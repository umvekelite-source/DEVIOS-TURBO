# DEVIOS Turbo

A CPU speculative-inference engine for local LLMs, designed to explore ways to improve token-generation throughput on memory-bandwidth-constrained x86-64 systems.

> **Benchmark note:** The performance figures below are preliminary measurements reported by the project author on one machine. They are not independently verified, and should not be treated as guaranteed results. Performance can vary with hardware, memory configuration, model, software versions, settings, workload, and system load. See [Benchmark Results](#3-benchmark-results) for the recorded setup and limitations.

---

## 1. Overview

Autoregressive generation for quantized LLMs can be limited by memory bandwidth: model weights must be read repeatedly during token generation. DEVIOS Turbo explores hierarchical speculative decoding, which proposes candidate tokens and verifies them in batches in an effort to make better use of available compute and memory bandwidth.

The project combines prompt-lookup drafting, optional corpus-based suffix memory, adaptive draft lengths, and an OpenAI-compatible HTTP/SSE server. The actual speedup depends on the workload and hardware; speculative decoding may provide less benefit when candidate acceptance is low.

## 2. Architecture

```text
User / Agent Prompt
        |
        v
+--------------------------------------------------+
| Draft Controller                                 |
|  - Prompt-lookup drafting using n-gram matching   |
|  - Optional static suffix-memory corpus          |
|  - Adaptive candidate window and fallback        |
+------------------------+-------------------------+
                         | Proposed candidates
                         v
+--------------------------------------------------+
| Inference pipeline                               |
|  - llama.cpp-based model inference                |
|  - Quantized KV-cache options and batch checking  |
+------------------------+-------------------------+
                         v
+--------------------------------------------------+
| HTTP/SSE server and execution tuning              |
|  - OpenAI-compatible endpoints                    |
|  - CPU affinity and low-latency streaming options  |
+--------------------------------------------------+
```

### Main components

- **Prompt-lookup drafting:** Searches prior token context for candidate continuations using n-gram matching, with a short n-gram fallback for transitions.
- **Suffix memory:** An optional corpus index intended to provide candidates from reusable text, schemas, or code patterns.
- **Adaptive speculative window:** Adjusts the number of proposed tokens based on recent acceptance behavior.
- **Fallback behavior:** Reduces or bypasses speculation when candidates are not being accepted.
- **Inference and server integration:** Integrates with llama.cpp and exposes an OpenAI-compatible API with server-sent event (SSE) streaming.
- **CPU tuning:** Includes affinity and thread-pool settings that may be hardware- and operating-system-specific.

Implementation details and available options can change as the project develops. Consult the source and build scripts for the exact behavior of the current revision.

---

## 3. Benchmark Results

The figures below are **author-reported measurements from a single local test setup**. They have not yet been independently reproduced. Treat them as preliminary observations, not as a performance guarantee or a general comparison across systems.

**Reported test machine:** AMD Ryzen 5 3600 (6 cores / 12 threads), 16 GB DDR4-3200 single-channel memory, Windows 11 x64.  
**Reported model:** `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` (approximately 5.3 GB).

| Mode | Reported generation speed | Reported acceptance rate | Notes |
| :--- | :---: | :---: | :--- |
| Baseline (standard decode) | 3.8 tok/s | N/A | Author-reported measurement |
| Vanilla n-gram speculation | 5.8 tok/s | ~30% | Author-reported measurement |
| DEVIOS Turbo (C++ / code workload) | 14.1 tok/s | 84.3% | Reported with multi-hop drafting (K=12) |
| DEVIOS Turbo (OpenCode ReAct workload) | 14.3 tok/s | 70.4% | Reported with prompt lookup and suffix memory |

The previously reported “effective memory bandwidth” values are estimates derived from throughput, not direct measurements of physical RAM bandwidth, and are therefore omitted from the comparison table.

Results can differ substantially with CPU, memory channels and timings, model and quantization, prompt and output length, context size, build configuration, background load, and benchmark procedure. The current published figures do not include a complete, independently reproducible test protocol or raw result files. We plan to improve this by documenting a fixed benchmark procedure and collecting comparable community submissions. Please include the exact hardware, model, software revision, settings, workload, and repeated-run statistics when sharing results.

---

## 4. Repository Structure

```text
configs/                  Client configuration guides
  jan_settings_guide.md   Jan setup guide
  lm_studio_guide.md      LM Studio connection guide
  opencode_rules.md       OpenCode agent execution rules
data/                     Example corpora and text data
scripts/                  Build, run, indexing, and benchmark scripts
src/                      C++ engine, corpus, drafting, and server code
tests/                    Tests
devios_turbo.patch        Patch for the pinned llama.cpp revision
```

---

## 5. Building

### Prerequisites

- Windows 10/11 x64 or Linux x86-64
- CMake 3.26+ and Ninja
- MSVC 2022 with C++ CMake tools, or GCC 12+ / Clang 15+
- Git

### Automated build (Windows)

```cmd
git clone https://github.com/umvekelite-source/DEVIOS-TURBO.git
cd DEVIOS-TURBO
scripts\setup_and_build.bat
```

### Manual build using the pinned upstream llama.cpp revision

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout 7d7f0287530bc8981690f361d8c1627dbae1c99a
git apply /path/to/devios_turbo.patch

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_AVX2=ON -DGGML_FMA=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release --target devios-turbo
```

---

## 6. Running the Server

Start the HTTP server on port `18080`:

```cmd
scripts\run_server.bat "C:\path\to\model.gguf" 18080
```

The server prints its active model and listening address at startup. CPU affinity and thread-pool behavior depend on the build and host system.

---

## 7. Client Integrations

The server is intended to work with OpenAI-compatible clients using `http://127.0.0.1:18080/v1` as the base URL.

- **OpenCode:** See `configs/opencode_rules.md`.
- **Jan:** Configure an OpenAI-compatible provider; see `configs/jan_settings_guide.md`.
- **LM Studio:** Add a custom OpenAI-compatible provider; see `configs/lm_studio_guide.md`.

---

## 8. HTTP API

- `GET /v1/models` — model discovery.
- `POST /v1/chat/completions` — chat completions, including SSE streaming with `stream: true`.
- `GET /v1/turbo/metrics` — runtime metrics such as tokens/sec, acceptance rate, active draft window, and corpus memory usage.
- `POST /v1/corpus/index` — index text into suffix memory without restarting the process.
- `POST /v1/corpus/clear` — clear the suffix-memory index.

---

## 9. License

MIT License. See [LICENSE](LICENSE) for details.
