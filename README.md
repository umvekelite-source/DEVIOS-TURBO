# DEVIOS Turbo

High-performance, hardware-optimized CPU speculative inference engine for local LLMs, designed to break through memory-bandwidth bottlenecks on commodity x86_64 hardware.

---

## 1. Overview & Problem Statement

On bandwidth-constrained systems (e.g., AMD Zen 2/Zen 3 with single-channel DDR4-3200 memory), autoregressive token generation for a ~4B parameter quantized model (e.g., Gemma 4B Q4_K_M, ~5.3 GB) is strictly bounded by RAM read speed:

$$\text{Throughput}_{\text{max}} \approx \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Size (GB)}} \approx \frac{20 \text{ GB/s}}{5.3 \text{ GB}} \approx 3.8 \text{ tokens/sec}$$

Computational cores sit mostly idle while waiting on memory fetches. 

**DEVIOS Turbo** solves this by trading idle compute cycles for memory bandwidth through **Hierarchical Speculative Decoding**: verifying multiple predicted candidate tokens ($K = 4 \dots 12$) in a single sequential read of model weights from RAM, achieving real-world generation speeds of **14+ tokens/sec** ($\times 3.6$ speedup) without quality degradation.

---

## 2. Core Architecture Innovations

```
User / Agent Prompt
        │
        ▼
┌────────────────────────────────────────────────────────┐
│               Draft Controller (Dual-Tier)             │
│                                                        │
│  [Tier 1] Dynamic Prompt-Lookup Drafting (PLD)         │
│           - Scans active KV-cache & conversation       │
│           - n-gram matching (n=5..3)                   │
│           - 2-gram trampoline for syntax anchors       │
│           - Multi-hop candidate chaining (K up to 12)  │
│                                                        │
│  [Tier 2] Static Suffix Memory (corpus.bin)            │
│           - Suffix Array binary search O(L log N)      │
│           - ReAct schemas, JSON specs, codebase tokens │
│                                                        │
│  [Fallback] Zero-Overhead Bypass                       │
│           - If acceptance == 0% for 2 steps:           │
│             Drop K=0 for 4 tokens (pure autoregressive)│
└──────────────────────────┬─────────────────────────────┘
                           │ Proposed Candidates
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Inference Pipeline                   │
│                                                        │
│  • KV-Cache: GGML_TYPE_Q4_0 (50% bandwidth reduction)   │
│  • Flash Attention Enabled                             │
│  • Process Memory Locked (MMAP_MLOCK)                  │
│  • Batch Verification: N tokens evaluated per memory read│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│          Zero-Latency Execution & Network Engine       │
│                                                        │
│  • Zen 2 Topology: Pinned to physical cores 0,2,4,6,8   │
│  • Hot Spin-Wait Threadpool (poll=100, _mm_pause)      │
│  • Network IO isolated on core 10                      │
│  • Socket TCP_NODELAY: 0ms Nagle buffering on SSE      │
└────────────────────────────────────────────────────────┘
```

### Key Pillars:
1. **Dynamic Prompt-Lookup Drafting with 2-Gram Trampoline**:
   - Searches full KV-cache token history using multi-hop n-gram matching ($n=5, 4, 3$).
   - When long n-grams fail on short punctuation/transitions (e.g., `": "`, `" {\n"`, `"->"`), a 2-gram trampoline drafts 2–3 tokens to bridge the gap, unlocking longer chains on the next step.
2. **Adaptive Window Scaling ($K = 4 \to 12$)**:
   - Scales draft length dynamically based on consecutive 100% acceptance streaks.
3. **Zero-Overhead Fallback**:
   - On novel or unpredictable text (0% acceptance for 2 consecutive steps), sets $K=0$ for 4 tokens to avoid wasting memory bus traffic on invalid speculative batches.
4. **Q4_0 KV-Cache Ultra-Compression**:
   - Compresses Key and Value states to `GGML_TYPE_Q4_0`, halving cache memory footprint and memory bus pressure.
5. **Zero-Latency Spin-Wait Threadpool**:
   - Pinned strictly to physical Zen 2 cores (`cpumask: 0x155`) with high thread priority and active polling (`poll=100`, `_mm_pause`), eliminating OS scheduler thread-wake latencies (~10–15 µs).
   - Core 10 is reserved for HTTP and socket operations.
6. **TCP_NODELAY Streaming Server**:
   - Full OpenAI-compatible SSE streaming (`/v1/chat/completions`) with Nagle's algorithm disabled, ensuring instant delivery of every generated token.

---

## 3. Real-World Benchmarks

**Hardware**: AMD Ryzen 5 3600 (6C/12T @ 3.6-4.2 GHz), 16 GB DDR4-3200 Single-Channel (1x16GB), Windows 11 x64.  
**Model**: `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` (5.3 GB).

| Mode | Generation Speed | Acceptance Rate | Effective Memory Bandwidth | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Standard Decode)** | **3.8 tok/s** | N/A | ~20.1 GB/s | Bounded by single-channel RAM limit |
| **Vanilla n-gram Speculation** | **5.8 tok/s** | ~30% | ~30.7 GB/s | Stalls on novel context |
| **DEVIOS Turbo (C++ / Code)** | **14.1 tok/s** | **84.3%** | **~74.7 GB/s equiv.** | Multi-hop chaining ($K=12$) |
| **DEVIOS Turbo (OpenCode ReAct)** | **14.3 tok/s** | **70.4%** | **~75.8 GB/s equiv.** | Dynamic PLD + Suffix Memory |

---

## 4. Directory Structure

```
devios-turbo/
├── configs/                  # Client configuration guides
│   ├── jan_settings_guide.md # Jan (v0.8.7+) setup guide
│   ├── lm_studio_guide.md    # LM Studio connection guide
│   └── opencode_rules.md     # OpenCode agent execution rules
├── data/                     # Raw corpus data
│   ├── hybrid_corpus.txt     # OpenCode schemas, Markdown & C++ templates
│   ├── sample_corpus.txt     # Example general corpus
│   └── sample_gemma.txt      # Example Gemma prompt corpus
├── scripts/                  # Automated build and run scripts
│   ├── build.bat             # MSVC + Ninja build script
│   ├── setup_and_build.bat   # Clone llama.cpp, patch, and build
│   ├── run_server.bat        # Start production daemon on port 18080
│   ├── index_corpus.bat      # Suffix Memory indexer
│   └── benchmark.ps1         # Telemetry and speed benchmark
├── src/                      # Core C++ engine sources
│   ├── CMakeLists.txt        # Engine CMake definition with MSVC overdrive
│   ├── devios_corpus.h/.cpp  # Suffix Array builder and binary search
│   ├── devios_drafter.h/.cpp # PLD + 2-gram trampoline + Adaptive K
│   ├── devios_engine.h/.cpp  # Threadpool, Q4_0 KV-cache, core pinning
│   ├── devios_server.h/.cpp  # OpenAI HTTP/SSE server with TCP_NODELAY
│   └── main.cpp              # CLI and daemon entrypoint
├── devios_turbo.patch        # Clean patch for upstream llama.cpp
├── .gitignore                # Production gitignore
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 5. Quick Start & Building

### Prerequisites
- Windows 10/11 x64 or Linux x86_64
- CMake 3.26+ and Ninja
- MSVC 2022 (with C++ CMake tools) or GCC 12+/Clang 15+
- Git

### Automated Build (Windows)
```cmd
git clone https://github.com/YourUsername/devios-turbo.git
cd devios-turbo
scripts\setup_and_build.bat
```

### Manual Build via Upstream llama.cpp
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

Start the OpenAI-compatible HTTP daemon on port `18080`:

```cmd
scripts\run_server.bat "C:\path\to\model.gguf" 18080
```

Console output upon launch:
```
[c4] CPU Affinity: process on Zen 2 physical cores 0..5 (mask: 0x555, priority: HIGH)
[c4] ThreadPool: attached zero-latency spin-wait threadpool (poll=100, strict_cpu=true, cores 0,2,4,6,8)
[c4] DEVIOS Turbo server listening on http://127.0.0.1:18080
[c4] Active model ID: Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Turbo
```

---

## 7. Client Integrations

### OpenCode
Place `configs/opencode_rules.md` into `.opencode/rules.md` in your project root. Configure your OpenCode model endpoint to `http://127.0.0.1:18080/v1`.

### Jan (jan.ai)
Open Jan -> Settings -> Model Providers -> OpenAI:
- **Base URL**: `http://127.0.0.1:18080/v1`
- **API Key**: `devios-turbo`
- Select `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Turbo`.  
See `configs/jan_settings_guide.md` for full details.

### LM Studio
Settings -> Providers -> Add Custom OpenAI Provider:
- **Base URL**: `http://127.0.0.1:18080/v1`  
See `configs/lm_studio_guide.md`.

---

## 8. HTTP API Reference

- `GET /v1/models`: OpenAI-compatible model discovery.
- `POST /v1/chat/completions`: Supports standard completions and SSE streaming (`stream: true`).
- `GET /v1/turbo/metrics`: Live telemetry endpoint returning tokens/sec, rolling acceptance rate, active $K$, and corpus memory usage.
- `POST /v1/corpus/index`: Ingests and compiles text into the live suffix array memory without restarting the process.
- `POST /v1/corpus/clear`: Clears the suffix memory index.

---

## 9. License

MIT License. See [LICENSE](LICENSE) for details.
