#include "devios_engine.h"
#include "devios_server.h"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

void print_help(const char * prog) {
    std::cout << "DEVIOS Turbo v0.1 - CPU Speculative Inference with Suffix Memory\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH         Path to GGUF model file (required)\n";
    std::cout << "  -c, --corpus PATH        Path to binary corpus index (corpus.bin)\n";
    std::cout << "  -p, --prompt TEXT        Prompt to generate\n";
    std::cout << "  -n, --predict N          Maximum tokens to generate (default: 256)\n";
    std::cout << "  -t, --threads N          Number of inference threads (default: 6)\n";
    std::cout << "      --ctx-size N         Context size in tokens (default: 4096)\n";
    std::cout << "      --jinja              Use embedded Jinja chat template\n";
    std::cout << "      --temp FLOAT         Sampling temperature (default: 0.0)\n";
    std::cout << "      --k N                Initial draft length K (default: 4)\n";
    std::cout << "      --baseline           Run baseline generation (disable speculation)\n";
    std::cout << "      --server             Run OpenAI-compatible HTTP server\n";
    std::cout << "      --host IP            Server host (default: 127.0.0.1)\n";
    std::cout << "      --port PORT          Server port (default: 8080)\n";
    std::cout << "      --index-file PATH    Index a text file into corpus\n";
    std::cout << "      --index-dir PATH     Recursively index a directory of files\n";
    std::cout << "      --save-corpus PATH   Save built index to binary file (default: corpus.bin)\n";
    std::cout << "  -h, --help               Show this help\n";
}

int main(int argc, char ** argv) {
    devios::EngineConfig cfg;
    std::string prompt = "";
    int32_t max_tokens = 256;
    bool is_server = false;
    std::string server_host = "127.0.0.1";
    int server_port = 8080;
    std::string index_file = "";
    std::string index_dir = "";
    std::string save_corpus_path = "corpus.bin";
    bool baseline = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i < argc) cfg.model_path = argv[i];
        } else if (arg == "-c" || arg == "--corpus") {
            if (++i < argc) cfg.corpus_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i < argc) prompt = argv[i];
        } else if (arg == "-n" || arg == "--predict") {
            if (++i < argc) max_tokens = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i < argc) cfg.n_threads = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i < argc) cfg.n_batch = std::stoi(argv[i]);
        } else if (arg == "--ctx-size" || arg == "-ctx" || arg == "--n-ctx") {
            if (++i < argc) cfg.n_ctx = std::stoi(argv[i]);
        } else if (arg == "--jinja") {
            // Force / enable jinja template formatting (active by default in format_chat)
        } else if (arg == "-fa" || arg == "--flash-attn") {
            // Flash attention active by default
        } else if (arg == "--cache-type-k") {
            if (++i < argc) { /* Q8_0 active by default */ }
        } else if (arg == "--cache-type-v") {
            if (++i < argc) { /* Q8_0 active by default */ }
        } else if (arg == "--mlock") {
            // MMAP + MLOCK active by default
        } else if (arg == "--temp") {
            if (++i < argc) cfg.temperature = std::stof(argv[i]);
        } else if (arg == "--k") {
            if (++i < argc) cfg.k_initial = std::stoul(argv[i]);
        } else if (arg == "--baseline") {
            baseline = true;
        } else if (arg == "--server") {
            is_server = true;
        } else if (arg == "--host") {
            if (++i < argc) server_host = argv[i];
        } else if (arg == "--port") {
            if (++i < argc) server_port = std::stoi(argv[i]);
        } else if (arg == "--index-file") {
            if (++i < argc) index_file = argv[i];
        } else if (arg == "--index-dir") {
            if (++i < argc) index_dir = argv[i];
        } else if (arg == "--save-corpus") {
            if (++i < argc) save_corpus_path = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "[c4] Error: --model is required.\n";
        print_help(argv[0]);
        return 1;
    }

    if (baseline) {
        cfg.k_initial = 0;
        cfg.corpus_path = "";
    }

    devios::TurboEngine engine(cfg);
    if (!engine.init()) {
        std::cerr << "[c4] Engine initialization failed.\n";
        return 1;
    }

    // Index file or directory if requested
    if (!index_file.empty()) {
        std::ifstream f(index_file);
        if (!f.is_open()) {
            std::cerr << "[c4] Failed to open " << index_file << std::endl;
            return 1;
        }
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::cout << "[c4] Indexing file: " << index_file << " (" << content.size() << " bytes)\n";
        engine.index_corpus_text(content, save_corpus_path);
        std::cout << "[c4] Corpus index saved to " << save_corpus_path
                  << " (" << engine.get_corpus().size() << " tokens)\n";
    } else if (!index_dir.empty()) {
        std::string accumulated = "";
        uint64_t n_files = 0;
        for (const auto & entry : fs::recursive_directory_iterator(index_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::string filename = entry.path().filename().string();
            // Privacy filter
            if (filename.find(".env") != std::string::npos ||
                filename.find(".pem") != std::string::npos ||
                filename.find(".key") != std::string::npos ||
                filename.find("id_rsa") != std::string::npos ||
                entry.path().string().find(".git") != std::string::npos) {
                continue;
            }
            // Code and text extensions
            if (ext == ".c" || ext == ".cpp" || ext == ".h" || ext == ".hpp" ||
                ext == ".py" || ext == ".js" || ext == ".ts" || ext == ".json" ||
                ext == ".md" || ext == ".txt") {
                std::ifstream f(entry.path());
                if (f.is_open()) {
                    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                    accumulated += "\n// FILE: " + entry.path().filename().string() + "\n" + content;
                    ++n_files;
                }
            }
        }
        std::cout << "[c4] Indexing " << n_files << " files from " << index_dir
                  << " (" << accumulated.size() << " bytes)\n";
        engine.index_corpus_text(accumulated, save_corpus_path);
        std::cout << "[c4] Corpus index saved to " << save_corpus_path
                  << " (" << engine.get_corpus().size() << " tokens)\n";
    }

    if (!index_file.empty() || !index_dir.empty()) {
        if (!is_server && prompt.empty()) {
            return 0;
        }
    }

    if (is_server) {
        devios::ServerConfig scfg;
        scfg.host = server_host;
        scfg.port = server_port;
        devios::TurboServer server(engine, scfg);
        server.start();
        return 0;
    }

    if (prompt.empty()) {
        std::cout << "[c4] Ready. Enter prompt (or Ctrl+C to exit):\n> ";
        std::getline(std::cin, prompt);
    }

    std::cout << "\n[c4] Generating (" << (baseline ? "BASELINE" : "TURBO") << "):\n";
    std::string response = engine.generate(prompt, max_tokens, [](const std::string & token) {
        std::cout << token << std::flush;
        return true;
    });

    devios::GenerationStats stats = engine.get_last_stats();
    devios::DraftMetrics dm = engine.get_drafter_metrics();

    printf("\n\n[DEVIOS TURBO GEMMA] Tokens: %d | Time: %.1fs | Real Speed: %.1f tok/s | Draft Acceptance: %.0f%%\n",
           stats.n_predicted, stats.t_gen_ms / 1000.0, stats.gen_speed_ts, stats.acceptance_rate * 100.0f);
    fflush(stdout);

    std::cout << "\n--- [c4] Diagnostics ---\n";
    std::cout << "Prompt tokens:     " << stats.n_prompt << " (" << stats.prompt_speed_ts << " t/s)\n";
    std::cout << "Generated tokens:  " << stats.n_predicted << " (" << stats.gen_speed_ts << " t/s)\n";
    if (!baseline) {
        std::cout << "Drafted tokens:    " << stats.n_drafted << "\n";
        std::cout << "Accepted tokens:   " << stats.n_accepted << "\n";
        std::cout << "Acceptance rate:   " << (stats.acceptance_rate * 100.0f) << "%\n";
        std::cout << "Avg accepted/step: " << stats.avg_accepted_per_step << "\n";
        std::cout << "Active K:          " << stats.final_k << "\n";
        std::cout << "Corpus tokens:     " << engine.get_corpus().size() << " ("
                  << (stats.memory_bytes / 1024.0 / 1024.0) << " MB)\n";
    }

    return 0;
}
