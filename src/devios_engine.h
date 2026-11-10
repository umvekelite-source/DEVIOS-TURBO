#pragma once

#include "devios_corpus.h"
#include "devios_drafter.h"

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <string>
#include <vector>
#include <functional>

namespace devios {

struct EngineConfig {
    std::string model_path;
    std::string corpus_path;
    int32_t n_ctx = 8192;
    int32_t n_threads = 5;       // Strictly 5 threads on Ryzen 5 3600 (single-channel DDR4-3200)
    int32_t n_batch = 2048;
    int32_t n_ubatch = 512;
    float temperature = 0.0f;     // Default deterministic
    uint32_t seed = 42;
    size_t k_initial = 4;
    size_t k_max = 12;
    size_t min_match_len = 3;
};

struct GenerationStats {
    int32_t n_prompt = 0;
    int32_t n_predicted = 0;
    int32_t n_drafted = 0;
    int32_t n_accepted = 0;
    double t_prompt_ms = 0.0;
    double t_gen_ms = 0.0;
    double prompt_speed_ts = 0.0;
    double gen_speed_ts = 0.0;
    float acceptance_rate = 0.0f;
    float avg_accepted_per_step = 0.0f;
    size_t final_k = 0;
    size_t memory_bytes = 0;
};

class TurboEngine {
public:
    explicit TurboEngine(const EngineConfig & config);
    ~TurboEngine();

    bool init();
    void close();

    // Rebuild or reload corpus
    bool load_corpus(const std::string & path);
    bool index_corpus_text(const std::string & text, const std::string & save_path);
    void clear_corpus();

    // Generation
    std::string generate(
        const std::string & prompt,
        int32_t max_tokens = 256,
        std::function<bool(const std::string &)> token_callback = nullptr
    );

    std::string get_model_id() const;
    std::string get_raw_model_name() const;
    std::string format_chat(const std::vector<std::pair<std::string, std::string>> & messages) const;

    GenerationStats get_last_stats() const { return last_stats; }
    DraftMetrics get_drafter_metrics() const { return drafter.get_metrics(); }
    const CorpusIndex & get_corpus() const { return corpus; }

    uint32_t compute_vocab_hash() const;

private:
    EngineConfig config;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;

    CorpusIndex corpus;
    DraftController drafter;
    GenerationStats last_stats;

    struct ggml_threadpool * tp = nullptr;
    bool initialized = false;
};

} // namespace devios
