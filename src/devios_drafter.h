#pragma once

#include "devios_corpus.h"
#include "llama.h"

#include <vector>
#include <deque>
#include <cstdint>
#include <cstddef>

namespace devios {

struct AdaptiveConfig {
    size_t k_min = 4;
    size_t k_max = 12;
    size_t k_initial = 4;
    size_t min_match_len = 3;
    size_t max_match_len = 5;
    size_t wake_up_match_len = 4;
    float accept_rate_high = 0.65f;
    float accept_rate_med = 0.40f;
    float accept_rate_low = 0.15f;
    size_t window_size = 16;
};

struct DraftMetrics {
    size_t total_generated = 0;
    size_t total_drafted = 0;
    size_t total_accepted = 0;
    size_t draft_attempts = 0;
    size_t accepted_steps = 0;
    size_t current_k = 4;
    float rolling_acceptance_rate = 0.0f;
    float avg_accepted_per_step = 0.0f;
};

class DraftController {
public:
    explicit DraftController(const AdaptiveConfig & cfg = AdaptiveConfig());

    void set_corpus(const CorpusIndex * corpus);

    // Level 1: Dynamic context (prompt lookup) n-gram search
    std::vector<llama_token> propose_dynamic_draft(
        const std::vector<llama_token> & context,
        size_t k = 4,
        size_t min_len = 3,
        size_t max_len = 5
    );

    // Level 2: Static corpus search from corpus.bin
    std::vector<llama_token> propose_corpus_draft(
        const std::vector<llama_token> & context,
        size_t k = 4
    );

    // Hybrid generator (Level 1 Dynamic Context -> Level 2 Static Corpus)
    std::vector<llama_token> propose_hybrid_draft(
        const std::vector<llama_token> & context,
        size_t k = 4
    );

    // Legacy method maintained for compatibility
    std::vector<uint32_t> propose_draft(const uint32_t * context_tail, size_t tail_len);

    // Record verification result
    void record_verification(size_t drafted_count, size_t accepted_count);

    // Bypass countdown tick (when K=0 zero-overhead fallback is active)
    void tick_bypass();

    void reset();

    DraftMetrics get_metrics() const;
    size_t get_current_k() const { return current_k; }

private:
    AdaptiveConfig config;
    const CorpusIndex * corpus_ptr = nullptr;

    size_t current_k = 4;
    size_t consecutive_perfect_steps = 0;
    size_t consecutive_zero_steps = 0;
    size_t zero_bypass_counter = 0;

    struct Trial {
        size_t drafted;
        size_t accepted;
    };
    std::deque<Trial> history;

    DraftMetrics metrics;

    void update_k();
};

} // namespace devios
