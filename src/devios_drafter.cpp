#include "devios_drafter.h"

#include <numeric>
#include <algorithm>

namespace devios {

DraftController::DraftController(const AdaptiveConfig & cfg)
    : config(cfg), current_k(cfg.k_initial) {
}

void DraftController::set_corpus(const CorpusIndex * corpus) {
    corpus_ptr = corpus;
}

void DraftController::reset() {
    current_k = config.k_initial;
    consecutive_perfect_steps = 0;
    consecutive_zero_steps = 0;
    zero_bypass_counter = 0;
    history.clear();
    metrics = DraftMetrics();
    metrics.current_k = current_k;
}

std::vector<llama_token> DraftController::propose_dynamic_draft(
    const std::vector<llama_token> & context,
    size_t k,
    size_t min_len,
    size_t max_len
) {
    if (context.size() < 2 || k == 0) {
        return {};
    }

    std::vector<llama_token> result;
    result.reserve(k);

    std::vector<llama_token> working = context;
    working.reserve(context.size() + k);

    while (result.size() < k) {
        const size_t n = working.size();
        const size_t needed = k - result.size();
        std::vector<llama_token> best_candidate;

        size_t start_len = std::min(max_len, (n > 1) ? (n - 1) : 1);
        for (size_t l = start_len; l >= min_len; --l) {
            if (l >= n) continue;
            const llama_token * query = &working[n - l];
            const llama_token first_tok = query[0];

            for (int64_t pos = static_cast<int64_t>(n - l - 1); pos >= 0; --pos) {
                if (working[pos] != first_tok) {
                    continue;
                }

                bool matched = true;
                for (size_t j = 1; j < l; ++j) {
                    if (working[pos + j] != query[j]) {
                        matched = false;
                        break;
                    }
                }

                if (matched) {
                    size_t draft_start = pos + l;
                    size_t draft_end = std::min(n, draft_start + needed);
                    if (draft_end > draft_start) {
                        size_t cand_len = draft_end - draft_start;
                        if (cand_len == needed) {
                            best_candidate.assign(working.begin() + draft_start, working.begin() + draft_end);
                            break;
                        }
                        if (cand_len > best_candidate.size()) {
                            best_candidate.assign(working.begin() + draft_start, working.begin() + draft_end);
                        }
                    }
                }
            }

            if (!best_candidate.empty()) {
                break;
            }
        }

        // Fast 2-gram trampoline: if 3..5 found nothing, try 2-gram for short syntax transitions
        if (best_candidate.empty() && n >= 3 && needed >= 2) {
            const llama_token * query = &working[n - 2];
            const llama_token first_tok = query[0];
            const llama_token second_tok = query[1];
            const size_t max_2g = std::min<size_t>(needed, 3);

            for (int64_t pos = static_cast<int64_t>(n - 3); pos >= 0; --pos) {
                if (working[pos] == first_tok && working[pos + 1] == second_tok) {
                    size_t draft_start = pos + 2;
                    size_t draft_end = std::min(n, draft_start + max_2g);
                    if (draft_end > draft_start) {
                        best_candidate.assign(working.begin() + draft_start, working.begin() + draft_end);
                        break;
                    }
                }
            }
        }

        if (best_candidate.empty()) {
            break;
        }

        result.insert(result.end(), best_candidate.begin(), best_candidate.end());
        working.insert(working.end(), best_candidate.begin(), best_candidate.end());
    }

    return result;
}

std::vector<llama_token> DraftController::propose_corpus_draft(
    const std::vector<llama_token> & context,
    size_t k
) {
    if (!corpus_ptr || corpus_ptr->size() == 0 || context.empty() || k == 0) {
        return {};
    }

    size_t tail_len = std::min<size_t>(context.size(), 32);
    const uint32_t * tail_ptr = reinterpret_cast<const uint32_t *>(context.data() + (context.size() - tail_len));

    MatchResult match = corpus_ptr->find_longest_match(tail_ptr, tail_len, config.min_match_len);
    if (match.corpus_offset < 0 || match.match_len < config.min_match_len) {
        return {};
    }

    std::vector<uint32_t> u32_draft = corpus_ptr->get_draft(match, k);
    std::vector<llama_token> draft;
    draft.reserve(u32_draft.size());
    for (uint32_t tok : u32_draft) {
        draft.push_back(static_cast<llama_token>(tok));
    }
    return draft;
}

std::vector<llama_token> DraftController::propose_hybrid_draft(
    const std::vector<llama_token> & context,
    size_t k
) {
    if (k == 0) {
        return {};
    }

    // Level 1: Dynamic Context (Prompt-Lookup Drafting across full KV history)
    std::vector<llama_token> draft = propose_dynamic_draft(context, k, config.min_match_len, config.max_match_len);
    if (!draft.empty()) {
        // ALWAYS prioritize dynamic context matches (OpenCode agent prompt / dialogue history)
        return draft;
    }

    // Level 2: Static Corpus fallback (when dynamic context has 0 matches)
    return propose_corpus_draft(context, k);
}

std::vector<uint32_t> DraftController::propose_draft(const uint32_t * context_tail, size_t tail_len) {
    if (!corpus_ptr || corpus_ptr->size() == 0 || tail_len == 0 || current_k == 0) {
        return {};
    }

    MatchResult match = corpus_ptr->find_longest_match(context_tail, tail_len, config.min_match_len);
    if (match.corpus_offset < 0 || match.match_len < config.min_match_len) {
        return {};
    }

    return corpus_ptr->get_draft(match, current_k);
}

void DraftController::record_verification(size_t drafted_count, size_t accepted_count) {
    if (drafted_count == 0) {
        return;
    }

    metrics.total_drafted += drafted_count;
    metrics.total_accepted += accepted_count;
    metrics.draft_attempts += 1;
    if (accepted_count > 0) {
        metrics.accepted_steps += 1;
    }

    history.push_back({drafted_count, accepted_count});
    if (history.size() > config.window_size) {
        history.pop_front();
    }

    // Zero-overhead fallback: if last 2 steps accepted 0, bypass K for 4 tokens
    if (accepted_count == 0) {
        consecutive_perfect_steps = 0;
        consecutive_zero_steps++;
        if (consecutive_zero_steps >= 2) {
            zero_bypass_counter = 4;
            current_k = 0;
        } else {
            current_k = config.k_initial;
        }
    } else {
        consecutive_zero_steps = 0;
        zero_bypass_counter = 0;

        // Adaptive K scaling:
        if (accepted_count == drafted_count && drafted_count > 0) {
            consecutive_perfect_steps++;
            current_k = std::min<size_t>(config.k_max, current_k + 2);
        } else if (accepted_count >= (drafted_count * 3) / 4 && drafted_count > 0) {
            current_k = std::min<size_t>(config.k_max, current_k + 1);
        } else {
            consecutive_perfect_steps = 0;
            current_k = std::max<size_t>(config.k_initial, accepted_count + 1);
        }
    }

    update_k();
}

void DraftController::tick_bypass() {
    if (zero_bypass_counter > 0) {
        --zero_bypass_counter;
        if (zero_bypass_counter == 0) {
            current_k = config.k_initial; // Restore K to 4
            consecutive_zero_steps = 0;
            metrics.current_k = current_k;
        }
    }
}

void DraftController::update_k() {
    if (history.empty()) {
        return;
    }

    size_t win_drafted = 0;
    size_t win_accepted = 0;
    for (const auto & t : history) {
        win_drafted += t.drafted;
        win_accepted += t.accepted;
    }

    float rate = (win_drafted > 0) ? (static_cast<float>(win_accepted) / static_cast<float>(win_drafted)) : 0.0f;
    metrics.rolling_acceptance_rate = rate;
    metrics.avg_accepted_per_step = (metrics.draft_attempts > 0)
        ? (static_cast<float>(metrics.total_accepted) / static_cast<float>(metrics.draft_attempts))
        : 0.0f;
    metrics.current_k = current_k;
}

DraftMetrics DraftController::get_metrics() const {
    return metrics;
}

} // namespace devios
