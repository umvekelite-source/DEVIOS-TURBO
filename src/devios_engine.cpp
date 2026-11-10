#include "devios_engine.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "log.h"

#include <chrono>
#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace devios {

TurboEngine::TurboEngine(const EngineConfig & cfg)
    : config(cfg) {
    AdaptiveConfig ad_cfg;
    ad_cfg.k_initial = cfg.k_initial;
    ad_cfg.k_max = cfg.k_max;
    ad_cfg.min_match_len = cfg.min_match_len;
    drafter = DraftController(ad_cfg);
}

TurboEngine::~TurboEngine() {
    close();
}

uint32_t TurboEngine::compute_vocab_hash() const {
    if (!vocab) {
        return 0;
    }
    const int32_t n_tokens = llama_vocab_n_tokens(vocab);
    uint32_t h = 5381;
    h = ((h << 5) + h) ^ static_cast<uint32_t>(n_tokens);

    // Hash a sample of token pieces across the vocab
    int step = std::max(1, n_tokens / 128);
    char buf[128];
    for (int32_t i = 0; i < n_tokens; i += step) {
        int len = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, false);
        for (int j = 0; j < len; ++j) {
            h = ((h << 5) + h) ^ static_cast<uint8_t>(buf[j]);
        }
    }
    return h;
}

bool TurboEngine::init() {
    if (initialized) {
        return true;
    }

    llama_backend_init();

#ifdef _WIN32
    // Pin process to physical Zen 2 cores (Logical CPUs 0, 2, 4, 6, 8 for GGML + 10 for HTTP/OS)
    // Mask: 0x555 (Cores 0, 1, 2, 3, 4, 5 physical)
    DWORD_PTR proc_mask = (1ULL << 0) | (1ULL << 2) | (1ULL << 4) | (1ULL << 6) | (1ULL << 8) | (1ULL << 10);
    if (SetProcessAffinityMask(GetCurrentProcess(), proc_mask)) {
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        fprintf(stdout, "[c4] CPU Affinity: process on Zen 2 physical cores 0..5 (mask: 0x555, priority: HIGH)\n");
        fflush(stdout);
    }
#endif

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK; // Force system to keep model in RAM rather than swapping

    model = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "[c4] failed to load model from: %s\n", config.model_path.c_str());
        return false;
    }

    vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.n_ubatch = config.n_ubatch;
    cparams.n_threads = (config.n_threads > 0) ? config.n_threads : 5; // 5 threads strictly for Zen 2 single-channel
    cparams.n_threads_batch = cparams.n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;           // Flash Attention enabled
    cparams.type_k = GGML_TYPE_Q4_0;                                    // KV-cache Q4_0 (ultra memory bandwidth reduction)
    cparams.type_v = GGML_TYPE_Q4_0;                                    // KV-cache Q4_0
    cparams.n_outputs_max = config.n_batch;
    cparams.n_outputs_max_per_seq = 64;                                 // Multi-token logits for speculative batch

    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[c4] failed to create llama context\n");
        llama_model_free(model);
        model = nullptr;
        return false;
    }

    // Attach zero-latency hot spin-wait threadpool
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(cparams.n_threads);
    tpp.n_threads = cparams.n_threads;
    tpp.prio = GGML_SCHED_PRIO_HIGH;
    tpp.poll = 100; // Aggressive spin-wait with _mm_pause (no thread sleep/wake latency)
    tpp.strict_cpu = true;
    memset(tpp.cpumask, 0, sizeof(tpp.cpumask));
    tpp.cpumask[0] = true;
    tpp.cpumask[2] = true;
    tpp.cpumask[4] = true;
    tpp.cpumask[6] = true;
    tpp.cpumask[8] = true;

    tp = ggml_threadpool_new(&tpp);
    if (tp) {
        llama_attach_threadpool(ctx, tp, tp);
        fprintf(stdout, "[c4] ThreadPool: attached zero-latency spin-wait threadpool (poll=100, strict_cpu=true, cores 0,2,4,6,8)\n");
        fflush(stdout);
    }

    // Try loading corpus if specified
    if (!config.corpus_path.empty()) {
        load_corpus(config.corpus_path);
    }

    drafter.set_corpus(&corpus);
    initialized = true;
    return true;
}

void TurboEngine::close() {
    if (tp) {
        if (ctx) {
            llama_detach_threadpool(ctx);
        }
        ggml_threadpool_free(tp);
        tp = nullptr;
    }
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    llama_backend_free();
    initialized = false;
}

bool TurboEngine::load_corpus(const std::string & path) {
    uint32_t expected_hash = compute_vocab_hash();
    bool ok = corpus.load(path, expected_hash);
    if (ok) {
        drafter.set_corpus(&corpus);
    }
    return ok;
}

bool TurboEngine::index_corpus_text(const std::string & text, const std::string & save_path) {
    if (!vocab) {
        return false;
    }
    std::vector<llama_token> tokens = common_tokenize(vocab, text, false, false);
    if (tokens.empty()) {
        return false;
    }

    std::vector<uint32_t> u32_tokens(tokens.begin(), tokens.end());
    uint32_t v_hash = compute_vocab_hash();
    corpus.build(u32_tokens, v_hash, 1);
    drafter.set_corpus(&corpus);

    if (!save_path.empty()) {
        return corpus.save(save_path);
    }
    return true;
}

void TurboEngine::clear_corpus() {
    corpus.clear();
    drafter.set_corpus(&corpus);
}

std::string TurboEngine::generate(
    const std::string & prompt,
    int32_t max_tokens,
    std::function<bool(const std::string &)> token_callback
) {
    if (!initialized && !init()) {
        return "";
    }

    drafter.reset();
    last_stats = GenerationStats();

    // Reset KV cache sequence 0 for new generation request
    llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);

    // Tokenize prompt
    std::vector<llama_token> inp = common_tokenize(vocab, prompt, true, true);
    if (inp.empty()) {
        return "";
    }

    // Guard against prompt exceeding context size
    if (inp.size() >= static_cast<size_t>(config.n_ctx)) {
        size_t max_keep = (config.n_ctx > 256) ? (config.n_ctx - 128) : (config.n_ctx / 2);
        if (inp.size() > max_keep) {
            inp.erase(inp.begin(), inp.end() - max_keep);
        }
    }

    last_stats.n_prompt = static_cast<int32_t>(inp.size());

    // Setup sampler
    common_params_sampling sparams;
    sparams.temp = config.temperature;
    sparams.seed = config.seed;
    struct common_sampler * smpl = common_sampler_init(model, sparams);

    // Warm-up / decode prompt in chunks of n_batch
    auto t_prompt_start = std::chrono::high_resolution_clock::now();

    const size_t batch_size = std::max<size_t>(1, config.n_batch);
    if (inp.size() > 1) {
        size_t n_eval = inp.size() - 1;
        for (size_t i = 0; i < n_eval; i += batch_size) {
            size_t n_chunk = std::min<size_t>(n_eval - i, batch_size);
            common_batch batch(ctx);
            auto mem = llama_get_memory(ctx);
            llama_pos pos = llama_memory_seq_pos_max(mem, 0) + 1;
            for (size_t j = 0; j < n_chunk; ++j) {
                batch.add(inp[i + j], pos++, 0, false);
            }
            llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        }
    }

    {
        common_batch last_batch(ctx);
        auto mem = llama_get_memory(ctx);
        llama_pos pos = llama_memory_seq_pos_max(mem, 0) + 1;
        last_batch.add(inp.back(), pos, 0, true);
        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, last_batch.get());
    }

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    last_stats.t_prompt_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();
    last_stats.prompt_speed_ts = (last_stats.t_prompt_ms > 0)
        ? (last_stats.n_prompt / (last_stats.t_prompt_ms / 1000.0))
        : 0.0;

    int n_past = static_cast<int>(inp.size());
    int n_predict = 0;
    bool has_eos = false;

    std::vector<llama_token> draft;
    common_batch batch_tgt(ctx);
    std::string full_response = "";

    auto t_gen_start = std::chrono::high_resolution_clock::now();

    while (n_predict < max_tokens && !has_eos) {
        int i_dft = 0;
        size_t accepted_in_this_batch = 0;

        while (true) {
            llama_token id = common_sampler_sample(smpl, ctx, i_dft);
            common_sampler_accept(smpl, id, true);

            std::string piece = common_token_to_piece(ctx, id);

            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
            }

            ++n_predict;

            // Check match with drafted token
            if (i_dft < static_cast<int>(draft.size()) && id == draft[i_dft]) {
                ++last_stats.n_accepted;
                ++accepted_in_this_batch;
                ++n_past;
                ++i_dft;
                inp.push_back(id);
                full_response += piece;
                if (token_callback) {
                    if (!token_callback(piece)) {
                        has_eos = true;
                        break;
                    }
                }
                if (has_eos || n_predict >= max_tokens) {
                    break;
                }
                continue;
            }

            // Mismatch or end of draft tokens
            full_response += piece;
            if (token_callback) {
                if (!token_callback(piece)) {
                    has_eos = true;
                    break;
                }
            }

            if (!draft.empty()) {
                drafter.record_verification(draft.size(), accepted_in_this_batch);
            }

            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            break;
        }

        if (has_eos || n_predict >= max_tokens) {
            break;
        }

        // Clean KV cache of unaccepted draft tokens
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);

        batch_tgt.clear();
        batch_tgt.add(draft[0], n_past, 0, true);

        // Propose draft: Level 1 Dynamic Context -> Level 2 Static Corpus (with Adaptive K: 4..12)
        const size_t draft_k = drafter.get_current_k();
        if (draft_k == 0) {
            drafter.tick_bypass();
        }
        std::vector<llama_token> proposed = drafter.propose_hybrid_draft(inp, draft_k);

        for (llama_token tok : proposed) {
            draft.push_back(tok);
        }

        for (size_t i = 1; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], n_past + static_cast<int>(i), 0, true);
        }

        last_stats.n_drafted += static_cast<int32_t>(draft.size() - 1);

        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        ++n_past;

        draft.erase(draft.begin());
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    last_stats.t_gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    last_stats.n_predicted = n_predict;
    last_stats.gen_speed_ts = (last_stats.t_gen_ms > 0)
        ? (n_predict / (last_stats.t_gen_ms / 1000.0))
        : 0.0;

    DraftMetrics dm = drafter.get_metrics();
    float accept_rate = (last_stats.n_drafted > 0)
        ? (static_cast<float>(last_stats.n_accepted) / static_cast<float>(last_stats.n_drafted))
        : 0.0f;
    last_stats.acceptance_rate = accept_rate;
    last_stats.avg_accepted_per_step = dm.avg_accepted_per_step;
    last_stats.final_k = drafter.get_current_k();
    last_stats.memory_bytes = corpus.memory_bytes();

    common_sampler_free(smpl);
    return full_response;
}

std::string TurboEngine::get_raw_model_name() const {
    if (config.model_path.empty()) {
        return "devios-turbo";
    }
    std::string path = config.model_path;
    size_t last_slash = path.find_last_of("/\\");
    std::string name = (last_slash != std::string::npos) ? path.substr(last_slash + 1) : path;
    if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf") {
        name = name.substr(0, name.size() - 5);
    }
    return name;
}

std::string TurboEngine::get_model_id() const {
    std::string name = get_raw_model_name();
    if (name == "devios-turbo") {
        return name;
    }

    const std::vector<std::string> quant_suffixes = {
        "-abliterated.Q8_0", ".Q8_0", ".q8_0",
        "-Q4_K_M", ".Q4_K_M", ".q4_k_m",
        "-Q4_0", ".Q4_0", ".q4_0",
        "-Q5_K_M", ".Q5_K_M", ".q5_k_m",
        "-abliterated"
    };

    std::string clean = name;
    for (const auto & qs : quant_suffixes) {
        if (clean.size() > qs.size() && clean.substr(clean.size() - qs.size()) == qs) {
            clean = clean.substr(0, clean.size() - qs.size());
            break;
        }
    }

    if (clean.find("-Turbo") == std::string::npos && clean.find("-turbo") == std::string::npos) {
        clean += "-Turbo";
    }
    return clean;
}

std::string TurboEngine::format_chat(const std::vector<std::pair<std::string, std::string>> & messages) const {
    if (messages.empty()) {
        return "";
    }

    if (model != nullptr) {
        const char * tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl != nullptr) {
            std::vector<llama_chat_message> chat;
            chat.reserve(messages.size());
            for (const auto & m : messages) {
                chat.push_back({ m.first.c_str(), m.second.c_str() });
            }

            std::vector<char> buf(8192);
            int32_t res = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), (int32_t)buf.size());
            if (res > (int32_t)buf.size()) {
                buf.resize(res + 1);
                res = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), (int32_t)buf.size());
            }
            if (res > 0) {
                return std::string(buf.data(), res);
            }
        }
    }

    // Fallback ChatML format
    std::string prompt;
    for (const auto & m : messages) {
        prompt += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

} // namespace devios
