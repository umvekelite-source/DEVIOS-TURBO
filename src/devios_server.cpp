#include "devios_server.h"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <sstream>
#include <ctime>

using json = nlohmann::json;

namespace devios {

TurboServer::TurboServer(TurboEngine & engine, const ServerConfig & config)
    : engine(engine), config(config) {
}

TurboServer::~TurboServer() {
    stop();
}

bool TurboServer::start() {
    httplib::Server svr;
    svr.set_tcp_nodelay(true); // Disable Nagle algorithm: immediate SSE chunk flush with zero latency

    const std::string model_id = engine.get_model_id();
    const std::string raw_stem = engine.get_raw_model_name();

    // Default CORS headers on all responses
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With"},
        {"Access-Control-Max-Age", "86400"}
    });

    // 1. CORS Preflight for all routes
    svr.Options(".*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
        res.set_header("Access-Control-Max-Age", "86400");
        res.status = 200;
        res.set_content("", "text/plain");
    });

    // 2. GET /v1/models — dynamic model discovery for LM Studio / OpenCode
    auto make_model_entry = [](const std::string & id) -> json {
        return {
            {"id", id},
            {"object", "model"},
            {"created", static_cast<int64_t>(time(nullptr))},
            {"owned_by", "devios-turbo"},
            {"permission", json::array({
                {
                    {"id", "modelperm-" + id},
                    {"object", "model_permission"},
                    {"created", static_cast<int64_t>(time(nullptr))},
                    {"allow_create_engine", false},
                    {"allow_sampling", true},
                    {"allow_logprobs", true},
                    {"allow_search_indices", false},
                    {"allow_view", true},
                    {"allow_fine_tuning", false},
                    {"organization", "*"},
                    {"group", nullptr},
                    {"is_blocking", false}
                }
            })},
            {"root", id},
            {"parent", nullptr}
        };
    };

    svr.Get("/v1/models", [model_id, raw_stem, make_model_entry](const httplib::Request &, httplib::Response & res) {
        json models_arr = json::array();
        models_arr.push_back(make_model_entry(model_id));

        if (model_id != "devios-turbo") {
            models_arr.push_back(make_model_entry("devios-turbo"));
        }
        if (!raw_stem.empty() && raw_stem != model_id && raw_stem != "devios-turbo") {
            models_arr.push_back(make_model_entry(raw_stem));
        }

        json resp = {
            {"object", "list"},
            {"data", models_arr}
        };

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_content(resp.dump(2), "application/json");
    });

    // GET /v1/models/:model (single model inspection)
    svr.Get(R"(/v1/models/(.+))", [make_model_entry](const httplib::Request & req, httplib::Response & res) {
        std::string req_id = req.matches[1];
        json resp = make_model_entry(req_id);
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.set_content(resp.dump(2), "application/json");
    });

    // 3. GET /v1/turbo/metrics — internal telemetry
    svr.Get("/v1/turbo/metrics", [this, model_id](const httplib::Request &, httplib::Response & res) {
        GenerationStats stats = engine.get_last_stats();
        DraftMetrics dm = engine.get_drafter_metrics();
        const CorpusIndex & corpus = engine.get_corpus();

        json j = {
            {"status", "ok"},
            {"model_id", model_id},
            {"tokens_per_sec", stats.gen_speed_ts},
            {"prompt_speed_ts", stats.prompt_speed_ts},
            {"rolling_acceptance_rate", dm.rolling_acceptance_rate},
            {"avg_accepted_per_step", dm.avg_accepted_per_step},
            {"current_k", dm.current_k},
            {"total_drafted", dm.total_drafted},
            {"total_accepted", dm.total_accepted},
            {"corpus_tokens", corpus.size()},
            {"corpus_memory_bytes", corpus.memory_bytes()},
            {"corpus_memory_mb", corpus.memory_bytes() / (1024.0 * 1024.0)}
        };
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_content(j.dump(2), "application/json");
    });

    // POST /v1/corpus/index
    svr.Post("/v1/corpus/index", [this](const httplib::Request & req, httplib::Response & res) {
        try {
            std::lock_guard<std::mutex> lock(mutex_engine);
            auto j = json::parse(req.body);
            std::string text = j.value("text", "");
            std::string save_path = j.value("save_path", "corpus.bin");

            bool ok = engine.index_corpus_text(text, save_path);
            json resp = {
                {"success", ok},
                {"corpus_tokens", engine.get_corpus().size()}
            };
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(std::string("{\"error\": \"") + e.what() + "\"}", "application/json");
        }
    });

    // POST /v1/corpus/clear
    svr.Post("/v1/corpus/clear", [this](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(mutex_engine);
        engine.clear_corpus();
        json resp = {
            {"success", true},
            {"message", "corpus cleared"}
        };
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(resp.dump(), "application/json");
    });

    // 4. POST /v1/chat/completions (OpenAI Chat API with SSE Streaming)
    svr.Post("/v1/chat/completions", [this, model_id](const httplib::Request & req, httplib::Response & res) {
        try {
            auto body = json::parse(req.body);
            bool stream = body.value("stream", false);

            int32_t max_tokens = 512;
            if (body.contains("max_tokens") && body["max_tokens"].is_number_integer()) {
                max_tokens = body["max_tokens"].get<int32_t>();
            } else if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number_integer()) {
                max_tokens = body["max_completion_tokens"].get<int32_t>();
            }
            if (max_tokens <= 0) {
                max_tokens = 2048;
            }

            std::string req_model = body.value("model", model_id);

            // Reconstruct prompt: structured messages or raw prompt
            std::string prompt;
            if (body.contains("messages") && body["messages"].is_array()) {
                std::vector<std::pair<std::string, std::string>> chat_msgs;
                for (const auto & msg : body["messages"]) {
                    std::string role = msg.value("role", "user");
                    std::string content = "";
                    if (msg.contains("content")) {
                        if (msg["content"].is_string()) {
                            content = msg["content"].get<std::string>();
                        } else if (msg["content"].is_array()) {
                            for (const auto & part : msg["content"]) {
                                if (part.is_object() && part.value("type", "") == "text") {
                                    content += part.value("text", "");
                                }
                            }
                        }
                    }
                    chat_msgs.push_back({ role, content });
                }
                prompt = engine.format_chat(chat_msgs);
            } else if (body.contains("prompt")) {
                if (body["prompt"].is_string()) {
                    prompt = body["prompt"].get<std::string>();
                }
            }

            if (prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\": \"empty prompt\"}", "application/json");
                return;
            }

            int64_t created_time = static_cast<int64_t>(time(nullptr));

            if (!stream) {
                // Non-streaming completion
                std::string output;
                GenerationStats stats;
                {
                    std::lock_guard<std::mutex> lock(mutex_engine);
                    output = engine.generate(prompt, max_tokens, nullptr);
                    stats = engine.get_last_stats();
                }

                // Live telemetry printout
                printf("[DEVIOS TURBO GEMMA] Tokens: %d | Time: %.1fs | Real Speed: %.1f tok/s | Draft Acceptance: %.0f%%\n",
                       stats.n_predicted, stats.t_gen_ms / 1000.0, stats.gen_speed_ts, stats.acceptance_rate * 100.0f);
                fflush(stdout);

                json jres = {
                    {"id", "chatcmpl-devios-turbo"},
                    {"object", "chat.completion"},
                    {"created", created_time},
                    {"model", req_model},
                    {"choices", json::array({
                        {
                            {"index", 0},
                            {"message", {
                                {"role", "assistant"},
                                {"content", output}
                            }},
                            {"finish_reason", "stop"}
                        }
                    })},
                    {"usage", {
                        {"prompt_tokens", stats.n_prompt},
                        {"completion_tokens", stats.n_predicted},
                        {"total_tokens", stats.n_prompt + stats.n_predicted}
                    }},
                    {"devios_turbo", {
                        {"tokens_per_sec", stats.gen_speed_ts},
                        {"acceptance_rate", stats.acceptance_rate},
                        {"avg_accepted_per_step", stats.avg_accepted_per_step},
                        {"final_k", stats.final_k}
                    }}
                };

                res.set_header("Access-Control-Allow-Origin", "*");
                res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
                res.set_content(jres.dump(), "application/json");
            } else {
                // Streaming SSE completion for LM Studio / OpenCode UI
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");
                res.set_header("Access-Control-Allow-Origin", "*");
                res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

                res.set_chunked_content_provider("text/event-stream",
                    [this, prompt, max_tokens, req_model, created_time](size_t /*offset*/, httplib::DataSink & sink) -> bool {
                        std::lock_guard<std::mutex> lock(mutex_engine);
                        engine.generate(prompt, max_tokens, [&sink, &req_model, created_time](const std::string & chunk) -> bool {
                            if (chunk.empty()) {
                                return true;
                            }
                            json jchunk = {
                                {"id", "chatcmpl-devios-turbo"},
                                {"object", "chat.completion.chunk"},
                                {"created", created_time},
                                {"model", req_model},
                                {"choices", json::array({
                                    {
                                        {"index", 0},
                                        {"delta", {{"content", chunk}}},
                                        {"finish_reason", nullptr}
                                    }
                                })}
                            };
                            std::string line = "data: " + jchunk.dump() + "\n\n";
                            return sink.write(line.data(), line.size());
                        });

                        // Final stop chunk
                        json jfinal = {
                            {"id", "chatcmpl-devios-turbo"},
                            {"object", "chat.completion.chunk"},
                            {"created", created_time},
                            {"model", req_model},
                            {"choices", json::array({
                                {
                                    {"index", 0},
                                    {"delta", json::object()},
                                    {"finish_reason", "stop"}
                                }
                            })}
                        };
                        std::string final_line = "data: " + jfinal.dump() + "\n\ndata: [DONE]\n\n";
                        sink.write(final_line.data(), final_line.size());
                        sink.done();

                        // Live telemetry printout
                        GenerationStats stats = engine.get_last_stats();
                        printf("[DEVIOS TURBO GEMMA] Tokens: %d | Time: %.1fs | Real Speed: %.1f tok/s | Draft Acceptance: %.0f%%\n",
                               stats.n_predicted, stats.t_gen_ms / 1000.0, stats.gen_speed_ts, stats.acceptance_rate * 100.0f);
                        fflush(stdout);

                        return false;
                    }
                );
            }
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_content(std::string("{\"error\": \"") + e.what() + "\"}", "application/json");
        }
    });

    std::cout << "[c4] DEVIOS Turbo server listening on http://" << config.host << ":" << config.port << std::endl;
    std::cout << "[c4] Active model ID: " << model_id << std::endl;
    running = true;
    return svr.listen(config.host.c_str(), config.port);
}

void TurboServer::stop() {
    running = false;
}

} // namespace devios
