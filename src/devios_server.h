#pragma once

#include "devios_engine.h"

#include <string>
#include <mutex>

namespace devios {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
};

class TurboServer {
public:
    TurboServer(TurboEngine & engine, const ServerConfig & config);
    ~TurboServer();

    bool start();
    void stop();

private:
    TurboEngine & engine;
    ServerConfig config;
    std::mutex mutex_engine;
    bool running = false;
};

} // namespace devios
