@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  DEVIOS Turbo -- Starting OpenAI-Compatible Daemon
echo ========================================================

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "BIN=%ROOT_DIR%\llama.cpp\build\bin\devios-turbo-extreme.exe"
if not exist "%BIN%" (
    set "BIN=%ROOT_DIR%\llama.cpp\build\bin\devios-turbo.exe"
)

if not exist "%BIN%" (
    echo [ERROR] DEVIOS Turbo binary not found. Run scripts\build.bat first.
    exit /b 1
)

:: Model path resolution
set "MODEL=%~1"
if "%MODEL%"=="" (
    set "DEFAULT_MODEL=%ROOT_DIR%\models\model.gguf"
    if exist "!DEFAULT_MODEL!" (
        set "MODEL=!DEFAULT_MODEL!"
    ) else (
        echo [ERROR] Model path not provided.
        echo Usage: run_server.bat "path\to\model.gguf" [port]
        exit /b 1
    )
)

set "PORT=%~2"
if "%PORT%"=="" set "PORT=18080"

set "CORPUS=%ROOT_DIR%\corpus.bin"
if not exist "%CORPUS%" (
    if exist "%ROOT_DIR%\data\hybrid_corpus.txt" (
        echo [*] corpus.bin not found, building from hybrid_corpus.txt...
        call "%SCRIPT_DIR%index_corpus.bat" "%MODEL%"
    )
)

echo [*] Binary:      %BIN%
echo [*] Model:       %MODEL%
echo [*] Corpus:      %CORPUS%
echo [*] Port:        %PORT%
echo [*] Physical CPU Cores: 5 (Affinity mask 0x555, Zero-Latency Spin-Wait)
echo [*] KV-Cache:    Q4_0 (Flash Attention enabled)
echo.

"%BIN%" ^
    -m "%MODEL%" ^
    -c "%CORPUS%" ^
    --server --host 127.0.0.1 --port %PORT% ^
    -t 5 --ctx-size 8192 --jinja -fa ^
    --cache-type-k q4_0 --cache-type-v q4_0 --mlock
