@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  DEVIOS Turbo -- Suffix Memory Corpus Builder
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

set "MODEL=%~1"
if "%MODEL%"=="" (
    set "MODEL=%ROOT_DIR%\models\model.gguf"
    if not exist "!MODEL!" (
        echo [ERROR] Model path not provided.
        echo Usage: index_corpus.bat "path\to\model.gguf" ["path\to\input.txt"]
        exit /b 1
    )
)

set "INPUT=%~2"
if "%INPUT%"=="" (
    set "INPUT=%ROOT_DIR%\data\hybrid_corpus.txt"
)

set "OUTPUT=%ROOT_DIR%\corpus.bin"

echo [*] Indexing: %INPUT%
echo [*] Target:   %OUTPUT%

"%BIN%" -m "%MODEL%" --index-file "%INPUT%" --save-corpus "%OUTPUT%"

if errorlevel 1 (
    echo [ERROR] Indexing failed.
    exit /b 1
)

echo [SUCCESS] Corpus index saved successfully.
