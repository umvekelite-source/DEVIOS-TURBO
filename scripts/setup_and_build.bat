@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  DEVIOS Turbo -- Setup and Build from Scratch
echo ========================================================

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "LLAMA_DIR=%ROOT_DIR%\llama.cpp"
set "PATCH_FILE=%ROOT_DIR%\devios_turbo.patch"

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH.
    exit /b 1
)

if not exist "%LLAMA_DIR%" (
    echo [*] Cloning upstream llama.cpp repository...
    git clone https://github.com/ggerganov/llama.cpp.git "%LLAMA_DIR%"
    if errorlevel 1 (
        echo [ERROR] Git clone failed.
        exit /b 1
    )
    cd /d "%LLAMA_DIR%"
    echo [*] Checking out verified stable commit 7d7f028...
    git checkout 7d7f0287530bc8981690f361d8c1627dbae1c99a
    echo [*] Applying DEVIOS Turbo patch...
    git apply "%PATCH_FILE%"
    if errorlevel 1 (
        echo [ERROR] Patch application failed.
        exit /b 1
    )
) else (
    echo [*] llama.cpp directory already exists at: %LLAMA_DIR%
)

call "%SCRIPT_DIR%build.bat"
