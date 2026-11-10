@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  DEVIOS Turbo -- Building High-Performance Binary
echo ========================================================

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "LLAMA_DIR=%ROOT_DIR%\llama.cpp"

if not exist "%LLAMA_DIR%" (
    echo [ERROR] llama.cpp directory not found at: %LLAMA_DIR%
    echo Please run setup_and_build.bat first.
    exit /b 1
)

:: Find MSVC vcvars64.bat
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    set "VS_DIR=%%i"
)

if not defined VS_DIR (
    echo [ERROR] Visual Studio 2022 with C++ x64 tools not found.
    exit /b 1
)

call "%VS_DIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to initialize vcvars64 environment.
    exit /b 1
)

cd /d "%LLAMA_DIR%"

echo [*] Configuring CMake with AVX2 and FMA...
cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_AVX2=ON ^
    -DGGML_FMA=ON ^
    -DGGML_AVX=ON ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF

if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo [*] Compiling devios-turbo target...
cmake --build build --config Release --target devios-turbo

if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo ========================================================
echo  [SUCCESS] Binary built at:
echo  %LLAMA_DIR%\build\bin\devios-turbo-extreme.exe
echo ========================================================
