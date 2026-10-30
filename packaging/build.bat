@echo off
rem Build the portable Windows zip: double-click this file.
rem Needs uv on PATH —  winget install --id=astral-sh.uv -e
setlocal
cd /d "%~dp0.."

where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo   uv was not found on PATH.
    echo   Install it once with:   winget install --id=astral-sh.uv -e
    echo   Then open a new terminal and run this file again.
    echo.
    pause
    exit /b 1
)

rem --no-project keeps uv from treating this repo's venv as the build environment:
rem the builder only needs the standard library, and it installs the real wheels itself.
uv run --no-project python "%~dp0build_bundle.py" %*
if errorlevel 1 (
    echo.
    echo   Build failed — see the output above.
    pause
    exit /b 1
)

echo.
pause
