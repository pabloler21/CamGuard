@echo off
cd /d "%~dp0"
where uv >nul 2>nul || powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.11.1/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run --no-dev python app.py
pause
