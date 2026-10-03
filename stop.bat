@echo off
rem Double-click to stop all services. Models, runs and data are kept.
cd /d "%~dp0"
docker compose --profile scheduler down
timeout /t 5
