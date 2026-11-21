@echo off
cd /d "%~dp0"
if not exist reports\screens mkdir reports\screens
rem Grafana: run inside its network namespace so that localhost:3000 (Grafana's root_url) resolves to it
for /f %%c in ('docker compose ps -q grafana') do set GRAFANA_ID=%%c
docker run --rm --network container:%GRAFANA_ID% -e ONLY=grafana -e GRAFANA_URL=http://localhost:3000 -v "%cd%\scripts:/scripts:ro" -v "%cd%\reports\screens:/out" mcr.microsoft.com/playwright/python:v1.49.1-noble sh -c "pip install -q --break-system-packages playwright==1.49.1 && python /scripts/screenshots.py /out"
pause
