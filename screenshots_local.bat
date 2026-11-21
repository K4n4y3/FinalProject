@echo off
rem Screenshots of the running stack (UI, Swagger, Grafana, Prometheus) -> reports\screens
cd /d "%~dp0"
if not exist reports\screens mkdir reports\screens
docker run --rm --network menufromnn_default -v "%cd%\scripts:/scripts:ro" -v "%cd%\reports\screens:/out" mcr.microsoft.com/playwright/python:v1.49.1-noble python /scripts/screenshots.py /out
pause
