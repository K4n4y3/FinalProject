@echo off
rem 2-minute load test against the running service; report -> reports\perf\local_full*.csv / .html
cd /d "%~dp0"
docker compose --profile loadtest run --rm loadtest
echo Report: reports\perf\local_full.html
pause
