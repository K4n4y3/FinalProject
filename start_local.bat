@echo off
rem Menu from Neural Networks - local deployment (Docker Desktop): service + Prometheus + Grafana
chcp 65001 >nul
cd /d "%~dp0"
where docker >nul 2>&1 || (echo [x] Docker not found. Install Docker Desktop. & pause & exit /b 1)
docker info >nul 2>&1 || (echo [x] Docker Desktop is not running. Start it, wait for "Engine running" and retry. & pause & exit /b 1)
if not exist artifacts\manifest.json (echo [x] artifacts\manifest.json not found: copy the Kaggle "bundle" folder contents into artifacts\ & pause & exit /b 1)
echo [1/3] Building and starting containers (first time ~5-10 min)...
docker compose up -d --build || (pause & exit /b 1)
echo [2/3] Waiting for the models to load (first start downloads Qwen ~1 GB)...
powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 180;$i++){ try { $h=Invoke-RestMethod http://localhost:7860/health -TimeoutSec 3; Write-Host ('  status: ' + $h.status); if($h.status -ne 'loading'){ $ok=$true; break } } catch { Write-Host '  starting...' }; Start-Sleep 5 }; if(-not $ok){ exit 1 }"
echo [3/3] Opening UI, API docs and Grafana
start "" http://localhost:7860/ui
start "" http://localhost:7860/docs
start "" http://localhost:3000
echo.
echo UI:          http://localhost:7860/ui
echo API docs:    http://localhost:7860/docs
echo Metrics:     http://localhost:7860/metrics
echo Prometheus:  http://localhost:9090
echo Grafana:     http://localhost:3000   (dashboard "Menu from Neural Networks")
echo Stop:        stop_local.bat
pause
