# One-click start: Docker -> services -> first model if needed -> browser.
# Run through start.bat (double-click) or directly: powershell -File scripts\start.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$appUrl = "http://localhost:8000"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

function Test-Engine {
    cmd /c "docker info >nul 2>&1"
    return $LASTEXITCODE -eq 0
}

function Start-DockerDesktop {
    $exe = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe) { Fail "Docker Desktop is not installed. Get it from https://www.docker.com/products/docker-desktop/" }

    # After an unclean shutdown Docker Desktop crashes on leftover Unix-socket files that Windows
    # can't open or delete. Moving their folders aside lets Docker recreate them.
    Get-Process | Where-Object { $_.ProcessName -match "^(Docker Desktop|com\.docker\..*)$" } |
        Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($dir in "$env:LOCALAPPDATA\Docker\run", "$env:LOCALAPPDATA\docker-secrets-engine") {
        if (Test-Path $dir) {
            try { Rename-Item $dir "$(Split-Path $dir -Leaf).stale-$stamp" } catch { }
        }
    }
    Start-Process $exe
    for ($i = 0; $i -lt 60; $i++) {
        if (Test-Engine) { return }
        Start-Sleep -Seconds 4
    }
    Fail "Docker did not start within 4 minutes. Open Docker Desktop to see what is wrong."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "the docker command was not found. Install Docker Desktop first."
}

# docker writes progress to stderr; that must not abort the script
$ErrorActionPreference = "Continue"

Step "Checking Docker"
if (Test-Engine) { Write-Host "Docker is running." }
else { Write-Host "Starting Docker Desktop (can take a minute)..."; Start-DockerDesktop; Write-Host "Docker is up." }

Step "Starting MLflow, Prefect and the web app (first run builds the image: a few minutes)"
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { Fail "docker compose up failed (see the output above)." }

Step "Waiting for the web app"
$health = $null
for ($i = 0; $i -lt 60; $i++) {
    try { $health = Invoke-RestMethod "$appUrl/health" -TimeoutSec 3; break } catch { Start-Sleep -Seconds 2 }
}
if (-not $health) { Fail "the web app did not answer. Logs: docker compose logs api" }

if (-not $health.model_loaded) {
    Step "No model yet: running the training pipeline once (about a minute)"
    docker compose run --rm jobs mlwf train
    if ($LASTEXITCODE -ne 0) { Fail "training failed (see the output above)." }
    Invoke-RestMethod "$appUrl/reload" -Method Post | Out-Null
}

Step "Opening $appUrl"
Start-Process $appUrl
Write-Host "`nRunning. MLflow: http://localhost:5000  Prefect: http://localhost:4200" -ForegroundColor Green
Write-Host "To stop everything, double-click stop.bat."
