# PowerShell Automation Script for Pi Cam Live
# Usage: .\deploy_and_run.ps1

$Username = "rspi-ai-reader"
$IP = "10.241.229.32"
$DestDir = "~/pi-cam-feed"
$Port = 8080

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Deploying & Launching Pi Cam Live" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Target Pi: $Username@$IP" -ForegroundColor Yellow
Write-Host "Local Directory: $PSScriptRoot" -ForegroundColor Yellow

# 1. SCP Transfer
Write-Host "`n[1/3] Copying files to Pi..." -ForegroundColor Cyan
scp -r "$PSScriptRoot" "${Username}@${IP}:${DestDir}"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to copy files via SCP. Please check network connection and password."
    exit $LASTEXITCODE
}

# 2. SSH Dependency Install & Launch
Write-Host "`n[2/3] Connecting to Pi to install dependencies..." -ForegroundColor Cyan
# Run pip install over SSH
ssh "${Username}@${IP}" "cd $DestDir && pip3 install --user -r requirements.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Dependency installation returned an error. Trying to proceed anyway..."
}

# 3. Launching
Write-Host "`n[3/3] Launching camera stream server on port ${Port}..." -ForegroundColor Cyan
Write-Host "Open your browser at: http://${IP}:${Port}" -ForegroundColor Green
Write-Host "Press Ctrl+C inside the SSH session to stop the stream." -ForegroundColor Yellow
Write-Host "--------------------------------------------------" -ForegroundColor Cyan

ssh -t "${Username}@${IP}" "cd $DestDir && python3 stream_server.py --port $Port"
