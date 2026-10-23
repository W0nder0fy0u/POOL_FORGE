# PoolForge Local Multi-Agent Simulation for Windows PowerShell
# Launches Coordinator + 2 Agents with different RAM caps

$ErrorActionPreference = "Stop"
$Port = 8000
$ServerUrl = "http://127.0.0.1:$Port"
$Py = ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

Write-Host "=========================================================="
Write-Host " Starting PoolForge Coordinator on $ServerUrl..."
Write-Host "=========================================================="

$CoordProc = Start-Process -FilePath $Py -ArgumentList "-m coordinator.server" -PassThru -NoNewWindow
Start-Sleep -Seconds 2

# Create room
$RoomResp = Invoke-RestMethod -Uri "$ServerUrl/api/rooms" -Method Post
$RoomCode = $RoomResp.room_code
$AdminToken = $RoomResp.admin_token

Write-Host "Created Pool Room: $RoomCode"
Write-Host "Admin Token:      $AdminToken"
Write-Host "Join URL:         $ServerUrl/join/$RoomCode"

# Launch Agent 1 (Capped at 1.0 GB RAM)
Write-Host "Starting Agent 1 (Capped at 1.0 GB RAM)..."
$Agent1Proc = Start-Process -FilePath $Py -ArgumentList "agent/cli.py join $RoomCode --server $ServerUrl --node-id laptop-alpha --ram-cap-gb 1.0" -PassThru -NoNewWindow

# Launch Agent 2 (Capped at 2.0 GB RAM)
Write-Host "Starting Agent 2 (Capped at 2.0 GB RAM)..."
$Agent2Proc = Start-Process -FilePath $Py -ArgumentList "agent/cli.py join $RoomCode --server $ServerUrl --node-id laptop-beta --ram-cap-gb 2.0" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "`nBoth agents running! Press Ctrl+C to terminate."

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "`nCleaning up processes..."
    Stop-Process -Id $Agent1Proc.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $Agent2Proc.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $CoordProc.Id -ErrorAction SilentlyContinue
}
