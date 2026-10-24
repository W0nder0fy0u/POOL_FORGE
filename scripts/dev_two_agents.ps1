# Run a whole pool on ONE Windows machine: coordinator + 2 agents with different RAM caps.
# If this machine has a GPU, laptop-a trains on it and laptop-b stays on the CPU.
#
#   powershell -ExecutionPolicy Bypass -File scripts\dev_two_agents.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\dev_two_agents.ps1 -Train
#
# Close this window or press Ctrl+C to stop everything.
param([switch]$Train, [int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Url = "http://127.0.0.1:$Port"
$Py = ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
New-Item -ItemType Directory -Force -Path run | Out-Null

$env:POOLFORGE_PORT = "$Port"
Write-Host "== Starting coordinator on $Url (log: run\coordinator.log)"
$coord = Start-Process -FilePath $Py -ArgumentList "-m", "coordinator.server" -PassThru -NoNewWindow `
    -RedirectStandardOutput run\coordinator.log -RedirectStandardError run\coordinator.err.log
$procs = @($coord)

try {
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try { Invoke-RestMethod "$Url/health" | Out-Null; $ready = $true; break } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw "Coordinator did not start; see run\coordinator.err.log" }

    $room = Invoke-RestMethod -Method Post "$Url/api/rooms"
    $code = $room.room_code
    $token = $room.admin_token

    Write-Host "== Starting agent laptop-a (1 GB cap) and laptop-b (2 GB cap)"
    $procs += Start-Process -FilePath $Py -PassThru -NoNewWindow -RedirectStandardOutput run\agent-a.log -RedirectStandardError run\agent-a.err.log `
        -ArgumentList "-m", "agent.cli", "join", $code, "--server", $Url, "--node-id", "laptop-a", "--ram-cap-gb", "1", "--workdir", "run\a"
    $procs += Start-Process -FilePath $Py -PassThru -NoNewWindow -RedirectStandardOutput run\agent-b.log -RedirectStandardError run\agent-b.err.log `
        -ArgumentList "-m", "agent.cli", "join", $code, "--server", $Url, "--node-id", "laptop-b", "--ram-cap-gb", "2", "--device", "cpu", "--workdir", "run\b"
    Start-Sleep -Seconds 4

    Write-Host ""
    Write-Host "  Room code   : $code"
    Write-Host "  Admin token : $token"
    Write-Host "  Dashboard   : $Url/room/$code#admin=$token"
    Write-Host "  Join link   : $Url/join/$code"
    Write-Host ""

    if ($Train) {
        Write-Host "== Starting training with ONE API call (mnist_cnn, auto split)"
        $body = '{"model":"mnist_cnn","batch_size":64,"epochs":1,"steps_per_epoch":100}'
        Invoke-RestMethod -Method Post "$Url/api/rooms/$code/jobs" -Headers @{ "X-Admin-Token" = $token } `
            -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
    }

    Write-Host "Open the dashboard link above. Press Ctrl+C to stop."
    while ($true) { Start-Sleep -Seconds 1 }
}
finally {
    Write-Host "Stopping..."
    foreach ($p in $procs) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
