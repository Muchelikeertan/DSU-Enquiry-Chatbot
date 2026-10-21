# ==============================================================================
# DSU Enquiry System - Multi-Server Local Shutdown Script (PowerShell)
# Safely stops python/uvicorn processes listening on ports 8000, 5500, and 5501.
# ==============================================================================

Write-Host "Stopping DSU Enquiry System servers..." -ForegroundColor Cyan

$Ports = @(8000, 5500, 5501)
$StoppedCount = 0

foreach ($Port in $Ports) {
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        if ($connections) {
            foreach ($conn in $connections) {
                $pidVal = $conn.OwningProcess
                if ($pidVal -and $pidVal -gt 4) {
                    $proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
                    if ($proc -and ($proc.ProcessName -match "python|uvicorn")) {
                        Write-Host "Terminating process on port $Port (PID: $pidVal, Name: $($proc.ProcessName))..." -ForegroundColor Yellow
                        Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
                        $StoppedCount++
                    }
                }
            }
        }
    } catch {
        # Ignore individual lookup failures
    }
}

Write-Host "Shutdown complete. $StoppedCount process(es) stopped." -ForegroundColor Green
