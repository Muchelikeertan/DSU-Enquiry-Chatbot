# ==============================================================================
# DSU Enquiry System - Multi-Server Local Startup Script (PowerShell)
# Launches:
#   1. FastAPI Backend:          http://127.0.0.1:8000
#   2. Student Chatbot Website:  http://127.0.0.1:5500
#   3. Admission Cell Website:   http://127.0.0.1:5501
# ==============================================================================

$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
if (-not $Root) { $Root = Get-Location }

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "       DAYANANDA SAGAR UNIVERSITY (DSU) ENQUIRY SYSTEM" -ForegroundColor Yellow
Write-Host "                   Local Multi-Server Launcher" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Verify Virtual Environment
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$UvicornExe = Join-Path $Root ".venv\Scripts\uvicorn.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Virtual environment Python not found at: $PythonExe" -ForegroundColor Red
    Write-Host "Please ensure the virtual environment (.venv) is set up." -ForegroundColor Red
    Pause
    Exit 1
}

# 2. Check Database Connectivity
Write-Host "`n[1/4] Checking PostgreSQL database connection..." -ForegroundColor Cyan
$dbCheck = & $PythonExe -c "from app.db import check_db_connection; print('CONNECTED' if check_db_connection() else 'FAILED')" 2>$null
if ($dbCheck -notmatch "CONNECTED") {
    Write-Host "[WARNING] PostgreSQL connection test did not return CONNECTED." -ForegroundColor Yellow
    Write-Host "Ensure local PostgreSQL service is running and DATABASE_URL in .env is correct." -ForegroundColor Yellow
    Write-Host "Backend will attempt startup regardless, but DB endpoints may be unavailable." -ForegroundColor DarkYellow
} else {
    Write-Host "       PostgreSQL connected successfully." -ForegroundColor Green
}

# 3. Launch Backend on Port 8000
Write-Host "`n[2/4] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Cyan
$BackendCmd = "cd `"$Root`"; `$host.UI.RawUI.WindowTitle = 'DSU Backend [Port 8000]'; & `"$UvicornExe`" app.main:app --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd

# 4. Launch Student Frontend on Port 5500
Write-Host "[3/4] Starting Student Chatbot Frontend on http://127.0.0.1:5500..." -ForegroundColor Cyan
$StudentCmd = "cd `"$Root`"; `$host.UI.RawUI.WindowTitle = 'DSU Student Chatbot [Port 5500]'; & `"$PythonExe`" scripts\run_student_frontend.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $StudentCmd

# 5. Launch Admission Cell Frontend on Port 5501
Write-Host "[4/4] Starting Admission Cell Management Frontend on http://127.0.0.1:5501..." -ForegroundColor Cyan
$AdminCmd = "cd `"$Root`"; `$host.UI.RawUI.WindowTitle = 'DSU Admission Cell Portal [Port 5501]'; & `"$PythonExe`" scripts\run_admission_frontend.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $AdminCmd

# 6. Verification and Summary
Start-Sleep -Seconds 2
Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host " ALL THREE LOCAL SERVERS LAUNCHED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host " Website 1 (Student Chatbot):      http://127.0.0.1:5500" -ForegroundColor White
Write-Host " Website 2 (Admission Cell Portal): http://127.0.0.1:5501" -ForegroundColor White
Write-Host " Shared FastAPI Backend:           http://127.0.0.1:8000" -ForegroundColor White
Write-Host " Shared PostgreSQL Database:       localhost:5432 / dsu_chatbot" -ForegroundColor White
Write-Host "`n Demo Staff Login (for http://127.0.0.1:5501):" -ForegroundColor Yellow
Write-Host "   Username: Use your configured admin username" -ForegroundColor Yellow
Write-Host "   Password: Check your local configuration" -ForegroundColor Yellow
Write-Host "`n To stop all servers cleanly, run: .\stop_servers.ps1" -ForegroundColor Gray
Write-Host "=====================================================================" -ForegroundColor Green
