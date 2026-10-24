@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  PilKB one-click launcher - double-click to run on Windows
::  - Backend: FastAPI/uvicorn -> http://127.0.0.1:8000
::  - Frontend: Next.js dev    -> http://localhost:3000
::  Browser opens automatically when frontend is ready.
::  Two service windows stay open; close a window to stop that service.
::  If the browser does not open, visit http://localhost:3000 manually.
::
::  Paths are derived from this script's own location, so moving the
::  project or switching drive letters needs no edit in this file.
:: ============================================================

:: ---------- config - auto-detected; override with env vars if needed ----------
:: %~dp0 is the folder holding this .bat, with a trailing backslash.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
:: Project virtualenv. Override with PILKB_PYTHON if yours lives elsewhere.
set "PYVENV=%ROOT%\.venv\Scripts\python.exe"
:: Leave NODEDIR empty when node/npm are already on PATH (the usual case).
:: Set it to a Node install dir to prepend that dir to PATH.
set "NODEDIR="

if defined PILKB_ROOT set "ROOT=%PILKB_ROOT%"
if defined PILKB_PYTHON set "PYVENV=%PILKB_PYTHON%"

:: Drop inherited values that would change how the services start.
:: PORT matters most: Next.js reads it first and would bind that port
:: instead of 3000 (PORT=0 picks a random free port).
set "PORT="
set "NODE_OPTIONS="
set "CODEBUDDY_SESSION_ID="
set "CLAUDE_SESSION_ID="
set "NEXT_PUBLIC_API_BASE=http://localhost:8000"

if defined NODEDIR set "PATH=%NODEDIR%;%PATH%"

echo [PilKB] root   = %ROOT%
echo [PilKB] python = %PYVENV%

echo [PilKB] checking runtimes...
if not exist "%PYVENV%" (
  echo [ERROR] project python not found:
  echo         %PYVENV%
  echo         Create it once with:
  echo             python -m venv .venv
  echo             .venv\Scripts\python -m pip install -r requirements.txt
  echo         or set PILKB_PYTHON to an interpreter that already has the deps.
  pause
  goto :eof
)
where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm not found on PATH.
  echo         Install Node.js, or set NODEDIR near the top of this file.
  pause
  goto :eof
)
if not exist "%ROOT%\app\main.py" (
  echo [ERROR] backend not found: %ROOT%\app\main.py
  pause
  goto :eof
)
if not exist "%ROOT%\frontend\package.json" (
  echo [ERROR] frontend not found: %ROOT%\frontend\package.json
  pause
  goto :eof
)

:: ---------- .env - the app loads it relative to its working dir ----------
if not exist "%ROOT%\.env" (
  if exist "%ROOT%\.env.example" (
    echo [PilKB] .env missing - creating it from .env.example
    copy /y "%ROOT%\.env.example" "%ROOT%\.env" >nul
  ) else (
    echo [WARN] .env missing and there is no .env.example to copy from.
  )
)

:: ---------- frontend deps ----------
if not exist "%ROOT%\frontend\node_modules" (
  echo [PilKB] frontend\node_modules missing - running npm install, this takes a while...
  pushd "%ROOT%\frontend"
  call npm install
  popd
)

:: ---------- port pre-check - avoid silent EADDRINUSE ----------
echo [PilKB] pre-checking ports 8000 and 3000 ...
curl -sf http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 (
  echo [ERROR] port 8000 is occupied - backend may already be running.
  echo         Close the previous backend window or free port 8000, then re-run.
  pause
  goto :eof
)
curl -sf http://localhost:3000 >nul 2>&1
if not errorlevel 1 (
  echo [ERROR] port 3000 is occupied - frontend may already be running.
  echo         Close the previous frontend window or free port 3000, then re-run.
  pause
  goto :eof
)

:: ---------- start backend ----------
:: /D sets the working directory (config.py loads .env from it), and the exe
:: path is quoted on its own so a project path containing spaces still works.
:: Do not fold this into one big "cd /d ... && ..." string: the && breaks the
:: quoting and cmd then fails with "not recognized as an internal command".
echo [PilKB] starting backend - uvicorn on :8000
start "PilKB-Backend" /D "%ROOT%" cmd /k "%PYVENV%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

:: ---------- wait for backend health ----------
set "ATT=0"
:wait_backend
curl -sf http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 goto backend_ok
set /a ATT+=1
if %ATT% geq 40 (
  echo [PilKB] backend start timed out. Check the PilKB-Backend window for errors.
  pause
  goto :eof
)
timeout /t 1 >nul
goto wait_backend
:backend_ok
echo [PilKB] backend ready: http://127.0.0.1:8000

:: ---------- start frontend ----------
echo [PilKB] starting frontend - Next.js on :3000
start "PilKB-Frontend" /D "%ROOT%\frontend" cmd /k "npm run dev -- -p 3000"

:: ---------- wait for frontend ----------
set "ATT=0"
:wait_frontend
curl -sf http://localhost:3000 >nul 2>&1
if not errorlevel 1 goto frontend_ok
set /a ATT+=1
if %ATT% geq 90 (
  echo [PilKB] frontend start timed out. Check the PilKB-Frontend window for errors.
  pause
  goto :eof
)
timeout /t 1 >nul
goto wait_frontend
:frontend_ok
echo [PilKB] frontend ready, opening browser...
start "" http://localhost:3000

:end
echo.
echo [PilKB] done. Keep the two service windows open. Close a window to stop that service.
echo If the browser did not open, visit http://localhost:3000 manually.
echo.
pause
endlocal
