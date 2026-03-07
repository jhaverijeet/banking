@echo off
REM ────────────────────────────────────────────────────────────────
REM  SONIA Rate Pipeline – Daily Scheduler Setup (Windows)
REM
REM  Creates a Windows Task Scheduler task that runs the SONIA
REM  pipeline daily at 11:00 AM (after BoE typically publishes
REM  the previous day's data by 10:00 AM).
REM
REM  Run this script once WITH ADMIN privileges:
REM    Right-click → "Run as administrator"
REM ────────────────────────────────────────────────────────────────

SET TASK_NAME=SONIA_Rate_Pipeline_Daily
SET PYTHON_EXE=py
SET SCRIPT_DIR=%~dp0
SET WORKING_DIR=%SCRIPT_DIR%..

echo.
echo ═══════════════════════════════════════════════════════
echo   Setting up SONIA Rate Pipeline – Daily Scheduler
echo ═══════════════════════════════════════════════════════
echo.
echo   Task Name   : %TASK_NAME%
echo   Schedule    : Daily at 11:00 AM
echo   Working Dir : %WORKING_DIR%
echo.

REM Delete existing task if present (ignore errors)
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

REM Create the scheduled task
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%PYTHON_EXE%\" -m sonia_pipeline run --mode daily" ^
  /SC DAILY ^
  /ST 11:00 ^
  /SD %date:~0,2%/%date:~3,2%/%date:~6,4% ^
  /RL HIGHEST ^
  /F

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo   [OK] Task created successfully!
    echo   The pipeline will run daily at 11:00 AM.
    echo.
    echo   To run it manually:
    echo     schtasks /Run /TN "%TASK_NAME%"
    echo.
    echo   To remove the task:
    echo     schtasks /Delete /TN "%TASK_NAME%" /F
    echo.
) ELSE (
    echo.
    echo   [ERROR] Failed to create task. Make sure you run
    echo          this script as Administrator.
    echo.
)

pause
