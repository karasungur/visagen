@echo off
REM Visagen Installation Script (Windows wrapper)
REM Usage: install.bat

python install.py %*
if errorlevel 1 (
    echo.
    echo Installation failed. Check install.log for details.
    pause
)
