@echo off
echo Setting up Edge-Aware Federated Learning System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Run setup script
python setup.py %*

echo.
echo Setup complete! Press any key to exit...
pause >nul
