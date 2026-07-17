@echo off
title Wildlife AI — Install Dependencies
echo ================================================
echo  Installing Python dependencies into venv
echo ================================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r BACKEND\requirements.txt

echo.
echo ================================================
echo  Done! You can now run start_backend.bat
echo ================================================
pause
