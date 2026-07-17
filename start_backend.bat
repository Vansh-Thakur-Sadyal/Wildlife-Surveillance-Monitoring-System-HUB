@echo off
title Wildlife AI — Backend Server
echo ================================================
echo  Wildlife AI Backend — Starting on port 5055
echo ================================================
echo.

cd /d "%~dp0"

REM Activate venv
call venv\Scripts\activate.bat

REM Install requirements if needed (comment out after first run)
REM pip install -r BACKEND\requirements.txt

cd BACKEND
python app.py

pause
