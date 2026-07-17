@echo off
echo ============================================================
echo   Audio BOOST Training - SE-ResNet + Focal Loss + Mixup
echo   Targets: CAT / LION / SHEEP (were at 33%% accuracy)
echo ============================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Run install_gpu.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
cd BACKEND
python training\train_audio_boost.py
cd ..

echo.
echo ============================================================
echo   Done! Boost model saved to BACKEND\saved_models\audio_classifier_boost.pth
echo   Restart the Flask server to load the new model.
echo ============================================================
pause
