@echo off
title Wildlife AI — GPU Setup (RTX 5070 Ti / CUDA 12.8)
echo ================================================================
echo  Wildlife AI — GPU Installation (RTX 5070 Ti, Blackwell sm_120)
echo ================================================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat

echo [Step 1/3] Installing PyTorch with CUDA 12.8 (for RTX 5070 Ti)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo.
echo [Step 2/3] Installing other requirements...
pip install -r BACKEND\requirements.txt

echo.
echo [Step 3/3] Verifying GPU access...
python -c "import torch; print('[OK] PyTorch:', torch.__version__); print('[OK] CUDA:', torch.cuda.is_available()); print('[OK] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
python -c "from ultralytics import YOLO; print('[OK] Ultralytics ready')"
python -c "import librosa; print('[OK] Librosa ready')"

echo.
echo ================================================================
echo  GPU setup complete!
echo  Next steps:
echo    1. run train_audio.bat   (audio CNN training on GPU)
echo    2. run train_yolo.bat    (YOLO fine-tuning on GPU)
echo    3. run start_backend.bat (start Flask server)
echo ================================================================
pause
