@echo off
title Wildlife AI - Train Custom YOLO Detector
echo ================================================================
echo  Wildlife AI - Custom Detection Training (our own model)
echo  Step 1: build dataset   Step 2: train yolo_custom_det.pt
echo ================================================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
cd BACKEND

echo [1/3] Preparing detection dataset...
python training\prepare_detection_dataset.py

echo.
echo [2/3] Adding camera-degradation copies (for live camera)...
python training\augment_detection_dataset.py

echo.
echo [3/3] Training detection model on GPU...
python training\train_yolo_detect.py

echo.
echo Done! Restart start_backend.bat to use the new model.
pause
