@echo off
title Wildlife AI — YOLO GPU Training
echo ================================================================
echo  YOLO Custom Classification Training on RTX 4070
echo  Dataset : IMAGES_DATASET\  (7 classes, 31,575 images)
echo  Model   : YOLOv8s-cls (small, best accuracy/speed balance)
echo  Epochs  : 50 (patience=15 early-stop)
echo  Output  : BACKEND\saved_models\yolo_custom_cls.pt
echo ================================================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
cd BACKEND

echo [1/2] Preparing dataset structure...
python training\prepare_yolo_dataset.py

echo.
echo [2/2] Training YOLOv8 on GPU...
python training\train_yolo_custom.py

echo.
echo ================================================================
echo  YOLO training complete!
echo  Training plots saved in: BACKEND\yolo_runs\animal_cls\
echo ================================================================
pause
