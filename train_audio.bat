@echo off
title Wildlife AI — Audio CNN GPU Training
echo ================================================================
echo  Audio CNN Training on RTX 4070 (PyTorch + CUDA)
echo  Dataset : AUDIO_DATASET\  (7 classes, ~350 WAV files)
echo  Augment : ON  (doubles dataset via noise + time-shift)
echo  Epochs  : 60 (early-stop at patience=12)
echo  Output  : BACKEND\saved_models\audio_classifier.pth
echo ================================================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
cd BACKEND

python training\train_audio_model.py

echo.
echo ================================================================
echo  Audio training complete!
echo  Check accuracy above. Expected: 85-95%% on RTX 4070
echo ================================================================
pause
