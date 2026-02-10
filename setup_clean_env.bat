@echo off
REM Clean environment setup for Contrastive Trajectory Encoders
REM This script creates a dedicated conda environment to avoid OpenMP conflicts

echo ========================================
echo Creating clean conda environment
echo ========================================

REM Create new conda environment with Python 3.10 (more stable for ML packages)
conda create -n rl_encoder python=3.10 -y

REM Activate the environment
call conda activate rl_encoder

echo.
echo ========================================
echo Installing PyTorch via conda (recommended)
echo ========================================

REM Install PyTorch through conda to avoid OpenMP conflicts
REM CPU version - change to CUDA if you have GPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

REM If you have CUDA 11.8, use this instead:
REM conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

REM If you have CUDA 12.1, use this instead:
REM conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo.
echo ========================================
echo Installing other dependencies via pip
echo ========================================

REM Install remaining packages via pip
pip install gymnasium>=0.29.0
pip install carl-bench>=1.1.0
pip install stable-baselines3>=2.0.0
pip install sb3-contrib>=2.0.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install scikit-learn>=1.3.0
pip install tensorboard>=2.13.0
pip install pyyaml>=6.0
pip install omegaconf>=2.3.0
pip install tqdm>=4.65.0
pip install einops>=0.7.0

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To use the environment:
echo   1. Run: conda activate rl_encoder
echo   2. Run: python run_pipeline.py
echo.
echo To deactivate: conda deactivate
echo ========================================

pause
