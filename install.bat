@echo off
echo ============================================
echo Quantum Respiratory Classification - Setup
echo ============================================
echo.

REM Check Python version
python --version
echo.

echo Choose installation type:
echo [1] CPU only (default)
echo [2] CUDA 11.8
echo [3] CUDA 12.1
echo.
set /p choice="Enter choice (1/2/3): "

if "%choice%"=="" set choice=1

echo.
echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

if "%choice%"=="1" (
    echo Installing PyTorch CPU...
    pip install torch torchvision torchaudio
) else if "%choice%"=="2" (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%choice%"=="3" (
    echo Installing PyTorch with CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo Installing remaining dependencies...
pip install pennylane==0.35.1
pip install pennylane-lightning>=0.35.0
pip install numpy pandas scipy scikit-learn
pip install librosa soundfile
pip install matplotlib seaborn
pip install jupyter ipywidgets tqdm

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo To activate the environment, run:
echo     .venv\Scripts\activate
echo.
echo To verify CUDA:
echo     python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo.
echo To start Jupyter:
echo     jupyter notebook notebooks/07_Final_Comparison.ipynb
echo.
pause

