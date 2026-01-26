@echo off
setlocal

:: Get the directory where this script is located
set "ROOT_DIR=%~dp0"
set "MAMBA_EXE=%ROOT_DIR%micromamba.exe"
set "ENV_DIR=%ROOT_DIR%env"

:: 1. Check if the environment already exists
if not exist "%ENV_DIR%" (
    echo ========================================================
    echo  FIRST TIME SETUP - This will take a few minutes...
    echo  Downloading Python, PyTorch, and Napari...
    echo ========================================================
    
    :: Create the environment locally in the /env folder
    call "%MAMBA_EXE%" create -p "%ENV_DIR%" -f "%ROOT_DIR%environment.yml" -y
    
    if errorlevel 1 (
        echo.
        echo [ERROR] Installation failed. Please check your internet connection.
        pause
        exit /b
    )
    echo.
    echo [SUCCESS] Installation complete!
)

:: 2. Activate the environment and run the app
echo Starting LatiSeg-Assist...
call "%MAMBA_EXE%" run -p "%ENV_DIR%" python "%ROOT_DIR%src\main.py"

if errorlevel 1 (
    echo.
    echo [ERROR] The application crashed. See the error message above.
    pause
)