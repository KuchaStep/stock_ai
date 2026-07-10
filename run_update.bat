@echo off
echo ==================================================
echo  Japan Stock AI - Incremental Update Script
echo ==================================================
echo.

echo [1/4] Fetching latest price data (update_latest.py)...
venv\Scripts\python.exe scripts/update_latest.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Data update failed. Stop process.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Recreating features (create_features_v4.py)...
venv\Scripts\python.exe scripts/create_features_v4.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Feature creation failed. Stop process.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Recreating targets (create_target_v4.py)...
venv\Scripts\python.exe scripts/create_target_v4.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Target creation failed. Stop process.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Retraining AI model (train_model_v4.py)...
venv\Scripts\python.exe scripts/train_model_v4.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Model training failed. Stop process.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==================================================
echo All steps completed successfully!
echo ==================================================
pause
