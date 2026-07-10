@echo off
echo ==================================================
echo  Japan Stock AI - Pipeline Execution Script
echo ==================================================
echo.

echo [1/4] Running data collection (collect_all_stocks_v2.py)...
venv\Scripts\python.exe scripts/collect_all_stocks_v2.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Data collection failed. Stop process.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Creating features (create_features_v4.py)...
venv\Scripts\python.exe scripts/create_features_v4.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Feature creation failed. Stop process.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Creating targets (create_target_v4.py)...
venv\Scripts\python.exe scripts/create_target_v4.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Target creation failed. Stop process.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Training AI model (train_model_v4.py)...
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
