@echo off
echo ==================================================
echo  Japan Stock AI - Starting Dashboard...
echo ==================================================
echo.
echo Open your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.
venv\Scripts\streamlit.exe run app/app.py
pause
