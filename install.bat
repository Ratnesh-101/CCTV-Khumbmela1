@echo off
cd /d "%~dp0"
echo Creating venv if missing...
if not exist ".venv\Scripts\python.exe" python -m venv .venv
echo.
echo Installing dependencies (PyTorch is ~115MB — this can take 10-20+ minutes)...
echo Run this window from Explorer or CMD and leave it open until you see "DONE".
echo.
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
  echo.
  echo INSTALL FAILED — copy the error above.
  pause
  exit /b 1
)
echo.
echo DONE. Start the app with:
echo   .venv\Scripts\streamlit.exe run app.py
echo.
pause
