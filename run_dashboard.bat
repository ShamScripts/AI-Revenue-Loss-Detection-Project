@echo off
cd /d "%~dp0"
echo Starting dashboard...
python -m streamlit run app/streamlit_app.py
if errorlevel 1 pause
