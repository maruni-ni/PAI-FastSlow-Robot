@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" view_demo.py
) else (
  "..\PAI-FastSlow-Robot\.venv311\Scripts\python.exe" view_demo.py
)
