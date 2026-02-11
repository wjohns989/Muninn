@echo off
:loop
echo [%TIME%] Starting Muninn Server... >> monitor.log
c:\Users\wjohn\miniconda3\python.exe server.py
echo [%TIME%] Server crashed or stopped. Restarting in 5 seconds... >> monitor.log
ping -n 5 127.0.0.1 >nul
goto loop
