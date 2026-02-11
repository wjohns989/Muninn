import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import subprocess
import time
from pathlib import Path

# Path to the server script
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
PYTHON_EXE = sys.executable

class MuninnService(win32serviceutil.ServiceFramework):
    _svc_name_ = "MuninnMemoryService"
    _svc_display_name_ = "Muninn Memory MCP Service"
    _svc_description_ = "Muninn Memory Backend for AI Assistants"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.process = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        if self.process:
            self.process.terminate()

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        self.main()

    def main(self):
        # Start server.py as a subprocess
        cmd = [PYTHON_EXE, SERVER_SCRIPT]
        
        # Ensure we run in the right directory
        cwd = os.path.dirname(SERVER_SCRIPT)
        
        self.process = subprocess.Popen(cmd, cwd=cwd, stdout=open(os.path.join(cwd, "service.log"), "a"), stderr=subprocess.STDOUT)
        
        # Wait for stop signal
        win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
        
        if self.process:
            self.process.terminate()

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(MuninnService)
