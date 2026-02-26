import sys
import subprocess
import os

def main():
    # Capture original args (skip script name)
    # The first arg after the script name is the command to run
    args = sys.argv[1:]
    if not args:
        print("Usage: python silent_mcp.py <command> [args...]")
        return

    # Windows-specific window suppression
    # 0x08000000 is CREATE_NO_WINDOW
    creationflags = 0
    if os.name == 'nt':
        creationflags = 0x08000000

    try:
        # Launch child process with pipes mirrored
        # We use shell=True for 'npx' on Windows to resolve the path correctly if needed
        # but here we prefer direct execution if possible.
        # Check if first arg is 'npx' - if so, we might need shell=True or absolute path
        cmd = args[0]
        if os.name == 'nt' and cmd.lower() == 'npx':
            cmd = 'npx.cmd'
            args[0] = cmd

        process = subprocess.Popen(
            args,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            creationflags=creationflags,
            bufsize=0, # Unbuffered
            universal_newlines=False
        )
        
        # Wait for the process to complete
        process.wait()
        sys.exit(process.returncode)
    except Exception as e:
        # Since this runs under pythonw (no console), errors might be invisible
        # Log to a temporary file if needed for debugging
        # with open("mcp_wrapper_error.log", "a") as f:
        #     f.write(f"Error launching {args}: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
