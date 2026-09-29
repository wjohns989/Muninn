"""Fast module entry point for hooks; avoid importing the Muninn package."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("muninn") / "hook_client.py"), run_name="__main__")
