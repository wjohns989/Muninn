import os
import shutil
from pathlib import Path

def repair_fastembed_cache():
    # FastEmbed uses Temp directory by default on Windows if not specified
    # The logs show it's in AppData\Local\Temp\fastembed_cache
    temp_dir = os.environ.get('TEMP') or os.environ.get('TMP')
    if not temp_dir:
        print("Could not determine TEMP directory.")
        return

    cache_root = Path(temp_dir) / "fastembed_cache"
    model_dir = cache_root / "models--nomic-ai--nomic-embed-text-v1.5"

    print(f"Checking for FastEmbed cache at: {model_dir}")

    if model_dir.exists():
        print(f"Corrupted cache found. Deleting {model_dir}...")
        try:
            shutil.rmtree(model_dir)
            print("Successfully cleared cache. FastEmbed will re-download model on next start.")
        except Exception as e:
            print(f"Error clearing cache: {e}")
            print("Please ensure no other processes are using the model files.")
    else:
        print("Problematic cache directory not found. Trying to clear entire fastembed_cache root...")
        if cache_root.exists():
             try:
                shutil.rmtree(cache_root)
                print(f"Cleared {cache_root}")
             except Exception as e:
                print(f"Error: {e}")
        else:
            print("FastEmbed cache root not found. Nothing to repair.")

if __name__ == "__main__":
    repair_fastembed_cache()
