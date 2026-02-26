import sys
import threading
import webbrowser
from pathlib import Path
from PIL import Image
import pystray
from pystray import MenuItem as item

def create_tray_icon(server_url, quit_callback):
    """Create and run the system tray icon."""
    # Find icon
    icon_path = Path(__file__).parent / "assets" / "muninn_banner.jpeg"
    if not icon_path.exists():
        # Fallback to a simple colored square if image not found
        image = Image.new('RGB', (64, 64), color=(138, 43, 226))
    else:
        try:
            image = Image.open(icon_path)
            # Resize and crop to square for tray
            width, height = image.size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            image = image.crop((left, top, right, bottom)).resize((64, 64))
        except Exception:
            image = Image.new('RGB', (64, 64), color=(138, 43, 226))

    def on_open_dashboard(icon, item):
        webbrowser.open(server_url)

    def on_quit(icon, item):
        icon.stop()
        if quit_callback:
            quit_callback()

    menu = (
        item('Open Dashboard', on_open_dashboard, default=True),
        item('Quit Muninn', on_quit),
    )

    icon = pystray.Icon("Muninn", image, "Muninn Memory Engine", menu)
    icon.run()

if __name__ == "__main__":
    # Test launch
    create_tray_icon("http://localhost:42069", lambda: sys.exit(0))
