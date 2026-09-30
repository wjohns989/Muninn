"""Read-only isolated Edge check of the live loopback dashboard.

Requires optional Python Playwright and an installed Edge channel. The main
token is read from the process or Windows user environment and never printed.
No transcript search, policy write, credential reveal, or inference occurs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import mkstemp
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def _token() -> str | None:
    value = os.environ.get("MUNINN_AUTH_TOKEN")
    if value or os.name != "nt":
        return value
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "MUNINN_AUTH_TOKEN")
            return value if isinstance(value, str) and value else None
    except FileNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:42069")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--expect-resources-ready", action="store_true")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save a temporary screenshot of nonsecret status UI")
    args = parser.parse_args()
    if not 320 <= args.width <= 2560 or not 600 <= args.height <= 1600:
        parser.error("Viewport is outside the bounded UI test range")
    origin = urlsplit(args.base)
    if (origin.scheme != "http" or origin.hostname != "127.0.0.1"
            or origin.port != 42069 or origin.path or origin.query or origin.fragment):
        parser.error("Only the single loopback Muninn origin is permitted")
    token = _token()
    if not token:
        parser.error("MUNINN_AUTH_TOKEN is unavailable in this process or Windows user environment")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True,
                                             args=["--disable-background-networking"])
        try:
            context = browser.new_context(viewport={"width": args.width, "height": args.height},
                                          service_workers="block", accept_downloads=False)

            def local_only(route):
                target = urlsplit(route.request.url)
                if target.scheme == "http" and target.hostname == "127.0.0.1" and target.port == 42069:
                    route.continue_()
                else:
                    route.abort()

            context.route("**/*", local_only)
            page = context.new_page()
            page.goto(args.base, wait_until="domcontentloaded", timeout=15000)
            expect(page.get_by_placeholder("Paste your Auth Token here...")).to_be_visible()
            page.get_by_placeholder("Paste your Auth Token here...").fill(token)
            page.get_by_role("button", name="Authenticate & Enter").click()
            expect(page.locator("#auth-modal")).to_be_hidden()
            page.get_by_role("button", name="Encrypted History").click()
            if not page.get_by_role("heading", name="Encrypted capture status").is_visible():
                failure = {"state": "history_hidden", "tab_class": page.locator("#tab-history").get_attribute("class"),
                           "nav_class": page.locator('[data-tab="history"]').get_attribute("class"),
                           "document_width": page.evaluate("document.documentElement.scrollWidth")}
                if args.screenshot:
                    handle, name = mkstemp(prefix="muninn-dashboard-fail-", suffix=".png")
                    os.close(handle)
                    page.screenshot(path=name, full_page=True)
                    failure["screenshot"] = str(Path(name).resolve())
                print(json.dumps(failure, sort_keys=True))
                return 2
            expect(page.get_by_role("heading", name="Encrypted capture status")).to_be_visible()
            expect(page.locator("#history-coverage-status")).to_contain_text("Archive generation")

            result = {"authenticated_history_visible": True,
                      "capture_status_visible": page.locator("#history-capture-status").is_visible(),
                      "resources_checked": False}
            if args.expect_resources_ready:
                page.get_by_role("button", name="Check GPU and Ollama").click()
                expect(page.locator("#local-resource-status")).not_to_contain_text(
                    "Sampling", timeout=15000)
                status = page.locator("#local-resource-status").inner_text()
                if "unavailable" in status.lower() or "sampling" in status.lower():
                    raise RuntimeError("Live resource status did not become ready")
                result["resources_checked"] = True
                result["installed_model_rows"] = page.locator("#local-resource-models li").count()
            if args.screenshot:
                handle, name = mkstemp(prefix="muninn-dashboard-", suffix=".png")
                os.close(handle)
                page.screenshot(path=name, full_page=True)
                result["screenshot"] = str(Path(name).resolve())
            print(json.dumps(result, sort_keys=True))
            context.close()
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
