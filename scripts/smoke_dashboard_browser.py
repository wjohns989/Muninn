"""Isolated Edge check of the live loopback dashboard or checked-out candidate.

Requires optional Python Playwright and an installed Edge channel. The main
token is read from the process or Windows user environment and never printed.
Transcript search enqueues a durable read job only with --transcript-query.
No policy write, credential reveal, or model inference occurs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from tempfile import mkstemp
from urllib.parse import urlsplit

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


def _assert_window_counts(report: dict, text: str) -> None:
    """Compare complete numeric fields, never count-containing substrings."""
    matched = re.match(r'^Recorded windows \(all capture jobs\): (\d+) total; (.*?)\. Retry includes ', text)
    try:
        actual = {}
        for field in matched.group(2).split(', '):
            number, label = field.split(' ', 1)
            key = label.replace(' ', '_')
            if key in actual or not number.isdecimal():
                raise ValueError
            actual[key] = int(number)
        expected = report['states']
        if (report.get('basis') != 'all_capture_lane_jobs'
                or int(matched.group(1)) != report['total']
                or any(actual.get(key, 0) != expected.get(key, 0)
                       for key in set(actual) | set(expected))):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Rendered window counts differ from their backend response') from exc


def main() -> int:
    from playwright.sync_api import expect, sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:42069")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--expect-resources-ready", action="store_true")
    parser.add_argument("--keyboard-nav", action="store_true",
                        help="Exercise every sidebar action with Enter or Space")
    parser.add_argument("--home-status", action="store_true",
                        help="Check authenticated service, archive, queue, and hook status on Home")
    parser.add_argument("--operating-status", action="store_true",
                        help="Check interpretation progress and actual provider-key usage display")
    parser.add_argument("--window-status", action="store_true",
                        help="Compare rendered window counts with their actual backend response")
    parser.add_argument("--credential-query", help="Nonsecret metadata query; prints only match count")
    parser.add_argument("--candidate-html", action="store_true",
                        help="Render checked-out dashboard HTML against the real loopback backend")
    parser.add_argument("--transcript-query", help="Real encrypted-history query; prints only page metadata")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save a temporary screenshot of nonsecret status UI")
    args = parser.parse_args()
    if not 320 <= args.width <= 2560 or not 600 <= args.height <= 1600:
        parser.error("Viewport is outside the bounded UI test range")
    if args.credential_query is not None and not 1 <= len(args.credential_query) <= 64:
        parser.error("Credential metadata query must be 1–64 characters")
    if (args.credential_query is not None or args.transcript_query is not None) and args.screenshot:
        parser.error("Do not save a screenshot of credential or transcript results")
    if args.transcript_query is not None and not 1 <= len(args.transcript_query) <= 240:
        parser.error("Transcript query must be 1–240 characters")
    origin = urlsplit(args.base)
    if (origin.scheme != "http" or origin.hostname != "127.0.0.1"
            or origin.port != 42069 or origin.path or origin.query or origin.fragment):
        parser.error("Only the single loopback Muninn origin is permitted")
    token = _token()
    if not token:
        parser.error("MUNINN_AUTH_TOKEN is unavailable in this process or Windows user environment")
    candidate = (Path(__file__).resolve().parents[1] / "dashboard.html").read_bytes() if args.candidate_html else None

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True,
                                             args=["--disable-background-networking"])
        try:
            context = browser.new_context(viewport={"width": args.width, "height": args.height},
                                          service_workers="block", accept_downloads=False)

            def local_only(route):
                target = urlsplit(route.request.url)
                if target.scheme == "http" and target.hostname == "127.0.0.1" and target.port == 42069:
                    if candidate is not None and target.path in ("", "/") and route.request.method == "GET":
                        route.fulfill(status=200, content_type="text/html; charset=utf-8", body=candidate)
                    else:
                        route.continue_()
                else:
                    route.abort()

            context.route("**/*", local_only)
            page = context.new_page()
            window_reports = []
            if args.window_status:
                def collect_status(response):
                    if response.url == args.base + '/history/status' and response.status == 200:
                        data = response.json().get('data', {}).get('capture_enrichment')
                        if isinstance(data, dict) and isinstance(data.get('window_jobs'), dict):
                            window_reports.append(data)
                page.on('response', collect_status)
            page.goto(args.base, wait_until="domcontentloaded", timeout=15000)
            expect(page.get_by_placeholder("Paste your Auth Token here...")).to_be_visible()
            page.get_by_placeholder("Paste your Auth Token here...").fill(token)
            page.get_by_role("button", name="Authenticate & Enter").click()
            expect(page.locator("#auth-modal")).to_be_hidden()
            if args.home_status:
                expect(page.locator("#overview-service-status")).to_contain_text(
                    "reports ok", timeout=15000)
                expect(page.locator("#overview-history-status")).to_contain_text(
                    "Archive generation", timeout=15000)
                expect(page.locator("#overview-queue-status")).to_contain_text(
                    "Capture intents", timeout=15000)
                expect(page.locator("#overview-hooks-status")).to_contain_text(
                    "accepted endpoint invocations", timeout=15000)
                page.get_by_role("button", name="Refresh local status").click()
                expect(page.locator("#overview-refreshed-status")).to_contain_text(
                    "Local status checked at", timeout=15000)
            keyboard_nav_checked = 0
            if args.keyboard_nav:
                page.get_by_role("button", name="Overview", exact=True).focus()
                page.keyboard.press("Tab")
                expect(page.get_by_role("button", name="Ingestion", exact=True)).to_be_focused()
                for index, (name, tab) in enumerate((
                    ("Overview", "overview"), ("Ingestion", "ingest"),
                    ("Ordinary Search", "search"), ("Encrypted History", "history"),
                    ("Credential Metadata", "credentials"), ("System", "system"),
                )):
                    nav = page.get_by_role("button", name=name, exact=True)
                    nav.focus()
                    expect(nav).to_be_focused()
                    page.keyboard.press("Enter" if index % 2 == 0 else "Space")
                    expect(page.locator(f"#tab-{tab}")).to_have_class(re.compile(r"\bactive\b"))
                    expect(nav).to_have_attribute("aria-current", "page")
                    keyboard_nav_checked += 1
                profile = page.get_by_role("button", name="User Profile", exact=True)
                profile.focus()
                page.keyboard.press("Enter")
                expect(page.locator("#profile-modal")).to_have_class(re.compile(r"\bactive\b"))
                page.locator("#profile-modal button", has_text="Cancel").click()
                keyboard_nav_checked += 1
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
                      "resources_checked": False, "candidate_html": args.candidate_html}
            if args.operating_status:
                expect(page.locator("#history-interpretation-status")).to_contain_text(
                    "pending source versions", timeout=15000)
                expect(page.locator("#history-interpretation-status")).to_contain_text(
                    "Enrollment is not interpretation completion.")
                expect(page.locator("#remote-key-status")).to_contain_text(
                    "Dedicated-key usage", timeout=15000)
                result["operating_status_checked"] = True
            if args.window_status:
                expect(page.locator('#history-window-status')).to_contain_text(
                    'Recorded windows (all capture jobs)', timeout=15000)
                expect(page.locator('#history-batch-status')).to_contain_text('Up to 128 windows per batch')
                if not window_reports:
                    raise RuntimeError('Rendered window counts have no backend response')
                report = window_reports[-1]['window_jobs']
                text = page.locator('#history-window-status').inner_text()
                _assert_window_counts(report, text)
                result['window_status_checked'] = True
                result['recorded_window_jobs'] = report['total']
            if args.keyboard_nav:
                result["keyboard_nav_checked"] = keyboard_nav_checked
            if args.home_status:
                result["home_status_checked"] = True
            if args.expect_resources_ready:
                page.get_by_role("button", name="Check GPU and Ollama").click()
                expect(page.locator("#local-resource-status")).not_to_contain_text(
                    "Sampling", timeout=15000)
                status = page.locator("#local-resource-status").inner_text()
                if "unavailable" in status.lower() or "sampling" in status.lower():
                    raise RuntimeError("Live resource status did not become ready")
                result["resources_checked"] = True
                result["installed_model_rows"] = page.locator("#local-resource-models li").count()
            if args.credential_query is not None:
                page.get_by_role("button", name="Credential Metadata").click()
                expect(page.get_by_role("heading", name="Credential metadata")).to_be_visible()
                page.get_by_label("Metadata query").fill(args.credential_query)
                page.get_by_role("button", name="Search metadata").click()
                expect(page.locator("#credential-metadata-status")).to_have_text(
                    re.compile(r"metadata match|No metadata matches", re.I), timeout=15000)
                result["credential_metadata_checked"] = True
                result["credential_match_count"] = page.locator(
                    "#credential-metadata-results .result-item").count()
            if args.transcript_query is not None:
                page.get_by_role("button", name="Encrypted History").click()
                page.get_by_label("Search encrypted history").fill(args.transcript_query)
                page.get_by_role("button", name="Search History").click()
                expect(page.locator("#history-results .result-item").first).to_be_visible(timeout=180000)
                page.get_by_role("button", name="Open redacted transcript").first.click()
                expect(page.locator("#history-transcript-status")).to_contain_text(
                    "page 1", timeout=300000)
                first_length = page.locator("#history-transcript-page").evaluate(
                    "node => Array.from(node.textContent).length")
                if not 0 <= first_length <= 4000:
                    raise RuntimeError("First transcript page exceeded bound")
                expect(page.get_by_role("button", name="Next page")).to_be_visible()
                page.get_by_role("button", name="Next page").click()
                expect(page.locator("#history-transcript-status")).to_contain_text("page 2", timeout=30000)
                second_length = page.locator("#history-transcript-page").evaluate(
                    "node => Array.from(node.textContent).length")
                if not 0 <= second_length <= 4000:
                    raise RuntimeError("Second transcript page exceeded bound")
                result["transcript_pages_checked"] = 2
                result["transcript_chars_checked"] = first_length + second_length
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
