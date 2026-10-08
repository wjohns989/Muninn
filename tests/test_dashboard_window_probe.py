"""Exact backend/render agreement must not confuse one with eleven."""
import pytest

from scripts.smoke_dashboard_browser import _assert_window_counts


@pytest.mark.parametrize('total,pending', [(1, 1), (11, 1), (1, 11), (11, 11)])
def test_counts_compare_as_numbers_not_substrings(total, pending):
    report = {'basis': 'all_capture_lane_jobs', 'total': 1, 'states': {'pending': 1}}
    text = (f'Recorded windows (all capture jobs): {total} total; 0 succeeded, {pending} pending, '
            '0 outcome unknown. Retry includes 0 privacy-parked windows, not runnable work.')
    if (total, pending) == (1, 1):
        _assert_window_counts(report, text)
    else:
        with pytest.raises(RuntimeError, match='^Rendered window counts differ'):
            _assert_window_counts(report, text)


def test_no_fields_or_duplicate_fields_do_not_prove_rendered_counts():
    report = {'basis': 'all_capture_lane_jobs', 'total': 1, 'states': {'pending': 1}}
    for text in ('unknown', 'Recorded windows (all capture jobs): 1 total; 1 pending, 1 pending. Retry includes '):
        with pytest.raises(RuntimeError, match='^Rendered window counts differ'):
            _assert_window_counts(report, text)


def test_browser_probe_matches_current_accessible_auth_and_all_sidebar_tabs():
    from html.parser import HTMLParser
    from pathlib import Path
    from scripts.smoke_dashboard_browser import AUTH_LABEL, NAV_LINKS
    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.fields, self.links = [], []
        def handle_starttag(self, tag, attrs):
            row = dict(attrs)
            if tag == "input" and row.get("type") == "password":
                self.fields.append(row.get("aria-label"))
            if row.get("role") == "button" and row.get("data-tab"):
                self.links.append((row.get("aria-label"), row["data-tab"]))
    page = Elements()
    page.feed((Path(__file__).resolve().parents[1] / "dashboard.html").read_text(encoding="utf-8"))
    assert AUTH_LABEL in page.fields
    assert tuple(page.links) == NAV_LINKS
