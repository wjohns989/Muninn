"""Render checked-out review alerts using synthetic, isolated browser responses.

All network traffic is intercepted; no service, credentials, workers or models
are contacted. This proves layout/interaction, not a real worker notification.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def main():
    from playwright.sync_api import expect, sync_playwright
    repo = Path(__file__).resolve().parents[1]
    html = (repo / 'dashboard.html').read_bytes()
    css = (repo / 'dashboard.css').read_bytes()
    artifacts = Path(tempfile.mkdtemp(prefix='muninn-triage-ui-'))
    reports = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True,
                                             args=['--disable-background-networking'])
        try:
            for width in (1280, 390):
                context = browser.new_context(viewport={'width': width, 'height': 900},
                    service_workers='block', accept_downloads=False)
                forbidden = []
                current = {'state': 'awaiting_passphrase', 'input_needed': True, 'worker_count': 1}
                def isolate(route):
                    url = urlsplit(route.request.url)
                    if route.request.method != 'GET' or url.netloc != '127.0.0.1:42069':
                        forbidden.append({'method': route.request.method})
                        route.abort()
                    elif url.path == '/':
                        route.fulfill(status=200, content_type='text/html', body=html)
                    elif url.path == '/dashboard.css':
                        route.fulfill(status=200, content_type='text/css', body=css)
                    elif url.path == '/auth/check':
                        route.fulfill(status=200, json={'status': 'ok'})
                    elif url.path == '/health':
                        route.fulfill(status=200, json={'status': 'ok', 'backend': 'isolated fixture',
                                                       'history_security_mode': 'strict'})
                    elif url.path == '/credentials/triage/status':
                        route.fulfill(status=200, json={'data': current})
                    else:
                        route.fulfill(status=404, json={'detail': 'unavailable in fixture'})
                context.route('**/*', isolate)
                page = context.new_page()
                failures = []
                page.on('pageerror', lambda error: failures.append(str(error)))
                page.goto('http://127.0.0.1:42069', wait_until='domcontentloaded')
                page.get_by_label('Local Muninn authentication token', exact=True).fill('synthetic-fixture-token')
                page.get_by_role('button', name='Authenticate & Enter', exact=True).click()
                alert = page.get_by_role('alert').filter(has=page.get_by_role('heading',
                    name='Local input needed: credential review is waiting', exact=True))
                expect(alert).to_be_visible()
                expect(page.locator('#auth-modal')).to_be_hidden()
                expect(alert).to_contain_text('existing local review window')
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                image = artifacts / f'triage-awaiting-{width}.png'
                page.screenshot(path=str(image))
                page.get_by_role('button', name='View review status', exact=True).click()
                expect(page.get_by_role('button', name='Refresh review status', exact=True)).to_be_visible()
                current.update(state='previous_run_failed', input_needed=False, worker_count=0)
                page.get_by_role('button', name='Refresh review status', exact=True).click()
                expect(page.get_by_role('heading', name='Credential review stopped', exact=True)).to_be_visible()
                expect(page.locator('#triage-attention-message')).to_contain_text('No automatic retry')
                page.get_by_role('button', name='Lock session', exact=True).click()
                expect(page.locator('#triage-attention')).to_be_hidden()
                expect(page.locator('#triage-attention-message')).to_have_text('')
                assert not failures, failures
                assert not forbidden, forbidden
                reports.append({'width': width, 'input_alert_rendered': True,
                    'failure_alert_rendered': True, 'lock_cleared': True, 'overflow': False,
                    'mutations_or_external_requests': 0, 'screenshot': str(image)})
                context.close()
        finally:
            browser.close()
    print(json.dumps({'basis': 'isolated_synthetic_ui_not_live_worker', 'reports': reports}))


if __name__ == '__main__':
    main()
