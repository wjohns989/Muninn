"""Triage operational status is main-token/loopback-only, without vault access."""
from types import SimpleNamespace

import httpx
import pytest

import server
from muninn.history import triage_status


@pytest.mark.asyncio
@pytest.mark.parametrize('peer,token,expected', [
    ('127.0.0.1', 'main', 200), ('::1', 'main', 200),
    ('127.0.0.1', 'generic', 401), ('127.0.0.1', 'credential', 401),
    ('127.0.0.1', 'none', 401), ('192.0.2.10', 'main', 404),
])
async def test_status_auth_and_no_vault_open(tmp_path, monkeypatch, peer, token, expected):
    tokens = {name: name + '-fixture-' + 'a' * 40 for name in ('main', 'generic', 'credential')}
    monkeypatch.setenv('MUNINN_AUTH_TOKEN', tokens['main'])
    monkeypatch.setenv('MUNINN_API_KEY', tokens['generic'])
    monkeypatch.setenv('MUNINN_CREDENTIAL_API_TOKEN', tokens['credential'])
    monkeypatch.setattr(server, 'is_security_enabled', lambda: True)
    monkeypatch.setattr(server, 'memory', SimpleNamespace(config=SimpleNamespace(data_dir=tmp_path)))
    def forbidden():
        pytest.fail('Status constructed the live vault')
    monkeypatch.setattr(server, '_credential_store_for_api', forbidden)
    calls = []
    def read(runtime, **kwargs):
        calls.append(runtime)
        return {'state': 'idle', 'input_needed': False, 'worker_count': 0}
    monkeypatch.setattr(triage_status, 'operational_status', read)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app,
        client=(peer, 12345)), base_url='http://localhost') as client:
        response = await client.get('/credentials/triage/status', headers={
            'Authorization': 'Bearer ' + tokens.get(token, 'invalid')})
    assert response.status_code == expected
    assert response.headers['cache-control'] == 'no-store'
    assert bool(calls) is (expected == 200)
    if expected == 200:
        assert response.json()['data']['input_needed'] is False


@pytest.mark.asyncio
async def test_disabled_auth_cannot_read_private_operational_status(monkeypatch):
    monkeypatch.setattr(server, 'is_security_enabled', lambda: False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app,
        client=('127.0.0.1', 12345)), base_url='http://localhost') as client:
        response = await client.get('/credentials/triage/status')
    assert response.status_code == 404
    assert response.headers['cache-control'] == 'no-store'
