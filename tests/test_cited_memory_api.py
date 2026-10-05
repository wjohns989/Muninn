"""Isolated encrypted-store agent API proof; no live sources/models/services."""
import asyncio
import threading
import json
import httpx
import pytest
import server
from tests.test_memory_ledger import fixture, record
from muninn.history.memory_ledger import MemoryLedger
from muninn.history.service import HistoryService


@pytest.mark.asyncio
async def test_review_queue_uses_existing_private_guard_without_decisions_or_attestations(tmp_path,monkeypatch):
    archive,entry,attempt,page=fixture(tmp_path,role="assistant")
    ledger=MemoryLedger(archive);ident=record(ledger,entry,attempt,page)
    before=ledger.verify_all()
    token="synthetic-main-token-aaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN",token)
    monkeypatch.setenv("MUNINN_API_KEY","different-generic-token-bbbbbbbbbbbbb")
    monkeypatch.setenv("MUNINN_NO_AUTH","0")
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY","strict")
    monkeypatch.setattr(server,"is_security_enabled",lambda:True)
    service=HistoryService(None,tmp_path/'unused',home=tmp_path,secure_archive_root=archive.root,
                           archive_passphrase="synthetic portable recovery phrase")
    monkeypatch.setattr(server,"_require_history",lambda:service)
    monkeypatch.setattr(server,"_secure_history_fetch_slots",asyncio.Semaphore(1))
    server._cited_memory_read_times.clear()
    from muninn.history.source_evidence import SourceEvidenceStore
    monkeypatch.setattr(SourceEvidenceStore,"_store_screen_info",lambda *a,**kw:pytest.fail("read wrote attestation"))
    headers={"Authorization":"Bearer "+token};route='/history/secure/memories/review-queue'
    transport=httpx.ASGITransport(app=server.app,client=("127.0.0.1",1234))
    try:
        async with httpx.AsyncClient(transport=transport,base_url="http://localhost") as client:
            assert (await client.post(route,json={})).status_code==401
            assert (await client.post(route,json={},headers={'Authorization':'Bearer different-generic-token-bbbbbbbbbbbbb'})).status_code==401
            response=await client.post(route,json={'limit':1},headers=headers)
            assert response.status_code==200 and response.headers['cache-control']=='no-store'
            assert response.json()['data']['matches'][0]['id']==ident
            assert response.json()['data']['matches'][0]['state']=='provisional'
            invalid=await client.post(route,json={'cursor':{'secret':'PRIVATE_INPUT_CANARY'}},headers=headers)
            assert invalid.status_code==422 and 'PRIVATE_INPUT_CANARY' not in invalid.text
            assert (await client.post(route,json={'limit':True},headers=headers)).status_code==422
            assert (await client.post(route,json={'cursor':'tampered'},headers=headers)).status_code==400
            assert (await client.post(route,json={'query':'PRIVATE_INPUT_CANARY'},headers=headers)).status_code==422
        remote=httpx.ASGITransport(app=server.app,client=('192.168.1.2',1234))
        async with httpx.AsyncClient(transport=remote,base_url='http://localhost') as client:
            assert (await client.post(route,json={},headers=headers)).status_code==404
        assert ledger.verify_all()==before
    finally:await service.stop()


@pytest.mark.asyncio
async def test_agent_lookup_search_and_exact_source_follow_to_full_redacted_transcript(tmp_path,monkeypatch):
    archive,entry,attempt,page=fixture(tmp_path)
    ledger=MemoryLedger(archive)
    ident=record(ledger,entry,attempt,page)
    token="synthetic-main-token-aaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("MUNINN_AUTH_TOKEN",token)
    monkeypatch.setenv("MUNINN_API_KEY","different-generic-token-bbbbbbbbbbbbb")
    monkeypatch.setenv("MUNINN_NO_AUTH","0")
    monkeypatch.setenv("MUNINN_HISTORY_SECURITY","strict")
    monkeypatch.setattr(server,"is_security_enabled",lambda:True)
    service=HistoryService(None,tmp_path/'unused',home=tmp_path,secure_archive_root=archive.root,
                           archive_passphrase="synthetic portable recovery phrase")
    monkeypatch.setattr(server,"_require_history",lambda:service)
    monkeypatch.setattr(server,"_secure_history_fetch_slots",asyncio.Semaphore(1))
    server._cited_memory_read_times.clear(); server._secure_history_page_times.clear()
    transport=httpx.ASGITransport(app=server.app,client=("127.0.0.1",1234))
    headers={"Authorization":"Bearer "+token}
    try:
        async with httpx.AsyncClient(transport=transport,base_url="http://localhost") as client:
            requests=[('search',{'query':'citations'}),('get',{'memory_ref':ident}),('source',{'memory_ref':ident})]
            for action,payload in requests:
                route='/history/secure/memories/'+action
                assert (await client.post(route,json=payload)).status_code==401
                assert (await client.post(route,json=payload,headers={'Authorization':'Bearer different-generic-token-bbbbbbbbbbbbb'})).status_code==401
                response=await client.post(route,json=payload,headers=headers)
                assert response.status_code==200 and response.headers['cache-control']=='no-store'
                if action=='search': assert response.json()['data']['matches'][0]['id']==ident
                if action=='get': assert response.json()['data']['truth_status']=='unverified_assertion'
                if action=='source': source=response.json()['data']
            assert source['context']=='I want source citations kept.'
            grant=source['transcript_capability']
            status=await client.post('/history/secure/transcript/start',json={'capability':grant},headers=headers)
            for _ in range(100):
                if status.json()['data']['state']!='pending': break
                await asyncio.sleep(.01)
                status=await client.post('/history/secure/transcript/poll',json={'capability':grant},headers=headers)
            assert status.json()['data']['state']=='ready'
            cursor=status.json()['data']['cursor']
            text=''
            while cursor:
                response=await client.post('/history/secure/transcript/page',json={'cursor':cursor},headers=headers)
                assert response.status_code==200
                data=response.json()['data']; text+=data['redacted_text']; cursor=data['next_cursor']
            assert 'I want source citations kept.' in text
            assert 'synthetic-project' not in text
            rejected=await client.post('/history/secure/memories/search',json={'query':'SERVICE_API_KEY=synthetic$private'},headers=headers)
            assert rejected.status_code==400 and 'private' not in rejected.text
            malformed=await client.post('/history/secure/memories/get',json={'memory_ref':{'secret':'PRIVATE_INPUT_CANARY'}},headers=headers)
            assert malformed.status_code==422 and 'PRIVATE_INPUT_CANARY' not in malformed.text
            missing=await client.post('/history/secure/memories/get',json={'memory_ref':'b'*64},headers=headers)
            assert missing.status_code==404 and missing.headers['cache-control']=='no-store'
        remote=httpx.ASGITransport(app=server.app,client=('192.168.1.2',1234))
        async with httpx.AsyncClient(transport=remote,base_url='http://localhost') as client:
            for action,payload in requests:
                assert (await client.post('/history/secure/memories/'+action,json=payload,headers=headers)).status_code==404
    finally: await service.stop()


@pytest.mark.asyncio
async def test_client_cancellation_keeps_cpu_slot_until_reader_really_finishes(monkeypatch):
    started,release=threading.Event(),threading.Event()
    slots=asyncio.Semaphore(1)
    monkeypatch.setattr(server,'_secure_history_fetch_slots',slots)
    server._cited_memory_read_times.clear()
    def reader():
        started.set(); release.wait(3)
        return {'safe':True}
    task=asyncio.create_task(server._cited_memory_read(reader))
    try:
        for _ in range(100):
            if started.is_set(): break
            await asyncio.sleep(.01)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert slots.locked()
        with pytest.raises(server.HTTPException) as busy: await server._cited_memory_read(lambda:{})
        assert busy.value.status_code==429
    finally:
        release.set()
        for _ in range(100):
            if not slots.locked(): break
            await asyncio.sleep(.01)
    assert not slots.locked()


@pytest.mark.asyncio
async def test_concurrent_last_rate_allowance_has_only_one_admission(monkeypatch):
    import time
    slots=asyncio.Semaphore(1)
    monkeypatch.setattr(server,'_secure_history_fetch_slots',slots)
    server._cited_memory_read_times.clear()
    server._cited_memory_read_times.extend([time.monotonic()]*59)
    await slots.acquire()
    first=asyncio.create_task(server._cited_memory_read(lambda:{'safe':True}))
    second=asyncio.create_task(server._cited_memory_read(lambda:{'safe':True}))
    await asyncio.sleep(.01)
    slots.release()
    results=await asyncio.gather(first,second,return_exceptions=True)
    assert sum(isinstance(r,server.HTTPException) and r.status_code==429 for r in results)==1
    assert sum(isinstance(r,server.JSONResponse) for r in results)==1
    assert len(server._cited_memory_read_times)==60 and not slots.locked()
    server._cited_memory_read_times.clear()
