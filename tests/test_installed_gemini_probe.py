"""Probe helpers only: no live settings, archives, commands or providers."""
from scripts.probe_installed_gemini_hooks import accepted, select_source, verify_capture


def test_acceptance_requires_a_new_capture_receipt_not_process_exit_zero():
    key = ('gemini_cli', 'AfterAgent')
    old = {key: {'accepted_invocations': 2, 'last_outcome': 'capture_intent'}}
    assert not accepted(old, old, 'AfterAgent')
    assert not accepted(old, {key: {'accepted_invocations': 3, 'last_outcome': 'no_transcript'}}, 'AfterAgent')
    assert accepted(old, {key: {'accepted_invocations': 3, 'last_outcome': 'capture_intent'}}, 'AfterAgent')


def test_source_selection_is_existing_archived_gemini_only(tmp_path):
    own = tmp_path / '.gemini/chats/small.json'
    own.parent.mkdir(parents=True)
    own.write_text('{}')
    other = tmp_path / 'outside.json'
    other.write_text('{}')
    manifest = {'files': {str(p): [{'provider': 'gemini_cli', 'kind': 'transcript'}] for p in (own, other)}}
    assert select_source(manifest, tmp_path) == own
    assert not select_source(manifest, tmp_path).is_relative_to(tmp_path / 'outside')


def test_capture_proof_authenticates_real_encrypted_bytes_and_detects_source_change(tmp_path):
    from muninn.history.secure_archive import SecureHistoryArchive
    archive = SecureHistoryArchive.create(tmp_path / 'archive', 'synthetic portable test phrase')
    source = tmp_path / 'session.json'
    source.write_text('{"message": "Synthetic ordinary memory"}')
    archive.archive_file(source, 'gemini_cli')
    assert verify_capture(archive, source)
    source.write_text('{"message": "Changed synthetic ordinary memory"}')
    assert not verify_capture(archive, source)
