from scripts.smoke_secure_search_live import _probe_succeeded, _safe_reason


def test_explicit_local_probe_requires_actual_ollama_result():
    base = {"state": "succeeded", "complete": True, "match_count": 1}
    assert not _probe_succeeded("local", False, {**base, "analysis_status": "deferred"})
    assert not _probe_succeeded("local", False, {**base, "analysis_status": "ok",
                                                    "analysis_provider": "openrouter"})
    assert _probe_succeeded("local", False, {**base, "analysis_status": "ok",
                                                 "analysis_provider": "ollama"})


def test_remote_probe_and_automatic_analysis_do_not_false_pass():
    base = {"state": "succeeded", "complete": True, "match_count": 1}
    assert not _probe_succeeded("remote", False, {**base, "analysis_status": "deferred"})
    assert _probe_succeeded("remote", False, {**base, "analysis_status": "ok",
                                                  "analysis_provider": "openrouter"})
    assert not _probe_succeeded(None, True, {**base, "analysis_queued": False})
    assert not _probe_succeeded(None, True, {**base, "analysis_queued": True,
                                               "auto_analysis_state": "deferred"})
    assert _probe_succeeded(None, True, {**base, "analysis_queued": True,
                                            "auto_analysis_state": "succeeded"})
    assert not _probe_succeeded("local", False, {**base, "match_count": 0,
                                                   "analysis_status": "ok",
                                                   "analysis_provider": "ollama"})


def test_reason_output_is_bounded_to_nonsecret_code():
    assert _safe_reason(None) is None
    assert _safe_reason("ollama_model_already_resident") == "ollama_model_already_resident"
    assert _safe_reason("token=secret") == "other"
    assert _safe_reason("a" * 100) == "other"
