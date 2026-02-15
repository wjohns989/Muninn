import muninn_standalone


def test_standalone_parser_defaults():
    args = muninn_standalone.build_arg_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 42069
    assert args.no_browser is False
    assert args.log_level == "info"


def test_standalone_main_skips_browser_when_no_browser_flag(monkeypatch):
    captured = {}

    def _fake_schedule(url, delay):
        captured["browser"] = (url, delay)

    def _fake_run(app, **kwargs):
        captured["uvicorn"] = {"app": app, **kwargs}

    monkeypatch.setattr(muninn_standalone, "_schedule_browser_launch", _fake_schedule)
    monkeypatch.setattr(muninn_standalone.uvicorn, "run", _fake_run)

    rc = muninn_standalone.main(
        ["--host", "127.0.0.1", "--port", "43001", "--no-browser", "--log-level", "debug"]
    )

    assert rc == 0
    assert "browser" not in captured
    assert captured["uvicorn"]["app"] == "server:app"
    assert captured["uvicorn"]["host"] == "127.0.0.1"
    assert captured["uvicorn"]["port"] == 43001
    assert captured["uvicorn"]["log_level"] == "debug"


def test_standalone_main_schedules_browser_by_default(monkeypatch):
    captured = {}

    def _fake_schedule(url, delay):
        captured["browser"] = (url, delay)

    def _fake_run(app, **kwargs):
        captured["uvicorn"] = {"app": app, **kwargs}

    monkeypatch.setattr(muninn_standalone, "_schedule_browser_launch", _fake_schedule)
    monkeypatch.setattr(muninn_standalone.uvicorn, "run", _fake_run)

    rc = muninn_standalone.main(["--host", "127.0.0.1", "--port", "43100"])

    assert rc == 0
    assert captured["browser"][0] == "http://127.0.0.1:43100/"
    assert captured["uvicorn"]["port"] == 43100
