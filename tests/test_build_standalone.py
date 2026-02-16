import os

from scripts import build_standalone


def test_build_standalone_default_output_name_is_huginn():
    args = build_standalone.build_parser().parse_args([])
    assert args.name == "HuginnControlCenter"


def test_build_standalone_constructs_pyinstaller_command(monkeypatch):
    captured = {}

    def _fake_run(cmd):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(build_standalone, "_run", _fake_run)
    rc = build_standalone.main(["--name", "MuninnTestApp", "--onefile", "--windowed", "--clean"])

    assert rc == 0
    cmd = captured["cmd"]
    assert cmd[:3] == [build_standalone.sys.executable, "-m", "PyInstaller"]
    assert "--name" in cmd
    assert "MuninnTestApp" in cmd
    assert "--onefile" in cmd
    assert "--noconsole" in cmd
    assert "--clean" in cmd
    assert "--add-data" in cmd
    add_data_value = cmd[cmd.index("--add-data") + 1]
    separator = ";" if os.name == "nt" else ":"
    assert separator in add_data_value
    assert add_data_value.endswith(f"{separator}.")
