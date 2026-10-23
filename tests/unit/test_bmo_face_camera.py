"""The CLI must own the camera — never quietly read someone else's stream.

``rpicam-vid`` fails fast ("Pipeline handler in use by another process") when
the web UI or the voice agent holds the camera, and its stderr used to be sent
to /dev/null.  The pipeline then listened on the default UDP port 5000, which is
exactly where ``web_app.py`` streams, so ``who`` answered from *that* feed and
looked healthy while its own capture was dead.
"""
import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "bmo_face.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("bmo_face_cli_camera", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bmo_face_cli_camera"] = module
    spec.loader.exec_module(module)
    return module


class DeadProc:
    returncode = 1

    def __init__(self):
        self.killed = False

    def wait(self, timeout=None):
        return 1

    def kill(self):
        self.killed = True


class LiveProc:
    returncode = None

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("rpicam-vid", timeout)

    def kill(self):
        pass

    def terminate(self):
        pass


def test_camera_reports_alive_while_running(cli):
    assert cli._camera_ok(LiveProc(), 0.1) is True


def test_camera_reports_dead_when_it_exits(cli):
    assert cli._camera_ok(DeadProc(), 0.1) is False


def test_default_port_is_not_the_web_app_port(cli):
    """Port 5000 carries web_app.py's stream; the diagnostic needs its own."""
    args = cli.build_parser().parse_args(["who"])
    assert args.port != 5000
    assert args.port == cli.DEFAULT_UDP_PORT


def test_camera_busy_surfaces_as_json(cli, monkeypatch):
    def holder(*_a, **_kw):
        raise cli.CameraBusy("failed to acquire camera")

    monkeypatch.setattr(cli, "_collect", holder)
    monkeypatch.setattr(cli, "_agent_service_active", lambda: False)
    out = cli.cmd_who(cli.build_parser().parse_args(["who"]), module=None, Pipeline=None)
    assert out["ok"] is False
    assert out["reason"] == "camera_busy"
    assert "failed to acquire camera" in out["error"]


def test_enroll_also_reports_camera_busy(cli, monkeypatch):
    def holder(*_a, **_kw):
        raise cli.CameraBusy("Pipeline handler in use by another process")

    monkeypatch.setattr(cli, "_collect", holder)
    monkeypatch.setattr(cli, "_agent_service_active", lambda: False)
    args = cli.build_parser().parse_args(["enroll", "--name", "Robert"])
    out = cli.cmd_enroll(args, module=None, Pipeline=None)
    assert out["ok"] is False
    assert out["reason"] == "camera_busy"


def test_camera_busy_is_a_runtime_error(cli):
    assert issubclass(cli.CameraBusy, RuntimeError)
