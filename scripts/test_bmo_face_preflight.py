"""``who``/``enroll`` must refuse cleanly while bmo-voice owns the NPU.

The Hailo device is single-tenant: with the agent running, HailoRT floods stderr
with ``HAILO_OUT_OF_PHYSICAL_DEVICES`` and BMO's caller gets no usable answer.
A preflight check turns that into one JSON line telling the operator what to do.
"""
import importlib.util
import pathlib
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "bmo_face.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("bmo_face_cli_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bmo_face_cli_preflight"] = module
    spec.loader.exec_module(module)
    return module


def runner(stdout="inactive", raises=None):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if raises:
            raise raises
        return types.SimpleNamespace(stdout=stdout, returncode=0, stderr="")

    run.calls = calls
    return run


def test_active_service_is_detected(cli):
    assert cli._agent_service_active(runner("active\n"))


def test_inactive_service_is_not_busy(cli):
    assert not cli._agent_service_active(runner("inactive\n"))


def test_missing_systemctl_is_not_busy(cli):
    """Not a Pi / no systemd — never block the command on that."""
    assert not cli._agent_service_active(runner(raises=FileNotFoundError("systemctl")))


def who_args(cli):
    return cli.build_parser().parse_args(["who"])


def test_who_refuses_while_the_agent_runs(cli, monkeypatch):
    monkeypatch.setattr(cli, "_agent_service_active", lambda: True)

    def boom(*_a, **_kw):
        raise AssertionError("must not touch the camera/NPU while bmo-voice runs")

    monkeypatch.setattr(cli, "_collect", boom)
    out = cli.cmd_who(who_args(cli), module=None, Pipeline=None)
    assert out["ok"] is False
    assert out["reason"] == "agent_running"
    assert "systemctl stop bmo-voice" in out["hint"]


def test_enroll_refuses_while_the_agent_runs(cli, monkeypatch):
    monkeypatch.setattr(cli, "_agent_service_active", lambda: True)

    def boom(*_a, **_kw):
        raise AssertionError("must not touch the camera/NPU while bmo-voice runs")

    monkeypatch.setattr(cli, "_collect", boom)
    args = cli.build_parser().parse_args(["enroll", "--name", "Robert"])
    out = cli.cmd_enroll(args, module=None, Pipeline=None)
    assert out["ok"] is False
    assert out["reason"] == "agent_running"


def test_list_does_not_need_the_npu(cli, monkeypatch):
    """Reading the DB must keep working while the agent runs."""
    monkeypatch.setattr(cli, "_agent_service_active", lambda: True)
    monkeypatch.setattr(cli, "cmd_list", cli.cmd_list)  # unchanged; documented here
    db = types.SimpleNamespace(get_all_records=lambda: {"a": {"label": "Robert"}})
    monkeypatch.setattr(cli, "_open_db", lambda *_a, **_kw: db)
    args = cli.build_parser().parse_args(["list"])
    out = cli.cmd_list(args, module=None, Pipeline=None)
    assert out["ok"] is True
    assert out["labels"] == ["Robert"]
