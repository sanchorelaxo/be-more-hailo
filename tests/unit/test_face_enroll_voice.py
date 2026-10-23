"""Saying "enroll this face as <name>" has to work while the robot is running.

Two things stood in the way of a spoken enrollment:

  * ``scripts/bmo_face.py`` refuses every run while ``bmo-voice.service`` is
    active — and a spoken enrollment is *started by that very service*, so the
    guard rejected its own caller (``reason: agent_running``).
  * enrollment refused a label that already existed (``already_enrolled``), so a
    stale or badly-posed record could never be replaced from voice.

The guard stays for a human at a shell; a caller that has already freed the NPU
and the camera (``FaceID``) announces itself with ``BMO_FACE_AGENT_HANDOFF=1``.
"""
import importlib.util
import pathlib
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "bmo_face.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("bmo_face_cli_handoff", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bmo_face_cli_handoff"] = module
    spec.loader.exec_module(module)
    return module


def runner(stdout="active\n"):
    def run(cmd, **kwargs):
        return types.SimpleNamespace(stdout=stdout, returncode=0, stderr="")
    return run


# --- the agent guard ---------------------------------------------------------------

def test_handoff_caller_is_not_treated_as_busy(cli, monkeypatch):
    monkeypatch.setenv("BMO_FACE_AGENT_HANDOFF", "1")
    assert cli._agent_service_active(runner("active\n")) is False


def test_human_run_is_still_blocked(cli, monkeypatch):
    monkeypatch.delenv("BMO_FACE_AGENT_HANDOFF", raising=False)
    assert cli._agent_service_active(runner("active\n")) is True


def test_handoff_value_must_be_explicit(cli, monkeypatch):
    monkeypatch.setenv("BMO_FACE_AGENT_HANDOFF", "0")
    assert cli._agent_service_active(runner("active\n")) is True


def test_enroll_runs_under_handoff_though_the_agent_is_active(cli, monkeypatch):
    monkeypatch.setenv("BMO_FACE_AGENT_HANDOFF", "1")
    monkeypatch.setattr(cli, "_agent_service_active", cli._agent_service_active)
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(
        run=lambda *a, **k: types.SimpleNamespace(stdout="active\n", returncode=0, stderr="")))
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: ([], None, FakeFrames()))
    args = cli.build_parser().parse_args(["enroll", "--name", "Johnny Knuckles"])
    out = cli.cmd_enroll(args, module=None, Pipeline=None)
    assert out.get("reason") != "agent_running"


# --- replacing an existing label ---------------------------------------------------

class FakeFrames:
    def get_latest(self):
        return None


class FakeDB:
    """Minimal stand-in for the app's DatabaseHandler."""

    def __init__(self, existing=None):
        self.records = list(existing or [])
        self.deleted = []
        self.created = []

    def get_all_records(self):
        return {r["global_id"]: r for r in self.records}

    def get_record_by_label(self, label):
        for r in self.records:
            if r.get("label") == label:
                return r
        return None

    def delete_record(self, global_id):
        self.deleted.append(global_id)
        self.records = [r for r in self.records if r["global_id"] != global_id]

    def create_record(self, embedding, sample, timestamp, label):
        record = {"global_id": "new-id", "label": label,
                  "samples_json": [{"id": "s1", "sample_path": sample,
                                    "embedding": list(embedding)}]}
        self.created.append(label)
        self.records.append(record)
        return record

    def insert_new_sample(self, record, embedding, sample, timestamp):
        record["samples_json"].append({"id": "s2", "sample_path": sample,
                                       "embedding": list(embedding)})


def enroll_args(cli, name="Johnny Knuckles", replace=False):
    argv = ["enroll", "--name", name]
    if replace:
        argv.append("--replace")
    return cli.build_parser().parse_args(argv)


def face_event(cli, ts):
    return {"track_id": 3, "label": "Unknown", "confidence": 1.0, "x": 0.3, "y": 0.2,
            "w": 0.3, "h": 0.3, "detection_confidence": 0.9, "ts": ts,
            "embedding": [0.5] * 8}


def wire(cli, monkeypatch, db, events):
    monkeypatch.setattr(cli, "_agent_service_active", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: (events, None, FakeFrames()))
    monkeypatch.setattr(cli, "_open_db", lambda *a, **k: db)
    monkeypatch.setattr(cli, "_save_sample", lambda frame, app_dir, label: "/tmp/x.jpg")


def test_replace_overwrites_an_existing_label(cli, monkeypatch):
    db = FakeDB(existing=[{"global_id": "old-id", "label": "Johnny Knuckles",
                           "samples_json": [{"id": "old", "embedding": [0.1] * 8}]}])
    wire(cli, monkeypatch, db, [face_event(cli, 1.0), face_event(cli, 1.5)])
    out = cli.cmd_enroll(enroll_args(cli, replace=True), module=None, Pipeline=None)
    assert out["ok"] is True
    assert db.deleted == ["old-id"]
    assert db.created == ["Johnny Knuckles"]
    assert out["replaced"] is True


def test_without_replace_an_existing_label_is_refused(cli, monkeypatch):
    db = FakeDB(existing=[{"global_id": "old-id", "label": "Johnny Knuckles"}])
    wire(cli, monkeypatch, db, [face_event(cli, 1.0), face_event(cli, 1.5)])
    out = cli.cmd_enroll(enroll_args(cli), module=None, Pipeline=None)
    assert out["ok"] is False and out["reason"] == "already_enrolled"
    assert db.deleted == []


# --- what FaceID sends -------------------------------------------------------------

def test_face_id_enroll_asks_for_a_replacement_and_announces_the_handoff(monkeypatch):
    from core.face_id import FaceID

    seen = {}

    def run(argv, timeout):
        seen["argv"] = list(argv)
        seen["env"] = None
        return {"ok": True, "label": "Johnny Knuckles", "samples": 4}

    face = FaceID(run=run, unload_llm=lambda: True, camera=None,
                  script="/tmp/bmo_face.py", app_dir="/tmp/app")
    face.enroll("Johnny Knuckles")
    assert seen["argv"][0] == "enroll"
    assert "--replace" in seen["argv"]
    assert face._env()["BMO_FACE_AGENT_HANDOFF"] == "1"


def test_recognition_also_announces_the_handoff():
    from core.face_id import FaceID

    face = FaceID(run=lambda argv, timeout: {"ok": True, "known": False},
                  unload_llm=lambda: True, camera=None,
                  script="/tmp/bmo_face.py", app_dir="/tmp/app")
    assert face._env()["BMO_FACE_AGENT_HANDOFF"] == "1"


# --- the camera loan has to be visible to the agent ---------------------------------

def _face(monkeypatch, run):
    from core.face_id import FaceID
    return FaceID(run=run, unload_llm=lambda: True, camera=None,
                  script="/tmp/bmo_face.py", app_dir="/tmp/app")


def test_busy_is_true_only_while_the_helper_runs():
    seen = {}
    face = None

    def run(argv, timeout):
        seen["busy"] = face.busy
        assert face._stop_camera() is False  # no camera handed over
        return {"ok": True, "known": False}

    face = _face(None, run)
    assert face.busy is False
    face.recognize()
    assert seen["busy"] is True, "the presence loop must be able to see the loan"
    assert face.busy is False


def test_busy_clears_when_the_helper_raises():
    def run(argv, timeout):
        raise RuntimeError("helper exploded")

    face = _face(None, run)
    try:
        face.recognize()
    except RuntimeError:
        pass
    assert face.busy is False


# --- what BMO says ------------------------------------------------------------------

def test_replace_is_spoken_as_an_update():
    from core.face_id import describe

    said = describe({"ok": True, "label": "Johnny Knuckles", "samples": 4,
                     "replaced": True}, "enroll")
    assert "Johnny Knuckles" in said
    assert "update" in said.lower() or "again" in said.lower()
