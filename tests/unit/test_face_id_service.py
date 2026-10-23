"""The handoff around the face pipeline: NPU freed first, camera handed back.

The Hailo-10H is single-tenant (hailo-ollama holds it for the LLM) and only one
process can own the camera, so ``FaceID`` must unload the LLM, stop the shared
camera, run the helper, then restart the camera — in that order, and the camera
must come back even when the helper explodes.
"""
import json
import subprocess

from core.face_id import FaceID


class FakeCamera:
    def __init__(self, running=True):
        self.running = running
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        self.running = False

    def start(self):
        self.calls.append("start")
        self.running = True
        return True


class FakeRun:
    """Stands in for ``scripts/bmo_face.py``; records argv + env."""

    def __init__(self, result=None, raises=None):
        self.result = result if result is not None else {"ok": True, "known": True,
                                                         "label": "Robert", "confidence": 0.9}
        self.raises = raises
        self.calls = []

    def __call__(self, argv, timeout):
        self.calls.append((list(argv), timeout))
        if self.raises:
            raise self.raises
        return self.result


def make(run=None, camera=None, unload=None, log=None):
    order = []

    def _unload():
        order.append("unload")
        return unload if unload is not None else True

    class OrderedCamera(FakeCamera):
        def stop(self):
            order.append("camera_stop")
            super().stop()

        def start(self):
            order.append("camera_start")
            super().start()

    cam = camera if camera is not None else OrderedCamera()
    face = FaceID(run=run or FakeRun(), unload_llm=_unload, camera=cam,
                  script="/tmp/bmo_face.py", app_dir="/tmp/app")
    return face, order, cam


def test_recognize_frees_npu_then_camera_and_hands_it_back():
    run = FakeRun()
    face, order, cam = make(run=run)
    result = face.recognize()

    assert result["label"] == "Robert"
    assert order == ["unload", "camera_stop", "camera_start"]
    assert cam.running is True
    argv, timeout = run.calls[0]
    assert argv[0] == "who"
    assert "/tmp/app" in argv
    assert timeout == face.recognize_timeout


def test_enroll_passes_the_name():
    run = FakeRun({"ok": True, "label": "Robert", "samples": 3})
    face, order, _cam = make(run=run)
    result = face.enroll("Robert")

    assert result["ok"] is True
    argv, _timeout = run.calls[0]
    assert argv[0] == "enroll" and "--name" in argv
    assert argv[argv.index("--name") + 1] == "Robert"
    assert order == ["unload", "camera_stop", "camera_start"]


def test_camera_restarts_even_when_the_helper_blows_up():
    face, order, cam = make(run=FakeRun(raises=RuntimeError("boom")))
    try:
        face.recognize()
    except RuntimeError:
        pass
    assert order[-1] == "camera_start"
    assert cam.running is True


def test_unload_failure_does_not_abort_the_turn():
    run = FakeRun()
    face, _order, _cam = make(run=run, unload=False)
    assert face.recognize()["label"] == "Robert"
    assert len(run.calls) == 1


def test_camera_that_is_not_running_is_left_alone():
    cam = FakeCamera(running=False)
    face, _order, _cam = make(camera=cam)
    face.recognize()
    assert cam.calls == []


def test_no_camera_object_is_fine():
    face = FaceID(run=FakeRun(), unload_llm=lambda: True, camera=None,
                  script="/tmp/bmo_face.py", app_dir="/tmp/app")
    assert face.recognize()["ok"] is True


def test_enrolled_reads_labels():
    run = FakeRun({"ok": True, "labels": ["Robert", "Mary"], "count": 2})
    face, _order, _cam = make(run=run)
    assert face.enrolled() == ["Robert", "Mary"]


def test_enrolled_is_empty_on_failure():
    face, _order, _cam = make(run=FakeRun({"ok": False, "reason": "exception"}))
    assert face.enrolled() == []


def test_forget_delegates():
    run = FakeRun({"ok": True, "label": "Robert", "removed": ["Robert"]})
    face, _order, _cam = make(run=run)
    assert face.forget("Robert")["removed"] == ["Robert"]
    assert run.calls[0][0][0] == "forget"


# --- parsing the helper's stdout ---------------------------------------------------

class Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_script_output_is_parsed_from_the_last_json_line():
    from core.face_id import _parse
    proc = Proc(stdout='[hailo] noise\n{"ok": true, "known": false}\n')
    assert _parse(proc) == {"ok": True, "known": False}


def test_garbage_output_becomes_a_speakable_failure():
    from core.face_id import _parse
    result = _parse(Proc(stdout="Traceback...", stderr="ImportError: no gi", returncode=1))
    assert result["ok"] is False
    assert "gi" in result["error"]


def test_run_script_reports_a_missing_helper(tmp_path):
    face = FaceID(script=str(tmp_path / "missing.py"), python="python3")
    result = face._run_script(["who"], 1)
    assert result["ok"] is False and result["reason"] == "missing_script"


def test_run_script_env_puts_app_dir_on_pythonpath(tmp_path, monkeypatch):
    helper = tmp_path / "bmo_face.py"
    helper.write_text("import json;print(json.dumps({'ok': True, 'args': __import__('sys').argv[1:]}))\n")
    face = FaceID(script=str(helper), python="python3", app_dir="/opt/face-app")
    result = face._run_script(["list"], 10)
    assert result["ok"] is True
    assert result["args"][0] == "list"
    assert "/opt/face-app" in face._env()["PYTHONPATH"]
