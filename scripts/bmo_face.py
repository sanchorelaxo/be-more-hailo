#!/usr/bin/env python3
"""Headless front end for the onboard Hailo face app (SCRFD → ArcFace → LanceDB).

The Pi's face stack lives in ``~/face-recog-web``: a GStreamer pipeline that
detects faces with SCRFD, tracks them, embeds them with ArcFace and looks the
embedding up in a LanceDB table.  That app is a Flask web viewer; BMO needs the
same brains without a browser, so this script drives the pipeline directly,
watches the camera for a few seconds and prints ONE json line on stdout.

    python3 scripts/bmo_face.py who                 # who is in front of BMO?
    python3 scripts/bmo_face.py enroll --name Robert
    python3 scripts/bmo_face.py list                # who does BMO know?
    python3 scripts/bmo_face.py forget --name Robert

Run it with the hailo-apps venv (``~/hailo-apps/venv_hailo_apps/bin/python3``):
hailo_platform, GStreamer and the pipeline's post-process SOs are only importable
there.  BMO's ``core/face_id.py`` does that, and frees the NPU + camera first.

Exit status is 0 when the JSON says ok, 1 otherwise — so a shell caller can
branch on it, while BMO reads the JSON.
"""
import argparse
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

DEFAULT_APP_DIR = os.path.expanduser("~/face-recog-web")
DEFAULT_HAILO_APPS = os.path.expanduser("~/hailo-apps")
# NOT 5000: that is where ``web_app.py`` streams its own capture, and a pipeline
# listening there happily answers from the web UI's camera instead of our own.
DEFAULT_UDP_PORT = 5199
HAILO_ENV_FILE = "/usr/local/hailo/resources/.env"
UNKNOWN = {"", "unknown", "none", "null"}
MIN_DETECTION_CONFIDENCE = 0.55
# The app's own recognition threshold: at or above this the app already calls it
# the same person, so a second label describing the same face must be folded in.
SAME_FACE_SIM = 0.7
# Boxes are normalised 0..1; anything under 0.5% of the frame is background
# clutter (a face on a poster, a reflection), not the person talking to BMO.
MIN_FACE_AREA = 0.005


# --- wiring -----------------------------------------------------------------------

def _load_hailo_env():
    """GStreamer/Hailo env vars the pip/apt install expects (hailo-tappas-core)."""
    try:
        with open(HAILO_ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


def _bootstrap(app_dir: str, hailo_apps: str):
    """Make the app's pipeline and hailo-apps helpers importable."""
    _load_hailo_env()
    os.environ.setdefault("GST_PLUGIN_FEATURE_RANK", "vaapidecodebin:NONE")
    for path in (
        app_dir,
        hailo_apps,
        os.path.join(hailo_apps, "hailo_apps/python/core/common"),
        os.path.join(hailo_apps, "hailo_apps/python/core/gstreamer"),
        os.path.join(hailo_apps, "hailo_apps/python/pipeline_apps/face_recognition"),
    ):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _load_pipeline_module(app_dir: str):
    """Import the app's ``pipeline.py`` by path (it is not an installed package)."""
    path = os.path.join(app_dir, "pipeline.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"face app pipeline not found at {path}")
    spec = importlib.util.spec_from_file_location("bmo_face_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pipeline_class(module):
    """FaceReconPipeline with a configurable UDP port.

    Upstream hardcodes ``udpsrc port=5000``; the web viewer may be running on the
    same box, so the CLI needs its own port.
    """

    class Pipeline(module.FaceReconPipeline):
        def __init__(self, *args, udp_port: int = 5000, **kwargs):
            self._udp_port = udp_port
            super().__init__(*args, **kwargs)

        def _source(self):
            if self._input == "rpi":
                return (f"udpsrc port={self._udp_port} ! h264parse config-interval=1 ! "
                        "avdec_h264 ! autovideoconvert")
            return super()._source()

        def _appsink(self):
            """Preview sink that does not pin a frame size.

            The aggregator emits the original frame at the detector's square
            geometry (640x640), while the stock sink caps to ``width x height``
            (640x480).  videoconvert cannot rescale, so negotiation fails, the
            aggregator blocks once its bypass queue fills, and the pipeline
            delivers a few frames then stalls — no face events ever arrive.
            """
            return ("queue name=app_q ! videoconvert ! video/x-raw,format=RGB ! "
                    "appsink name=web_appsink emit-signals=true sync=false "
                    "max-buffers=2 drop=true")

    return Pipeline


def _db_path(app_dir: str) -> str:
    return os.path.join(app_dir, "face_db", "persons.db")


# --- camera capture ---------------------------------------------------------------

def _port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class CameraBusy(RuntimeError):
    """The camera is held by someone else (web UI, voice agent, another probe)."""


def _camera_ok(cam, settle: float) -> bool:
    """True when ``cam`` is still capturing after ``settle`` seconds.

    ``rpicam-vid`` gives up immediately when another process owns libcamera
    ("Pipeline handler in use by another process"), so a quick exit is the only
    reliable signal that we never got the camera.
    """
    try:
        cam.wait(timeout=settle)
    except subprocess.TimeoutExpired:
        return True
    return False


def _start_camera_stream(port: int, width: int, height: int, settle: float = 1.5):
    cmd = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
    if cmd is None:
        raise CameraBusy("rpicam-vid not installed")
    # Keep rpicam's chatter out of the JSON on stdout, but capture it so a dead
    # camera can be reported with the reason instead of a silent empty result.
    log = tempfile.TemporaryFile(mode="w+")
    cam = subprocess.Popen(
        [cmd, "--width", str(width), "--height", str(height), "--framerate", "30",
         "--nopreview", "-t", "0", "-o", f"udp://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=log)
    if _camera_ok(cam, settle):
        cam._bmo_log = log
        return cam
    log.seek(0)
    detail = [line for line in log.read().splitlines() if line.strip()]
    log.close()
    message = detail[-1].strip() if detail else f"rpicam-vid exited ({cam.returncode})"
    raise CameraBusy(message)


def _collect(module, Pipeline, app_dir: str, seconds: float, port: int,
             width: int, height: int):
    """Watch the camera for ``seconds``; return (events, pipeline, camera_proc)."""
    frames = module.FrameQueue()
    faces = module.FaceEventQueue()
    cam = _start_camera_stream(port, width, height)
    time.sleep(1.0)  # let the H.264 stream establish before the pipeline tunes in
    pipeline = Pipeline(frames, faces, input_source="rpi", width=width, height=height,
                        db_path=_db_path(app_dir), udp_port=port)
    if not pipeline.start():
        _stop(pipeline, cam)
        raise RuntimeError("pipeline failed to start (see hailo logs)")
    events = []
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            try:
                events.append(faces.get(block=False))
            except queue.Empty:
                time.sleep(0.05)
    finally:
        _stop(pipeline, cam)
    return events, pipeline, frames


def _stop(pipeline, cam):
    try:
        pipeline.stop()
    except Exception:
        pass
    try:
        cam.terminate()
        cam.wait(timeout=5)
    except Exception:
        try:
            cam.kill()
        except Exception:
            pass
    log = getattr(cam, "_bmo_log", None)
    if log is not None:
        try:
            log.close()
        except Exception:
            pass


# --- choosing a face ---------------------------------------------------------------

def _area(event: dict) -> float:
    return float(event.get("w") or 0) * float(event.get("h") or 0)


def _usable(event: dict) -> bool:
    return (_area(event) >= MIN_FACE_AREA
            and float(event.get("detection_confidence") or 0) >= MIN_DETECTION_CONFIDENCE)


def _best_face(events):
    usable = [e for e in events if _usable(e)]
    return max(usable, key=_area) if usable else None


def _dominant_face(events):
    """The biggest face, plus the best known label seen for its track.

    The app stamps unrecognised faces with confidence 1.0, so the label cannot
    come from the top-confidence event — that always lands on ``Unknown`` even
    when the same track was just matched to an enrolled face.
    """
    best = _best_face(events)
    if best is None:
        return None
    track = [e for e in events if e.get("track_id") == best["track_id"]]
    faces = {}
    for event in track:
        label = (event.get("label") or "Unknown").strip()
        conf = float(event.get("confidence") or 0)
        if label.lower() in UNKNOWN:
            continue
        if conf >= faces.get(label, 0):
            faces[label] = conf
    if faces:
        label = max(faces, key=lambda name: faces[name])
        confidence = faces[label]
    else:
        label = "Unknown"
        confidence = max((float(e.get("confidence") or 0) for e in track), default=0.0)
    return {
        "track_id": best["track_id"],
        "label": label,
        "confidence": confidence,
        "detection_confidence": float(best.get("detection_confidence") or 0),
        "area": round(_area(best), 1),
        "faces": [{"label": k, "confidence": round(v, 4)} for k, v in faces.items()],
        "track": track,
    }


# --- preflight --------------------------------------------------------------------

AGENT_SERVICE = "bmo-voice"

# A spoken "enroll this face as X" is started BY bmo-voice.service, so the
# service check would reject its own caller.  ``core.face_id.FaceID`` frees the
# NPU and the shared camera before it shells out and announces that handoff in
# the environment; a human at a shell (no flag) still gets the refusal.
HANDOFF_ENV = "BMO_FACE_AGENT_HANDOFF"

_AGENT_BUSY = {
    "ok": False,
    "reason": "agent_running",
    "hint": (f"stop {AGENT_SERVICE} first: sudo systemctl stop {AGENT_SERVICE} "
             "(it holds the NPU and the camera)"),
}


def _agent_service_active(runner=None) -> bool:
    """Is the voice agent running?  It owns the single-tenancy Hailo device."""
    if os.environ.get(HANDOFF_ENV) == "1":
        return False
    runner = runner or subprocess.run
    try:
        result = runner(["systemctl", "is-active", AGENT_SERVICE],
                        capture_output=True, text=True, timeout=5)
    except Exception:  # no systemd, no systemctl, timeout — assume it is free
        return False
    return (getattr(result, "stdout", "") or "").strip() == "active"


_CAMERA_BUSY_HINT = ("the camera is held by another process — "
                     "stop bmo-voice and bmo-web first "
                     "(sudo systemctl stop bmo-voice bmo-web)")


def _camera_busy_result(exc: Exception) -> dict:
    return {"ok": False, "reason": "camera_busy", "error": str(exc),
            "hint": _CAMERA_BUSY_HINT}


# --- commands ---------------------------------------------------------------------

def cmd_who(args, module, Pipeline) -> dict:
    if _agent_service_active():
        return dict(_AGENT_BUSY)
    started = time.time()
    try:
        events, _pipeline, frames = _collect(module, Pipeline, args.app_dir, args.timeout,
                                             args.port, args.width, args.height)
    except CameraBusy as exc:
        return _camera_busy_result(exc)
    face = _dominant_face(events)
    if face is None:
        return {"ok": False, "reason": "no_face", "seconds": round(time.time() - started, 1),
                "events": len(events)}
    label = (face["label"] or "Unknown").strip()
    known = label.lower() not in UNKNOWN
    frame = frames.get_latest() if hasattr(frames, "get_latest") else None
    return {
        "ok": True,
        "known": known,
        "label": label if known else "Unknown",
        "confidence": round(face["confidence"], 4),
        "detection_confidence": round(face["detection_confidence"], 4),
        "faces": face["faces"],
        "seconds": round(time.time() - started, 1),
        "events": len(events),
        "sample": _save_sample(frame, args.app_dir, label if known else "unknown"),
    }


def _save_sample(frame, app_dir: str, label: str) -> str | None:
    """Keep the frame we judged — handy for confirming a bad enrollment later."""
    if not frame or not frame[0]:
        return None
    try:
        samples = os.path.join(app_dir, "face_db", "recognitions")
        os.makedirs(samples, exist_ok=True)
        safe = "".join(c for c in label if c.isalnum() or c in "-_") or "unknown"
        path = os.path.join(samples, f"{int(time.time())}_{safe}.jpg")
        with open(path, "wb") as f:
            f.write(frame[0])
        return path
    except Exception:
        return None


def cmd_enroll(args, module, Pipeline) -> dict:
    if not args.name:
        return {"ok": False, "reason": "needs_name"}
    if _agent_service_active():
        return dict(_AGENT_BUSY)
    started = time.time()
    try:
        events, _pipeline, frames = _collect(module, Pipeline, args.app_dir, args.timeout,
                                             args.port, args.width, args.height)
    except CameraBusy as exc:
        return _camera_busy_result(exc)
    face = _dominant_face(events)
    if face is None:
        return {"ok": False, "reason": "no_face", "seconds": round(time.time() - started, 1)}

    schar = [e for e in face["track"] if e.get("embedding") and _usable(e)]
    if not schar:
        return {"ok": False, "reason": "no_embedding",
                "seconds": round(time.time() - started, 1)}
    schar.sort(key=_area, reverse=True)
    picked, seen_ts = [], []
    for event in schar:  # spaced samples of the same face, not 12 copies of one frame
        if any(abs(event["ts"] - ts) < 0.25 for ts in seen_ts):
            continue
        picked.append(event)
        seen_ts.append(event["ts"])
        if len(picked) >= args.samples:
            break

    import numpy as np
    best = picked[0]
    frame = frames.get_latest() if hasattr(frames, "get_latest") else None
    sample = _save_sample(frame, args.app_dir, args.name) or ""
    db = _open_db(module, Pipeline, args.app_dir)
    existing = _find_records(db, args.name, exact=True)
    if existing and not args.replace:
        return {"ok": False, "reason": "already_enrolled", "label": args.name,
                "seconds": round(time.time() - started, 1)}
    twins = _same_face_elsewhere(db, args.name, best["embedding"])
    for old in existing:  # --replace: a stale/ badly-posed record is dropped whole
        db.delete_record(old["global_id"])
    merged = []
    if args.replace:  # ...and so is the same face filed under an older name
        for twin, _sim in twins:
            db.delete_record(twin["global_id"])
            merged.append(twin.get("label"))
    record = db.create_record(
        embedding=np.asarray(best["embedding"], dtype="float32"),
        sample=sample,
        timestamp=int(time.time()),
        label=args.name,
    )
    for extra in picked[1:]:  # the DB keeps an average of every sample it is given
        db.insert_new_sample(
            record=_record_for_update(db, record),
            embedding=np.asarray(extra["embedding"], dtype="float32"),
            sample=sample,
            timestamp=int(time.time()),
        )
    stored = db.get_record_by_label(args.name) or {}
    return {
        "ok": True,
        "label": args.name,
        "samples": len(picked),
        "global_id": record.get("global_id"),
        "replaced": bool(existing),
        "merged": merged,
        "looks_like": [twin.get("label") for twin, _sim in twins],
        "stored_samples": len(stored.get("samples_json") or []) if isinstance(
            stored.get("samples_json"), list) else None,
        "seconds": round(time.time() - started, 1),
        "sample": sample or None,
    }


def _record_for_update(db, record: dict) -> dict:
    """insert_new_sample wants ``samples_json`` already parsed."""
    fresh = db.get_record_by_label(record.get("label")) or {}
    sample_list = fresh.get("samples_json")
    if isinstance(sample_list, str):
        sample_list = json.loads(sample_list)
    return {**fresh, "samples_json": sample_list or []}


def _record_embeddings(record: dict) -> list:
    """Every stored sample vector of one person, ``samples_json`` parsed."""
    import numpy as np

    samples = (record or {}).get("samples_json") or []
    if isinstance(samples, str):
        try:
            samples = json.loads(samples)
        except Exception:
            return []
    out = []
    for sample in samples if isinstance(samples, list) else []:
        emb = sample.get("embedding") if isinstance(sample, dict) else None
        if not emb:
            continue
        vec = np.asarray(emb, dtype="float32").ravel()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6 and vec.size:
            out.append(vec / norm)
    return out


def _same_face_elsewhere(db, name: str, embedding, threshold: float = SAME_FACE_SIM) -> list:
    """Other labels whose stored samples are this same face at recognition level.

    Learning one face under two names is how BMO ends up calling you the wrong
    one: both records match above 0.7 and the higher score wins.
    """
    import numpy as np

    probe = np.asarray(embedding, dtype="float32").ravel()
    norm = float(np.linalg.norm(probe))
    if not norm:
        return []
    probe = probe / norm
    twins = []
    for record in _all_records(db):
        label = ((record or {}).get("label") or "").strip()
        if not label or label.lower() == name.lower() or label.lower() in UNKNOWN:
            continue
        stored = _record_embeddings(record)
        if not stored:
            continue
        sim = float(np.mean([float(probe @ vec) for vec in stored]))
        if sim >= threshold:
            twins.append((record, sim))
    twins.sort(key=lambda pair: pair[1], reverse=True)
    return twins


def _open_db(module, Pipeline, app_dir: str):
    """Path resolution only — no GStreamer, no NPU, so this is safe to call alone."""
    pipeline = Pipeline(module.FrameQueue(), module.FaceEventQueue(),
                        input_source="rpi", db_path=_db_path(app_dir))
    return pipeline.get_db()


def _all_records(db) -> list:
    records = db.get_all_records() or {}
    return list(records.values()) if isinstance(records, dict) else list(records)


def _find_records(db, name: str, exact: bool = True) -> list:
    values = _all_records(db)
    out = []
    for record in values:
        label = (record or {}).get("label") or ""
        if (label.lower() == name.lower()) if exact else (name.lower() in label.lower()):
            out.append(record)
    return out


def cmd_list(args, module, Pipeline) -> dict:
    db = _open_db(module, Pipeline, args.app_dir)
    records = db.get_all_records() or {}
    values = list(records.values()) if isinstance(records, dict) else list(records)
    labels = sorted({(r.get("label") or "").strip() for r in values
                     if (r.get("label") or "").strip().lower() not in UNKNOWN})
    return {"ok": True, "labels": labels, "count": len(labels)}


def cmd_forget(args, module, Pipeline) -> dict:
    db = _open_db(module, Pipeline, args.app_dir)
    matches = _find_records(db, args.name, exact=False)
    if not matches:
        return {"ok": False, "reason": "not_enrolled", "label": args.name}
    removed = []
    for record in matches:
        db.delete_record(record["global_id"])
        removed.append(record.get("label"))
    return {"ok": True, "label": args.name, "removed": removed}


# --- main -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--app-dir", default=os.environ.get("HAILO_APP_DIR", DEFAULT_APP_DIR),
                        help="face app directory (pipeline.py + face_db/)")
    common.add_argument("--hailo-apps", default=os.environ.get("BMO_HAILO_APPS_DIR", DEFAULT_HAILO_APPS))

    parser = argparse.ArgumentParser(prog="bmo_face.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    who = sub.add_parser("who", parents=[common], help="who is in front of the camera?")
    who.add_argument("--timeout", type=float, default=8.0)
    who.add_argument("--port", type=int, default=int(os.environ.get("BMO_FACE_UDP_PORT", DEFAULT_UDP_PORT)))
    who.add_argument("--width", type=int, default=640)
    who.add_argument("--height", type=int, default=480)

    enroll = sub.add_parser("enroll", parents=[common], help="learn a new face")
    enroll.add_argument("--name", required=False, help="name to remember this face by")
    enroll.add_argument("--samples", type=int, default=3, help="distinct frames to average")
    enroll.add_argument("--replace", action="store_true",
                        help="overwrite an existing label instead of refusing")
    enroll.add_argument("--timeout", type=float, default=12.0)
    enroll.add_argument("--port", type=int, default=int(os.environ.get("BMO_FACE_UDP_PORT", DEFAULT_UDP_PORT)))
    enroll.add_argument("--width", type=int, default=640)
    enroll.add_argument("--height", type=int, default=480)

    listing = sub.add_parser("list", parents=[common], help="who does BMO know?")
    forget = sub.add_parser("forget", parents=[common], help="drop a face from BMO's memory")
    forget.add_argument("--name", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    emit = {"ok": False, "reason": "unknown_command"}
    try:
        _bootstrap(args.app_dir, args.hailo_apps)
        module = _load_pipeline_module(args.app_dir)
        Pipeline = _pipeline_class(module)
        handler = {"who": cmd_who, "enroll": cmd_enroll,
                   "list": cmd_list, "forget": cmd_forget}[args.command]
        emit = handler(args, module, Pipeline)
    except Exception as exc:  # one JSON line always — BMO parses stdout only
        import traceback
        traceback.print_exc()
        emit = {"ok": False, "reason": "exception", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(emit))
    sys.stdout.flush()
    return 0 if emit.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
