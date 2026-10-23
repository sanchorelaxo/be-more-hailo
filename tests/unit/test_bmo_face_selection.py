"""Picking the face BMO should answer about, and the frames it enrolls from.

The pipeline emits one event per tracked face per analysed frame, so a turn
produces dozens of events for one person plus a few for a poster on the wall.
``_best_face``/``_dominant_face`` pick the biggest real face and the best label
seen for its track.  Pure functions — no GStreamer, no import of the face app.
"""
import importlib.util
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "bmo_face.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("bmo_face_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bmo_face_cli"] = module
    spec.loader.exec_module(module)
    return module


def event(track_id=1, label="Robert", confidence=0.9, w=0.2, h=0.3, det=0.9,
          ts=0.0, embedding=None):
    ev = {"track_id": track_id, "label": label, "confidence": confidence,
          "w": w, "h": h, "detection_confidence": det, "x": 0.4, "y": 0.3, "ts": ts,
          "frame_w": 640, "frame_h": 480}
    if embedding is not None:
        ev["embedding"] = embedding
    return ev


def test_biggest_face_wins(cli):
    events = [event(track_id=1, w=0.05, h=0.05), event(track_id=2, w=0.4, h=0.4)]
    assert cli._best_face(events)["track_id"] == 2


def test_tiny_or_unsure_detections_are_ignored(cli):
    assert cli._best_face([event(w=0.01, h=0.01)]) is None
    assert cli._best_face([event(det=0.2)]) is None
    assert cli._best_face([]) is None


def test_dominant_face_uses_the_best_label_of_its_track(cli):
    events = [event(track_id=7, label="Unknown", confidence=0.0),
              event(track_id=7, label="Robert", confidence=0.83),
              event(track_id=3, label="Poster", confidence=0.99, w=0.02, h=0.02)]
    face = cli._dominant_face(events)
    assert face["label"] == "Robert"
    assert face["confidence"] == pytest.approx(0.83)
    assert face["faces"] == [{"label": "Robert", "confidence": 0.83}]


def test_unknown_events_do_not_mask_a_known_label(cli):
    """The app stamps unknown faces with confidence 1.0.

    Picking the label off the highest-confidence event therefore always chose
    "Unknown", even when the same track was matched to an enrolled face.
    """
    events = [event(track_id=7, label="Unknown", confidence=1.0),
              event(track_id=7, label="Unknown", confidence=1.0),
              event(track_id=7, label="Robert", confidence=0.854)]
    face = cli._dominant_face(events)
    assert face["label"] == "Robert"
    assert face["confidence"] == pytest.approx(0.854)


def test_most_confident_known_label_wins(cli):
    events = [event(track_id=7, label="Unknown", confidence=1.0),
              event(track_id=7, label="Robert", confidence=0.70),
              event(track_id=7, label="Alice", confidence=0.91)]
    face = cli._dominant_face(events)
    assert face["label"] == "Alice"
    assert face["confidence"] == pytest.approx(0.91)


def test_unknown_only_track_stays_unknown(cli):
    events = [event(track_id=7, label="Unknown", confidence=1.0),
              event(track_id=7, label="Unknown", confidence=1.0)]
    face = cli._dominant_face(events)
    assert face["label"] == "Unknown"
    assert face["faces"] == []


def test_unknown_labels_are_not_offered_as_names(cli):
    events = [event(label="Unknown", confidence=0.0), event(label="unknown", confidence=0.0)]
    face = cli._dominant_face(events)
    assert face["label"] == "Unknown"
    assert face["faces"] == []


def test_no_face_at_all(cli):
    assert cli._dominant_face([]) is None


def test_enroll_prefers_spaced_frames(cli):
    # Twelve events in the same second are one look, not twelve samples.
    events = [event(ts=100.0, embedding=[0.1] * 4) for _ in range(12)]
    events += [event(ts=100.5, embedding=[0.2] * 4), event(ts=101.1, embedding=[0.3] * 4)]
    face = cli._dominant_face(events)
    picked = []
    for ev in face["track"]:
        if any(abs(ev["ts"] - t) < 0.25 for t in picked):
            continue
        picked.append(ev["ts"])
    assert picked == [100.0, 100.5, 101.1]
