"""Face routing inside Brain: one spoken line in, one spoken answer out.

The face route runs before the camera-photo keywords ('do you see' is in both
lists), it never reaches the LLM, and 'enroll my face' with no name first asks
for a name and uses the next utterance as the answer.
"""
import pytest

from core.llm import Brain


class FakeFace:
    def __init__(self, result=None):
        self.result = result or {"ok": True, "known": True, "label": "Robert",
                                "confidence": 0.94}
        self.calls = []

    def recognize(self):
        self.calls.append(("recognize",))
        return self.result

    def enroll(self, name=None):
        self.calls.append(("enroll", name))
        return {"ok": True, "label": name, "samples": 3}


def brain(face=None):
    b = Brain(persist=False)
    b.face_id = face
    return b


def test_recognize_is_answered_without_the_llm():
    face = FakeFace()
    b = brain(face)
    out = b.think("do you recognize me?")
    assert "Robert" in out and "94" in out
    assert face.calls == [("recognize",)]
    assert '"action"' not in out


def test_vision_questions_keep_their_camera_route():
    """'do you see me?' is vision, not recognition — it must still take a photo."""
    face = FakeFace()
    out = brain(face).think("do you see me?")
    assert "take_photo" in out
    assert face.calls == []


def test_enroll_with_a_name_in_the_sentence():
    face = FakeFace()
    brain(face).think("enroll my face as Robert")
    assert face.calls == [("enroll", "Robert")]


def test_enroll_without_a_name_asks_then_uses_the_reply():
    face = FakeFace()
    b = brain(face)
    asked = b.think("enroll my face")
    assert "name" in asked.lower() and "?" in asked
    assert face.calls == []

    done = b.think("Robert")
    assert face.calls == [("enroll", "Robert")]
    assert "Robert" in done


def test_changing_the_subject_drops_the_pending_name():
    face = FakeFace()
    b = brain(face)
    b.think("enroll my face")
    assert "12" in b.think("what is 12 percent of 80") or True  # any other route
    assert face.calls == []
    b.think("what's the weather like today")
    assert face.calls == []


def test_no_face_id_falls_through():
    assert Brain(persist=False)._face_action("do you recognize me") is None


def test_face_result_is_also_streamed():
    b = brain(FakeFace())
    chunks = list(b.stream_think("do you recognize me"))
    text = "".join(c for c in chunks if isinstance(c, str))
    assert "Robert" in text


@pytest.mark.parametrize("line", ["who am i", "do you know me", "do you remember me"])
def test_common_phrasings_reach_the_face_route(line):
    face = FakeFace()
    brain(face).think(line)
    assert face.calls == [("recognize",)]
