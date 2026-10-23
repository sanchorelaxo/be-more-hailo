"""Face intents are parsed in Python, never left to the 1.7B model.

``match_face_intent`` decides between "who is this?" and "learn my face"; the
name comes from "as Robert" / "my name is Robert".  ``describe`` is the spoken
half — every branch must produce something BMO can actually say.
"""
from core.face_id import match_face_intent, describe, parse_name_reply


def test_recognize_needs_the_word_recognize():
    """The trigger is the word itself, wherever it sits in the sentence.

    On the robot "Recognize me." missed the old pattern list, fell through to the
    1.7B model and came back with a name from the chat history — the one thing a
    face query must never do is invent an answer.
    """
    for text in ("do you recognize me", "Do you recognise me?", "recognize me.",
                 "Recognise me", "recognize me please", "recognize this face",
                 "can you recognise the person in front of you?",
                 "do you recognise whoever is standing here",
                 "guess what, recognize me now"):
        assert match_face_intent(text) == ("recognize", None), text


def test_without_the_word_recognize_it_is_not_recognition():
    """A deliberate rule: no "recognize", no scan.

    "who am i" / "do you know me" used to route here. They must not any more —
    the robot only looks at the camera when it is asked to recognize.
    """
    for text in ("who am i", "do you know me", "do you know who i am",
                 "who is that", "who is it", "what's my name", "identify me",
                 "do you remember me", "do you remember my face",
                 "can you tell me who i am"):
        assert match_face_intent(text) is None, text


def test_enroll_phrasings():
    for text in ("enroll my face", "register this face", "learn my face",
                 "remember my face", "save my face", "enroll me"):
        intent = match_face_intent(text)
        assert intent is not None and intent[0] == "enroll", text
        assert intent[1] is None, text


def test_enroll_with_name():
    assert match_face_intent("enroll my face as Robert") == ("enroll", "Robert")
    assert match_face_intent("register this face as Mary Jane") == ("enroll", "Mary Jane")
    assert match_face_intent("my name is Robert, enroll my face") == ("enroll", "Robert")
    assert match_face_intent("enroll this face as Johnny Knuckles") == (
        "enroll", "Johnny Knuckles")


def test_enroll_verbs_people_actually_use():
    """"Hold this face as anonymous" was said to the robot and went to the LLM."""
    assert match_face_intent("hold this face as anonymous") == ("enroll", "anonymous")
    assert match_face_intent("learn this face as Rob") == ("enroll", "Rob")
    assert match_face_intent("add my face as Rob") == ("enroll", "Rob")
    assert match_face_intent("memorize this face as Rob") == ("enroll", "Rob")


def test_name_tail_is_not_a_name():
    assert match_face_intent("enroll my face as soon as possible") == ("enroll", None)


def test_unrelated_text_is_ignored():
    for text in ("what's the weather", "take a photo", "who won the game",
                 "play music", "", "do you see the cat"):
        assert match_face_intent(text) is None, text


def test_vision_questions_are_not_recognition():
    """Vision validation and recognition validation are different jobs.

    "do you see me?" asks the VLM what is in view and must keep its existing
    camera route; only the stipulated recognition phrasing goes to the face app.
    """
    for text in ("do you see me?", "do you see me", "do you see a face",
                 "who do you see", "can you see my face", "what do you see",
                 "what is this", "look at this", "check the camera",
                 "take a photo of me", "show me what you see"):
        assert match_face_intent(text) is None, text


def test_describe_known_face():
    line = describe({"ok": True, "known": True, "label": "Robert", "confidence": 0.91})
    assert "Robert" in line and "91" in line


def test_describe_unknown_face():
    line = describe({"ok": True, "known": False, "label": "Unknown",
                     "faces": [{"label": "Unknown", "confidence": 0}], "confidence": 0})
    assert "don't know" in line.lower() or "do not know" in line.lower()
    assert "enroll" in line.lower()


def test_describe_no_face():
    line = describe({"ok": False, "reason": "no_face"})
    assert "face" in line.lower() and "camera" in line.lower()


def test_describe_enroll():
    line = describe({"ok": True, "label": "Robert", "samples": 3}, command="enroll")
    assert "Robert" in line and "3" in line


def test_describe_already_enrolled():
    line = describe({"ok": False, "reason": "already_enrolled", "label": "Robert"},
                    command="enroll")
    assert "already" in line.lower() and "Robert" in line


def test_describe_asks_for_a_name():
    line = describe({"ok": False, "reason": "needs_name"}, command="enroll")
    assert "?" in line and "name" in line.lower()


def test_parse_name_reply():
    assert parse_name_reply("Robert") == "Robert"
    assert parse_name_reply("Robert Jodouin") == "Robert Jodouin"
    assert parse_name_reply("my name is Robert") == "Robert"
    assert parse_name_reply("call me Rob") == "Rob"
    assert parse_name_reply("it's Robert") == "Robert"
    assert parse_name_reply("Mary Jane.") == "Mary Jane"


def test_parse_name_reply_rejects_non_names():
    for text in ("yes", "no thanks", "I don't want to", "what's the weather like today",
                 "", "take a photo of the cat"):
        assert parse_name_reply(text) is None, text


def test_describe_failure_is_speakable():
    line = describe({"ok": False, "reason": "missing_script", "error": "nothing"})
    assert line and line[0].isupper() and line.endswith((".", "!"))
