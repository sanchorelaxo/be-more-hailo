"""When the tiny on-NPU model can't answer ("I don't know" / "I can't answer"),
the turn should be handed to Hermes instead of leaving the user with a dead-end.
``_is_cant_answer`` is the pure detector; ``_delegation_preface`` supplies the
spoken "BMO came up with nothing, let's try Hermes" hand-off line.
"""
from core.llm import _is_cant_answer, _delegation_preface, _DELEGATION_PREFACES


def test_dont_know_is_cant_answer():
    assert _is_cant_answer("I don't know, friend!")
    assert _is_cant_answer("BMO doesn't know that one.")
    assert _is_cant_answer("I do not know that.")


def test_cant_answer_phrasings():
    assert _is_cant_answer("I can't answer that, friend.")
    assert _is_cant_answer("BMO cannot answer that.")
    assert _is_cant_answer("I couldn't tell you that.")
    assert _is_cant_answer("I couldn't say.")


def test_no_idea_and_not_sure():
    assert _is_cant_answer("BMO has no idea about that.")
    assert _is_cant_answer("I'm not sure, friend.")


def test_case_insensitive():
    assert _is_cant_answer("I DON'T KNOW THAT.")


def test_a_real_answer_is_not_cant_answer():
    assert not _is_cant_answer("The capital of France is Paris!")
    assert not _is_cant_answer("Paris is the capital of France.")
    assert not _is_cant_answer("Oh my! Let me tell you about it.")


def test_empty_is_not_cant_answer():
    assert not _is_cant_answer("")
    assert not _is_cant_answer("   ")


def test_preface_is_one_of_the_variants():
    for _ in range(20):
        assert _delegation_preface() in _DELEGATION_PREFACES


def test_preface_mentions_hermes():
    # Every variant must actually name Hermes so the hand-off is audible.
    assert all("Hermes" in p for p in _DELEGATION_PREFACES)