"""The 1.7B model can collapse into answering with only a JSON action and no
words.  If that bare action is persisted to history it is echoed back as the
model's own last turn (``_with_current_context`` trims history to system + last
2 messages), so the model mimics it and every later question returns
action-only -> the "tape deck got jammed" fallback.  ``_history_safe`` stops
the loop by storing a spoken line instead of a bare action.
"""
from core.llm import _history_safe


def test_action_only_is_replaced():
    assert _history_safe('{"action": "set_expression", "value": "happy"}') == "Okay friend!"


def test_text_plus_action_is_kept():
    out = _history_safe('Paris! {"action": "set_expression", "value": "happy"}')
    assert out == 'Paris! {"action": "set_expression", "value": "happy"}'


def test_empty_content_gets_fallback():
    assert _history_safe("") == "Okay friend!"
    assert _history_safe(None) == "Okay friend!"


def test_punctuation_only_is_replaced():
    # An emoji-only reply must not be persisted as "speech" either.
    assert _history_safe("😊") == "Okay friend!"


def test_nested_action_objects_are_all_peeled():
    # Multiple actions with no words still count as action-only.
    out = _history_safe('{"action": "set_expression", "value": "heart"}{"action": "play", "value": "song"}')
    assert out == "Okay friend!"


def test_whitespace_only_is_replaced():
    assert _history_safe("   \n\t  ") == "Okay friend!"


def test_normal_text_is_unchanged():
    assert _history_safe("The capital of France is Paris!") == "The capital of France is Paris!"