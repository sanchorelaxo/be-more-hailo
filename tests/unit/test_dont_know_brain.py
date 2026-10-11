"""End-to-end: a "don't know" / "can't answer" reply must hand the turn to
Hermes inside a real Brain.think()/stream_think() call, and a genuine answer
must not.  The LLM HTTP layer and delegate_to_hermes are faked.
"""
import core.llm as llm


def _brain():
    # Bypass Brain.__init__ (avoids network warm-up and memory.json).
    b = llm.Brain.__new__(llm.Brain)
    b.persist = False
    b.history = [{"role": "system", "content": "You are BMO."}]
    b.briefing_ready = None
    b.reminders = None
    b.home = None
    return b


class _Resp:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


class _StreamResp(_Resp):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        import json
        yield json.dumps({"message": {"content": self._content}})


def test_think_delegates_on_dont_know(monkeypatch):
    delegated = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp("I don't know, friend!"))
    monkeypatch.setattr(llm, "delegate_to_hermes",
                        lambda prompt: delegated.append(prompt) or "The capital of Australia is Canberra.")
    out = _brain().think("What is the capital of Australia?")
    assert any(out.startswith(p) for p in llm._DELEGATION_PREFACES)
    assert "Canberra" in out
    assert delegated == ["What is the capital of Australia?"]


def test_think_does_not_delegate_on_real_answer(monkeypatch):
    delegated = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp("The capital of France is Paris!"))
    monkeypatch.setattr(llm, "delegate_to_hermes",
                        lambda prompt: delegated.append(prompt) or "unused")
    out = _brain().think("What is the capital of France?")
    assert "Paris" in out
    assert delegated == []


def test_stream_think_delegates_on_cant_answer(monkeypatch):
    delegated = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _StreamResp("I can't answer that, friend."))
    monkeypatch.setattr(llm, "delegate_to_hermes",
                        lambda prompt: delegated.append(prompt) or "The capital of Australia is Canberra.")
    yielded = list(_brain().stream_think("What is the capital of Australia?"))
    joined = " | ".join(yielded)
    assert "Canberra" in joined
    assert any(p in joined for p in llm._DELEGATION_PREFACES)
    assert delegated == ["What is the capital of Australia?"]


def test_stream_think_does_not_delegate_on_real_answer(monkeypatch):
    delegated = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _StreamResp("Paris is the capital of France."))
    monkeypatch.setattr(llm, "delegate_to_hermes",
                        lambda prompt: delegated.append(prompt) or "unused")
    yielded = list(_brain().stream_think("What is the capital of France?"))
    assert "Paris" in " ".join(yielded)
    assert delegated == []