"""Delegated turns must carry BMO's own conversation memory.

BMO keeps a rolling history in ``Brain.history`` (persisted to ``memory.json``)
and feeds it to the tiny on-NPU model every turn.  ``delegate_to_hermes`` used
to shell out to a *cold* ``hermes -z`` with no conversation at all, so a
follow-up such as "repeat the last answer" was answered from Hermes' own stale
session instead of the turn the user had just heard.

These tests pin the fix: the recent conversation is summarised into the prompt
that Hermes receives, the current request is never duplicated, and every
delegation call site forwards ``self.history``.
"""
import core.llm as llm


def _brain(history=None):
    """A Brain with no network warm-up and no memory.json writes."""
    b = llm.Brain.__new__(llm.Brain)
    b.persist = False
    b.history = history or [{"role": "system", "content": "You are BMO."}]
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


# --------------------------------------------------------------------------
# _delegation_context: pure formatter
# --------------------------------------------------------------------------

def test_empty_history_yields_no_context():
    assert llm._delegation_context(None) == ""
    assert llm._delegation_context([]) == ""


def test_system_prompt_is_never_leaked():
    ctx = llm._delegation_context([
        {"role": "system", "content": "You are BMO. SECRET INSTRUCTIONS."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi friend"},
    ])
    assert "SECRET INSTRUCTIONS" not in ctx
    assert "hello" in ctx and "hi friend" in ctx


def test_prior_exchange_is_preserved():
    ctx = llm._delegation_context([
        {"role": "user", "content": "use the spotykach skill"},
        {"role": "assistant", "content": "pstretch, reel, slice"},
    ])
    assert "spotykach" in ctx
    assert "pstretch, reel, slice" in ctx
    # Oldest first, so Hermes reads it as a conversation.
    assert ctx.index("spotykach") < ctx.index("pstretch")


def test_current_request_is_not_duplicated():
    # think() appends the user turn before delegating, so the last message IS
    # the current request and must not appear in the context block too.
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "repeat the last answer"},
    ]
    ctx = llm._delegation_context(history, current="repeat the last answer")
    assert "old answer" in ctx
    assert "repeat the last answer" not in ctx


def test_dedupe_is_case_and_punctuation_insensitive():
    history = [{"role": "user", "content": "Repeat the last answer!"}]
    assert llm._delegation_context(history, current="repeat the last answer") == ""


def test_only_the_most_recent_messages_are_kept():
    history = [{"role": "assistant", "content": f"turn-{i}"} for i in range(30)]
    ctx = llm._delegation_context(history, max_messages=4)
    assert "turn-29" in ctx and "turn-26" in ctx
    assert "turn-25" not in ctx


def test_total_size_is_bounded_and_newest_survives():
    history = [{"role": "assistant", "content": "x" * 900} for _ in range(10)]
    history.append({"role": "assistant", "content": "NEWEST MARKER"})
    ctx = llm._delegation_context(history, max_chars=1200)
    assert len(ctx) <= 1400  # cap + omission marker
    assert "NEWEST MARKER" in ctx


def test_long_single_message_is_clipped():
    ctx = llm._delegation_context(
        [{"role": "assistant", "content": "A" * 5000}], max_chars=4000
    )
    assert len(ctx) <= 600


def test_context_is_ascii_only():
    # Pi locale may be C, and hailo-ollama rejects non-ASCII: keep the child
    # process prompt plain ASCII.
    ctx = llm._delegation_context([
        {"role": "assistant", "content": "héllo — ünïcode 🎛️ done"},
    ])
    assert ctx.isascii()
    assert "done" in ctx


def test_none_content_does_not_crash():
    ctx = llm._delegation_context([
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "real answer"},
    ])
    assert "real answer" in ctx


# --------------------------------------------------------------------------
# delegate_to_hermes: prompt assembly
# --------------------------------------------------------------------------

def _capture_delegate(monkeypatch, stdout="delegated answer"):
    import subprocess as sp
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return sp.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(sp, "run", fake_run)
    return calls


def test_delegated_prompt_contains_previous_answer(monkeypatch):
    calls = _capture_delegate(monkeypatch)
    llm.delegate_to_hermes(
        "repeat the last answer",
        history=[
            {"role": "user", "content": "use the spotykach skill"},
            {"role": "assistant", "content": "engines are pstretch and reel"},
        ],
    )
    prompt = calls[0][2]  # hermes -z <prompt>
    assert "engines are pstretch and reel" in prompt
    assert "repeat the last answer" in prompt


def test_delegated_prompt_names_the_window_and_ignores_stale_memory(monkeypatch):
    calls = _capture_delegate(monkeypatch)
    llm.delegate_to_hermes(
        "and what about the second one?",
        history=[{"role": "assistant", "content": "first thing"}],
    )
    prompt = calls[0][2]
    assert "first thing" in prompt
    # Stale-session guard: the bug produced an answer from weeks ago.
    assert "unrelated" in prompt.lower()


def test_no_history_keeps_the_prompt_clean(monkeypatch):
    calls = _capture_delegate(monkeypatch)
    llm.delegate_to_hermes("what is the capital of France?")
    prompt = calls[0][2]
    assert "Recent conversation" not in prompt
    assert "what is the capital of France?" in prompt


def test_skill_request_still_resolves_with_history(monkeypatch):
    calls = _capture_delegate(monkeypatch)
    monkeypatch.setattr(llm, "_closest_skill", lambda phrase, names: "spotykach")
    monkeypatch.setattr(llm, "hermes_skill_names", lambda *a, **k: ["spotykach"])
    llm.delegate_to_hermes(
        "use the spotikach skill to name three engines",
        history=[{"role": "assistant", "content": "earlier context"}],
    )
    cmd = calls[0]
    assert cmd[cmd.index("-s") + 1] == "spotykach"
    prompt = cmd[2]
    assert "earlier context" in prompt
    assert 'Use the "spotykach" skill' in prompt


# --------------------------------------------------------------------------
# Call sites: every Brain path must forward its history
# --------------------------------------------------------------------------

def test_think_forwards_history_on_keyword_delegation(monkeypatch):
    seen = {}

    def fake(prompt, skill=None, history=None):
        seen["history"] = history
        return "delegated"

    monkeypatch.setattr(llm, "delegate_to_hermes", fake)
    b = _brain()
    b.think("use hermes to name three engines")
    assert seen["history"] is not None
    assert any(m.get("content") == "use hermes to name three engines" for m in seen["history"])


def test_think_forwards_history_on_dont_know_delegation(monkeypatch):
    """The reported bug: the local model shrugs, the follow-up is delegated."""
    seen = {}

    def fake(prompt, skill=None, history=None):
        seen["prompt"] = prompt
        seen["history"] = history
        return "engines are pstretch and reel"

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp("I don't know, friend!"))
    monkeypatch.setattr(llm, "delegate_to_hermes", fake)
    b = _brain()
    b.history += [
        {"role": "user", "content": "use the spotykach skill to name engines"},
        {"role": "assistant", "content": "engines are pstretch and reel"},
    ]
    out = b.think("repeat the last answer")
    assert "pstretch" in out
    # The just-spoken answer is available to the delegated turn.
    assert any("pstretch" in (m.get("content") or "") for m in seen["history"])


def test_stream_think_forwards_history(monkeypatch):
    seen = {}

    def fake(prompt, skill=None, history=None):
        seen["history"] = history
        return "delegated"

    monkeypatch.setattr(llm, "delegate_to_hermes", fake)
    b = _brain()
    list(b.stream_think("use hermes to name three engines"))
    assert seen["history"] is not None


# --------------------------------------------------------------------------
# "the last answer" must be unambiguous — delegated AND local replies alike
# --------------------------------------------------------------------------

def test_most_recent_answer_is_marked():
    ctx = llm._delegation_context([
        {"role": "user", "content": "use the op-1 skill to summarise the tape engine"},
        {"role": "assistant", "content": "TAPE ENGINE ANSWER"},
        {"role": "user", "content": "use the op-1 skill to write a gorillas prompt"},
        {"role": "assistant", "content": "GORILLAS ANSWER"},
    ])
    assert "BMO (most recent answer): GORILLAS ANSWER" in ctx
    # Only the newest one is marked; older answers stay plain.
    assert "BMO: TAPE ENGINE ANSWER" in ctx
    assert ctx.count("most recent answer") == 1


def test_a_local_non_delegated_reply_counts_as_most_recent():
    # Alternating delegated / non-delegated turns: the last thing spoken may
    # come from local qwen3, and a delegated "repeat that" must still get it.
    ctx = llm._delegation_context([
        {"role": "user", "content": "use the op-1 skill to summarise the tape engine"},
        {"role": "assistant", "content": "TAPE ENGINE ANSWER"},
        {"role": "user", "content": "tell me a fun fact about dogs"},
        {"role": "assistant", "content": "DOGS ANSWER"},
    ])
    assert "BMO (most recent answer): DOGS ANSWER" in ctx
    assert "BMO: TAPE ENGINE ANSWER" in ctx


def test_marker_survives_budget_trimming():
    ctx = llm._delegation_context([
        {"role": "assistant", "content": "OLD " + "x" * 900},
        {"role": "assistant", "content": "NEWEST ANSWER"},
    ], max_chars=300)
    assert "most recent answer): NEWEST ANSWER" in ctx


def test_recall_instruction_tells_hermes_what_to_repeat(monkeypatch):
    calls = _capture_delegate(monkeypatch)
    llm.delegate_to_hermes(
        "repeat the last answer",
        history=[{"role": "assistant", "content": "GORILLAS ANSWER"}],
    )
    prompt = calls[0][2]
    low = prompt.lower()
    assert "asks you to repeat" in low
    assert "most recent answer" in low
    assert "never an earlier bmo line" in low
    assert "GORILLAS ANSWER" in prompt


def test_injected_timestamp_prefix_is_stripped():
    ctx = llm._delegation_context([
        {"role": "user", "content": "[Now: 04:37 PM, Friday, October 23, 2026] tell me a fun fact about dogs"},
        {"role": "assistant", "content": "DOGS ANSWER"},
    ])
    assert "Now:" not in ctx
    assert "User: tell me a fun fact about dogs" in ctx


def test_timestamp_does_not_defeat_current_ask_dedupe():
    ask = "repeat the last answer"
    ctx = llm._delegation_context([
        {"role": "user", "content": "tell me a fun fact about dogs"},
        {"role": "assistant", "content": "DOGS ANSWER"},
        {"role": "user", "content": "[Now: 04:37 PM, Friday, October 23, 2026] " + ask},
    ], current=ask)
    assert ctx.count("repeat the last answer") == 0
