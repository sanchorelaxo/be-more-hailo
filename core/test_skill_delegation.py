"""``use <x> skill`` — the skill flavour of Hermes delegation.

A prompt that contains "use <name> skill" (with or without the word "hermes")
must be handed to the Pi's Hermes Agent exactly like the "use hermes ..."
trigger, except that Hermes is also told to preload (``-s``) the installed
skill whose name is closest to what the user said — voice transcription
mangles skill names ("spotikach" for "spotykach"), so matching is fuzzy.

Everything here is hardware-free: the skills directory is a tmp_path fixture
and ``subprocess`` is a fake module.
"""
import sys
import types

import core.config as config
import core.llm as llm


# --------------------------------------------------------------------------- #
#  Parsing the spoken request
# --------------------------------------------------------------------------- #

def test_extracts_skill_name_and_rest_of_ask():
    phrase, ask = llm._extract_skill_request(
        "use the weather skill to tell me tomorrow's forecast"
    )
    assert phrase == "weather"
    assert "forecast" in ask
    assert "skill" not in ask.lower()


def test_extraction_is_case_insensitive():
    phrase, _ = llm._extract_skill_request("Use Spotykach Skill and make a beat")
    assert phrase == "spotykach"


def test_extraction_accepts_plural_skills():
    phrase, _ = llm._extract_skill_request("use my ffmpeg skills to trim the film")
    assert phrase == "ffmpeg"


def test_no_skill_phrase_for_plain_prompt():
    assert llm._extract_skill_request("tell me a joke") is None
    assert llm._extract_skill_request("use hermes what's the weather") is None
    assert llm._extract_skill_request("what skills do you have?") is None


def test_multiword_skill_phrase():
    phrase, _ = llm._extract_skill_request("use the hermes agent skill to explain")
    assert phrase == "hermes agent"


# --------------------------------------------------------------------------- #
#  Discovering skills + fuzzy match
# --------------------------------------------------------------------------- #

def _make_skill(root, category, name, frontmatter_name=None):
    d = root / category / name if category else root / name
    d.mkdir(parents=True)
    fm = f"---\nname: {frontmatter_name or name}\ndescription: test\n---\n\n# {name}\n"
    (d / "SKILL.md").write_text(fm, encoding="utf-8")
    return d


def test_discovers_nested_skill_names(tmp_path):
    _make_skill(tmp_path, "music", "spotykach")
    _make_skill(tmp_path, "devops", "be-more-hailo")
    _make_skill(tmp_path, "", "flat-skill")
    names = llm.hermes_skill_names(tmp_path)
    assert "spotykach" in names
    assert "be-more-hailo" in names
    assert "flat-skill" in names


def test_missing_skills_dir_is_empty_not_fatal(tmp_path):
    assert llm.hermes_skill_names(tmp_path / "nope") == []


def test_closest_skill_tolerates_transcription_typo():
    names = ["spotykach", "ffmpeg", "be-more-hailo"]
    assert llm._closest_skill("spotikach", names) == "spotykach"


def test_closest_skill_prefers_substring():
    names = ["ffmpeg", "be-more-hailo", "hermes-agent"]
    assert llm._closest_skill("hermes agent", names) == "hermes-agent"


def test_closest_skill_returns_none_when_nothing_is_close():
    assert llm._closest_skill("zebra wrangling", ["spotykach", "ffmpeg"]) is None


def test_short_generic_word_does_not_match_an_unrelated_skill():
    # Live regression: "use the video skill" fuzzy-matched the "ideation"
    # skill because difflib scored two short, unrelated words above cutoff.
    assert llm._closest_skill("video", ["ideation", "ffmpeg", "box"]) is None


def test_short_name_with_typo_still_matches_on_shared_prefix():
    assert llm._closest_skill("videoo", ["videos", "ideation"]) == "videos"


# --------------------------------------------------------------------------- #
#  Trigger detection used by think() / stream_think()
# --------------------------------------------------------------------------- #

def test_hermes_request_matches_keyword_and_skill_triggers():
    assert llm._is_hermes_request("use hermes, what is the capital of France?")
    assert llm._is_hermes_request("ask hermes to summarise this")
    assert llm._is_hermes_request("use the spotykach skill to make a beat")


def test_hermes_request_ignores_ordinary_prompts():
    assert not llm._is_hermes_request("tell me a joke")
    assert not llm._is_hermes_request("what skills do you have?")


# --------------------------------------------------------------------------- #
#  CLI invocation
# --------------------------------------------------------------------------- #

class _FakeProc:
    def __init__(self, stdout="Hermes answer.", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _install_fake_subprocess(monkeypatch, proc=None):
    calls = []
    fake = types.ModuleType("subprocess")

    class TimeoutExpired(Exception):
        pass

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        return proc or _FakeProc()

    fake.run = run
    fake.TimeoutExpired = TimeoutExpired
    monkeypatch.setitem(sys.modules, "subprocess", fake)
    return calls


def test_delegation_preloads_closest_skill(tmp_path, monkeypatch):
    _make_skill(tmp_path, "music", "spotykach")
    _make_skill(tmp_path, "media", "ffmpeg")
    monkeypatch.setattr(config, "HERMES_SKILLS_DIR", str(tmp_path))
    calls = _install_fake_subprocess(monkeypatch)

    out = llm.delegate_to_hermes("use the spotikach skill to make a lo-fi beat")

    assert out == "Hermes answer."
    cmd = calls[0]
    assert cmd[:2] == ["hermes", "-z"]
    assert "-s" in cmd
    assert cmd[cmd.index("-s") + 1] == "spotykach"
    prompt = cmd[2]
    assert "spotykach" in prompt.lower()
    assert "lo-fi beat" in prompt
    # The trigger phrase itself must not be forwarded verbatim.
    assert "use the spotikach skill" not in prompt.lower()


def test_delegation_without_skill_match_still_delegates(tmp_path, monkeypatch):
    _make_skill(tmp_path, "music", "spotykach")
    monkeypatch.setattr(config, "HERMES_SKILLS_DIR", str(tmp_path))
    calls = _install_fake_subprocess(monkeypatch)

    out = llm.delegate_to_hermes("use the zebra wrangling skill to tame a zebra")

    assert out == "Hermes answer."
    cmd = calls[0]
    assert "-s" not in cmd
    # With no local match the full ask (incl. the phrase) goes to Hermes so its
    # own skill matcher can still have a go.
    assert "zebra" in cmd[2].lower()


def test_plain_hermes_trigger_has_no_skill_flag(tmp_path, monkeypatch):
    _make_skill(tmp_path, "music", "spotykach")
    monkeypatch.setattr(config, "HERMES_SKILLS_DIR", str(tmp_path))
    calls = _install_fake_subprocess(monkeypatch)

    llm.delegate_to_hermes("use hermes what is the capital of France?")

    cmd = calls[0]
    assert "-s" not in cmd
    assert "capital of France" in cmd[2]
    assert "use hermes" not in cmd[2].lower()


# --------------------------------------------------------------------------- #
#  End to end through a real Brain
# --------------------------------------------------------------------------- #

def _brain():
    b = llm.Brain.__new__(llm.Brain)
    b.persist = False
    b.history = [{"role": "system", "content": "You are BMO."}]
    b.briefing_ready = None
    b.reminders = None
    b.home = None
    return b


def test_think_routes_skill_prompt_to_hermes(tmp_path, monkeypatch):
    _make_skill(tmp_path, "music", "spotykach")
    monkeypatch.setattr(config, "HERMES_SKILLS_DIR", str(tmp_path))
    _install_fake_subprocess(monkeypatch, _FakeProc(stdout="Loaded the looper skill."))

    def _boom(*a, **k):
        raise AssertionError("local LLM must not be called for a skill request")

    monkeypatch.setattr(llm.requests, "post", _boom)

    out = _brain().think("use the spotikach skill to make a lo-fi beat")
    assert out == "Loaded the looper skill."


def test_stream_think_routes_skill_prompt_to_hermes(tmp_path, monkeypatch):
    _make_skill(tmp_path, "music", "spotykach")
    monkeypatch.setattr(config, "HERMES_SKILLS_DIR", str(tmp_path))
    _install_fake_subprocess(monkeypatch, _FakeProc(stdout="Loaded the looper skill."))

    def _boom(*a, **k):
        raise AssertionError("local LLM must not be called for a skill request")

    monkeypatch.setattr(llm.requests, "post", _boom)

    yielded = list(_brain().stream_think("use the spotikach skill to make a lo-fi beat"))
    assert "Loaded the looper skill." in " ".join(yielded)
