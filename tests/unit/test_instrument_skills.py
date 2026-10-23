"""Deployment check for the music/instruments skill pack (spotykach as probe).

The ``"use the spotykach skill ..."`` voice trigger only does anything useful
if that skill is actually installed where BMO's Hermes Agent looks for skills
(``HERMES_SKILLS_DIR``, i.e. ``~/.hermes/skills`` on the Pi).  A skill pack
copied only halfway — missing ``references/``, a truncated SKILL.md, an empty
``scripts/`` — still resolves by name and then gives Hermes nothing to work
with, which is a silent failure on a voice device.

These tests read the real skill tree: no subprocess, no hardware.  They skip
when the pack was never installed on this machine, unless
``BMO_REQUIRE_INSTRUMENT_SKILLS=1`` is set, in which case a missing pack is a
failure — use that on the Pi (or in CI) to assert the deployment.
"""
import ast
import os
import re
import sys
import types
from pathlib import Path

import pytest

import core.config as config
import core.llm as llm

PACK_RELATIVE = Path("music") / "hermes-music-sampling" / "skills" / "instruments"
# Every skill in the pack must be present after a full copy.
EXPECTED_INSTRUMENTS = (
    "fx-wizard",
    "hypno-2",
    "kaoss-dj-mixer",
    "kaossilator-v1",
    "kaossilator-v2",
    "minikp-v1",
    "minikp-v2",
    "op-1",
    "quneo",
    "qunexus",
    "sp404-mk2",
    "spotykach",
    "spotykach-engine-authoring",
    "wave-bard",
    "wavedrum-mini",
)


def _candidates():
    env = os.environ.get("BMO_INSTRUMENT_SKILLS_DIR")
    if env:
        yield Path(env)
    hermes_root = Path(os.path.expanduser(str(config.HERMES_SKILLS_DIR)))
    yield hermes_root / PACK_RELATIVE
    yield hermes_root / "music" / "instruments"
    yield Path("~/.hermes/skills").expanduser() / PACK_RELATIVE
    yield Path("~/Documents/git/hermes-music-sampling/skills/instruments").expanduser()


def pack_root() -> Path:
    """Root of the installed instrument pack, or skip/fail when it is absent."""
    for candidate in _candidates():
        if (candidate / "SKILL.md").is_file():
            return candidate
    message = (
        "music/instruments skill pack is not installed (looked in: "
        + ", ".join(str(c) for c in _candidates())
        + ")"
    )
    if os.environ.get("BMO_REQUIRE_INSTRUMENT_SKILLS") == "1":
        pytest.fail(message)
    pytest.skip(message)
    raise AssertionError(message)  # unreachable: pytest.skip always raises


def _frontmatter(skill_md: Path) -> str:
    return skill_md.read_text(encoding="utf-8", errors="replace").split("---")[1]


def _frontmatter_field(skill_md: Path, field: str) -> str | None:
    m = re.search(rf"(?m)^{field}:\s*(.+)$", _frontmatter(skill_md))
    return m.group(1).strip().strip("'\"") if m else None


# --------------------------------------------------------------------------- #
#  The pack itself
# --------------------------------------------------------------------------- #

def test_pack_has_every_expected_instrument():
    root = pack_root()
    present = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    missing = [name for name in EXPECTED_INSTRUMENTS if name not in present]
    assert not missing, f"instrument skills missing from {root}: {missing}"


def test_every_instrument_skill_has_usable_frontmatter():
    root = pack_root()
    problems = []
    for name in EXPECTED_INSTRUMENTS:
        skill_md = root / name / "SKILL.md"
        if not skill_md.is_file():
            problems.append(f"{name}: no SKILL.md")
            continue
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        description = _frontmatter_field(skill_md, "description")
        if _frontmatter_field(skill_md, "name") is None:
            problems.append(f"{name}: frontmatter has no name")
        if not description:
            problems.append(f"{name}: frontmatter has no description")
        elif len(description) > 1024:
            problems.append(f"{name}: description is {len(description)} chars (max 1024)")
        if len(text) < 500 or "# " not in text:
            problems.append(f"{name}: SKILL.md looks truncated ({len(text)} chars)")
    assert not problems, "broken skills in pack:\n" + "\n".join(problems)


# --------------------------------------------------------------------------- #
#  Spotykach in detail: the voice-path probe
# --------------------------------------------------------------------------- #

def test_spotykach_skill_ships_its_references_and_scripts():
    skill = pack_root() / "spotykach"
    assert (skill / "SKILL.md").is_file()

    engines = skill / "references" / "engines"
    engine_docs = sorted(p.name for p in engines.glob("*.md"))
    # The engine reference set is the substance of this skill; a partial copy
    # that drops it leaves Hermes with prose and no engine detail.
    assert len(engine_docs) >= 15, f"only {len(engine_docs)} engine refs in {engines}"
    for expected in ("engines.md", "firmware.md", "midi-cc.md", "sd-card.md"):
        assert (skill / "references" / expected).is_file(), f"missing references/{expected}"

    scripts = skill / "scripts"
    assert (scripts / "sk_engines.py").is_file()
    assert (scripts / "wav_builder.py").is_file()


def test_spotykach_skill_parses_as_python():
    scripts = pack_root() / "spotykach" / "scripts"
    for path in sorted(scripts.glob("*.py")):
        # ast.parse catches a truncated or binary-mangled copy.
        ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))


def test_spotykach_frontmatter_name_matches_directory():
    skill_md = pack_root() / "spotykach" / "SKILL.md"
    assert _frontmatter_field(skill_md, "name") == "spotykach"
    # llm's own parser must agree with the file (it is what the voice path uses).
    assert llm._skill_frontmatter_name(str(skill_md)) == "spotykach"


def test_spoken_typo_resolves_to_spotykach_not_the_engine_skill():
    root = pack_root()
    names = llm.hermes_skill_names(root)
    assert "spotykach" in names
    # Both "spotykach" and "spotykach-engine-authoring" are installed; the
    # spoken name must never jump to the authoring skill.
    assert llm._closest_skill("spotikach", names) == "spotykach"
    assert llm._closest_skill("spotykach", names) == "spotykach"


def test_spoken_engine_wording_reaches_the_authoring_skill():
    names = llm.hermes_skill_names(pack_root())
    assert llm._closest_skill("spotykach engine authoring", names) == "spotykach-engine-authoring"


def _install_fake_subprocess(monkeypatch, stdout="Loaded the looper skill."):
    calls = []

    class _FakeProc:
        returncode = 0

        def __init__(self):
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProc()

    fake = types.ModuleType("subprocess")
    fake.run = fake_run
    fake.TimeoutExpired = type("TimeoutExpired", (Exception,), {})
    fake.SubprocessError = type("SubprocessError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "subprocess", fake)
    return calls


def test_delegation_preloads_the_installed_spotykach_skill(monkeypatch):
    """End to end against the real pack: spoken request -> `hermes -s spotykach`."""
    pack = pack_root()
    monkeypatch.setattr(config, "HERMES_SKILLS_DIR", str(pack))
    calls = _install_fake_subprocess(monkeypatch)

    out = llm.delegate_to_hermes("use the spotikach skill to describe the tape engine")

    assert out == "Loaded the looper skill."
    cmd = calls[0]
    assert cmd[:2] == ["hermes", "-z"]
    assert cmd[cmd.index("-s") + 1] == "spotykach"
    prompt = cmd[2]
    assert "spotykach" in prompt.lower()
    assert "tape engine" in prompt.lower()
    assert "spotykach-engine-authoring" not in prompt
