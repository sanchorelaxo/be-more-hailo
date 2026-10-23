"""Who's in front of BMO? — enroll and recognise faces on the onboard Hailo app.

The Pi already has a complete face stack: ``~/face-recog-web`` runs a GStreamer
pipeline (SCRFD detection → tracker → ArcFace embeddings → LanceDB) over the Pi
camera.  This module is the seam between BMO and that pipeline:

  * :func:`match_face_intent` — turns "do you recognize me" / "enroll my face as
    Rob" into a command, with no model in the loop (the tiny on-NPU LLM is not
    reliable at emitting structured JSON).
  * :func:`describe` — turns a result dict into something BMO can say.
  * :class:`FaceID` — owns the handoff around ``scripts/bmo_face.py``:
      - the Hailo-10H is single-tenant, so hailo-ollama must unload the LLM
        first (``core.llm._unload_llm_from_npu``); the LLM reloads lazily on
        BMO's next turn, so no service is ever restarted;
      - only one process can hold the camera, so the shared ``core.camera.Camera``
        is stopped for the duration and started again afterwards.

Everything hardware-touching is injectable (``run``, ``unload_llm``, ``camera``)
so the orchestration is testable on a laptop with no NPU and no camera.
"""
import json
import logging
import os
import re
import subprocess

from core.config import (
    FACE_APP_DIR,
    FACE_ENROLL_TIMEOUT,
    FACE_PYTHON,
    FACE_RECOGNIZE_TIMEOUT,
    FACE_SCRIPT,
)

logger = logging.getLogger(__name__)

UNKNOWN_LABELS = {"", "unknown", "none", "null"}

# --- intents ---------------------------------------------------------------------

# Recognition is opt-in on one word: the utterance has to contain "recognize"
# (or "recognise").  Owner's rule — no word, no scan — because the alternative is
# the 1.7B model answering identity questions out of the chat history and making
# a name up: "Recognize me." once missed this list and came back as the *old*
# label.  Any sentence carrying the word counts, wherever it sits.
#
# This also keeps "who do you see?" / "what is this?" on the camera/VLM route:
# asking what is in view is not asking who it is.
_RECOGNIZE_PATTERNS = [
    r"\brecogni[sz]e\b",
]

# "enroll my face", "register this face as Rob", "hold this face as anonymous",
# "remember my face", "enroll me"
_ENROLL_PATTERNS = [
    r"\b(?:enroll|enrol|register|remember|save|learn|store|add|hold|memori[sz]e)"
    r"\s+(?:my|this|that|his|her)\s+face\b",
    r"\b(?:enroll|enrol|register|remember|save|learn|store|add|hold|memori[sz]e)\s+me\b",
    r"\bput my face in (?:your|the) (?:memory|database|list)\b",
]

# "do you…", "can you…", "would you…" — a question, never an instruction.
_QUESTION_OPENER = re.compile(r"^\s*(?:so,?\s*)?(?:do|can|could|would|will|are|is|did|have)\s+you\b", re.I)

_NAME_AS = re.compile(r"\bas\s+([A-Za-z][A-Za-z .'-]{0,40})", re.I)
_NAME_IS = re.compile(r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z .'-]{0,40})", re.I)
_TRAILING = " .,!?;:"


def _extract_name(text: str) -> str | None:
    """'enroll my face as Robert' → 'Robert'; nothing → None."""
    for pattern in (_NAME_AS, _NAME_IS):
        m = pattern.search(text)
        if m:
            name = m.group(1).strip(_TRAILING).strip()
            # Reject "as soon as possible"-style tails and anything too long.
            if name and len(name.split()) <= 3 and not name.lower().startswith(
                    ("soon", "well", "much", "if", "you")):
                return name
    return None


def match_face_intent(text: str) -> "tuple[str, str | None] | None":
    """('recognize', None) / ('enroll', name|None) / None.

    Only *recognition* phrasings match.  Vision questions — "do you see me?",
    "who do you see", "what is this" — return None and keep the camera/VLM
    route: asking what is in view is not the same as asking who it is.

    A bare "enroll" with no name still returns ('enroll', None) — BMO then asks
    what to call the person instead of guessing a label.
    """
    if not text:
        return None
    low = text.lower()
    for pattern in _RECOGNIZE_PATTERNS:
        if re.search(pattern, low):
            return ("recognize", None)
    # An interrogative is asking whether BMO *can* remember, not asking it to
    # learn a face: "do you remember my face?" must not start an enrollment.
    if _QUESTION_OPENER.match(low):
        return None
    for pattern in _ENROLL_PATTERNS:
        if re.search(pattern, low):
            return ("enroll", _extract_name(text))
    return None


# BMO asks "what should I call you?" when the enroll line carried no name; the
# reply to that question is a name, not an instruction.
_NAME_REPLY_PREFIXES = [
    re.compile(r"^(?:my name is|my name's|call me|i am|i'm|it's|its|as)\s+(.+)$", re.I),
]

_NAME_STOPWORDS = {
    "yes", "no", "yeah", "nope", "ok", "okay", "sure", "maybe", "thanks", "thank",
    "you", "please", "stop", "never", "mind", "not", "don't", "dont", "do", "does",
    "want", "wanna", "like", "the", "a", "an", "of", "to", "for", "what", "what's",
    "weather", "photo", "picture", "take", "cat", "dog", "it", "that", "this",
}


def parse_name_reply(text: str) -> str | None:
    """The answer to 'what should I call you?' → 'Robert' (or None).

    Accepts a bare name ('Robert', 'Mary Jane'), or the usual phrasings
    ('my name is Robert', 'call me Rob').  Rejects anything that reads like a
    normal sentence, so BMO doesn't enroll someone as 'no thanks'.
    """
    if not text:
        return None
    candidate = text.strip().strip(_TRAILING).strip()
    for pattern in _NAME_REPLY_PREFIXES:
        m = pattern.match(candidate)
        if m:
            candidate = m.group(1).strip(_TRAILING).strip()
            break
    words = candidate.split()
    if not candidate or len(words) > 3 or len(candidate) > 40:
        return None
    if any(w.lower().strip(".,'\"") in _NAME_STOPWORDS for w in words):
        return None
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", w) for w in words):
        return None
    return candidate


# --- speaking ---------------------------------------------------------------------

def _pct(confidence) -> int:
    try:
        return int(round(float(confidence) * 100))
    except (TypeError, ValueError):
        return 0


def describe(result: dict, command: str = "recognize") -> str:
    """Plain words for a ``bmo_face.py`` result dict."""
    if not result or not result.get("ok"):
        reason = (result or {}).get("reason", "")
        if reason == "no_face":
            return ("I can't see a face right now. Come closer and look straight at "
                    "my camera, friend!")
        if reason == "camera_busy":
            return "My camera is busy with something else right now. Try me again in a moment!"
        if reason == "already_enrolled":
            label = result.get("label") or "that face"
            return f"I already know that face as {label}!"
        if reason == "timeout":
            return ("I looked but nothing came through from my camera in time. "
                    "Let's try that again!")
        if reason == "needs_name":
            return "Sure! What name should I remember you by?"
        detail = (result.get("error") or reason or "unknown problem").strip().rstrip(".")
        return f"My face scanner didn't start: {detail}."

    if command == "enroll":
        label = result.get("label") or "you"
        samples = result.get("samples") or 1
        looks = f" from {samples} looks" if samples > 1 else ""
        if result.get("merged"):  # same face, older name — say where it went
            old = ", ".join(str(name) for name in result["merged"])
            return (f"Got it! I used to know that face as {old}, so I moved it over to "
                    f"{label}{looks}.")
        if result.get("replaced"):
            return (f"Got it! I knew that face as {label}, but I updated it{looks} — "
                    "so it should recognise you better now.")
        if samples > 1:
            return f"Got it! I learned your face from {samples} looks. I'll remember you as {label}."
        return f"Got it! I'll remember your face as {label}."

    if result.get("known"):
        label = result.get("label")
        conf = _pct(result.get("confidence"))
        if conf >= 80:
            return f"Yes! You're {label}. I'm {conf} percent sure."
        return f"I think you're {label} — about {conf} percent sure."

    faces = result.get("faces") or []
    if faces:
        return ("I see a face, but I don't know who it is yet. Say 'enroll my face as "
                "your name' and I'll remember you!")
    return ("I can't see a face right now. Come closer and look straight at my camera, friend!")


# --- handoff ----------------------------------------------------------------------

class FaceID:
    """Runs ``scripts/bmo_face.py`` on the Pi's onboard Hailo face pipeline.

    ``run``           — (argv: list[str], timeout: float) -> dict of parsed JSON
    ``unload_llm``    — () -> bool; frees the NPU (default: core.llm's helper)
    ``camera``        — core.camera.Camera (or None when presence is off)
    """

    def __init__(self, run=None, unload_llm=None, camera=None,
                 python: str | None = None, script: str | None = None,
                 app_dir: str | None = None,
                 recognize_timeout: float | None = None,
                 enroll_timeout: float | None = None,
                 log=None):
        self._run = run or self._run_script
        self._unload_llm = unload_llm or self._unload_default
        self.camera = camera
        self.python = python or FACE_PYTHON
        self.script = script or FACE_SCRIPT
        self.app_dir = app_dir or FACE_APP_DIR
        self.recognize_timeout = recognize_timeout or FACE_RECOGNIZE_TIMEOUT
        self.enroll_timeout = enroll_timeout or FACE_ENROLL_TIMEOUT
        self.log = log or logger
        # True while a helper run holds the camera on loan.  The agent's
        # presence loop reads it so it does not re-open /dev/video0 mid-run.
        self.busy = False

    # -- public ---------------------------------------------------------------
    def available(self) -> bool:
        return os.path.exists(self.script)

    def recognize(self) -> dict:
        return self._borrow(lambda: self._run(
            ["who", "--app-dir", self.app_dir], self.recognize_timeout))

    def enroll(self, name: str | None = None, replace: bool = True) -> dict:
        """Learn (or re-learn) a face under ``name``.

        ``replace`` defaults on: a spoken "enroll this face as X" must not be
        answered with "I already know that face as X" when the stored samples
        are stale (a profile-posed enrollment silently makes recognition fail).
        """
        argv = ["enroll", "--app-dir", self.app_dir]
        if name:
            argv += ["--name", name]
        if replace:
            argv.append("--replace")
        return self._borrow(lambda: self._run(argv, self.enroll_timeout))

    def enrolled(self) -> list[str]:
        result = self._run(["list", "--app-dir", self.app_dir], 30)
        return list(result.get("labels") or []) if result.get("ok") else []

    def forget(self, name: str) -> dict:
        return self._run(["forget", "--name", name, "--app-dir", self.app_dir], 30)

    # -- handoff --------------------------------------------------------------
    def _borrow(self, action):
        """NPU + camera are exclusive: free both, work, hand the camera back."""
        self._release_npu()
        stopped = self._stop_camera()
        self.busy = True
        try:
            return action()
        finally:
            self.busy = False
            if stopped:  # only restart a camera we took away
                self._start_camera()

    def _release_npu(self):
        try:
            if not self._unload_llm():
                self.log.warning("FaceID: LLM unload failed; the face pipeline may be starved.")
        except Exception as exc:  # never let the handoff kill the turn
            self.log.warning(f"FaceID: LLM unload raised: {exc}")

    def _stop_camera(self) -> bool:
        """Stop the shared camera; True only if we actually took it away."""
        if self.camera is not None and getattr(self.camera, "running", False):
            try:
                self.camera.stop()
                return True
            except Exception as exc:
                self.log.warning(f"FaceID: camera stop failed: {exc}")
        return False

    def _start_camera(self):
        if self.camera is not None:
            try:
                self.camera.start()
            except Exception as exc:
                self.log.warning(f"FaceID: camera restart failed: {exc}")

    # -- defaults -------------------------------------------------------------
    @staticmethod
    def _unload_default() -> bool:
        from core.llm import _unload_llm_from_npu
        return _unload_llm_from_npu()

    def _run_script(self, argv: list[str], timeout: float) -> dict:
        if not self.available():
            return {"ok": False, "reason": "missing_script",
                    "error": f"face helper not found at {self.script}"}
        cmd = [self.python, self.script] + list(argv)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout + 15, env=self._env())
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout"}
        except Exception as exc:
            return {"ok": False, "reason": "exec_failed", "error": str(exc)}
        return _parse(proc)

    def _env(self) -> dict:
        env = dict(os.environ)
        paths = [self.app_dir, os.environ.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = ":".join(p for p in paths if p)
        env.setdefault("HAILO_APP_DIR", self.app_dir)
        # Tell the helper we are the caller it is refusing for ("bmo-voice is
        # active").  _borrow already freed the NPU and the camera;
        # without this a spoken enroll/recognize can never run on a live robot.
        env["BMO_FACE_AGENT_HANDOFF"] = "1"
        return env


def _parse(proc) -> dict:
    """One JSON line on stdout, diagnostics on stderr."""
    out = (proc.stdout or "").strip()
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
    return {"ok": False, "reason": "bad_output",
            "error": tail[0][:200], "returncode": proc.returncode}
