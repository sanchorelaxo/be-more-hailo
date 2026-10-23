# Hermes delegation — how BMO hands off to the Pi's Hermes Agent

BMO's on-NPU model (qwen3:1.7b) is tiny. For anything it can't do well, BMO
shells out to the Pi's own Hermes Agent CLI and speaks the richer answer. This
gives BMO a far larger model + full tool access while staying in-character.

## The three triggers (all in `core/llm.py`)

### 0. Skill trigger — "use <x> skill" (no "hermes" needed)

`_extract_skill_request()` matches `use [the|a|an|my|your] <name> skill[s]` and
the turn is delegated exactly like the keyword trigger, with one addition: the
named skill is fuzzy-matched against the skills installed for the Pi's Hermes
(`hermes_skill_names()` walks `HERMES_SKILLS_DIR`, default `~/.hermes/skills`,
skipping dot-dirs such as `.archive/` and support dirs
`references/templates/scripts/assets`) and the winner is preloaded in the
session with `hermes -s <skill>` plus named in the prompt
(`Use the "<skill>" skill to answer this. <ask>`).

Match tiers: exact normalised name (`"hermes agent"` == `"hermes-agent"`),
then substring either way, then difflib `SequenceMatcher.ratio()` >=
`HERMES_SKILL_MATCH_CUTOFF` (0.6). Names shorter than 8 chars must ALSO share
a >=3-char prefix — without that guard "use the video skill" matched the
unrelated `ideation` skill (live regression, covered in
`tests/unit/test_skill_delegation.py`).

No local match is not a failure: the full prompt (trigger included) goes to
Hermes without `-s`, so Hermes' own matcher still gets a shot. Verified live on
the Pi: "use the wrongname skill to say hello in French" logs `-> None` and
still answers gracefully.

### 1. Keyword trigger (`HERMES_TRIGGERS`)

Saying any of these hands the turn to Hermes:

- `use hermes <ask>`
- `ask hermes <ask>`
- `hermes, <ask>`
- `hermes: <ask>`

Trigger words are stripped (word-anchored) from the prompt so Hermes sees the
real ask; falls back to the full prompt if stripping empties it.

### 2. Don't-know auto-trigger (added later — `_is_cant_answer`)

When the local qwen3 returns a shrug, BMO no longer gives up — it emits a
preface and delegates the follow-up. `_is_cant_answer()` matches (all
word-boundary, case-insensitive):

- `don't know` / `do not know` / `doesn't know` / `does not know`
- `can't answer` / `cannot answer` / `can not answer`
- `couldn't tell|say|answer`
- `have|has|got no idea`
- `I'm not sure`

Preface lines are randomized from five ASCII variants (match the user's
requested phrasing), e.g. "BMO came up with nothing. Let's try my big friend
Hermes!" and "BMO drew a blank there. Let me ask Hermes!".

Wired in BOTH `think()` and `stream_think()`. Fail-safe: if delegation returns
empty, BMO keeps its original shrug rather than clobbering it.

## Conversation memory — delegated turns are COLD one-shots

`hermes -z` with no session flags starts fresh, every time. BMO's own memory
(`Brain.history`, persisted to `memory.json`, capped at `MAX_HISTORY_MESSAGES`
/ `MAX_CONTEXT_CHARS`) was already fed to local qwen3, but delegation used to
pass nothing — so a follow-up like "repeat the last answer" was answered from
Hermes' OWN stale session (the observed symptom: an answer from weeks ago).

Fix: `delegate_to_hermes(prompt, skill=None, history=None)` renders a bounded
transcript into the prompt via `_delegation_context()`:

```
Be ultra-succinct. ...

Recent conversation between the user and BMO, oldest first:
User: use the op-1 skill to summarise the tape engine
BMO: 6-track recorder, overdub, bar markers, lift/bounce ...
User: use the op-1 skill to explain seeing gorillas on the device screen
BMO (most recent answer): That is the OP-1 hidden gorilla display ...

Continue from this conversation. Ignore any unrelated or older conversation
you may remember; only the turns above are relevant. If the current request
asks you to repeat, restate or recall the last answer, echo the line labelled
"BMO (most recent answer)" verbatim - never an earlier BMO line.

Current request: repeat the last answer
```

The `BMO (most recent answer):` label is what makes recall deterministic. With
several BMO answers in the window, a bare "repeat the last answer" was observed
picking an EARLIER one (two op-1 delegations: tape-engine then the gorilla-screen question;
the recall after the gorillas turn returned the tape answer). The label is
applied after trimming so it can never be the line that falls off, and the
prompt tells Hermes explicitly to echo that line.

`_delegation_context()` also strips the `[Now: 04:37 PM, ...]` prefix that
`think()` stamps onto user turns — otherwise the transcript is noisy AND the
current-ask de-duplication misses (the stamped copy no longer equals the ask).

Alternating delegated and local turns is the strongest recall test: the last
thing spoken may have come from local qwen3, not Hermes. Verified live on the
Pi (7 turns through `stream_think`): delegated → local → delegated recall,
delegated → local → delegated recall, and a purely local recall all returned
the correct newest answer.

Rules the helper enforces (all pinned by `tests/unit/test_delegation_memory.py`):

- system prompt NEVER included (no instruction leakage)
- current request de-duplicated — `think()`/`stream_think()` append the user
  turn BEFORE delegating, so the last history entry is the ask itself
- window = last `HERMES_DELEGATE_HISTORY_MESSAGES` messages, each clipped to
  `HERMES_DELEGATE_HISTORY_MSG_CHARS`, whole block to
  `HERMES_DELEGATE_HISTORY_CHARS` (newest-first trim, `[earlier turns omitted]`
  marker when trimmed)
- ASCII-only output — the Pi locale may be `C` and the child-process prompt
  must never carry emoji
- newest BMO line labelled `BMO (most recent answer):`; `[Now: ...]` prefixes
  stripped from user turns
- all FOUR call sites pass `history=self.history` (keyword trigger in
  `think()` and `stream_think()`, plus both don't-know auto-delegations)

`web_app.py /api/chat` calls `brain.set_history(request.history)` first, so the
browser-owned conversation is included too.

## Verifying recall live (multi-turn probe)

`scripts/live_recall_probe.py` is the re-runnable probe; run it on the Pi with
the deployed venv, passing the context asks plus the recall ask last:

    ssh sanchobot@100.119.186.62 'cd ~/be-more-agent && ./venv/bin/python \
      ~/.hermes/skills/devops/be-more-hailo/scripts/live_recall_probe.py \
      "use the op-1 skill to summarise the tape engine" \
      "use the op-1 skill to explain seeing gorillas on the device screen" \
      "use the op-1 skill to repeat the last answer"'

It builds `Brain.__new__(llm.Brain)` with `persist=False` (never writes
`memory.json`), `history=[system prompt]`, `briefing_ready/reminders/home=None`,
and consumes `stream_think()` so the VOICE path is exercised. On failure it
dumps the actual `_delegation_context()` the child saw — usually enough to tell
"missing turn" from "ambiguous turn" at a glance.

Pass criteria: >= 0.90 similarity to the immediately preceding answer and
< 0.50 to every earlier one. Do NOT assert exact equality or an expected
answer: Hermes rewords between runs, and the same question can legitimately
come back with different content (the OP-1 gorilla screen was answered once as
a disk-mode easter egg and once as the Finger sequencer). Anything between 0.5
and 0.9 is ambiguous — re-run before filing it as a bug.

When a reported recall bug does not reproduce, instrument instead of guessing:
wrap `llm.delegate_to_hermes` in a shim that prints
`llm._delegation_context(history, current=prompt)` (or patch
`subprocess.run` to dump `cmd`) and run the reported scenario verbatim. This is
how "the tape answer came back" was traced to an unmarked transcript rather
than a lost turn: every turn was present, the model simply had to guess which
BMO line was "the last answer".

Use the user's real phrasing in probes. A prompt the user would never say
("write a short gorillas themed prompt") gets corrected and exercises the wrong
path — probe the behaviour they actually described.

Rejected alternative: a dedicated long-lived Hermes session (`-c <name>`).
It grows unbounded on the Pi's SD card, interleaves the web and voice services,
and silently re-introduces the exact stale-context bug when the session is old.

## CLI invocation

```python
subprocess.run(["hermes", "-z", prompt, "-m", model], timeout=..., ...)
# skill requests append:  ["-s", resolved_skill]
```

- `-z <prompt>` = the global one-shot flag (`hermes -z PROMPT`, NOT a
  `hermes chat` subcommand). Prints ONLY the final response to stdout — no
  banner, spinner, tool previews, or trailing `session_id:` line. Clean output
  AND it keeps verbose answers out of BMO's persistent history (which triggers
  the KV-cache overflow).
- `-m <model>` = pin the model. BMO uses `HERMES_DELEGATE_MODEL`
  (default `deepseek/deepseek-v4-flash-0731`).
- Do NOT use `hermes chat -q -Q` — a noisier equivalent that emits a trailing
  `session_id:` line needing stripping and stuffs more into history.

## Config (core/config.py — env-overridable)

| Var | Default | Effect |
|-----|---------|--------|
| `HERMES_DELEGATE_MODEL` | `deepseek/deepseek-v4-flash-0731` | model Hermes runs |
| `HERMES_DELEGATE_TIMEOUT` | `180` | seconds to wait |
| `HERMES_DELEGATE_SUCCINCT` | "Be ultra-succinct. Keep your answer to one or two short sentences." | suffix appended to the delegated prompt |
| `HERMES_SKILLS_DIR` | `~/.hermes/skills` | where `use <x> skill` looks up installed skills |
| `HERMES_SKILL_MATCH_CUTOFF` | `0.6` | difflib similarity floor for a skill-name match |
| `HERMES_DELEGATE_HISTORY_MESSAGES` | `6` | last N messages of BMO history sent to Hermes |
| `HERMES_DELEGATE_HISTORY_CHARS` | `2000` | total character budget for that transcript |
| `HERMES_DELEGATE_HISTORY_MSG_CHARS` | `500` | per-message clip |

## Robustness — never freeze or crash offline

`delegate_to_hermes()` never raises. Every failure maps to a graceful BMO line:

| Failure | BMO says |
|---------|----------|
| `FileNotFoundError` (CLI missing) | "isn't installed right now" |
| non-zero exit (404 model, auth, CLI error) | "couldn't answer that one — it hit an error" |
| `subprocess.TimeoutExpired` | "taking too long to think" |
| any other exception | "hit a snag" |

## Pitfalls

- **Model must be a Hermes-catalog model — NOT BMO's local qwen3.** `-m`
  resolves against Hermes's provider catalog (Nous/OpenRouter), not
  hailo-ollama. Setting `HERMES_DELEGATE_MODEL=qwen3:1.7b` makes the CLI exit 1
  with `HTTP 404: Model 'qwen3:1.7b' not found` (graceful error, no crash).
  Delegating back to local qwen would require registering hailo-ollama as a
  custom provider in `~/.hermes/config.yaml`, which contends with BMO over the
  single-tenant NPU — not advisable.
- `subprocess` is imported INSIDE `delegate_to_hermes()` (module scope has no
  `subprocess`), and config attrs live in `core.config`, not `core.llm` — so
  tests must patch via env vars, not `core.llm.subprocess` / `core.llm.HERMES_*`.
- `hermes -z` runs as `sanchobot` on the Pi, using the Pi's hermes profile +
  credentials.
- Real delegation returns ONE yield (the delegated answer) vs. local qwen3's
  multiple chunks — that's how you tell it worked.
- Force the failure path in a test: `HERMES_DELEGATE_MODEL=nonsense/x` → CLI
  exits non-zero → friendly error in ~5s, no hang.
- `-s/--skills` is NOT validated before the model call — a bogus skill name
  still runs, so a bad fuzzy match degrades to "Hermes ignores the directive"
  rather than erroring. Keep the matcher conservative.
- After deploying `core/llm.py`, restart `bmo-web.service` AND
  `bmo-voice.service`: both import it at startup, so without a restart the old
  trigger set stays live.

## "Tap to speak doesn't work after a delegation" — usually not a bug

1. Taps during speech are ignored by design — a long Hermes answer takes
   ~30-40 s to read out; taps log `[CLICK] Ignored in state speaking`. Wait for
   `[TTS] aplay finished.` + `[EARS] Waiting for wake word...`.
2. Triple-tap (5 taps <0.8 s) = quit — rapid taps exit BMO and look like a
   crash.
3. Check `/tmp/bmo.log` `[CLICK]` lines before touching audio code.