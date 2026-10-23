---
name: be-more-hailo
description: Use for the BMO robot (be-more-hailo) on the Hailo Pi.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [bmo, hailo, raspberry-pi, voice-assistant, llm, robotics]
    related_skills: [hailo10h-pi5-setup]
---

# BMO robot (be-more-hailo)

Application layer for the BMO voice robot: `agent_hailo.py`, `core/llm.py`, the face rig, TTS, wake word, briefings. The NPU/HailoRT *driver* layer lives in the `hailo10h-pi5-setup` skill — keep driver repair and app work separate.

## When to Use

Load when deploying to, restarting, or debugging the BMO robot — especially when it stops answering, hits a fallback phrase ("tape deck got jammed"), or the wake word / voice changes. Skip it for pure NPU-driver / HailoRT problems (those go to `hailo10h-pi5-setup`).

Project layout: desktop fork at `~/Documents/git/be-more-hailo` (origin `sanchorelaxo/be-more-hailo`, upstream `moorew/be-more-hailo`); deployed app at `sanchobot@raspberrypi:~/be-more-agent`. The Pi has no GitHub creds — deploy via scp/rsync, not `git pull`.

## Deploy + test loop

0. Local unit tests run in a dedicated venv (the repo has none on the desktop): `~/.venvs/bmo-test/bin/python -m pytest tests/unit` — it needs `pytest requests python-dotenv numpy ddgs Pillow soundfile fastapi httpx` (import-time deps of `core.llm` / `core.stt` / `web_app`). Keep it outside the repo so it never gets rsynced.
1. rsync desktop → Pi, EXCLUDING device state and heavy deps: `memory.json`, `settings.json`, `pronunciations.json`, `voices.json`, `venv/`, `models/`, `piper/`, `whisper.cpp/`, logs. Overwriting device state clobbers the live robot's memory/voice config.
2. Run ONLY `tests/unit/` on the Pi venv (`./venv/bin/python -m pytest tests/unit/`). Never run `test_hailo*.py`, `test_wakeword.py`, or `tests/integration/` — they need the mic/NPU/face and are flaky-over-SSH noise. Launch it as a background SSH job with completion notification and write junitxml, then parse the XML for the counts: a full ~800-test run can exhaust the Pi's memory, after which `systemctl restart` fails with "unavailable resources" and sshd starts resetting connections (ping still works, so it looks like a network fault). Let it settle or reboot before restarting services — and do the live verification before the suite, not after.
3. Smoke-test live paths (weather, Hermes delegation, an open LLM prompt) after deploy before declaring done.

## Restart the agent (systemd: bmo-voice.service)

The voice agent (`agent_hailo.py`) runs under `bmo-voice.service` — a systemd unit installed by `setup_services.sh`, alongside `bmo-ollama.service` and `bmo-web.service`. It starts on boot and restarts on crash.

Restart: `sudo systemctl restart bmo-voice.service`
Logs:   `journalctl -u bmo-voice -f`
Status: `systemctl status bmo-voice.service`

The unit sets `DISPLAY=:0 XAUTHORITY=... XDG_RUNTIME_DIR=/run/user/1000 PYTHONUNBUFFERED=1` and runs `venv/bin/python agent_hailo.py` directly — skipping `start_agent.sh`'s per-boot `pip install` and its redundant hailo-ollama auto-start (bmo-ollama.service owns port 8000). `Restart=always` + `RestartSec=5` cover startup races (display :0 not yet up, mic not yet enumerated). The mic failure itself is self-healing inside the agent (`wait_for_wakeword` retries); a full process restart is the fallback that re-runs `find_audio_devices()` if the device index shifted.

Manual fallback (debugging only — NOT for normal restarts):

    ssh sanchobot@raspberrypi 'cd ~/be-more-agent && \
      export DISPLAY=:0 XAUTHORITY=/home/sanchobot/.Xauthority XDG_RUNTIME_DIR=/run/user/1000 PYTHONUNBUFFERED=1 && \
      nohup ./venv/bin/python agent_hailo.py > /tmp/agent_hailo.log 2>&1 &'

The SSH command will not return cleanly (the detached child holds the channel) — expected, the timeout is harmless. Verify with `ps aux | grep agent_hailo.py` and a log tail showing `[STATE] IDLE: Tap to speak` (or watch `journalctl -u bmo-voice -f`).

## Failure mode: "tape deck got jammed" (LLM history poisoning)

Symptom: BMO answers every prompt with only an action object (`{"action":"set_expression","value":"happy"}`) and zero spoken text, then hits the "tape deck got jammed" fallback in `agent_hailo.py`.

Root cause is history poisoning, NOT a broken backend. `_with_current_context` in `core/llm.py` trims conversation history to the system prompt + the last 2 messages; if the trailing turns are action-only (JSON, no speech), the small model (qwen3:1.7b) mimics them — emitting a bare action and nothing speakable, which loops forever.

Fix (already landed): `_history_safe()` peels balanced JSON action objects off each persisted assistant turn and substitutes `"Okay friend!"` when no alphanumeric speech remains. The action still executes (face still updates); only the stored history copy changes, breaking the mimic loop. Applied at all three history-append sites in `core/llm.py` (stream_think main, stream_think finally-salvage, think). Regression tests in `tests/unit/test_history_safe.py`.

After the code fix, also clean the LIVE `memory.json`: apply `_history_safe` to the existing poisoned assistant turns (or the next turn still mimics stale data). The running agent only reads `memory.json` on start and rewrites on save — clean it after killing the process, before restarting.

Diagnostic pitfall: verify any LLM-path fix with an OPEN prompt (e.g. "Tell me a fun fact about dogs"), never an arithmetic question — `2+2` is intercepted by the `_instant_reply` math route and never reaches the model.

## Wake word invariant

The working wake word is "hey bmo" (user-confirmed empirically — it wakes the agent). Do NOT trust the model filename: `install.sh` (~line 325) downloads the stock openWakeWord `hey_jarvis_v0.1.onnx` and renames it to `wakeword.onnx`, and its progress echo prints "Hey BMO". Despite the `hey_jarvis` URL and `wakeword` filename, the phrase that actually fires is "hey bmo". When debugging "BMO won't wake", verify against the phrase the user actually says, not the model filename.

## Editing BMO's identity / backstory

BMO's "who are you" answer is hardcoded in several places, not just the system prompt. When renaming a companion or relocating the setting (e.g. "Finn and Jake" -> "Sancho", "the Tree Fort" -> "an Airstream"), grep for the term in every one of these, on BOTH the desktop repo and the Pi:

- `core/config.py` — the persona appears TWICE: `get_system_prompt()` (active, short) and `_legacy_system_prompt()` (superseded, commented "kept for reference"). Change both.
- `memory.json` (Pi-only, gitignored runtime memory) — the system prompt AND earlier self-description turns ("I'm BMO ... who lives in ...") are persisted here and re-sent every session. Change config.py but not memory.json and the old phrasing still resurfaces from memory.
- `agent_hailo.py` and `web_app.py` — duplicated `fallback_phrases` lists ("I wonder what <name> is doing right now").
- `scripts/generate_sounds.py` — pre-generated greeting/thinking TTS lines.
- `core/quick_answers.py` — instant pre-LLM answers (time/date/math/units); one canned character string ("Even BMO can't divide by zero!") lives at its divide-by-zero branch.

Rename/replace workflow:
1. Word-boundary, case-insensitive grep on both machines; exclude `voices.json` (contains the language name "Finnish") and `venv/ models/ whisper.cpp/ piper/` dirs so you don't over- or under-match.
2. Edit the tracked files on the desktop repo, then `scp` them to the Pi; `sed` `memory.json` directly on the Pi (it has no desktop counterpart).
3. Fix verb agreement when replacing a plural with a singular: "Finn and Jake are doing" -> "Sancho is doing".
4. Restart `bmo-voice.service` AND `bmo-web.service` (both carry the prompt/fallback text).
5. Commit the desktop repo; the `memory.json` edit lives only on the Pi (gitignored).

## Interacting with BMO

Three interfaces, all on the Pi:

- **Voice + tap (primary):** tap the body button to enter listening -> say "hey bmo" -> LLM -> Piper TTS reply. Runs under `bmo-voice.service` (DISPLAY=:0, local USB mic/speaker). Logs/status under "Restart the agent" above.
- **Web UI:** `bmo-web.service` serves the agent at http://<pi>:8080 (uvicorn `web_app:app`). Use when headless over Tailscale/SSH tunnel; the Tkinter face needs DISPLAY=:0.
- **Raw LLM:** `curl -H 'Content-Type: application/json' 127.0.0.1:8000/api/chat -d '{"model":"qwen3:1.7b","messages":[{"role":"user","content":"hi"}]}'` talks to hailo-ollama directly, bypassing BMO's voice/TTS stack.

## Configuring BMO

Config lives in `core/config.py` (module constants + a few env-overridable). USB mic/speaker are auto-detected at startup via sounddevice — plug in USB audio and restart; no config edit needed.

Env-overridable (set before the service starts):

- `HERMES_DELEGATE_MODEL` = `deepseek/deepseek-v4-flash-0731` (Hermes catalog model for delegation)
- `HERMES_DELEGATE_TIMEOUT` = `180` (seconds)
- `HERMES_DELEGATE_SUCCINCT` = "Be ultra-succinct…" (suffix appended to delegated prompts)
- `HERMES_DELEGATE_HISTORY_MESSAGES` = `6` / `HERMES_DELEGATE_HISTORY_CHARS` = `2000` / `..._MSG_CHARS` = `500` (conversation window sent with each delegated turn)
- `BMO_NPU_STT` = `1` to run STT on the NPU (off by default — NPU is single-tenant, the LLM owns it)
- `VLM_HEF_PATH` = `models/Qwen3-VL-2B-Instruct.hef`

Fixed constants (edit code, not env):

- `LLM_URL` = `http://127.0.0.1:8000/api/chat`, `LLM_MODEL` = `qwen3:1.7b` (native Hailo)
- `VOLUME` = `0.75` (software gain), `MIC_SAMPLE_RATE` = `48000`, `WAKE_WORD_THRESHOLD` = `0.35`
- TTS = Piper (`piper/piper` + `bmo.onnx`); STT = whisper.cpp (`whisper-cli`, CPU) by default

Most knobs are hardcoded, not env-var driven — a key limitation when scripting config changes.

## Hermes delegation

BMO hands richer work to the Pi's own Hermes Agent (deepseek + full tool access) by shelling out to `hermes -z <prompt> -m <model>`. Three triggers, all in `core/llm.py`:

1. **Keyword:** "use hermes <ask>" / "ask hermes …" / "hermes, …" / "hermes: …" (`HERMES_TRIGGERS`).
2. **Skill:** "use <x> skill [to] <ask>" — no need to say "hermes". Delegates AND preloads the closest installed skill (`hermes -s <skill>`), fuzzy-matched so transcription typos still resolve. Resolver: `_extract_skill_request()` / `_closest_skill()` / `hermes_skill_names()`; no local match still delegates, just without `-s`.
3. **Don't-know auto-trigger:** when local qwen3 shrugs ("I don't know", "I can't answer", "I'm not sure", "I have no idea"), BMO emits a preface like "BMO came up with nothing — let me ask my big friend Hermes!" and delegates the follow-up automatically instead of giving up.

**Conversation memory:** each delegation is a cold `hermes -z` one-shot, so `delegate_to_hermes(prompt, skill=None, history=None)` renders BMO's own recent turns (last 6 messages / 2000 chars, system prompt excluded, current ask de-duplicated, ASCII-only) into the prompt via `_delegation_context()`. All four call sites pass `history=self.history`. Without this, "repeat the last answer" is answered from Hermes' stale session — it once returned a weeks-old answer.

The newest BMO line is labelled `BMO (most recent answer):` and the prompt orders Hermes to echo that line for repeat/recall asks. Without the label, a recall after two delegations was observed returning the EARLIER answer (op-1 tape-engine summary instead of the gorilla-screen answer that had just been spoken). `[Now: ...]` timestamps are stripped from the transcript. Recall must be tested with ALTERNATING delegated and local turns — the last thing spoken may come from local qwen3, not Hermes.

**Verifying recall live** — `scripts/live_recall_probe.py` runs real turns through the deployed `Brain` (no hardware, no service restart) and reports whether the recall turn reproduced the PREVIOUS answer. Rules:

- Ask what the user would actually ask. Invented synthetic prompts ("write a short gorillas themed prompt") are nonsensical to the user and exercise a different path than the real request ("explain seeing gorillas on the device screen").
- Alternate delegated and local turns — the answer to repeat may come from local qwen3, not Hermes.
- Compare with difflib similarity (>=0.90 vs the previous answer, <0.50 vs earlier ones), never exact text or expected content: Hermes rewords, and it can answer the same question differently across runs (the OP-1 gorilla-screen question returned the hidden-easter-egg answer once and the Finger-sequencer answer once).
- If a recall bug will not reproduce, do not chase model non-determinism: instrument the child prompt (monkeypatch `delegate_to_hermes` to print `_delegation_context(history, current=prompt)`) and remove the ambiguity from the prompt instead.

Delegation blocks until Hermes finishes, then speaks the answer in-character. It never crashes offline — every failure maps to a graceful BMO line. Full reference (CLI flags, failure table, trigger regexes, pitfalls): [references/hermes-delegation.md](mdc:references/hermes-delegation.md)

## Syncing skills to the Pi's Hermes

BMO delegates to the Pi's OWN Hermes instance (`sanchobot@raspberrypi:~/.hermes/skills/`, same layout as this desktop) — not the desktop Hermes. That instance must carry `be-more-hailo` and `hailo10h-pi5-setup` so BMO's local Hermes can configure/troubleshoot BMO itself. Whenever either skill changes on the desktop, copy it over and confirm byte-identity:

    scp -r ~/.hermes/skills/devops/<name> sanchobot@raspberrypi:~/.hermes/skills/devops/
    md5sum ~/.hermes/skills/devops/<name>/SKILL.md   # run on both ends; hashes must match

`scp -r` on the skill dir preserves structure (unlike `scp a b host:dir/`, which flattens loose files into `dir/`). Verify hashes, not just the scp exit code. The Pi's running Hermes won't see a new/updated skill until a new session starts.

### The music/instruments pack (15 instrument skills)

Source of truth on the desktop: `~/.hermes/skills/music/hermes-music-sampling/skills/instruments/`
(git repo `~/Documents/git/hermes-music-sampling`, symlinked into the skills dir).
Upstream repo: https://github.com/sanchorelaxo/hermes-music-sampling (public) —
clone with `git clone https://github.com/sanchorelaxo/hermes-music-sampling.git`
if the pack is missing, then re-sync below.
On the Pi it is installed at the SAME relative path:

    ~/.hermes/skills/music/hermes-music-sampling/skills/instruments/

Mirror that path (not `music/instruments/`) so the skills' internal relative
references keep resolving. Sync with rsync, never `--delete`, and skip pycache:

    SRC=~/.hermes/skills/music/hermes-music-sampling/skills/instruments
    ssh sanchobot@100.119.186.62 'mkdir -p ~/.hermes/skills/music/hermes-music-sampling/skills'
    rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store \
      "$SRC/" sanchobot@100.119.186.62:'~/.hermes/skills/music/hermes-music-sampling/skills/instruments/'

Then prove it: 115 files / 6.5 MB, `find . -type f -exec md5sum {} +` identical on
both ends, and `hermes_skill_names()` on the Pi grows from 60 to 81 names.
Deployment is covered by `tests/unit/test_instrument_skills.py` (spotykach is the
probe): `BMO_REQUIRE_INSTRUMENT_SKILLS=1 ./venv/bin/python -m pytest tests/unit/test_instrument_skills.py`
makes a missing/half-copied pack FAIL instead of skip — run that on the Pi after
every sync. `HERMES_SKILLS_DIR` is read per delegation, so no service restart is
needed for skill-pack changes (only for `core/llm.py` edits).

## Other known invariants

- NPU is single-tenant: unload the active model before loading another (driver mechanics in `hailo10h-pi5-setup`).
- hailo-ollama (the NPU LLM backend) errors on non-ASCII input; `_sanitize_messages` in `core/llm.py` strips it before the call.
- LLM backend is qwen3:1.7b at `http://127.0.0.1:8000/api/chat` (port 8000, NOT the usual 11434).
