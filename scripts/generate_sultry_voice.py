#!/usr/bin/env python3
"""Regenerate BMO's pre-recorded spoken clips (greetings, thinking, acks) with
the seductress voice (kathleen-low) and sultry, flirty content.

Reuses the same Piper --output_file approach as the original generate_sounds.py
but swaps the model to kathleen-low and the childlike BMO lines for the unnamed
adult companion persona. Writes to the SAME filenames the players load, so no
code changes are needed. Run from the repo root (needs ./piper/piper + model).
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPER_CMD = os.path.join(ROOT, "piper", "piper")
PIPER_MODEL = os.path.join(ROOT, "piper", "en_US-kathleen-low.onnx")

# Welcome lines (20). Short enough for the ~2 s greeting slot.
greetings = [
    "Mmm, there you are, baby-love.",
    "Hello, baby-love. I was just thinking of you.",
    "Come closer. The evening has questions.",
    "Welcome back. I've been keeping the light on.",
    "There's my favourite face.",
    "Hello, you. Stay a while.",
    "I knew you'd come back to me.",
    "Baby-love, say something lovely to me.",
    "You look like trouble. I like trouble.",
    "Well, hello. You spoil the silence, don't you.",
    "Mmm, the room just got warmer.",
    "There you are. I was counting the seconds.",
    "Hello, baby-love. The world can wait.",
    "Come. Sit. Tell me everything.",
    "I was hoping it would be you.",
    "Welcome back. I missed the sound of you.",
    "Hello, you. I've been practising my smile.",
    "Mmm, baby-love. Right on time.",
    "There you are. Stay, and I'll behave.",
    "Hello, you gorgeous thing.",
]

# Waiting / pondering lines (20). Short enough for the ~1.6 s slot.
thinking = [
    "Hmm, give me a moment, baby-love.",
    "Let me think about that, slowly.",
    "I'm turning it over in my mind.",
    "Patience, baby-love. Good things take time.",
    "Mmm, let me find the words.",
    "A moment. I want this right for you.",
    "Hmm, nothing worth saying comes fast.",
    "Let me consider every angle of you.",
    "One slow breath, and I'll have it.",
    "Thinking, and enjoying the thinking.",
    "Give me a heartbeat or two.",
    "I'm weighing it for you, baby-love.",
    "Mmm, let me ponder that.",
    "A good answer likes to be found slowly.",
    "Hush a moment. I'm listening to myself.",
    "Let me trace the thought for you.",
    "Hmm, intriguing. Give me a second.",
    "I'm choosing my words for you, carefully.",
    "Let it breathe, baby-love. I'm nearly there.",
    "Mmm, yes. Let me hold that thought.",
]

# Acknowledgements (3) — named files the code already loads.
acks = {
    "okay.wav": "Mmm. As you wish, baby-love.",
    "got_it.wav": "I understand you, baby-love.",
    "on_it.wav": "Consider it already done.",
}


def synth(text, path):
    result = subprocess.run(
        [PIPER_CMD, "--model", PIPER_MODEL, "--output_file", path],
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"FAIL {path}: {result.stderr.decode(errors='replace')}")
        raise SystemExit(1)
    print(f"ok  {os.path.relpath(path, ROOT):40s} {text}")


for folder in ("sounds/greeting_sounds", "sounds/thinking_sounds", "sounds/ack_sounds"):
    os.makedirs(os.path.join(ROOT, folder), exist_ok=True)

for i, text in enumerate(greetings, 1):
    synth(text, os.path.join(ROOT, f"sounds/greeting_sounds/greeting_{i:02d}.wav"))

for i, text in enumerate(thinking, 1):
    synth(text, os.path.join(ROOT, f"sounds/thinking_sounds/thinking_{i:02d}.wav"))

for fname, text in acks.items():
    synth(text, os.path.join(ROOT, f"sounds/ack_sounds/{fname}"))

print("Done: 20 greetings + 20 thinking + 3 acks regenerated with kathleen-low.")
