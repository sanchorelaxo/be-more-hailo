#!/usr/bin/env python3
"""Live multi-turn recall probe for BMO's delegated Hermes turns.

Runs real turns through the deployed core.llm.Brain (no hardware, no services
restart needed) and reports whether a recall/"repeat" turn reproduces the
IMMEDIATELY PRECEDING answer rather than an earlier one. Hermes rewords the
same question across runs, so matching is by difflib similarity, never by
exact text.

Run ON THE PI with the deployed venv:

    ssh sanchobot@100.119.186.62 \
      'cd ~/be-more-agent && ./venv/bin/python \
         ~/.hermes/skills/devops/be-more-hailo/scripts/live_recall_probe.py \
         "use the op-1 skill to summarise the tape engine" \
         "use the op-1 skill to explain seeing gorillas on the device screen" \
         "use the op-1 skill to repeat the last answer"'

The LAST ask is treated as the recall turn; every earlier ask is a context
turn. Mix delegated (`use ... skill` / `use hermes ...`) and local turns
(e.g. "tell me a fun fact about dogs") to cover the case where the answer to
repeat came from local qwen3, not Hermes.

Thresholds: >= 0.90 vs the previous answer = pass, < 0.50 vs any earlier
answer = correctly NOT repeated. Anything in between is ambiguous; re-run
before calling it a bug (Hermes is non-deterministic).
"""

import difflib
import os
import sys

REPO = os.environ.get("BMO_REPO", "/home/sanchobot/be-more-agent")
sys.path.insert(0, REPO)

import core.llm as llm  # noqa: E402


def norm(text: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() or c.isspace() else " "
                            for c in text or "").split())


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def main(asks: list) -> int:
    if len(asks) < 2:
        print("need at least one context ask and a recall ask")
        return 2

    brain = llm.Brain.__new__(llm.Brain)
    brain.persist = False                      # never touch memory.json
    brain.history = [{"role": "system", "content": "You are BMO."}]
    brain.briefing_ready = None
    brain.reminders = None
    brain.home = None

    answers = []
    for i, ask in enumerate(asks, 1):
        out = "".join(brain.stream_think(ask))
        answers.append(out)
        print("T%d ASK  : %s" % (i, ask))
        print("T%d ANSWER: %s" % (i, (out or "")[:200].replace("\n", " ")))

    recall = answers[-1]
    previous = answers[-2]
    prev_score = ratio(recall, previous)
    print("\nsimilarity(recall, previous answer) = %.3f" % prev_score)
    earlier = [(i + 1, ratio(recall, a))
               for i, a in enumerate(answers[:-2])]
    worst = 0.0
    for turn, score in earlier:
        print("similarity(recall, T%d answer)      = %.3f" % (turn, score))
        worst = max(worst, score)

    ok = prev_score >= 0.9 and worst < 0.5
    print("VERDICT:", "PASS" if ok else "INVESTIGATE")
    if not ok:
        print("\nDelegated context the model actually saw for the recall turn:")
        print(llm._delegation_context(brain.history, current=asks[-1]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
