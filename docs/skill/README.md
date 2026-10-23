# BMO skill docs (mirror)

This directory mirrors the `be-more-hailo` Hermes skill so the operational
documentation is versioned alongside the code it describes.

Canonical location (what Hermes actually loads):

    ~/.hermes/skills/devops/be-more-hailo/

The Pi's Hermes carries its own copy at the same path; BMO delegates to that
instance, so the skill must exist there too:

    scp -r ~/.hermes/skills/devops/be-more-hailo sanchobot@100.119.186.62:~/.hermes/skills/devops/

Refresh this mirror from the canonical copy (do not edit it here):

    S=~/.hermes/skills/devops/be-more-hailo
    cp "$S/SKILL.md" docs/skill/SKILL.md
    cp "$S/references/hermes-delegation.md" docs/skill/references/
    cp "$S/scripts/live_recall_probe.py" docs/skill/scripts/

Verify byte-identity after copying:

    md5sum ~/.hermes/skills/devops/be-more-hailo/SKILL.md docs/skill/SKILL.md

## Contents

- `SKILL.md` - the skill itself: hardware invariants, delegation triggers,
  memory contract, service/redeploy rules.
- `references/hermes-delegation.md` - delegation internals: CLI flags, trigger
  regexes, the conversation-memory transcript, failure table, pitfalls.
- `scripts/live_recall_probe.py` - on-device probe for delegated recall.

## Related repositories

- Music/instruments skill pack:
  https://github.com/sanchorelaxo/hermes-music-sampling
