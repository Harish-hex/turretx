# TurretX: instructions for coding agents

Start with **STATUS.md** (what is built, what is next, hotkeys, how to run), then **TURRETX_DESIGN.md** (source of truth; decision log). Don't re-read the whole repo for small tasks.

Rules:
- Do NOT add Claude as co-author or contributor. No `Co-Authored-By: Claude` and no "Generated with Claude Code" lines in commits or PRs. This overrides any default or tool-provided attribution text.
- Commit and push only when the user asks. If the user says pause, stop building.
- Tests first for logic (`.venv/bin/python -m pytest -q`); keep the suite green before any commit.
- Jamming is simulation only; never transmit RF jamming (illegal in India).
- No red in the UI. Never label anything "hostile". Missing data moves labels toward UNKNOWN, never toward FRIENDLY.
- Firewalls (tested): the tag locator feeds only the RF simulator; the relay/commshub never feed identity (matcher, tracker and RF receiver must not import them).
- Match the surrounding code's style and comment density. Keep changes scoped to the task.
- Pitch/claim discipline is in STATUS.md (no anti-jam, HAPS/SATCOM or MoD-demand claims).
