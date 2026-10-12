"""Event log (design §5.7): label changes, RF rejections, red-team actions, RF DENIED, ghosts.

Every event goes to logs/<timestamp>.jsonl (for tuning thresholds afterwards) and to an
in-memory ticker for the display. A vehicle newly turning SUSPICIOUS beeps once.
"""
import json
import platform
import time
from collections import deque
from pathlib import Path


def beep():
    if platform.system() == "Windows":
        import winsound

        winsound.Beep(1200, 120)
    else:
        print("\a", end="", flush=True)


class EventLog:
    def __init__(self, log_dir="logs", sound=True, write=True):
        self.ticker = deque(maxlen=6)
        self.prev = {}         # track_id → (state, reason, rf_id)
        self.since = {}        # track_id → t the current state started
        self.prev_denied = False
        self.prev_ghosts = set()
        self.sound = sound
        self.file = None
        if write:
            Path(log_dir).mkdir(exist_ok=True)
            self.path = Path(log_dir) / time.strftime("%Y%m%d-%H%M%S.jsonl")
            self.file = open(self.path, "a")

    def add(self, t, kind, text, **data):
        self.ticker.append((t, kind, text))
        if self.file:
            self.file.write(json.dumps({"t": round(t, 3), "kind": kind, "text": text, **data}) + "\n")
            self.file.flush()

    def update(self, t, out, rejections=()):
        """Diff the matcher output against the previous frame and log what changed."""
        for r in rejections:
            self.add(t, "rf_rejected", f"RF rejected: {r['reason'].replace('_', ' ')}", reason=r["reason"])
        for tid, lab in out["labels"].items():
            cur = (lab["state"], lab["reason"], lab["rf_id"])
            if self.prev.get(tid) != cur:
                if self.prev.get(tid, (None,))[0] != lab["state"]:
                    self.since[tid] = t
                name = f"#{tid}" + (f" {lab['rf_id']}" if lab["rf_id"] else "")
                what = lab["state"] + (f" {lab['reason'].replace('_', ' ')}" if lab["reason"] else "")
                self.add(t, "label", f"{name}: {what}", track_id=tid, state=lab["state"],
                         reason=lab["reason"], rf_id=lab["rf_id"], confidence=round(lab["confidence"], 3),
                         scores={k: round(v, 3) for k, v in lab["scores"].items()})
                if lab["state"] == "SUSPICIOUS" and self.prev.get(tid, (None,))[0] != "SUSPICIOUS" and self.sound:
                    beep()
                self.prev[tid] = cur
        for tid in [k for k in self.prev if k not in out["labels"]]:
            del self.prev[tid]
            self.since.pop(tid, None)
        if out["rf_denied"] != self.prev_denied:
            self.add(t, "rf_denied", "RF DENIED" if out["rf_denied"] else "RF restored", on=out["rf_denied"])
            self.prev_denied = out["rf_denied"]
        ghosts = {g["rf_id"] for g in out["ghosts"]}
        for rid in ghosts - self.prev_ghosts:
            self.add(t, "ghost", f"GHOST {rid}: claim with no vehicle", rf_id=rid)
        self.prev_ghosts = ghosts

    def time_in_state(self, tid, t):
        return t - self.since.get(tid, t)

    def close(self):
        if self.file:
            self.file.close()
