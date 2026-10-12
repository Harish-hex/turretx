"""End-to-end regression: synthetic camera → tracker → sticker GPS → RF sim → receiver → matcher.

Locks in the Phase 3 end-to-end findings: through crossings the real friendly must never be
labelled SUSPICIOUS (it was, before velocity resets and merged-blob handling), a clone flags
both vehicles, and jamming coasts the friendly until custody breaks, then raises RF DENIED.
"""
import numpy as np
import pytest

from turretx import synthetic, wire
from turretx.config import load_config
from turretx.homography import Homography
from turretx.matcher import Matcher
from turretx.rfrx import Interrogator, Receiver
from turretx.rfsim import RFSim
from turretx.taglocator import TagLocator
from turretx.tracker import Tracker


def run(actions=None, seconds=10.0):
    cfg = load_config()
    cfg["table_cm"] = list(synthetic.TABLE_CM)
    cfg["detector"].update(backend="diff", ground_point_frac=0.5)
    homo = Homography(synthetic.TABLE_CM)
    assert homo.calibrate(synthetic.render(0))
    tracker = Tracker(cfg, homo)
    tracker.backend.set_background(synthetic.render(0, cars=False))
    tags, sim = TagLocator(cfg["tags"], homo), RFSim(cfg, seed=1)
    inter = Interrogator(cfg["rf"]["challenge_hz"])
    rx, matcher = Receiver(wire.rf_key(cfg["rf"]["key"]), inter), Matcher(cfg)
    actions = dict(actions or {})
    rows = []  # (t, car name, state, reason, rf_denied)
    for i in range(int(seconds * synthetic.FPS)):
        t = i / synthetic.FPS
        frame = synthetic.render(t)
        tracks, fixes = tracker.update(frame, t), tags.update(frame, t)
        sim.observe(t, fixes, tracks)
        for at, action in list(actions.items()):
            if t >= at:
                sim.red_team(action, t)
                del actions[at]
        ch = inter.poll(t)
        if ch:
            sim.on_challenge(ch, t)
        claims, _ = rx.receive(sim.poll(t), t)
        out = matcher.update(t, tracks, claims)
        truth = synthetic.car_states(t)
        for k in tracks:
            name = min(truth, key=lambda n: np.hypot(truth[n][0] - k["x"], truth[n][1] - k["y"]))
            lab = out["labels"][k["track_id"]]
            rows.append((t, name, lab["state"], lab["reason"], out["rf_denied"]))
    return rows


@pytest.fixture(scope="module")
def normal():
    return run()


def test_friendly_identified_quickly(normal):
    first = min(t for t, n, s, _, _ in normal if n == "F-1" and s == "FRIENDLY")
    assert first <= 1.6


def test_real_friendly_never_suspicious_through_crossings(normal):
    assert not [t for t, n, s, _, _ in normal if n == "F-1" and s == "SUSPICIOUS"]


def test_unknown_vehicles_never_friendly_or_suspicious(normal):
    assert not [(t, n, s) for t, n, s, _, _ in normal if n != "F-1" and s != "UNKNOWN"]


def test_friendly_reacquired_after_first_crossing(normal):
    assert any(n == "F-1" and s == "FRIENDLY" for t, n, s, _, _ in normal if 4.5 < t < 5.6)


def test_clone_flags_both_vehicles():
    rows = run({5.0: "clone"}, seconds=8.0)
    window = [(n, s, r) for t, n, s, r, _ in rows if 5.3 < t < 8.0]
    assert all((s, r) == ("SUSPICIOUS", "ID_CONFLICT") for n, s, r in window if n in ("F-1", "U-2"))
    assert all(s == "UNKNOWN" for n, s, r in window if n == "U-3")


def test_jam_coasts_then_custody_breaks_then_rf_denied():
    rows = run({2.0: "jam"}, seconds=6.0)
    f1 = {round(t, 2): (s, r) for t, n, s, r, _ in rows if n == "F-1"}
    assert f1[2.5] == ("FRIENDLY", "COASTING")           # silent, still in custody
    later = [sr for t, sr in f1.items() if t > 3.0]
    assert later and all(s != "FRIENDLY" for s, _ in later)  # custody broke in the crossing
    assert not any(s == "SUSPICIOUS" for s, _ in f1.values())
    denied = [t for t, n, s, r, d in rows if d]
    assert denied and 3.9 <= min(denied) <= 4.2           # ~2 s after the last claim
