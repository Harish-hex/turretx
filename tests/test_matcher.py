"""Matcher tests: example scenarios for every label + metamorphic relations (no oracle needed).

Scenario helper: vehicles move along known paths (tracks every frame); emitters send radio
claims at 5 Hz with 100 ms latency, like rfsim → rfrx. Two emitters with the same rf_id in
one round are flagged conflict, like the receiver does.
"""
import copy
import math
import random

import pytest

from turretx.config import load_config
from turretx.matcher import Matcher
from turretx.rfmodel import expected_rssi

ARENA = (200, 150)


def make_cfg(r_weight=None):
    cfg = load_config()
    cfg["table_cm"] = list(ARENA)
    if r_weight is not None:  # geometric relations: RSSI depends on absolute position
        w = cfg["matcher"]["weights"]
        w["R"] = r_weight
        w["P"] = w["M"] = (1 - r_weight) / 2
    return cfg


def still(x, y):
    return lambda t: (x, y)


def line(x0, y0, vx, vy):
    return lambda t: (x0 + vx * t, y0 + vy * t)


def vel(path, t, dt=1e-3):
    (x0, y0), (x1, y1) = path(t - dt), path(t + dt)
    return (x1 - x0) / (2 * dt), (y1 - y0) / (2 * dt)


def simulate(cfg, seconds, vehicles, emitters, fps=30, latency=0.1, noise_cm=0.0, seed=0,
             silence=(), transform=None, start_t=0.0, track_ids=None, record=None):
    """vehicles: {track_id: path}; emitters: [(rf_id, path)]; silence: [(t0, t1)] radio off.
    transform: f(x, y, vx, vy) -> (x, y, vx, vy) applied to everything (metamorphic tests).
    record: optional list that receives every (t, tracks, claims) fed to the matcher.
    Returns [(t, output)] for every frame."""
    rng = random.Random(seed)
    m = Matcher(cfg)
    xf = transform or (lambda x, y, vx, vy: (x, y, vx, vy))
    rf = cfg["rf"]
    in_flight, outs, next_round, rnd = [], [], start_t, 0
    n = int(round(seconds * fps))
    for i in range(n + 1):
        t = start_t + i / fps
        while t >= next_round - 1e-9:
            tc = next_round
            if not any(a <= tc < b for a, b in silence):
                counts = {}
                for rf_id, _ in emitters:
                    counts[rf_id] = counts.get(rf_id, 0) + 1
                for rf_id, path in emitters:
                    x, y = path(tc)
                    vx, vy = vel(path, tc)
                    x, y = x + rng.gauss(0, noise_cm), y + rng.gauss(0, noise_cm)
                    x, y, vx, vy = xf(x, y, vx, vy)
                    in_flight.append((tc + latency, {
                        "rf_id": rf_id, "nonce": f"n{rnd}", "t": tc, "x": x, "y": y, "vx": vx, "vy": vy,
                        "rssi": expected_rssi(x, y, ARENA, cfg["scale_m_per_cm"], rf["rssi"]),
                        "conflict": counts[rf_id] > 1}))
            next_round += 1 / rf["challenge_hz"]
            rnd += 1
        due = [c for dt, c in in_flight if dt <= t + 1e-9]
        in_flight = [(dt, c) for dt, c in in_flight if dt > t + 1e-9]
        tracks = []
        for tid, path in vehicles.items():
            p = path(t)
            if p is None:
                continue
            x, y, vx, vy = xf(*p, *vel(path, t))
            tid = track_ids.get(tid, tid) if track_ids else tid
            tracks.append({"track_id": tid, "t": t, "x": x, "y": y, "vx": vx, "vy": vy,
                           "speed": math.hypot(vx, vy), "heading": math.atan2(vy, vx)})
        if record is not None:
            record.append((t, copy.deepcopy(tracks), copy.deepcopy(due)))
        outs.append((t, m.update(t, tracks, due)))
    return outs


def at(outs, t):
    """Output of the frame closest to time t."""
    return min(outs, key=lambda o: abs(o[0] - t))[1]


def state(out, tid):
    lab = out["labels"].get(tid)
    return (lab["state"], lab["reason"]) if lab else None


def final(outs, tid):
    return state(outs[-1][1], tid)


def first_time(outs, tid, st):
    return next((t for t, o in outs if state(o, tid) and state(o, tid)[0] == st), None)


# --- example scenarios ---------------------------------------------------------------------

def test_matched_friendly_becomes_friendly_after_hysteresis():
    cfg = make_cfg()
    car = line(40, 50, 15, 5)
    outs = simulate(cfg, 4.0, {1: car}, [("F-1", car)], noise_cm=1.5)
    t_f = first_time(outs, 1, "FRIENDLY")
    assert t_f is not None and 1.0 <= t_f <= 2.2   # needs ≥1 s of agreement, not instant
    assert final(outs, 1) == ("FRIENDLY", None)
    lab = outs[-1][1]["labels"][1]
    assert lab["rf_id"] == "F-1" and lab["confidence"] >= 0.7
    assert set(lab["scores"]) == {"P", "M", "R"}


def test_vehicle_without_claim_is_unknown_no_id():
    cfg = make_cfg()
    outs = simulate(cfg, 3.0, {1: line(40, 50, 15, 0), 2: still(150, 100)}, [("F-1", line(40, 50, 15, 0))])
    assert final(outs, 2) == ("UNKNOWN", "NO_ID")
    assert all(state(o, 2)[0] == "UNKNOWN" for _, o in outs)


def test_two_vehicles_side_by_side_is_too_close_not_a_guess():
    cfg = make_cfg()
    a, b = line(60, 70, 12, 0), line(60, 76, 12, 0)  # 6 cm apart, same motion
    claim = line(60, 73, 12, 0)                      # claim right between them
    outs = simulate(cfg, 3.0, {1: a, 2: b}, [("F-1", claim)])
    assert final(outs, 1) == ("UNKNOWN", "TOO_CLOSE") or final(outs, 2) == ("UNKNOWN", "TOO_CLOSE")
    assert "FRIENDLY" not in {final(outs, 1)[0], final(outs, 2)[0]}


def test_clone_flags_both_vehicles_id_conflict_immediately():
    cfg = make_cfg()
    real, other = line(40, 40, 12, 0), line(150, 110, -10, 0)
    outs = simulate(cfg, 3.0, {1: real, 2: other}, [("F-1", real), ("F-1", other)])
    first_claim_t = 0.1
    t1 = first_time(outs, 1, "SUSPICIOUS")
    assert t1 is not None and t1 <= first_claim_t + 0.05     # no 1 s wait for a security alert
    assert final(outs, 1) == ("SUSPICIOUS", "ID_CONFLICT")
    assert final(outs, 2) == ("SUSPICIOUS", "ID_CONFLICT")


def test_clone_after_friendly_established_overrides_friendly():
    cfg = make_cfg()
    real, other = line(40, 40, 12, 0), line(150, 110, -10, 0)
    outs = simulate(cfg, 5.0, {1: real, 2: other}, [("F-1", real)])
    assert at(outs, 2.9)["labels"][1]["state"] == "FRIENDLY"
    both = [("F-1", real), ("F-1", other)]
    m = Matcher(cfg)
    inputs_a, inputs_b = [], []
    simulate(cfg, 3.0, {1: real, 2: other}, [("F-1", real)], record=inputs_a)
    simulate(cfg, 2.0, {1: real, 2: other}, both, start_t=3.0, record=inputs_b)
    for t, tracks, claims in inputs_a + inputs_b[1:]:
        out = m.update(t, tracks, claims)
        if t > 3.2:
            assert state(out, 1) == ("SUSPICIOUS", "ID_CONFLICT")
            assert state(out, 2) == ("SUSPICIOUS", "ID_CONFLICT")


def test_motion_mismatch_claim_says_driving_vehicle_is_parked():
    """A valid claim sits on a parked vehicle but reports 20 cm/s: position agrees, motion doesn't."""
    cfg = make_cfg()
    m = Matcher(cfg)
    rssi = expected_rssi(100, 75, ARENA, cfg["scale_m_per_cm"], cfg["rf"]["rssi"])
    for i in range(int(3.0 * 30)):
        t = i / 30
        tracks = [{"track_id": 1, "t": t, "x": 100.0, "y": 75.0, "vx": 0.0, "vy": 0.0, "speed": 0.0, "heading": 0.0}]
        claims = []
        if i % 6 == 3:  # 5 Hz; measured 0.1 s ago; propagated by the matcher it lands on the car
            claims = [{"rf_id": "F-1", "nonce": f"n{i}", "t": t - 0.1, "x": 98.0, "y": 75.0,
                       "vx": 20.0, "vy": 0.0, "rssi": rssi, "conflict": False}]
        out = m.update(t, tracks, claims)
    assert state(out, 1) == ("SUSPICIOUS", "MOTION_MISMATCH")
    assert out["labels"][1]["scores"]["P"] > 0.8 and out["labels"][1]["scores"]["M"] < 0.3


def test_parked_friendly_with_parked_claim_is_friendly():
    cfg = make_cfg()
    car = still(100, 75)
    outs = simulate(cfg, 3.0, {1: car}, [("F-1", car)])
    assert final(outs, 1) == ("FRIENDLY", None)
    assert "low_info" in outs[-1][1]["labels"][1]["flags"]


def test_silence_coasts_with_custody_and_never_becomes_suspicious():
    cfg = make_cfg()
    car = line(30, 40, 10, 4)
    outs = simulate(cfg, 12.0, {1: car, 2: still(180, 130)}, [("F-1", car)], silence=[(3.0, 99)])
    late = [state(o, 1) for t, o in outs if t > 3.5]
    assert all(s == ("FRIENDLY", "COASTING") for s in late)
    assert outs[-1][1]["labels"][1]["coast_s"] == pytest.approx(12.0 - 3.0, abs=0.4)
    assert not any(s[0] == "SUSPICIOUS" for t, o in outs for s in [state(o, 1)])


def test_coasting_ends_when_another_vehicle_comes_close():
    cfg = make_cfg()
    car = line(30, 75, 10, 0)
    intruder = line(140, 75, -10, 0)  # head-on: gap closes 20 cm/s; < 15 cm at t ≈ 4.75 s
    outs = simulate(cfg, 6.0, {1: car, 2: intruder}, [("F-1", car)], silence=[(2.5, 99)])
    assert state(at(outs, 3.5), 1) == ("FRIENDLY", "COASTING")
    assert final(outs, 1) == ("UNKNOWN", "NO_ID")
    t_drop = next(t for t, o in outs if t > 2.5 and state(o, 1) == ("UNKNOWN", "NO_ID"))
    assert 4.6 <= t_drop <= 4.9  # drops the instant custody is doubtful, no hysteresis wait


def test_coasting_ends_after_max_time():
    cfg = make_cfg()
    cfg["matcher"]["coast_max_s"] = 4
    car = still(100, 75)
    outs = simulate(cfg, 9.0, {1: car}, [("F-1", car)], silence=[(3.0, 99)])
    assert final(outs, 1) == ("UNKNOWN", "NO_ID")
    t_drop = next(t for t, o in outs if t > 3 and state(o, 1)[0] == "UNKNOWN")
    assert 6.9 <= t_drop <= 7.3


def test_rf_denied_raised_after_silence_and_cleared_on_return():
    cfg = make_cfg()
    car = line(30, 40, 10, 4)
    outs = simulate(cfg, 9.0, {1: car}, [("F-1", car)], silence=[(3.0, 6.0)])
    assert at(outs, 2.9)["rf_denied"] is False
    assert at(outs, 4.6)["rf_denied"] is False   # < 2 s since the last claim (~3.0 s)
    assert at(outs, 5.5)["rf_denied"] is True
    assert at(outs, 6.5)["rf_denied"] is False   # claims are back


def test_no_rf_denied_before_any_claim_was_ever_heard():
    cfg = make_cfg()
    outs = simulate(cfg, 4.0, {1: still(50, 50)}, [])
    assert not any(o["rf_denied"] for _, o in outs)


def test_lost_friendly_leaves_last_known_marker():
    cfg = make_cfg()
    car = line(30, 40, 10, 0)
    gone = lambda t: car(t) if t < 3.0 else None  # noqa: E731 - track disappears (e.g. under cover)
    outs = simulate(cfg, 5.0, {1: gone}, [("F-1", car)], silence=[(3.0, 99)])
    lost = outs[-1][1]["lost_friendlies"]
    assert len(lost) == 1 and lost[0]["rf_id"] == "F-1"
    assert lost[0]["x"] == pytest.approx(60, abs=1.5)
    assert lost[0]["age_s"] == pytest.approx(2.0, abs=0.1)


def test_ghost_claim_without_vehicle_after_grace():
    cfg = make_cfg()
    outs = simulate(cfg, 3.0, {1: still(50, 50)}, [("F-1", still(150, 120))])
    early = [o["ghosts"] for t, o in outs if t < 0.9]
    assert all(g == [] for g in early)  # a missed detection isn't instantly a ghost
    ghosts = outs[-1][1]["ghosts"]
    assert len(ghosts) == 1 and ghosts[0]["rf_id"] == "F-1"
    assert final(outs, 1) == ("UNKNOWN", "NO_ID")


def test_all_pairs_gated_does_not_crash():
    cfg = make_cfg()
    outs = simulate(cfg, 1.0, {1: still(10, 10), 2: still(20, 10)}, [("F-1", still(190, 140)), ("F-2", still(180, 140))])
    assert final(outs, 1) == ("UNKNOWN", "NO_ID")


def test_no_flicker_under_noise():
    cfg = make_cfg()
    car = line(30, 40, 12, 6)
    outs = simulate(cfg, 8.0, {1: car}, [("F-1", car)], noise_cm=3.0, seed=4)
    seq = [state(o, 1) for t, o in outs if t > 2.0]
    changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    assert changes == 0 and seq[-1] == ("FRIENDLY", None)


def test_recent_friendly_time_is_exposed_for_the_turret_interlock():
    cfg = make_cfg()
    car = line(30, 40, 10, 4)
    outs = simulate(cfg, 3.0, {1: car}, [("F-1", car)])
    lab = outs[-1][1]["labels"][1]
    assert lab["last_friendly_t"] == pytest.approx(outs[-1][0], abs=1e-6)


# --- metamorphic relations -----------------------------------------------------------------

SCENE = {
    "vehicles": {1: line(40, 40, 12, 3), 2: line(150, 110, -10, 0), 3: still(100, 30)},
    "emitters": [("F-1", line(40, 40, 12, 3)), ("F-1", line(150, 110, -10, 0)), ("F-2", still(100, 75))],
}


def labels_seq(outs):
    return [{tid: (l["state"], l["reason"], l["rf_id"]) for tid, l in o["labels"].items()} for _, o in outs]


def test_mr_translation_invariance():
    cfg = make_cfg(r_weight=0.0)
    base = simulate(cfg, 4.0, **SCENE, noise_cm=1.0)
    moved = simulate(cfg, 4.0, **SCENE, noise_cm=1.0,
                     transform=lambda x, y, vx, vy: (x + 37.5, y - 22.0, vx, vy))
    assert labels_seq(base) == labels_seq(moved)


def test_mr_rotation_invariance():
    cfg = make_cfg(r_weight=0.0)
    base = simulate(cfg, 4.0, **SCENE, noise_cm=1.0)
    rot = simulate(cfg, 4.0, **SCENE, noise_cm=1.0,
                   transform=lambda x, y, vx, vy: (-y, x, -vy, vx))  # 90° turn
    assert labels_seq(base) == labels_seq(rot)


def test_mr_track_id_renaming_permutes_labels():
    cfg = make_cfg()
    base = simulate(cfg, 4.0, **SCENE)
    mapping = {1: 70, 2: 5, 3: 12}
    renamed = simulate(cfg, 4.0, **SCENE, track_ids=mapping)
    for (_, a), (_, b) in zip(base, renamed):
        for tid, lab in a["labels"].items():
            other = b["labels"][mapping[tid]]
            assert (lab["state"], lab["reason"], lab["rf_id"]) == (other["state"], other["reason"], other["rf_id"])


def test_mr_input_order_invariance():
    cfg = make_cfg()
    inputs = []
    simulate(cfg, 3.0, **SCENE, record=inputs)
    a, b, rng = Matcher(cfg), Matcher(cfg), random.Random(3)
    for t, tracks, claims in inputs:
        oa = a.update(t, tracks, claims)
        tr2, cl2 = copy.deepcopy(tracks), copy.deepcopy(claims)
        rng.shuffle(tr2)
        rng.shuffle(cl2)
        ob = b.update(t, tr2, cl2)
        assert {k: (v["state"], v["reason"]) for k, v in oa["labels"].items()} == \
               {k: (v["state"], v["reason"]) for k, v in ob["labels"].items()}


def test_mr_frame_rate_invariance():
    cfg = make_cfg()
    fast = simulate(cfg, 6.0, **SCENE, fps=30)
    slow = simulate(cfg, 6.0, **SCENE, fps=15)
    assert labels_seq(fast)[-1] == labels_seq(slow)[-1]
    for tid in (1, 2, 3):  # transitions happen at the same time within one slow frame
        for st in ("FRIENDLY", "SUSPICIOUS"):
            a, b = first_time(fast, tid, st), first_time(slow, tid, st)
            assert (a is None) == (b is None)
            if a is not None:
                assert abs(a - b) <= 1 / 15 + 1e-6


def test_mr_irrelevant_far_vehicle_changes_nothing():
    cfg = make_cfg()
    base = simulate(cfg, 4.0, **SCENE)
    extra = {**SCENE["vehicles"], 9: still(195, 5)}
    more = simulate(cfg, 4.0, vehicles=extra, emitters=SCENE["emitters"])
    for (_, a), (_, b) in zip(base, more):
        for tid in (1, 2, 3):
            assert state(a, tid) == state(b, tid)
        assert state(b, 9) == ("UNKNOWN", "NO_ID")


def test_mr_position_score_decreases_with_distance():
    cfg = make_cfg()
    ps = []
    for off in (0, 3, 6, 10, 15, 20):
        car = line(60, 60, 10, 0)
        claim = line(60 + off, 60, 10, 0)
        outs = simulate(cfg, 1.5, {1: car}, [("F-1", claim)])
        ps.append(outs[-1][1]["labels"][1]["scores"]["P"])
    assert all(a > b for a, b in zip(ps, ps[1:]))


# --- tests added after mutation testing (each kills a mutant the suite above missed) -------

def direct_run(cfg, seconds, track_v, claim_v, fps=30, claim_offset=lambda t: 0.0):
    """One parked vehicle at (100, 75) whose tracker reports velocity track_v (estimator noise);
    one emitter at (100 + claim_offset, 75) reporting claim_v, 5 Hz, 0.1 s latency. A non-zero
    claim_v makes the propagated claim drift off the car, so motion tests widen sigma_pos_cm."""
    m = Matcher(cfg)
    rssi = expected_rssi(100, 75, ARENA, cfg["scale_m_per_cm"], cfg["rf"]["rssi"])
    outs, next_round = [], 0.0
    pending = []
    for i in range(int(round(seconds * fps)) + 1):
        t = i / fps
        while t >= next_round - 1e-9:
            tc = next_round
            pending.append((tc + 0.1, {"rf_id": "F-1", "nonce": f"{tc:.1f}", "t": tc,
                                       "x": 100 + claim_offset(tc), "y": 75.0,
                                       "vx": claim_v[0], "vy": claim_v[1], "rssi": rssi, "conflict": False}))
            next_round += 0.2
        due = [c for d, c in pending if d <= t + 1e-9]
        pending = [(d, c) for d, c in pending if d > t + 1e-9]
        # the vehicle sits still; its *reported* velocity is track_v (estimator noise)
        sp = math.hypot(*track_v)
        tracks = [{"track_id": 1, "t": t, "x": 100.0, "y": 75.0, "vx": track_v[0], "vy": track_v[1],
                   "speed": sp, "heading": math.atan2(track_v[1], track_v[0])}]
        outs.append((t, m.update(t, tracks, due)))
    return outs


def test_creeping_car_with_flipped_velocity_noise_stays_friendly():
    """R8: at 3 cm/s, heading estimates are noise. Opposite headings must not mean mismatch."""
    cfg = make_cfg()
    cfg["matcher"]["sigma_pos_cm"] = 50  # isolate the motion term from propagation drift
    outs = direct_run(cfg, 3.0, track_v=(3.0, 0.0), claim_v=(-3.0, 0.0))
    assert final(outs, 1) == ("FRIENDLY", None)


def test_borderline_speed_disagreement_is_not_called_a_mismatch():
    """Conservative rule: MOTION MISMATCH needs informative evidence (heading active, or a
    speed gap > 2σv). A 9 cm/s gap with the car parked stays UNKNOWN, not SUSPICIOUS."""
    cfg = make_cfg()
    cfg["matcher"]["sigma_pos_cm"] = 50
    outs = direct_run(cfg, 3.0, track_v=(0.0, 0.0), claim_v=(9.0, 0.0))
    assert final(outs, 1)[0] == "UNKNOWN"
    assert outs[-1][1]["labels"][1]["scores"]["M"] < 0.3


def test_mr_frame_rate_invariance_while_evidence_changes():
    """Confidence ramps as a 14 cm offset closes at t=2 s: entry time must not depend on fps."""
    cfg = make_cfg()
    offset = lambda tc: 14.0 if tc < 2.0 else 0.0  # noqa: E731
    fast = direct_run(cfg, 5.0, (0.0, 0.0), (0.0, 0.0), fps=30, claim_offset=offset)
    slow = direct_run(cfg, 5.0, (0.0, 0.0), (0.0, 0.0), fps=10, claim_offset=offset)
    a, b = first_time(fast, 1, "FRIENDLY"), first_time(slow, 1, "FRIENDLY")
    assert a is not None and b is not None
    assert a > 2.5  # it really had to wait for the evidence to improve
    assert abs(a - b) <= 1 / 10 + 1e-6


def test_unsettled_track_velocity_is_never_a_mismatch():
    """Right after a merge/split the track's velocity is junk; motion must count as no-info."""
    cfg = make_cfg()
    m = Matcher(cfg)
    rssi = expected_rssi(100, 75, ARENA, cfg["scale_m_per_cm"], cfg["rf"]["rssi"])
    for i in range(int(3.0 * 30)):
        t = i / 30
        tracks = [{"track_id": 1, "t": t, "x": 100.0, "y": 75.0, "vx": 60.0, "vy": 0.0, "speed": 60.0,
                   "heading": 0.0, "settled": False}]
        claims = []
        if i % 6 == 3:
            claims = [{"rf_id": "F-1", "nonce": f"n{i}", "t": t - 0.1, "x": 100.0, "y": 75.0,
                       "vx": 0.0, "vy": 0.0, "rssi": rssi, "conflict": False}]
        out = m.update(t, tracks, claims)
    assert state(out, 1)[0] != "SUSPICIOUS"


def test_merged_box_is_too_close_but_conflict_still_shows():
    cfg = make_cfg()
    car = line(60, 60, 10, 0)

    def run(emitters):
        m = Matcher(cfg)
        recs = []
        simulate(cfg, 3.0, {1: car}, emitters, record=recs)
        for t, tracks, claims in recs:
            for k in tracks:
                k["merged"] = True
            out = m.update(t, tracks, claims)
        return state(out, 1)

    assert run([("F-1", car)]) == ("UNKNOWN", "TOO_CLOSE")
    assert run([("F-1", car), ("F-1", still(150, 120))]) == ("SUSPICIOUS", "ID_CONFLICT")
