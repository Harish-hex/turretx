import random

import numpy as np
import pytest

from turretx import synthetic, wire
from turretx.config import load_config
from turretx.homography import Homography
from turretx.rfrx import Interrogator, Receiver
from turretx.rfsim import RFSim, rssi_at
from turretx.taglocator import TagLocator
from turretx.tracker import Tracker

KEY = b"test-key"


def pkt(nonce, rf_id="F-1", x=50.0, key=KEY, t=1.0):
    return wire.make_packet(key, wire.encode_payload(rf_id, nonce, t, x, 30.0, 5.0, 0.0, -90.0))


@pytest.fixture
def link():
    inter = Interrogator(challenge_hz=5)
    return inter, Receiver(KEY, inter)


# --- wire format -------------------------------------------------------------------------

def test_payload_roundtrip_integer_fields():
    p = wire.encode_payload("F-1", "ab12", 1.2345, 12.34, -5.67, 8.91, 0.04, -87.65)
    assert b"." not in p  # integers only on the wire (R14)
    d = wire.decode_payload(p)
    assert d["rf_id"] == "F-1" and d["nonce"] == "ab12"
    assert d["t"] == pytest.approx(1.234, abs=1e-3)
    assert (d["x"], d["y"], d["vx"]) == (pytest.approx(12.3), pytest.approx(-5.7), pytest.approx(8.9))
    assert d["rssi"] == pytest.approx(-87.6, abs=0.06)


@pytest.mark.parametrize("payload", [
    b"not json", b"[]", b'{"id":"F-1"}',
    b'{"id":"F-1","n":"a","t":1,"x":1.5,"y":1,"vx":1,"vy":1,"rssi":-900}',      # float field
    b'{"id":"F-1","n":"a","t":1,"x":true,"y":1,"vx":1,"vy":1,"rssi":-900}',     # bool field
    b'{"id":"F-1","n":"a","t":1,"x":999999999,"y":1,"vx":1,"vy":1,"rssi":-900}',  # out of range
    b'{"id":"","n":"a","t":1,"x":1,"y":1,"vx":1,"vy":1,"rssi":-900}',           # empty id
])
def test_decode_rejects_malformed(payload):
    assert wire.decode_payload(payload) is None


def test_key_env_override(monkeypatch):
    monkeypatch.setenv("TURRETX_RF_KEY", "from-env")
    assert wire.rf_key("DEMO") == b"from-env"
    monkeypatch.delenv("TURRETX_RF_KEY")
    assert wire.rf_key("DEMO") == b"DEMO"


# --- receiver checks, in order ------------------------------------------------------------

def test_valid_reply_becomes_claim(link):
    inter, rx = link
    ch = inter.poll(0.0)
    claims, rej = rx.receive([pkt(ch["nonce"])], 0.1)
    assert rej == [] and len(claims) == 1
    assert claims[0]["rf_id"] == "F-1" and claims[0]["conflict"] is False


def test_tampered_payload_is_bad_signature(link):
    inter, rx = link
    p = pkt(inter.poll(0.0)["nonce"])
    p["payload"] = p["payload"].replace(b'"x":500', b'"x":900')
    claims, rej = rx.receive([p], 0.1)
    assert claims == [] and rej[0]["reason"] == "bad_signature" and rej[0]["rf_id"] is None


def test_wrong_key_is_bad_signature(link):
    inter, rx = link
    claims, rej = rx.receive([pkt(inter.poll(0.0)["nonce"], key=b"guess")], 0.1)
    assert claims == [] and rej[0]["reason"] == "bad_signature"


def test_signed_garbage_and_oversize_are_malformed(link):
    inter, rx = link
    inter.poll(0.0)
    garbage = wire.make_packet(KEY, b'{"hello":1}')
    huge = wire.make_packet(KEY, b"x" * (wire.MAX_PAYLOAD_BYTES + 1))
    claims, rej = rx.receive([garbage, huge, {"sig": "nope"}], 0.1)
    assert claims == [] and [r["reason"] for r in rej] == ["malformed"] * 3


def test_old_nonce_is_stale(link):
    inter, rx = link
    old = inter.poll(0.0)["nonce"]
    for i in range(1, 4):  # three newer challenges push it out of the window
        inter.poll(i * 0.2)
    claims, rej = rx.receive([pkt(old)], 0.7)
    assert claims == [] and rej[0]["reason"] == "stale_nonce" and rej[0]["rf_id"] == "F-1"


def test_exact_duplicate_is_replayed(link):
    inter, rx = link
    p = pkt(inter.poll(0.0)["nonce"])
    rx.receive([p], 0.1)
    claims, rej = rx.receive([p], 0.15)
    assert claims == [] and rej[0]["reason"] == "replayed"


def test_two_valid_answers_to_one_challenge_is_conflict(link):
    inter, rx = link
    n = inter.poll(0.0)["nonce"]
    claims, rej = rx.receive([pkt(n, x=20.0), pkt(n, x=80.0)], 0.1)
    assert rej == [] and len(claims) == 2 and all(c["conflict"] for c in claims)
    # the next round's genuine reply stays flagged for CONFLICT_HOLD_S
    n2 = inter.poll(0.2)["nonce"]
    later, _ = rx.receive([pkt(n2, x=20.0)], 0.3)
    assert later[0]["conflict"] is True
    n3 = inter.poll(1.4)["nonce"]
    clear, _ = rx.receive([pkt(n3, x=20.0)], 1.5)
    assert clear[0]["conflict"] is False


def test_different_ids_same_nonce_no_conflict(link):
    inter, rx = link
    n = inter.poll(0.0)["nonce"]
    claims, _ = rx.receive([pkt(n, "F-1"), pkt(n, "F-2")], 0.1)
    assert not any(c["conflict"] for c in claims)


def test_interrogator_rate_and_fresh_nonces():
    inter = Interrogator(challenge_hz=5)
    issued = [c for t in np.arange(0, 2.0, 1 / 30) if (c := inter.poll(float(t)))]
    assert len(issued) == 10
    assert len({c["nonce"] for c in issued}) == 10


def test_rssi_falls_with_distance():
    cfg = dict(load_config()["rf"], scale_m_per_cm=10)
    cfg["rssi"] = dict(cfg["rssi"], noise_db=0)
    rng = random.Random(0)
    centre = rssi_at(100, 75, (200, 150), cfg, rng)
    corner = rssi_at(0, 0, (200, 150), cfg, rng)
    assert corner < centre - 3  # meaningful spread across the arena at the scenario scale (R9)


# --- end to end on the synthetic clip ------------------------------------------------------

def run_link(seconds=6.0, actions=(), hide_sticker_after=None):
    """Synthetic scene → tracker + sticker GPS → RFSim → channel → Receiver."""
    cfg = load_config()
    cfg["table_cm"] = list(synthetic.TABLE_CM)
    cfg["detector"].update(backend="diff", ground_point_frac=0.5)
    homo = Homography(synthetic.TABLE_CM)
    assert homo.calibrate(synthetic.render(0.0))
    tracker = Tracker(cfg, homo)
    tracker.backend.set_background(synthetic.render(0, cars=False))
    tags = TagLocator(cfg["tags"], homo)
    sim = RFSim(cfg, seed=1)
    inter = Interrogator(cfg["rf"]["challenge_hz"])
    rx = Receiver(wire.rf_key(cfg["rf"]["key"]), inter)
    actions = dict(actions)
    out = {"claims": [], "rejections": [], "banners": []}
    for i in range(int(seconds * synthetic.FPS)):
        t = i / synthetic.FPS
        frame = synthetic.render(t)
        tracks = tracker.update(frame, t)
        fixes = tags.update(frame, t)
        if hide_sticker_after is not None and t >= hide_sticker_after:
            fixes = [dict(rf_id=f["rf_id"], t=t, visible=False) for f in fixes]
        sim.observe(t, fixes, tracks)
        for at, action in list(actions.items()):
            if t >= at:
                out["banners"].append(sim.red_team(action, t))
                del actions[at]
        ch = inter.poll(t)
        if ch:
            sim.on_challenge(ch, t)
        claims, rej = rx.receive(sim.poll(t), t)
        for c in claims:
            c["rx_t"] = t
        out["claims"] += claims
        out["rejections"] += rej
    return out


def test_friendly_claims_rate_latency_and_accuracy():
    out = run_link(seconds=6.0)
    claims = [c for c in out["claims"] if c["rf_id"] == "F-1"]
    assert out["rejections"] == []
    assert 27 <= len(claims) <= 31  # 5 Hz for ~6 s
    lat = [c["rx_t"] - c["t"] for c in claims]
    assert 0.09 <= np.mean(lat) <= 0.15
    err = [np.hypot(c["x"] - synthetic.car_states(c["t"])["F-1"][0],
                    c["y"] - synthetic.car_states(c["t"])["F-1"][1]) for c in claims]
    assert np.mean(err) < 6  # 3 cm noise per axis + sticker fix error
    assert not any(c["conflict"] for c in claims)


def test_forged_and_replay_are_rejected_not_claimed():
    out = run_link(seconds=4.0, actions={1.5: "forged", 3.0: "replay"})
    reasons = [r["reason"] for r in out["rejections"]]
    assert reasons.count("bad_signature") == 1
    assert reasons.count("stale_nonce") == 1
    assert any("forged" in b for b in out["banners"]) and any("replayed" in b for b in out["banners"])
    assert not any(c["conflict"] for c in out["claims"])


def test_clone_raises_conflict_on_both_vehicles():
    out = run_link(seconds=10.0, actions={1.0: "clone"})
    after = [c for c in out["claims"] if c["rx_t"] > 1.3]
    assert after and all(c["conflict"] for c in after)
    by_round = {}
    for c in after:
        by_round.setdefault(c["nonce"], []).append(c)
    assert sum(len(v) == 2 for v in by_round.values()) >= 0.8 * len(by_round)


def test_clone_stays_on_its_own_vehicle_through_the_crossing():
    """Regression: the clone followed a tracker ID, which swapped onto the real friendly when
    F-1 and U-3 crossed (t≈3 s). It must follow the physical vehicle (U-3) instead."""
    out = run_link(seconds=10.0, actions={1.0: "clone"})
    by_round = {}
    for c in out["claims"]:
        if c["rx_t"] > 1.3:
            by_round.setdefault(c["nonce"], []).append(c)
    for claims in by_round.values():
        if len(claims) != 2:
            continue
        t = claims[0]["t"]
        s = synthetic.car_states(t)
        f1, u3 = s["F-1"], s["U-3"]
        clone = max(claims, key=lambda c: np.hypot(c["x"] - f1[0], c["y"] - f1[1]))
        assert np.hypot(clone["x"] - u3[0], clone["y"] - u3[1]) < 8, f"clone left U-3 at t={t:.2f}"


def test_ghost_claims_an_empty_spot():
    out = run_link(seconds=3.0, actions={1.0: "ghost"})
    ghost_t = 1.3
    late = [c for c in out["claims"] if c["rx_t"] > ghost_t]
    real = synthetic.car_states(2.0)
    far = [c for c in late if min(np.hypot(c["x"] - x, c["y"] - y) for x, y, _ in real.values()) > 25]
    assert far  # some claims are at a spot with no vehicle


def test_jamming_stops_all_claims_and_resumes():
    out = run_link(seconds=5.0, actions={1.0: "jam", 3.0: "jam"})
    during = [c for c in out["claims"] if 1.05 < c["rx_t"] < 3.0]
    after = [c for c in out["claims"] if c["rx_t"] > 3.2]
    assert during == [] and after
    assert any("JAMMING ON" in b for b in out["banners"])


def test_hidden_sticker_dead_reckons_then_falls_silent():
    out = run_link(seconds=4.0, hide_sticker_after=2.0)
    f1 = [c for c in out["claims"] if c["rf_id"] == "F-1"]
    assert any(2.0 < c["t"] <= 3.0 for c in f1)        # keeps answering ~1 s on dead reckoning
    assert not any(c["t"] > 3.05 for c in f1)          # then silence (comms loss)
