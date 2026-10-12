"""Glue between the failsafe, the RF simulation and the relay (turretx/commshub.py)."""
from turretx import relay
from turretx.commshub import CommsHub

CFG = {"enabled": True, "nodes": ["F-1", "F-2"], "channels": 4, "hop_restores_iff": True, "pos_every_s": 2.0}
KEY = b"hub-key"


def hub(**over):
    return CommsHub(KEY, {**CFG, **over})


def claims(t, ids=("F-1",), x=60.0):
    return {i: {"rf_id": i, "t": t, "x": x, "y": 40.0, "vx": 0.0, "vy": 0.0} for i in ids}


def run(h, t0, t1, rung="L0", last=None, dt=0.1):
    ev, t = [], t0
    while t < t1:
        ev += h.step(t, rung, last or {})
        t += dt
    return ev


def delivered(h, mtype):
    return [m for m in h.net.log if m["type"] == mtype and m["route"] in ("direct", "via drone", "drone")]


def test_camera_loss_broadcasts_one_alert_and_restoration_one_status():
    h = hub()
    run(h, 0.0, 1.0, "L0")
    run(h, 1.0, 3.0, "L2", claims(1.0))
    run(h, 3.0, 4.0, "L0")
    texts = [m["text"] for m in delivered(h, "sys")]
    assert len(texts) == 2 and "CAMERA LOST" in texts[0] and "restored" in texts[1].lower()
    assert delivered(h, "sys")[0]["route"] == "drone" and delivered(h, "sys")[0]["dst"] == "*"


def test_no_alert_while_camera_stays_healthy():
    h = hub()
    run(h, 0.0, 5.0, "L0")
    assert delivered(h, "sys") == []


def test_blue_force_picture_broadcast_only_in_rf_only_levels():
    h = hub()
    run(h, 0.0, 6.0, "L0", claims(0.0))
    assert delivered(h, "pos") == []
    h = hub()
    run(h, 0.0, 6.0, "L2", claims(0.0, ("F-1", "F-2")))
    pos = delivered(h, "pos")
    assert {m["text"].split()[0] for m in pos} == {"F-1", "F-2"}
    assert 2 <= len(pos) // 2 <= 4                  # ~ every 2 s for 6 s
    h = hub()
    run(h, 0.0, 6.0, "L4", {})
    assert delivered(h, "pos") == []


def test_stale_positions_carry_their_age():
    h = hub()
    run(h, 10.0, 12.5, "L3", claims(0.0))
    assert any("age 10" in m["text"] or "age 11" in m["text"] for m in delivered(h, "pos"))


def test_hop_restores_iff_by_default_and_not_when_disabled():
    h = hub()
    h.set_jam(True)
    assert h.iff_jammed()
    h.net.send(0.0, "F-1", "F-2", "a")
    h.net.send(0.1, "F-1", "F-2", "b")
    h.net.send(0.2, "F-1", "F-2", "c")
    run(h, 0.0, 3.0)
    assert h.net.hops >= 1 and not h.iff_jammed()
    h = hub(hop_restores_iff=False)
    h.set_jam(True)
    for i in range(3):
        h.net.send(0.1 * i, "F-1", "F-2", str(i))
    run(h, 0.0, 3.0)
    assert h.net.hops >= 1 and h.iff_jammed()


def test_jam_off_clears_the_jammer():
    h = hub()
    h.set_jam(True)
    h.set_jam(False)
    assert not h.iff_jammed() and not h.net.link.jam_channels


def test_barrier_toggle_routes_via_drone():
    h = hub()
    assert "ON" in h.toggle_barrier()
    h.send_demo(0.0)
    run(h, 0.0, 1.5)
    assert delivered(h, "txt")[0]["route"] == "via drone"
    assert "OFF" in h.toggle_barrier()


def test_red_team_forged_and_replay_are_rejected_and_reported():
    h = hub()
    assert "rejected" in h.forged(0.0)
    assert "nothing" in h.replay(0.0).lower()
    h.send_demo(0.0)
    run(h, 0.0, 1.0)
    assert "rejected" in h.replay(1.0) and "replay" in h.replay(1.1)


def test_step_reports_hop_and_rejection_events_once():
    h = hub()
    h.forged(0.0)
    ev = run(h, 0.0, 0.5)
    assert sum(1 for k, _ in ev if k == "relay_rejected") == 1
    assert sum(1 for k, _ in run(h, 0.5, 1.0) if k == "relay_rejected") == 0


def test_demo_messages_cycle():
    h = hub()
    h.send_demo(0.0)
    h.send_demo(0.1)
    run(h, 0.0, 1.5)
    assert len({m["text"] for m in delivered(h, "txt")}) == 2


def test_summary_is_renderable_data():
    h = hub()
    h.send_demo(0.0)
    run(h, 0.0, 1.0)
    s = h.summary()
    assert set(s) >= {"channel", "hops", "jam_suspected", "queued", "barrier", "messages"}
    assert s["messages"] and s["messages"][-1]["state"] == "receipted"
