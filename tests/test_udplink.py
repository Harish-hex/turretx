"""Real friendly beacons over WiFi UDP (firmware/turretx_rover) → the existing receiver and matcher."""
import json
import socket
import time

import pytest

from turretx import wire
from turretx.config import load_config
from turretx.matcher import Matcher
from turretx.rfrx import Interrogator, Receiver
from turretx.rfsim import RFSim
from turretx.udplink import UdpBeaconLink

KEY = wire.rf_key("DEMO-KEY-turretx")


def _sock():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2.0)
    return s


def _link(targets, challenge_port=4210):
    return UdpBeaconLink({"targets": targets, "challenge_port": challenge_port, "reply_port": 0,
                          "telemetry_port": 0}, KEY)


def _poll_until(link, n=1, timeout=2.0):
    got, end = [], time.monotonic() + timeout
    while len(got) < n and time.monotonic() < end:
        got += link.poll()
        time.sleep(0.01)
    return got


def _datagram(payload, key=KEY):
    return json.dumps({"payload": payload.hex(), "sig": wire.sign(key, payload)}).encode()


def test_challenge_is_compact_json_sent_to_each_target():
    rover = _sock()
    link = _link(["127.0.0.1"], challenge_port=rover.getsockname()[1])
    link.send_challenge({"nonce": "ab12cd34", "t": 1.25})
    data, _ = rover.recvfrom(256)
    assert data == b'{"n":"ab12cd34","t":1250}'  # compact: the firmware's parser needs it


def test_signed_reply_becomes_a_claim_marked_rssi_unknown():
    inter = Interrogator(5)
    rx = Receiver(KEY, inter)
    ch = inter.poll(10.0)
    link = _link([])
    rover = _sock()
    payload = wire.encode_payload("F-1", ch["nonce"], ch["t"], 52.3, 30.0, -12.0, 0.0, -49.0)
    rover.sendto(_datagram(payload), ("127.0.0.1", link.reply_port))
    claims, rejections = rx.receive(_poll_until(link), 10.1)
    assert not rejections and len(claims) == 1
    c = claims[0]
    assert (c["rf_id"], c["x"], c["vx"], c["rssi_known"]) == ("F-1", 52.3, -12.0, False)


def test_garbage_and_forged_datagrams_are_rejected():
    inter = Interrogator(5)
    rx = Receiver(KEY, inter)
    ch = inter.poll(0.0)
    link = _link([])
    rover = _sock()
    rover.sendto(b"not json at all", ("127.0.0.1", link.reply_port))
    forged = wire.encode_payload("F-1", ch["nonce"], 0.0, 1.0, 1.0, 0.0, 0.0, -49.0)
    rover.sendto(_datagram(forged, key=b"guessed-key"), ("127.0.0.1", link.reply_port))
    claims, rejections = rx.receive(_poll_until(link, n=2), 0.1)
    assert claims == [] and sorted(r["reason"] for r in rejections) == ["bad_signature", "malformed"]


def test_beacon_address_is_learned_only_from_signed_telemetry():
    link = _link(["auto"])
    rover = _sock()
    tel = b'{"id":"F-1","type":"tel","seq":1,"x":200,"y":200}'
    rover.sendto(_datagram(tel, key=b"wrong"), ("127.0.0.1", link.telemetry_port))
    _poll_until(link, n=99, timeout=0.3)
    assert link.targets == set()  # a forged telemetry packet can't redirect challenges
    rover.sendto(_datagram(tel), ("127.0.0.1", link.telemetry_port))
    _poll_until(link, n=99, timeout=0.3)
    assert link.targets == {"127.0.0.1"} and link.telemetry["127.0.0.1"]["seq"] == 1


def _parked_run(rssi_known):
    cfg = load_config()
    cfg["table_cm"] = [100, 60]
    m = Matcher(cfg)
    out = None
    for i in range(75):  # 2.5 s at 30 fps, claim every 0.2 s
        t = i / 30
        track = {"track_id": 1, "t": t, "x": 50.0, "y": 30.0, "vx": 0.0, "vy": 0.0, "speed": 0.0,
                 "heading": 0.0, "settled": True, "merged": False}
        claims = []
        if i % 6 == 0:
            claims = [{"rf_id": "F-1", "t": t, "x": 50.5, "y": 30.0, "vx": 0.0, "vy": 0.0,
                       "rssi": -49.0, "nonce": f"n{i}", "conflict": False, "rssi_known": rssi_known}]
        out = m.update(t, [track], claims)
    return out["labels"][1]


def test_parked_hardware_friendly_is_identified_without_drone_rssi():
    """The rover's RSSI is its own WiFi signal, not the drone's view: R must be ignored, else a
    parked real friendly (M = 0.5, low info) could never reach the 0.70 threshold."""
    assert _parked_run(rssi_known=True)["state"] == "UNKNOWN"   # R ~ 0 drags it under 0.70
    lab = _parked_run(rssi_known=False)
    assert lab["state"] == "FRIENDLY" and lab["rf_id"] == "F-1" and lab["confidence"] >= 0.70


def test_real_ids_switch_off_their_simulated_beacon_but_not_the_attacker():
    cfg = load_config()
    sim = RFSim(cfg, seed=1, exclude=["F-1"])
    assert "F-1" not in sim.beacons and sim.attacker.target_id == "F-1"


@pytest.mark.parametrize("module", ["matcher.py", "tracker.py", "rfrx.py"])
def test_identity_modules_never_import_the_udp_link(module):
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parent.parent / "turretx" / module).read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("udplink" in m for m in names)
