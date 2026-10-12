"""Comms relay (RELAY_INTEGRATION.md): signed messages, drone relay when the direct link is blocked,
consistency-based jam detection, keyed channel hop, priority/TTL/dedup/receipts, bounded queue."""
import ast
import pathlib

from turretx import relay, wire

KEY = b"test-key"
ROOT = pathlib.Path(__file__).resolve().parent.parent / "turretx"


def net(**kw):
    return relay.CommsNet(KEY, ["F-1", "F-2"], channels=4, **kw)


def run(n, t0, t1):
    t = t0
    while t < t1:
        n.step(t)
        t += 0.1
    return t


def delivered(n, types=("txt", "pos", "sys")):
    return [m for m in n.log if m["route"] in ("direct", "via drone", "drone") and m["type"] in types]


def jam_now(n):
    n.link.jam_channels = {n.channel}


# ---------------------------------------------------------------- routing
def test_direct_delivery_when_clear():
    n = net()
    n.send(0.0, "F-1", "F-2", "ok")
    run(n, 0.0, 1.0)
    (m,) = delivered(n)
    assert (m["route"], m["text"], m["src"], m["dst"]) == ("direct", "ok", "F-1", "F-2")


def test_barrier_routes_via_drone_without_calling_it_jamming():
    n = net()
    n.link.direct_blocked = True
    n.send(0.0, "F-1", "F-2", "need water")
    run(n, 0.0, 2.0)
    (m,) = delivered(n)
    assert m["route"] == "via drone"
    assert not n.jam_suspected and n.hops == 0


# ---------------------------------------------------------------- jamming
def test_jam_blocks_then_is_detected_then_hop_then_queue_flushes_in_order():
    n = net()
    jam_now(n)
    jammed = n.channel
    for i in range(3):
        n.send(0.1 * i, "F-1", "F-2", f"m{i}")
    run(n, 0.0, 0.2)
    assert delivered(n) == []                        # nothing gets through the jammer
    run(n, 0.2, 6.0)                                 # 3rd failed attempt -> detected -> hop -> flush
    assert n.hops >= 1 and n.channel != jammed       # hopped off the jammed channel
    assert [m["text"] for m in delivered(n)] == ["m0", "m1", "m2"]


def test_jam_banner_shows_then_clears():
    n = net()
    jam_now(n)
    for i in range(3):
        n.send(0.1 * i, "F-1", "F-2", f"m{i}")
    run(n, 0.0, 0.6)
    assert n.jam_suspected
    run(n, 0.6, 8.0)
    assert not n.jam_suspected


def test_priority_alert_overtakes_chat_when_the_queue_flushes():
    n = net()
    jam_now(n)
    n.send(0.0, "F-1", "F-2", "chat-1", prio=relay.CHAT)
    n.send(0.1, "F-1", "F-2", "chat-2", prio=relay.CHAT)
    n.send(0.2, "F-1", "F-2", "ALERT", mtype="sys", prio=relay.ALERT)
    run(n, 0.0, 6.0)
    order = [m["text"] for m in delivered(n)]
    assert order == ["ALERT", "chat-1", "chat-2"]


def test_ttl_expires_messages_that_cannot_be_delivered():
    n = net()
    n.link.jam_channels = set(range(4))              # every channel jammed: hopping can't help
    mid = n.send(0.0, "F-1", "F-2", "stale", ttl=2)
    run(n, 0.0, 4.0)
    assert n.status[mid] == "expired"
    assert n.queued == 0 and not delivered(n)


def test_bounded_queue_drops_oldest_lowest_priority_and_keeps_alerts():
    n = net(queue_max=3)
    jam_now(n)
    chats = [n.send(0.1 * i, "F-1", "F-2", f"c{i}") for i in range(3)]
    alert = n.send(0.3, "F-1", "F-2", "ALERT", mtype="sys", prio=relay.ALERT)
    assert n.queued == 3
    assert n.status[chats[0]] == "dropped" and n.status[alert] == "queued"
    late_chat = n.send(0.4, "F-1", "F-2", "late chat")      # queue full of >= priority: new one loses
    assert n.status[late_chat] != "dropped" or n.queued == 3


def test_full_queue_of_alerts_drops_a_new_chat():
    n = net(queue_max=2)
    jam_now(n)
    n.send(0.0, "F-1", "F-2", "a1", mtype="sys", prio=relay.ALERT)
    n.send(0.1, "F-1", "F-2", "a2", mtype="sys", prio=relay.ALERT)
    chat = n.send(0.2, "F-1", "F-2", "chat")
    assert n.status[chat] == "dropped" and n.queued == 2


# ---------------------------------------------------------------- authentication
def test_forged_signature_rejected_by_drone():
    n = net()
    good = relay.make_message(KEY, "F-1", "F-2", 1, 0.0, "hello")
    bad = {"payload": good["payload"], "sig": "0" * 64}
    assert n.inject(0.0, bad) == "rejected: bad signature"
    assert not delivered(n)


def test_message_signed_with_other_key_rejected():
    n = net()
    pkt = relay.make_message(b"wrong", "F-1", "F-2", 1, 0.0, "x")
    assert n.inject(0.0, pkt) == "rejected: bad signature"


def test_replay_rejected_but_first_copy_accepted():
    n = net()
    pkt = relay.make_message(KEY, "F-1", "F-2", 5, 0.0, "once")
    assert n.inject(0.0, pkt).startswith("accepted")
    assert n.inject(1.0, pkt) == "rejected: replay"
    assert len(delivered(n)) == 1


def test_out_of_order_sequence_numbers_are_accepted():
    n = net()
    assert n.inject(0.0, relay.make_message(KEY, "F-1", "F-2", 5, 0.0, "later")).startswith("accepted")
    assert n.inject(0.1, relay.make_message(KEY, "F-1", "F-2", 3, 0.0, "earlier")).startswith("accepted")


def test_old_packet_replayed_after_its_ttl_is_rejected_as_expired():
    n = net()
    pkt = relay.make_message(KEY, "F-1", "F-2", 1, 0.0, "old", ttl=5)
    assert n.inject(1.0, pkt).startswith("accepted")
    n.step(100.0)                                    # the seen-entry is forgotten ...
    assert n.inject(100.0, pkt) == "rejected: expired"   # ... but freshness still refuses it


def test_malformed_payload_rejected_not_crashing():
    n = net()
    assert n.inject(0.0, wire.make_packet(KEY, b"not json at all")).startswith("rejected")
    bad_type = wire.make_packet(KEY, b'{"s":"a","d":"b","n":1,"t":0,"y":"boom","p":1,"l":5,"m":"x"}')
    assert n.inject(0.0, bad_type) == "rejected: malformed"


# ---------------------------------------------------------------- receipts, broadcast
def test_unicast_delivery_returns_a_receipt_to_the_sender():
    n = net()
    mid = n.send(0.0, "F-1", "F-2", "ping")
    assert n.status[mid] == "queued"
    run(n, 0.0, 2.0)
    assert n.status[mid] == "receipted"
    assert any(m["type"] == "rcpt" and m["src"] == "F-2" and m["dst"] == "F-1" for m in n.log)


def test_receipt_is_not_receipted_and_not_shown_as_a_message():
    n = net()
    n.send(0.0, "F-1", "F-2", "ping")
    run(n, 0.0, 3.0)
    assert len([m for m in n.log if m["type"] == "rcpt"]) == 1
    assert len(delivered(n)) == 1


def test_broadcast_from_the_drone_needs_no_receipt():
    n = net()
    n.send(0.0, "drone", "*", "CAMERA LOST", mtype="sys", prio=relay.ALERT)
    run(n, 0.0, 2.0)
    (m,) = delivered(n)
    assert (m["route"], m["dst"]) == ("drone", "*")
    assert not [m for m in n.log if m["type"] == "rcpt"]


def test_receipt_survives_a_jam_and_still_arrives():
    n = net()
    n.send(0.0, "F-1", "F-2", "ping")
    run(n, 0.0, 0.2)
    jam_now(n)
    run(n, 0.2, 8.0)
    assert n.status[(("F-1", 1))] in ("delivered", "receipted")


# ---------------------------------------------------------------- keyed hop schedule
def test_hop_schedule_is_keyed_and_deterministic():
    a = [relay.hop_channel(KEY, k, 4) for k in range(8)]
    assert a == [relay.hop_channel(KEY, k, 4) for k in range(8)]
    assert a != [relay.hop_channel(b"other", k, 4) for k in range(8)]
    assert all(0 <= c < 4 for c in a)


# ---------------------------------------------------------------- link independence, firewall
class AlwaysUp:
    n_channels, channel = 4, 0

    def set_channel(self, c):
        self.channel = c

    def transmit(self, packet, src, dst):
        return relay.LinkResult(True, 0.0, "direct")


def test_protocol_works_over_any_link_object():
    n = relay.CommsNet(KEY, ["F-1", "F-2"], link=AlwaysUp())
    n.send(0.0, "F-1", "F-2", "hi")
    run(n, 0.0, 1.0)
    assert len(delivered(n)) == 1


def test_identity_modules_never_import_the_relay():
    for name in ("matcher.py", "tracker.py", "rfrx.py", "rfmodel.py", "kinematics.py"):
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""] + [a.name for a in node.names]
            assert not any(m.split(".")[-1] == "relay" for m in mods), f"{name} imports relay"
