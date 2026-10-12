"""Drone comms relay, simulated (RELAY_INTEGRATION.md; research: reports/Camera failsafe and airborne relay.md).

Ground units send short signed messages. The drone is the relay: when the direct ground link is
blocked it forwards them. When the channel is jammed nothing gets through; the nodes tell jamming
apart from a blocked path by CONSISTENCY (low delivery AND high noise floor; a barrier lowers
delivery but not the noise), hop to the next channel on a schedule derived from the shared key,
and the queued messages are flushed highest priority first.

Protocol (this module) and delivery (a Link) are separate: SimLink is the simulation; a serial
bridge to ESP32 radios can replace it without touching the protocol.

SIMULATION ONLY. No RF is transmitted. "Jamming" is a flag that makes a channel undeliverable;
running a real jammer is illegal in India (Wireless Telegraphy Act 1933; DoT advisory 2022).

FIREWALL: comms never decide identity. matcher.py, tracker.py and rfrx.py must not import this
module (tests/test_relay.py enforces it).

Limits, stated honestly: the shared HMAC key proves membership, not which member sent a message;
a jammer that follows the hop schedule (knows the key) defeats hopping; ESP-NOW is a demo radio,
not a fielded one.
"""
import hashlib
import hmac
import json
from collections import defaultdict
from typing import NamedTuple, Optional

from turretx import wire

TYPES = ("txt", "pos", "sys", "rcpt")
ALERT, STATUS, CHAT = 0, 1, 2          # priorities: lower number goes first
WINDOW = 6            # recent delivery attempts the detector looks at
MIN_ATTEMPTS = 3
PDR_JAM = 0.3         # delivery ratio below this ...
NOISE_JAM = 0.5       # ... together with this noise level (0-1) means jamming, not a blocked path
RETRY_S = 0.5
JAM_BANNER_S = 3.0
DEFAULT_TTL_S = 30
QUEUE_MAX = 20


def hop_channel(key, k, n):
    """Channel for hop number k: a keyed pseudo-random sequence, identical on every node."""
    return int.from_bytes(hmac.new(key, f"hop{k}".encode(), hashlib.sha256).digest()[:4], "big") % n


def make_message(key, src, dst, seq, t, text, mtype="txt", prio=CHAT, ttl=DEFAULT_TTL_S):
    payload = json.dumps({"s": src, "d": dst, "n": seq, "t": round(t * 1000), "y": mtype,
                          "p": prio, "l": ttl, "m": text}, separators=(",", ":")).encode()
    return wire.make_packet(key, payload)


FIELDS = {"s", "d", "n", "t", "y", "p", "l", "m"}


def parse_message(payload):
    try:
        d = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict) or set(d) != FIELDS:
        return None
    if not (isinstance(d["s"], str) and isinstance(d["d"], str) and isinstance(d["m"], str)
            and len(d["m"]) <= 120 and d["y"] in TYPES
            and all(type(d[k]) is int for k in ("n", "t", "p", "l"))
            and 0 <= d["p"] <= 2 and 1 <= d["l"] <= 3600):
        return None
    return d


# ---------------------------------------------------------------------- delivery
class LinkResult(NamedTuple):
    ok: bool
    noise: float            # 0-1 channel energy seen while trying
    route: Optional[str]    # "direct" | "via drone" | "drone"


class SimLink:
    """Simulated shared RF channel with an optional barrier and jammer."""

    def __init__(self, channels=4):
        self.n_channels = channels
        self.channel = 0
        self.jam_channels = set()     # channels the (simulated) jammer sits on
        self.direct_blocked = False   # barrier between the ground units

    def set_channel(self, c):
        self.channel = c

    def transmit(self, packet, src, dst):
        if self.channel in self.jam_channels:
            return LinkResult(False, 0.9, None)
        if src == "drone":
            return LinkResult(True, 0.1, "drone")
        return LinkResult(True, 0.1, "via drone" if self.direct_blocked else "direct")


# ---------------------------------------------------------------------- protocol
class CommsNet:
    def __init__(self, key, nodes, link=None, channels=4, queue_max=QUEUE_MAX):
        self.key, self.nodes, self.queue_max = key, list(nodes), queue_max
        self.link = link or SimLink(channels)
        self.n_channels = self.link.n_channels
        self.hop_k, self.hops = 0, 0
        self.link.set_channel(hop_channel(key, 0, self.n_channels))
        self.log = []                  # dicts: t, src, dst, text, route, type, id, detail
        self.status = {}               # (src, seq) -> queued | delivered | receipted | expired | dropped | rejected
        self._pending = []
        self._seq = defaultdict(int)
        self._seen = {}                # (src, seq) -> time the entry may be forgotten
        self._attempts = []            # (delivered, noise) over the last WINDOW attempts
        self._banner_until = -1.0
        self._now = 0.0

    # ------------------------------------------------------------------ public API
    @property
    def channel(self):
        return self.link.channel

    @property
    def jam_suspected(self):
        return self._now < self._banner_until

    @property
    def queued(self):
        return len(self._pending)

    def send(self, t, src, dst, text, mtype="txt", prio=CHAT, ttl=DEFAULT_TTL_S):
        """Queue a message. Returns its id (src, seq)."""
        self._seq[src] += 1
        seq = self._seq[src]
        mid = (src, seq)
        entry = {"packet": make_message(self.key, src, dst, seq, t, text, mtype, prio, ttl), "id": mid,
                 "src": src, "dst": dst, "type": mtype, "prio": prio, "text": text, "created": t,
                 "ttl": ttl, "next": t, "tries": 0}
        if len(self._pending) >= self.queue_max:
            victim = max(self._pending + [entry], key=lambda e: (e["prio"], -e["created"], -e["id"][1]))
            self.status[victim["id"]] = "dropped"
            self._record(t, victim, "dropped", "queue full")
            if victim is entry:
                return mid
            self._pending.remove(victim)
        self._pending.append(entry)
        self.status[mid] = "queued"
        return mid

    def inject(self, t, packet):
        """A packet arriving at the drone from outside the queue (red team, or a real radio)."""
        ok, why, msg = self._accept(packet, t)
        if not ok:
            self._log(t, "?", "?", "", "rejected", "txt", None, why)
            return f"rejected: {why}"
        self._delivered(t, msg, "via drone")
        return "accepted: delivered via drone"

    def step(self, t):
        self._now = t
        self._seen = {k: v for k, v in self._seen.items() if v > t}
        # A hop restores the link, so the whole queue is retried at once, highest priority first.
        # One restart per step: if every channel is jammed we must not loop.
        for _ in range(2):
            hops = self.hops
            for e in sorted(self._pending, key=lambda e: (e["prio"], e["created"], e["id"][1])):
                if e["next"] > t or e not in self._pending:
                    continue
                self._attempt(t, e)
                if self.hops != hops:
                    break
            if self.hops == hops:
                break

    def _attempt(self, t, e):
        if t > e["created"] + e["ttl"]:
            self._pending.remove(e)
            self.status[e["id"]] = "expired"
            self._record(t, e, "expired", f"ttl {e['ttl']}s")
            return
        e["tries"] += 1
        res = self.link.transmit(e["packet"], e["src"], e["dst"])
        self._attempts = (self._attempts + [(res.ok, res.noise)])[-WINDOW:]
        if not res.ok:
            e["next"] = t + RETRY_S
            if e["tries"] == 1:
                self._record(t, e, "queued", "channel jammed")
            self._detect(t)
            return
        self._pending.remove(e)
        ok, why, msg = self._accept(e["packet"], t)
        if ok:
            self._delivered(t, msg, res.route)
        else:
            self.status[e["id"]] = "rejected"
            self._record(t, e, "rejected", why)

    # ------------------------------------------------------------------ internals
    def _accept(self, packet, t):
        if not wire.signature_ok(self.key, packet):
            return False, "bad signature", None
        msg = parse_message(packet["payload"])
        if msg is None:
            return False, "malformed", None
        if t - msg["t"] / 1000 > msg["l"]:
            return False, "expired", None
        mid = (msg["s"], msg["n"])
        if mid in self._seen:
            return False, "replay", None
        self._seen[mid] = msg["t"] / 1000 + msg["l"]
        return True, "", msg

    def _delivered(self, t, msg, route):
        mid = (msg["s"], msg["n"])
        self.status[mid] = "delivered"
        self._log(t, msg["s"], msg["d"], msg["m"], route, msg["y"], mid, "")
        if msg["y"] == "rcpt":
            orig_src, _, orig_n = msg["m"].partition(":")
            if orig_n.isdigit():
                self.status[(orig_src, int(orig_n))] = "receipted"
        elif msg["d"] in self.nodes:  # unicast to a ground unit: it answers with a receipt
            self.send(t, msg["d"], msg["s"], f"{msg['s']}:{msg['n']}", "rcpt", STATUS)

    def _detect(self, t):
        """Low delivery with a high noise floor is jamming; low delivery on a quiet channel is just
        a blocked path."""
        if len(self._attempts) < MIN_ATTEMPTS:
            return
        pdr = sum(ok for ok, _ in self._attempts) / len(self._attempts)
        noise = sum(n for _, n in self._attempts) / len(self._attempts)
        if pdr < PDR_JAM and noise > NOISE_JAM:
            self._banner_until = t + JAM_BANNER_S
            self._hop(t)

    def _hop(self, t):
        old = self.channel
        new = old
        while new == old:
            self.hop_k += 1
            new = hop_channel(self.key, self.hop_k, self.n_channels)
        self.link.set_channel(new)
        self.hops += 1
        self._attempts = []
        for e in self._pending:
            e["next"] = t
        self._log(t, "drone", "*", "", "hop", "sys", None, f"channel {old} -> {new}")

    def _record(self, t, e, route, detail=""):
        self._log(t, e["src"], e["dst"], e["text"], route, e["type"], e["id"], detail)

    def _log(self, t, src, dst, text, route, mtype, mid, detail):
        self.log.append({"t": t, "src": src, "dst": dst, "text": text, "route": route,
                         "type": mtype, "id": mid, "detail": detail})
