"""Drone side of the RF link (design §5.4–5.5): issue challenges, verify replies → claims.

Every received packet is treated as hostile. Checks, in order:
  1. size limit                    → malformed
  2. HMAC over the raw bytes       → bad_signature   (before any parsing)
  3. schema / ranges               → malformed
  4. nonce is one of the last 3    → stale_nonce     (freshness; no clock sync needed, R15)
  5. exact byte duplicate          → replayed
  6. a second, different valid reply for the same (nonce, rf_id) → both flagged conflict (R5)
This module never imports the simulator or the tag locator: it only sees packets.
"""
import secrets
from collections import deque

from turretx import wire

NONCE_WINDOW = 3          # challenges a reply may answer (0.6 s at 5 Hz)
CONFLICT_HOLD_S = 1.0     # keep flagging an rf_id this long after its last conflicting round


class Interrogator:
    """Issues a fresh random challenge nonce at `challenge_hz`, driven by frame time."""

    def __init__(self, challenge_hz):
        self.period = 1.0 / challenge_hz
        self.next_t = None
        self.recent = deque(maxlen=NONCE_WINDOW)

    def poll(self, t):
        """A new challenge {"nonce", "t"} if one is due at time t, else None."""
        if self.next_t is not None and t < self.next_t:
            return None
        self.next_t = t + self.period if self.next_t is None else max(self.next_t + self.period, t)
        nonce = secrets.token_hex(8)
        self.recent.append(nonce)
        return {"nonce": nonce, "t": t}

    def is_current(self, nonce):
        return nonce in self.recent


class Receiver:
    def __init__(self, key, interrogator):
        self.key = key
        self.interrogator = interrogator
        self.seen_payloads = deque(maxlen=200)
        self.replies = {}            # (nonce, rf_id) → first accepted claim
        self.conflict_until = {}     # rf_id → t

    def receive(self, packets, t):
        """(claims, rejections). Claims follow contract §6 plus "nonce"; rejections are
        {"t", "reason", "rf_id"} (rf_id only once the packet is authenticated)."""
        claims, rejections = [], []
        for pkt in packets:
            payload = pkt.get("payload")
            if not isinstance(payload, bytes) or len(payload) > wire.MAX_PAYLOAD_BYTES:
                rejections.append({"t": t, "reason": "malformed", "rf_id": None})
                continue
            if not wire.signature_ok(self.key, pkt):
                rejections.append({"t": t, "reason": "bad_signature", "rf_id": None})
                continue
            claim = wire.decode_payload(payload)
            if claim is None:
                rejections.append({"t": t, "reason": "malformed", "rf_id": None})
                continue
            # hardware beacons report their own WiFi RSSI, not the drone's view (udplink.py)
            claim["rssi_known"] = pkt.get("rssi_known", True)
            if not self.interrogator.is_current(claim["nonce"]):
                rejections.append({"t": t, "reason": "stale_nonce", "rf_id": claim["rf_id"]})
                continue
            if payload in self.seen_payloads:
                rejections.append({"t": t, "reason": "replayed", "rf_id": claim["rf_id"]})
                continue
            self.seen_payloads.append(payload)

            key = (claim["nonce"], claim["rf_id"])
            first = self.replies.get(key)
            if first is None:
                self.replies[key] = claim
            else:  # two different valid answers to one challenge under one identity: clone
                self.conflict_until[claim["rf_id"]] = t + CONFLICT_HOLD_S
                first["conflict"] = True
            claim["conflict"] = t < self.conflict_until.get(claim["rf_id"], -1.0)
            claims.append(claim)

        current = set(self.interrogator.recent)
        self.replies = {k: v for k, v in self.replies.items() if k[0] in current}
        return claims, rejections
