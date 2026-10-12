"""Glue between the failsafe, the RF simulation and the relay (RELAY_INTEGRATION.md §4, §5).

main.py owns one CommsHub. It does NOT feed identity: nothing here touches the matcher, and the
matcher never imports the relay (tests enforce it). The hub only

  * steps the relay each loop,
  * broadcasts a priority alert when the camera is lost / restored,
  * in the RF-only levels, broadcasts the blue-force picture (from authenticated claims) so ground
    units keep a picture the drone can no longer verify visually,
  * shares one simulated jam between the IFF link and the relay, and reports whether a hop restored
    the IFF link,
  * offers the demo hotkeys (barrier, canned message, forged and replayed relay packets).

SIMULATION ONLY: no RF is transmitted.
"""
from turretx import relay

DEMO_MESSAGES = [("F-1", "F-2", "need water"), ("F-2", "F-1", "contact east"),
                 ("F-1", "F-2", "moving to grid 4"), ("F-2", "F-1", "all clear")]
LOST_LEVELS = ("L2", "L3", "L4")


class CommsHub:
    def __init__(self, key, cfg):
        self.key = key
        self.net = relay.CommsNet(key, cfg["nodes"], channels=cfg["channels"])
        self.hop_restores_iff = cfg["hop_restores_iff"]
        self.pos_every_s = cfg["pos_every_s"]
        self.jam_on = False
        self._last_rung = None
        self._next_pos = 0.0
        self._scanned = 0
        self._demo_i = 0
        self._last_demo = None
        self._forged_n = 0

    # ------------------------------------------------------------------ per-loop
    def step(self, t, rung, last_claims):
        """Advance the relay. Returns [(kind, text)] events for the operator log."""
        prev, self._last_rung = self._last_rung, rung
        if prev is not None:
            if prev not in LOST_LEVELS and rung in LOST_LEVELS:
                self.net.send(t, "drone", "*", "CAMERA LOST - RF-only picture follows", "sys", relay.ALERT)
                self._next_pos = t
            elif prev != "L0" and rung == "L0":
                self.net.send(t, "drone", "*", "Camera restored - designation needs re-arm", "sys", relay.STATUS)
        if rung in ("L2", "L3") and t >= self._next_pos:
            self._next_pos = t + self.pos_every_s
            for rf_id, c in sorted(last_claims.items()):
                age = max(0.0, t - c["t"])
                self.net.send(t, "drone", "*", f"{rf_id} {c['x']:.0f},{c['y']:.0f} age {age:.0f}s",
                              "pos", relay.STATUS, ttl=10)
        self.net.step(t)
        return self._scan()

    def _scan(self):
        events = []
        for m in self.net.log[self._scanned:]:
            if m["route"] == "hop":
                events.append(("relay_hop", f"RELAY: jam detected, {m['detail']}"))
            elif m["route"] == "rejected":
                events.append(("relay_rejected", f"RELAY: rejected packet ({m['detail']})"))
            elif m["route"] in ("expired", "dropped"):
                events.append(("relay_drop", f"RELAY: {m['route']} '{m['text'][:24]}' ({m['detail']})"))
        self._scanned = len(self.net.log)
        return events

    # ------------------------------------------------------------------ simulated RF conditions
    def set_jam(self, on):
        self.jam_on = bool(on)
        self.net.link.jam_channels = {self.net.channel} if on else set()

    def iff_jammed(self):
        """Whether the IFF link is jammed. With hop_restores_iff, hopping off the jammed channel
        restores it; without, the jam stays on (jam-first demo)."""
        if not self.jam_on:
            return False
        return self.net.channel in self.net.link.jam_channels if self.hop_restores_iff else True

    def toggle_barrier(self):
        link = self.net.link
        link.direct_blocked = not link.direct_blocked
        return "BARRIER ON: ground units cannot reach each other directly" if link.direct_blocked \
            else "BARRIER OFF"

    # ------------------------------------------------------------------ demo hotkeys
    def send_demo(self, t):
        src, dst, text = DEMO_MESSAGES[self._demo_i % len(DEMO_MESSAGES)]
        self._demo_i += 1
        mid = self.net.send(t, src, dst, text)
        self._last_demo = mid
        return f"{src} -> {dst}: {text}"

    def forged(self, t):
        self._forged_n += 1
        pkt = relay.make_message(b"attacker-key", "F-1", "F-2", 9000 + self._forged_n, t, "forged order")
        return f"RED TEAM: forged relay message -> {self.net.inject(t, pkt)}"

    def replay(self, t):
        pkt = self.net.packets.get(self._last_demo)
        if pkt is None:
            return "RED TEAM: nothing to replay yet (send a message with n first)"
        return f"RED TEAM: replayed relay message -> {self.net.inject(t, pkt)}"

    def summary(self):
        return self.net.summary()
