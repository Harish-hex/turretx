"""RF simulator (design §5.4): friendly beacons, the red-team attacker, and the radio channel.

Friendly beacons answer each challenge with their OWN position: the tag locator's sticker fix
(the vehicle's GPS stand-in, R1) plus GPS-like noise, ~100 ms later. If the sticker is not
seen they dead-reckon for up to 1 s, then fall silent (comms loss → coasting).

Everything is driven by frame time `t` from the main loop; there are no threads, so recorded
clips replay identically at any speed (R3).
"""
import math
import random

from turretx import wire

DEAD_RECKON_S = 1.0
CLONE_KEEP_OFF_CM = 12   # the cloned emitter never "becomes" the real friendly
CLONE_FOLLOW_CM = 15     # max jump per frame when following the attacker's vehicle


def rssi_at(x_cm, y_cm, arena_cm, cfg_rf, rng):
    """Log-distance path loss from a virtual drone above the arena centre, in scenario metres (R9)."""
    r = cfg_rf["rssi"]
    scale = cfg_rf["scale_m_per_cm"]
    dx = (x_cm - arena_cm[0] / 2) * scale
    dy = (y_cm - arena_cm[1] / 2) * scale
    d = math.sqrt(dx * dx + dy * dy + r["drone_alt_m"] ** 2)
    return r["p0"] - 10 * r["n"] * math.log10(d) + rng.gauss(0, r["noise_db"])


class Channel:
    """Delivers packets after their latency; jamming drops everything on the air."""

    def __init__(self):
        self.queue = []          # (deliver_t, packet)
        self.jammed = False
        self.history = []        # (sent_t, packet) of genuine friendly replies, for replays

    def send(self, packet, deliver_t):
        self.queue.append((deliver_t, packet))

    def poll(self, t):
        due = [p for dt, p in self.queue if dt <= t]
        self.queue = [(dt, p) for dt, p in self.queue if dt > t]
        return [] if self.jammed else due


class FriendlyBeacon:
    def __init__(self, rf_id, key, cfg_rf, arena_cm, rng):
        self.rf_id, self.key, self.cfg, self.arena, self.rng = rf_id, key, cfg_rf, arena_cm, rng
        self.fix = None  # last visible TagFix

    def observe(self, fix):
        if fix.get("visible"):
            self.fix = fix

    def answer(self, challenge, t):
        """Signed reply packet to `challenge` at time t, or None if position unknown."""
        if self.fix is None or t - self.fix["t"] > DEAD_RECKON_S:
            return None
        dt = t - self.fix["t"]  # dead-reckon if the sticker wasn't seen this frame
        noise = self.cfg["pos_noise_cm"]
        x = self.fix["x"] + self.fix["vx"] * dt + self.rng.gauss(0, noise)
        y = self.fix["y"] + self.fix["vy"] * dt + self.rng.gauss(0, noise)
        vx = self.fix["vx"] + self.rng.gauss(0, 1.0)
        vy = self.fix["vy"] + self.rng.gauss(0, 1.0)
        rssi = rssi_at(x, y, self.arena, self.cfg, self.rng)
        payload = wire.encode_payload(self.rf_id, challenge["nonce"], t, x, y, vx, vy, rssi)
        return wire.make_packet(self.key, payload)


class Attacker:
    """Red-team actions (§5.4). One-shot: forged, replay. Toggles: clone, ghost."""

    def __init__(self, key, cfg_rf, arena_cm, rng, target_id="F-1"):
        self.key, self.cfg, self.arena, self.rng = key, cfg_rf, arena_cm, rng
        self.target_id = target_id
        self.clone_pos = None      # (x, y) of the vehicle the attacker sits on, followed by position
        self.ghost_pos = None
        self.pending = []          # one-shot packets to send at the next challenge

    def _payload(self, nonce, t, x, y, vx=0.0, vy=0.0):
        return wire.encode_payload(self.target_id, nonce, t, x, y, vx, vy,
                                   rssi_at(x, y, self.arena, self.cfg, self.rng))

    def forge(self, t, tracks):
        """No key: signs with a guessed key, claiming to be the friendly at some vehicle."""
        x, y = (tracks[0]["x"], tracks[0]["y"]) if tracks else (self.arena[0] / 2, self.arena[1] / 2)
        self.pending.append(("forged", x, y))
        return "RED TEAM: forged message injected (no key)"

    def replay(self, t, channel):
        """Re-sends a genuine friendly packet captured ≥1 s ago."""
        old = [p for st, p in channel.history if t - st >= 1.0]
        if not old:
            return "RED TEAM: replay - nothing captured yet"
        self.pending.append(("replay", old[-1]))
        return "RED TEAM: captured friendly message replayed"

    @staticmethod
    def _away_from(tracks, friendly_fix):
        if friendly_fix is None:
            return list(tracks)
        return [k for k in tracks if math.hypot(k["x"] - friendly_fix["x"],
                                                k["y"] - friendly_fix["y"]) > CLONE_KEEP_OFF_CM]

    def toggle_clone(self, tracks, friendly_fix):
        """Captured key: answer challenges as the friendly, from ANOTHER vehicle's position."""
        if self.clone_pos is not None:
            self.clone_pos = None
            return "RED TEAM: cloned ID stopped"
        others = self._away_from(tracks, friendly_fix)
        if friendly_fix is not None:  # start on the vehicle farthest from the real friendly
            others.sort(key=lambda k: -math.hypot(k["x"] - friendly_fix["x"], k["y"] - friendly_fix["y"]))
        if not others:
            return "RED TEAM: clone - no other vehicle to sit on"
        self.clone_pos = (others[0]["x"], others[0]["y"])
        return f"RED TEAM: cloned {self.target_id} ID on vehicle #{others[0]['track_id']}"

    def follow(self, tracks, friendly_fix):
        """Stay on the same PHYSICAL vehicle: nearest track to where it was, never the friendly.
        (Following a track ID breaks when the tracker swaps IDs in a crossing.)"""
        if self.clone_pos is None:
            return None
        cands = self._away_from(tracks, friendly_fix)
        if not cands:
            return None
        k = min(cands, key=lambda k: math.hypot(k["x"] - self.clone_pos[0], k["y"] - self.clone_pos[1]))
        if math.hypot(k["x"] - self.clone_pos[0], k["y"] - self.clone_pos[1]) > CLONE_FOLLOW_CM:
            return None  # lost it this frame (e.g. merged with another car); keep last position
        self.clone_pos = (k["x"], k["y"])
        return k

    def toggle_ghost(self, tracks):
        """Captured key: claim the friendly ID at an empty spot (decoy)."""
        if self.ghost_pos is not None:
            self.ghost_pos = None
            return "RED TEAM: ghost ID stopped"
        best, best_d = None, -1.0
        for gx in range(1, 10):
            for gy in range(1, 6):
                p = (self.arena[0] * gx / 10, self.arena[1] * gy / 6)
                d = min((math.hypot(p[0] - k["x"], p[1] - k["y"]) for k in tracks), default=1e9)
                if d > best_d:
                    best, best_d = p, d
        self.ghost_pos = best
        return f"RED TEAM: ghost {self.target_id} at empty spot ({best[0]:.0f}, {best[1]:.0f}) cm"

    def answer(self, challenge, t, tracks, friendly_fix=None):
        packets = []
        for item in self.pending:
            if item[0] == "forged":
                payload = self._payload(challenge["nonce"], t, item[1], item[2])
                packets.append({"payload": payload, "sig": wire.sign(b"guessed-key", payload)})
            elif item[0] == "replay":
                packets.append(item[1])
        self.pending = []
        if self.clone_pos is not None:
            k = self.follow(tracks, friendly_fix)
            if k is not None:
                packets.append(wire.make_packet(self.key, self._payload(
                    challenge["nonce"], t, k["x"], k["y"], k["vx"], k["vy"])))
        if self.ghost_pos is not None:
            packets.append(wire.make_packet(self.key, self._payload(
                challenge["nonce"], t, *self.ghost_pos)))
        return packets


class RFSim:
    """Owns the friendly beacons, the attacker and the channel. Main loop, every frame:
         sim.observe(t, tag_fixes, tracks)
         if (ch := interrogator.poll(t)): sim.on_challenge(ch, t)
         claims, rejections = receiver.receive(sim.poll(t), t)
    """

    def __init__(self, cfg, seed=None):
        rf = dict(cfg["rf"], scale_m_per_cm=cfg["scale_m_per_cm"])
        self.latency = rf["latency_ms"] / 1000
        key = wire.rf_key(rf["key"])
        rng = random.Random(seed)
        arena = cfg["table_cm"]
        self.beacons = {rf_id: FriendlyBeacon(rf_id, key, rf, arena, rng) for rf_id in cfg["tags"]}
        self.attacker = Attacker(key, rf, arena, rng, target_id=next(iter(cfg["tags"]), "F-1"))
        self.channel = Channel()
        self.tracks = []

    def observe(self, t, tag_fixes, tracks):
        for fix in tag_fixes:
            if fix["rf_id"] in self.beacons:
                self.beacons[fix["rf_id"]].observe(fix)
        self.tracks = tracks  # only the attacker uses these (its own vehicle's position)
        self.attacker.follow(tracks, self._friendly_fix())

    def _friendly_fix(self):
        b = self.beacons.get(self.attacker.target_id)
        return b.fix if b else None

    def on_challenge(self, challenge, t):
        for beacon in self.beacons.values():
            pkt = beacon.answer(challenge, t)
            if pkt is not None:
                self.channel.send(pkt, t + self.latency)
                self.channel.history.append((t, pkt))
        self.channel.history = self.channel.history[-100:]
        for pkt in self.attacker.answer(challenge, t, self.tracks, self._friendly_fix()):
            self.channel.send(pkt, t + self.latency)

    def poll(self, t):
        return self.channel.poll(t)

    # red-team hotkeys → banner text
    def red_team(self, action, t):
        a = self.attacker
        if action == "forged":
            return a.forge(t, self.tracks)
        if action == "replay":
            return a.replay(t, self.channel)
        if action == "clone":
            return a.toggle_clone(self.tracks, self._friendly_fix())
        if action == "ghost":
            return a.toggle_ghost(self.tracks)
        if action == "jam":
            self.channel.jammed = not self.channel.jammed
            return "RED TEAM: RF JAMMING ON" if self.channel.jammed else "RED TEAM: jamming off"
        raise ValueError(action)
