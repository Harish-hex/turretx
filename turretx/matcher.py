"""Matching engine (design §5.6, §5.6a): links verified RF claims to camera tracks and labels
every vehicle FRIENDLY / UNKNOWN / SUSPICIOUS with a reason and explainable scores.

Never imports the tag locator or the simulator: it only sees tracks (camera) and claims (radio).
Never outputs "hostile": unidentified is UNKNOWN, contradictory is SUSPICIOUS (verify).
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from turretx.kinematics import STATIONARY_CM_S
from turretx.rfmodel import expected_rssi

GATED = 1e6          # finite "impossible" cost: scipy rejects inf (R10)
LOST_KEEP_S = 60.0   # how long a lost friendly's last-known marker is shown

FRIENDLY, UNKNOWN, SUSPICIOUS = "FRIENDLY", "UNKNOWN", "SUSPICIOUS"


class _Track:
    def __init__(self, t):
        self.state, self.reason, self.rf_id = UNKNOWN, "NO_ID", None
        self.pending, self.pending_since = None, None
        self.last_friendly_t = None
        self.confidence = 0.0
        self.scores = {"P": 0.0, "M": 0.0, "R": 0.0}
        self.flags = []
        self.last = None  # last Track dict seen


class Matcher:
    def __init__(self, cfg):
        self.c = cfg["matcher"]
        self.arena = cfg["table_cm"]
        self.scale = cfg["scale_m_per_cm"]
        self.rssi_cfg = cfg["rf"]["rssi"]
        self.claims = {}         # rf_id → claims of its latest radio round
        self.rf_last_rx = {}     # rf_id → receive time of its latest claim
        self.last_rx = None      # receive time of the latest claim of any id
        self.smooth = {}         # (rf_id, track_id) → {"P","M","R","t"}
        self.tracks = {}         # track_id → _Track
        self.ghost_since = {}    # rf_id → t
        self.lost = []           # last-known positions of friendlies whose tracks vanished
        self.rssi_known = {}     # rf_id → False for hardware beacons (their RSSI isn't the drone's view)

    # ---------------------------------------------------------------- scoring
    def _instant(self, claim, track):
        """Instantaneous (P, M, R, informative, low_info) for one pair, or None if gated."""
        c = self.c
        dt = track["t"] - claim["t"]  # R2: propagate the claim to the frame's time
        cx, cy = claim["x"] + claim["vx"] * dt, claim["y"] + claim["vy"] * dt
        d = math.hypot(cx - track["x"], cy - track["y"])
        if d > c["gate_cm"]:
            return None
        P = math.exp(-d * d / (2 * c["sigma_pos_cm"] ** 2))

        s_claim = math.hypot(claim["vx"], claim["vy"])
        s_track = track["speed"]
        ds = s_claim - s_track
        M = math.exp(-ds * ds / (2 * c["sigma_speed"] ** 2))
        heading_on = s_claim > c["heading_min_speed"] and s_track > c["heading_min_speed"]
        if heading_on:  # R8: heading is noise at low speed
            dth = math.atan2(claim["vy"], claim["vx"]) - track["heading"]
            M *= (1 + math.cos(dth)) / 2
        low_info = s_claim < STATIONARY_CM_S and s_track < STATIONARY_CM_S
        # a track whose velocity history was just reset (merge/split/ID swap) has no motion
        # evidence yet: treat like low_info, never as a mismatch (Phase 3 end-to-end finding)
        unsettled = not track.get("settled", True)
        if low_info or unsettled:
            M = 0.5
        informative = (heading_on or abs(ds) > 2 * c["sigma_speed"]) and not unsettled

        R = 0.0  # no evidence (shown as 0, left out of the confidence by _combined)
        if claim.get("rssi_known", True):
            exp_rssi = expected_rssi(track["x"], track["y"], self.arena, self.scale, self.rssi_cfg)
            R = math.exp(-(claim["rssi"] - exp_rssi) ** 2 / (2 * c["sigma_rssi_db"] ** 2))
        return P, M, R, informative, low_info

    def _combined(self, s, rid):
        w = self.c["weights"]
        if not self.rssi_known.get(rid, True):  # no RSSI evidence: P and M carry the weight
            return (w["P"] * s["P"] + w["M"] * s["M"]) / (w["P"] + w["M"])
        return w["P"] * s["P"] + w["M"] * s["M"] + w["R"] * s["R"]

    def _smoothed(self, key, P, M, R, t):
        """Time-based EMA (frame-rate independent): alpha = 1 - exp(-dt / tau)."""
        s = self.smooth.get(key)
        if s is None:
            s = self.smooth[key] = {"P": P, "M": M, "R": R, "t": t}
            return s
        a = 1 - math.exp(-max(t - s["t"], 0.0) / self.c["ema_tau_s"])
        for k, v in (("P", P), ("M", M), ("R", R)):
            s[k] += a * (v - s[k])
        s["t"] = t
        return s

    # ---------------------------------------------------------------- main step
    def update(self, t, tracks, new_claims):
        c = self.c
        tracks = sorted(tracks, key=lambda k: k["track_id"])  # deterministic, order-invariant
        for cl in sorted(new_claims, key=lambda q: (q["rf_id"], q["t"], q["x"], q["y"])):
            rid = cl["rf_id"]
            cur = self.claims.get(rid)
            if not cur or cl["t"] > cur[0]["t"] + 1e-6:
                self.claims[rid] = [cl]
            elif abs(cl["t"] - cur[0]["t"]) <= 1e-6:
                cur.append(cl)
            self.rf_last_rx[rid] = t
            self.last_rx = t
            self.rssi_known[rid] = cl.get("rssi_known", True)

        active = [cl for rid, lst in sorted(self.claims.items())
                  if t - lst[0]["t"] <= c["claim_max_age_s"] for cl in lst]
        active_ids = {cl["rf_id"] for cl in active}
        conflict_ids = {cl["rf_id"] for cl in active if cl.get("conflict")}

        # 1) instantaneous scores per (claim, track), smoothed per (rf_id, track)
        inst = {}  # (claim index, track_id) → tuple
        for i, cl in enumerate(active):
            for k in tracks:
                r = self._instant(cl, k)
                if r is not None:
                    inst[(i, k["track_id"])] = r
        smooth_now = {}
        for rid in active_ids:
            rows = [i for i, cl in enumerate(active) if cl["rf_id"] == rid]
            for k in tracks:
                cands = [inst[(i, k["track_id"])] for i in rows if (i, k["track_id"]) in inst]
                if cands:
                    P, M, R, _, _ = max(cands, key=lambda r: r[0])
                else:
                    P, M, R = 0.0, 0.0, 0.0
                smooth_now[(rid, k["track_id"])] = self._smoothed((rid, k["track_id"]), P, M, R, t)

        # 2) assignment: rows = active claims, cols = tracks, gated pairs impossible
        assigned = {}  # track_id → (claim index, smoothed scores)
        if active and tracks:
            cost = np.full((len(active), len(tracks)), GATED)
            for i, cl in enumerate(active):
                for j, k in enumerate(tracks):
                    if (i, k["track_id"]) in inst:
                        cost[i, j] = 1 - self._combined(smooth_now[(cl["rf_id"], k["track_id"])], cl["rf_id"])
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < GATED:
                    tid = tracks[j]["track_id"]
                    assigned[tid] = (i, smooth_now[(active[i]["rf_id"], tid)])

        # 3) ambiguity: a single-claim id whose runner-up track is nearly as good → TOO_CLOSE
        ambiguous = set()
        for tid, (i, s) in assigned.items():
            rid = active[i]["rf_id"]
            if rid in conflict_ids:
                continue
            best = self._combined(s, rid)
            for k in tracks:
                o = k["track_id"]
                if o == tid or (i, o) not in inst:
                    continue
                r = self._combined(smooth_now[(rid, o)], rid)
                if r >= c["ambiguity_floor"] and best - r <= c["ambiguity_margin"]:
                    ambiguous.update({tid, o})

        # 4) ghosts: active claims with no vehicle inside the gate, for longer than the grace
        ghost_claims = {rid: [cl for i, cl in enumerate(active) if cl["rf_id"] == rid
                              and not any((i, k["track_id"]) in inst for k in tracks)]
                        for rid in active_ids}
        ghosts = []
        for rid, cls in ghost_claims.items():
            if cls:
                self.ghost_since.setdefault(rid, t)
                if t - self.ghost_since[rid] >= c["ghost_grace_s"]:
                    ghosts += cls
            else:
                self.ghost_since.pop(rid, None)
        for rid in list(self.ghost_since):
            if rid not in active_ids:
                del self.ghost_since[rid]

        # 5) per-track raw state, custody/coasting, hysteresis
        seen = set()
        positions = {k["track_id"]: (k["x"], k["y"]) for k in tracks}
        for k in tracks:
            tid = k["track_id"]
            seen.add(tid)
            st = self.tracks.setdefault(tid, _Track(t))
            st.last = k
            raw, reason, rid, conf, scores, flags = UNKNOWN, "NO_ID", None, 0.0, None, []
            if tid in assigned:
                i, s = assigned[tid]
                rid = active[i]["rf_id"]
                _, _, _, informative, low_info = inst[(i, tid)]
                conf = self._combined(s, rid)
                scores = {"P": s["P"], "M": s["M"], "R": s["R"]}
                if low_info:
                    flags.append("low_info")
                if rid in conflict_ids:
                    raw, reason = SUSPICIOUS, "ID_CONFLICT"
                elif tid in ambiguous:
                    raw, reason = UNKNOWN, "TOO_CLOSE"
                elif conf >= c["friendly_threshold"]:
                    raw, reason = FRIENDLY, None
                elif informative and s["M"] < c["mismatch_threshold"]:
                    raw, reason = SUSPICIOUS, "MOTION_MISMATCH"
            elif tid in ambiguous:
                raw, reason = UNKNOWN, "TOO_CLOSE"
            if k.get("merged") and reason != "ID_CONFLICT":
                # one box holding two touching vehicles: identity can't be pinned on either
                # (an ID conflict still shows: it's a security alert)
                raw, reason = UNKNOWN, "TOO_CLOSE"

            # track custody: a silent friendly keeps its identity while custody holds (§5.6a)
            coasting = (st.state == FRIENDLY and raw == UNKNOWN and reason == "NO_ID"
                        and st.rf_id is not None and st.rf_id not in active_ids)
            if coasting:
                coast_s = t - self.rf_last_rx.get(st.rf_id, t)
                crowded = any(math.hypot(x - k["x"], y - k["y"]) < c["custody_radius_cm"]
                              for o, (x, y) in positions.items() if o != tid)
                if crowded or coast_s > c["coast_max_s"]:
                    st.state, st.reason, st.rf_id = UNKNOWN, "NO_ID", None  # immediately
                    st.pending = None
                else:
                    st.reason = "COASTING"
                    st.flags = ["coasting"]
                    st.coast_s = coast_s
                    st.last_friendly_t = t
                    continue
            st.coast_s = None

            if scores is not None:
                st.scores = scores
            st.flags = flags
            if raw == st.state:
                st.reason, st.pending = reason, None
                if rid is not None:
                    st.rf_id, st.confidence = rid, conf
            elif raw == SUSPICIOUS and reason == "ID_CONFLICT":  # security alert: no wait
                st.state, st.reason, st.rf_id, st.confidence, st.pending = raw, reason, rid, conf, None
            else:
                if st.pending != raw:
                    st.pending, st.pending_since = raw, t
                hold = c["exit_s"] if st.state in (FRIENDLY, SUSPICIOUS) else c["enter_s"]
                if t - st.pending_since >= hold - 1e-9:
                    st.state, st.reason, st.pending = raw, reason, None
                    st.rf_id, st.confidence = rid, conf
            if st.state == FRIENDLY:
                st.last_friendly_t = t

        # 6) vanished tracks: remember friendlies' last-known positions
        for tid in [o for o in self.tracks if o not in seen]:
            st = self.tracks.pop(tid)
            if st.state == FRIENDLY and st.last is not None:
                self.lost = [l for l in self.lost if l["rf_id"] != st.rf_id]
                self.lost.append({"rf_id": st.rf_id, "x": st.last["x"], "y": st.last["y"],
                                  "vx": st.last["vx"], "vy": st.last["vy"], "t_lost": st.last["t"]})
        relocated = {st.rf_id for st in self.tracks.values() if st.state == FRIENDLY}
        self.lost = [l for l in self.lost if t - l["t_lost"] <= LOST_KEEP_S and l["rf_id"] not in relocated]
        for rid in list(self.smooth):
            if rid[1] not in seen:
                del self.smooth[rid]

        labels = {}
        for tid, st in self.tracks.items():
            labels[tid] = {"track_id": tid, "state": st.state,
                           "reason": st.reason if st.state != FRIENDLY or st.reason == "COASTING" else None,
                           "rf_id": st.rf_id if st.state != UNKNOWN else None,
                           "confidence": st.confidence, "scores": dict(st.scores),
                           "flags": list(st.flags), "coast_s": st.coast_s,
                           "last_friendly_t": st.last_friendly_t}
        return {
            "labels": labels,
            "ghosts": ghosts,
            "rf_denied": self.last_rx is not None and t - self.last_rx > c["rf_denied_s"],
            "lost_friendlies": [dict(l, age_s=t - l["t_lost"]) for l in self.lost],
        }
