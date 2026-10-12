"""Operator display (design §5.7): video panel + side panel (mini-map, selected vehicle, events).

Colour + icon + text for every state (colour-blind safe, readable on a projector). No red is
used for any label: nothing on screen should read as "enemy" (R6, R21).
"""
import math

import cv2
import numpy as np

GREEN = (80, 200, 60)
AMBER = (0, 190, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DARK = (32, 32, 32)
STATE_COLOUR = {"FRIENDLY": GREEN, "UNKNOWN": AMBER, "SUSPICIOUS": MAGENTA}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale, colour, thick=2, outline=True):
    org = (int(org[0]), int(org[1]))
    if outline:
        cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def icon(img, state, centre, r, colour):
    """✓ friendly, ? unknown, ! in a triangle for suspicious (drawn, not font glyphs)."""
    x, y = int(centre[0]), int(centre[1])
    cv2.circle(img, (x, y), r, (0, 0, 0), -1)
    if state == "FRIENDLY":
        pts = np.array([[x - r // 2, y], [x - r // 8, y + r // 2], [x + r // 2, y - r // 2]], np.int32)
        cv2.polylines(img, [pts], False, colour, max(2, r // 4), cv2.LINE_AA)
    elif state == "SUSPICIOUS":
        tri = np.array([[x, y - r + 2], [x - r + 2, y + r - 3], [x + r - 2, y + r - 3]], np.int32)
        cv2.polylines(img, [tri], True, colour, 2, cv2.LINE_AA)
        cv2.line(img, (x, y - r // 3), (x, y + r // 5), colour, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y + r // 2), 2, colour, -1)
    else:
        text(img, "?", (x - r // 2 + 1, y + r // 2), r / 14, colour, 2, outline=False)


def dashed_rect(img, p1, p2, colour, thick, dash=12):
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(int(math.hypot(b[0] - a[0], b[1] - a[1]) // dash), 1)
        for i in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
            e = (int(a[0] + (b[0] - a[0]) * min(i + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(i + 1, n) / n))
            cv2.line(img, s, e, colour, thick)


def short_label(lab):
    st, why, rid = lab["state"], lab["reason"], lab["rf_id"]
    if st == "FRIENDLY":
        if why == "COASTING":
            return f"{rid} NO RF {lab['coast_s']:.0f}s"
        return f"{rid} {lab['confidence']:.2f}"
    if st == "SUSPICIOUS":
        return "VERIFY: ID CONFLICT" if why == "ID_CONFLICT" else "VERIFY: MOTION"
    return "TOO CLOSE" if why == "TOO_CLOSE" else "NO ID"


class Display:
    def __init__(self, cfg, homo):
        self.homo = homo
        self.scale_m = cfg["scale_m_per_cm"]

    # ---------------------------------------------------------------- video panel
    def _video(self, vis, view, s):
        homo, out = self.homo, view["out"]
        labels = out["labels"]
        for k in view["tracks"]:
            lab = labels.get(k["track_id"])
            if lab is None:
                continue
            colour = STATE_COLOUR[lab["state"]]
            x1, y1, x2, y2 = (int(v) for v in k["bbox"])
            thick = max(2, int(3 * s)) + (2 if lab["state"] == "SUSPICIOUS" else 0)
            if lab["reason"] == "COASTING":
                dashed_rect(vis, (x1, y1), (x2, y2), colour, thick)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thick)
            if k["track_id"] == view.get("selected"):
                cv2.rectangle(vis, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), WHITE, 1)
            r = int(13 * s)
            icon(vis, lab["state"], (x1 + r, y1 - r - 4), r, colour)
            text(vis, short_label(lab), (x1 + 2 * r + 6, y1 - 8), 0.7 * s, colour)
        for c in view["claims"]:
            px, py = homo.to_px([(c["x"], c["y"])])[0]
            colour = MAGENTA if c["conflict"] else CYAN
            cv2.circle(vis, (int(px), int(py)), int(16 * s), colour, 2, cv2.LINE_AA)
        for g in out["ghosts"]:
            px, py = homo.to_px([(g["x"], g["y"])])[0]
            cv2.circle(vis, (int(px), int(py)), int(26 * s), MAGENTA, 3, cv2.LINE_AA)
            text(vis, f"GHOST {g['rf_id']}", (px + 30 * s, py + 6), 0.7 * s, MAGENTA)

    def _rf_only(self, vis, view, s):
        """Camera lost: replace the video with a schematic arena of authenticated friendlies only."""
        h, w = vis.shape[:2]
        vis[:] = (30, 30, 30)
        aw, ah = self.homo.w, self.homo.h
        k = min(w * 0.86 / aw, h * 0.72 / ah)
        x0, y0 = int((w - aw * k) / 2), int(h * 0.2)

        def P(x, y):
            return int(x0 + x * k), int(y0 + y * k)

        cv2.rectangle(vis, P(0, 0), P(aw, ah), GREY, 2)
        text(vis, "RF-ONLY PICTURE: authenticated friendlies only", (x0, y0 - int(14 * s)), 0.8 * s, WHITE, 2, False)
        text(vis, "Anything not transmitting is not shown. Absence of a marker does not mean safe.",
             (x0, y0 + int(ah * k) + int(34 * s)), 0.6 * s, AMBER, 1, False)
        for rf_id, c in view["last_claims"].items():
            age = max(0.0, view["t"] - c["t"])
            fresh = age < 1.0
            colour = MAGENTA if c.get("conflict") else (GREEN if fresh else AMBER)
            px, py = P(c["x"] + c["vx"] * min(age, 5), c["y"] + c["vy"] * min(age, 5))
            ring = max(int((5 + 3 * age) * k), int(10 * s))  # no-fire ring grows while the beacon is silent
            cv2.circle(vis, (px, py), ring, colour, 2 if not fresh else 1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), int(9 * s), colour, -1, cv2.LINE_AA)
            tail = "live" if fresh else f"LOST FRIENDLY - last beacon {age:.0f}s ago"
            text(vis, f"{rf_id}  {tail}", (px + int(14 * s), py - int(12 * s)), 0.7 * s, colour, 2)

    # ---------------------------------------------------------------- side panel
    def _map(self, panel, view, x0, y0, w, s):
        arena_w, arena_h = self.homo.w, self.homo.h
        h = int(w * arena_h / arena_w)
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), -1)
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), GREY, 1)

        def P(x, y):
            return int(x0 + x / arena_w * w), int(y0 + y / arena_h * h)

        km = arena_w * self.scale_m / 1000
        text(panel, f"0 - {km:.1f} km", (x0, y0 + h + 22 * s), 0.5 * s, GREY, 1, False)
        out = view["out"]
        for lost in out["lost_friendlies"]:
            age = lost["age_s"]
            px = lost["x"] + lost["vx"] * min(age, 5)
            py = lost["y"] + lost["vy"] * min(age, 5)
            rad = int((5 + 3 * age) / arena_w * w)
            fade = max(0.3, 1 - age / 60)
            col = tuple(int(c * fade) for c in GREEN)
            cv2.circle(panel, P(px, py), max(rad, 4), col, 1, cv2.LINE_AA)
            text(panel, f"{lost['rf_id']} last seen {age:.0f}s", (P(px, py)[0] + 6, P(px, py)[1] - 6),
                 0.42 * s, col, 1, False)
        claims_by_id = {}
        for c in view["claims"]:
            claims_by_id.setdefault(c["rf_id"], []).append(c)
            cv2.circle(panel, P(c["x"], c["y"]), int(7 * s), MAGENTA if c["conflict"] else CYAN, 1, cv2.LINE_AA)
        for g in out["ghosts"]:
            cv2.circle(panel, P(g["x"], g["y"]), int(10 * s), MAGENTA, 2, cv2.LINE_AA)
        conflict_pts = {}
        for k in view["tracks"]:
            lab = out["labels"].get(k["track_id"])
            if lab is None:
                continue
            colour = STATE_COLOUR[lab["state"]]
            p = P(k["x"], k["y"])
            for c in claims_by_id.get(lab["rf_id"], []) if lab["rf_id"] else []:
                cv2.line(panel, p, P(c["x"], c["y"]), colour, 1, cv2.LINE_AA)
            if lab["reason"] == "ID_CONFLICT":
                conflict_pts.setdefault(lab["rf_id"], []).append(p)
            cv2.circle(panel, p, int(6 * s), colour, -1, cv2.LINE_AA)
            if k["speed"] > 2:
                tip = P(k["x"] + k["vx"] * 0.8, k["y"] + k["vy"] * 0.8)
                cv2.arrowedLine(panel, p, tip, colour, 1, cv2.LINE_AA, tipLength=0.3)
            text(panel, f"#{k['track_id']}", (p[0] + 8, p[1] + 4), 0.45 * s, colour, 1, False)
        for pts in conflict_pts.values():
            for a, b in zip(pts, pts[1:]):
                cv2.line(panel, a, b, MAGENTA, 2, cv2.LINE_AA)
        return y0 + h + int(30 * s)

    def _selected(self, panel, view, x0, y, w, s):
        lab = view["out"]["labels"].get(view.get("selected"))
        if lab is None:
            text(panel, "Click a vehicle (or Tab) for details", (x0, y + 20 * s), 0.5 * s, GREY, 1, False)
            return y + int(40 * s)
        colour = STATE_COLOUR[lab["state"]]
        head = f"#{lab['track_id']}" + (f"  {lab['rf_id']}" if lab["rf_id"] else "")
        text(panel, head, (x0, y + 24 * s), 0.75 * s, WHITE, 2, False)
        state = lab["state"] + (f" - {lab['reason'].replace('_', ' ')}" if lab["reason"] else "")
        text(panel, state, (x0, y + 52 * s), 0.6 * s, colour, 2, False)
        text(panel, f"for {view['time_in_state'](lab['track_id']):.1f} s", (x0, y + 74 * s), 0.48 * s, GREY, 1, False)
        yy = y + int(90 * s)
        rows = [("P position", lab["scores"]["P"]), ("M motion", lab["scores"]["M"]),
                ("R signal", lab["scores"]["R"]), ("Confidence", lab["confidence"])]
        for name, v in rows:
            text(panel, name, (x0, yy + 14 * s), 0.48 * s, WHITE, 1, False)
            bx = x0 + int(115 * s)
            bw = w - int(165 * s)
            cv2.rectangle(panel, (bx, yy), (bx + bw, yy + int(16 * s)), (70, 70, 70), -1)
            cv2.rectangle(panel, (bx, yy), (bx + int(bw * max(0, min(v, 1))), yy + int(16 * s)), colour, -1)
            text(panel, f"{v:.2f}", (bx + bw + 6, yy + 14 * s), 0.48 * s, WHITE, 1, False)
            yy += int(24 * s)
        if lab["flags"]:
            text(panel, "flags: " + ", ".join(lab["flags"]), (x0, yy + 14 * s), 0.45 * s, GREY, 1, False)
            yy += int(22 * s)
        return yy + int(10 * s)

    def render(self, frame, view):
        """frame + side panel. view: t, fps, latency_ms, backend, tracks, out (matcher), claims,
        rejected, jammed, selected, banners [(text, colour)], events, time_in_state(tid)."""
        h, w = frame.shape[:2]
        s = h / 1080 * 1.4
        vis = frame.copy()
        if view.get("rung", "L0") in ("L2", "L3", "L4"):
            self._rf_only(vis, view, s)
        else:
            self._video(vis, view, s)

        banners = list(view["banners"])
        if view["out"]["rf_denied"]:
            banners.insert(0, ("RF DENIED: identification degraded. Unidentified != hostile.", MAGENTA))
        for i, (msg, colour) in enumerate(banners[:3]):
            y = int((14 + 44 * i) * s)
            cv2.rectangle(vis, (0, y), (w, y + int(38 * s)), (0, 0, 0), -1)
            text(vis, msg, (int(14 * s), y + int(28 * s)), 0.8 * s, colour, 2, False)

        pw = int(h * 0.42)
        panel = np.full((h, pw, 3), DARK, np.uint8)
        x0, inner = int(16 * s), pw - int(32 * s)
        text(panel, "TURRETX  air-to-ground ID", (x0, 34 * s), 0.7 * s, WHITE, 2, False)
        rf = "JAMMED" if view["jammed"] else ("DENIED" if view["out"]["rf_denied"] else "OK")
        lat = f"  lat {view['latency_ms']:.0f}ms" if view["latency_ms"] is not None else ""
        text(panel, f"{view['fps']:.0f} fps{lat}  [{view['backend']}]", (x0, 62 * s), 0.5 * s, GREY, 1, False)
        text(panel, f"RF {rf}   rejected {view['rejected']}", (x0, 86 * s), 0.5 * s,
             MAGENTA if rf != "OK" else GREY, 1, False)
        y = self._map(panel, view, x0, int(102 * s), inner, s)
        y = self._selected(panel, view, x0, y, inner, s)

        legend = [("FRIENDLY", GREEN, "verified ID matches"), ("UNKNOWN", AMBER, "no / unclear ID"),
                  ("SUSPICIOUS", MAGENTA, "conflict - VERIFY")]
        for name, colour, why in legend:
            icon(panel, name, (x0 + 10 * s, y + 10 * s), int(10 * s), colour)
            text(panel, f"{name}  {why}", (x0 + 28 * s, y + 16 * s), 0.45 * s, colour, 1, False)
            y += int(26 * s)
        y += int(8 * s)
        text(panel, "Events", (x0, y + 14 * s), 0.5 * s, WHITE, 1, False)
        y += int(22 * s)
        for et, kind, msg in reversed(view["events"]):
            colour = MAGENTA if ("SUSPICIOUS" in msg or kind in ("rf_rejected", "rf_denied", "ghost", "redteam")) else GREY
            text(panel, f"{et:6.1f}  {msg}"[:52], (x0, y + 14 * s), 0.42 * s, colour, 1, False)
            y += int(20 * s)
        return np.hstack([vis, panel])

    @staticmethod
    def pick(view, x, y):
        """Track id whose box contains video pixel (x, y), else None."""
        for k in view["tracks"]:
            x1, y1, x2, y2 = k["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return k["track_id"]
        return None
