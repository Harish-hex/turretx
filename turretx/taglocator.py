"""Simulator-only "GPS": finds each friendly's roof sticker (design §5.3, R1).

Two detection modes per tag (config `tags.<rf_id>.mode`):
  disc  a round sticker: dark disc inside a light ring (e.g. the SIST2026 event stickers).
        Found by SHAPE, not colour: dark regions whose outline, projected into arena cm (which
        removes the camera's perspective), is round and the right physical size. The light ring
        separates the disc even on a dark roof. ArUco markers (squares), shadows and car bodies
        (elongated) fail the roundness tests.
        Supports ONE tag (identical discs can't be told apart from ~2 m).
  hsv   a single-colour patch, by HSV range.

The sticker sits on the roof but the homography maps the arena plane, and the tracker uses its
own ground point, so sticker and track disagree by a few cm on real cars. Each tag's `offset_cm`
(measured with scripts/measure_tag_offset.py) is subtracted to line them up.

FIREWALL: output goes ONLY to the RF simulator. tracker.py and matcher.py must never import
this module. Identity is decided by the signed RF claims, never by the sticker.
"""
import math

import cv2
import numpy as np

from turretx.kinematics import History

MIN_BLOB_PX = 30
ARENA_SLACK_CM = 5        # hsv tags are only searched inside the arena (+ this margin): green/blue
                          # things around the arena (bags, tape, clothes) must not become the friendly
DISC_DARK_V = 80          # disc pixels are darker than this (HSV value, 0-255)
# Shape tests run in arena cm (perspective removed), on the convex hull (rim text notches gone).
# The SIST2026 sticker has rim text top AND bottom, which at ~2 m eats the disc's top and bottom
# edges: what's left is a lens (circularity ≈ 0.87, fill ≈ 0.6). So circularity carries the
# decision (square 0.785, car body < 0.70) and fill is only a sanity check.
DISC_MIN_CIRCULARITY = 0.82
DISC_MIN_FILL = 0.50
CORNER_EXCLUSION = 0.10   # × arena diagonal: no tags accepted this close to a corner (ArUco markers)
DISC_SIZE_RANGE = (0.75, 1.35)  # accepted diameter vs configured diameter_cm
DISC_RING_CONTRAST = 30   # the ring just outside must be this much brighter than the disc


class TagLocator:
    def __init__(self, tags_cfg, homo, velocity_window_s=0.5):
        """tags_cfg: {rf_id: {"mode": "disc", "diameter_cm": 5.5} | {"mode": "hsv", "hsv_lo", "hsv_hi"}}"""
        self.homo = homo
        self.cfg = {rf_id: dict(c) for rf_id, c in tags_cfg.items()}
        for c in self.cfg.values():
            c.setdefault("mode", "hsv")
        if sum(c["mode"] == "disc" for c in self.cfg.values()) > 1:
            raise ValueError("disc mode supports one tag: identical round stickers can't be told apart")
        self.offsets = {rf_id: c.get("offset_cm", [0.0, 0.0]) for rf_id, c in self.cfg.items()}
        self.histories = {rf_id: History(velocity_window_s) for rf_id in self.cfg}
        self.last_px = {}
        self._kernel = np.ones((3, 3), np.uint8)
        self._arena_key, self._arena = None, None

    def _arena_mask(self, shape):
        """Arena (+ slack) as a pixel mask, cached until the calibration or frame size changes."""
        key = (self.homo.H.tobytes(), shape[:2])
        if key != self._arena_key:
            s = ARENA_SLACK_CM
            poly = self.homo.to_px([(-s, -s), (self.homo.w + s, -s),
                                    (self.homo.w + s, self.homo.h + s), (-s, self.homo.h + s)])
            self._arena = np.zeros(shape[:2], np.uint8)
            cv2.fillPoly(self._arena, [poly.astype(np.int32)], 255)
            self._arena_key = key
        return self._arena

    # ------------------------------------------------------------------ detectors
    def _find_hsv(self, hsv, c):
        mask = cv2.inRange(hsv, np.array(c["hsv_lo"], np.uint8), np.array(c["hsv_hi"], np.uint8))
        mask = cv2.bitwise_and(mask, self._arena_mask(hsv.shape))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob = max(contours, key=cv2.contourArea, default=None)
        if blob is None or cv2.contourArea(blob) < MIN_BLOB_PX:
            return None
        m = cv2.moments(blob)
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    def _near_corner(self, x, y):
        w, h = self.homo.w, self.homo.h
        r = CORNER_EXCLUSION * math.hypot(w, h)
        return any(math.hypot(x - cx, y - cy) < r for cx, cy in ((0, 0), (w, 0), (w, h), (0, h)))

    def _close_kernel(self, diameter_cm):
        """Odd kernel ≈ 30% of the disc's diameter in pixels (at the arena centre), min 5."""
        cx, cy = self.homo.w / 2, self.homo.h / 2
        a, b = self.homo.to_px([(cx - diameter_cm / 2, cy), (cx + diameter_cm / 2, cy)])
        k = max(5, int(0.3 * np.linalg.norm(a - b)))
        return k + 1 if k % 2 == 0 else k

    def _find_disc(self, hsv, c, rf_id):
        v = hsv[:, :, 2]
        raw = (v < DISC_DARK_V).astype(np.uint8) * 255
        # Three gap-healing strengths: a sticker-sized one bridges the white rim text and icon;
        # gentle / none keep a thin light ring (1-2 px from far away) separating the disc from a
        # dark car roof. Each pass adds candidates; the shape tests pick the real disc.
        contours = []
        for k in sorted({1, 3, self._close_kernel(c["diameter_cm"])}):
            dark = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) if k > 1 else raw
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, self._kernel) if k > 1 else dark
            contours += cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        best, best_score = None, -1.0
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_BLOB_PX:
                continue
            hull_px = cv2.convexHull(cnt)  # rim text notches the outline; roundness is about the hull
            hull_cm = self.homo.to_cm(hull_px.reshape(-1, 2)).astype(np.float32)
            area = cv2.contourArea(hull_cm)
            perim = cv2.arcLength(hull_cm, True)
            if area <= 0 or perim == 0:
                continue
            (xcm, ycm), r_cm = cv2.minEnclosingCircle(hull_cm)
            circularity = 4 * math.pi * area / (perim * perim)
            fill = area / (math.pi * r_cm * r_cm)
            size = 2 * r_cm / c["diameter_cm"]
            if (circularity < DISC_MIN_CIRCULARITY or fill < DISC_MIN_FILL
                    or not DISC_SIZE_RANGE[0] <= size <= DISC_SIZE_RANGE[1]
                    or not self.homo.inside(xcm, ycm) or self._near_corner(xcm, ycm)):
                continue
            # light ring just outside the disc (orange ring / cream border)
            (cx, cy), r_px = cv2.minEnclosingCircle(hull_px)
            ring = np.zeros(v.shape, np.uint8)
            cv2.circle(ring, (int(cx), int(cy)), int(r_px * 1.3) + 2, 255, -1)
            cv2.circle(ring, (int(cx), int(cy)), int(r_px * 1.05) + 1, 0, -1)
            inner = np.zeros(v.shape, np.uint8)
            cv2.drawContours(inner, [hull_px], -1, 255, -1)
            if cv2.mean(v, ring)[0] - cv2.mean(v, inner)[0] < DISC_RING_CONTRAST:
                continue
            m = cv2.moments(hull_px)
            centre = (m["m10"] / m["m00"], m["m01"] / m["m00"])
            score = circularity + fill
            if rf_id in self.last_px:  # prefer the candidate nearest to where it was
                d = math.hypot(centre[0] - self.last_px[rf_id][0], centre[1] - self.last_px[rf_id][1])
                score -= d / max(r_px * 8, 1)
            if score > best_score:
                best, best_score = centre, score
        return best

    # ------------------------------------------------------------------ main step
    def update(self, frame, t):
        """TagFix dicts (contract §6), one per configured tag. visible=False if not found."""
        if not self.homo.ready:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fixes = []
        for rf_id, c in self.cfg.items():
            px = self._find_disc(hsv, c, rf_id) if c["mode"] == "disc" else self._find_hsv(hsv, c)
            if px is None:
                fixes.append({"rf_id": rf_id, "t": t, "visible": False})
                continue
            self.last_px[rf_id] = px
            x, y = self.homo.to_cm([px])[0]
            x, y = x - self.offsets[rf_id][0], y - self.offsets[rf_id][1]
            hist = self.histories[rf_id]
            hist.add(t, float(x), float(y))
            vx, vy = hist.velocity()
            fixes.append({"rf_id": rf_id, "t": t, "x": float(x), "y": float(y),
                          "vx": vx, "vy": vy, "visible": True})
        return fixes
