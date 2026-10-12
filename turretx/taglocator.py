"""Simulator-only "GPS": finds each friendly's roof sticker by colour (design §5.3, R1).

The sticker sits on the roof but the homography maps the table plane, and the tracker uses the
bottom of each box, so sticker and track disagree by a few cm on real cars. Each tag's
`offset_cm` (measured with scripts/measure_tag_offset.py) is subtracted to line them up.

FIREWALL: output goes ONLY to the RF simulator. tracker.py and matcher.py must never import
this module. Identity is decided by the signed RF claims, never by the sticker.
"""
import cv2
import numpy as np

from turretx.kinematics import History

MIN_BLOB_PX = 30


class TagLocator:
    def __init__(self, tags_cfg, homo, velocity_window_s=0.5):
        """tags_cfg: {rf_id: {"hsv_lo": [h, s, v], "hsv_hi": [h, s, v]}}"""
        self.homo = homo
        self.tags = {rf_id: (np.array(c["hsv_lo"], np.uint8), np.array(c["hsv_hi"], np.uint8))
                     for rf_id, c in tags_cfg.items()}
        self.offsets = {rf_id: c.get("offset_cm", [0.0, 0.0]) for rf_id, c in tags_cfg.items()}
        self.histories = {rf_id: History(velocity_window_s) for rf_id in self.tags}
        self._kernel = np.ones((3, 3), np.uint8)

    def update(self, frame, t):
        """TagFix dicts (contract §6), one per configured tag. visible=False if not found."""
        if not self.homo.ready:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fixes = []
        for rf_id, (lo, hi) in self.tags.items():
            mask = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, self._kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blob = max(contours, key=cv2.contourArea, default=None)
            if blob is None or cv2.contourArea(blob) < MIN_BLOB_PX:
                fixes.append({"rf_id": rf_id, "t": t, "visible": False})
                continue
            m = cv2.moments(blob)
            x, y = self.homo.to_cm([(m["m10"] / m["m00"], m["m01"] / m["m00"])])[0]
            x, y = x - self.offsets[rf_id][0], y - self.offsets[rf_id][1]
            hist = self.histories[rf_id]
            hist.add(t, float(x), float(y))
            vx, vy = hist.velocity()
            fixes.append({"rf_id": rf_id, "t": t, "x": float(x), "y": float(y),
                          "vx": vx, "vy": vy, "visible": True})
        return fixes
