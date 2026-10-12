"""Pixel ↔ table-cm mapping from 4 ArUco corner markers.

Table frame: origin at the CENTRE of the top-left marker, x → towards the top-right marker,
y → towards the bottom-left marker, in cm. So `table_cm` in config is the distance between
marker centres. Calibrate once (frozen), then only check for drift (R12).
"""
import json
from pathlib import Path

import cv2
import numpy as np


class Homography:
    def __init__(self, table_cm, aruco_dict="DICT_4X4_50", ids=(0, 1, 2, 3)):
        self.w, self.h = table_cm
        self.ids = list(ids)  # TL, TR, BR, BL
        self.dst_cm = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
        self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.H = None       # px → cm
        self.H_inv = None   # cm → px
        self.ref_px = None  # marker centres (px) at calibration, for drift checks

    @property
    def ready(self):
        return self.H is not None

    def detect_markers(self, frame):
        """{marker_id: centre (x, y) px} for our 4 ids that are visible."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        found = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) in self.ids:
                    found[int(i)] = c.reshape(4, 2).mean(axis=0)
        return found

    def set_points(self, px4):
        """Calibrate from 4 pixel points in order TL, TR, BR, BL (ArUco or manual clicks)."""
        src = np.float32(px4)
        self.H = cv2.getPerspectiveTransform(src, self.dst_cm)
        self.H_inv = np.linalg.inv(self.H)
        self.ref_px = src

    def calibrate(self, frame):
        """True if all 4 markers were found and the homography was set."""
        found = self.detect_markers(frame)
        if len(found) < 4:
            return False
        self.set_points([found[i] for i in self.ids])
        return True

    def drift_px(self, frame):
        """Max pixel distance between visible markers and their calibrated positions.
        None if fewer than 3 markers are visible (cars or sticks may cover some)."""
        if not self.ready:
            return None
        found = self.detect_markers(frame)
        if len(found) < 3:
            return None
        return max(float(np.linalg.norm(found[i] - self.ref_px[k]))
                   for k, i in enumerate(self.ids) if i in found)

    def to_cm(self, pts_px):
        pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_px(self, pts_cm):
        pts = np.asarray(pts_cm, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def inside(self, x_cm, y_cm, margin=0.0):
        return margin <= x_cm <= self.w - margin and margin <= y_cm <= self.h - margin

    def save(self, path):
        Path(path).write_text(json.dumps({"ref_px": self.ref_px.tolist(), "table_cm": [self.w, self.h]}))

    def load(self, path):
        """Load a saved calibration if it exists and matches the configured table size."""
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text())
        if data["table_cm"] != [self.w, self.h]:
            return False
        self.set_points(data["ref_px"])
        return True
