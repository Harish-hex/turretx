"""Vehicle detection + tracking → Track dicts in table cm (design §5.1, contract §6).

Backends (config detector.backend):
  yolo       Ultralytics YOLO (yolov8n.pt, yolo26n.pt, or fine-tuned models/vehicle.pt) + ByteTrack
  yoloworld  YOLO-World with text classes (e.g. "toy car") + ByteTrack
  diff       difference from an empty-table reference frame + a small centroid tracker
Identity is NOT decided here; this module only finds and follows vehicles.
"""
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from turretx.kinematics import History, motion

BYTETRACK_YAML = str(Path(__file__).with_name("bytetrack.yaml"))
TABLE_SLACK_CM = 5      # keep vehicles slightly outside the marker-centre rectangle
PRUNE_AFTER_S = 2.0     # forget a track's history after this long unseen
AREA_JUMP = 1.5         # box area changing by more than this factor in one frame = merge/split
MERGED_AREA = 1.7       # a box this much bigger than the typical vehicle = two cars in one blob


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloBackend:
    def __init__(self, dcfg, world=False):
        from ultralytics import YOLO, YOLOWorld

        self.device = pick_device()
        self.conf = dcfg["conf"]
        self.imgsz = dcfg["imgsz"]
        if world:
            self.model = YOLOWorld(dcfg.get("world_weights", "yolov8s-world.pt"))
            self.model.set_classes(dcfg["classes_text"])
            self.classes = None
        else:
            self.model = YOLO(dcfg["weights"])
            excluded = set(dcfg.get("exclude_classes", []))
            self.classes = [i for i, n in self.model.names.items() if n not in excluded]  # R7: no people/hands

    def update(self, frame):
        """[(track_id, (x1, y1, x2, y2), conf)]"""
        r = self.model.track(frame, persist=True, tracker=BYTETRACK_YAML, conf=self.conf,
                             imgsz=self.imgsz, classes=self.classes, device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        ids = r.boxes.id.int().tolist()
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        return [(i, tuple(b), float(c)) for i, b, c in zip(ids, boxes, confs)]


class CentroidTracker:
    """Hungarian assignment on centroid distance (px). Enough for the diff backend."""

    def __init__(self, max_dist_px=60, max_missed=30):
        self.max_dist = max_dist_px
        self.max_missed = max_missed
        self.tracks = {}  # id → {"c": (x, y), "missed": n}
        self.next_id = 1

    def update(self, boxes):
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        ids = list(self.tracks)
        assigned = {}
        if ids and cents:
            cost = np.array([[np.hypot(cx - self.tracks[i]["c"][0], cy - self.tracks[i]["c"][1])
                              for cx, cy in cents] for i in ids])
            gated = np.where(cost > self.max_dist, 1e6, cost)  # finite cost, never inf (R10)
            for r, c in zip(*linear_sum_assignment(gated)):
                if gated[r, c] < 1e6:
                    assigned[c] = ids[r]
        out = []
        for k, (b, c) in enumerate(zip(boxes, cents)):
            tid = assigned.get(k)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = {"c": c, "missed": 0}
            out.append((tid, b))
        seen = {tid for tid, _ in out}
        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return out


class DiffBackend:
    def __init__(self, dcfg, homo):
        self.homo = homo
        self.min_area = dcfg["diff"]["min_area"]
        self.max_area = dcfg["diff"]["max_area"]
        self.thresh = dcfg["diff"].get("thresh", 30)
        self.background = None
        bg = dcfg["diff"].get("background")
        if bg and Path(bg).exists():
            self.background = cv2.imread(bg)
        self.tracker = CentroidTracker()
        self._zone_key = None  # zone masks are cached until the calibration changes

    def _zone_masks(self, shape):
        """(zone, inner): table area + slack, and the same eroded by ~3 px. A blob with any
        outline point outside `inner` touches the zone edge: an arm/stick from outside (R7)."""
        key = (self.homo.H.tobytes(), shape[:2])
        if key != self._zone_key:
            s = TABLE_SLACK_CM
            poly = self.homo.to_px([(-s, -s), (self.homo.w + s, -s),
                                    (self.homo.w + s, self.homo.h + s), (-s, self.homo.h + s)])
            self._zone = np.zeros(shape[:2], np.uint8)
            cv2.fillPoly(self._zone, [poly.astype(np.int32)], 255)
            self._inner = cv2.erode(self._zone, np.ones((7, 7), np.uint8))
            self._zone_key = key
        return self._zone, self._inner

    def set_background(self, frame):
        self.background = frame.copy()

    def update(self, frame):
        if self.background is None or not self.homo.ready:
            return []
        zone, inner = self._zone_masks(frame.shape)
        diff = cv2.absdiff(frame, self.background).max(axis=2)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, zone)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if not self.min_area <= area <= self.max_area:
                continue
            pts = c[:, 0, :]
            if (inner[pts[:, 1], pts[:, 0]] == 0).any():  # touches the zone edge (R7)
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
        return [(tid, b, 1.0) for tid, b in self.tracker.update(boxes)]


class Tracker:
    def __init__(self, cfg, homo):
        dcfg = cfg["detector"]
        self.homo = homo
        self.ground_frac = dcfg.get("ground_point_frac", 1.0)  # 1.0 = bbox bottom, 0.5 = centre (R11)
        backend = dcfg["backend"]
        if backend == "diff":
            self.backend = DiffBackend(dcfg, homo)
        elif backend in ("yolo", "yoloworld"):
            self.backend = YoloBackend(dcfg, world=backend == "yoloworld")
        else:
            raise ValueError(f"unknown detector backend {backend!r}")
        self.velocity_window_s = cfg["tracker"].get("velocity_window_s", 0.5)
        self.histories = {}
        self.areas = {}
        self.typical_area = deque(maxlen=900)  # recent box areas; the median ≈ one vehicle

    def update(self, frame, t):
        """Track dicts (contract §6) for vehicles inside the table area."""
        if not self.homo.ready:
            return []
        dets = self.backend.update(frame)
        if not dets:
            self._prune(t)
            return []
        gp = [((b[0] + b[2]) / 2, b[1] + self.ground_frac * (b[3] - b[1])) for _, b, _ in dets]
        areas_now = [max((b[2] - b[0]) * (b[3] - b[1]), 1.0) for _, b, _ in dets]
        self.typical_area.extend(areas_now)
        typical = float(np.median(self.typical_area)) if len(self.typical_area) >= 30 else None
        cm = self.homo.to_cm(gp)
        tracks = []
        for (tid, box, conf), (x, y) in zip(dets, cm):
            if not self.homo.inside(x, y, margin=-TABLE_SLACK_CM):
                continue
            hist = self.histories.setdefault(tid, History(self.velocity_window_s))
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            merged = typical is not None and area > MERGED_AREA * typical
            prev_area = self.areas.get(tid)
            if prev_area and not 1 / AREA_JUMP <= area / prev_area <= AREA_JUMP:
                hist.reset()  # two cars merged into one blob, or one split off: not one motion
            self.areas[tid] = area
            hist.add(t, float(x), float(y))
            vx, vy = hist.velocity()
            speed, heading = motion(vx, vy)
            tracks.append({"track_id": int(tid), "t": t, "x": float(x), "y": float(y),
                           "vx": vx, "vy": vy, "speed": speed, "heading": heading,
                           "settled": hist.settled, "merged": merged,
                           "bbox": [float(v) for v in box], "det_conf": conf})
        self._prune(t)
        return tracks

    def _prune(self, t):
        for tid in [k for k, h in self.histories.items() if h.last_t is None or t - h.last_t > PRUNE_AFTER_S]:
            del self.histories[tid]
            self.areas.pop(tid, None)
