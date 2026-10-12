"""Vehicle detection + tracking → Track dicts in table cm (design §5.1, contract §6).

Backends (config detector.backend):
  yolo       Ultralytics YOLO + ByteTrack. detector.weights is one file or a list, e.g.
             [yolov8n.pt, models/vehicle.pt]: every model runs, overlapping boxes are merged into one
             and a single ByteTrack follows the merged boxes (one vehicle = one track)
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
MERGED_AREA = 1.7       # a box suddenly this much bigger than its OWN recent size = two cars in one blob
BASELINE_FRAMES = 15    # a track's recent unmerged box areas (~0.5 s): slow turning is absorbed, a merge isn't
MERGED_RESET_S = 3.0    # "merged" this long = the vehicle itself changed (turned, came closer): re-learn its size


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _overlap(a, b):
    """(IoU, intersection ÷ smaller box area) of two xyxy boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0, 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter), inter / max(min(area_a, area_b), 1e-9)


def merge_overlapping(dets, iou=0.3, contain=0.7):
    """Merge boxes that are the same vehicle seen twice (two models, or two classes of one model).
    dets: [((x1, y1, x2, y2), conf)]. Two boxes merge if IoU >= `iou` or the smaller one is at least
    `contain` inside the other; merging is transitive. Each group → (union box, best conf), ordered
    by best conf. Boxes that only touch stay separate (two cars parked side by side)."""
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            o_iou, o_contain = _overlap(dets[i][0], dets[j][0])
            if o_iou >= iou or o_contain >= contain:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(dets[i])
    merged = []
    for members in groups.values():
        boxes = [b for b, _ in members]
        union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes))
        merged.append((tuple(float(v) for v in union), max(float(c) for _, c in members)))
    return sorted(merged, key=lambda m: (-m[1], m[0]))


class YoloBackend:
    def __init__(self, dcfg, world=False):
        from ultralytics import YOLO, YOLOWorld

        if world:
            model = YOLOWorld(dcfg.get("world_weights", "yolov8s-world.pt"))
            model.set_classes(dcfg["classes_text"])
            models = [(model, None)]
        else:
            weights = dcfg["weights"]
            excluded = set(dcfg.get("exclude_classes", []))
            models = []
            for w in [weights] if isinstance(weights, str) else list(weights):
                model = YOLO(w)
                models.append((model, [i for i, n in model.names.items() if n not in excluded]))  # R7: no people/hands
        self._init_tracking(dcfg, models)

    def _init_tracking(self, dcfg, models):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace

        self.device = pick_device()
        conf = dcfg["conf"]  # one value, or one per weights file (a custom model can be stricter)
        self.confs = list(conf) if isinstance(conf, (list, tuple)) else [conf] * len(models)
        if len(self.confs) != len(models):
            raise ValueError(f"detector.conf has {len(self.confs)} values for {len(models)} weights")
        self.conf = min(self.confs)
        self.imgsz = dcfg["imgsz"]
        self.merge_iou = dcfg.get("merge_iou", 0.3)
        self.merge_contain = dcfg.get("merge_contain", 0.7)
        self.models = models
        self.model, self.classes = models[0]
        self.tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(BYTETRACK_YAML)))

    def update(self, frame):
        """[(track_id, (x1, y1, x2, y2), conf)]"""
        from ultralytics.engine.results import Boxes

        dets = []
        for (model, classes), conf in zip(self.models, self.confs):
            r = model.predict(frame, conf=conf, imgsz=self.imgsz, classes=classes, device=self.device,
                              agnostic_nms=True, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                dets += [(tuple(b), float(c)) for b, c in
                         zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        merged = merge_overlapping(dets, self.merge_iou, self.merge_contain)
        data = np.array([[*b, c, 0] for b, c in merged], np.float32).reshape(-1, 6)
        tracked = self.tracker.update(Boxes(data, frame.shape[:2]), frame)
        return [(int(t[4]), tuple(float(v) for v in t[:4]), float(t[5])) for t in tracked]


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
        # Per-track size baseline, not one "typical vehicle" for all: a big vehicle next to a small
        # one is not a merged blob (live-arena finding: mixed sizes were flagged TOO CLOSE).
        self.baselines = {}     # track id → deque of its recent unmerged box areas
        self.merged_since = {}  # track id → t when its "merged" flag started

    def update(self, frame, t):
        """Track dicts (contract §6) for vehicles inside the table area."""
        if not self.homo.ready:
            return []
        dets = self.backend.update(frame)
        if not dets:
            self._prune(t)
            return []
        gp = [((b[0] + b[2]) / 2, b[1] + self.ground_frac * (b[3] - b[1])) for _, b, _ in dets]
        cm = self.homo.to_cm(gp)
        tracks = []
        for (tid, box, conf), (x, y) in zip(dets, cm):
            if not self.homo.inside(x, y, margin=-TABLE_SLACK_CM):
                continue
            hist = self.histories.setdefault(tid, History(self.velocity_window_s))
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            merged = self._merged(tid, area, t)
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

    def _merged(self, tid, area, t):
        """True if this track's box suddenly grew well past its own recent size (two cars in one
        blob). A new track has no baseline yet, so it is never flagged."""
        base = self.baselines.setdefault(tid, deque(maxlen=BASELINE_FRAMES))
        merged = len(base) >= 5 and area > MERGED_AREA * float(np.median(base))
        if merged:
            if t - self.merged_since.setdefault(tid, t) > MERGED_RESET_S:
                base.clear()
                del self.merged_since[tid]
                merged = False
        else:
            self.merged_since.pop(tid, None)
        if not merged:
            base.append(area)  # merged areas never become the baseline
        return merged

    def _prune(self, t):
        for tid in [k for k, h in self.histories.items() if h.last_t is None or t - h.last_t > PRUNE_AFTER_S]:
            del self.histories[tid]
            self.areas.pop(tid, None)
            self.baselines.pop(tid, None)
            self.merged_since.pop(tid, None)
