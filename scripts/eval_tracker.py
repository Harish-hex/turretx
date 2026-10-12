"""Hour-2 checkpoint: compare detector backends on a clip.

  python scripts/eval_tracker.py --backend diff --source recordings/synthetic.mp4 --truth
  python scripts/eval_tracker.py --backend yolo --weights yolo26n.pt --source recordings/real1.mp4

With --truth (synthetic clip only) it scores against exact ground truth: recall, position and
velocity error, ID switches. Without it (real clips) it reports track stability: vehicles per
frame and how many distinct IDs were created (fewer = steadier tracking).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turretx import synthetic  # noqa: E402
from turretx.camera import Camera  # noqa: E402
from turretx.config import load_config  # noqa: E402
from turretx.homography import Homography  # noqa: E402
from turretx.tracker import Tracker  # noqa: E402

MATCH_CM = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="recordings/synthetic.mp4")
    ap.add_argument("--backend")
    ap.add_argument("--weights")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--ground-frac", type=float)
    ap.add_argument("--background", help="empty-table image for the diff backend")
    ap.add_argument("--truth", action="store_true", help="score against synthetic ground truth")
    args = ap.parse_args()

    cfg = load_config()
    d = cfg["detector"]
    if args.truth:
        cfg["table_cm"] = list(synthetic.TABLE_CM)
        d["diff"]["background"] = "recordings/synthetic_bg.png"
        d["ground_point_frac"] = 0.5  # synthetic cars are flat: their centre IS the ground point
    for key, val in [("backend", args.backend), ("weights", args.weights), ("conf", args.conf),
                     ("ground_point_frac", args.ground_frac)]:
        if val is not None:
            d[key] = [w.strip() for w in val.split(",")] if key == "weights" else val
    if args.background:
        d["diff"]["background"] = args.background

    cam = Camera(args.source)
    homo = Homography(cfg["table_cm"], cfg["aruco"]["dict"], cfg["aruco"]["ids"])
    tracker = None

    frames, per_frame, ids = 0, [], set()
    hits = total = 0
    pos_err, vel_err, last_id, switches = [], [], {}, 0
    t0 = None  # timer starts after the first tracked frame: model load + GPU warm-up excluded
    while True:
        frame, t = cam.read()
        if frame is None:
            break
        if not homo.ready and not homo.calibrate(frame):
            continue
        tracker = tracker or Tracker(cfg, homo)
        tracks = tracker.update(frame, t)
        if t0 is None:
            t0 = time.monotonic()
            continue
        frames += 1
        per_frame.append(len(tracks))
        ids.update(tr["track_id"] for tr in tracks)
        if not args.truth:
            continue
        for name, (x, y, _) in synthetic.car_states(t).items():
            total += 1
            if not tracks:
                continue
            best = min(tracks, key=lambda tr: np.hypot(tr["x"] - x, tr["y"] - y))
            dist = float(np.hypot(best["x"] - x, best["y"] - y))
            if dist > MATCH_CM:
                continue
            hits += 1
            pos_err.append(dist)
            if t > 1.5:  # velocity needs ~1 s of history
                tvx, tvy = synthetic.car_velocity(name, t)
                vel_err.append(float(np.hypot(best["vx"] - tvx, best["vy"] - tvy)))
            if name in last_id and last_id[name] != best["track_id"]:
                switches += 1
            last_id[name] = best["track_id"]
    elapsed = time.monotonic() - t0 if t0 else 0.0
    cam.release()

    print(f"backend={cfg['detector']['backend']} weights={d.get('weights')} frames={frames} "
          f"fps={frames / max(elapsed, 1e-6):.1f} (excl. first frame)")
    print(f"  vehicles/frame mean={np.mean(per_frame) if per_frame else 0:.2f}  distinct IDs={len(ids)}")
    if args.truth:
        recall = hits / max(total, 1)
        print(f"  recall={recall:.1%}  ID switches={switches}")
        if pos_err:
            print(f"  position error cm: mean={np.mean(pos_err):.2f}  p95={np.percentile(pos_err, 95):.2f}")
        if vel_err:
            print(f"  velocity error cm/s: mean={np.mean(vel_err):.2f}  p95={np.percentile(vel_err, 95):.2f}")


if __name__ == "__main__":
    main()
