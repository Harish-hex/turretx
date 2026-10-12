"""Measure how far each roof sticker sits from its car's tracked position, on a real clip.

  python scripts/measure_tag_offset.py --source recordings/real1.mp4 [--backend yolo --weights yolo26n.pt]

The sticker is on the roof (above the table plane) and the tracker uses the bottom of the box,
so the two disagree by a few cm. This reports the median (sticker − track) vector per tag;
paste it into config.yaml as that tag's `offset_cm`. Run it with the detector you'll demo with.
Use a clip where the stickered car drives around most of the table, away from other cars.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turretx.camera import Camera  # noqa: E402
from turretx.config import load_config  # noqa: E402
from turretx.homography import Homography  # noqa: E402
from turretx.taglocator import TagLocator  # noqa: E402
from turretx.tracker import Tracker  # noqa: E402

PAIR_CM = 10  # nearest track must be this close to count, and no second track nearby


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backend")
    ap.add_argument("--weights")
    ap.add_argument("--background", help="empty-table image for the diff backend")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["detector"]
    for key, val in [("backend", args.backend), ("weights", args.weights)]:
        if val is not None:
            d[key] = val
    if args.background:
        d["diff"]["background"] = args.background
    for tag in cfg["tags"].values():
        tag["offset_cm"] = [0.0, 0.0]  # measure the raw offset

    cam = Camera(args.source)
    homo = Homography(cfg["table_cm"], cfg["aruco"]["dict"], cfg["aruco"]["ids"])
    tracker = locator = None
    deltas = {rf_id: [] for rf_id in cfg["tags"]}
    while True:
        frame, t = cam.read()
        if frame is None:
            break
        if not homo.ready and not homo.calibrate(frame):
            continue
        tracker = tracker or Tracker(cfg, homo)
        locator = locator or TagLocator(cfg["tags"], homo)
        tracks = tracker.update(frame, t)
        for fix in locator.update(frame, t):
            if not fix["visible"] or not tracks:
                continue
            dist = sorted((np.hypot(fix["x"] - k["x"], fix["y"] - k["y"]), i) for i, k in enumerate(tracks))
            if dist[0][0] > PAIR_CM or (len(dist) > 1 and dist[1][0] < 2 * PAIR_CM):
                continue  # unmatched or ambiguous (another car close by)
            k = tracks[dist[0][1]]
            deltas[fix["rf_id"]].append((fix["x"] - k["x"], fix["y"] - k["y"]))
    cam.release()

    for rf_id, ds in deltas.items():
        if len(ds) < 30:
            print(f"{rf_id}: only {len(ds)} usable frames; record a longer clip with the car in clear space")
            continue
        ds = np.array(ds)
        med = np.median(ds, axis=0)
        spread = np.percentile(np.hypot(*(ds - med).T), 90)
        print(f"{rf_id}: {len(ds)} frames  median offset = [{med[0]:.1f}, {med[1]:.1f}] cm  "
              f"(90% of frames within {spread:.1f} cm of it)")
        print(f"  config.yaml →  {rf_id}: {{..., offset_cm: [{med[0]:.1f}, {med[1]:.1f}]}}")
        if spread > 4:
            print("  NOTE: offset varies a lot across the table; the camera may be too shallow (<60°)")


if __name__ == "__main__":
    main()
