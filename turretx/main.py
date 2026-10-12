"""TurretX main loop.

Phase 2: camera → homography → vehicle tracking (+ simulator-only tag locator).
Run:  python -m turretx.main [--source 1 | recordings/clip.mp4] [--backend yolo|yoloworld|diff]
                             [--weights yolo26n.pt] [--debug-tags] [--headless --max-frames N]
Keys: h calibrate (ArUco; falls back to clicking TL, TR, BR, BL) · b capture empty-table
      background (diff backend) · p pause · q quit
"""
import argparse
import time

import cv2

from turretx.camera import Camera
from turretx.config import load_config, parse_source
from turretx.homography import Homography
from turretx.taglocator import TagLocator
from turretx.tracker import DiffBackend, Tracker

WINDOW = "TurretX"
DRIFT_CHECK_EVERY = 30  # frames
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def draw_table(vis, homo, step_cm=10):
    """Project a 10 cm grid and the table outline onto the camera image."""
    for x in range(0, int(homo.w) + 1, step_cm):
        a, b = homo.to_px([(x, 0), (x, homo.h)]).astype(int)
        cv2.line(vis, tuple(a), tuple(b), (90, 90, 90), 1)
    for y in range(0, int(homo.h) + 1, step_cm):
        a, b = homo.to_px([(0, y), (homo.w, y)]).astype(int)
        cv2.line(vis, tuple(a), tuple(b), (90, 90, 90), 1)
    outline = homo.to_px([(0, 0), (homo.w, 0), (homo.w, homo.h), (0, homo.h)]).astype(int)
    cv2.polylines(vis, [outline], True, (255, 255, 0), 2)


def draw_tracks(vis, tracks):
    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
        label = f"#{tr['track_id']} {tr['speed']:.0f}cm/s"
        cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)


def draw_tag_fixes(vis, homo, fixes):
    """Debug only (--debug-tags): the simulator's GPS stand-in. Hidden in the demo (§5.3)."""
    for f in fixes:
        if f["visible"]:
            px, py = homo.to_px([(f["x"], f["y"])])[0].astype(int)
            cv2.drawMarker(vis, (int(px), int(py)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(vis, f"GPS {f['rf_id']}", (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def put_lines(vis, lines, colour=(255, 255, 255)):
    for i, text in enumerate(lines):
        y = 30 + 30 * i
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="webcam index, video file or URL (overrides config)")
    ap.add_argument("--backend", help="detector backend: yolo | yoloworld | diff")
    ap.add_argument("--weights", help="YOLO weights, e.g. yolo26n.pt or models/vehicle.pt")
    ap.add_argument("--debug-tags", action="store_true", help="show the simulator-only sticker GPS")
    ap.add_argument("--headless", action="store_true", help="no window (smoke tests)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg["detector"]["backend"] = args.backend
    if args.weights:
        cfg["detector"]["weights"] = args.weights
    source = parse_source(args.source if args.source is not None else cfg["camera"]["source"])
    cam = Camera(source, cfg["camera"]["width"], cfg["camera"]["height"])
    homo = Homography(cfg["table_cm"], cfg["aruco"]["dict"], cfg["aruco"]["ids"])
    calib_file = cfg["homography_file"]
    if homo.load(calib_file):
        print(f"Loaded calibration from {calib_file} (press h to recalibrate)")

    tracker = Tracker(cfg, homo)
    tag_locator = TagLocator(cfg["tags"], homo, cfg["tracker"]["velocity_window_s"])
    tracks, fixes = [], []

    clicks = []
    state = {"click_mode": False}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and state["click_mode"]:
            clicks.append((x, y))
            if len(clicks) == 4:
                homo.set_points(clicks)
                homo.save(calib_file)
                state["click_mode"] = False
                print("Calibrated from 4 clicks")

    if not args.headless:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

    frame, t = None, 0.0
    frames, paused, moved = 0, False, False
    fps, last = 0.0, time.monotonic()
    while True:
        if not paused or frame is None:
            new_frame, new_t = cam.read()
            if new_frame is None:
                break
            frame, t = new_frame, new_t
            frames += 1
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            if not homo.ready and frames == 1 and homo.calibrate(frame):
                homo.save(calib_file)
                print("Auto-calibrated from ArUco markers")
            if homo.ready and frames % DRIFT_CHECK_EVERY == 0:
                d = homo.drift_px(frame)
                moved = d is not None and d > cfg["camera_moved_px"]

            tracks = tracker.update(frame, t)
            fixes = tag_locator.update(frame, t)  # → RF simulator only (Phase 3)

        if args.headless:
            if args.max_frames and frames >= args.max_frames:
                break
            continue

        vis = frame.copy()
        if homo.ready:
            draw_table(vis, homo)
            draw_tracks(vis, tracks)
            if args.debug_tags:
                draw_tag_fixes(vis, homo, fixes)
        lines = [f"t={t:7.2f}s  fps={fps:4.1f}  vehicles={len(tracks)}  [{cfg['detector']['backend']}]"
                 + ("  PAUSED" if paused else "")]
        if state["click_mode"]:
            lines.append(f"Click the {CORNER_NAMES[len(clicks)]} marker centre ({len(clicks)}/4)")
            for p in clicks:
                cv2.circle(vis, p, 6, (0, 255, 255), -1)
        elif not homo.ready:
            lines.append("NOT CALIBRATED - press h")
        put_lines(vis, lines)
        if moved:
            put_lines(vis, ["", "", "CAMERA MOVED - press h"], (255, 0, 255))
        cv2.imshow(WINDOW, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b") and isinstance(tracker.backend, DiffBackend):
            tracker.backend.set_background(frame)
            print("Captured empty-table background (make sure no cars were on the table)")
        if key == ord("p"):
            paused = not paused
        if key == ord("h"):
            moved = False
            if homo.calibrate(frame):
                homo.save(calib_file)
                print("Calibrated from ArUco markers")
            else:
                print("Markers not all visible: click the 4 corners (TL, TR, BR, BL)")
                clicks.clear()
                state["click_mode"] = True
        if args.max_frames and frames >= args.max_frames:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Processed {frames} frames, last t={t:.2f}s, calibrated={homo.ready}, "
          f"vehicles in last frame={len(tracks)}")


if __name__ == "__main__":
    main()
