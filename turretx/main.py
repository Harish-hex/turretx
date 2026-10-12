"""TurretX main loop.

Phase 2: camera → homography → vehicle tracking; sticker GPS → RF sim → signed claims.
Run:  python -m turretx.main [--source 1 | recordings/clip.mp4] [--backend yolo|yoloworld|diff]
                             [--weights yolo26n.pt] [--debug-tags] [--headless --max-frames N]
Keys: h calibrate (ArUco; falls back to clicking TL, TR, BR, BL: right-click undo, Esc cancel)
      · b capture empty-table background (diff backend) · p pause · q quit
      Red team: f forged · r replay · d cloned ID (toggle) · g ghost ID (toggle) · j jam (toggle)
"""
import argparse
import time

import cv2
import numpy as np

from turretx.camera import Camera
from turretx.config import load_config, parse_source
from turretx.homography import Homography
from turretx.rfrx import Interrogator, Receiver
from turretx.rfsim import RFSim
from turretx import wire
from turretx.taglocator import TagLocator
from turretx.tracker import DiffBackend, Tracker

WINDOW = "TurretX"
DRIFT_CHECK_EVERY = 30  # frames
AUTO_CALIBRATE_EVERY = 15  # frames, until calibrated
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


def draw_claims(vis, homo, claims):
    """RF claims as rings at the claimed position: cyan, magenta if the ID is in conflict."""
    for c in claims:
        px, py = homo.to_px([(c["x"], c["y"])])[0].astype(int)
        colour = (255, 0, 255) if c["conflict"] else (255, 255, 0)
        cv2.circle(vis, (int(px), int(py)), 16, colour, 2)
        cv2.putText(vis, c["rf_id"], (int(px) + 18, int(py) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def draw_tag_fixes(vis, homo, fixes):
    """Debug only (--debug-tags): the simulator's GPS stand-in. Hidden in the demo (§5.3)."""
    for f in fixes:
        if f["visible"]:
            px, py = homo.to_px([(f["x"], f["y"])])[0].astype(int)
            cv2.drawMarker(vis, (int(px), int(py)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(vis, f"GPS {f['rf_id']}", (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def draw_clicks(vis, clicks):
    """Numbered click points joined in order, so a wrong click is easy to spot (and undo)."""
    for i, p in enumerate(clicks):
        if i:
            cv2.line(vis, clicks[i - 1], p, (0, 255, 255), 2)
        cv2.circle(vis, p, 8, (0, 255, 255), -1)
        cv2.putText(vis, str(i + 1), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(vis, str(i + 1), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


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
    rf_sim = RFSim(cfg)
    interrogator = Interrogator(cfg["rf"]["challenge_hz"])
    receiver = Receiver(wire.rf_key(cfg["rf"]["key"]), interrogator)
    recent_claims, rejected = [], 0
    RED_TEAM_KEYS = {ord("f"): "forged", ord("r"): "replay", ord("d"): "clone",
                     ord("g"): "ghost", ord("j"): "jam"}

    clicks = []
    state = {"click_mode": False}
    notice = {"text": "", "until": 0.0}

    def notify(text, seconds=3.0):
        print(text)
        notice.update(text=text, until=time.monotonic() + seconds)

    def on_mouse(event, x, y, *_):
        if not state["click_mode"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            if len(clicks) == 4:
                homo.set_points(clicks)
                homo.save(calib_file)
                state["click_mode"] = False
                notify("Calibrated from 4 clicks")
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    if not args.headless:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

    frame, t = None, 0.0
    frames, paused, moved, no_signal = 0, False, False, False
    dt_avg, last = None, time.monotonic()
    while True:
        if not paused or frame is None:
            new_frame, new_t = cam.read()
            if new_frame is None:
                if cam.is_file:
                    break  # end of the recording
                no_signal = True  # live source dropped: keep the window up, camera reconnects
                if frame is None:
                    frame = np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), np.uint8)
            else:
                no_signal = False
                frame, t = new_frame, new_t
                frames += 1
                now = time.monotonic()
                dt = now - last
                last = now
                dt_avg = dt if dt_avg is None else 0.9 * dt_avg + 0.1 * dt

                # keep trying until the markers are seen (people/cars may cover them at start)
                if not homo.ready and not state["click_mode"] and frames % AUTO_CALIBRATE_EVERY == 1:
                    if homo.calibrate(frame):
                        homo.save(calib_file)
                        notify("Auto-calibrated from ArUco markers")
                if homo.ready and frames % DRIFT_CHECK_EVERY == 0:
                    d = homo.drift_px(frame)
                    if d is not None:  # covered markers keep the previous verdict (R12)
                        moved = d > cfg["camera_moved_px"]

                tracks = tracker.update(frame, t)
                fixes = tag_locator.update(frame, t)  # → RF simulator only, never the matcher

                rf_sim.observe(t, fixes, tracks)
                challenge = interrogator.poll(t)
                if challenge:
                    rf_sim.on_challenge(challenge, t)
                claims, rejections = receiver.receive(rf_sim.poll(t), t)
                recent_claims = [c for c in recent_claims if t - c["t"] < 0.45] + claims
                for r in rejections:
                    rejected += 1
                    notify(f"RF REJECTED: {r['reason'].replace('_', ' ')}", 2.5)

        if args.headless:
            if args.max_frames and frames >= args.max_frames:
                break
            continue

        vis = frame.copy()
        if homo.ready:
            draw_table(vis, homo)
            draw_tracks(vis, tracks)
            draw_claims(vis, homo, recent_claims)
            if args.debug_tags:
                draw_tag_fixes(vis, homo, fixes)
        fps = 1.0 / dt_avg if dt_avg else 0.0
        lines = [f"t={t:7.2f}s  fps={fps:4.1f}  vehicles={len(tracks)}  [{cfg['detector']['backend']}]"
                 + f"  RF rejected={rejected}" + ("  JAMMED" if rf_sim.channel.jammed else "")
                 + ("  PAUSED" if paused else "")]
        if state["click_mode"]:
            draw_clicks(vis, clicks)
            lines.append(f"CLICK the {CORNER_NAMES[len(clicks)].upper()} marker centre ({len(clicks) + 1}/4)")
            lines.append("right-click = undo, Esc = cancel")
        elif not homo.ready:
            lines.append("NOT CALIBRATED - press h")
        put_lines(vis, lines)
        alerts = []
        if no_signal:
            alerts.append("NO SIGNAL - reconnecting camera...")
        if moved:
            alerts.append("CAMERA MOVED - press h")
        if time.monotonic() < notice["until"]:
            alerts.append(notice["text"])
        if alerts:
            put_lines(vis, [""] * len(lines) + alerts, (255, 0, 255))
        cv2.imshow(WINDOW, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 27 and state["click_mode"]:
            state["click_mode"] = False
            notify("Click calibration cancelled")
        if key == ord("b"):
            if isinstance(tracker.backend, DiffBackend):
                tracker.backend.set_background(frame)
                notify("Captured empty-table background (table must be empty)")
            else:
                notify(f"b only applies to the diff backend (current: {cfg['detector']['backend']})")
        if key in RED_TEAM_KEYS:
            notify(rf_sim.red_team(RED_TEAM_KEYS[key], t), 4.0)
        if key == ord("p"):
            paused = not paused
        if key == ord("h"):
            moved = False
            if homo.calibrate(frame):
                homo.save(calib_file)
                notify("Calibrated from ArUco markers")
            else:
                notify("Markers not all visible: click the 4 corners instead", 5.0)
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
