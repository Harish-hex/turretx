"""TurretX main loop: camera → tracker → (sticker GPS → RF sim) → receiver → matcher → display.

Run:  python -m turretx.main [--source 1 | recordings/clip.mp4] [--backend yolo|yoloworld|diff]
                             [--weights yolo26n.pt] [--debug-tags] [--no-log] [--headless --max-frames N]
Keys: h calibrate (ArUco; falls back to clicking TL, TR, BR, BL: right-click undo, Esc cancel)
      · b capture empty-arena background (diff backend) · click / Tab select vehicle
      · p pause · q quit
      Red team: f forged · r replay · d cloned ID (toggle) · g ghost ID (toggle) · j jam (toggle)
"""
import argparse
import time

import cv2
import numpy as np

from turretx import wire
from turretx.camera import Camera
from turretx.config import load_config, parse_source
from turretx.display import AMBER, MAGENTA, WHITE, Display
from turretx.events import EventLog
from turretx.homography import Homography
from turretx.matcher import Matcher
from turretx.rfrx import Interrogator, Receiver
from turretx.rfsim import RFSim
from turretx.taglocator import TagLocator
from turretx.tracker import DiffBackend, Tracker

WINDOW = "TurretX"
DRIFT_CHECK_EVERY = 30      # frames
AUTO_CALIBRATE_EVERY = 15   # frames, until calibrated
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]
RED_TEAM_KEYS = {ord("f"): "forged", ord("r"): "replay", ord("d"): "clone", ord("g"): "ghost", ord("j"): "jam"}
TAB, ESC = 9, 27


def draw_table(vis, homo):
    """Project a grid and the arena outline onto the camera image."""
    step = 10 if homo.w <= 120 else 25
    for x in range(0, int(homo.w) + 1, step):
        a, b = homo.to_px([(x, 0), (x, homo.h)]).astype(int)
        cv2.line(vis, tuple(a), tuple(b), (90, 90, 90), 1)
    for y in range(0, int(homo.h) + 1, step):
        a, b = homo.to_px([(0, y), (homo.w, y)]).astype(int)
        cv2.line(vis, tuple(a), tuple(b), (90, 90, 90), 1)
    outline = homo.to_px([(0, 0), (homo.w, 0), (homo.w, homo.h), (0, homo.h)]).astype(int)
    cv2.polylines(vis, [outline], True, (255, 255, 0), 2)


def draw_clicks(vis, clicks):
    """Numbered click points joined in order, so a wrong click is easy to spot (and undo)."""
    for i, p in enumerate(clicks):
        if i:
            cv2.line(vis, clicks[i - 1], p, (0, 255, 255), 2)
        cv2.circle(vis, p, 8, (0, 255, 255), -1)
        cv2.putText(vis, str(i + 1), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(vis, str(i + 1), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def draw_tag_fixes(vis, homo, fixes):
    """Debug only (--debug-tags): the simulator's GPS stand-in. Hidden in the demo (§5.3)."""
    for f in fixes:
        if f["visible"]:
            px, py = homo.to_px([(f["x"], f["y"])])[0].astype(int)
            cv2.drawMarker(vis, (int(px), int(py)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(vis, f"GPS {f['rf_id']}", (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="webcam index, video file or URL (overrides config)")
    ap.add_argument("--backend", help="detector backend: yolo | yoloworld | diff")
    ap.add_argument("--weights", help="YOLO weights, e.g. yolo26n.pt or models/vehicle.pt")
    ap.add_argument("--debug-tags", action="store_true", help="show the simulator-only sticker GPS")
    ap.add_argument("--no-log", action="store_true", help="don't write logs/<time>.jsonl")
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
    rf_sim = RFSim(cfg)
    interrogator = Interrogator(cfg["rf"]["challenge_hz"])
    receiver = Receiver(wire.rf_key(cfg["rf"]["key"]), interrogator)
    matcher = Matcher(cfg)
    events = EventLog(sound=not args.headless, write=not args.no_log)
    display = Display(cfg, homo)
    max_w = cfg.get("display", {}).get("max_width", 1600)

    ui = {"click_mode": False, "clicks": [], "selected": None, "scale": 1.0, "video_w": 0, "view": None}
    notice = {"text": "", "until": 0.0, "colour": WHITE}

    def notify(msg, seconds=3.0, colour=WHITE):
        print(msg)
        notice.update(text=msg, until=time.monotonic() + seconds, colour=colour)

    def on_mouse(event, x, y, *_):
        x, y = int(x / ui["scale"]), int(y / ui["scale"])
        if x >= ui["video_w"]:
            return  # side panel
        if ui["click_mode"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                ui["clicks"].append((x, y))
                if len(ui["clicks"]) == 4:
                    homo.set_points(ui["clicks"])
                    homo.save(calib_file)
                    ui["click_mode"] = False
                    notify("Calibrated from 4 clicks")
            elif event == cv2.EVENT_RBUTTONDOWN and ui["clicks"]:
                ui["clicks"].pop()
        elif event == cv2.EVENT_LBUTTONDOWN and ui["view"]:
            ui["selected"] = Display.pick(ui["view"], x, y)

    if not args.headless:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

    frame, t = None, 0.0
    frames, paused, moved, no_signal, rejected = 0, False, False, False, 0
    dt_avg, last = None, time.monotonic()
    tracks, fixes, recent_claims = [], [], []
    out = {"labels": {}, "ghosts": [], "rf_denied": False, "lost_friendlies": []}
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
                if not homo.ready and not ui["click_mode"] and frames % AUTO_CALIBRATE_EVERY == 1:
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
                rejected += len(rejections)
                recent_claims = [c for c in recent_claims if t - c["t"] < 0.45] + claims
                out = matcher.update(t, tracks, claims)
                events.update(t, out, rejections)

        if args.headless:
            if args.max_frames and frames >= args.max_frames:
                break
            continue

        vis = frame.copy()
        if homo.ready:
            draw_table(vis, homo)
            if args.debug_tags:
                draw_tag_fixes(vis, homo, fixes)
        banners = []
        if ui["click_mode"]:
            draw_clicks(vis, ui["clicks"])
            banners.append((f"CLICK the {CORNER_NAMES[len(ui['clicks'])].upper()} marker centre "
                            f"({len(ui['clicks']) + 1}/4) - right-click undo, Esc cancel", AMBER))
        elif not homo.ready:
            banners.append(("NOT CALIBRATED - press h", AMBER))
        if no_signal:
            banners.append(("NO SIGNAL - reconnecting camera...", MAGENTA))
        if moved:
            banners.append(("CAMERA MOVED - press h", MAGENTA))
        verify = [f"#{tid} {lab['reason'].replace('_', ' ')}" for tid, lab in out["labels"].items()
                  if lab["state"] == "SUSPICIOUS"]
        if verify:
            banners.append(("VERIFY: " + ", ".join(verify), MAGENTA))
        if time.monotonic() < notice["until"]:
            banners.append((notice["text"], notice["colour"]))
        if paused:
            banners.append(("PAUSED", WHITE))

        if ui["selected"] not in out["labels"]:
            ui["selected"] = None
        view = {"t": t, "fps": 1.0 / dt_avg if dt_avg else 0.0, "latency_ms": cam.latency_ms(t),
                "backend": cfg["detector"]["backend"], "tracks": tracks if homo.ready else [],
                "out": out, "claims": recent_claims if homo.ready else [], "rejected": rejected,
                "jammed": rf_sim.channel.jammed, "selected": ui["selected"], "banners": banners,
                "events": list(events.ticker), "time_in_state": lambda tid: events.time_in_state(tid, t)}
        ui["view"], ui["video_w"] = view, vis.shape[1]
        canvas = display.render(vis, view)
        ui["scale"] = min(1.0, max_w / canvas.shape[1])
        if ui["scale"] < 1.0:
            canvas = cv2.resize(canvas, None, fx=ui["scale"], fy=ui["scale"], interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ESC and ui["click_mode"]:
            ui["click_mode"] = False
            notify("Click calibration cancelled")
        if key == TAB and out["labels"]:
            ids = sorted(out["labels"])
            cur = ui["selected"]
            ui["selected"] = ids[(ids.index(cur) + 1) % len(ids)] if cur in ids else ids[0]
        if key == ord("b"):
            if isinstance(tracker.backend, DiffBackend):
                tracker.backend.set_background(frame)
                notify("Captured empty-arena background (arena must be empty)")
            else:
                notify(f"b only applies to the diff backend (current: {cfg['detector']['backend']})")
        if key in RED_TEAM_KEYS:
            msg = rf_sim.red_team(RED_TEAM_KEYS[key], t)
            events.add(t, "redteam", msg)
            notify(msg, 4.0, MAGENTA)
        if key == ord("p"):
            paused = not paused
        if key == ord("h"):
            moved = False
            if homo.calibrate(frame):
                homo.save(calib_file)
                notify("Calibrated from ArUco markers")
            else:
                notify("Markers not all visible: click the 4 corners instead", 5.0, AMBER)
                ui["clicks"].clear()
                ui["click_mode"] = True
        if args.max_frames and frames >= args.max_frames:
            break

    cam.release()
    events.close()
    cv2.destroyAllWindows()
    friendly = sum(1 for lab in out["labels"].values() if lab["state"] == "FRIENDLY")
    print(f"Processed {frames} frames, last t={t:.2f}s, calibrated={homo.ready}, "
          f"vehicles in last frame={len(tracks)}, friendly={friendly}")


if __name__ == "__main__":
    main()
