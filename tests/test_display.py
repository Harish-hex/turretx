import numpy as np

from turretx import synthetic
from turretx.config import load_config
from turretx.display import AMBER, GREEN, MAGENTA, STATE_COLOUR, Display, short_label
from turretx.events import EventLog
from turretx.homography import Homography


def lab(tid, state, reason=None, rf_id=None, conf=0.0, coast_s=None):
    return {"track_id": tid, "state": state, "reason": reason, "rf_id": rf_id, "confidence": conf,
            "scores": {"P": conf, "M": conf, "R": conf}, "flags": [], "coast_s": coast_s, "last_friendly_t": None}


def view(out, tracks, claims=(), selected=None):
    return {"t": 1.0, "fps": 30.0, "latency_ms": 42.0, "backend": "diff", "tracks": tracks, "out": out,
            "claims": list(claims), "rejected": 0, "jammed": False, "selected": selected, "banners": [],
            "events": [(0.5, "label", "#1 F-1: FRIENDLY")], "time_in_state": lambda tid: 2.0}


def test_no_red_is_used_for_any_label():
    """R6/R21: nothing should read as 'enemy'. Pure-ish red (BGR, R high, G/B low) is banned."""
    for colour in STATE_COLOUR.values():
        b, g, r = colour
        assert not (r > 200 and g < 80 and b < 80)


def test_short_labels():
    assert short_label(lab(1, "FRIENDLY", None, "F-1", 0.86)) == "F-1 0.86"
    assert short_label(lab(1, "FRIENDLY", "COASTING", "F-1", 0.9, coast_s=23.4)) == "F-1 NO RF 23s"
    assert short_label(lab(1, "UNKNOWN", "NO_ID")) == "NO ID"
    assert short_label(lab(1, "UNKNOWN", "TOO_CLOSE")) == "TOO CLOSE"
    assert short_label(lab(1, "SUSPICIOUS", "ID_CONFLICT", "F-1")) == "VERIFY: ID CONFLICT"
    assert short_label(lab(1, "SUSPICIOUS", "MOTION_MISMATCH", "F-1")) == "VERIFY: MOTION"


def test_render_every_state_and_overlay():
    homo = Homography(synthetic.TABLE_CM)
    assert homo.calibrate(synthetic.render(0))
    d = Display(load_config(), homo)
    frame = synthetic.render(1.0)
    tracks = [{"track_id": i, "x": 20.0 * i, "y": 30.0, "vx": 5.0, "vy": 0.0, "speed": 5.0,
               "bbox": [100 * i, 300, 100 * i + 60, 340]} for i in range(1, 5)]
    out = {"labels": {1: lab(1, "FRIENDLY", None, "F-1", 0.9), 2: lab(2, "FRIENDLY", "COASTING", "F-2", 0.8, 12.0),
                      3: lab(3, "UNKNOWN", "TOO_CLOSE"), 4: lab(4, "SUSPICIOUS", "ID_CONFLICT", "F-1", 0.7)},
           "ghosts": [{"rf_id": "F-1", "x": 90.0, "y": 50.0, "conflict": True}], "rf_denied": True,
           "lost_friendlies": [{"rf_id": "F-3", "x": 50.0, "y": 20.0, "vx": 1.0, "vy": 0.0, "age_s": 7.0}]}
    claims = [{"rf_id": "F-1", "x": 20.0, "y": 30.0, "conflict": True}]
    img = d.render(frame, view(out, tracks, claims, selected=4))
    h, w = frame.shape[:2]
    assert img.shape[0] == h and img.shape[1] > w  # side panel appended
    assert (img[:, w:] != 0).any()


def test_pick_selects_vehicle_under_click():
    tracks = [{"track_id": 7, "bbox": [10, 10, 50, 50]}]
    v = {"tracks": tracks}
    assert Display.pick(v, 30, 30) == 7
    assert Display.pick(v, 80, 80) is None


def test_event_log_records_changes_once(tmp_path):
    ev = EventLog(log_dir=tmp_path, sound=False)
    base = {"ghosts": [], "rf_denied": False, "lost_friendlies": []}
    ev.update(0.0, dict(base, labels={1: lab(1, "UNKNOWN", "NO_ID")}))
    ev.update(0.1, dict(base, labels={1: lab(1, "UNKNOWN", "NO_ID")}))           # no change
    ev.update(1.2, dict(base, labels={1: lab(1, "FRIENDLY", None, "F-1", 0.9)}))
    ev.update(3.0, dict(base, labels={1: lab(1, "FRIENDLY", None, "F-1", 0.9)}, rf_denied=True),
              rejections=[{"t": 3.0, "reason": "bad_signature", "rf_id": None}])
    ev.close()
    kinds = [k for _, k, _ in ev.ticker]
    assert kinds == ["label", "label", "rf_rejected", "rf_denied"]
    assert abs(ev.time_in_state(1, 3.0) - 1.8) < 1e-9
    lines = ev.path.read_text().splitlines()
    assert len(lines) == 4 and '"state": "FRIENDLY"' in lines[1]


def test_rf_only_view_draws_friendlies_without_video_and_uses_no_red():
    homo = Homography(synthetic.TABLE_CM)
    assert homo.calibrate(synthetic.render(0))
    d = Display(load_config(), homo)
    frame = synthetic.render(1.0)
    out = {"labels": {}, "ghosts": [], "rf_denied": False, "lost_friendlies": []}
    claims = {"F-1": {"rf_id": "F-1", "t": 0.9, "x": 60.0, "y": 40.0, "vx": 3.0, "vy": 0.0, "conflict": False},
              "F-2": {"rf_id": "F-2", "t": -9.0, "x": 120.0, "y": 90.0, "vx": 0.0, "vy": 0.0, "conflict": False}}
    for rung in ("L2", "L3", "L4"):
        v = dict(view(out, []), rung=rung, last_claims=claims)
        img = d.render(frame, v)
        w = frame.shape[1]
        # the video is gone: left half is the flat dark background plus overlay colours only
        assert not np.array_equal(img[:, :w], frame)
        px = img[:, :w].reshape(-1, 3)
        red = (px[:, 2] > 180) & (px[:, 1] < 80) & (px[:, 0] < 80)
        assert not red.any()
