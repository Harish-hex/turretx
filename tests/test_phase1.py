import numpy as np
import pytest

from turretx import synthetic
from turretx.camera import Camera
from turretx.config import load_config, parse_source
from turretx.homography import Homography


def make_h():
    return Homography(synthetic.TABLE_CM)


def test_config_loads_and_parses_source():
    cfg = load_config()
    assert cfg["table_cm"] == [100, 60]
    assert parse_source(1) == 1
    assert parse_source("2") == 2
    assert parse_source("recordings/a.mp4") == "recordings/a.mp4"


def test_calibrate_from_aruco_maps_ground_truth_within_1cm():
    h = make_h()
    assert h.calibrate(synthetic.render(0.0))
    truth_cm = [(50, 30), (10, 50), (90, 5), (0, 0), (100, 60)]
    est_cm = h.to_cm(synthetic.cm_to_out_px(truth_cm))
    assert np.abs(est_cm - np.float32(truth_cm)).max() < 1.0


def test_roundtrip_cm_px_cm():
    h = make_h()
    assert h.calibrate(synthetic.render(0.0))
    pts = np.float32([[12.5, 33.0], [80, 10]])
    assert np.allclose(h.to_cm(h.to_px(pts)), pts, atol=1e-3)


def test_calibrate_fails_without_all_markers():
    frame = synthetic.render(0.0)
    frame[:, :400] = 60  # hide the left-hand markers
    assert not make_h().calibrate(frame)


def test_drift_detects_moved_camera():
    h = make_h()
    frame = synthetic.render(0.0)
    assert h.calibrate(frame)
    assert h.drift_px(frame) < 2
    shifted = np.roll(frame, 25, axis=1)  # camera knocked 25 px sideways
    assert h.drift_px(shifted) > 20


def test_inside_zone():
    h = make_h()
    assert h.inside(50, 30)
    assert not h.inside(-1, 30)
    assert not h.inside(99, 30, margin=3)


def test_save_and_load_calibration(tmp_path):
    h = make_h()
    assert h.calibrate(synthetic.render(0.0))
    h.save(tmp_path / "h.json")
    h2 = make_h()
    assert h2.load(tmp_path / "h.json")
    assert np.allclose(h.H, h2.H)
    assert not Homography((50, 50)).load(tmp_path / "h.json")  # different table size


def test_file_camera_timestamps_follow_video_not_wall_clock(tmp_path):
    clip = tmp_path / "clip.mp4"
    synthetic.write_clip(clip, seconds=1)
    cam = Camera(str(clip))
    ts = []
    while True:
        frame, t = cam.read()
        if frame is None:
            break
        ts.append(t)
    cam.release()
    assert len(ts) == pytest.approx(synthetic.FPS, abs=1)
    steps = np.diff(ts)
    assert np.all(steps > 0)
    assert np.allclose(steps, 1 / synthetic.FPS, atol=0.005)
