import ast
from pathlib import Path

import cv2
import numpy as np
import pytest

from turretx import synthetic
from turretx.config import load_config
from turretx.homography import Homography
from turretx.kinematics import History
from turretx.taglocator import TagLocator
from turretx.tracker import CentroidTracker, Tracker


@pytest.fixture(scope="module")
def homo():
    h = Homography(synthetic.TABLE_CM)
    assert h.calibrate(synthetic.render(0.0))
    return h


def diff_cfg():
    cfg = load_config()
    cfg["detector"].update(backend="diff", ground_point_frac=0.5)  # synthetic cars are flat
    cfg["detector"]["diff"]["background"] = None
    return cfg


def separated(states, min_cm=14):
    pts = [(x, y) for x, y, _ in states.values()]
    return all(np.hypot(a[0] - b[0], a[1] - b[1]) >= min_cm
               for i, a in enumerate(pts) for b in pts[i + 1:])


def test_history_velocity_linear_motion():
    h = History()
    for i in range(31):
        t = i / 30
        h.add(t, 10 + 12 * t, 5 - 4 * t)
    vx, vy = h.velocity()
    assert vx == pytest.approx(12, abs=1e-6) and vy == pytest.approx(-4, abs=1e-6)


def test_history_stationary_jitter_is_zero():
    rng = np.random.default_rng(0)
    h = History()
    for i in range(31):
        h.add(i / 30, 50 + rng.normal(0, 0.2), 30 + rng.normal(0, 0.2))
    assert h.velocity() == (0.0, 0.0)


def test_centroid_tracker_keeps_ids_and_survives_short_gaps():
    ct = CentroidTracker(max_dist_px=50, max_missed=5)
    a, b = ct.update([(0, 0, 10, 10), (200, 0, 210, 10)])
    assert (a[0], b[0]) == (1, 2)
    moved = ct.update([(205, 5, 215, 15), (5, 5, 15, 15)])  # order swapped, small moves
    assert {tid for tid, _ in moved} == {1, 2}
    assert dict((tuple(bx), tid) for tid, bx in moved)[(5, 5, 15, 15)] == 1
    ct.update([(5, 5, 15, 15)])  # car 2 missed one frame
    back = ct.update([(5, 5, 15, 15), (205, 5, 215, 15)])
    assert sorted(tid for tid, _ in back) == [1, 2]
    far = ct.update([(600, 600, 610, 610)])  # beyond gate → new id
    assert far[0][0] == 3


def test_diff_tracker_accuracy_on_synthetic(homo):
    tr = Tracker(diff_cfg(), homo)
    tr.backend.set_background(synthetic.render(0, cars=False))
    pos_err, vel_err, misses = [], [], 0
    for i in range(300):
        t = i / synthetic.FPS
        tracks = tr.update(synthetic.render(t), t)
        states = synthetic.car_states(t)
        if not separated(states):
            continue  # merged blobs are a known limitation (R16)
        for name, (x, y, _) in states.items():
            d = [np.hypot(k["x"] - x, k["y"] - y) for k in tracks]
            if not d or min(d) > 6:
                misses += 1
                continue
            k = tracks[int(np.argmin(d))]
            pos_err.append(min(d))
            if t > 1.5:
                tvx, tvy = synthetic.car_velocity(name, t)
                vel_err.append(np.hypot(k["vx"] - tvx, k["vy"] - tvy))
    assert misses == 0
    assert np.mean(pos_err) < 1.0
    assert np.mean(vel_err) < 4.0


def test_diff_rejects_arm_reaching_in_from_outside(homo):
    tr = Tracker(diff_cfg(), homo)
    bg = synthetic.render(0, cars=False)
    tr.backend.set_background(bg)
    frame = bg.copy()
    # an "arm" from the bottom image edge onto the table, touching nothing else
    cv2.rectangle(frame, (600, 520), (660, 719), (90, 120, 160), -1)
    assert tr.update(frame, 0.0) == []
    cv2.rectangle(frame, (400, 300), (460, 330), (30, 90, 200), -1)  # a real car-sized blob
    assert len(tr.update(frame, 0.04)) == 1


def test_tag_locator_tracks_friendly_sticker(homo):
    loc = TagLocator(load_config()["tags"], homo)
    pos_err, vel_err = [], []
    for i in range(150):
        t = i / synthetic.FPS
        fix = loc.update(synthetic.render(t), t)[0]
        assert fix["rf_id"] == "F-1" and fix["visible"]
        x, y, _ = synthetic.car_states(t)["F-1"]
        pos_err.append(np.hypot(fix["x"] - x, fix["y"] - y))
        if t > 1.5:
            tvx, tvy = synthetic.car_velocity("F-1", t)
            vel_err.append(np.hypot(fix["vx"] - tvx, fix["vy"] - tvy))
    assert np.max(pos_err) < 1.0
    assert np.mean(vel_err) < 3.0


def test_tag_locator_reports_invisible_when_sticker_hidden(homo):
    loc = TagLocator(load_config()["tags"], homo)
    assert loc.update(synthetic.render(0, cars=False), 0.0) == [{"rf_id": "F-1", "t": 0.0, "visible": False}]


@pytest.mark.parametrize("module", ["tracker.py", "matcher.py"])
def test_firewall_identity_pipeline_never_imports_taglocator(module):
    """R1: the sticker is the simulator's GPS stand-in only; identification must not see it."""
    path = Path(__file__).resolve().parent.parent / "turretx" / module
    if not path.exists():
        pytest.skip(f"{module} not written yet")
    tree = ast.parse(path.read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("taglocator" in m for m in imported)


def test_yolo_backend_excludes_people():
    from turretx.tracker import YoloBackend

    cfg = load_config()["detector"]
    b = YoloBackend(cfg)
    names = b.model.names
    assert all(names[i] != "person" for i in b.classes)
    assert any(names[i] == "car" for i in b.classes)


def test_tag_offset_is_subtracted(homo):
    cfg = load_config()["tags"]
    plain = TagLocator(cfg, homo).update(synthetic.render(1.0), 1.0)[0]
    cfg["F-1"]["offset_cm"] = [1.5, -4.0]
    shifted = TagLocator(cfg, homo).update(synthetic.render(1.0), 1.0)[0]
    assert shifted["x"] == pytest.approx(plain["x"] - 1.5)
    assert shifted["y"] == pytest.approx(plain["y"] + 4.0)
