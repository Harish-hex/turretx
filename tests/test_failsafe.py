"""Camera-loss failsafe (EVAL2_PLAN §1): health classifier, fallback rung, re-arm interlock."""
import numpy as np

from turretx.failsafe import Failsafe, select_rung


def img(v=120, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    f = np.full((60, 80, 3), v, np.int16)
    if noise:
        f += rng.integers(-noise, noise + 1, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def feed(fs, n, t0=0.0, dt=1 / 30, frame=lambda i: img(120, 3, i), **kw):
    rung = None
    for i in range(n):
        rung = fs.update(t0 + i * dt, frame(i), **kw)
    return rung


# ---------------------------------------------------------------- rung selection
def test_rung_table():
    assert select_rung("OK", True, True) == "L0"
    assert select_rung("DEGRADED", True, True) == "L1"
    assert select_rung("LOST", True, True) == "L2"
    assert select_rung("LOST", False, True) == "L3"
    assert select_rung("LOST", False, False) == "L4"


def test_camera_ok_never_leaves_l0_even_if_rf_dies():
    # RF loss is the matcher's RF DENIED business; the camera still sees vehicles
    assert select_rung("OK", False, False) == "L0"


# ---------------------------------------------------------------- health
def test_healthy_stream_stays_l0():
    assert feed(Failsafe(), 90, rf_alive=True) == "L0"


def test_no_frames_for_half_a_second_is_lost():
    fs = Failsafe()
    feed(fs, 10, rf_alive=True)
    assert fs.update(0.30, None, rf_alive=True) == "L0"  # a short gap is tolerated
    assert fs.update(0.9, None, rf_alive=True) == "L2"


def test_black_frame_is_lost_and_recovers_when_light_returns():
    fs = Failsafe()
    feed(fs, 10, rf_alive=True)
    assert fs.update(0.5, img(2), rf_alive=True) == "L2"
    assert feed(fs, 10, t0=0.6, rf_alive=True) == "L0"


def test_frozen_frame_lost_only_when_rf_says_vehicles_move():
    still = lambda i: img(120, 0)  # byte-identical frames
    fs = Failsafe()
    assert feed(fs, 90, frame=still, rf_alive=True, rf_moving=False) == "L0"  # empty static scene
    fs = Failsafe()
    assert feed(fs, 90, frame=still, rf_alive=True, rf_moving=True) == "L2"


def test_noisy_live_stream_is_not_frozen():
    assert feed(Failsafe(), 90, rf_alive=True, rf_moving=True) == "L0"


def test_homography_drift_is_degraded_not_lost():
    assert feed(Failsafe(), 10, rf_alive=True, drifted=True) == "L1"


def test_operator_kill_is_lost_until_released():
    fs = Failsafe()
    feed(fs, 5, rf_alive=True)
    fs.kill(True)
    assert feed(fs, 5, t0=0.2, rf_alive=True) == "L2"
    fs.kill(False)
    assert feed(fs, 5, t0=0.4, rf_alive=True) == "L0"


def test_rf_stale_drops_to_last_known_then_safe_state():
    fs = Failsafe()
    fs.kill(True)
    assert feed(fs, 3, rf_alive=False, has_last_known=True) == "L3"
    assert feed(fs, 3, t0=0.2, rf_alive=False, has_last_known=False) == "L4"


# ---------------------------------------------------------------- interlock
def test_designation_allowed_only_in_l0():
    fs = Failsafe()
    feed(fs, 5, rf_alive=True)
    assert fs.designation_allowed


def test_no_automatic_rearm_after_camera_returns():
    fs = Failsafe()
    feed(fs, 5, rf_alive=True)
    fs.kill(True)
    feed(fs, 5, t0=0.2, rf_alive=True)
    assert not fs.designation_allowed
    fs.kill(False)
    assert feed(fs, 5, t0=0.4, rf_alive=True) == "L0"
    assert not fs.designation_allowed          # camera is back, but nobody has re-confirmed
    assert fs.rearm()
    assert fs.designation_allowed


def test_rearm_refused_while_camera_is_down():
    fs = Failsafe()
    fs.kill(True)
    feed(fs, 5, rf_alive=True)
    assert not fs.rearm()
    assert not fs.designation_allowed


def test_drift_also_locks_designation():
    fs = Failsafe()
    feed(fs, 5, rf_alive=True, drifted=True)
    assert not fs.designation_allowed
    feed(fs, 5, t0=0.2, rf_alive=True)
    assert not fs.designation_allowed
    assert fs.rearm() and fs.designation_allowed
