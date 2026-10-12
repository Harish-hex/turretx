"""Camera-loss failsafe (design: reports/Camera failsafe and airborne relay.md, EVAL2_PLAN §1).

Pure logic, no I/O: main.py feeds it one call per loop iteration and acts on the rung.

  L0 full fusion       camera healthy
  L1 camera degraded   homography drift: labels stay, designation locked
  L2 RF-only           camera lost, authenticated claims still arriving: friendly map, no video labels
  L3 last-known        camera lost and claims stale: last-known positions with growing no-fire rings
  L4 safe state        camera lost, nothing known: display frozen / stale

Two rules: missing data only moves labels toward UNKNOWN and designation toward OFF; and nothing
re-arms by itself. Once the camera is back the operator must confirm (`rearm`).

Camera health checks the silent failures too, not only a dead feed: a black frame (lens covered)
and a frozen frame (stream alive but repeating). A frozen frame only counts when RF says vehicles
are moving, since a genuinely empty, noiseless scene looks identical to a freeze.
"""
import cv2
import numpy as np

LOST_S = 0.5           # no new frame for this long -> LOST   (engineering choice, not a standard)
BLACK_MEAN = 8.0       # mean grey level below this -> lens covered / sensor dead
FROZEN_FRAMES = 30     # ~1 s of byte-identical frames while RF says things move
FROZEN_DIFF = 0.05     # mean abs grey difference (0-255) treated as "identical"


def select_rung(camera, rf_alive, has_last_known):
    if camera == "OK":
        return "L0"
    if camera == "DEGRADED":
        return "L1"
    if rf_alive:
        return "L2"
    return "L3" if has_last_known else "L4"


class Failsafe:
    def __init__(self, lost_s=LOST_S, black_mean=BLACK_MEAN, frozen_frames=FROZEN_FRAMES,
                 frozen_diff=FROZEN_DIFF):
        self.lost_s, self.black_mean = lost_s, black_mean
        self.frozen_frames, self.frozen_diff = frozen_frames, frozen_diff
        self.camera = "OK"
        self.rung = "L0"
        self.reason = ""
        self._killed = False
        self._locked = False
        self._last_frame_at = None
        self._prev = None
        self._same = 0

    def kill(self, on):
        """Operator / demo switch: treat the camera as lost."""
        self._killed = bool(on)

    @property
    def killed(self):
        return self._killed

    def _grey(self, frame):
        return cv2.cvtColor(cv2.resize(frame, (80, 45), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.int16)

    def _health(self, now, frame, rf_moving, drifted):
        if self._killed:
            return "LOST", "camera disabled by operator"
        if frame is None:
            if self._last_frame_at is None:
                self._last_frame_at = now
            if now - self._last_frame_at > self.lost_s:
                return "LOST", "no frames"
            return self.camera, self.reason  # short gap: keep the previous verdict
        self._last_frame_at = now
        g = self._grey(frame)
        if g.mean() < self.black_mean:
            self._prev, self._same = g, 0
            return "LOST", "black frame (lens covered?)"
        same = self._prev is not None and np.abs(g - self._prev).mean() < self.frozen_diff
        self._same = self._same + 1 if same else 0
        self._prev = g
        if rf_moving and self._same >= self.frozen_frames:
            return "LOST", "frozen frame"
        if drifted:
            return "DEGRADED", "camera moved"
        return "OK", ""

    def update(self, now, frame=None, rf_alive=True, has_last_known=False, rf_moving=False,
               drifted=False):
        """One call per loop. frame=None means no new frame arrived. Returns the rung."""
        self.camera, self.reason = self._health(now, frame, rf_moving, drifted)
        self.rung = select_rung(self.camera, rf_alive, has_last_known)
        if self.rung != "L0":
            self._locked = True
        return self.rung

    @property
    def designation_allowed(self):
        return self.rung == "L0" and not self._locked

    def rearm(self):
        """Operator confirmation. Refused unless the camera is healthy again."""
        if self.rung != "L0":
            return False
        self._locked = False
        return True

    @property
    def locked(self):
        return self._locked
