"""Velocity from a short position history (least-squares slope over the last ~0.5 s).

A line fit over a window reports roughly the velocity at the window's midpoint, so longer
windows lag on curves; 0.5 s balanced lag against jitter on the synthetic test (Phase 2).
"""
import math
from collections import deque

import numpy as np

STATIONARY_CM_S = 2.0
DEFAULT_WINDOW_S = 0.5


class History:
    def __init__(self, window_s=DEFAULT_WINDOW_S):
        self.window_s = window_s
        self.samples = deque()  # (t, x, y)

    def add(self, t, x, y):
        self.samples.append((t, x, y))
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    @property
    def last_t(self):
        return self.samples[-1][0] if self.samples else None

    def velocity(self):
        """(vx, vy) in cm/s; zero until there are ≥3 samples over ≥0.2 s, and below STATIONARY_CM_S."""
        if len(self.samples) < 3:
            return 0.0, 0.0
        s = np.asarray(self.samples, dtype=np.float64)
        t = s[:, 0] - s[0, 0]
        if t[-1] < 0.2:
            return 0.0, 0.0
        vx = np.polyfit(t, s[:, 1], 1)[0]
        vy = np.polyfit(t, s[:, 2], 1)[0]
        if math.hypot(vx, vy) < STATIONARY_CM_S:
            return 0.0, 0.0
        return float(vx), float(vy)


def motion(vx, vy):
    """(speed cm/s, heading rad) where heading 0 = +x, π/2 = +y (table frame)."""
    return math.hypot(vx, vy), math.atan2(vy, vx)
