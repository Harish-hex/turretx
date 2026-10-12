"""Velocity from a short position history (least-squares slope over the last ~0.5 s).

A line fit over a window reports roughly the velocity at the window's midpoint, so longer
windows lag on curves; 0.5 s balanced lag against jitter on the synthetic test (Phase 2).
"""
import math
from collections import deque

import numpy as np

STATIONARY_CM_S = 2.0
DEFAULT_WINDOW_S = 0.5
MAX_SPEED_CM_S = 250.0  # faster than any RC car here: a bigger jump is a merge/split/ID swap


class History:
    def __init__(self, window_s=DEFAULT_WINDOW_S):
        self.window_s = window_s
        self.samples = deque()  # (t, x, y)

    def add(self, t, x, y):
        """Add a sample. An implausible jump means the track is not the same physical motion
        (blobs merging/splitting, tracker ID swap): drop the history instead of fitting a line
        through the jump, which would fake a huge velocity (found in the Phase 3 end-to-end run)."""
        if self.samples:
            pt, px, py = self.samples[-1]
            if t > pt and math.hypot(x - px, y - py) / (t - pt) > MAX_SPEED_CM_S:
                self.reset()
        self.samples.append((t, x, y))
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def reset(self):
        self.samples.clear()

    @property
    def settled(self):
        """True once the history spans most of the window: the velocity can be trusted."""
        return len(self.samples) >= 3 and self.samples[-1][0] - self.samples[0][0] >= 0.8 * self.window_s

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
