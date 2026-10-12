"""Frame source: webcam index, video file, or stream URL.

Every frame comes with its timestamp `t` (seconds). This is the ONE clock for the whole
pipeline (design R3): the video's own timestamp for files, so playback speed doesn't
matter, and time.monotonic() at grab time for live sources.

Live sources use a grabber thread that keeps only the newest frame, so OpenCV's internal
buffer can't make latency build up (R13).
"""
import platform
import threading
import time

import cv2


def _is_file(source):
    return isinstance(source, str) and not source.lower().startswith(("http://", "https://", "rtsp://"))


class Camera:
    def __init__(self, source, width=None, height=None):
        self.source = source
        self.is_file = _is_file(source)
        if isinstance(source, int) and platform.system() == "Windows":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # R20: indices match check_env.py
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source {source!r}")
        if not self.is_file and width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._latest = None  # (seq, frame, t)
        self._last_seq = -1
        self._running = False
        self._cond = threading.Condition()
        if not self.is_file:
            self._running = True
            threading.Thread(target=self._grab_loop, daemon=True).start()

    def _grab_loop(self):
        seq = 0
        while self._running:
            ok, frame = self.cap.read()
            t = time.monotonic()
            with self._cond:
                if not ok:
                    self._running = False
                else:
                    self._latest = (seq, frame, t)
                    seq += 1
                self._cond.notify_all()

    def read(self, timeout=2.0):
        """Return (frame, t), or (None, None) at end of file / when the live source dies."""
        if self.is_file:
            ok, frame = self.cap.read()
            if not ok:
                return None, None
            return frame, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        with self._cond:
            self._cond.wait_for(
                lambda: not self._running or (self._latest and self._latest[0] != self._last_seq),
                timeout=timeout,
            )
            if not self._latest or self._latest[0] == self._last_seq:
                return None, None
            seq, frame, t = self._latest
            self._last_seq = seq
            return frame, t

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def release(self):
        self._running = False
        self.cap.release()
