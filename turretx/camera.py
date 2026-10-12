"""Frame source: webcam index, video file, or stream URL.

Every frame comes with its timestamp `t` (seconds). This is the ONE clock for the whole
pipeline (design R3): the video's own timestamp for files, so playback speed doesn't
matter, and seconds since the camera was opened (monotonic, at grab time) for live sources.

Live sources use a grabber thread that keeps only the newest frame, so OpenCV's internal
buffer can't make latency build up (R13). If the live source drops (USB unplugged, phone app
closed), the thread keeps trying to reopen it; read() returns (None, None) meanwhile and
`connected` is False, so the app can show NO SIGNAL instead of exiting.
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
        self.width, self.height = width, height
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source {source!r}")
        self.connected = True
        self._t0 = time.monotonic()

        self._latest = None  # (seq, frame, t)
        self._last_seq = -1
        self._running = False
        self._cond = threading.Condition()
        if not self.is_file:
            self._running = True
            threading.Thread(target=self._grab_loop, daemon=True).start()

    def _open(self):
        if isinstance(self.source, int) and platform.system() == "Windows":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # R20: indices match check_env.py
        else:
            cap = cv2.VideoCapture(self.source)
        if not self.is_file and self.width and self.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _grab_loop(self):
        seq = 0
        while self._running:
            ok, frame = self.cap.read()
            t = time.monotonic() - self._t0
            if not ok:
                self.connected = False
                self.cap.release()
                time.sleep(1.0)
                if self._running:
                    self.cap = self._open()
                continue
            self.connected = True
            with self._cond:
                self._latest = (seq, frame, t)
                seq += 1
                self._cond.notify_all()

    def read(self, timeout=2.0):
        """Return (frame, t). (None, None) at end of file, or if no new live frame arrives within
        `timeout` (source dropped; it is being reopened in the background)."""
        if self.is_file:
            ok, frame = self.cap.read()
            if not ok:
                return None, None
            return frame, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        with self._cond:
            self._cond.wait_for(lambda: self._latest and self._latest[0] != self._last_seq,
                                timeout=timeout)
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
