"""Synthetic overhead scene with exact ground truth, for tests and offline development.

A flat table with the 4 ArUco markers and 3 cars, seen from an angled "phone" via a known
perspective warp. Car 'F-1' carries the pink roof sticker (the RF simulator's GPS stand-in).
"""
import math

import cv2
import numpy as np

TABLE_CM = (100, 60)        # distance between marker centres
PX_PER_CM = 10              # top-down canvas resolution
MARGIN_CM = 10              # canvas border around the marker centres
MARKER_CM = 8
OUT_SIZE = (1280, 720)
FPS = 30

CANVAS = ((TABLE_CM[0] + 2 * MARGIN_CM) * PX_PER_CM, (TABLE_CM[1] + 2 * MARGIN_CM) * PX_PER_CM)
# Where the canvas corners land in the output image: a trapezoid = camera tilted ~60°.
_SRC = np.float32([[0, 0], [CANVAS[0], 0], [CANVAS[0], CANVAS[1]], [0, CANVAS[1]]])
_DST = np.float32([[240, 90], [1040, 90], [1220, 690], [60, 690]])
WARP = cv2.getPerspectiveTransform(_SRC, _DST)  # canvas px → output px

CAR_CM = (8, 4)
CARS = {  # name: (BGR colour, sticker?)
    "F-1": ((200, 120, 40), True),
    "U-2": ((40, 160, 40), False),
    "U-3": ((30, 90, 200), False),
}


def canvas_px(x_cm, y_cm):
    return ((x_cm + MARGIN_CM) * PX_PER_CM, (y_cm + MARGIN_CM) * PX_PER_CM)


def cm_to_out_px(pts_cm):
    """Exact ground-truth projection of table cm → output image px."""
    pts = np.float32([canvas_px(x, y) for x, y in pts_cm]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, WARP).reshape(-1, 2)


def car_states(t):
    """Ground-truth {name: (x_cm, y_cm, heading_rad)} at time t (s)."""
    a = 0.6 * t  # F-1 drives an ellipse around the table centre
    f1 = (50 + 30 * math.cos(a), 30 + 15 * math.sin(a), math.atan2(15 * math.cos(a), -30 * math.sin(a)))
    u3x = 20 + 60 * (0.5 - 0.5 * math.cos(0.4 * t))  # U-3 slides back and forth along the bottom
    return {"F-1": f1, "U-2": (70, 20, 0.0), "U-3": (u3x, 48, 0.0)}


def _draw_car(img, x, y, heading, colour, sticker):
    cx, cy = canvas_px(x, y)
    size = (CAR_CM[0] * PX_PER_CM, CAR_CM[1] * PX_PER_CM)
    box = cv2.boxPoints(((cx, cy), size, math.degrees(heading))).astype(np.int32)
    cv2.fillPoly(img, [box], colour)
    if sticker:
        r = int(1.5 * PX_PER_CM)
        cv2.rectangle(img, (int(cx) - r, int(cy) - r), (int(cx) + r, int(cy) + r), (180, 40, 230), -1)


def render(t, cars=True):
    img = np.full((CANVAS[1], CANVAS[0], 3), 205, np.uint8)  # light grey table
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = MARKER_CM * PX_PER_CM
    for marker_id, (x, y) in enumerate([(0, 0), (TABLE_CM[0], 0), TABLE_CM, (0, TABLE_CM[1])]):
        cx, cy = canvas_px(x, y)
        tag = cv2.aruco.generateImageMarker(dictionary, marker_id, m)
        # white quiet zone around the marker, as on a printed sheet
        x0, y0 = int(cx - m / 2), int(cy - m / 2)
        cv2.rectangle(img, (x0 - 15, y0 - 15), (x0 + m + 15, y0 + m + 15), (255, 255, 255), -1)
        img[y0:y0 + m, x0:x0 + m] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    for name, (x, y, h) in (car_states(t).items() if cars else []):
        colour, sticker = CARS[name]
        _draw_car(img, x, y, h, colour, sticker)
    out = cv2.warpPerspective(img, WARP, OUT_SIZE, borderValue=(60, 60, 60))
    return cv2.GaussianBlur(out, (3, 3), 0)


def car_velocity(name, t, dt=1e-3):
    """Ground-truth (vx, vy) in cm/s by central difference."""
    (x0, y0, _), (x1, y1, _) = car_states(t - dt)[name], car_states(t + dt)[name]
    return (x1 - x0) / (2 * dt), (y1 - y0) / (2 * dt)


def write_clip(path, seconds=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, OUT_SIZE)
    for i in range(int(seconds * FPS)):
        writer.write(render(i / FPS))
    writer.release()
