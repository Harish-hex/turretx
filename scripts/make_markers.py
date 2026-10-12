"""Write markers/markers.pdf (and marker_<id>.png): the 4 corner ArUco markers (DICT_4X4_50,
ids 0-3) at 300 DPI with a white border and a label. Default 15 cm, sized for a floor arena
filmed from ~2 m up (one per A4 page); use --cm 8 for a small table setup.
Print at 100% scale ("actual size", no fit-to-page), then check with a ruler that the black
square is the stated size. Tape them at the arena corners: 0 = top-left, 1 = top-right,
2 = bottom-right, 3 = bottom-left (as seen in the camera image), then measure the distance
between marker CENTRES into config.yaml (table_cm).
Run from the repo root:  python scripts/make_markers.py [--cm 15]"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image  # installed with torchvision; embeds DPI so prints come out 8 cm

DPI = 300
BORDER_CM = 1
NAMES = ["0 TOP-LEFT", "1 TOP-RIGHT", "2 BOTTOM-RIGHT", "3 BOTTOM-LEFT"]

ap = argparse.ArgumentParser()
ap.add_argument("--cm", type=float, default=15, help="black square size in cm (max ~18 for A4)")
MARKER_CM = ap.parse_args().cm
px = round(MARKER_CM / 2.54 * DPI)
border = round(BORDER_CM / 2.54 * DPI)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
out = Path("markers")
out.mkdir(exist_ok=True)
pages = []
for marker_id, name in enumerate(NAMES):
    tag = cv2.aruco.generateImageMarker(dictionary, marker_id, px)
    sheet = np.full((px + 2 * border + 120, px + 2 * border), 255, np.uint8)
    sheet[border:border + px, border:border + px] = tag
    cv2.putText(sheet, f"{name}  ({MARKER_CM:g} cm)", (border, px + 2 * border + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    page = Image.fromarray(sheet)
    path = out / f"marker_{marker_id}.png"
    page.save(path, dpi=(DPI, DPI))
    pages.append(page)
    print(f"wrote {path}")
pages[0].save(out / "markers.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
print(f"wrote {out / 'markers.pdf'} (one marker per page)")
print(f"Print at 100% / actual size, then check the black square measures {MARKER_CM:g} cm.")
