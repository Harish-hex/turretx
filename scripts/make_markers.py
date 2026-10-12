"""Write markers/markers.pdf (and marker_<id>.png): the 4 corner ArUco markers (DICT_4X4_50, ids 0-3), 8 cm wide
at 300 DPI with a white border and a label. Print at 100% scale ("actual size", no fit-to-page),
then check with a ruler that the black square is 8.0 cm.
Tape them at the table corners: 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left
(as seen from the camera), then measure the distance between marker CENTRES into config.yaml.
Run from the repo root:  python scripts/make_markers.py"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image  # installed with torchvision; embeds DPI so prints come out 8 cm

DPI = 300
MARKER_CM = 8
BORDER_CM = 1
NAMES = ["0 TOP-LEFT", "1 TOP-RIGHT", "2 BOTTOM-RIGHT", "3 BOTTOM-LEFT"]

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
    cv2.putText(sheet, f"{name}  ({MARKER_CM} cm)", (border, px + 2 * border + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    page = Image.fromarray(sheet)
    path = out / f"marker_{marker_id}.png"
    page.save(path, dpi=(DPI, DPI))
    pages.append(page)
    print(f"wrote {path}")
pages[0].save(out / "markers.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
print(f"wrote {out / 'markers.pdf'} (one marker per page)")
print("Print at 100% / actual size, then check the black square measures 8.0 cm.")
