"""Write recordings/synthetic.mp4: a 10 s fake overhead scene with known car positions.
Run from the repo root:  python scripts/make_synthetic_clip.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402

from turretx.synthetic import render, write_clip  # noqa: E402

out = Path("recordings/synthetic.mp4")
out.parent.mkdir(exist_ok=True)
write_clip(out)
cv2.imwrite("recordings/synthetic_bg.png", render(0, cars=False))  # empty table for the diff backend
print(f"wrote {out} and recordings/synthetic_bg.png")
