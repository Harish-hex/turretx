"""Write recordings/synthetic.mp4: a 10 s fake overhead scene with known car positions.
Run from the repo root:  python scripts/make_synthetic_clip.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turretx.synthetic import write_clip  # noqa: E402

out = Path("recordings/synthetic.mp4")
out.parent.mkdir(exist_ok=True)
write_clip(out)
print(f"wrote {out}")
