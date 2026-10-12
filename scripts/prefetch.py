"""Download model weights and trigger Ultralytics' lazy installs now, while internet is good,
so nothing tries to download mid-demo. Safe to re-run."""
import numpy as np

blank = np.zeros((480, 640, 3), dtype=np.uint8)


def step(name, fn):
    try:
        fn()
        print(f"[ok]   {name}")
    except Exception as e:  # keep going; report what failed
        print(f"[FAIL] {name}: {e}")


def yolo():
    from ultralytics import YOLO

    YOLO("yolov8n.pt")(blank, verbose=False)


def bytetrack():
    from ultralytics import YOLO

    YOLO("yolov8n.pt").track(blank, tracker="bytetrack.yaml", persist=True, verbose=False)


def yolo_world():
    from ultralytics import YOLOWorld

    m = YOLOWorld("yolov8s-world.pt")
    m.set_classes(["toy car"])  # first call installs the CLIP text encoder (needs git)
    m(blank, verbose=False)


step("YOLOv8n weights", yolo)
step("ByteTrack tracker", bytetrack)
step("YOLO-World weights + CLIP", yolo_world)
