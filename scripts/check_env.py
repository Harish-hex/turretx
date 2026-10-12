"""Verify the TurretX environment: Python, torch + GPU, OpenCV/ArUco, and which camera indices open."""
import platform
import sys


def main():
    print(f"Python      {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    if sys.version_info[:2] != (3, 12):
        print("  WARNING: expected Python 3.12 (run inside .venv)")

    import torch

    print(f"torch       {torch.__version__}")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU       {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        print("  GPU       Apple MPS")
    else:
        device = "cpu"
        print("  WARNING: no GPU detected, running on CPU (slow)")
    x = torch.randn(512, 512, device=device)
    _ = (x @ x).sum().item()
    print(f"  matmul on {device}: OK")

    import cv2

    print(f"opencv      {cv2.__version__}")
    print(f"  aruco     {'OK' if hasattr(cv2, 'aruco') else 'MISSING'}")

    import scipy
    import ultralytics

    print(f"scipy       {scipy.__version__}")
    print(f"ultralytics {ultralytics.__version__}")

    # Phone webcams (Iriun/DroidCam) usually appear as index 1 or 2; index 0 is the built-in camera.
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    print("cameras:")
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            h, w = frame.shape[:2]
            print(f"  index {i}: {w}x{h}")
        cap.release()


if __name__ == "__main__":
    main()
