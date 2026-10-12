# TurretX: Dependencies & Machine Setup

Two machines, one git repo:

| Machine | Role | GPU path |
|---|---|---|
| **Windows laptop, RTX 3050** | **Main machine**: runs the live demo with the phone camera | CUDA (`cu126` PyTorch build) |
| **Mac, Apple M1** | Writing code; develops against recorded clips | MPS (Apple GPU) |

Workflow: write code on the Mac and `git push`, then `git pull` on the 3050 and run it live. Record short clips on the 3050, commit them to `recordings/`, and develop against them on the Mac.

---

## 1. Software stack

| Package | Why | Notes |
|---|---|---|
| Python **3.12** | Runtime | Managed by `uv`. Don't use the Mac's system Python 3.14, which is too new for torch/ultralytics. |
| `uv` | Python + venv + package manager | Installed by the setup scripts |
| `torch`, `torchvision` | Runs YOLO on the GPU | **Windows: CUDA 12.6 build.** Mac: default build (MPS). Installed by the setup script, **not** from requirements.txt. |
| `ultralytics` | YOLOv8, YOLO-World, ByteTrack tracking, training | |
| `opencv-python` | Camera capture, homography, ArUco markers, display window | `cv2.aruco` is included |
| `numpy`, `scipy` | Maths, Hungarian assignment | |
| `lap` | ByteTrack dependency | Pre-installed so Ultralytics doesn't auto-install it mid-demo |
| `pyyaml` | `config.yaml` | |
| CLIP (auto) | YOLO-World text encoder | Auto-installed by `scripts/prefetch.py` on first use; **needs `git`** |

Model weights (auto-downloaded by `prefetch.py`, gitignored): `yolov8n.pt` (~6 MB), `yolov8s-world.pt` (~25 MB).
Fine-tuned weights go in `models/` and **are** committed.

> ⚠️ Only load weights from official Ultralytics downloads or ones we trained ourselves. `.pt` files are pickles and can run code when loaded, so don't use unknown third-party models.

---

## 2. Windows RTX 3050 laptop (do this FIRST, it's a ~3 GB download)

### Install manually beforehand
1. **NVIDIA driver**: update to the latest Game Ready or Studio driver (needed for the CUDA 12.6 build).
2. **Git**: https://git-scm.com/download/win
3. **Iriun Webcam** Windows client: https://iriun.com (this makes the phone show up as a webcam)
4. *(Fallback)* **DroidCam** Windows client: https://www.dev47apps.com

### Run the setup
```powershell
git clone <your-repo-url> turretx
cd turretx
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```
It installs uv and Python 3.12, creates `.venv`, installs CUDA PyTorch and the requirements, checks the GPU, lists camera indices, and pre-downloads the models.

### Expected output from `check_env.py`
```
torch       2.x.x+cu126
  GPU       NVIDIA GeForce RTX 3050 ... (CUDA 12.6)
  matmul on cuda: OK
opencv      4.x
  aruco     OK
cameras:
  index 0: 1280x720     <- built-in webcam
  index 1: 1280x720     <- phone via Iriun (use this one)
```
**If it says "no GPU detected":** the driver is too old, or the CPU-only torch got installed. Re-run
`uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126` inside the venv.

### Laptop power settings (RTX 3050 has 4 GB VRAM and throttles on battery)
- Always **on charger** while running or training.
- Windows: Settings → System → Power → **Best performance**.
- Don't fine-tune YOLO while the live pipeline is running (VRAM).
- Before judging, run the full pipeline for 10 minutes and watch that FPS stays steady (thermal soak).

### Daily use
```powershell
.\.venv\Scripts\Activate.ps1
git pull
python -m turretx.main            # (once the app exists)
```

---

## 3. Mac M1 (dev machine)
```bash
cd ~/sedhacks
bash setup_mac.sh
source .venv/bin/activate
```
The Mac runs the full pipeline on recorded clips at roughly 15–25 FPS on MPS. That's fine for development; the live demo runs on the 3050.

---

## 4. Android phone (camera)
1. Install **Iriun Webcam** from the Play Store. *(Fallback: DroidCam. Its USB mode needs Developer options → **USB debugging** on.)*
2. Connect to the 3050 by **USB** (lower latency, and avoids the crowded venue WiFi). If you use WiFi, use your **own hotspot**.
3. In the camera app: **lock focus + exposure**, set **720p**, and use 60 FPS if available.
4. Phone settings: keep it plugged in, set auto-lock to never, turn on Do Not Disturb.
5. Mount at **about 45–60° looking down** at the table, with **all 4 corner markers in frame**.

The camera source is a setting in `config.yaml`: a webcam index (e.g. `1`), a video file (`recordings/clip1.mp4`), or a URL (`http://<phone-ip>:8080/video` for the IP Webcam app).

---

## 5. Physical scene checklist (teammates)
- [ ] Plain, single-colour, matte surface (table or floor), about 100 × 60 cm. Measure it exactly and put the numbers in `config.yaml`.
- [ ] 4 **ArUco markers** (DICT_4X4_50, IDs 0–3), printed about 8 cm wide, taped at the corners: 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left. Generator: https://chev.me/arucogen/
- [ ] Even, diffuse lighting, no harsh moving shadows (avoid windows behind)
- [ ] 3–4 toy/RC cars that look different from each other and contrast with the surface
- [ ] **No hands in frame:** use RC cars, or move toy cars with thin sticks or thread. Hands get detected as vehicles.
- [ ] **Roof sticker on each friendly car** (~3×3 cm, matte), in a colour that appears **nowhere else** in the scene (bright pink/magenta or neon green usually works). Friendly-1 and Friendly-2 get different colours. This sticker is the simulator's stand-in for the vehicle's GPS.
- [ ] Phone mount: tripod with an arm, a selfie stick taped to a chair, or a broomstick across two chairs. **Angle ≥60°** (steeper is more accurate). Once calibrated, **don't touch it**.
- [ ] A marked start spot + direction on the table for the ESP32 car (stretch goal only)

### Laser turret (electronics team), see TURRETX_DESIGN.md §5.9
- [ ] ESP32 + 2 servos (pan/tilt bracket) + **Class 2 (<1 mW) laser pointer** switched through a transistor (not straight off a GPIO pin)
- [ ] Servos on their **own 5 V supply** with common ground (servos browning out the ESP32 is the classic failure)
- [ ] Firmware: serial 115200 baud, commands `P <pan*10> <tilt*10>`, `L 0|1|2`, `H` (reply `OK` / `ERR`). **Clamp servo angles to the table area. Laser OFF at power-up. Laser OFF if no `H` heartbeat for 500 ms**
- [ ] Mount it rigidly above or beside the table so it can reach the whole designation zone. Once calibrated, don't move it
- [ ] Note the COM port in Device Manager → Ports and put it in `config.yaml` (`turret.port`)

---

## 6. Optional: fine-tuning the detector (Roboflow, done by teammates)
1. Harish records 2–3 minutes of the real scene, moving cars included, and extracts about 250 frames.
2. Teammates upload them to **Roboflow** (free, browser-based), with one class: `vehicle`. Draw boxes (auto-label pre-fills most of them), then export as **YOLOv8**.
3. On the 3050:
   ```powershell
   yolo train model=yolov8n.pt data=<export>/data.yaml epochs=50 imgsz=640 batch=8
   ```
   This takes about 15–25 minutes. If you run out of memory, use `batch=4`. Copy `runs/detect/train/weights/best.pt` to `models/vehicle.pt` and commit it.

---

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Update the NVIDIA driver, then reinstall the cu126 torch (see above) |
| A camera index won't open | Close Iriun/DroidCam preview windows; try the next index; replug USB |
| Laggy or frozen frames | Use USB instead of WiFi; 720p, not 1080p |
| YOLO-World fails on `set_classes` | Git is missing or there's no internet. Re-run `python scripts/prefetch.py` |
| `CUDA out of memory` while training | `batch=4` |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| FPS drops after a few minutes | On battery or thermal throttling: plug in, Best performance mode, raise the laptop for airflow |
| Roof sticker not found / found everywhere | Re-tune the HSV range in `config.yaml` under the venue lighting; pick a colour nothing else has |
