# TurretX: Design & Project Context

> **For a new Claude Code session (or teammate) opening this repo:** this file holds all the context and decisions from the brainstorming session on 2026-10-12. Read it fully before writing code. Machine setup is in [DEPENDENCIES.md](DEPENDENCIES.md). Harish is the **only** software developer, so keep everything simple, single-language (Python) and single-process.

---

## 1. Understanding summary

- **Event:** SEDHACKS'26, a 24-hour hackathon. Track: *Space Applications & Defense Technology*. Category: **Hardware**. Team **WhatTheHack** (lead: Adithya Varman G), 4 people.
- **What:** TurretX is an **air-to-ground friend-or-foe identification** system. An overhead camera (standing in for a drone or attack helicopter sensor) tracks ground vehicles and fuses them with **authenticated RF identity messages**. Each vehicle is labelled **FRIENDLY / UNKNOWN / SUSPICIOUS** with an explainable confidence score.
- **Why:** iDEX / DRDO **DISC 3, "Friend or Foe Identification System"**: in mechanised warfare there's often no clear forward line of troops (FLOT), so aircraft giving close support struggle to identify their own ground forces. IFF says *a friendly is nearby*; a camera says *there are 5 vehicles*. **Nothing links an RF identity to a specific vehicle in the image.** TurretX closes that gap.
- **Core principle:** show uncertainty and **never assume an unidentified vehicle is hostile**. Human in the loop: the system assists the operator and **takes no action on its own**.
- **Jurors:** **Aerion Aerospace** (aerionaerospace.in), a UAV maker for agriculture, industry and defence (*Vanguard C50* defence UAV). Their site stresses AI/ML, edge computing, computer vision and SLAM. Pitch TurretX as **a payload/software module for a defence UAV**. Expect questions on edge compute, latency, moving platforms and failure modes.
- **Non-goals:** taking action on its own, real military crypto, training models from scratch, actual flight, night/IR, sharing data across multiple drones.

### Pitch deck notes (the original PDF is gitignored)
The deck pitched a *ground-vehicle turret* framing. **It is now re-framed as air-to-ground.** The deck's prototype bill of materials: camera ₹2k, LoRa/SDR ₹3k, Raspberry Pi ₹4k, GNSS/IMU ₹1.5k (= ₹10.5k). The evidence it cites: iDEX DISC 1/3/5, DRDO IFF, PIB fratricide data (1/3/7/1 incidents in 2012–2015).
**Fix these typos in the deck:** "Kaiman" → Kalman, "Suspicous" → Suspicious, "battleships" → battlefields, "Cable of" → Capable of, "full-the-shelf" → off-the-shelf, "cures" → cues.

---

## 2. Constraints & assumptions

| Item | Value |
|---|---|
| Developers | **1** (Harish). Teammates: electronics, deck, research, scene, car-pushing, labelling |
| Run machine | Windows laptop, **RTX 3050** (main/demo) |
| Code machine | Mac M1, developing against recorded clips (MPS) |
| Camera | **Android phone** via Iriun (or DroidCam) over USB, angled **about 45–60°** at a table |
| Scene | Toy/RC cars on a plain table, about 100×60 cm, **4 ArUco corner markers** |
| Available hardware | ESP32-S3, Raspberry Pi 5, MPU6050, one AS5600 encoder, ultrasonics, laser pointers (extras TBC). **No GPS, LoRa or radar.** Same kit as `~/harzard-poc` |
| Performance target | ≥10 FPS end to end (3050 should manage 30+), labels update within 1 s |
| Scale | 3–5 vehicles in view, 1–2 friendly RF IDs |
| Security model | RF "authentication" = **HMAC-SHA256 with a shared key**, plus a nonce/challenge. Honest and simple; not military crypto |
| Demo slot | Unknown; assume **5 min + Q&A** |

---

## 3. Scope (locked)

**Base product**
1. Live vehicle detection + tracking from the phone camera
2. Homography → vehicle positions in **table centimetres**, plus speed/heading
3. RF simulator: signed friendly-identity messages (ground truth + noise, see §5.2)
4. RF receiver: signature + freshness verification
5. **Matching engine**: position + motion + RSSI scores → confidence → assignment → label state machine
6. Operator display (OpenCV): coloured boxes, labels, mini-map, alerts, event log

**Add-ons (cheap, chosen for a solo developer)**
- **#3 Explainable confidence:** show `P .92 · M .81 · R .70 → 0.86` per vehicle
- **#6 Comms-loss coasting:** when RF is lost, the identity holds with decaying confidence, then drops to UNKNOWN
- **#2 Red-team hotkeys:** forged, replayed, duplicate-ID (cloned), and ghost-ID attacks
- **#1 Challenge-response:** the drone issues a signed challenge; friendlies only transmit in reply (emission control, EMCON)

**Stretch (only if ahead at about hour 18)**
- **#5 Real ESP32-S3 beacon:** real WiFi/ESP-NOW signed packets, with position from **encoder + MPU6050 dead reckoning** (reuse `~/harzard-poc` firmware). This replaces the simulated position with an independent real one.

**Cut** (mention as roadmap): moving-camera homography, laser pan-tilt "marker" (only if electronics can do it entirely themselves; must be operator-triggered), web dashboard, Jetson deployment, multi-drone sharing, night/IR.

---

## 4. Architecture

Single Python process. Modules exchange **plain dicts** (contracts in §6), so each one can be tested offline on recorded data.

```
 CAMERA / VIDEO FILE
        │
 [tracker.py]  detect (yolo | yoloworld | diff) → track (ByteTrack / simple tracker)
        │       → ground point = bbox BOTTOM-CENTRE → homography → cm
        │       → velocity by linear regression over the last ~1 s
        ▼
   tracks[]  ───────────────────────────────┐
                                             ▼
 [rfsim.py] friendly beacons + attacker ─► [rfrx.py] verify HMAC,   ─► claims[]
   (challenge-response, noise, RSSI)         nonce, freshness             │
                                                                          ▼
                                                              [matcher.py]
                                         pairwise scores → EMA → Hungarian → states
                                                                          │
                                                                          ▼
                                                     [display.py] overlay + mini-map + alerts
                                                     [main.py] loop, hotkeys, config, JSONL log
```

### Proposed layout
```
turretx/
  __init__.py
  config.py      # loads config.yaml
  tracker.py
  homography.py  # ArUco corner detection, manual 4-click fallback, pixel→cm
  rfsim.py
  rfrx.py
  matcher.py
  display.py
  main.py        # python -m turretx.main [--source N|file|url]
config.yaml
models/          # fine-tuned weights (committed)
recordings/      # short test clips (keep each well under 50 MB; GitHub's hard limit is 100 MB)
logs/            # JSONL event logs (gitignored)
tests/           # matcher unit tests against synthetic tracks
```

---

## 5. Component details

### 5.1 Tracker
- **Detector backends** (set in config; compare all three by the **hour 2 checkpoint**):
  - `yolo`: official `yolov8n.pt` (or `models/vehicle.pt` once fine-tuned). Low confidence threshold (**0.15–0.25**), **class-agnostic** (any box inside the table area counts as a vehicle), `imgsz` 640–960.
  - `yoloworld`: `yolov8s-world.pt` with `set_classes(["toy car"])`.
  - `diff`: **difference from a reference frame of the empty table** (hotkey `b` captures it). absdiff → threshold → morphology → contours with an area filter. Very reliable in a controlled scene, including stopped cars. **This is the guaranteed fallback.** It needs locked phone exposure.
- **Tracking:** ByteTrack via `model.track(persist=True, tracker=...)` with `track_buffer` raised to about 60. For the `diff` backend, use a small centroid+IoU tracker with Hungarian assignment (scipy).
- **Ground point:** use the **bottom-centre of the box**, not its centre. The homography maps the *ground plane*, and with an angled camera the box centre is above the ground.
- **Kinematics:** keep about 1 s of (t, x_cm, y_cm) per track; velocity = least-squares slope; speed and heading derived from it. Below 2 cm/s, treat the vehicle as stationary.
- **Motion blur** is the main cause of unreliable detection while moving: lock exposure, light the table well, use 60 FPS if available.

### 5.2 RF simulator (`rfsim.py`)
- **Binding:** press `1` (or `2`), then click a vehicle to bind **Friendly-1** to that track.
- **Claims are ground truth plus noise** (the standard way to simulate GNSS): track position + N(0, 3 cm), velocity + noise, ~100 ms latency, 5 Hz. If the bound track is lost, the beacon keeps dead-reckoning at constant velocity, as a real vehicle would.
- **RSSI:** log-distance path loss from a virtual "drone" position (centre of the table, raised): `rssi = P0 − 10·n·log10(d) + N(0, 3 dB)`, with P0 = −40 dBm and n = 2.2.
- **Be upfront about this in the pitch:** "RF is simulated from ground truth plus realistic noise; stretch goal #5 replaces it with a real ESP32 beacon using its own dead reckoning."
- **Challenge-response (#1):** every 1 s the drone broadcasts `{challenge_nonce, t}`. Beacons reply with `HMAC(key, nonce‖rf_id‖x‖y‖vx‖vy‖t)`. A reply without the *current* nonce is invalid. This also prevents replays.
- **Attacker (red-team hotkeys):**
  - `f` **forged**: a message with a bad signature → rejected
  - `r` **replay**: re-sends an old valid message → rejected (stale nonce)
  - `d` **duplicate / cloned ID**: an attacker holding a captured key broadcasts Friendly-1's ID **from another vehicle's position** → the same ID appears in two places → **SUSPICIOUS**
  - `g` **ghost ID**: a valid claim at a position where no vehicle exists → a SUSPICIOUS claim marker on the map
  - `j` **jam**: toggles RF loss → demonstrates coasting

### 5.3 RF receiver (`rfrx.py`)
Verify the HMAC → check the nonce matches the current challenge and is unused → check `|now − t| < 1.5 s`. Each rejection is logged with its reason (`bad_signature`, `stale_nonce`, `replayed`, `expired`) and shown in the event log. Output: verified `claims[]`.

### 5.4 Matching engine (`matcher.py`), the core logic
For every (claim *c*, track *k*) pair:

| Score | Formula (starting values, tune with real data) |
|---|---|
| **Position P** | `exp(−d² / 2σp²)`, σp = 8 cm; **gate**: d > 25 cm → pair not allowed |
| **Motion M** | speed term `exp(−Δs² / 2σv²)` (σv = 5 cm/s) × heading term `(1 + cos Δθ) / 2`. If both vehicles are stationary → M = 0.5 with a "low info" flag |
| **RSSI R** | `exp(−(rssi − rssi_expected(track))² / 2σr²)`, σr = 6 dB |
| **Combined C** | `0.45·P + 0.40·M + 0.15·R`, then an EMA over time per pair (α = 0.3) |

- **Assignment:** Hungarian (`scipy.optimize.linear_sum_assignment`) on cost `1 − C`, over allowed pairs only.
- **Ambiguity:** if the second-best track's C is within **0.10** of the best → the vehicle is **UNKNOWN (ambiguous: too close together)**. Don't guess.
- **Label state machine per track (with hysteresis):**
  - **FRIENDLY:** assigned claim with C ≥ **0.70** held for ≥ 1.0 s
  - **SUSPICIOUS:** (a) assigned claim with M < 0.3 held for ≥ 1.0 s (motion contradicts the claim), or (b) the same `rf_id` is claimed at two places more than the gate apart (duplicate ID; flag every track involved), or (c) a verified claim with no track within the gate (ghost; shown on the map as a red hollow marker)
  - **UNKNOWN:** everything else, including no RF, low confidence, or ambiguous
  - Leaving a state requires 1.5 s of contrary evidence, so labels don't flicker.
- **Coasting (#6):** a FRIENDLY track whose claims stop keeps its label marked `COASTING`, with C decaying by 0.1/s. Below 0.5 it becomes UNKNOWN. **It never becomes SUSPICIOUS from silence alone.**

### 5.5 Display (`display.py`)
- **Video panel:** boxes coloured green for FRIENDLY, amber for UNKNOWN, red for SUSPICIOUS. Label text: `FRIENDLY F-1 0.86  [P.92 M.81 R.70]`.
- **Mini-map:** the table from above in cm. Filled dots are tracks; hollow rings are RF claims; a line joins matched pairs.
- **Banner alerts** for SUSPICIOUS events, with a beep (`winsound` on Windows, terminal bell elsewhere).
- **Event ticker:** the last 6 events. FPS + latency shown in the corner.
- **Every** label change and RF rejection is written to `logs/<timestamp>.jsonl`.

### 5.6 Hotkeys
`h` calibrate homography (ArUco; falls back to clicking the 4 corners) · `b` capture empty-table background · `1`/`2` + click: bind friendly · `f` forged · `r` replay · `d` duplicate ID · `g` ghost · `j` jam toggle · `p` pause · `q` quit

---

## 6. Data contracts

```python
# tracker → matcher (one per vehicle, every frame)
Track = {"track_id": int, "t": float, "x": float, "y": float,      # cm, table frame
         "vx": float, "vy": float, "speed": float, "heading": float, # cm/s, radians
         "bbox": [x1, y1, x2, y2], "det_conf": float}

# rfsim → rfrx (wire format, JSON)
RFMessage = {"rf_id": "F-1", "t": float, "nonce": str,
             "x": float, "y": float, "vx": float, "vy": float,
             "rssi": float, "sig": str}          # sig = hex HMAC-SHA256

# rfrx → matcher
Claim = {"rf_id": str, "t": float, "x": float, "y": float,
         "vx": float, "vy": float, "rssi": float}

# matcher → display
Label = {"track_id": int, "state": "FRIENDLY|UNKNOWN|SUSPICIOUS",
         "rf_id": str | None, "confidence": float,
         "scores": {"P": float, "M": float, "R": float},
         "flags": ["coasting" | "ambiguous" | "duplicate_id" | "low_info"]}
```

### `config.yaml` (initial)
```yaml
camera:
  source: 1                 # webcam index, video path, or URL
  width: 1280
  height: 720
table_cm: [100, 60]         # measure the real table!
aruco: {dict: DICT_4X4_50, ids: [0, 1, 2, 3]}   # TL, TR, BR, BL
detector:
  backend: yolo             # yolo | yoloworld | diff
  weights: yolov8n.pt
  conf: 0.2
  imgsz: 640
  classes_text: ["toy car"] # yoloworld only
tracker: {track_buffer: 60}
rf:
  key: "turretx-demo-key"   # shared HMAC key (demo only)
  rate_hz: 5
  pos_noise_cm: 3
  rssi: {p0: -40, n: 2.2, noise_db: 3}
matcher:
  sigma_pos_cm: 8
  gate_cm: 25
  sigma_speed: 5
  sigma_rssi_db: 6
  weights: {P: 0.45, M: 0.40, R: 0.15}
  ema_alpha: 0.3
  friendly_threshold: 0.70
  ambiguity_margin: 0.10
  enter_s: 1.0
  exit_s: 1.5
  coast_decay_per_s: 0.1
```

---

## 7. Build plan (solo, 24 h)

| Hours | Harish (software) | Teammates (parallel) |
|---|---|---|
| 0–1 | Run setup on the 3050, camera feed via Iriun | Build the scene (table, markers, mount, lighting, cars). **Record test clips.** |
| 1–3 | Detector + tracker + homography. **Hour 2 checkpoint: pick the best backend** | Fix deck typos; re-pitch the deck as air-to-ground |
| 1–4 | — | **Label about 250 frames in Roboflow** (DEPENDENCIES.md §6) |
| 3–4 | Kinematics, Track contract | Write the 5-minute demo script |
| 4–6 | rfsim + rfrx (HMAC, challenge-response) | Prepare Q&A answers (§9) |
| ~5 | Fine-tune YOLO (~20 min of GPU time, runs in the background) | |
| 6–10 | **matcher.py**, with unit tests on synthetic tracks | Rehearse the car-pushing scenarios (§8) |
| 10–12 | display.py, with explainable confidence | One-slide architecture diagram |
| 12–13.5 | Red-team hotkeys, coasting | Electronics: get the ESP32 beacon hardware ready (stretch) |
| 13.5–15 | Run the scenarios and tune thresholds | Push the cars |
| 15–18 | **Sleep** / buffer | Practice the pitch |
| 18–20 | Stretch #5 *or* fixes | |
| 20–22 | **Freeze the code.** Record a backup demo video | |
| 22–24 | Rehearse twice, with Q&A | |

**How to build:** one module at a time, each tested before the next. The matcher gets unit tests (`tests/`) using synthetic tracks, so it can be developed on the Mac without a camera.

---

## 8. Demo scenarios (5 min)
1. **Baseline:** 3 cars, Friendly-1 bound → it turns green; the other two stay amber as UNKNOWN ("we don't assume hostile").
2. **Crossing:** Friendly-1 crosses an unknown car → briefly *ambiguous* → re-resolves correctly afterwards.
3. **Forged / replay** (`f`, `r`) → rejected in the event log; labels unchanged.
4. **Cloned ID** (`d`) → both vehicles flagged red SUSPICIOUS, with an alert.
5. **Ghost** (`g`) → a red hollow marker where no vehicle exists.
6. **Jamming** (`j`) → Friendly-1 shows COASTING, then fades to UNKNOWN; it is never labelled hostile.
7. *(Stretch)* the real ESP32 beacon car drives on its own odometry.

---

## 9. Likely judge questions → answers
- **"Isn't this just ADS-B/radar or AIS track correlation?"** The idea is related, but those assume cooperative, well-separated aircraft or ships. Ground vehicles bunch together, cross paths and hide under cover, so we add motion consistency, explicit ambiguity, spoof detection, and a low-cost edge build.
- **"Doesn't broadcasting position give friendlies away?"** No. Challenge-response means they only transmit when an authenticated drone asks, the same principle as IFF Mode 5.
- **"What about jamming?"** The vehicle coasts with decaying confidence, then goes to UNKNOWN. **Never hostile by default.**
- **"Does it work on a moving drone?"** In production, the camera pose comes from the drone's GNSS/IMU/gimbal (or SLAM) to project the image onto the ground. In the demo, the corner markers stand in for that. Moving-camera homography is on the roadmap.
- **"Why not just use IFF?"** IFF says *someone friendly is nearby*. Close air support needs *which vehicle in the sight picture*.
- **"Edge compute?"** YOLOv8n runs in real time on a laptop GPU, and a Jetson Orin-class module is the target. Quote the measured FPS and latency from the display.
- **"Who decides to engage?"** The human operator. TurretX only labels and alerts.

---

## 10. Risks
| Risk | Mitigation |
|---|---|
| YOLO misses toy cars, especially while moving | Low conf threshold + ByteTrack buffer; YOLO-World; **`diff` fallback**; Roboflow fine-tune |
| Motion blur | Light, locked exposure, 60 FPS |
| Lighting changes break `diff` | Lock exposure; re-capture background with `b` |
| Threshold tuning takes long | Synthetic unit tests first; tune from logged JSONL |
| CUDA/driver setup on the 3050 | Run setup early; `check_env.py`; CPU fallback still works at low FPS |
| Venue WiFi | USB phone connection; models pre-fetched |
| Live demo fails | Recorded backup video; video-file source mode |
| Solo developer burnout | Scope cut already; sleep 15–18 h |

---

## 11. Decision log
| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | Real camera tracking + simulated RF | Full hardware; fully simulated | Visible part real, unverifiable part simulated; fits 24 h |
| 2 | Toy cars on a table, overhead/angled camera | People with phones; recorded drone footage | Controlled; easy homography |
| 3 | **Air-to-ground framing** | Keep ground turret; both | Matches DISC 3 wording (air support, no FLOT) and the UAV jurors |
| 4 | Off-the-shelf detection, no training from scratch | Custom model | No CV specialist on the team |
| 5 | No ArUco markers *on the cars* (only on table corners) | Markers on cars | Would undermine the premise; enemies don't carry your markers |
| 6 | Python, single process | Go backend + Python vision | Solo developer; less glue code |
| 7 | RF sim = ground truth + noise | Scripted paths | Standard GNSS simulation; robust. Disclosed openly; upgrade path is #5 |
| 8 | Add-ons #3, #6, #2, #1; #5 stretch; cut #4, laser, web UI | Larger set | ~3.5 h budget for one developer |
| 9 | ESP32 WiFi/ESP-NOW beacon instead of LoRa | LoRa | No LoRa on hand; ESP32-S3 + Pi 5 + odometry firmware already exist |
| 10 | Phone camera (Iriun, USB), angled 45–60° | Laptop webcam overhead | Easier mounting; homography handles the angle |
| 11 | Drop the friend's unknown pretrained model | Keep using it | Unknown provenance can't be debugged; `.pt` pickle can run code |
| 12 | Official YOLOv8n → compare YOLO-World / diff → optional Roboflow fine-tune | Use the unknown model | Known, debuggable, fast to fine-tune |
| 13 | Code on Mac, run on Windows RTX 3050, sync via git | Single machine | User preference; Mac has Claude Code; 3050 is the demo machine |
| 14 | HMAC + challenge nonce for "authentication" | Real crypto/IFF | Honest, simple, demonstrable |
