# TurretX: Design & Project Context

> **For a new Claude Code session (or teammate) opening this repo:** this file holds all the context and decisions from the design sessions on 2026-10-12 (brainstorming, then a multi-agent design review). Read it fully before writing code. Machine setup is in [DEPENDENCIES.md](DEPENDENCIES.md). Harish is the **only** software developer, so keep everything simple, single-language (Python) and single-process. Review findings are referenced as **R1–R24** (see §12).

---

## 1. Understanding summary

- **Event:** SEDHACKS'26, a 24-hour hackathon. Track: *Space Applications & Defense Technology*. Category: **Hardware**. Team **WhatTheHack** (lead: Adithya Varman G), 4 people.
- **What:** TurretX is an **air-to-ground friend-or-foe identification** system. An overhead camera (standing in for a drone or attack helicopter sensor) tracks ground vehicles and fuses them with **authenticated RF identity messages**. Each vehicle is labelled **FRIENDLY / UNKNOWN / SUSPICIOUS** with an explainable confidence score.
- **Why:** iDEX / DRDO **DISC 3, "Friend or Foe Identification System"**: in mechanised warfare there's often no clear forward line of troops (FLOT), so aircraft giving close support struggle to identify their own ground forces. IFF says *a friendly is nearby*; a camera says *there are 5 vehicles*. **Nothing links an RF identity to a specific vehicle in the image.** TurretX closes that gap.
- **Core principle:** show uncertainty and **never assume an unidentified vehicle is hostile**. SUSPICIOUS means *conflicting information: verify*, never *hostile* (R6). Human in the loop: the system assists the operator and **takes no action on its own**.
- **Jurors:** **Aerion Aerospace** (aerionaerospace.in), a UAV maker for agriculture, industry and defence (*Vanguard C50* defence UAV). Their site stresses AI/ML, edge computing, computer vision and SLAM. Pitch TurretX as **a payload/software module for a defence UAV**. Expect questions on edge compute, latency, moving platforms and failure modes.
- **Non-goals:** taking action on its own, real military crypto, training models from scratch, actual flight, night/IR, sharing data across multiple drones.

### Pitch deck notes (the original PDF is gitignored)
The deck pitched a *ground-vehicle turret* framing. **It is now re-framed as air-to-ground.** The deck's prototype bill of materials: camera ₹2k, LoRa/SDR ₹3k, Raspberry Pi ₹4k, GNSS/IMU ₹1.5k (= ₹10.5k). The evidence it cites: iDEX DISC 1/3/5, DRDO IFF, PIB fratricide data (1/3/7/1 incidents in 2012–2015).
**Fix these typos in the deck:** "Kaiman" → Kalman, "Suspicous" → Suspicious, "battleships" → battlefields, "Cable of" → Capable of, "full-the-shelf" → off-the-shelf, "cures" → cues.

---

## 2. Constraints & assumptions

| Item | Value |
|---|---|
| Developers | **1** (Harish). Teammates: electronics, deck, research, scene, car driving, labelling |
| Run machine | Windows laptop, **RTX 3050** (4 GB VRAM) — main/demo |
| Code machine | Mac M1, developing against recorded clips (MPS) |
| Camera | **Android phone** via Iriun (or DroidCam) over USB at **1080p**, **rigidly mounted ~2 m above the arena, as close to straight down as possible** (steeper = better, R11). **Never handheld:** the calibration assumes a fixed camera |
| Scene | **1 RC car = Friendly-1** (roof sticker, R1) + **2+ toy cars = unknowns**, moved with sticks or thread (**no hands in frame**, R7), on a **floor arena**: a plain matte sheet about 200×150 cm with **4 ArUco corner markers (15 cm)**. The code calls the arena "table"; any flat area works |
| Laser turret | ESP32 + 2 servos (pan/tilt) + low-power laser pointer, mounted high near the camera, pointing down at the arena, connected to the 3050 by USB serial. Built by the electronics team (§5.9) |
| Scenario scale | **1 cm on the arena = 10 m** → a 200×150 cm arena represents 2 km × 1.5 km; virtual drone at 300 m altitude (R9) |
| Available hardware | ESP32-S3, Raspberry Pi 5, MPU6050, one AS5600 encoder, ultrasonics, laser pointers (extras TBC). **No GPS, LoRa or radar.** Same kit as `~/harzard-poc` |
| Performance target | ≥10 FPS end to end (3050 should manage 30+), labels update within 1 s |
| Scale | 3–5 vehicles in view, 1–2 friendly RF IDs |
| Security model | RF "authentication" = **HMAC-SHA256 with a shared key**, plus a per-challenge nonce. Honest and simple; not military crypto |
| Demo slot | Unknown; assume **5 min + Q&A** |

---

## 3. Scope (locked)

**Base product**
1. Live vehicle detection + tracking from the phone camera
2. Homography → vehicle positions in **table centimetres** (shown in scenario metres), plus speed/heading
3. **Tag locator (simulator-only "GPS")**: a colour detector finds each friendly's roof sticker. Its output feeds **only the RF simulator**, never the identification pipeline (R1)
4. RF simulator: challenge-response, signed friendly-identity replies built from tag position + noise
5. RF receiver: signature + nonce verification, clone detection
6. **Matching engine**: time-aligned position + motion + RSSI scores → confidence → assignment → label state machine
7. Operator display (OpenCV): coloured boxes + icons, mini-map, side panel, alerts, event log

**Add-ons (cheap, chosen for a solo developer)**
- **#3 Explainable confidence:** side panel shows `P .92 · M .81 · R .70 → 0.86` for the selected vehicle
- **#6 Comms-loss coasting:** when RF is lost, the identity holds with decaying confidence, then drops to UNKNOWN
- **#2 Red-team hotkeys:** forged, replayed, cloned-ID and ghost-ID attacks, with an on-screen banner
- **#1 Challenge-response:** the drone issues a signed challenge; friendlies only transmit in reply (EMCON). This is also what makes clone detection exact (R4/R5)

**Stretch (only if ahead at about hour 18)**
- **#5 Real ESP32-S3 beacon:** real WiFi/ESP-NOW signed packets, with position from **encoder + MPU6050 dead reckoning** (reuse `~/harzard-poc` firmware), starting from a marked pose (R15).

**Hardware add-on: laser "verification marker" turret** (§5.9). The electronics team builds the turret and firmware; Harish spends ~1.5 h on the serial link, calibration and the cue/mark logic. **The model cues, the interlock protects friendlies, the human marks.**

**Cut** (mention as roadmap): moving-camera homography, web dashboard, Jetson deployment, multi-drone sharing, night/IR.

---

## 4. Architecture

Single Python process. Modules exchange **plain dicts** (contracts in §6), so each one can be tested offline on recorded data.

```
 CAMERA (grabber thread, latest frame only) / VIDEO FILE
        │                         clock: video PTS (file) | monotonic (live)   (R3)
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 [tracker.py]  detect (yolo | yoloworld | diff)   [taglocator.py]  HSV blob per sticker colour
        │       → track → bbox bottom-centre            │   → homography → cm
        │       → homography → cm → velocity             │   ("the friendly vehicle's own GPS")
        ▼                                                ▼   SIMULATOR ONLY — never reaches matcher
   tracks[]                                 [rfsim.py] challenge @5 Hz → signed replies
        │                                     (+ red-team injections, banner events)
        │                                                │  raw payload bytes
        │                                                ▼
        │                                   [rfrx.py] verify HMAC over raw bytes, nonce,
        │                                     one reply per (nonce, rf_id) → clone detect
        │                                                │
        │                                                ▼  claims[], conflicts[]
        └───────────────────────► [matcher.py] ◄────────┘
                       propagate claims to track time → scores → EMA
                       → gated Hungarian → label state machine
                                     │
                                     ▼
                [display.py] video + icons, mini-map, side panel, banners
                [main.py] loop, hotkeys, config, JSONL log, camera-moved check
                [turret.py] zone check → friendly interlock → cue/mark → USB serial → ESP32 laser
```

### Proposed layout
```
turretx/
  __init__.py
  config.py      # loads config.yaml
  camera.py      # source opening (CAP_DSHOW on Windows, R20), grabber thread (R13), frame timestamps = THE clock (R3)
  synthetic.py   # fake overhead scene with exact ground truth, for tests
  homography.py  # ArUco calibration, manual 4-click fallback, pixel→cm, drift check (R12)
  tracker.py
  taglocator.py  # simulator-only sticker detector (R1)
  rfsim.py
  rfrx.py
  matcher.py
  display.py
  turret.py      # designation zone, friendly interlock, cue/mark state, serial link, calibration
  main.py        # python -m turretx.main [--source N|file|url]
firmware/turret/ # ESP32 sketch: serial commands → servos + laser (electronics team)
config.yaml
models/          # fine-tuned weights (committed)
recordings/      # short test clips (keep each well under 50 MB; GitHub's hard limit is 100 MB)
logs/            # JSONL event logs (gitignored)
tests/           # matcher unit + metamorphic tests on synthetic tracks
```

---

## 5. Component details

### 5.0 Clock & camera (R3, R13, R20)
- **One clock** for everything: **each frame's timestamp `t`** from `Camera.read()`. That's the **video timestamp** (`CAP_PROP_POS_MSEC`) in file mode and `time.monotonic()` at grab time in live mode. Every Track, Claim and RF timer uses the current frame's `t`. The RF simulator and turret are **driven per frame by the main loop** (they handle whatever is due by the current `t`), never by their own wall-clock threads, so playback speed on the Mac doesn't matter. *(Implemented without a separate `clock.py`: the frame timestamp is the clock.)*
- **Live mode:** a grabber thread reads continuously and keeps only the latest frame (OpenCV otherwise buffers and latency builds up).
- **Windows:** always open cameras with `cv2.CAP_DSHOW` so indices match `check_env.py`.

### 5.1 Tracker
- **Detector backends** (set in config; compare all three by the **hour 2 checkpoint**):
  - `yolo`: official `yolov8n.pt` **or `yolo26n.pt`** (newer, NMS-free, better on small objects; compare both) or `models/vehicle.pt` once fine-tuned. Low confidence threshold (**0.15–0.25**). Accept any class **except `person`** (R7), and only boxes inside the table area. `imgsz` 640–960.
  - `yoloworld`: `yolov8s-world.pt` with `set_classes(["toy car"])`.
  - `diff`: **difference from a reference frame of the empty table** (hotkey `b` captures it). absdiff → threshold → morphology → contours. Reject blobs **touching the table border** or outside `[min_area, max_area]` (hands, sticks, R7). Very reliable in a controlled scene, including stopped cars. **This is the guaranteed fallback.** It needs locked phone exposure and diffuse light (shadows inflate blobs). Known limitation: cars touching each other merge into one blob (R16), so run the crossing scenario on a YOLO backend.
- **Tracking:** ByteTrack via `model.track(persist=True, tracker=...)` with `track_buffer` raised to about 60. For the `diff` backend, use a small centroid+IoU tracker with Hungarian assignment (scipy).
- **Ground point:** `ground_point_frac` picks where in the box the car's position is. **0.5 (centre)** for the planned near-overhead camera; **1.0 (bottom edge)** for a shallow, angled camera, where the box centre sits above the ground. Confirm with `scripts/measure_tag_offset.py`, and tune σp on real data (R11).
- **Kinematics:** keep **0.5 s** of (t, x_cm, y_cm) per track (`tracker.velocity_window_s`); velocity = least-squares slope; speed and heading derived from it. Below 2 cm/s, treat the vehicle as stationary. *(Phase 2 finding: a 1 s window reports the velocity from ~0.5 s ago and was ~4 cm/s off on curves; 0.5 s halves that while staying robust to jitter.)*
- **Re-acquisition:** a lost-then-found vehicle gets a new track ID and its confidence history restarts, so it needs about 1 s to regain FRIENDLY. This is intended, honest behaviour (R17).
- **Robustness (from code review):** a dropped live camera shows NO SIGNAL and reconnects instead of exiting; auto-calibration retries every 15 frames until the markers are seen; covered markers keep the last CAMERA MOVED verdict; live timestamps start at 0 when the camera opens.
- **Hour-2 checkpoint tool:** `python scripts/eval_tracker.py --backend <b> [--weights w] --source <clip>`. On the synthetic clip add `--truth` (uses the box centre automatically, since synthetic cars are flat) to score recall, position/velocity error and ID switches. FPS excludes model loading. On real clips it reports vehicles per frame and distinct IDs (fewer = steadier). Phase 2 synthetic baseline for `diff`: 100% recall, 0.5 cm mean position error, ~95 FPS on the M1; all ID switches came from cars within ~10 cm merging (R16).
- **Motion blur** is the main cause of unreliable detection while moving: lock exposure, light the table well, use 60 FPS if available.

### 5.2 Homography (R12)
- `h` calibrates from the 4 ArUco markers (clicking the corners is the fallback), and the result is then **frozen**. The camera must not move.
- Every ~30 frames, if ≥3 markers are visible, compare their detected positions with the calibrated ones. If the error is above ~8 px → banner **"CAMERA MOVED — press h"**.

### 5.3 Tag locator (simulator-only GPS) (R1)
- **UPDATE (Eval 2 prep): disc mode.** The only sticker the team has is the round SIST2026 event sticker, so `tags.<rf_id>.mode: disc` finds a dark disc inside a light ring **by shape** (outline projected to arena cm, circularity ≥ 0.82, fill ≥ 0.5, size 0.75–1.35× `diameter_cm`; ArUco markers, arena corners and elongated car bodies are rejected; a light ring separates it from a dark roof). Supports **one** disc tag (identical discs can't be told apart at ~2 m). The colour (`hsv`) mode below still works. The bullets below describe the original colour mode.
- Each friendly car has a **matte coloured sticker** (about 3×3 cm) on its roof, in a colour that appears nowhere else in the scene (pick it at scene setup; HSV range in config). Friendly-1 = colour 1, Friendly-2 = colour 2. **No click-binding is needed.**
- Per frame: HSV threshold → largest blob → centroid → homography → cm. Velocity by least-squares over ~1 s.
- **Firewall:** `taglocator` output goes **only** to `rfsim`. The tracker and matcher never import it. This is what makes the matching non-circular: if the tracker swaps IDs in a crossing, the RF truth still follows the *physical* car, so the swap shows up as a mismatch.
- **Offset correction:** the sticker sits on the roof (above the table plane) and the tracker uses the bottom of each box, so sticker and track disagree by a few cm (≈4 cm on the synthetic clip). Measure it once on a real clip with `scripts/measure_tag_offset.py` (same detector as the demo) and put the median into the tag's `offset_cm`. If the reported spread is large (>4 cm), the camera is too shallow; steepen it.
- If a sticker isn't visible (e.g. a stick blocks it), the beacon keeps dead-reckoning at constant velocity for up to 1 s, then stops replying (comms-loss → coasting).
- **Pitch line:** "Each friendly reports its own position from its own navigation. Here, a sticker read by a *separate* sensor stands in for the vehicle's GPS. Stretch goal #5 replaces it with real on-board odometry."
- **Perception risk:** a judge seeing a sticker may assume it identifies friendlies. **Say the pitch line before they can wonder, then immediately prove it:** press `j` (jam) and the stickered car fades to UNKNOWN *with the sticker still visible*; then `d` (cloned ID) flags a car *without* a sticker as ID CONFLICT. Only the RC car wears a sticker. Put it on **before recording dev clips** so the tag locator can be built offline.

### 5.4 RF simulator (`rfsim.py`)
- **Challenge-response (#1):** the drone issues a challenge **at 5 Hz**: `{nonce, t}` with a fresh random nonce. Each friendly sends **exactly one reply per nonce** (R4).
- **Reply content:** tag position + N(0, 3 cm), tag velocity + noise, ~100 ms latency, `t` = the time the position was measured.
- **Scenario scale (R9):** RSSI uses scenario metres (1 cm = 10 m) from a virtual drone at 300 m altitude above the table centre: `rssi = P0 − 10·n·log10(d_m) + N(0, 3 dB)`, with P0 = −40 dBm at 1 m and n = 2.2. Keep R's weight low; it's a consistency check, not a discriminator.
- **Wire format (R14):** payload = compact JSON with **integer** fields (mm, mm/s, dBm×10, ms). Signature = `HMAC-SHA256(key, payload_bytes)` sent alongside the payload, never computed over re-serialised JSON.
- **Attacker (red-team hotkeys, each also emits a banner event, R24):**
  - `f` **forged**: a reply with a bad signature → rejected
  - `r` **replay**: re-sends an old valid reply → rejected (stale nonce)
  - `d` **cloned ID** (toggle): an attacker holding a captured key answers the challenge as Friendly-1 **from another vehicle's position** → two valid replies for one (nonce, rf_id) → **ID CONFLICT**. The attacker follows its **physical** vehicle by position (nearest track to where it was, never within 12 cm of the friendly's sticker), not by track ID. *(Phase 2b bug: following a track ID made the clone jump onto the real friendly when the tracker swapped IDs in a crossing.)*
  - `g` **ghost ID** (toggle): a valid reply at the emptiest spot in the arena → ghost marker
  - `j` **jam**: toggles RF loss → demonstrates coasting

**Implementation (Phase 2b):** `wire.py` (payload encode/decode, HMAC, key from `TURRETX_RF_KEY` env var or the config demo key), `rfsim.py` (`FriendlyBeacon`, `Attacker`, `Channel` with jamming, `RFSim` orchestrator), `rfrx.py` (`Interrogator`, `Receiver`). Synthetic results: 5 replies/s per friendly, ~107 ms latency, 3.3 cm mean position error (3 cm noise per axis), RSSI spans ~11 dB across the arena.

### 5.5 RF receiver (`rfrx.py`)
0. Size limit (256 bytes) → otherwise `malformed`. **No parsing happens before the signature check.**
1. Recompute the HMAC over the **received payload bytes**; compare with `hmac.compare_digest` (R14) → otherwise `bad_signature`. Then schema/range validation of the authenticated payload → otherwise `malformed`.
2. Nonce must be one of the **last 3 issued** challenges → otherwise `stale_nonce`. (In live mode the nonce provides freshness; there's no clock sync with real hardware, R15.)
3. **Second distinct valid reply for the same (nonce, rf_id)** → raise a `conflict` for that rf_id carrying both positions (R5). An exact byte duplicate → `replayed`.
4. Output: `claims[]` (one per rf_id per nonce, or both sides of a conflict, flagged) and `rejections[]` (logged and shown in the event ticker).

### 5.6 Matching engine (`matcher.py`), the core logic
**Step 0: time alignment (R2).** Before scoring, propagate every claim to the track's timestamp: `x' = x + vx·(t_track − t_claim)`, same for y. Without this, latency alone (~0.3 s at 30 cm/s ≈ 9 cm) would push moving friendlies outside σp.

For every (claim *c*, track *k*) pair:

| Score | Formula (starting values, tune with real data) |
|---|---|
| **Position P** | `exp(−d² / 2σp²)`, σp = 8 cm; **gate**: d > 25 cm → pair not allowed |
| **Motion M** | speed term `exp(−Δs² / 2σv²)` (σv = 5 cm/s). **Heading term `(1 + cos Δθ) / 2` is multiplied in only if both speeds > 5 cm/s** (R8). If both are stationary → M = 0.5 with flag `low_info` |
| **RSSI R** | `exp(−(rssi − rssi_expected(track))² / 2σr²)`, σr = 6 dB, in scenario metres |
| **Combined C** | `0.45·P + 0.45·M + 0.10·R`, then an EMA over time per (rf_id, track_id) pair (α = 0.3) |

- **Assignment:** Hungarian (`scipy.optimize.linear_sum_assignment`) on cost `1 − C`. **Gated pairs get a large finite cost (e.g. 1e6), not `inf`** (scipy raises "cost matrix is infeasible"), and any assignment with that cost is discarded afterwards (R10).
- **Ambiguity:** if the second-best track's C is within **0.10** of the best → the vehicle is **UNKNOWN · TOO CLOSE**. Don't guess.
- **Label state machine per track (with hysteresis):**

| State | Condition | Display (R6, R21, R23) |
|---|---|---|
| **FRIENDLY** | assigned claim with C ≥ **0.70** held ≥ 1.0 s | green · ✓ · `F-1 0.86` |
| FRIENDLY · COASTING | was FRIENDLY, its rf_id went **silent** (not contradicted). Held while **track custody** is intact (§5.6a), up to 60 s | green dashed · ✓ · `F-1 · NO RF 23 s` |
| **UNKNOWN · NO ID** | no claim assigned | amber · ? |
| **UNKNOWN · TOO CLOSE** | ambiguity rule above | amber · ? · `TOO CLOSE` |
| **SUSPICIOUS · MOTION MISMATCH** | assigned claim with M < 0.3 held ≥ 1.0 s (only when the heading term is active or Δs is large) | **magenta** · ⚠ · `VERIFY` |
| **SUSPICIOUS · ID CONFLICT** | track matched to either side of an rf_id conflict | **magenta** · ⚠ · `ID CONFLICT — VERIFY`, the two vehicles linked by a line |
| (claim only) **GHOST** | verified claim with no track within the gate | magenta hollow ring on the mini-map · `GHOST ID` |

  - Leaving a state requires 1.5 s of contrary evidence, so labels don't flicker.
  - Custody broken or 60 s elapsed → UNKNOWN · NO ID. **Silence alone never produces SUSPICIOUS.**
  - **ID CONFLICT enters immediately** (no 1 s wait): it is a security alert and the turret interlock must apply at once. Leaving it still needs 1.5 s.
  - Smoothing is **time-based** (`ema_tau_s`), so results don't depend on frame rate.

### 5.6a Track custody & RF DENIED (from the jamming discussion)
- **Identity belongs to the camera track; the radio refreshes it.** When a FRIENDLY's rf_id goes silent, the track keeps FRIENDLY · COASTING for as long as the drone keeps continuous custody, up to `coast_max_s` (60 s).
- **Custody breaks, and the label drops to UNKNOWN · NO ID immediately, when:** the track is lost (a re-found vehicle gets a new track ID), or any other vehicle comes within `custody_radius_cm` (15 cm), since they could swap in a crossing.
- **RF DENIED:** if friendly claims were being received and **no** valid claims arrive for `rf_denied_s` (2 s), the matcher raises `rf_denied`. The display shows **"RF DENIED: identification degraded. Unidentified ≠ hostile."** and the turret cues nothing (no fresh claims means no MOTION MISMATCH cues; coasting friendlies stay interlocked).
- **Last-known friendlies:** when a FRIENDLY/COASTING track is lost, its last position and velocity are kept for 60 s and drawn as a fading marker with a growing uncertainty circle.
- **Pitch line:** "Under total jamming we can't identify anyone, and we say so. Identity we've already confirmed stays with each vehicle's track while the drone keeps continuous custody."
  - **Red is not used for any label.** Nothing on screen should read as "enemy".

### 5.7 Display (`display.py`)
- **Video panel:** boxes coloured per state, each with an **icon + short text** (readable on a projector, colour-blind safe). Large fonts.
- **Side panel (R22):** the selected vehicle (click it, or press Tab to cycle) shows `P · M · R → C`, state, rf_id, and time in state.
- **Mini-map:** the table from above, axes labelled in **scenario metres**. Filled dots are tracks; hollow rings are RF claims; a line joins matched pairs; a magenta line joins conflict pairs.
- **Banners:** SUSPICIOUS events, **red-team actions** ("RED TEAM: cloned ID injected", R24), "CAMERA MOVED". Beep via `winsound` on Windows, terminal bell elsewhere.
- **Event ticker:** the last 6 events. FPS + end-to-end latency shown in the corner.
- **Every** label change, conflict and RF rejection is written to `logs/<timestamp>.jsonl`.
- **Implementation:** `display.py` (video overlay with drawn icons, dashed box for COASTING, side panel with mini-map in scenario km, selected-vehicle P/M/R/confidence bars and time in state, legend, event ticker; window scaled to `display.max_width`), `events.py` (diffs matcher output each frame → ticker + JSONL; beeps once when a vehicle turns SUSPICIOUS). A persistent `VERIFY: #n …` banner lists every SUSPICIOUS vehicle.

### 5.8 Hotkeys
`h` calibrate homography · `b` capture empty-table background · Tab / click: select vehicle · `m` **mark** the cued/selected vehicle (laser blinks) · `x` cancel mark / laser off · `t` turret calibration · `f` forged · `r` replay · `d` cloned ID · `g` ghost · `j` jam toggle (IFF link **and** relay) · `c` kill/restore camera (failsafe demo) · `a` re-arm designation · `k` relay barrier · `n` send relay message · `v` forged relay msg · `e` replay relay msg · `p` pause · `q` quit

### 5.11 Camera-loss failsafe (`failsafe.py`) — added after the first jury review
Jury question: *"do you have a failsafe in case the camera fails?"* Research: `reports/Camera failsafe and airborne relay.md`.
- **Health** (per loop): LOST if no frame for 0.5 s, a black frame (mean grey < 8, lens covered), a frozen frame (~1 s of identical frames **while RF says vehicles move**), or the operator kill (`c`); DEGRADED on homography drift. The sticker locator keeps running after a `c` kill (it stands in for GPS, which needs no camera); physically covering the lens loses it too, so the display then falls to L3.
- **Rungs:** L0 full fusion · L1 degraded (labels stay, designation locked) · L2 RF-only (camera lost, claims alive: schematic map of authenticated friendlies with beacon age, banner "non-transmitting vehicles NOT visible") · L3 last-known (claims stale: amber markers, growing no-fire ring, LOST FRIENDLY) · L4 safe state (nothing known).
- **Rules:** labels only move toward UNKNOWN and designation only toward OFF; **nothing re-arms by itself**: after the camera returns the operator presses `a` (refused unless L0). `designation_allowed` is exposed for `turret.py` (not built yet). Thresholds (0.5 s, 1 s, grey 8) are engineering choices, not standards.
- Tests: `tests/test_failsafe.py`, RF-only render in `tests/test_display.py`.

### 5.12 Comms relay (`relay.py`, `commshub.py`) — "comms centre in the sky"
Jury idea: the drone also acts as a communication relay for friendlies when ground comms are jammed, so it stays useful if its camera is lost. Approach: `RELAY_INTEGRATION.md`. Evidence: `reports/Camera failsafe and airborne relay.md`, `turretx_airborne_relay_market_study.xlsx` (claim discipline: tactical authenticated relay + blue-force tracking only; **no** anti-jam performance, HAPS or SATCOM claims; ESP-NOW is a demo radio).
- **Simulation only.** No RF is transmitted. "Jamming" is a flag (`j`); a barrier (`k`) blocks the direct ground link so traffic goes via the drone. Real jammers are illegal in India (Wireless Telegraphy Act 1933; DoT advisory 2022).
- **Protocol (`relay.py`):** HMAC-signed JSON messages (types txt/pos/sys/rcpt, priority alert > status > chat, TTL); seen-set replay protection (not a monotonic counter) plus freshness; bounded queue (drops oldest lowest-priority); signed delivery receipts; jam detection by **consistency** (low delivery **and** high noise = jamming; low delivery on a quiet channel = blocked path); keyed channel hop (HMAC-derived schedule) that flushes the queue in priority order. Delivery is behind a `Link` interface (`SimLink` now; an ESP32 serial bridge later).
- **Hub (`commshub.py`):** broadcasts a priority alert when the camera is lost / restored; in L2/L3 broadcasts the blue-force picture (positions + age) from authenticated claims; shares one simulated jam between IFF and relay (`relay.hop_restores_iff`: true = IFF recovers after the hop, false = keep the jam-first demo); demo hotkeys.
- **Firewall:** comms never decide identity. `matcher.py`, `tracker.py`, `rfrx.py` etc. must not import `relay` or `commshub` (AST test in `tests/test_relay.py`).
- **Display panel:** step 3 (not built at the time of writing; see `STATUS.md`).

### 5.10 Real ESP32 beacon: MPU6050-only motion signature (stretch #5, decided 2026-10-12)
The RC car carries an ESP32-S3 + MPU6050 and **no wheel encoder**. Integrated speed/distance/position from an accelerometer drift within seconds (a parked car would soon "report" motion, the exact false-MISMATCH trap fixed in Phase 3), so the real beacon does **not** send position. It sends a **motion signature**, and the matcher asks *"is exactly one tracked vehicle moving the way this radio says it is?"*

**Firmware (electronics team):**
- Sample the MPU6050 at **100 Hz**; low-pass filter.
- **Gyro bias:** average gyro Z for 3 s at power-on while still (set `cal=1` when done); re-estimate whenever stationary.
- **Stationary detector (ZUPT):** |gyro Z| < 2 °/s and accelerometer variance small over 0.3 s → `mv=0`, else `mv=1`.
- On each challenge (5 Hz), reply with the **last 0.2 s as 4 samples at 20 Hz**:
  - `wz`: yaw rate (gyro Z, bias-removed), units 0.1 °/s, oldest first
  - `ax`: forward acceleration (accelerometer along the car's nose, gravity/bias removed), units mm/s²
- **Do not send** roll/pitch/yaw angle, integrated speed or distance: they drift and would cause false alarms. (Fine to keep for local debugging.)
- Payload: compact JSON, integers only, ≤ 256 bytes, signed with **HMAC-SHA256** over the exact bytes (mbedtls `mbedtls_md_hmac`, same shared key):
  `{"id":"F-1","n":"<nonce>","seq":123,"cal":1,"mv":1,"wz":[152,160,171,169],"ax":[-40,10,25,30]}`
- Transport: **UDP over WiFi** on the same hotspot as the 3050: the laptop broadcasts `{"n":"<nonce>"}` challenges on one port; the car replies to another. (ESP-NOW to a USB-attached ESP32 is the alternative.)
- `seq` increments per reply (loss detection). The ESP32 clock is **not** used for timing: the laptop timestamps samples from the challenge send time.

**Software (Harish):**
- `wire.py`: second payload schema (`wz`, `ax`, `mv`, `cal`, `seq`) with the same verification order.
- Camera side: per track, **yaw rate** = d(heading)/dt (only while speed > 5 cm/s) and forward acceleration = d(speed)/dt, from the same velocity history.
- Matcher: new **signature score S** over a 2.5 s window: correlation of beacon yaw rate with each track's yaw rate, plus agreement of moving/stationary (`mv`) and start/stop events. S replaces P (no position) for these claims; it is weak while everything is parked (both "stationary" = low info, like today).
- `cal=0` → motion fields ignored (no information), never SUSPICIOUS.
- **Why it's good for the pitch:** no position needed, nothing to drift, harder to spoof (the attacker must mimic the friendly's live motion), and it works on a real drone even when friendly GPS is degraded.
- Estimated cost: ~2–3 h of software once the firmware sends packets.

 (`turret.py` + ESP32)
**Principle: the model cues, the interlock protects friendlies, the human marks.** The laser never blinks (marks) without an operator keypress, so the system takes no autonomous action.

**Designation zone.** A polygon in table cm (default: the calibrated table area minus a 3 cm margin; configurable). The turret only ever points at coordinates **inside the zone**. Anything outside is refused in software, and the firmware also clamps the servos to mechanical limits so the laser physically cannot leave the table area (eye safety).

**Friendly interlock (hard rule, checked before every command).** The turret refuses to point at a vehicle that is:
- FRIENDLY (including COASTING), or
- SUSPICIOUS · ID CONFLICT (a real friendly may be one side of the conflict, R6), or
- was FRIENDLY at any time in the last **5 s**, or
- within **10 cm** of a FRIENDLY or ID CONFLICT vehicle (too close to point at safely; also blocks TOO CLOSE situations).
A refused request shows `BLOCKED: FRIENDLY INTERLOCK` and is logged. Nothing can override this, including the operator.

**Turret states**
| State | Trigger | Laser |
|---|---|---|
| `IDLE` | nothing to do | off, parked |
| `CUEING <track>` | **automatic**: a vehicle is SUSPICIOUS · **MOTION MISMATCH** inside the zone and passes the interlock (if several, the highest-confidence mismatch). Or the operator selects any vehicle that passes the interlock | slews to it and follows it, **solid on** = "look here" |
| `MARKED <track>` | operator presses `m` while cueing | **blinks** (2 Hz) on the vehicle while it moves |
| `BLOCKED` | a cue/mark target fails the interlock or leaves the zone, or its label changes | off; banner shows the reason |
- Automatic cueing happens **only** for MOTION MISMATCH, never for UNKNOWN (unidentified ≠ hostile) or ID CONFLICT.
- The interlock is re-checked **every frame** while cueing or marked. If the target becomes FRIENDLY or ID CONFLICT, or drifts within 10 cm of one, the laser switches off immediately.
- `x` cancels at any time. If the target track is lost for more than 1 s → IDLE.

**Pointing at a moving car.** Aim at the track position propagated forward by the turret's latency (~100 ms + servo travel), `x + vx·dt`. Limit commands to 20 Hz, and skip moves under 0.5° to avoid servo jitter.

**Calibration (`t`, ~2 min).** The laser is manually jogged (arrow keys) onto each of the 4 ArUco corners plus the table centre, and the servo angles are recorded. A mapping from table cm to (pan, tilt) is fitted: a 2-D homography from table cm to (tan pan, tan tilt) is accurate enough if the turret sits above the table; otherwise use least-squares quadratics over the 5 points. The result is saved to `turret_calib.json`. SG90-class servos give ±1° ≈ ±2 cm at 1 m, fine for toy cars.

**Serial protocol** (115200 baud, one line per command, ESP32 replies `OK` or `ERR <reason>`):
```
P <pan_deg*10> <tilt_deg*10>   # move; firmware clamps to its own min/max limits
L 0|1|2                        # laser off | solid | blink 2 Hz
H                              # heartbeat every 200 ms; firmware turns the laser OFF if none for 500 ms
```
The heartbeat matters: if the laptop app crashes or the cable is pulled, the laser shuts itself off.

**Display.** The designation zone outline on the mini-map; turret status line (`IDLE / CUEING #3 / MARKED #3 / BLOCKED: FRIENDLY INTERLOCK`); the cued vehicle gets a crosshair icon.

**Safety.** Use a **Class 2 (<1 mW)** laser pointer only. Firmware servo limits keep it pointed at the table. Nobody's face at table height during the demo. The laser is off by default at power-up.

---

## 6. Data contracts

```python
# tracker → matcher (one per vehicle, every frame)
Track = {"track_id": int, "t": float, "x": float, "y": float,      # cm, table frame; t = frame timestamp
         "vx": float, "vy": float, "speed": float, "heading": float, # cm/s, radians
         "bbox": [x1, y1, x2, y2], "det_conf": float}

# taglocator → rfsim ONLY
TagFix = {"rf_id": "F-1", "t": float, "x": float, "y": float, "vx": float, "vy": float, "visible": bool}

# rfsim → rfrx (wire: two fields)
RFPacket = {"payload": b'{"id":"F-1","n":"<nonce hex>","t":123456,"x":512,"y":304,"vx":-80,"vy":12,"rssi":-873}',
            "sig": "<hex HMAC-SHA256 of payload bytes>"}   # integer mm, mm/s, ms, dBm*10

# rfrx → matcher
Claim = {"rf_id": str, "t": float, "x": float, "y": float,
         "vx": float, "vy": float, "rssi": float, "conflict": bool}

# matcher → display
Label = {"track_id": int,
         "state": "FRIENDLY|UNKNOWN|SUSPICIOUS",
         "reason": "COASTING|NO_ID|TOO_CLOSE|MOTION_MISMATCH|ID_CONFLICT|None",
         "rf_id": str | None, "confidence": float,
         "scores": {"P": float, "M": float, "R": float},
         "flags": ["low_info"]}
```

### `config.yaml` (initial)
```yaml
camera:
  source: 1                 # webcam index, video path, or URL
  width: 1280
  height: 720
table_cm: [100, 60]         # measure the real table!
scale_m_per_cm: 10          # scenario scale (R9)
aruco: {dict: DICT_4X4_50, ids: [0, 1, 2, 3]}   # TL, TR, BR, BL
camera_moved_px: 8
detector:
  backend: yolo             # yolo | yoloworld | diff
  weights: yolov8n.pt
  conf: 0.2
  imgsz: 640
  exclude_classes: [person]
  classes_text: ["toy car"] # yoloworld only
  diff: {min_area: 400, max_area: 20000}
tracker: {track_buffer: 60}
tags:                       # simulator-only GPS stand-in (R1); tune HSV at the venue
  F-1: {hsv_lo: [140, 120, 120], hsv_hi: [170, 255, 255]}
turret:
  port: COM5                # check Device Manager → Ports on the 3050
  baud: 115200
  zone_margin_cm: 3
  interlock_recent_s: 5
  interlock_radius_cm: 10
  lead_ms: 150
  max_cmd_hz: 20
  min_move_deg: 0.5
  heartbeat_ms: 200
rf:
  key: "DEMO-KEY-turretx"   # demo only (R19)
  challenge_hz: 5
  latency_ms: 100
  pos_noise_cm: 3
  rssi: {p0: -40, n: 2.2, noise_db: 3, drone_alt_m: 300}
matcher:
  sigma_pos_cm: 8
  gate_cm: 25
  sigma_speed: 5
  heading_min_speed: 5
  sigma_rssi_db: 6
  weights: {P: 0.45, M: 0.45, R: 0.10}
  ema_alpha: 0.3
  friendly_threshold: 0.70
  mismatch_threshold: 0.30
  ambiguity_margin: 0.10
  enter_s: 1.0
  exit_s: 1.5
  coast_decay_per_s: 0.1
```

---

## 7. Build plan (solo, 24 h)

| Hours | Harish (software) | Teammates (parallel) |
|---|---|---|
| 0–1 | Run setup on the 3050, camera feed via Iriun | Build the scene (floor mat, 15 cm markers, rigid overhead mount, lighting, RC car + toy cars + sticks, roof sticker). **Record test clips.** Electronics: **start the laser turret** (servos, laser via transistor, firmware per §5.9 protocol, mechanical limits, heartbeat timeout) |
| 1–3 | camera + homography + detector/tracker. **Hour 2 checkpoint: pick the best backend** | Fix deck typos; re-pitch the deck as air-to-ground |
| 1–4 | — | **Label about 250 frames in Roboflow** (DEPENDENCIES.md §6) |
| 3–4 | Kinematics, Track contract, taglocator | Write the 5-minute demo script |
| 4–6 | rfsim + rfrx (challenge-response, raw-bytes HMAC, conflicts) | Prepare Q&A answers (§9) |
| ~5 | Fine-tune YOLO (~20 min of GPU time; **pause the live pipeline while training**, R18) | |
| 6–10 | **matcher.py**, tests first: unit + metamorphic tests on synthetic tracks | Rehearse the driving scenarios (§8) |
| 10–12 | display.py (icons, side panel, mini-map, banners) | One-slide architecture diagram |
| 12–13.5 | Red-team hotkeys, coasting, camera-moved check | Electronics: turret firmware tested on its own over serial monitor |
| 13.5–15 | **turret.py**: serial link, calibration, zone + interlock, cue/mark (~1.5 h) | Mount the turret; help calibrate |
| 15–16.5 | Run the scenarios and tune thresholds from the JSONL logs | Drive the cars |
| 16.5–19 | **Sleep** / buffer | Practice the pitch |
| 19–20 | Fixes (stretch #5 only if everything else is solid) | |
| 20–22 | **Freeze the code.** 10-min thermal soak test on the 3050 (R18). Record a backup demo video | |
| 22–24 | Rehearse twice, with Q&A | |

**How to build:** one module at a time, each tested before the next. The matcher gets unit and metamorphic tests (`tests/`) using synthetic tracks, so it can be developed on the Mac without a camera.

---

## 8. Demo scenarios (5 min)
0. **Sticker disclaimer first** (§5.3): say the sticker is the GPS stand-in, then show jamming (`j`) early so judges see the stickered car drop to UNKNOWN.
1. **Baseline:** RC car = Friendly-1 (stickered) → green ✓; the two toy cars amber `NO ID` ("we don't assume hostile"). Laser stays IDLE.
2. **Crossing:** Friendly-1 crosses an unknown car → `TOO CLOSE` → re-resolves to the correct physical car afterwards (possible because the RF truth comes from the sticker, not the tracker, R1). Use a YOLO backend.
3. **Forged / replay** (`f`, `r`) → banner + rejected in the event log; labels unchanged.
4. **Cloned ID** (`d`) → both vehicles magenta `ID CONFLICT — VERIFY`, linked. The real friendly is *not* shown as hostile, and **the laser refuses to cue either one** (interlock); the operator verifies.
5. **Ghost** (`g`) → magenta hollow ring where no vehicle exists.
5b. **Motion mismatch + laser:** a claim's motion contradicts the vehicle it matches (e.g. a stick-moved car) → magenta `MOTION MISMATCH` → the **laser automatically cues** (solid) on it → operator presses `m` → the laser **blinks**. Then press Tab to try cueing the friendly RC car → `BLOCKED: FRIENDLY INTERLOCK`. This is the hardware highlight.
6. **Jamming** (`j`) → Friendly-1 `COASTING`, then fades to UNKNOWN; it is never labelled hostile.
7. *(Stretch)* the real ESP32 beacon car drives on its own odometry.

---

## 9. Likely judge questions → answers
- **"Isn't this just ADS-B/radar or AIS track correlation?"** The idea is related, but those assume cooperative, well-separated aircraft or ships. Ground vehicles bunch together, cross paths and hide under cover, so we add motion consistency, explicit ambiguity, clone/ghost detection, and a low-cost edge build.
- **"Doesn't broadcasting position give friendlies away?"** No. Challenge-response means they only transmit when an authenticated drone asks, the same principle as IFF Mode 5.
- **"What if the enemy captures a friendly radio and key?"** They can answer challenges, but then two vehicles answer as the same ID, or the claim doesn't match any vehicle's motion. We show **ID CONFLICT — VERIFY** for the operator instead of trusting either.
- **"What about jamming?"** The vehicle coasts with decaying confidence, then goes to UNKNOWN. **Never hostile by default.**
- **"Is your RF just copying the camera?"** No. The simulated beacon gets its position from a separate sensor (the sticker stands in for the vehicle's own GPS), so tracker errors show up as mismatches. The stretch goal uses the car's real odometry.
- **"Does it work on a moving drone? You can't put markers on a battlefield."** Right: the markers stand in for the drone's **navigation pose**, just as the sticker stands in for the vehicle's GPS. A real UAV georeferences every frame directly: GNSS position + altitude + IMU attitude + gimbal angles + calibrated camera intrinsics give the camera pose, and for flat ground that pose produces exactly the same pixel→ground matrix the markers give us (`H = K·[r1 r2 t]`). Everything after it (tracking, matching, interlock, laser) is unchanged. Hilly terrain adds a DEM or a gimbal laser rangefinder; visual SLAM refines the pose between GNSS updates (see §14).
- **"Why not just use IFF?"** IFF says *someone friendly is nearby*. Close air support needs *which vehicle in the sight picture*.
- **"Edge compute?"** YOLOv8n runs in real time on a laptop RTX 3050, and a Jetson Orin-class module is the target. Quote the measured FPS and latency from the display.
- **"Who decides to engage?"** The human operator. TurretX only labels and alerts, and never paints anything as "enemy".
- **"Isn't an auto-pointing laser an autonomous targeting system?"** The model only *cues* ("look here") and only for motion mismatches inside a defined designation zone. A hard-coded **friendly interlock** means it physically cannot point at anything with a friendly identity claim, even if the operator asks. Only the human *marks*.
- **"What if the software crashes while the laser is on?"** The firmware needs a heartbeat every 200 ms and switches the laser off within 500 ms without one. Servo limits keep it on the table.

---

## 10. Risks
| Risk | Mitigation |
|---|---|
| YOLO misses toy cars, especially while moving | Low conf threshold + ByteTrack buffer; YOLO-World; **`diff` fallback**; Roboflow fine-tune |
| Hands/sticks detected as vehicles (R7) | RC cars or thin sticks/thread; exclude `person`; diff blob filters |
| Motion blur | Light, locked exposure, 60 FPS |
| Lighting changes break `diff` / sticker HSV | Lock exposure; re-capture background with `b`; tune HSV at the venue |
| Timing mismatch between camera and RF (R2/R3) | Frame timestamp as the single clock; claim propagation |
| Camera bumped (R12) | Drift check + banner; recalibrate with `h` |
| Threshold tuning takes long | Synthetic + metamorphic tests first; tune from logged JSONL |
| CUDA/driver setup on the 3050 | Run setup early; `check_env.py`; CPU fallback still works at low FPS |
| 4 GB VRAM / thermal throttling (R18) | No training during live runs; on charger, Best-performance mode; soak test |
| Venue WiFi | USB phone connection; models pre-fetched |
| Live demo fails | Recorded backup video; video-file source mode |
| Solo developer burnout | Scope cut already; sleep ~16.5–19 h |
| Laser points at the wrong place or a person | Zone check + firmware servo limits + heartbeat timeout + Class 2 laser + off at power-up |
| Turret calibration off / servo jitter | 5-point calibration (`t`); lead compensation; min-move deadband; recalibrate if the turret is bumped |
| Turret not ready in time | The software runs without it (`turret.port: null` → status `TURRET OFFLINE`); demo the cue/mark logic on screen |

---

## 11. Decision log
| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | Real camera tracking + simulated RF | Full hardware; fully simulated | Visible part real, unverifiable part simulated; fits 24 h |
| 2 | Toy/RC cars on a table, angled camera | People with phones; recorded drone footage | Controlled; easy homography |
| 3 | **Air-to-ground framing** | Keep ground turret; both | Matches DISC 3 wording (air support, no FLOT) and the UAV jurors |
| 4 | Off-the-shelf detection, no training from scratch | Custom model | No CV specialist on the team |
| 5 | No ArUco markers *on the cars* (only on table corners) | Markers on cars | Would undermine the premise; enemies don't carry your markers |
| 6 | Python, single process | Go backend + Python vision | Solo developer; less glue code |
| 7 | ~~RF sim = tracker ground truth + noise~~ → **superseded by #15** | Scripted paths | — |
| 8 | Add-ons #3, #6, #2, #1; #5 stretch; cut #4, laser, web UI | Larger set | ~3.5 h budget for one developer |
| 9 | ESP32 WiFi/ESP-NOW beacon instead of LoRa | LoRa | No LoRa on hand; ESP32-S3 + Pi 5 + odometry firmware already exist |
| 10 | Phone camera (Iriun, USB), angled ≥60° | Laptop webcam overhead | Easier mounting; homography handles the angle |
| 10b | **Floor arena (~200×150 cm mat), phone rigidly mounted ~2 m up, near-overhead, 1080p, 15 cm markers, ground point = box centre** | Table with angled phone; handheld phone | More room for the RC car; steeper view cuts sticker/box bias; handheld would break calibration, motion checks and laser aim |
| 11 | Drop the friend's unknown pretrained model | Keep using it | Unknown provenance can't be debugged; `.pt` pickle can run code |
| 12 | Official YOLOv8n → compare YOLO-World / diff → optional Roboflow fine-tune | Use the unknown model | Known, debuggable, fast to fine-tune |
| 13 | Code on Mac, run on Windows RTX 3050, sync via git (Harish-hex/turretx, private) | Single machine | User preference; Mac has Claude Code; 3050 is the demo machine |
| 14 | HMAC + challenge nonce for "authentication" | Real crypto/IFF | Honest, simple, demonstrable |
| 15 | **RF truth from a roof-sticker colour detector, firewalled from the matcher** (R1) | Tracker truth + noise; ESP32 beacon early | Makes matching non-circular and the crossing scenario verifiable for ~45 min of work |
| 16 | Single clock (video PTS / monotonic) + claim time-propagation (R2, R3) | Wall clock, no alignment | Latency alone would break position matching; Mac playback speed differs |
| 17 | Challenge at 5 Hz, one reply per (nonce, rf_id); a second valid reply = clone (R4, R5) | 1 Hz challenge with 5 Hz replies | The original rejected 4 of 5 genuine messages and couldn't separate clones |
| 18 | SUSPICIOUS shown magenta with "VERIFY"; no red anywhere; icons + sub-labels (R6, R21, R23) | Red for suspicious | Red would paint a real friendly as hostile under a clone attack (fratricide risk) |
| 19 | Scenario scale 1 cm = 10 m; RSSI weight 0.10 (R9) | Raw table cm; weight 0.15 | RSSI was constant across the table; scale also improves the pitch |
| 20 | HMAC over raw payload bytes, integer fields, `compare_digest` (R14) | Sign re-serialised JSON | Float formatting differs between Python and ESP32 C |
| 21 | **Laser turret back in scope** as a "verification marker", with electronics owning the hardware and firmware | Keep it cut; automatic marking; marking anything not FRIENDLY | Strongest hardware element for the Hardware category; electronics team has capacity |
| 22 | **Designation zone + friendly interlock + auto-cue (MOTION MISMATCH only) + operator-confirmed mark** | Fully operator-driven; automatic blinking; zone-only | The automatic cue keeps the drama, the interlock prevents painting a friendly (clone case), the human mark keeps "no autonomous action" |
| 23 | Vehicle cast: RC car = Friendly-1, toy cars on sticks = unknowns | All toy cars | Only one remote-controlled car; the friendly needs the cleanest motion for the motion check |
| 24 | Laser safety: Class 2, servo limits, 500 ms heartbeat cut-off, off at power-up | Software-only checks | A crash or bug must never leave a laser on or pointing off the table |
| 25 | **Track custody** (coast up to 60 s while custody holds) + **RF DENIED** mode + last-known friendly markers | 2–3 s decay only; custody only | Jamming is the obvious judge question; confirmed identity shouldn't vanish while the drone still watches the vehicle, but must drop the instant custody is doubtful |
| 26 | Floor arena, rigid overhead mount (see 10b) | — | — |
| 27 | Matcher smoothing is time-based (τ), not per-frame α; ID CONFLICT enters immediately | Per-frame EMA α=0.3; 1 s entry for all states | Frame-rate independence (metamorphic test); security alerts and the interlock must not wait |
| 28 | Velocity history resets on implausible jumps / box-area jumps; `settled` and `merged` flags; unsettled → no mismatch; merged blob → TOO CLOSE | Fit through jumps | End-to-end run showed the real friendly flagged MOTION MISMATCH after every crossing |
| 29 | **Real beacon = MPU6050 motion signature (yaw rate + start/stop), no position** (§5.10) | Integrated speed/position; add wheel encoder | No encoder on the car; integrated IMU drifts within seconds; signature matching needs no position and is harder to spoof |
| 30 | **Disc-sticker detection by shape** for the one round event sticker (§5.3) | Colour patch only | The only sticker available is round and multi-coloured; colour thresholds fail on it |
| 31 | **Camera-loss failsafe ladder L0–L4**, labels only toward UNKNOWN, no automatic re-arm (§5.11) | Shut down on camera loss | Jury asked for a worst-case failsafe; fail-soft stays useful (RF-only blue-force picture) and is the safe direction |
| 32 | **Comms relay as simulated, authenticated, protocol-complete** (§5.12) | Real radios / real jammers; analog repeater | Hackathon timebox; real jamming is illegal in India; the HMAC identity layer is reused as the trust anchor |
| 33 | **Comms are firewalled from identity** | Let relay messages feed the matcher | A message must never be able to make a vehicle FRIENDLY |
| 34 | **One shared simulated jam; a hop restores IFF** (flag `hop_restores_iff`) | Separate jams | Coherent demo story: jam → RF DENIED → detect → hop → recovery |

---

## 12. Design review log (multi-agent-brainstorming, 2026-10-12)
**Disposition: APPROVED after revisions** (all accepted fixes are merged into the sections above).

| # | Reviewer | Finding | Resolution |
|---|---|---|---|
| R1 | Skeptic | RF sim truth = tracker output → circular; ID swaps invisible | Accepted → sticker tag locator (§5.3, D15) |
| R2 | Skeptic | Claim/track timestamps differ (~9 cm error at speed) | Accepted → claim propagation (§5.6) |
| R3 | Skeptic | Wall clock wrong on recorded video | Accepted → frame timestamp as the clock (§5.0) |
| R4 | Skeptic | 1 Hz nonce vs 5 Hz replies rejects genuine messages | Accepted → 5 Hz challenge (§5.4) |
| R5 | Skeptic | Clone replies overwrite each other | Accepted → one reply per (nonce, rf_id) (§5.5) |
| R6 | Skeptic | Clone attack paints the real friendly red | Accepted → magenta VERIFY (§5.6) |
| R7 | Skeptic | Hands detected as vehicles | Accepted → RC/sticks, exclude person, blob filters |
| R8 | Skeptic | Heading noise at low speed → false SUSPICIOUS | Accepted → heading term only above 5 cm/s |
| R9 | Skeptic | RSSI constant across the table | Accepted → scenario scale, weight 0.10 |
| R10 | Skeptic | `linear_sum_assignment` fails with `inf` | Accepted → large finite cost |
| R11 | Skeptic | Bottom-centre bias at an angle | Accepted → near-overhead mount, `ground_point_frac`, measured tag offset, tune σp |
| R12 | Skeptic | Bumped camera = silent error | Accepted → drift check banner |
| R13 | Skeptic | OpenCV frame buffering latency | Accepted → grabber thread |
| R14 | Skeptic | HMAC over re-serialised JSON | Accepted → raw bytes, ints, compare_digest |
| R15 | Skeptic | ESP32 frame/drift/clock | Accepted → marked start pose, nonce freshness |
| R16 | Skeptic | `diff` merges close cars | Accepted as a limitation → crossing demo on YOLO |
| R17 | Skeptic | Re-acquired track resets confidence | **Rejected as a bug** → documented behaviour |
| R18 | Constraint | 4 GB VRAM, thermals | Accepted → no training during live runs, soak test |
| R19 | Constraint | HMAC key in repo | Accepted as-is → labelled DEMO KEY |
| R20 | Constraint | Windows camera backend | Accepted → CAP_DSHOW everywhere |
| R21 | User | Red/green semantics & colour-blindness | Accepted → merged with R6 |
| R22 | User | Score text clutter | Accepted → side panel |
| R23 | User | Two kinds of UNKNOWN look the same | Accepted → NO ID / TOO CLOSE |
| R24 | User | Red-team actions invisible to judges | Accepted → banners |

---

## 13. Skills to use per phase (Claude Code)
| Phase | Skills | Why |
|---|---|---|
| 0. Design ✅ | `brainstorming`, `multi-agent-brainstorming` | Done |
| 1. Skeleton & setup | `uv-package-manager`, `agent-protocols:incremental-implementation`, `agent-protocols:karpathy-guidelines` | Correct uv usage; small runnable steps; minimal code |
| 2a. Tracker, homography, tag locator | `computer-vision-expert`, `python-performance-optimization` | Makes up for no CV specialist; keeps FPS up |
| 2b. RF sim / receiver | `agent-protocols:security-and-hardening`, `constant-time-analysis` | HMAC, nonces, replay, raw-bytes signing |
| 2c. Matcher | `agent-protocols:test-driven-development`, `coding-skills:metamorphic-testing`, `coding-skills:clocks-and-time`, `coding-skills:floating-point` | Tests first; property tests (translation/ID-swap invariance); timing; exp/cos edge cases |
| 3. Integration & tuning | `systematic-debugging`, `agent-protocols:debugging-and-error-recovery` | Root-cause from JSONL logs, not guesswork |
| Every commit | `/simplify`, `/code-review`, `commit` | Clean solo code; readable history to roll back |
| 5. Freeze | `verification-before-completion`, `codebase-audit-pre-push` | Prove each scenario works before recording the backup |
| Laser turret (`turret.py`) | `agent-protocols:test-driven-development` | Unit-test the zone + interlock rules before connecting hardware |
| Stretch: ESP32 / turret firmware | `c-pro` | Firmware edits (`harzard-poc` beacon, turret sketch) |
| Fine-tune | `computer-vision-expert` | Check Roboflow export and training settings |

---

## 14. Path to a real UAV (pitch + roadmap)
**What changes is only where the pixel→ground mapping comes from.**

```
Demo:   4 ArUco markers ─────────────┐
                                     ├──► H (pixels → ground) ──► tracker → matcher → turret
Drone:  GNSS + IMU + gimbal + K ─────┘    recomputed every frame
```

| Demo stand-in | Real UAV equivalent |
|---|---|
| ArUco corner markers → homography | **Direct georeferencing:** camera pose from GNSS (RTK for cm), barometer/laser altimeter, IMU attitude, gimbal encoders, and intrinsics `K` from a one-time checkerboard calibration. For flat ground, `H = K·[r1 r2 t]` |
| Flat table plane | Flat-earth near the target, or a **DEM** (terrain model) / **gimbal laser rangefinder** for hills |
| Table cm, 1 cm = 10 m scale | Real lat/lon / local metres. Friendly radios already report GNSS, so **both sides share world coordinates** |
| Roof sticker → simulated GPS | The friendly vehicle's own GNSS/INS in its IFF reply |
| Phone over USB → laptop RTX 3050 | Gimbal camera → onboard edge computer (Jetson Orin-class) |
| Laser pointer turret | Operator-confirmed designation on the aircraft's sensor (interlock unchanged) |

**What gets harder:**
- **Accuracy is metres, not centimetres.** A 0.5° attitude error at 300 m slant range is ≈2.6 m on the ground; GNSS without RTK adds 2–5 m; frame/IMU time-sync errors add more. σp and the gate scale to metres, which is exactly why the **motion-consistency check** matters: position alone can't separate vehicles 5 m apart; motion over time can.
- **Time sync:** every frame must be stamped against the same clock as the IMU and gimbal (the same principle as our single frame-timestamp clock, R3).
- **Pose refinement:** visual odometry / SLAM between GNSS updates, or registration against satellite imagery. **This plays to Aerion's stated SLAM and edge-CV strengths, so it's a natural integration conversation.**

