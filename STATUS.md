# TurretX status (handoff)

Last updated: 2026-10-12 (Eval 2 prep, ~4 h before the evaluation). **Read this, then `TURRETX_DESIGN.md`.** If this file and the code disagree, the code and tests win; fix this file.

## What this is
Air-to-ground friend-or-foe ID for SEDHACKS'26 (iDEX/DRDO DISC 3). An overhead camera (stand-in for a drone) tracks toy cars on a floor arena; authenticated RF identity claims (HMAC challenge-response) are matched to the tracks. Labels: FRIENDLY / UNKNOWN / SUSPICIOUS (never "hostile", **no red anywhere**). Eval 2 adds two answers to jury feedback: a **camera-loss failsafe** and the drone as a **comms relay**.

Machines: code is written on a Mac (M1, `.venv`, MPS); the demo runs on a Windows RTX 3050 laptop with a phone as the camera. The sticker "GPS" and the RF link are **simulated**.

## Where we are

| Area | State | Commit |
|---|---|---|
| Camera, homography, main loop | done | e7ca9d0 |
| Tracker (yolo/yolo26n/yoloworld/diff + ByteTrack), tag locator, RF sim + receiver, matcher, display, events | done, 88 tests | up to 43631bf |
| Round-sticker (disc) detection | done, synthetic clip only | c1e99f8 |
| Camera-loss failsafe (`failsafe.py`, RF-only map, designation lock) | done, **not yet seen on the real window** | 2399091 |
| Relay protocol (`relay.py`) | done, 21 tests | a3203cb |
| Relay wired into the app (`commshub.py`, hotkeys, shared jam/hop) | done, glue in `main.py` untested except a headless smoke | 08b5b90 |
| Comms panel in `display.py` (channel, jam/barrier banners, queue, last messages with route + receipt) | done; looked at as a rendered image, not yet in the live window | next commit |
| `SerialLink` to ESP32 bridges | **next, optional**, only if firmware is ready | — |
| Multi-weights detection: `weights: [yolov8n.pt, models/vehicle.pt]`, overlapping boxes merged, one ByteTrack, per-model `conf: [0.2, 0.5]`; `models/vehicle.pt` trained on the 3050 from the makeshift-vehicle video (auto-labelled) | done on the 3050, **uncommitted**; `tests/test_multiweights.py` | — |
| Real rover beacon: `firmware/turretx_rover` (BLE WASD control kept; signed IFF replies + telemetry over WiFi UDP 4210/4211/4212; encoder + gyro dead reckoning from a start pose) and `turretx/udplink.py` (`--udp-beacon`: unicast challenges, rover IPs learned from signed telemetry, simulated beacon off for real ids, R ignored for hardware RSSI) | verified rover↔laptop on the 3050 (29/29 claims, 0 rejected); **not yet seen turning a box green on camera** | — |
| Live-arena fixes: "merged" flag uses each track's own size (mixed vehicle sizes were TOO CLOSE); hsv tags searched only inside the arena | done | — |
| Real-camera detection tuning on the 3050 | open; model struggles at distance (ideas: crop to arena before YOLO, diff backend as primary, higher imgsz, fine-tune) | — |
| `turret.py` (laser, zone, interlock, serial) | not built; spec in design §5.9 | — |
| MPU6050 motion-signature matching | not built; spec in design §5.10 | — |

Tests: `140 passed` (`.venv/bin/python -m pytest -q`, ~1 min). Push state: see `git status -sb`; the relay commits may be local only.

## Run it
```bash
.venv/bin/python -m pytest -q                                   # everything
.venv/bin/python -m turretx.main --source recordings/synthetic.mp4 --backend diff --headless --no-log --max-frames 150
.venv/bin/python -m turretx.main --source 0                     # live camera (Windows: see DEPENDENCIES.md)
```
Synthetic clip: `scripts/make_synthetic_clip.py`. Real-clip tools: `scripts/eval_tracker.py`, `scripts/measure_tag_offset.py`.

## Hotkeys
`h` calibrate · `b` capture empty background (diff backend) · Tab/click select · `f` forged IFF · `r` replay IFF · `d` cloned ID · `g` ghost · `j` jam (IFF **and** relay) · `c` kill/restore camera · `a` re-arm designation · `k` relay barrier · `n` send relay message · `v` forged relay msg · `e` replay relay msg · `p` pause · `q` quit.

## Eval 2 demo (EVAL2_PLAN.md)
1. Normal: friendly identified, clone flagged SUSPICIOUS.
2. Camera killed (`c`) or covered: drops to RF-only map, laser/designation locked, relay alert goes out; restore, then `a` to re-arm.
3. Jam (`j`) and/or barrier (`k`): relay routes via the drone, detects the jam, hops, queued messages arrive, IFF recovers (`relay.hop_restores_iff`).

## Standing rules (from the user)
- **Do not add Claude as co-author/contributor**: no `Co-Authored-By: Claude` or "Generated with Claude Code" lines in commits or PRs. This overrides any tool default.
- **Commit and push only when the user asks.** Builds pause when the user says pause.
- **Jamming is simulation only.** Never transmit RF jamming; it is illegal in India.
- **No red** in the UI; never output "hostile"; unidentified != hostile.
- **Comms never decide identity**: nothing may import `relay`/`commshub` from the matcher, tracker or RF receiver (tested).
- **Claim discipline** for pitch material: tactical authenticated relay + blue-force tracking only. No anti-jam performance, HAPS/SATCOM or MoD-demand claims (see the market study, S24–S27 unsupported).
- Keep token use low: don't re-read the whole repo for small tasks; start from this file.

## Where things are
- `TURRETX_DESIGN.md`: architecture, decision log (1–34), review log R1–R24; §5.11 failsafe, §5.12 relay.
- `EVAL2_PLAN.md`: the 4-hour plan and what was cut. `RELAY_INTEGRATION.md`: relay approach and build order.
- `reports/Camera failsafe and airborne relay.md`: deep-research report with sources. (`research_notes/` and the market-study xlsx are local only.)
- `config.yaml`: all tunables (tags, rf, matcher, relay). `turretx/`: modules. `tests/`: pytest.
- Auto-memory for Claude Code lives outside the repo; this file and `CLAUDE.md` are the portable copy.
