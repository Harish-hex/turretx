# Eval 2 plan (4 hours)

Goal for the eval: answer the jury's two points **live**, not on a slide.
1. Camera fails → the system degrades and stays useful (RF-only friendly map, laser locked).
2. The drone doubles as a comms relay when ground links are jammed.

Everything else becomes the roadmap slide.

## Who does what

| Time | Software (Harish) | Electronics team | Deck/research |
|---|---|---|---|
| 0:00–0:20 | Close out the round-sticker work: fix the broken test, commit | ESP-NOW relay firmware: node A → drone node → node B | Slides: failsafe ladder, relay precedents (from the report) |
| 0:20–1:00 | Real-camera detection: diff backend as the primary on the real arena, crop to the arena before YOLO; test on the 3050 | (continue) | (continue) |
| 1:00–2:00 | **Camera failsafe** (below) | (continue) | Demo script and the wording for the caveats |
| 2:00–2:50 | **Comms relay MVP** (below) | Hand over the serial output format | |
| 2:50–3:30 | Integrate on the 3050, rehearse the 3 scenes, **record a backup video** | Help rehearse | Time the pitch |
| 3:30–4:00 | Buffer. Nothing new starts after 3:30. | | |

## 1. Camera failsafe (about 60 min)

- `camera.py`: a health check runs every frame and returns `OK / DEGRADED / LOST`.
  - **LOST:** no new frame for more than 0.5 s, or a black frame (camera covered).
  - **DEGRADED:** a frozen frame (practically no change across about 1 s while RF reports friendlies moving), or a homography drift latch.
- `main.py`: picks the rung from camera health and RF state.
  - **L0** camera OK.
  - **L2 RF-only**: camera LOST, RF alive.
  - **L3 last-known**: RF stale too.
  - **L4 safe**: both gone. L1 is folded into L0/L2 for now.
- `display.py`: when the camera is LOST, draw a schematic arena map instead of the video.
  - Friendlies shown at their signed claimed positions, with beacon age.
  - Banner: "CAMERA LOST: RF-ONLY. Non-emitting vehicles NOT visible".
  - Stale friendlies shown with a growing no-fire ring.
  - A "DESIGNATION LOCKED" indicator that only re-arms when the operator presses a key once the camera is back.
- Hotkey `c` simulates a camera kill. Physically covering the lens also works, via the black-frame check.
- Tests: the health classifier, the rung selection, and that labels never move away from UNKNOWN when data is missing.

## 2. Comms relay MVP (about 50 min)

- `relay.py` (software sim first):
  - Ground units send short text and status messages to each other.
  - The direct path can be blocked or jammed (hotkey). Messages are then routed **via the drone**.
  - Messages are signed with the existing HMAC key and have replay counters, and a forged message is rejected.
  - Queued messages are delivered when the link returns.
- Jam detector: low packet delivery while signal stays strong → "JAMMING SUSPECTED" → hop to the next channel on a schedule derived from the key.
- Display: a comms side panel with the message log, the route taken (direct / via the drone), and the queue.
- **Stretch:** if the electronics team's ESP-NOW nodes print to serial by 2:00, `relay.py` reads real packets instead of the simulation. If not, demo the simulation and show the hardware separately.

## Demo scenes (rehearse in this order)

1. **Normal:** the friendly is identified; a cloned ID is flagged SUSPICIOUS.
2. **Camera covered:**
   - Drops to RF-only. Friendlies are still on the map, and the laser is locked.
   - Uncover the lens; the operator re-arms.
3. **Jam:**
   - The direct link fails and comms reroute through the drone.
   - The jam is detected, the channel hops, and the queued messages are delivered.
   - The jam is a simulation toggle or a metal barrier. Never a real jammer: that's illegal in India.

## Cut (roadmap slide only)

- Laser turret code
- RF direction-finding (servo-swept antenna)
- LED optical uplink
- LoRa
- MPU6050 motion-signature matching
- Thermal or second camera

## Risks

- **Real camera detection** is the biggest risk. If the diff backend isn't solid by 1:00, run the demo on a recorded real clip and stop tuning.
- **Relay hardware may not be ready.** The software simulation is the fallback and is enough to make the point.
- **The plan has no slack.** If anything slips, cut in this order: hardware relay, the comms side panel polish, the L3 rung.
