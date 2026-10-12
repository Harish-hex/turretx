# Relay integration approach

Inputs: `reports/Camera failsafe and airborne relay.md`, `turretx_airborne_relay_market_study.xlsx` (S01–S30), EVAL2_PLAN §2.
Draft code on disk (uncommitted, not wired): `turretx/relay.py`, `tests/test_relay.py`.

## 1. Claim discipline (what the study allows)

Say: *authenticated relay + blue-force tracking that keeps working after camera loss; jamming is simulated.*
Do not say: anti-jam performance, HAPS/SATCOM, production data rates, MoD relay demand (S24–S27 are unsupported).
Prototype radio is ESP-NOW (S30): a demo link, not a fielded radio. Real radio stack, link budget and WPC/DoT approval (S15, S16) go on the roadmap slide as the gap we know about.

## 2. Architecture rules

1. **Comms never decide identity.** The relay shares the HMAC key but is a separate lane. `matcher.py`, `tracker.py` and `rfrx.py` must never import `relay.py` (AST test, same as the sticker firewall). A message saying "I am F-1" changes nothing about F-1's label.
2. **Protocol is separate from the channel.** `relay.py` = protocol (sign, dedup, TTL, priority queue, receipts, jam detector, hop schedule). A small `Link` interface = delivery. Two links: `SimLink` (barrier/jam flags, used for the eval) and `SerialLink` (ESP32 bridges, if ready). Swapping them must not touch protocol code.
3. **Fail toward silence.** Anything that fails verification is dropped and logged, never forwarded.

## 3. Protocol changes needed (draft is missing these; study sheet 5 lists them)

| Item | Draft today | Change |
|---|---|---|
| Replay | strictly increasing seq per sender | **seen-set of (src, seq) within TTL.** Strictly increasing breaks as soon as priority reorders the queue |
| TTL | none | each message expires (default 30 s); expired = dropped and logged |
| Dedup | implicit | explicit; a message delivered twice (direct + via drone) shows once |
| Priority | FIFO | 0 alert > 1 position/status > 2 chat; bounded queue (e.g. 20), drop the lowest priority first |
| Receipts | none | destination returns a signed receipt; sender's log shows `sent → delivered`. Receipt goes through the same relay |
| Message types | text only | `txt`, `pos` (friendly position report), `sys` (drone notices) |

## 4. How it ties to the failsafe (the point of the pitch)

- When the failsafe leaves L0, the drone broadcasts a `sys` alert: "CAMERA LOST - RF-only picture follows" at priority 0.
- In L2/L3 the drone relays `pos` messages built from its authenticated claims, so ground units still receive the blue-force picture the drone can no longer verify visually. It stays an information centre without the camera.
- Those `pos` messages carry age, so stale positions are labelled stale, same as the drone's own RF-only map.

## 5. One shared channel (decision for the demo story)

Jam hotkey `j` already jams the IFF link (RF DENIED, COASTING). Make it one RF channel: jam sets the jammed channel, the IFF link and the relay both go dark, the relay detects it (low delivery + high noise), hops on the keyed schedule, and **IFF recovers on the new channel**. Config flag `relay.hop_restores_iff: true`; set `false` to keep the old jam-first demo (RF DENIED persists).
New hotkey `k` toggles the barrier (direct ground link blocked, drone relay still works).
Both are simulation flags. No RF is transmitted by any of this.

## 6. Hardware path (electronics team, parallel, optional)

Keep the ESP32s dumb: ESP-NOW **passthrough bridges**, no crypto or logic on them.
- Frame on the serial line (laptop ↔ ESP32): one line of JSON, `{"payload":"<hex>","sig":"<hex>"}`, max 256 payload bytes (same as `wire.MAX_PAYLOAD_BYTES`). The laptop does all verification, dedup and queueing.
- Ground node A and B each get an ESP32 plugged into a laptop USB port (or the Pi); the "drone node" is the laptop itself.
- `SerialLink` reads and writes those lines. If it isn't ready at hour 2:00 we demo `SimLink` and show the hardware separately. Do not block on it.

## 7. Display

Side-panel comms block: channel number, `JAM SUSPECTED` banner (labelled SIMULATED), barrier indicator, queue length, last ~6 messages with route tag (`direct` / `via drone` / `queued` / `rejected`) and receipt state. Red team hotkeys: forged message and replay get a visible "rejected: bad signature / replay" line.

## 8. Build order and time

| Step | Work | Est. |
|---|---|---|
| 1 | Protocol rework in `relay.py` + tests (seen-set replay, TTL, dedup, priority, receipts, message types, `Link` interface, `SimLink`) | 30 min |
| 2 | Firewall AST test + wire into `main.py` (step each loop, `j`/`k`, `pos`/`sys` messages from failsafe and claims, shared jam/hop) | 25 min |
| 3 | Comms panel in `display.py` + render test | 20 min |
| 4 | `SerialLink`, only if firmware is ready | 20 min |
| 5 | Rehearse scenes; record backup video | with the group |

## 9. Cut / not now

Real RF jamming, LoRa, optical uplink, frequency-hopping radios, mesh routing beyond one drone, HAPS/SATCOM. All stay on the roadmap slide.
