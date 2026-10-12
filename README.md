# TurretX

Air-to-ground friend-or-foe identification: matches authenticated RF identities to camera-tracked ground vehicles and labels each one **FRIENDLY / UNKNOWN / SUSPICIOUS** with an explainable confidence score. Built for SEDHACKS'26 (Space Applications & Defense Technology, Hardware) by team WhatTheHack.

- **[TURRETX_DESIGN.md](TURRETX_DESIGN.md)**: full context, scope, architecture, matching logic, build plan, decision log. **Read this first.**
- **[DEPENDENCIES.md](DEPENDENCIES.md)**: machine setup (Windows RTX 3050 = demo, Mac M1 = dev), phone camera, scene checklist.

Quick setup:
```powershell
# Windows (RTX 3050)
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```
```bash
# Mac
bash setup_mac.sh
```
