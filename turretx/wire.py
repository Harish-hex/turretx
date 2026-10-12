"""RF wire format shared by the friendly beacons (rfsim) and the drone receiver (rfrx).

A packet is {"payload": bytes, "sig": hex str}. The payload is compact JSON with INTEGER
fields only (mm, mm/s, ms, dBm×10), and the signature is HMAC-SHA256 over the payload bytes
exactly as transmitted, never over re-serialised JSON (R14: Python and ESP32 C format floats
differently).
"""
import hashlib
import hmac
import json
import os

MAX_PAYLOAD_BYTES = 256
FIELDS = ("id", "n", "t", "x", "y", "vx", "vy", "rssi")


def rf_key(cfg_key):
    """Shared key: TURRETX_RF_KEY env var if set, else the labelled demo key from config (R19)."""
    return os.environ.get("TURRETX_RF_KEY", cfg_key).encode()


def encode_payload(rf_id, nonce, t, x_cm, y_cm, vx_cm_s, vy_cm_s, rssi_dbm):
    fields = {"id": rf_id, "n": nonce, "t": round(t * 1000),
              "x": round(x_cm * 10), "y": round(y_cm * 10),
              "vx": round(vx_cm_s * 10), "vy": round(vy_cm_s * 10),
              "rssi": round(rssi_dbm * 10)}
    return json.dumps(fields, separators=(",", ":")).encode()


def sign(key, payload):
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def make_packet(key, payload):
    return {"payload": payload, "sig": sign(key, payload)}


def signature_ok(key, packet):
    sig = packet.get("sig")
    if not isinstance(sig, str) or len(sig) != 64:
        return False
    return hmac.compare_digest(sign(key, packet["payload"]), sig)  # constant-time compare


def decode_payload(payload):
    """Parse an AUTHENTICATED payload into a dict in cm / cm/s / s / dBm; None if malformed."""
    try:
        d = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict) or set(d) != set(FIELDS):
        return None
    if not (isinstance(d["id"], str) and 0 < len(d["id"]) <= 16
            and isinstance(d["n"], str) and 0 < len(d["n"]) <= 32):
        return None
    ints = ("t", "x", "y", "vx", "vy", "rssi")
    if not all(type(d[k]) is int for k in ints):  # bool is not accepted
        return None
    if not (abs(d["x"]) <= 100_000 and abs(d["y"]) <= 100_000 and abs(d["vx"]) <= 50_000
            and abs(d["vy"]) <= 50_000 and -2000 <= d["rssi"] <= 0 and d["t"] >= 0):
        return None
    return {"rf_id": d["id"], "nonce": d["n"], "t": d["t"] / 1000,
            "x": d["x"] / 10, "y": d["y"] / 10, "vx": d["vx"] / 10, "vy": d["vy"] / 10,
            "rssi": d["rssi"] / 10}
