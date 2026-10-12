"""Radio propagation model shared by the simulator (adds noise) and the matcher (expects it).
Log-distance path loss from a virtual drone above the arena centre, in scenario metres (R9)."""
import math


def expected_rssi(x_cm, y_cm, arena_cm, scale_m_per_cm, rssi_cfg):
    dx = (x_cm - arena_cm[0] / 2) * scale_m_per_cm
    dy = (y_cm - arena_cm[1] / 2) * scale_m_per_cm
    d = math.sqrt(dx * dx + dy * dy + rssi_cfg["drone_alt_m"] ** 2)
    return rssi_cfg["p0"] - 10 * rssi_cfg["n"] * math.log10(d)
