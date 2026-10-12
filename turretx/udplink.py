"""Real friendly beacons over WiFi UDP (firmware/turretx_rover): the drone side of the IFF link.

Challenges go out as compact JSON {"n":"<nonce>","t":<ms>} (the firmware's parser needs it
compact) by UNICAST to every known beacon. Measured on the demo laptop: 255.255.255.255 never
reached the rover (Windows routed it to another adapter), subnet broadcast arrived 6/10 (an ESP32
running BLE + WiFi sleeps through broadcasts), unicast 14/14.

Beacon addresses come from config, or with "auto" are learned from each beacon's own telemetry
broadcasts (UDP 4212), only after its HMAC checks out, so a forged packet can't redirect challenges.

Replies {"payload":"<hex>","sig":"<hex>"} are handed to rfrx.Receiver as ordinary packets; it does
all verification (signature, schema, nonce freshness, replay, clone). A datagram that doesn't even
parse is passed on with no payload, so the receiver logs it as malformed.

The RSSI in a hardware claim is the beacon's own WiFi signal, not what the drone receives, so
packets from here carry rssi_known=False and the matcher leaves R out of the confidence.
"""
import json
import socket

from turretx import wire

MAX_DATAGRAM = 2048


class UdpBeaconLink:
    def __init__(self, ucfg, key):
        """ucfg: {"targets": ["auto" | "<ip>", ...], "challenge_port", "reply_port", "telemetry_port"}"""
        self.key = key
        self.challenge_port = ucfg.get("challenge_port", 4210)
        targets = ucfg.get("targets", ["auto"])
        self.auto = "auto" in targets
        self.targets = {t for t in targets if t != "auto"}
        self.telemetry = {}  # beacon ip → its latest verified telemetry dict
        self.rx = self._bind(ucfg.get("reply_port", 4211))
        self.tel = self._bind(ucfg.get("telemetry_port", 4212)) if self.auto else None
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @staticmethod
    def _bind(port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", port))
        s.setblocking(False)
        return s

    @property
    def reply_port(self):
        return self.rx.getsockname()[1]

    @property
    def telemetry_port(self):
        return self.tel.getsockname()[1] if self.tel else None

    def send_challenge(self, challenge):
        """Send a rfrx.Interrogator challenge {"nonce", "t" (s)} to every known beacon."""
        msg = json.dumps({"n": challenge["nonce"], "t": max(0, round(challenge["t"] * 1000))},
                         separators=(",", ":")).encode()
        for ip in sorted(self.targets):
            try:
                self.tx.sendto(msg, (ip, self.challenge_port))
            except OSError:
                pass  # beacon gone or network down: it simply won't answer

    @staticmethod
    def _drain(sock):
        out = []
        while True:
            try:
                out.append(sock.recvfrom(MAX_DATAGRAM))
            except BlockingIOError:
                return out
            except ConnectionResetError:
                continue  # Windows reports an earlier ICMP "port unreachable" here; skip it
            except OSError:
                return out

    @staticmethod
    def _packet(data):
        try:
            d = json.loads(data)
            return {"payload": bytes.fromhex(d["payload"]), "sig": d["sig"], "rssi_known": False}
        except (ValueError, KeyError, TypeError):
            return {"payload": None, "sig": None, "rssi_known": False}

    def poll(self):
        """Reply packets received since the last call, for rfrx.Receiver.receive()."""
        if self.tel is not None:
            for data, (ip, _) in self._drain(self.tel):
                pkt = self._packet(data)
                if (isinstance(pkt["payload"], bytes) and len(pkt["payload"]) <= wire.MAX_PAYLOAD_BYTES
                        and wire.signature_ok(self.key, pkt)):
                    self.targets.add(ip)
                    try:
                        self.telemetry[ip] = json.loads(pkt["payload"])
                    except ValueError:
                        pass
        return [self._packet(data) for data, _ in self._drain(self.rx)]

    def close(self):
        for s in (self.rx, self.tel, self.tx):
            if s is not None:
                s.close()
