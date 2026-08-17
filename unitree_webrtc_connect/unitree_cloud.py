"""
Minimal Unitree cloud client used for the Remote (STA-T) WebRTC signaling
flow and for fetching the per-device AES-128 key.

Why curl_cffi?  The Unitree cloud is fronted by Cloudflare-style TLS / JA3
fingerprinting. Plain `requests` gets blocked. `curl_cffi` impersonates a
real Chrome handshake, which is what the Unitree apk does on the wire.

Region and device_type are first-class parameters because the robot
families hit different app-name headers on the same host — the Go2 apk
uses `global-robot-api.unitree.com` with `AppName: Go2`, while the
humanoids (G1 and R1) both ship under Unitree Explorer and send
`AppName: B2`. See APP_NAME below. The `cn` region lives at
`robot-api.unitree.com` and is required for accounts registered in China.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from curl_cffi import requests as cffi_requests


# ─── Constants ────────────────────────────────────────────────────────

APP_SIGN_SECRET = "XyvkwK45hp5PHfA8"

BASE_URLS = {
    "global": "https://global-robot-api.unitree.com/",
    "cn": "https://robot-api.unitree.com/",
}

# Mirror the apk's headers verbatim — minor mismatches (e.g. AppVersion)
# can flip the cloud's response from `code:100` to `code:1003`.
_BASE_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "DeviceId": "Samsung/Samsung/SM-S931B/s24/14/34",
    "DevicePlatform": "Android",
    "DeviceModel": "SM-S931B",
    "SystemVersion": "34",
    "AppVersion": "1.11.4",
    "AppLocale": "en_US",
    "Channel": "UMENG_CHANNEL",
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 14; SM-S931B Build/AP3A.240905.015.A2; wv) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/127.0.6533.103 "
        "Mobile Safari/537.36"
    ),
}

VALID_REGIONS = ("global", "cn")
VALID_DEVICE_TYPES = ("Go2", "G1", "R1")

# `device_type` is the robot you're talking to; `AppName` is the account
# namespace the cloud signs against, and the two are not the same thing.
# Go2 has its own mobile app and its own namespace. The humanoids all ship
# under Unitree Explorer, which sends `AppName: B2` for the whole line — so
# G1 and R1 share one account and one namespace.
APP_NAME = {
    "Go2": "Go2",
    "G1":  "B2",
    "R1":  "B2",
}


# ─── Errors ───────────────────────────────────────────────────────────

class UnitreeCloudError(RuntimeError):
    """Raised when a cloud API call returns a non-success code."""

    def __init__(self, action: str, code, msg: str):
        super().__init__(f"{action} failed: code={code} msg={msg!r}")
        self.action = action
        self.code = code
        self.msg = msg


# ─── Data class ───────────────────────────────────────────────────────

@dataclass
class RobotDevice:
    sn: str = ""
    alias: str = ""
    series: str = ""
    model: str = ""
    mac: str = ""
    online: Optional[bool] = None
    key: str = ""           # The AES-128 key (32 hex chars) used for data2=3.
    raw: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "RobotDevice":
        return RobotDevice(
            sn=d.get("sn", ""),
            alias=d.get("alias", ""),
            series=d.get("series", ""),
            model=d.get("model", ""),
            mac=d.get("mac", ""),
            online=d.get("online"),
            key=d.get("key", "") or d.get("gcm_key", ""),
            raw=d,
        )


# ─── Cloud client ─────────────────────────────────────────────────────

class UnitreeCloud:
    """Talks to the Unitree cloud the same way the official apk does."""

    def __init__(self, region: str = "global", device_type: str = "Go2",
                 access_token: str = "", refresh_token: str = ""):
        if region not in VALID_REGIONS:
            raise ValueError(
                f"region must be one of {VALID_REGIONS}, got {region!r}"
            )
        if device_type not in VALID_DEVICE_TYPES:
            raise ValueError(
                f"device_type must be one of {VALID_DEVICE_TYPES}, got {device_type!r}"
            )
        self.region = region
        self.device_type = device_type
        self.base_url = BASE_URLS[region]
        self.access_token = access_token
        self.refresh_token = refresh_token
        # Chrome impersonation — bypasses the JA3/TLS fingerprint filter.
        self._session = cffi_requests.Session(impersonate="chrome120")

    # ── header / sign helpers ─────────────────────────────────────────

    def _headers(self) -> dict:
        ts = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex
        sign = hashlib.md5(f"{APP_SIGN_SECRET}{ts}{nonce}".encode()).hexdigest()
        return {
            **_BASE_HEADERS,
            "AppTimezone": time.strftime("%Z") or "UTC",
            "AppTimestamp": ts,
            "AppNonce": nonce,
            "AppSign": sign,
            "AppName": APP_NAME[self.device_type],
            "Token": self.access_token,
        }

    # ── low-level request with one auto-refresh on 1001 ───────────────

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = self.base_url + path
        headers = self._headers()

        if method == "GET":
            resp = self._session.get(url, params=body or {}, headers=headers)
        elif method == "POST":
            resp = self._session.post(url, data=body or {}, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        resp.raise_for_status()
        result = resp.json()

        if result.get("code") == 1001 and self.refresh_token:
            if self._refresh():
                headers = self._headers()
                if method == "GET":
                    resp = self._session.get(url, params=body or {}, headers=headers)
                else:
                    resp = self._session.post(url, data=body or {}, headers=headers)
                resp.raise_for_status()
                result = resp.json()

        return result

    def _refresh(self) -> bool:
        url = self.base_url + "token/refresh"
        resp = self._session.post(
            url,
            data={"refreshToken": self.refresh_token},
            headers=self._headers(),
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") == 100 and result.get("data"):
            self.access_token = result["data"].get("accessToken", "")
            self.refresh_token = result["data"].get("refreshToken", self.refresh_token)
            return True
        return False

    def _check(self, result: dict, action: str):
        if result.get("code") != 100:
            raise UnitreeCloudError(action, result.get("code"), result.get("errorMsg", ""))
        return result.get("data")

    # ── auth ──────────────────────────────────────────────────────────

    def login_email(self, email: str, password: str) -> str:
        """Login with email/password. Returns the access token (also stored
        on the client for subsequent calls)."""
        result = self._request("POST", "login/email", {
            "email": email,
            "password": hashlib.md5(password.encode()).hexdigest(),
        })
        data = self._check(result, "login/email")
        self.access_token = data.get("accessToken", "")
        self.refresh_token = data.get("refreshToken", "")
        return self.access_token

    # ── device ────────────────────────────────────────────────────────

    def list_devices(self) -> list[RobotDevice]:
        """`device/bind/list` — returns every robot bound to the account.
        On V3-capable firmware (G1 ≥ 1.5.1, Go2 ≥ 1.1.15, all R1) the
        per-device AES-128 key is in `dev.key`. `dev.series` identifies the
        family — "Go2", "G1" or "R1"; note that G1 and R1 are bound to the
        same account, so a humanoid listing can contain both."""
        result = self._request("GET", "device/bind/list")
        data = self._check(result, "device/bind/list") or []
        return [RobotDevice.from_dict(d) for d in data]

    # ── webrtc ────────────────────────────────────────────────────────

    def get_pub_key(self) -> str:
        """`system/pubKey` — base64 PEM of the cloud's RSA public key,
        used to wrap session-AES keys for `webrtc/account` and
        `webrtc/connect`. No auth required."""
        result = self._request("GET", "system/pubKey")
        return self._check(result, "system/pubKey") or ""

    def webrtc_account(self, sn: str, sk_rsa_b64: str) -> str:
        """`webrtc/account` — returns the AES-encrypted TURN server config
        (URL + temporary credentials). Caller decrypts with the AES key
        whose RSA-wrapped form was sent as `sk`."""
        result = self._request("POST", "webrtc/account", {
            "sn": sn,
            "sk": sk_rsa_b64,
        })
        return self._check(result, "webrtc/account") or ""

    def webrtc_connect(self, sn: str, sk_rsa_b64: str, data_aes_b64: str,
                        timeout: int = 5) -> str:
        """`webrtc/connect` — submits the AES-encrypted SDP offer and
        receives the AES-encrypted SDP answer. The cloud routes by SN to
        whichever robot is currently registered with the relay; if the
        robot isn't online, the cloud returns code 1000 ("device not
        online")."""
        result = self._request("POST", "webrtc/connect", {
            "sn": sn,
            "sk": sk_rsa_b64,
            "data": data_aes_b64,
            "timeout": timeout,
        })
        # webrtc/connect returns the encrypted answer in `data`; we let the
        # caller decode and surface a friendlier error on `code:1000`.
        if result.get("code") == 1000:
            raise UnitreeCloudError(
                "webrtc/connect",
                1000,
                "Device not online — the robot hasn't registered with the "
                "cloud relay. Check the network/Internet-remote toggle on "
                "the robot side, or use LocalSTA / LocalAP instead.",
            )
        return self._check(result, "webrtc/connect") or ""


# ─── Convenience helpers ──────────────────────────────────────────────

def fetch_aes_key(email: str, password: str, sn: str,
                   region: str = "global", device_type: str = "Go2") -> str:
    """One-shot lookup: login → device/bind/list → return `dev.key` for `sn`.
    Raises `UnitreeCloudError` if the account isn't logged in or the SN
    isn't bound."""
    cloud = UnitreeCloud(region=region, device_type=device_type)
    cloud.login_email(email, password)
    devices = cloud.list_devices()
    for d in devices:
        if d.sn == sn:
            if not d.key:
                raise UnitreeCloudError(
                    "fetch_aes_key", 0,
                    f"Device {sn} is bound but the cloud returned an empty "
                    f"`dev.key`. Check the firmware version (data2=3 / V3 "
                    f"is required — G1 ≥ 1.5.1, Go2 ≥ 1.1.15, or any R1).",
                )
            return d.key
    raise UnitreeCloudError(
        "fetch_aes_key", 0,
        f"SN {sn} is not bound to this account. Pair the robot through the "
        f"official apk first, or check that you're using the right region "
        f"({region!r}).",
    )
