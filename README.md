# Unitree WebRTC Connect

Python WebRTC driver for Unitree Go2, G1 and R1 robots. Provides high-level control through the same WebRTC protocol used by the Unitree Go/Unitree Explore mobile apps — no jailbreak or firmware modification required.

![Screenshot](https://github.com/legion1581/unitree_webrtc_connect/raw/master/images/screenshot_1.png)

[![PyPI](https://img.shields.io/pypi/v/unitree-webrtc-connect.svg)](https://pypi.org/project/unitree-webrtc-connect/)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/legion1581/unitree_webrtc_connect)](https://github.com/legion1581/unitree_webrtc_connect/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Supported Models

| Model | Variants |
|-------|----------|
| **Go2** | AIR, PRO, EDU |
| **G1** | AIR, EDU |
| **R1** | AIR, PRO, EDU |

## Supported Firmware

| Robot | Firmware Versions | Auth |
|-------|-------------------|------|
| **Go2** | 1.0.19 – 1.0.25<br>1.1.1 – 1.1.14<br>1.1.15+ *(latest)* | static GCM key (`data2=2`)<br>static GCM key (`data2=2`)<br>per-device AES-128 key (`data2=3`) — see [AES-128 Key](#aes-128-key-data23-g1--151--go2--1115--all-r1) |
| **G1** | 1.2.0 – 1.4.5<br>1.5.1+ *(latest)* | static GCM key (`data2=2`)<br>per-device AES-128 key (`data2=3`) — see [AES-128 Key](#aes-128-key-data23-g1--151--go2--1115--all-r1) |
| **R1** | 1.4.2 *(latest)* | per-device AES-128 key (`data2=3`) — see [AES-128 Key](#aes-128-key-data23-g1--151--go2--1115--all-r1) |

## Features

| Feature | Go2 | G1 | R1 |
|---------|:---:|:--:|:--:|
| Data channel (pub/sub, RPC) | yes | yes | yes |
| Sport / arm-action control | yes | yes | yes |
| FSM state control (`fsm_id`) | — | yes | yes |
| Video stream (receive) | yes | yes | yes |
| Audio stream (send/receive) | yes | — | — |
| LiDAR point cloud decoding | yes | — | — |
| VUI (LED, brightness, volume) | yes | — | — |
| AudioHub (audio file management) | yes | — | — |
| Obstacle avoidance API | yes | — | — |
| Multicast device discovery | yes | SN-targeted | SN-targeted |
| `data2=3` per-device key auth | yes (Go2 ≥ 1.1.15) | yes (G1 ≥ 1.5.1) | yes (always) |

"SN-targeted" discovery means the firmware only answers a query carrying its
own serial number, so it can confirm an address but not enumerate the network.

The humanoids (G1 and R1) share the Unitree Explorer account namespace, one
control channel and one topic set. Where they differ is the FSM state table:
R1's Run is `811`, not G1's `801`, and it carries its own dance / martial-arts
states. See `R1_FSM` in `constants.py`.

## Installation

### PyPI (recommended)

```sh
sudo apt update
sudo apt install -y python3-pip portaudio19-dev
pip install unitree_webrtc_connect
```

### From source

```sh
sudo apt update
sudo apt install -y python3-pip portaudio19-dev
git clone https://github.com/legion1581/unitree_webrtc_connect.git
cd unitree_webrtc_connect
pip install -e .
```

## Quick Start

```python
from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod

# Legacy firmware (Go2 < 1.1.15, G1 < 1.5.1) — no extra auth needed
conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.18")
await conn.connect()

# V3-capable firmware (G1 ≥ 1.5.1, Go2 ≥ 1.1.15) — per-device AES-128 key
# required for the LAN handshake (see "AES-128 Key" below for how to fetch it)
conn = UnitreeWebRTCConnection(
    WebRTCConnectionMethod.LocalSTA,
    ip="192.168.10.225",
    aes_128_key="<32-hex-chars>",
)
await conn.connect()
```

## Connection Methods

### AP Mode
Robot is in Access Point mode, client connects directly to the robot's WiFi.

```python
UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

# V3-capable firmware (G1 ≥ 1.5.1, Go2 ≥ 1.1.15) in AP mode
UnitreeWebRTCConnection(
    WebRTCConnectionMethod.LocalAP,
    aes_128_key="<32-hex-chars>",
)
```

### STA-L Mode (Local Network)
Robot and client are on the same local network. Requires IP or serial number.

```python
# By IP
UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")

# By serial number (uses multicast discovery). Pass device_type so the
# scanner queries the right multicast group — the humanoids don't announce
# on the Go2 group. Note that G1 >= 1.5.1 and every R1 ignore untargeted
# queries and reply only to one naming their own SN, so discovery finds
# them only when you already know the serial.
UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
UnitreeWebRTCConnection(
    WebRTCConnectionMethod.LocalSTA,
    serialNumber="E21D6000XXXXXXXX",
    device_type="R1",              # or "G1" — same multicast groups
    aes_128_key="<32-hex-chars>",
)

# V3-capable firmware (G1 ≥ 1.5.1, Go2 ≥ 1.1.15) by IP, with per-device key
UnitreeWebRTCConnection(
    WebRTCConnectionMethod.LocalSTA,
    ip="192.168.10.225",
    aes_128_key="<32-hex-chars>",
)
```

### STA-T Mode (Remote)
Remote connection through Unitree's TURN server. Control your robot from a different network. Requires Unitree account credentials.

`region` and `device_type` pick the cloud endpoint and the `AppName` header — `Go2` hits `global-robot-api.unitree.com` with `AppName: Go2`, `G1` does the same host with `AppName: G1`. Use `region="cn"` for accounts registered in China.

```python
# Go2 (default)
UnitreeWebRTCConnection(
    WebRTCConnectionMethod.Remote,
    serialNumber="B42D2000XXXXXXXX",
    username="email@gmail.com",
    password="pass",
)

# G1 in China region
UnitreeWebRTCConnection(
    WebRTCConnectionMethod.Remote,
    serialNumber="E21D6000XXXXXXXX",
    username="email@gmail.com",
    password="pass",
    region="cn",
    device_type="G1",
)
```

## AES-128 Key (`data2=3`, G1 ≥ 1.5.1 / Go2 ≥ 1.1.15 / all R1)

Starting with G1 firmware **1.5.1** (back-ported to Go2 firmware **1.1.15**), the LAN signaling handshake (`con_notify`) returns `data2=3`, which means the embedded RSA public key is wrapped under a **per-device AES-128-GCM key**. Without that key the WebRTC handshake can't decrypt the public key and the connection never starts. (Older firmware — G1 < 1.5.1 / Go2 < 1.1.15 — uses a static AES-GCM key, handled automatically.) **R1 shipped after the cutover, so every R1 needs a key.**

The key is **per device**, **stable across re-pairings**, stored on the robot at `/unitree/etc/key/aes_key.bin` (RSA-wrapped), and surfaced to the cloud as `dev.key` in `device/bind/list`.

### Fetch via the bundled CLI

After `pip install unitree_webrtc_connect` you get a console script:

```sh
# By default: region=global, device family=G1.
# Use --device-type Go2 for Unitree Go2 (≥ 1.1.15), or R1 for Unitree R1.
# G1 and R1 share one Unitree Explorer account, so either lists both.
unitree-fetch-aes-key --email you@example.com --password '...'

# Go2 in global region:
unitree-fetch-aes-key --email you@example.com --password '...' --device-type Go2

# China region:
unitree-fetch-aes-key --email you@example.com --password '...' --region cn

# Single SN, scriptable (bare key on stdout):
unitree-fetch-aes-key --email you@example.com --sn E21D6000XXXXXXXX --quiet

# Pre-existing access token (skip login):
unitree-fetch-aes-key --token <accessToken> --sn E21D6000XXXXXXXX
```

Equivalent if you don't have it on `$PATH`:
```sh
python -m unitree_webrtc_connect._cli --help
```

The interactive output is a labelled panel (SN / alias / online / region / key); `--quiet` swaps that for a bare key on stdout so you can pipe it:

```sh
KEY=$(unitree-fetch-aes-key --email ... --sn E21D... --quiet)
```

### Fetch programmatically

```python
from unitree_webrtc_connect import UnitreeCloud, fetch_aes_key

# One-shot lookup
key = fetch_aes_key("you@example.com", "...", sn="E21D6000XXXXXXXX",
                    region="global", device_type="G1")

# Or keep the cloud client around for other calls
cloud = UnitreeCloud(region="global", device_type="G1")
cloud.login_email("you@example.com", "...")
for d in cloud.list_devices():
    print(d.sn, d.alias, d.key)
```

### Typed errors

When you supply the wrong / missing key, the SDK raises typed exceptions you can catch instead of relying on stack traces:

```python
from unitree_webrtc_connect import (
    AesKeyRequiredError,    # data2=3 robot reached without a key
    AesKeyRejectedError,    # GCM tag check failed (wrong key)
    LocalSignalingPortError, # neither :9991 nor :8081 reachable on the IP
    RobotBusyError,          # robot rejected — another WebRTC client is connected
    NoSdpAnswerError,        # signaling round-trip returned no SDP
    DataChannelTimeoutError, # data channel didn't validate in time
)

try:
    await conn.connect()
except AesKeyRejectedError as e:
    ...
```

## Examples

Examples are organized by robot model under the `/examples` directory.

All examples default to `ip="192.168.8.181"` but read the `UNITREE_ROBOT_IP`
environment variable if set, so you can point them at your robot without
editing the source:

```bash
export UNITREE_ROBOT_IP=192.168.8.181
python examples/go2/data_channel/sportmode/sportmode.py
```

On V3-capable firmware (G1 ≥ 1.5.1, Go2 ≥ 1.1.15), also set
`UNITREE_AES_128_KEY` (fetch via `unitree-fetch-aes-key`).

### Go2

| Category | Example | Description |
|----------|---------|-------------|
| **Data Channel** | `data_channel/sportmode/` | Sport mode movement commands |
| | `data_channel/sportmode_mcf/` | Interactive MCF sport menu (firmware ≥ 1.1.7) |
| | `data_channel/sportmodestate/` | Subscribe to sport mode state |
| | `data_channel/lowstate/` | Subscribe to low-level state (IMU, motors) |
| | `data_channel/multiplestate/` | Subscribe to multiple state topics |
| | `data_channel/vui/` | VUI control (LED, volume, brightness) |
| | `data_channel/obstacles_avoid/` | Toggle obstacle avoidance + WASD drive |
| | `data_channel/lidar/lidar_stream.py` | LiDAR point cloud subscription |
| | `data_channel/lidar/plot_lidar_stream.py` | LiDAR 3D visualization (Three.js) |
| **Audio** | `audio/live_audio/` | Live audio receive |
| | `audio/save_audio/` | Save audio to file |
| | `audio/mp3_player/` | Play MP3 through robot speaker |
| | `audio/internet_radio/` | Stream internet radio |
| **Video** | `video/camera_stream/` | Display video stream |

### G1

| Category | Example | Description |
|----------|---------|-------------|
| **Data Channel** | `data_channel/sport_mode/` | Sport mode movement commands |
| **Video** | `video/camera_stream/` | Display video stream |

### R1

| Category | Example | Description |
|----------|---------|-------------|
| **Data Channel** | `data_channel/fsm_mode/` | Interactive FSM state menu — switch states, read `fsm_id` back, decode refusals |
| | `data_channel/arm_actions/` | Interactive upper-limb gesture menu |
| | `data_channel/teleop/` | Realtime WASD teleop — hold to drive, ramped, auto-stop on release |
| | `data_channel/lowstate/` | Live battery / motors / IMUs / board temperature |
| **Video** | `video/camera_stream/` | Display video stream |

## Imports

All public classes, helpers and exception types are exported from the package root:

```python
from unitree_webrtc_connect import (
    # Core driver
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
    WebRTCDataChannel,
    WebRTCDataChannelPubSub,
    DATA_CHANNEL_TYPE,
    RTC_TOPIC,
    SPORT_CMD,
    discover_ip_sn,

    # Humanoid (G1 / R1) control
    LOCO_API,
    LOCO_FSM_ERRORS,
    R1_FSM,
    ARM_ACTION,

    # Cloud helpers (data2=3, account / TURN flow, bind list)
    UnitreeCloud,
    UnitreeCloudError,
    RobotDevice,
    fetch_aes_key,

    # Typed errors
    AesKeyRequiredError,
    AesKeyRejectedError,
    DataChannelTimeoutError,
    LocalSignalingPortError,
    NoSdpAnswerError,
    RobotBusyError,
)
```

## Acknowledgements

A big thank you to RoboLegion community! Visit us at [RoboLegion](https://robolegion.com) for more information and support.

Special thanks to the [tfoldi WebRTC project](https://github.com/tfoldi/go2-webrtc) and [abizovnuralem](https://github.com/abizovnuralem) for adding LiDAR support, [MrRobotoW](https://github.com/MrRobotoW) for the LiDAR visualization example, and [Nico](https://github.com/oulianov) for the aiortc monkey patch.

## Support

If you like this project, please consider buying me a coffee:

<a href="https://www.buymeacoffee.com/legion1581" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
