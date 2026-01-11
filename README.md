# Unitree Go2 and G1 WebRTC Driver

This repository contains a Python implementation of the WebRTC driver to connect to the Unitree Go2 and G1 Robots. WebRTC is used by the Unitree Go/Unitree Explore APP and provides high-level control through it. Therefore, no jailbreak or firmware manipulation is required. It works out of the box for Go2 AIR/PRO/EDU models and G1 AIR/EDU.

![Description of the image](https://github.com/legion1581/unitree_webrtc_connect/raw/master/images/screenshot_1.png)


[![PyPI](https://img.shields.io/pypi/v/unitree-webrtc-connect.svg)](https://pypi.org/project/unitree-webrtc-connect/)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/legion1581/unitree_webrtc_connect)](https://github.com/legion1581/unitree_webrtc_connect/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Currently Supported Firmware Versions

**Go2:**
- **1.1.x series**: 1.1.1 – 1.1.11 *(latest available)*
- **1.0.x series**: 1.0.19 – 1.0.25

**G1:**
- **1.4.0** *(latest available)*

## Audio and Video Support (Go2 only)

There are video (recvonly) and audio (sendrecv) channels in WebRTC that you can connect to. Check out the examples in the `/example` folder.

## Lidar support (Go2 only)

There is a lidar decoder built in, so you can handle decoded PoinClouds directly. Check out the examples in the `/example` folder.

## Connection Methods

The driver supports three types of connection methods:

1. **AP Mode**: Go2/G1 is in AP mode, and the WebRTC client is connected directly to it:

    ```python
    UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
    ```

2. **STA-L Mode**: Go2/G1 and the WebRTC client are on the same local network. An IP or Serial number is required:

    ```python
    UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    ```

    Alternatively, you can set the environement variable `ROBOT_IP`, for example

    ```bash
    export ROBOT_IP="192.168.8.181"
    ```

    If the IP is unknown, you can specify only the serial number, and the driver will try to find the IP using the special Multicast discovery feature available on Go2:

    ```python
    UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    ```

3. **STA-T mode**: Remote connection through remote Unitrees TURN server. Could control your Go2/G1 even being on the diffrent network. Requires username and pass from Unitree account

    ```python
    UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    ```

## Multicast scanner (Go2 only)
The driver has a built-in Multicast scanner to find the Unitree Go2 on the local network and connect using only the serial number.

## PIP installation (Recommended)

```sh
cd ~
sudo apt update
sudo apt install -y python3-pip portaudio19-dev
pip install unitree_webrtc_connect
```


## Manual Installation 

```sh
cd ~
sudo apt update
sudo apt install -y python3-pip portaudio19-dev
pip install --upgrade setuptools pip
git clone https://github.com/legion1581/unitree_webrtc_connect.git
cd unitree_webrtc_connect
pip install -e .
```

## Usage 
Example programs are located in the /example directory. Set the IP address of the robot using the environment variable `ROBOT_IP` or modify the example.

### Thanks

A big thank you to TheRoboVerse community! Visit us at [TheRoboVerse](https://theroboverse.com) for more information and support.

Special thanks to the [tfoldi WebRTC project](https://github.com/tfoldi/go2-webrtc) and [abizovnuralem](https://github.com/abizovnuralem) for adding LiDAR support, [MrRobotow](https://github.com/MrRobotoW) for providing a plot LiDAR example. Special thanks to [Nico](https://github.com/oulianov) for the aiortc monkey patch.

 
### Support

If you like this project, please consider buying me a coffee:

<a href="https://www.buymeacoffee.com/legion1581" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
