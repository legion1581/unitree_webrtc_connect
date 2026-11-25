"""
Play local audio files through the robot speaker over the WebRTC audio
track, one after another on a single connection.

QueuedAudioTrack keeps the outgoing audio stream continuous (silence when
nothing is queued) and starts with a few seconds of silence: the robot drops
the first seconds of a stream that starts after an idle period, which used
to make the 2nd/3rd run of this example silent (#28).
"""
import asyncio
import logging
import os
import sys

from unitree_webrtc_connect import QueuedAudioTrack
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.8.181")
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY")  # needed on Go2 >= 1.1.15

HERE = os.path.dirname(__file__)
FILES = [
    os.path.join(HERE, "dog-barking.wav"),
    os.path.join(HERE, "dora-doradura-mp3.mp3"),
]


async def main():
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_128_KEY)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()

    player = QueuedAudioTrack()  # default 9 s silent warm-up before the first file
    conn.pc.addTrack(player)
    try:
        for path in FILES:
            print(f"Playing {os.path.basename(path)}")
            await player.play(path)  # returns once the whole file was sent
        await asyncio.sleep(1)       # let the tail reach the speaker
    finally:
        await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
