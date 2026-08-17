"""R1 arm gestures — every upper-limb action from an interactive menu.

Gestures ride their own topic (`rt/api/arm/request`) and api id
(SET_ARM_TASK = 7106), separate from the FSM state channel. The parameter
is {"data": <action id>}; see ARM_ACTION in constants.py for the ids.

They are independent of the locomotion state, so a gesture plays while the
robot stands in Lock. `Release` (action id 99) cancels whatever is running
and brings the arms back to neutral — run it between gestures.
"""

import asyncio
import logging
import os
import sys

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, LOCO_API, ARM_ACTION

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY")

# The gestures R1 ships, in the order its app lists them. G1 additionally
# has ArmHeart (20) and RightHeart (21); R1's app doesn't expose them.
# Release is first so returning to neutral is always index 0.
MENU = [
    ("Release",      "99  Cancel the current gesture, arms back to neutral"),
    ("Handshake",    "27  Offer a hand to shake"),
    ("HighFive",     "18  High five"),
    ("Hug",          "19  Hug"),
    ("HighWave",     "26  Wave, arm high"),
    ("Clap",         "17  Clap"),
    ("FaceWave",     "25  Wave in front of the face"),
    ("LeftKiss",     "12  Blow a kiss, left hand"),
    ("XRay",         "24  Ultraman ray pose"),
    ("HandsUp",      "15  Both hands up"),
    ("RightHandUp",  "23  Right hand up"),
    ("Reject",       "22  Refuse gesture"),
    ("ForwardPush",  "36  Push forward"),
]


async def play(conn, name):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["ARM_REQUEST"],
        {"api_id": LOCO_API["SET_ARM_TASK"], "parameter": {"data": ARM_ACTION[name]}},
    )
    return response.get("data", {}).get("header", {}).get("status", {}).get("code", -1)


async def main():
    print("WARNING: Ensure there is clear space around the robot's arms.")
    await asyncio.to_thread(input, "Press Enter to continue...")

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_128_KEY,
    )
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=AES_128_KEY)
    await conn.connect()

    print("\nAvailable gestures:")
    for i, (name, desc) in enumerate(MENU):
        print(f"  {i:2d}: {name:14s}  {desc}")
    print("   q: Quit")

    while True:
        raw = (await asyncio.to_thread(input, "\nGesture #: ")).strip()
        if raw.lower() == "q":
            break
        try:
            idx = int(raw)
        except ValueError:
            print("Invalid input")
            continue
        if not (0 <= idx < len(MENU)):
            print("Unknown gesture")
            continue

        name = MENU[idx][0]
        try:
            code = await play(conn, name)
        except Exception as e:
            print(f"  -> {name} failed: {e}")
            continue
        if code == 0:
            print(f"  -> {name} ({ARM_ACTION[name]}) accepted")
        else:
            print(f"  -> {name} ({ARM_ACTION[name]}) returned code={code}")
        await asyncio.sleep(0.5)

    await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting")
        sys.exit(0)
