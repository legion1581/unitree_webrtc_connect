"""R1 FSM modes — every state switchable from an interactive menu.

Everything persistent on R1 is a single FSM state selected through one api:
SET_FSM_ID (7101) on `rt/api/sport/request`, parameter {"data": <fsm_id>}.
The robot echoes the state it is actually in as `fsm_id` on
`rt/lf/sportmodestate`, which this example subscribes to and prints.

Three things that surprise people:

  * Reachability is enforced per state on the robot. Most motions are only
    reachable from Lock (4) — not from Run. A refused switch answers 1001.
    Damping (1) is accepted from every state, which makes it the stop.
  * On an AIR chassis a request for Run (811) is redirected on-robot to
    Locomotion20Dofs (830), so the state reported back differs from the
    one you asked for. That's expected, not an error.
  * The robot stops publishing sportmodestate entirely while in a state
    that isn't on its report white-list (5, 6, 7, 800, 813, 814 among
    others), so the live readout freezes rather than updating. Ask for
    Damping to get it talking again.
"""

import asyncio
import logging
import os
import sys

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, LOCO_API, R1_FSM, LOCO_FSM_ERRORS

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")
# R1 always speaks data2=3, so the per-device key is required on the LAN.
# Fetch with `unitree-fetch-aes-key --device-type R1`.
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY")

# Menu rows: (state key, "<fsm_id>  <description>"). Damping is first so the
# stop is always index 0.
MENU = [
    ("Damping",           "  1  Soft stop — accepted from every state"),
    ("Lock",              "  4  Stance / 'Lock' in the app. The hub: switch here first"),
    ("ZeroTorque",        "  0  Motors free (only from Damping)"),
    ("Run",               "811  Locomotion. AIR chassis redirects this to 830"),
    ("StandUp",           "701  Get up off the ground (face-up or face-down)"),
    ("LieDown",           "702  Lie down"),
    ("SitDown",           "  7  Sit down"),
    ("Dance1",            "601  Dance 1"),
    ("Dance2",            "602  Dance 2"),
    ("Dance3",            "603  Dance 3"),
    ("Twist",             "604  Twist (niuniuwu)"),
    ("KungFu",            "607  Kung Fu"),
    ("JeetKuneDo",        "608  Jeet Kune Do"),
    ("Keep",              "  5  Hold the current pose"),
    ("MoveTo",            "  6  MoveTo"),
    ("Motion",            "800  Motion super-state"),
    ("AmpMotion",         "812  AMP motion, 24-dof"),
    ("WalkStraightKnee",  "813  Walk, straight knee"),
    ("Walk",              "814  Walk, 24-dof"),
    ("AmpLocomotion",     "815  AMP locomotion"),
    ("ArmSdkLoco",        "816  Locomotion + arm-SDK"),
    ("Loco20Dof",         "830  20-dof locomotion"),
    ("LocoArmSdk",        "831  Locomotion + arm-SDK"),
]

# Not every state is built on every firmware — asking for one that isn't
# answers 1003 rather than doing nothing, so the menu can stay complete.

current_fsm_id = None


def on_sportmodestate(message):
    global current_fsm_id
    fsm_id = message.get("data", {}).get("fsm_id")
    if fsm_id is not None and fsm_id != current_fsm_id:
        current_fsm_id = fsm_id
        print(f"  [robot] now in fsm_id={fsm_id} ({fsm_name(fsm_id)})")


def fsm_name(fsm_id):
    return next((k for k, v in R1_FSM.items() if v == fsm_id), "unknown")


async def set_fsm(conn, name):
    """Request a state by name. Returns the robot's status code."""
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": LOCO_API["SET_FSM_ID"], "parameter": {"data": R1_FSM[name]}},
    )
    return response.get("data", {}).get("header", {}).get("status", {}).get("code", -1)


async def main():
    print("WARNING: Ensure there is clear space around the robot.")
    print("Menu item 0 (Damping) is the stop — it is accepted from every state.")
    await asyncio.to_thread(input, "Press Enter to continue...")

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_128_KEY,
    )
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=AES_128_KEY)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA,
    #                                serialNumber="E39N2000XXXXXXXX",
    #                                aes_128_key=AES_128_KEY, device_type="R1")
    await conn.connect()

    # Watch the reported state so every switch below is confirmed.
    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LF_SPORT_MOD_STATE"], on_sportmodestate,
    )
    await asyncio.sleep(1)

    print("\nAvailable states:")
    for i, (name, desc) in enumerate(MENU):
        print(f"  {i:2d}: {name:18s}  {desc}")
    print("   s: Show the last reported state")
    print("   q: Quit")

    while True:
        raw = (await asyncio.to_thread(input, "\nState #: ")).strip()
        if raw.lower() == "q":
            break
        if raw.lower() == "s":
            if current_fsm_id is None:
                print("  Nothing reported yet — the robot goes quiet in "
                      "states that aren't on its report white-list.")
            else:
                print(f"  fsm_id={current_fsm_id} ({fsm_name(current_fsm_id)})")
            continue
        try:
            idx = int(raw)
        except ValueError:
            print("Invalid input")
            continue
        if not (0 <= idx < len(MENU)):
            print("Unknown state")
            continue

        name = MENU[idx][0]
        try:
            code = await set_fsm(conn, name)
        except Exception as e:
            print(f"  -> {name} failed: {e}")
            continue

        if code == 0:
            print(f"  -> {name} ({R1_FSM[name]}) accepted")
        else:
            print(f"  -> {name} ({R1_FSM[name]}) refused: code={code}"
                  f" — {LOCO_FSM_ERRORS.get(code, 'unknown error')}")
            if code in (1001, 1002):
                print("     HINT: most states are only reachable from Lock (4). "
                      "Go there first, then retry.")
        await asyncio.sleep(0.5)

    await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting")
        sys.exit(0)
