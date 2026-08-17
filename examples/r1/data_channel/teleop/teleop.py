"""R1 realtime teleop — hold WASD to drive, release to stop.

Unlike a burst-per-keypress example, this publishes continuously at a fixed
rate and derives the joystick state from which keys are currently held. The
terminal has no key-up event, so "held" is inferred from auto-repeat: a key
keeps its axis alive for KEY_HOLD_TIMEOUT after the last repeat, and the
axis falls back to zero once the repeats stop.

That has one consequence worth knowing: most terminals wait ~0.5 s before
they start repeating a held key, so the very first moment of a press is a
single event. RAMP_UP smooths the start so the robot eases in rather than
lurching, and holding past the initial delay gives smooth continuous motion.

Axis mapping matches the on-robot `update_state()`, which reads the
wirelesscontroller message as [ly, -lx, -rx] -> [vx, vy, omega]:

    ly  +forward      lx  +right strafe      rx  +clockwise

Note the robot scales these normalised axes by its own limit (1.0 m/s
forward by default), so 1.0 here is full speed as the robot defines it.
"""

import asyncio
import logging
import os
import select
import sys
import termios
import time
import tty

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, LOCO_API, R1_FSM

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY")

PUBLISH_HZ = 20.0
TICK = 1.0 / PUBLISH_HZ
# How long an axis stays alive after the last keypress. Long enough to ride
# out the gaps between auto-repeats, short enough to stop promptly on
# release. Raise it if your key-repeat rate is unusually slow.
KEY_HOLD_TIMEOUT = 0.35
# Per-tick change toward the target, so a press eases in and a release eases
# out instead of stepping. 0.15 at 20 Hz reaches full scale in ~0.35 s.
RAMP_UP = 0.15
RAMP_DOWN = 0.30

#          key -> (axis, direction)
KEY_AXES = {
    "w": ("ly", 1.0), "\x1b[A": ("ly", 1.0),
    "s": ("ly", -1.0), "\x1b[B": ("ly", -1.0),
    "a": ("lx", -1.0), "\x1b[D": ("lx", -1.0),
    "d": ("lx", 1.0), "\x1b[C": ("lx", 1.0),
    "q": ("rx", -1.0),
    "e": ("rx", 1.0),
}


class KeyReader:
    """Non-blocking stdin in cbreak mode. cbreak rather than raw so Ctrl-C
    still interrupts — this drives a real robot."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read_keys(self):
        """Return every key pressed since the last call. Never blocks."""
        if not select.select([sys.stdin], [], [], 0)[0]:
            return []
        buf = os.read(self.fd, 1024).decode("utf-8", "ignore")
        keys, i = [], 0
        while i < len(buf):
            if buf[i] == "\x1b" and i + 2 < len(buf) and buf[i + 1] == "[":
                keys.append(buf[i:i + 3])
                i += 3
            else:
                keys.append(buf[i])
                i += 1
        return keys


def publish(pub_sub, lx=0.0, ly=0.0, rx=0.0, ry=0.0):
    pub_sub.publish_without_callback(
        RTC_TOPIC["WIRELESS_CONTROLLER"],
        {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "keys": 0},
    )


async def set_fsm(conn, name):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": LOCO_API["SET_FSM_ID"], "parameter": {"data": R1_FSM[name]}},
    )
    return response.get("data", {}).get("header", {}).get("status", {}).get("code", -1)


def approach(current, target, up, down):
    """Step `current` toward `target`, faster when returning to zero."""
    rate = up if abs(target) > abs(current) else down
    if current < target:
        return min(current + rate, target)
    if current > target:
        return max(current - rate, target)
    return current


current_fsm_id = None


def on_sportmodestate(message):
    global current_fsm_id
    fsm_id = message.get("data", {}).get("fsm_id")
    if fsm_id is not None:
        current_fsm_id = fsm_id


def fsm_name(fsm_id):
    return next((k for k, v in R1_FSM.items() if v == fsm_id), "?")


HELP = """
  Realtime teleop
  ---------------
    W / S  or  Up / Down     forward / backward
    A / D  or  Left / Right  strafe left / right
    Q / E                    turn left / right
    + / -                    speed scale
    1                        Lock  (fsm 4)
    2                        Run   (fsm 811) — required before moving
    SPACE                    stop moving
    X                        Damping (fsm 1) — the stop, from any state
    Ctrl-C                   quit (stops moving, leaves the robot standing)

  Hold a key to keep moving. Diagonals need both keys tapped alternately:
  a terminal only auto-repeats the most recently held key.
"""


async def main():
    print("WARNING: The robot will walk. Ensure there is clear space around it.")
    print("X damps at any time. Ctrl-C just exits — it stops the robot")
    print("moving but leaves it standing, so quitting is never a drop.")
    await asyncio.to_thread(input, "Press Enter to continue...")

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_128_KEY,
    )
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=AES_128_KEY)
    await conn.connect()

    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["LF_SPORT_MOD_STATE"], on_sportmodestate,
    )
    pub_sub = conn.datachannel.pub_sub

    print(HELP)
    print("  Press 1 (Lock) then 2 (Run), wait for it to stand, then drive.\n")

    axes = {"lx": 0.0, "ly": 0.0, "rx": 0.0}   # smoothed, what we publish
    held = {}                                   # axis -> (direction, last seen)
    scale = 0.6
    pending = []                                # fsm requests to run off-tick

    try:
        with KeyReader() as reader:
            while True:
                now = time.monotonic()

                for key in reader.read_keys():
                    lowered = key.lower() if len(key) == 1 else key
                    if lowered in KEY_AXES:
                        axis, direction = KEY_AXES[lowered]
                        held[axis] = (direction, now)
                    elif lowered == " ":
                        held.clear()
                    elif lowered == "x":
                        held.clear()
                        pending.append("Damping")
                    elif lowered == "1":
                        pending.append("Lock")
                    elif lowered == "2":
                        pending.append("Run")
                    elif key in ("+", "="):
                        scale = min(1.0, round(scale + 0.1, 2))
                    elif key in ("-", "_"):
                        scale = max(0.1, round(scale - 0.1, 2))

                # Drop axes whose key stopped repeating.
                held = {a: v for a, v in held.items()
                        if now - v[1] < KEY_HOLD_TIMEOUT}

                for axis in axes:
                    direction = held[axis][0] if axis in held else 0.0
                    axes[axis] = approach(
                        axes[axis], direction * scale, RAMP_UP, RAMP_DOWN,
                    )

                publish(pub_sub, lx=axes["lx"], ly=axes["ly"], rx=axes["rx"])

                state = f"{current_fsm_id} ({fsm_name(current_fsm_id)})" \
                    if current_fsm_id is not None else "unreported"
                sys.stdout.write(
                    f"\r  state={state:<22} scale={scale:.1f}  "
                    f"ly={axes['ly']:+.2f} lx={axes['lx']:+.2f} rx={axes['rx']:+.2f}   "
                )
                sys.stdout.flush()

                # FSM switches are request/response, so run them between
                # ticks rather than stalling the publish loop.
                while pending:
                    name = pending.pop(0)
                    code = await set_fsm(conn, name)
                    sys.stdout.write(f"\n  -> {name} ({R1_FSM[name]}) code={code}\n")
                    if code and name == "Run":
                        sys.stdout.write("     Run needs Lock first — press 1.\n")
                    sys.stdout.flush()

                await asyncio.sleep(TICK)

    except KeyboardInterrupt:
        pass
    finally:
        # Zero the sticks so it stops walking — but do NOT damp. The robot
        # is standing at this point and damping would drop it. Leaving the
        # state alone means quitting the script is safe at any moment; use
        # X deliberately when you actually want it on the ground.
        for _ in range(3):
            publish(pub_sub)
            await asyncio.sleep(0.05)
        print("\n  Stopped moving. Robot left in its current state.")
        await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting")
        sys.exit(0)
