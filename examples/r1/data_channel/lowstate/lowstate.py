"""R1 low-level state — battery, motors, IMUs, board temperature.

The humanoids split what Go2 packs into one message across four topics:

    rt/lf/lowstate        motor_state[], imu_state (this is the TORSO imu)
    rt/lf/bmsstate        the bms struct on its own topic, not nested
    rt/lf/secondary_imu   the pelvis imu
    rt/lf/mainboardstate  body/chassis temperature (Go2 uses temperature_ntc1)

Two shape differences from Go2 worth knowing:

  * Per-motor `temperature` is a two-element array [casing, winding], not a
    scalar. Both are unsigned bytes carrying signed values, so 250 means
    -6 C, not 250 C — see sign_byte(). The winding figure is the one the
    official app shows as "motor temperature".
  * The motor array has 29 slots (the G1 joint order). R1 is 20-DOF and
    fills a subset: legs 0-11, left arm 15-18, right arm 22-25. The waist
    and wrist slots stay zeroed, so this example prints only the slots R1
    actually drives.

Battery voltages arrive in millivolts: `bmsvoltage` is [pack_mV, bat_mV, _]
and `cell_vol` is per-cell mV padded out to 40 entries.
"""

import asyncio
import logging
import os
import sys

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import (
    RTC_TOPIC, MOTOR_NAMES_HUMANOID, R1_MOTOR_INDICES, sign_byte,
)

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.12.1")
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY")

# Latest payload per topic; the display redraws from whatever has arrived.
state = {"low": None, "bms": None, "imu2": None, "board": None}


def fmt(value, spec="7.3f", missing="      —"):
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return missing


def motor_rows(low):
    motors = low.get("motor_state") or []
    for idx in R1_MOTOR_INDICES:
        if idx >= len(motors):
            continue
        m = motors[idx]
        temp = m.get("temperature")
        if isinstance(temp, (list, tuple)):
            casing = sign_byte(temp[0]) if len(temp) > 0 else None
            winding = sign_byte(temp[1]) if len(temp) > 1 else None
        else:
            casing, winding = sign_byte(temp), None
        yield idx, MOTOR_NAMES_HUMANOID[idx], m, casing, winding


def render():
    low, bms, imu2, board = state["low"], state["bms"], state["imu2"], state["board"]
    out = ["\033[H\033[J", "R1 Low-Level State", "=" * 78, ""]

    # ── Battery ──
    out.append("Battery")
    if bms:
        volt = bms.get("bmsvoltage") or []
        pack = bms.get("pack_voltage", volt[0] if len(volt) > 0 else None)
        bat = bms.get("bat_voltage", volt[1] if len(volt) > 1 else None)
        cells = [c for c in (bms.get("cell_vol") or []) if isinstance(c, (int, float)) and c > 0]
        temps = [t for t in (bms.get("temperature") or bms.get("temps") or [])
                 if isinstance(t, (int, float))]
        out.append(f"  SOC        : {bms.get('soc', '—')}%")
        out.append(f"  Current    : {bms.get('current', '—')} mA")
        out.append(f"  Cycles     : {bms.get('cycle', '—')}")
        out.append(f"  Pack       : {fmt(pack and pack / 1000, '6.2f')} V"
                   f"   Battery: {fmt(bat and bat / 1000, '6.2f')} V")
        if cells:
            out.append(f"  Cells      : {len(cells)} live, "
                       f"min {min(cells)} mV / max {max(cells)} mV "
                       f"(spread {max(cells) - min(cells)} mV)")
        if temps:
            # [MOS, _, BAT1, RES, ...] per the app's battery view model.
            out.append(f"  Temps      : MOS {temps[0]}C"
                       + (f"  BAT {temps[2]}C" if len(temps) > 2 else "")
                       + f"   (raw {temps})")
    else:
        out.append("  waiting for rt/lf/bmsstate ...")
    out.append("")

    # ── IMUs ──
    out.append("IMU")
    torso = (low or {}).get("imu_state") or {}
    for label, imu in (("Torso ", torso), ("Pelvis", imu2 or {})):
        rpy = imu.get("rpy") or []
        if len(rpy) >= 3:
            out.append(f"  {label}     : roll {fmt(rpy[0], '7.3f')}  "
                       f"pitch {fmt(rpy[1], '7.3f')}  yaw {fmt(rpy[2], '7.3f')}"
                       f"   {imu.get('temperature', '—')}C")
        else:
            out.append(f"  {label}     : —")
    if board:
        temps = board.get("temperature") or []
        if temps:
            out.append(f"  Mainboard  : {sign_byte(temps[0])}C   (raw {temps})")
    out.append("")

    # ── Motors ──
    out.append("Motors (20 DOF — the slots R1 drives out of the 29-slot array)")
    out.append(f"  {'#':>3}  {'joint':<9} {'q (rad)':>9} {'dq':>8} {'tau':>8}"
               f" {'case':>5} {'wind':>5} {'lost':>5}")
    out.append("  " + "-" * 62)
    if low:
        for idx, name, m, casing, winding in motor_rows(low):
            out.append(
                f"  {idx:>3}  {name:<9} {fmt(m.get('q'), '9.4f')}"
                f" {fmt(m.get('dq'), '8.3f')} {fmt(m.get('tau_est'), '8.3f')}"
                f" {str(casing) if casing is not None else '—':>5}"
                f" {str(winding) if winding is not None else '—':>5}"
                f" {m.get('lost', '—'):>5}"
            )
    else:
        out.append("  waiting for rt/lf/lowstate ...")

    out.append("")
    out.append("Ctrl-C to exit")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


async def main():
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_128_KEY,
    )
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=AES_128_KEY)
    await conn.connect()

    def store(key):
        def callback(message):
            state[key] = message.get("data")
        return callback

    pub_sub = conn.datachannel.pub_sub
    pub_sub.subscribe(RTC_TOPIC["LOW_STATE"], store("low"))
    pub_sub.subscribe(RTC_TOPIC["BMS_STATE"], store("bms"))
    pub_sub.subscribe(RTC_TOPIC["SECONDARY_IMU"], store("imu2"))
    pub_sub.subscribe(RTC_TOPIC["MAIN_BOARD_STATE"], store("board"))

    # Redraw on a timer rather than per message: four topics at different
    # rates would otherwise fight over the screen.
    while True:
        render()
        await asyncio.sleep(0.2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
