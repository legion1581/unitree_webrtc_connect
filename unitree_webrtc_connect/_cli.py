"""
CLI for fetching the per-device AES-128 key (`dev.key` from the Unitree
cloud, required for the LAN data2=3 handshake on R1, G1 firmware ≥ 1.5.1
and Go2 firmware ≥ 1.1.15).

Registered as the `unitree-fetch-aes-key` console script in pyproject
once the package is pip-installed. Also runnable as
`python -m unitree_webrtc_connect._cli` or via the thin wrapper at
`examples/fetch_aes_key.py`.

Status messages go to stderr so stdout stays parsable. With `--sn`,
stdout is the bare key (or empty + non-zero exit) — good for shell
substitution: `KEY=$(unitree-fetch-aes-key --sn ... --email ...)`.
Without `--sn`, stdout is a human-readable table.
"""

import argparse
import getpass
import sys
import time

from .unitree_cloud import APP_NAME, UnitreeCloud, UnitreeCloudError


# ─── tiny stderr logger so we don't drag in the logging config ────────

def _step(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"  ▸ [{ts}] {msg}", file=sys.stderr, flush=True)


def _ok(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"  ✓ [{ts}] {msg}", file=sys.stderr, flush=True)


def _fail(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"  ✗ [{ts}] {msg}", file=sys.stderr, flush=True)


# ─── output helpers ───────────────────────────────────────────────────

def _print_key_box(device, region: str, device_type: str) -> None:
    """Pretty single-key panel for interactive use. Goes to stdout."""
    online = "?" if device.online is None else ("online" if device.online else "offline")
    alias = device.alias or "(no alias)"
    width = max(len(device.sn), len(alias), len(device.key), 56) + 2

    line = "─" * width
    print(f"\n┌{line}┐")
    print(f"│ {'AES-128 Key (data2=3)'.ljust(width - 1)}│")
    print(f"├{line}┤")
    print(f"│ {('SN     : ' + device.sn).ljust(width - 1)}│")
    print(f"│ {('Alias  : ' + alias).ljust(width - 1)}│")
    print(f"│ {('Status : ' + online).ljust(width - 1)}│")
    label = f"Region : {region}  Family : {device_type}  AppName : {APP_NAME[device_type]}"
    print(f"│ {label.ljust(width - 1)}│")
    print(f"│ {('Key    : ' + device.key).ljust(width - 1)}│")
    print(f"└{line}┘")
    print(
        "\n  Copy the key into UnitreeWebRTCConnection(..., aes_128_key=...)",
        file=sys.stderr,
    )


def _print_table(devices, region: str, device_type: str) -> None:
    """All-bound-devices table. Goes to stdout."""
    print(
        f"\nDevices bound to this account ({len(devices)} total) — "
        f"region={region}, family={device_type}, AppName={APP_NAME[device_type]}\n",
        file=sys.stderr,
    )
    sn_w = max((len(d.sn) for d in devices), default=2)
    al_w = max((len(d.alias or "(none)") for d in devices), default=5)
    print(f"{'SN':<{sn_w}}  {'alias':<{al_w}}  {'online':<8}  {'AES-128 key (data2=3)'}")
    print(f"{'-' * sn_w}  {'-' * al_w}  {'-' * 8}  {'-' * 32}")
    for d in devices:
        online = "?" if d.online is None else ("yes" if d.online else "no")
        alias = d.alias or "(none)"
        key = d.key or "(empty)"
        print(f"{d.sn:<{sn_w}}  {alias:<{al_w}}  {online:<8}  {key}")
    print()


# ─── arg parsing ──────────────────────────────────────────────────────

def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="unitree-fetch-aes-key",
        description=(
            "Fetch the per-device AES-128 key (data2=3, required on "
            "G1 ≥ 1.5.1, Go2 ≥ 1.1.15 and every R1) from the Unitree cloud."
        ),
        epilog=(
            "Examples:\n"
            "  unitree-fetch-aes-key --email you@example.com --password ...\n"
            "  unitree-fetch-aes-key --token <accessToken> --region cn --device-type G1\n"
            "  unitree-fetch-aes-key --email ... --device-type R1\n"
            "  unitree-fetch-aes-key --email ... --sn B42D2000XXXXXXXX --quiet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--email", help="account email (omit to use --token)")
    p.add_argument("--password",
                   help="account password (interactive prompt if missing)")
    p.add_argument("--token", help="pre-existing access token (skips login)")
    p.add_argument("--region", choices=["global", "cn"], default="global",
                   help="cloud region (default: global)")
    p.add_argument("--device-type", choices=["Go2", "G1", "R1"], default="G1",
                   help="robot family to sign requests as (default: G1). "
                        "G1 and R1 share one Unitree Explorer account, so "
                        "either lists both")
    p.add_argument("--sn", help="serial number to look up "
                                "(if omitted, prints every bound device)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="with --sn, print only the bare key on stdout "
                        "(suitable for shell substitution)")
    return p.parse_args(argv)


# ─── main ─────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = _parse_args(argv)

    if not args.token and not args.email:
        print("error: pass either --token or --email/--password", file=sys.stderr)
        return 2

    _ok(f"Using region={args.region}, device family={args.device_type}.")

    cloud = UnitreeCloud(
        region=args.region,
        device_type=args.device_type,
        access_token=args.token or "",
    )

    if args.email:
        password = args.password or getpass.getpass(f"Password for {args.email}: ")
        _step(f"Logging in to {args.region} cloud as {args.email}…")
        try:
            cloud.login_email(args.email, password)
            _ok("Login OK, access token cached.")
        except UnitreeCloudError as e:
            _fail(str(e))
            return 1
    else:
        _ok("Using --token; skipping login.")

    _step("Fetching device list (device/bind/list)…")
    try:
        devices = cloud.list_devices()
    except UnitreeCloudError as e:
        _fail(str(e))
        return 1

    if not devices:
        _ok("Cloud returned 0 devices.")
        print("(no robots bound to this account)")
        return 0

    _ok(f"Cloud returned {len(devices)} device(s).")

    # Single-SN lookup ────────────────────────────────────────────────
    if args.sn:
        for d in devices:
            if d.sn == args.sn:
                if not d.key:
                    _fail(
                        f"SN {d.sn} is bound but `dev.key` is empty — "
                        f"firmware is probably below the data2=3 cutover "
                        f"(G1 < 1.5.1 / Go2 < 1.1.15), in which case no "
                        f"per-device key is needed. R1 is always data2=3."
                    )
                    return 1
                if args.quiet:
                    print(d.key)
                else:
                    _print_key_box(d, args.region, args.device_type)
                return 0
        _fail(
            f"SN {args.sn!r} is not bound to this account "
            f"(region={args.region!r}). Did you mean a different region, "
            f"or pair the robot via the apk first?"
        )
        return 1

    # Full table ──────────────────────────────────────────────────────
    _print_table(devices, args.region, args.device_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
