"""UDP multicast discovery — maps serial numbers to IP addresses.

Two things decide whether a robot answers:

1. **The multicast group.** Go2 announces on `231.1.1.1`; the Explorer-line
   humanoids (G1 and R1) use `231.1.1.2` and `239.255.1.1`. Querying the
   wrong group is why SN discovery only ever worked for Go2.

2. **Whether the query names it.** V3-capable firmware — G1 >= 1.5.1 and
   *every* R1 — silently ignores an untargeted query. It replies only when
   the query carries its own `sn`. Older firmware answers either form. So
   for those robots `sn` is not an optimisation, it is required, and you
   have to know the serial before you can find the address.

Replies all arrive on port 10134 regardless of family, so a scan can pick up
robots outside the family you asked for. That's harmless: the caller matches
on the serial it wants. We deliberately do NOT filter by SN prefix, because
the prefix is not a reliable family signal.
"""

import socket
import struct
import json
import logging

RECV_PORT = 10134       # Port where the devices send their multicast replies
MULTICAST_PORT = 10131  # Port to send the multicast query to

# Multicast groups to query, per robot family.
FAMILY_GROUPS = {
    "Go2": ("231.1.1.1",),
    "G1":  ("231.1.1.2", "239.255.1.1"),
    "R1":  ("231.1.1.2", "239.255.1.1"),
}

# Kept for backwards compatibility with code that imported the old constant.
MULTICAST_GROUP = FAMILY_GROUPS["Go2"][0]

_QUERY_NAME = "unitree_dapengche"
_QUERY_BURST = 3        # Datagrams are lossy — repeat the query a few times.
_QUERY_INTERVAL = 0.2


def discover_ip_sn(timeout=2, device_type="Go2", sn=None):
    """Discover robots on the LAN, returning `{serial_number: ip_address}`.

    :param timeout: seconds to listen for replies.
    :param device_type: which family's multicast groups to query
        (`"Go2"`, `"G1"` or `"R1"`). Unknown values fall back to Go2.
    :param sn: serial number to target. **Required to find G1 >= 1.5.1 or
        any R1** — that firmware ignores untargeted queries entirely. When
        given, both an untargeted and a targeted query go out, so one call
        covers old and new firmware alike. Replies are NOT filtered by it:
        the caller still sees every robot that answered, so it can tell
        "nothing on the network" apart from "that serial isn't here".
    """
    print("Discovering devices on the network...")

    groups = FAMILY_GROUPS.get(device_type, FAMILY_GROUPS["Go2"])
    serial_to_ip = {}

    # Send both forms when we have an SN: the untargeted one finds legacy
    # robots, the targeted one is the only thing V3 firmware replies to.
    queries = [json.dumps({"name": _QUERY_NAME}).encode("utf-8")]
    if sn:
        queries.append(json.dumps({"name": _QUERY_NAME, "sn": sn}).encode("utf-8"))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", RECV_PORT))

    # Join every group for this family on all interfaces.
    for group in groups:
        try:
            mreq = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError as e:
            logging.warning(f"Could not join multicast group {group}: {e}")

    # Send the query burst, then listen. The socket is shared for send and
    # receive, so a short per-send timeout keeps the burst from blocking.
    sock.settimeout(_QUERY_INTERVAL)
    for attempt in range(_QUERY_BURST):
        for group in groups:
            for query_message in queries:
                try:
                    sock.sendto(query_message, (group, MULTICAST_PORT))
                except Exception as e:
                    logging.warning(f"Error sending multicast query to {group}: {e}")
        if attempt < _QUERY_BURST - 1:
            # Drain anything that answers between bursts instead of sleeping.
            _collect(sock, serial_to_ip)

    sock.settimeout(timeout)
    _collect(sock, serial_to_ip)

    for group in groups:
        try:
            mreq = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
        except OSError:
            pass
    sock.close()

    return serial_to_ip


def _collect(sock, serial_to_ip):
    """Read replies until the socket's timeout expires."""
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            try:
                message_dict = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue  # Not one of ours — the port is shared.
            serial_number = message_dict.get("sn")
            if not serial_number:
                continue
            if serial_number in serial_to_ip:
                continue
            ip_address = message_dict.get("ip", addr[0])
            serial_to_ip[serial_number] = ip_address
            print(f"Discovered device: {serial_number} at {ip_address}")
    except socket.timeout:
        pass
    except Exception as e:
        logging.error(f"An error occurred while receiving: {e}")


if __name__ == '__main__':
    import sys

    # Usage: python -m unitree_webrtc_connect.multicast_scanner [family] [sn]
    # The SN is mandatory in practice for G1 >= 1.5.1 and every R1.
    family = sys.argv[1] if len(sys.argv) > 1 else "Go2"
    sn = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Scanning for {family} robots{f' (sn={sn})' if sn else ''}...")
    if family in ("G1", "R1") and not sn:
        print("Note: G1 >= 1.5.1 and all R1 ignore untargeted queries — "
              "pass the serial number as the second argument to find one.")
    serial_to_ip = discover_ip_sn(timeout=3, device_type=family, sn=sn)
    print("\nDiscovered devices:")
    for serial_number, ip_address in serial_to_ip.items():
        print(f"Serial Number: {serial_number}, IP Address: {ip_address}")
