# Monkey-patch aioice.Connection to use a fixed username and password accross all instances.

import aioice


class Connection(aioice.Connection):
    local_username = aioice.utils.random_string(4)
    local_password = aioice.utils.random_string(22)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_username = Connection.local_username
        self.local_password = Connection.local_password


aioice.Connection = Connection  # type: ignore


# Monkey-patch aiortc.rtcdtlstransport.X509_DIGEST_ALGORITHMS to remove extra SHA algorithms
# Extra SHA algorithms introduced in aiortc 1.10.0 causes Unity Go2 to use the new SCTP format, despite aiortc using the old SCTP syntax.
# This new format is not supported by aiortc version as of today (2025-06-02)


import aiortc
from packaging.version import Version


if Version(aiortc.__version__) == Version("1.10.0"):
    X509_DIGEST_ALGORITHMS = {
        "sha-256": "SHA256",
    }
    aiortc.rtcdtlstransport.X509_DIGEST_ALGORITHMS = X509_DIGEST_ALGORITHMS

elif Version(aiortc.__version__) >= Version("1.11.0"):
    # Syntax changed in aiortc 1.11.0, so we need to use the hashes module
    from cryptography.hazmat.primitives import hashes

    X509_DIGEST_ALGORITHMS = {
        "sha-256": hashes.SHA256(),  # type: ignore
    }
    aiortc.rtcdtlstransport.X509_DIGEST_ALGORITHMS = X509_DIGEST_ALGORITHMS


# Public API
from .webrtc_driver import UnitreeWebRTCConnection  # noqa: E402
from .webrtc_datachannel import WebRTCDataChannel  # noqa: E402
from .constants import (  # noqa: E402
    WebRTCConnectionMethod,
    DATA_CHANNEL_TYPE,
    RTC_TOPIC,
    SPORT_CMD,
    SPORT_CMD_MCF,
    OBSTACLES_AVOID_API,
    LOCO_API,
    LOCO_API_SERVED_R1,
    LOCO_FSM_ERRORS,
    R1_FSM,
    ARM_ACTION,
    MOTOR_NAMES_HUMANOID,
    R1_MOTOR_INDICES,
    sign_byte,
)
from .multicast_scanner import discover_ip_sn  # noqa: E402
from .msgs.pub_sub import WebRTCDataChannelPubSub  # noqa: E402
from .unitree_cloud import (  # noqa: E402
    UnitreeCloud,
    UnitreeCloudError,
    RobotDevice,
    fetch_aes_key,
)
from .unitree_auth import (  # noqa: E402
    AesKeyRequiredError,
    AesKeyRejectedError,
    DataChannelTimeoutError,
    LocalSignalingPortError,
    NoSdpAnswerError,
    RobotBusyError,
)

__all__ = [
    "UnitreeWebRTCConnection",
    "WebRTCConnectionMethod",
    "WebRTCDataChannel",
    "WebRTCDataChannelPubSub",
    "DATA_CHANNEL_TYPE",
    "RTC_TOPIC",
    "SPORT_CMD",
    "SPORT_CMD_MCF",
    "OBSTACLES_AVOID_API",
    "LOCO_API",
    "LOCO_API_SERVED_R1",
    "LOCO_FSM_ERRORS",
    "R1_FSM",
    "ARM_ACTION",
    "MOTOR_NAMES_HUMANOID",
    "R1_MOTOR_INDICES",
    "sign_byte",
    "discover_ip_sn",
    "UnitreeCloud",
    "UnitreeCloudError",
    "RobotDevice",
    "fetch_aes_key",
    "AesKeyRequiredError",
    "AesKeyRejectedError",
    "DataChannelTimeoutError",
    "LocalSignalingPortError",
    "NoSdpAnswerError",
    "RobotBusyError",
]