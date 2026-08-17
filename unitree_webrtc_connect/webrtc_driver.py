import asyncio
import logging
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import MediaStreamError
from .unitree_auth import (
    NoSdpAnswerError,
    RobotBusyError,
    send_sdp_to_local_peer,
    send_sdp_to_remote_peer,
)
from .webrtc_datachannel import WebRTCDataChannel
from .webrtc_audio import WebRTCAudioChannel
from .webrtc_video import WebRTCVideoChannel
from .constants import DATA_CHANNEL_TYPE, WebRTCConnectionMethod
from .util import fetch_public_key, fetch_token, fetch_turn_server_info, print_status
from .multicast_scanner import discover_ip_sn

# # Enable logging for debugging
# logging.basicConfig(level=logging.INFO)

class UnitreeWebRTCConnection:
    def __init__(
        self,
        connectionMethod: WebRTCConnectionMethod,
        serialNumber=None,
        ip=None,
        username=None,
        password=None,
        aes_128_key: str = None,
        region: str = "global",
        device_type: str = "Go2",
    ) -> None:
        """`aes_128_key` is the per-device 16-byte key (32 hex chars) the
        cloud returns as `dev.key` in `device/bind/list`. Required on
        G1 firmware ≥ 1.5.1, Go2 firmware ≥ 1.1.15 and every R1 for the
        LAN flow (con_notify returns `data2 === 3`); ignored on older
        firmware. Fetch it via `examples/fetch_aes_key.py` once per robot
        and cache locally.

        `region` (`"global"`/`"cn"`) and `device_type` (`"Go2"`/`"G1"`/
        `"R1"`) select the correct cloud endpoint + AppName header for the
        Remote signaling flow. `device_type` also picks which multicast
        groups serial-number discovery queries — the humanoids do not
        announce on the Go2 group."""
        self.pc = None
        self.sn = serialNumber
        self.ip = ip
        self.connectionMethod = connectionMethod
        self.isConnected = False
        self.aes_128_key = aes_128_key
        self.region = region
        self.device_type = device_type
        self.token = (
            fetch_token(username, password, region=region, device_type=device_type)
            if username and password
            else ""
        )

    async def connect(self):
        print_status("WebRTC connection", "🟡 started")
        if self.connectionMethod == WebRTCConnectionMethod.Remote:
            self.public_key = fetch_public_key(
                region=self.region, device_type=self.device_type,
            )
            turn_server_info = fetch_turn_server_info(
                self.sn, self.token, self.public_key,
                region=self.region, device_type=self.device_type,
            )
            await self.init_webrtc(turn_server_info)
        elif self.connectionMethod == WebRTCConnectionMethod.LocalSTA:
            if not self.ip and self.sn:
                discovered_ip_sn_addresses = discover_ip_sn(
                    device_type=self.device_type, sn=self.sn,
                )
                
                if discovered_ip_sn_addresses:
                    if self.sn in discovered_ip_sn_addresses:
                        self.ip = discovered_ip_sn_addresses[self.sn]
                    else:
                        raise ValueError("The provided serial number wasn't found on the network. Provide an IP address instead.")
                else:
                    raise ValueError("No devices found on the network. Provide an IP address instead.")

            await self.init_webrtc(ip=self.ip)
        elif self.connectionMethod == WebRTCConnectionMethod.LocalAP:
            self.ip = "192.168.12.1"
            await self.init_webrtc(ip=self.ip)
    
    async def disconnect(self):
        if self.pc:
            await self.pc.close()
            self.pc = None
        self.isConnected = False
        print_status("WebRTC connection", "🔴 disconnected")

    async def reconnect(self):
        await self.disconnect()
        await self.connect()
        print_status("WebRTC connection", "🟢 reconnected")

    def create_webrtc_configuration(self, turn_server_info, stunEnable=True, turnEnable=True) -> RTCConfiguration:
        ice_servers = []

        if turn_server_info:
            username = turn_server_info.get("user")
            credential = turn_server_info.get("passwd")
            turn_url = turn_server_info.get("realm")
            
            if username and credential and turn_url:
                if turnEnable:
                    ice_servers.append(
                        RTCIceServer(
                            urls=[turn_url],
                            username=username,
                            credential=credential
                        )
                    )
                if stunEnable:
                    # Use Google's public STUN server
                    stun_url = "stun:stun.l.google.com:19302"
                    ice_servers.append(
                        RTCIceServer(
                            urls=[stun_url]
                        )
                    )
            else:
                raise ValueError("Invalid TURN server information")
        
        configuration = RTCConfiguration(
            iceServers=ice_servers
        )
        
        return configuration

    async def init_webrtc(self, turn_server_info=None, ip=None):
        configuration = self.create_webrtc_configuration(turn_server_info)
        self.pc = RTCPeerConnection(configuration)


        self.datachannel = WebRTCDataChannel(self, self.pc)

        self.audio = WebRTCAudioChannel(self.pc, self.datachannel)
        self.video = WebRTCVideoChannel(self.pc, self.datachannel)

        @self.pc.on("icegatheringstatechange")
        async def on_ice_gathering_state_change():
            state = self.pc.iceGatheringState
            if state == "new":
                print_status("ICE Gathering State", "🔵 new")
            elif state == "gathering":
                print_status("ICE Gathering State", "🟡 gathering")
            elif state == "complete":
                print_status("ICE Gathering State", "🟢 complete")


        @self.pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            state = self.pc.iceConnectionState
            if state == "checking":
                print_status("ICE Connection State", "🔵 checking")
            elif state == "completed":
                print_status("ICE Connection State", "🟢 completed")
            elif state == "failed":
                print_status("ICE Connection State", "🔴 failed")
            elif state == "closed":
                print_status("ICE Connection State", "⚫ closed")


        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = self.pc.connectionState
            if state == "connecting":
                print_status("Peer Connection State", "🔵 connecting")
            elif state == "connected":
                self.isConnected= True
                print_status("Peer Connection State", "🟢 connected")
            elif state == "closed":
                self.isConnected= False
                print_status("Peer Connection State", "⚫ closed")
            elif state == "failed":
                print_status("Peer Connection State", "🔴 failed")
        
        @self.pc.on("signalingstatechange")
        async def on_signaling_state_change():
            state = self.pc.signalingState
            if state == "stable":
                print_status("Signaling State", "🟢 stable")
            elif state == "have-local-offer":
                print_status("Signaling State", "🟡 have-local-offer")
            elif state == "have-remote-offer":
                print_status("Signaling State", "🟡 have-remote-offer")
            elif state == "closed":
                print_status("Signaling State", "⚫ closed")
        
        @self.pc.on("track")
        async def on_track(track):
            logging.info("Track received: %s", track.kind)

            # `MediaStreamError` is how aiortc signals end-of-track when the
            # peer closes — wrap both reader loops so a clean disconnect
            # doesn't dump a pyee traceback into the user's console.
            try:
                if track.kind == "video":
                    # Discard first frame, then hand the track to the video reader.
                    await track.recv()
                    await self.video.track_handler(track)

                elif track.kind == "audio":
                    await track.recv()  # warm-up
                    while True:
                        frame = await track.recv()
                        await self.audio.frame_handler(frame)
            except MediaStreamError:
                logging.debug("Track %s ended", track.kind)

        logging.info("Creating offer...")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        if self.connectionMethod == WebRTCConnectionMethod.Remote:
            peer_answer_json = await self.get_answer_from_remote_peer(self.pc, turn_server_info)
        elif self.connectionMethod == WebRTCConnectionMethod.LocalSTA or self.connectionMethod == WebRTCConnectionMethod.LocalAP:
            peer_answer_json = await self.get_answer_from_local_peer(self.pc, self.ip)

        if peer_answer_json is None:
            raise NoSdpAnswerError()
        peer_answer = json.loads(peer_answer_json)

        if peer_answer['sdp'] == "reject":
            raise RobotBusyError()

        remote_sdp = RTCSessionDescription(sdp=peer_answer['sdp'], type=peer_answer['type']) 
        await self.pc.setRemoteDescription(remote_sdp)
   
        await self.datachannel.wait_datachannel_open()

    
    async def get_answer_from_remote_peer(self, pc, turn_server_info):
        sdp_offer = pc.localDescription

        sdp_offer_json = {
            "id": "",
            "turnserver": turn_server_info,
            "sdp": sdp_offer.sdp,
            "type": sdp_offer.type,
            "token": self.token
        }

        logging.debug("Local SDP created: %s", sdp_offer_json)

        peer_answer_json = send_sdp_to_remote_peer(
            self.sn,
            json.dumps(sdp_offer_json),
            self.token,
            self.public_key,
            region=self.region,
            device_type=self.device_type,
        )

        return peer_answer_json

    async def get_answer_from_local_peer(self, pc, ip):
        sdp_offer = pc.localDescription

        sdp_offer_json = {
            "id": "STA_localNetwork" if self.connectionMethod == WebRTCConnectionMethod.LocalSTA else "",
            "sdp": sdp_offer.sdp,
            "type": sdp_offer.type,
            "token": self.token
        }

        peer_answer_json = send_sdp_to_local_peer(
            ip, json.dumps(sdp_offer_json), aes_128_key=self.aes_128_key,
        )

        return peer_answer_json


