import asyncio
import logging
import os
import sys
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.8.181")
# G1 firmware >= 1.5.1 requires an AES-128 key (data2=3 auth path).
AES_128_KEY = os.environ.get("UNITREE_AES_128_KEY")
    
async def main():
    try:
        # Choose a connection method (uncomment the correct one)
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_128_KEY)
        # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
        # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
        # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

        # Connect to the WebRTC service.
        await conn.connect()

        #
        # Actions
        #

        print("Performing Action...")
        await conn.datachannel.pub_sub.publish_request_new(
            "rt/api/arm/request", 
            {
                "api_id": 7106,
                "parameter": {"data": 27}   # 27 - Handshake, 18 - High Five, 19 - Hug, 26 - High Wave 
                                            # 17 - Clap, 25 - Face Wave, 12 - Left Kiss, 20 - Arm Heart
                                            # 21 - Right Heart, 15 - hands up, 24 - X-Ray, 23 - Right Hand up
                                            # 22 - Reject
            }
        )
        await asyncio.sleep(5)  # Wait 

        print("Returning Hand back from any Action...")
        await conn.datachannel.pub_sub.publish_request_new(
            "rt/api/arm/request", 
            {
                "api_id": 7106,
                "parameter": {"data": 99} # 99 - Canel any action, return hands back 
            }
        )
        await asyncio.sleep(5)  # Wait 

        #
        # Modes
        #

        # Switch Mode
        print("Switching Mode...")
        await conn.datachannel.pub_sub.publish_request_new(
            "rt/api/sport/request", 
            {
                "api_id": 7101,
                "parameter": {"data": 500}  # 500 - Walk, 501 - Walk(Control waist), 801 - Run
                                            # 
            }
        )
        await asyncio.sleep(5)  # Wait

        # Move
        print("Move...")
        conn.datachannel.pub_sub.publish_without_callback(
            "rt/wirelesscontroller", 
            {
                "lx": 0.0,
                "ly": 0.0,
                "rx": 1.0,
                "ry": 0.0,
                "keys": 0                                           
            }
        )
        await asyncio.sleep(3)  # Wait

        # STOP
        print("STOP...")
        conn.datachannel.pub_sub.publish_without_callback(
            "rt/wirelesscontroller", 
            {
                "lx": 0.0,
                "ly": 0.0,
                "rx": 0.0,
                "ry": 0.0,
                "keys": 0                                           
            }
        )


        

        # Keep the program running for a while
        await asyncio.sleep(3600)
    
    except ValueError as e:
        # Log any value errors that occur during the process.
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C to exit gracefully.
        print("\nProgram interrupted by user")
        sys.exit(0)