from enum import Enum

DATA_CHANNEL_TYPE = {
    "VALIDATION": "validation",
    "SUBSCRIBE": "subscribe",
    "UNSUBSCRIBE": "unsubscribe",
    "MSG": "msg",
    "REQUEST": "req",
    "RESPONSE": "res",
    "VID": "vid",
    "AUD": "aud",
    "ERR": "err",
    "HEARTBEAT": "heartbeat",
    "RTC_INNER_REQ": "rtc_inner_req",
    "RTC_REPORT": "rtc_report",
    "ADD_ERROR": "add_error",
    "RM_ERROR": "rm_error",
    "ERRORS": "errors",
}

class WebRTCConnectionMethod(Enum):
    LocalAP = 1
    LocalSTA = 2
    Remote = 3

app_error_messages = {
    "app_error_code_100_1": "DDS message timeout",
    "app_error_code_100_10": "Battery communication error",
    "app_error_code_100_2": "Distribution switch abnormal",
    "app_error_code_100_20": "Abnormal mote control communication",
    "app_error_code_100_40": "MCU communication error",
    "app_error_code_100_80": "Motor communication error",
    "app_error_code_200_1": "Rear left fan jammed",
    "app_error_code_200_2": "Rear right fan jammed",
    "app_error_code_200_4": "Front fan jammed",
    "app_error_code_300_1": "Overcurrent",
    "app_error_code_300_10": "Winding overheating",
    "app_error_code_300_100": "Motor communication interruption",
    "app_error_code_300_2": "Overvoltage",
    "app_error_code_300_20": "Encoder abnormal",
    "app_error_code_300_4": "Driver overheating",
    "app_error_code_300_8": "Generatrix undervoltage",
    "app_error_code_400_1": "Motor rotate speed abnormal",
    "app_error_code_400_10": "Abnormal dirt index",
    "app_error_code_400_2": "PointCloud data abnormal",
    "app_error_code_400_4": "Serial port data abnormal",
    "app_error_code_500_1": "UWB serial port open abnormal",
    "app_error_code_500_2": "Robot dog information retrieval abnormal",
    "app_error_code_600_4": "Overheating software protection",
    "app_error_code_600_8": "Low battery software protection",
    "app_error_source_100": "Communication firmware malfunction",
    "app_error_source_200": "Communication firmware malfunction",
    "app_error_source_300": "Motor malfunction",
    "app_error_source_400": "Radar malfunction",
    "app_error_source_500": "UWB malfunction",
    "app_error_source_600": "Motion Control",
    "app_error_wheel_300_100": "Motor Communication Interruption",
    "app_error_wheel_300_40": "Calibration Data Abnormality",
    "app_error_wheel_300_80": "Abnormal Reset"
}

RTC_TOPIC = {
    "LOW_STATE": "rt/lf/lowstate",
    "MULTIPLE_STATE": "rt/multiplestate",
    "FRONT_PHOTO_REQ": "rt/api/videohub/request",
    "ULIDAR_SWITCH": "rt/utlidar/switch",
    "ULIDAR": "rt/utlidar/voxel_map",
    "ULIDAR_ARRAY": "rt/utlidar/voxel_map_compressed",
    "ULIDAR_STATE": "rt/utlidar/lidar_state",
    "ROBOTODOM": "rt/utlidar/robot_pose",
    "UWB_REQ": "rt/api/uwbswitch/request",
    "UWB_STATE": "rt/uwbstate",
    "LOW_CMD": "rt/lowcmd",
    "WIRELESS_CONTROLLER": "rt/wirelesscontroller",
    "SPORT_MOD": "rt/api/sport/request",
    "SPORT_MOD_STATE": "rt/sportmodestate",
    "LF_SPORT_MOD_STATE": "rt/lf/sportmodestate",
    "BASH_REQ": "rt/api/bashrunner/request",
    "SELF_TEST": "rt/selftest",
    "GRID_MAP": "rt/mapping/grid_map",
    "SERVICE_STATE": "rt/servicestate",
    "GPT_FEEDBACK": "rt/gptflowfeedback",
    "VUI": "rt/api/vui/request",
    "OBSTACLES_AVOID": "rt/api/obstacles_avoid/request",
    "SLAM_QT_COMMAND": "rt/qt_command",
    "SLAM_ADD_NODE": "rt/qt_add_node",
    "SLAM_ADD_EDGE": "rt/qt_add_edge",
    "SLAM_QT_NOTICE": "rt/qt_notice",
    "SLAM_PC_TO_IMAGE_LOCAL": "rt/pctoimage_local",
    "SLAM_ODOMETRY": "rt/lio_sam_ros2/mapping/odometry",
    "ARM_COMMAND": "rt/arm_Command",
    "ARM_FEEDBACK": "rt/arm_Feedback",
    # Humanoid (G1 / R1) upper-limb request channel. Distinct from
    # ARM_COMMAND above: this one carries api_id requests (see ARM_ACTION),
    # the other is the low-level arm command stream.
    "ARM_REQUEST": "rt/api/arm/request",
    # Humanoid state topics. Go2 packs battery into lowstate and ships one
    # IMU; the humanoids split them out:
    #   BMS_STATE       — the bms struct on its own topic, not nested
    #   SECONDARY_IMU   — pelvis IMU (lowstate.imu_state is the torso one)
    #   MAIN_BOARD_STATE— body/chassis temperature (Go2 uses temperature_ntc1)
    "BMS_STATE": "rt/lf/bmsstate",
    "SECONDARY_IMU": "rt/lf/secondary_imu",
    "MAIN_BOARD_STATE": "rt/lf/mainboardstate",
    "AUDIO_HUB_REQ": "rt/api/audiohub/request",
    "AUDIO_HUB_PLAY_STATE": "rt/audiohub/player/state",
    "GAS_SENSOR": "rt/gas_sensor",
    "GAS_SENSOR_REQ": "rt/api/gas_sensor/request",
    "LIDAR_MAPPING_CMD": "rt/uslam/client_command",
    "LIDAR_MAPPING_CLOUD_POINT": "rt/uslam/frontend/cloud_world_ds",
    "LIDAR_MAPPING_ODOM": "rt/uslam/frontend/odom",
    "LIDAR_MAPPING_PCD_FILE": "rt/uslam/cloud_map",
    "LIDAR_MAPPING_SERVER_LOG": "rt/uslam/server_log",
    "LIDAR_LOCALIZATION_ODOM": "rt/uslam/localization/odom",
    "LIDAR_NAVIGATION_GLOBAL_PATH": "rt/uslam/navigation/global_path",
    "LIDAR_LOCALIZATION_CLOUD_POINT": "rt/uslam/localization/cloud_world",
    "PROGRAMMING_ACTUATOR_CMD": "rt/programming_actuator/command",
    "ASSISTANT_RECORDER": "rt/api/assistant_recorder/request",
    "MOTION_SWITCHER": "rt/api/motion_switcher/request"
}

SPORT_CMD = {
    "Damp": 1001,
    "BalanceStand": 1002,
    "StopMove": 1003,
    "StandUp": 1004,
    "StandDown": 1005,
    "RecoveryStand": 1006,
    "Euler": 1007,
    "Move": 1008,
    "Sit": 1009,
    "RiseSit": 1010,
    "SwitchGait": 1011,
    "Trigger": 1012,
    "BodyHeight": 1013,
    "FootRaiseHeight": 1014,
    "SpeedLevel": 1015,
    "Hello": 1016,
    "Stretch": 1017,
    "TrajectoryFollow": 1018,
    "ContinuousGait": 1019,
    "Content": 1020,
    "Wallow": 1021,
    "Dance1": 1022,
    "Dance2": 1023,
    "GetBodyHeight": 1024,
    "GetFootRaiseHeight": 1025,
    "GetSpeedLevel": 1026,
    "SwitchJoystick": 1027,
    "Pose": 1028,
    "Scrape": 1029,
    "FrontFlip": 1030,
    "LeftFlip": 1042,
    "RightFlip": 1043,
    "BackFlip": 1044,
    "FrontJump": 1031,
    "FrontPounce": 1032,
    "WiggleHips": 1033,
    "GetState": 1034,
    "EconomicGait": 1035,
    "LeadFollow": 1045,
    "FingerHeart": 1036,
    "Bound": 1304,
    "MoonWalk": 1305,
    "OnesidedStep": 1303,
    "CrossStep": 1302,
    "Handstand": 1301,
    "StandOut": 1039,
    "FreeWalk": 1045,
    "Standup": 1050,
    "CrossWalk": 1051
}

# MCF (Multi-Control Framework) sport api_ids — introduced in Unitree firmware
# 1.1.7 and used since. Shares the `rt/api/sport/request` topic with normal
# mode but uses a different api_id space (e.g. BackFlip is 2043 in MCF vs 1044
# in normal). Robot must already be in MCF mode (no motion_switcher handshake).
SPORT_CMD_MCF = {
    "Damp":             1001,
    "BalanceStand":     1002,
    "StopMove":         1003,
    "StandUp":          1004,
    "StandDown":        1005,
    "RecoveryStand":    1006,
    "Euler":            1007,
    "Move":             1008,
    "Sit":              1009,
    "RiseSit":          1010,
    "SpeedLevel":       1015,
    "Hello":            1016,
    "Stretch":          1017,
    "ContinuousGait":   1019,
    "Content":          1020,
    "Dance1":           1022,
    "Dance2":           1023,
    "GetSpeedLevel":    1026,
    "SwitchJoystick":   1027,
    "Pose":             1028,
    "Scrape":           1029,
    "FrontFlip":        1030,
    "FrontJump":        1031,
    "FrontPounce":      1032,
    "GetState":         1034,
    "Heart":            1036,
    "StaticWalk":       1061,
    "TrotRun":          1062,
    "EconomicGait":     1063,
    "LeftFlip":         2041,
    "BackFlip":         2043,
    "HandStand":        2044,
    "FreeWalk":         2045,
    "FreeBound":        2046,
    "FreeJump":         2047,
    "FreeAvoid":        2048,
    "ClassicWalk":      2049,
    "BackStand":        2050,
    "CrossStep":        2051,
    "SetAutoRecovery":  2054,
    "GetAutoRecovery":  2055,
    "LeadFollow":       2056,
    "SwitchAvoidMode":  2058,
}

# Obstacle-avoidance api_ids — topic is `rt/api/obstacles_avoid/request`
# (= RTC_TOPIC["OBSTACLES_AVOID"]).
OBSTACLES_AVOID_API = {
    "SWITCH_SET":                  1001,  # {"enable": bool}
    "SWITCH_GET":                  1002,  # {} -> {"enable": bool}
    "MOVE":                        1003,  # {"x", "y", "yaw", "mode": 0} (no-reply)
    "USE_REMOTE_COMMAND_FROM_API": 1004,  # {"is_remote_commands_from_api": bool}
}

# ─── Humanoid (G1 / R1) ───────────────────────────────────────────────
#
# The humanoids don't use the Go2 sport command space. Every persistent
# posture, gait and motion is a single FSM state selected through ONE api
# on `rt/api/sport/request` (= RTC_TOPIC["SPORT_MOD"]): you send
# SET_FSM_ID and change the `fsm_id`, rather than picking a command id.
# The robot echoes the current state back in `fsm_id` on
# `rt/lf/sportmodestate` (= RTC_TOPIC["LF_SPORT_MOD_STATE"]).

LOCO_API = {
    "GET_FSM_ID":         7001,
    "GET_FSM_MODE":       7002,
    "GET_ARM_SDK_STATUS": 7007,
    "SET_FSM_ID":         7101,
    "SET_VELOCITY":       7105,  # {"velocity":[vx,vy,omega],"duration":<s>}
    "SET_ARM_TASK":       7106,  # on RTC_TOPIC["ARM_REQUEST"], not SPORT_MOD
    "SET_SPEED_MODE":     7107,
    "SET_MOTION":         7108,  # parameter is an ARRAY, not {"data": N}
    "SET_ARM_SDK_STATUS": 7109,
}

# Only a subset is actually served: R1's loco server registers
# 7001 / 7101 / 7105 / 7108. The rest exist as constants in the firmware
# but are not bound to a handler, so they answer nothing.
LOCO_API_SERVED_R1 = ("GET_FSM_ID", "SET_FSM_ID", "SET_VELOCITY", "SET_MOTION")

# `SET_FSM_ID` error codes, from the on-robot handler.
LOCO_FSM_ERRORS = {
    1001: "transition refused by the current state (white/black list)",
    1002: "transition refused by the current state (white/black list)",
    1003: "invalid fsm id — that state does not exist on this firmware",
}

# R1 FSM states. Note Run is 811 (`AmpMotion22Dof`), NOT G1's 801 — and on
# an AIR chassis the firmware redirects a request for 811 to 830
# (`Locomotion20Dofs`), so the state reported back may differ from the one
# requested. States marked "app" are the four the official R1 app exposes.
R1_FSM = {
    "ZeroTorque":   0,    # app
    "Damping":      1,    # app — also the universal escape hatch
    "Lock":         4,    # app (internally `Stance`)
    "Keep":         5,
    "MoveTo":       6,
    "SitDown":      7,
    "Dance1":       601,
    "Dance2":       602,
    "Dance3":       603,
    "Twist":        604,  # niuniuwu 扭扭舞
    "KungFu":       607,  # gongfu 功夫
    "JeetKuneDo":   608,  # jiequandao 截拳道
    "StandUp":      701,  # Qishen 起身 — recovers from face-up or face-down
    "LieDown":      702,  # Tangxia 躺下
    "Motion":       800,
    "Run":          811,  # app — AmpMotion22Dof
    "AmpMotion":    812,
    "WalkStraightKnee": 813,
    "Walk":         814,
    "AmpLocomotion": 815,
    "ArmSdkLoco":   816,
    "Loco20Dof":    830,
    "LocoArmSdk":   831,
}

# Reachability is enforced on the robot, per state, and Damping is the only
# transition accepted from everywhere. Most motions are reachable only from
# Lock (4), not from Run. A refused switch answers 1001; a state this
# firmware doesn't build answers 1003.

# Motor index -> name for the humanoid lowstate array. G1 fills all 29
# slots (12 legs + 3 waist + 14 arms). R1 is 20-DOF and populates a subset
# of the SAME array: legs 0-11, left arm 15-18, right arm 22-25 — it has no
# waist and no wrists, so those slots stay zeroed.
MOTOR_NAMES_HUMANOID = [
    "L Hip P", "L Hip R", "L Hip Y", "L Knee", "L Ank P", "L Ank R",
    "R Hip P", "R Hip R", "R Hip Y", "R Knee", "R Ank P", "R Ank R",
    "Waist Y", "Waist R", "Waist P",
    "L Sho P", "L Sho R", "L Sho Y", "L Elbow", "L Wri R", "L Wri P", "L Wri Y",
    "R Sho P", "R Sho R", "R Sho Y", "R Elbow", "R Wri R", "R Wri P", "R Wri Y",
]

# The slots R1 actually drives, in a sensible reading order.
R1_MOTOR_INDICES = list(range(0, 12)) + list(range(15, 19)) + list(range(22, 26))


def sign_byte(value):
    """Temperatures arrive as unsigned bytes but represent signed values."""
    return value - 256 if isinstance(value, (int, float)) and value > 127 else value


# Upper-limb gestures — api_id LOCO_API["SET_ARM_TASK"] on
# RTC_TOPIC["ARM_REQUEST"], parameter {"data": <id>}. Shared with G1 except
# the two heart gestures (ArmHeart 20 / RightHeart 21), which R1 omits.
ARM_ACTION = {
    "Release":      99,   # cancel any gesture, return the arms
    "LeftKiss":     12,
    "HandsUp":      15,
    "Clap":         17,
    "HighFive":     18,
    "Hug":          19,
    "ArmHeart":     20,   # G1 only
    "RightHeart":   21,   # G1 only
    "Reject":       22,
    "RightHandUp":  23,
    "XRay":         24,
    "FaceWave":     25,
    "HighWave":     26,
    "Handshake":    27,
    "ForwardPush":  36,
}

class VUI_COLOR:
    WHITE: str = 'white'
    RED: str = 'red'
    YELLOW: str = 'yellow'
    BLUE: str = 'blue'
    GREEN: str = 'green'
    CYAN: str = 'cyan'
    PURPLE: str = 'purple'

# Audio API IDs
AUDIO_API = {
    # Audio Player Commands
    "GET_AUDIO_LIST": 1001,
    "SELECT_START_PLAY": 1002,
    "PAUSE": 1003,
    "UNSUSPEND": 1004,
    "SELECT_PREV_START_PLAY": 1005,
    "SELECT_NEXT_START_PLAY": 1006,
    "SET_PLAY_MODE": 1007,
    "SELECT_RENAME": 1008,
    "SELECT_DELETE": 1009,
    "GET_PLAY_MODE": 1010,
    
    # Audio Upload
    "UPLOAD_AUDIO_FILE": 2001,
    
    # Internal Corpus
    "PLAY_START_OBSTACLE_AVOIDANCE": 3001,
    "PLAY_EXIT_OBSTACLE_AVOIDANCE": 3002,
    "PLAY_START_COMPANION_MODE": 3003,
    "PLAY_EXIT_COMPANION_MODE": 3004,
    
    # Megaphone
    "ENTER_MEGAPHONE": 4001,
    "EXIT_MEGAPHONE": 4002,
    "UPLOAD_MEGAPHONE": 4003,
    
    # Internal Long Corpus
    "INTERNAL_LONG_CORPUS_SELECT_TO_PLAY": 5001,
    "INTERNAL_LONG_CORPUS_PLAYBACK_COMPLETED": 5002,
    "INTERNAL_LONG_CORPUS_STOP_PLAYING": 5003
}