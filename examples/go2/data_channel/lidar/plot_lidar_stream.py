""" @MrRobotoW at The RoboVerse Discord """
""" robert.wagoner@gmail.com """
""" 01/30/2025 """
""" Inspired from lidar_stream.py by @legion1581 at The RoboVerse Discord """

VERSION = "1.0.18"

import asyncio
import logging
import csv
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
import argparse
from datetime import datetime
import os
import sys
import ast

# Increase the field size limit for CSV reading
csv.field_size_limit(sys.maxsize)

# Flask app and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

logging.basicConfig(level=logging.FATAL)

# Constants to enable/disable features
ENABLE_POINT_CLOUD = True
SAVE_LIDAR_DATA = True

# File paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LIDAR_CSV_FILE = f"lidar_data_{timestamp}.csv"

# Global variables
lidar_csv_file = None
lidar_csv_writer = None

lidar_buffer = []
message_count = 0  # Counter for processed LIDAR messages
reconnect_interval = 5  # Time (seconds) before retrying connection

# Constants
MAX_RETRY_ATTEMPTS = 10

ROTATE_X_ANGLE = np.pi / 2  # 90 degrees
ROTATE_Z_ANGLE = np.pi      # 90 degrees

minYValue = 0
maxYValue = 100

# Parse command-line arguments
parser = argparse.ArgumentParser(description=f"LIDAR Viz v{VERSION}")
parser.add_argument("--version", action="version", version=f"LIDAR Viz v{VERSION}")
parser.add_argument("--cam-center", action="store_true", help="Put Camera at the Center")
parser.add_argument("--type-voxel", action="store_true", help="Voxel View")
parser.add_argument("--csv-read", type=str, help="Read from CSV files instead of WebRTC")
parser.add_argument("--csv-write", action="store_true", help="Write CSV data file")
parser.add_argument("--skip-mod", type=int, default=1, help="Skip messages using modulus (default: 1, no skipping)")
parser.add_argument('--minYValue', type=int, default=0, help='Minimum Y value for the plot')
parser.add_argument('--maxYValue', type=int, default=100, help='Maximum Y value for the plot')
args = parser.parse_args()

minYValue = args.minYValue
maxYValue = args.maxYValue
SAVE_LIDAR_DATA = args.csv_write

@socketio.on('check_args')
def handle_check_args():
    global ack_received
    typeFlag = 0b0101 # default iso cam & point cloud
    if args.cam_center:
        typeFlag |= 0b0010
    if args.type_voxel:
        typeFlag &= ~0b1000 # disable point cloud
        typeFlag |=  0b1000  # Set voxel flag
    typeFlagBinary = format(typeFlag, "04b")
    socketio.emit("check_args_ack", {"type": typeFlagBinary})

def setup_csv_output():
    """Set up CSV files for LIDAR output."""
    global lidar_csv_file, lidar_csv_writer

    if SAVE_LIDAR_DATA:
        lidar_csv_file = open(LIDAR_CSV_FILE, mode='w', newline='', encoding='utf-8')
        lidar_csv_writer = csv.writer(lidar_csv_file)
        lidar_csv_writer.writerow(['stamp', 'frame_id', 'resolution', 'src_size', 'origin', 'width', 
                                   'point_count', 'positions'])
        lidar_csv_file.flush()  # Ensure the header row is flushed to disk

def close_csv_output():
    """Close CSV files."""
    global lidar_csv_file

    if lidar_csv_file:
        lidar_csv_file.close()
        lidar_csv_file = None

def filter_points(points, percentage):
    """Filter points to skip plotting points within a certain percentage of distance to each other."""
    return points  # No filtering

def rotate_points(points, x_angle, z_angle):
    """Rotate points around the x and z axes by given angles."""
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])
    
    rotation_matrix_z = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0],
        [np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])
    
    points = points @ rotation_matrix_x.T
    points = points @ rotation_matrix_z.T
    return points

async def lidar_webrtc_connection():
    """Connect to WebRTC and process LIDAR data."""
    global lidar_buffer, message_count
    retry_attempts = 0

    # checkArgs()

    while retry_attempts < MAX_RETRY_ATTEMPTS:
        try:
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA)  # WebRTC IP
            # _webrtc_connection = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
            # _webrtc_connection = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

            # Connect to WebRTC
            logging.info("Connecting to WebRTC...")
            await conn.connect()
            logging.info("Connected to WebRTC.")
            retry_attempts = 0  # Reset retry attempts on successful connection

            # Disable traffic saving mode
            await conn.datachannel.disableTrafficSaving(True)

            # Turn LIDAR sensor on
            conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")

            # Set up CSV outputs
            setup_csv_output()

            async def lidar_callback_task(message):
                """Task to process incoming LIDAR data."""
                if not ENABLE_POINT_CLOUD:
                    return

                try:
                    global message_count
                    if message_count % args.skip_mod != 0:
                        message_count += 1
                        return

                    positions = message["data"]["data"].get("positions", [])
                    origin = message["data"].get("origin", [])
                    points = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)
                    total_points = len(points)
                    unique_points = np.unique(points, axis=0)

                    # Save to CSV
                    if SAVE_LIDAR_DATA and lidar_csv_writer:
                        lidar_csv_writer.writerow([
                            message["data"]["stamp"],
                            message["data"]["frame_id"],
                            message["data"]["resolution"],
                            message["data"]["src_size"],
                            message["data"]["origin"],
                            message["data"]["width"],
                            len(unique_points),
                            unique_points.tolist()  # Save full data
                        ])
                        lidar_csv_file.flush()  # Flush data to disk

                    points = rotate_points(unique_points, ROTATE_X_ANGLE, ROTATE_Z_ANGLE)  # Rotate points
                    points = points[(points[:, 1] >= minYValue) & (points[:, 1] <= maxYValue)]

                    # Calculate center coordinates
                    center_x = float(np.mean(points[:, 0]))
                    center_y = float(np.mean(points[:, 1]))
                    center_z = float(np.mean(points[:, 2]))

                    # Offset points by center coordinates
                    offset_points = points - np.array([center_x, center_y, center_z])

                    # Count and log points
                    message_count += 1
                    print(f"LIDAR Message {message_count}: Total points={total_points}, Unique points={len(unique_points)}")

                    # Emit data to Socket.IO
                    scalars = np.linalg.norm(offset_points, axis=1)  # Color by distance
                    socketio.emit("lidar_data", {
                        "points": offset_points.tolist(),
                        "scalars": scalars.tolist(),
                        "center": {"x": center_x, "y": center_y, "z": center_z}
                    })

                except Exception as e:
                    logging.error(f"Error in LIDAR callback: {e}")

            # Subscribe to LIDAR voxel map messages
            conn.datachannel.pub_sub.subscribe(
                "rt/utlidar/voxel_map_compressed",
                lambda message: asyncio.create_task(lidar_callback_task(message))
            )

            # Keep the connection active
            while True:
                await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.info(f"Reconnecting in {reconnect_interval} seconds... (Attempt {retry_attempts + 1}/{MAX_RETRY_ATTEMPTS})")
            close_csv_output()
            try:
                await conn.disconnect()
            except Exception as e:
                logging.error(f"Error during disconnect: {e}")
            await asyncio.sleep(reconnect_interval)
            retry_attempts += 1

    logging.error("Max retry attempts reached. Exiting.")

async def read_csv_and_emit(csv_file):
    """Continuously read CSV files and emit data without delay."""
    global message_count

    # checkArgs()

    while True:  # Infinite loop to restart at EOF
        try:
            total_messages = sum(1 for _ in open(csv_file)) - 1  # Calculate total messages

            with open(csv_file, mode='r', newline='', encoding='utf-8') as lidar_file:
                lidar_reader = csv.DictReader(lidar_file)

                for lidar_row in lidar_reader:
                    if message_count % args.skip_mod == 0:
                        try:
                            # Extract and validate positions
                            positions = ast.literal_eval(lidar_row.get("positions", "[]"))
                            if isinstance(positions, list) and all(isinstance(item, list) and len(item) == 3 for item in positions):
                                points = np.array(positions, dtype=np.float32)
                            else:
                                points = np.array([item for item in positions if isinstance(item, list) and len(item) == 3], dtype=np.float32)

                            # Extract and compute origin, resolution, width, and center
                            origin = np.array(eval(lidar_row.get("origin", "[]")), dtype=np.float32)
                            resolution = float(lidar_row.get("resolution", 0.05))
                            width = np.array(eval(lidar_row.get("width", "[128, 128, 38]")), dtype=np.float32)
                            center = origin + (width * resolution) / 2

                            # Process points
                            if points.size > 0:
                                points = rotate_points(points, ROTATE_X_ANGLE, ROTATE_Z_ANGLE)
                                points = points[(points[:, 1] >= minYValue) & (points[:, 1] <= maxYValue)]
                                unique_points = np.unique(points, axis=0)
                                # Calculate center coordinates
                                center_x = float(np.mean(unique_points[:, 0]))
                                center_y = float(np.mean(unique_points[:, 1]))
                                center_z = float(np.mean(unique_points[:, 2]))

                                # Offset points by center coordinates
                                offset_points = unique_points - np.array([center_x, center_y, center_z])
                            else:
                                unique_points = np.empty((0, 3), dtype=np.float32)
                                offset_points = unique_points

                            # Emit data to Socket.IO
                            scalars = np.linalg.norm(offset_points, axis=1)
                            socketio.emit("lidar_data", {
                                "points": offset_points.tolist(),
                                "scalars": scalars.tolist(),
                                "center": {"x": center_x, "y": center_y, "z": center_z}
                            })

                            # Print message details
                            print(f"LIDAR Message {message_count}/{total_messages}: Unique points={len(unique_points)}")

                        except Exception as e:
                            logging.error(f"Exception during processing: {e}")

                    # Increment message count
                    message_count += 1

            # Restart file reading when EOF is reached
            message_count = 0  # Reset counter if needed

        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>LIDAR Viz v{{ version }}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style> body { margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; } canvas { display: block; } </style>
    </head>
    <body>
        <script>
            let scene, camera, renderer, controls, pointCloud, voxelMesh;
            let voxelSize = 1.0;
            let transparency = .5;
            let wireframe = false;
            let lightIntensity = .5;
            let pointCloudEnable = 1;
            let pollingInterval;
            const socket = io();
            document.addEventListener("DOMContentLoaded", () => {                                 
                function init() {
                    // Initialize the scene
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x333333);

                    const sceneRotationDegrees = -90;  // Change this to any angle (e.g., 90, 180, -90)
                    scene.rotation.y = THREE.MathUtils.degToRad(sceneRotationDegrees); // Convert to radians

                    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
                    camera.position.set(-100, 100, -100); // Adjust camera for rotated scene
                    camera.lookAt(0, 0, 0); // Ensure it's looking at the center

                    // Initialize the renderer
                    const renderer = new THREE.WebGLRenderer({ antialias: true });
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    document.body.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.target.set(0, 0, 0); // Ensure the rotation works around the center
                    controls.enableDamping = true; // Smooth movement
                    controls.dampingFactor = 0.05;
                    controls.maxPolarAngle = Math.PI; // Allow full rotation
                    controls.screenSpacePanning = true;
                    controls.update();

                    const ambientLight = new THREE.AmbientLight(0x555555, 0.5); // Soft background light
                    scene.add(ambientLight);

                    const directionalLight = new THREE.DirectionalLight(0xffffff, 1); // Main light source
                    directionalLight.position.set(0, 100, 0); // Position above the scene
                    directionalLight.castShadow = true;
                    scene.add(directionalLight);

                    const axesHelper = new THREE.AxesHelper(5);
                    scene.add(axesHelper);
                                                                                                                              
                    socket.on("connect", () => {
                        console.log("Socket connected...");
                        pollArgs();
                    });
                                  
                    socket.on("check_args_ack", (data) => {
                        console.log("Received check_args event:", data);
                         const typeFlag = parseInt(data.type, 2);                         
                        if (typeFlag & 0b0001) {  
                            camera.position.set(-100, 100, -100); // Adjust camera for rotated scene
                            camera.lookAt(0, 0, 0); // Ensure it's looking at the center
                        }
                        if (typeFlag & 0b0010) {  
                            camera.position.set(0, 0, 10); // Set camera at the center of the scene
                            camera.lookAt(0, 0, -1); // Look slightly forward       
                         }
                        if (typeFlag & 0b0100) {  
                            pointCloudEnable = 1;
                            console.log("ptcloud:", pointCloudEnable);
                        }
                        if (typeFlag & 0b1000) {  
                            pointCloudEnable = 0;
                            console.log("ptcloud:", pointCloudEnable);
                        }
                        controls.update();
                        clearInterval(pollingInterval);
                     });
                                  
                    // Handle LIDAR data
                    socket.on("lidar_data", (data) => {
                        if (!data.handled) {
                            data.handled = true; // Prevent re-triggering
                            console.log("Received LIDAR data");
                            const points = data.points || [];
                            const scalars = data.scalars || [];

                            if (pointCloudEnable > 0) {
                                if (pointCloud) scene.remove(pointCloud);
                                if (voxelMesh) {
                                    scene.remove(voxelMesh);
                                    voxelMesh = null;
                                }

                                const geometry = new THREE.BufferGeometry();
                                const vertices = new Float32Array(points.flat());
                                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

                                const colors = new Float32Array(scalars.length * 3);
                                const maxScalar = Math.max.apply(null, scalars);
                                scalars.forEach((scalar, i) => {
                                    const color = new THREE.Color();
                                    color.setHSL(scalar / maxScalar, 1.0, 0.5);
                                    colors.set([color.r, color.g, color.b], i * 3);
                                });

                                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

                                const material = new THREE.PointsMaterial({ size: 0.3, vertexColors: true });
                                pointCloud = new THREE.Points(geometry, material);

                                scene.add(pointCloud);
                            } else {
                                if (voxelMesh) scene.remove(voxelMesh);
                                voxelMesh = createVoxelMesh(points, scalars, voxelSize, Infinity);
                                if (voxelMesh instanceof THREE.Object3D) {
                                    scene.add(voxelMesh);
                                }
                                // Remove any existing point cloud
                                if (pointCloud) {
                                    scene.remove(pointCloud);
                                    pointCloud = null;
                                }
                            }
                        }
                    });

                    function animate() {
                        requestAnimationFrame(animate);
                        controls.update();
                        renderer.render(scene, camera);
                    }

                    animate();
                }

                init();
            });

            function pollArgs() {
                pollingInterval = setInterval(() => {
                    socket.emit('check_args');
                }, 1000); // Poll every second
            }                     
                                     
            /**
            * Creates a voxel mesh from point data and scalar data.
            */
            function createVoxelMesh(points, scalars, voxelSize, maxVoxelsToShow = Infinity) {
                const geometry = new THREE.BufferGeometry();

                try {
                    // Precompute cube vertex offsets
                    const halfSize = voxelSize / 2;
                    const cubeVertexOffsets = [
                        [-halfSize, -halfSize, -halfSize],
                        [halfSize, -halfSize, -halfSize],
                        [halfSize, halfSize, -halfSize],
                        [-halfSize, halfSize, -halfSize],
                        [-halfSize, -halfSize, halfSize],
                        [halfSize, -halfSize, halfSize],
                        [halfSize, halfSize, halfSize],
                        [-halfSize, halfSize, halfSize]
                    ];

                    // Precompute indices for a unit cube
                    const cubeIndices = [
                        0, 1, 2, 2, 3, 0, // Back
                        4, 5, 6, 6, 7, 4, // Front
                        0, 1, 5, 5, 4, 0, // Bottom
                        2, 3, 7, 7, 6, 2, // Top
                        0, 3, 7, 7, 4, 0, // Left
                        1, 2, 6, 6, 5, 1  // Right
                    ];

                    const maxVoxels = Math.min(maxVoxelsToShow, points.length);
                    const maxScalar = Math.max(...scalars);

                    // Typed arrays for better performance
                    const positions = new Float32Array(maxVoxels * 8 * 3); // 8 vertices * 3 coords per voxel
                    const colors = new Float32Array(maxVoxels * 8 * 3);    // 8 vertices * 3 color channels per voxel
                    const indices = new Uint32Array(maxVoxels * 36);       // 12 triangles (36 indices) per voxel

                    let positionOffset = 0;
                    let colorOffset = 0;
                    let indexOffset = 0;

                    for (let i = 0; i < maxVoxels; i++) {
                        const centerX = points[i][0];
                        const centerY = points[i][1];
                        const centerZ = points[i][2];

                        // Compute color based on scalar
                        const normalizedScalar = scalars[i] / maxScalar;
                        const color = new THREE.Color();
                        color.setHSL(normalizedScalar * 0.7, 1.0, 0.5);

                        // Add vertices and colors
                        for (let j = 0; j < 8; j++) {
                            const [dx, dy, dz] = cubeVertexOffsets[j];
                            positions[positionOffset++] = centerX + dx;
                            positions[positionOffset++] = centerY + dy;
                            positions[positionOffset++] = centerZ + dz;

                            colors[colorOffset++] = color.r;
                            colors[colorOffset++] = color.g;
                            colors[colorOffset++] = color.b;
                        }

                        // Add indices with offsets
                        for (let j = 0; j < cubeIndices.length; j++) {
                            indices[indexOffset++] = cubeIndices[j] + i * 8;
                        }
                    }

                    // Set attributes and indices in the geometry
                    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
                    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
                    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

                    } catch (error) {
                                  THREE.Cache.clear();
                        if (error instanceof RangeError) {
                            console.error("🚨 Array buffer allocation failed:", error);

                            // Clear existing memory to free up space
                            THREE.Cache.clear();

                            // Optional: Trigger garbage collection (only works in Chrome DevTools)
                            if (window.gc) window.gc();

                            // Return an empty geometry to prevent further errors
                            return new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
                        } else {
                            throw error;  // Re-throw other errors
                        }
                    }

                // Create the material
                const material = new THREE.MeshBasicMaterial({
                    vertexColors: true,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: transparency,
                    wireframe: wireframe
                });

                // Return the voxel mesh
                return new THREE.Mesh(geometry, material);
            }
        </script>
    </body>
    </html>
    """, version=VERSION)

def start_webrtc():
    """Run WebRTC connection in a separate asyncio loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(lidar_webrtc_connection())

if __name__ == "__main__":
    import threading
    if args.csv_read:
        csv_thread = threading.Thread(target=lambda: asyncio.run(read_csv_and_emit(args.csv_read)), daemon=True)
        csv_thread.start()
    else:
        webrtc_thread = threading.Thread(target=start_webrtc, daemon=True)
        webrtc_thread.start()

    socketio.run(app, host="127.0.0.1", port=8080, debug=False)
