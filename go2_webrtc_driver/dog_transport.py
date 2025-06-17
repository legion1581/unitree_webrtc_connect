"""
DogTransport - A Pipecat transport for the Unitree Go2 robot.

This transport integrates with the Go2 robot's WebRTC audio streams,
providing a clean interface for Pipecat pipelines.
"""

import asyncio
import logging
import numpy as np
from typing import Optional, Any, Awaitable, Callable
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

from aiortc import AudioStreamTrack
from av import AudioFrame

logger = logging.getLogger(__name__)


class DogCallbacks(BaseModel):
    """Callback handlers for Dog transport events."""
    on_app_message: Callable[[Any], Awaitable[None]]
    on_client_connected: Callable[[Any], Awaitable[None]]
    on_client_disconnected: Callable[[Any], Awaitable[None]]
    on_client_closed: Callable[[Any], Awaitable[None]]


class DogAudioStreamTrack(AudioStreamTrack):
    """Custom audio stream track for sending audio to the Go2 robot."""
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.audio_queue = asyncio.Queue()
        self._timestamp = 0
        # Go2 expects 48kHz stereo audio
        self.sample_rate = sample_rate
        self.channels = 2
        self.samples_per_frame = 960  # 20ms of audio at 48kHz
        self._recv_count = 0
        logger.info(f"DogAudioStreamTrack initialized with sample_rate={sample_rate}")
        
    async def recv(self):
        """Receive the next audio frame for WebRTC."""
        logger.debug("DogAudioStreamTrack: recv() called")
        
        try:
            # Try to get audio data from queue without blocking too long
            audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.02)  # 20ms timeout
            logger.debug(f"DogAudioStreamTrack: Got audio data of size {len(audio_data)}")
        except asyncio.TimeoutError:
            # Generate silence if no audio data is available
            silence_size = self.samples_per_frame * 2 * 2  # stereo * 2 bytes per sample
            audio_data = bytes(silence_size)
            logger.debug("DogAudioStreamTrack: Generated silence frame")
        
        # Create audio frame
        frame = AudioFrame(format='s16', layout='stereo', samples=self.samples_per_frame)
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        self._timestamp += self.samples_per_frame
        
        # Fill frame with audio data
        frame.planes[0].update(audio_data)
        
        return frame
        
    async def add_audio(self, audio_data: bytes):
        """Add audio data to the queue."""
        await self.audio_queue.put(audio_data)


class DogClient:
    """Client that manages the Go2 WebRTC connection and audio streaming."""
    
    def __init__(self, go2_connection, callbacks: DogCallbacks):
        self._go2_connection = go2_connection
        self._callbacks = callbacks
        self._audio_queue = asyncio.Queue()
        self._audio_track = None
        self._audio_buffer = bytearray()
        self._running = False
        self._params = None
        self._in_sample_rate = None
        self._out_sample_rate = None
        self._connected = False
        
    async def setup(self, params: TransportParams, frame: StartFrame):
        """Setup the client with transport parameters."""
        self._params = params
        self._in_sample_rate = params.audio_in_sample_rate or frame.audio_in_sample_rate or 16000
        # Always use 48kHz for output to the robot, regardless of input
        self._out_sample_rate = 48000  # Go2 robot requires 48kHz
        
        # Only create audio output track if we don't already have one
        if params.audio_out_enabled and not self._audio_track:
            self._audio_track = DogAudioStreamTrack(self._out_sample_rate)
            logger.info(f"DogClient: Created new audio track in setup with {self._out_sample_rate}Hz")
        
        # If we already have an audio track, ensure _out_sample_rate matches
        if self._audio_track:
            self._out_sample_rate = self._audio_track.sample_rate
            logger.info(f"DogClient: Using existing audio track with {self._out_sample_rate}Hz")
                
    async def connect(self):
        """Connect to the Go2 robot (already connected via Go2WebRTCConnection)."""
        if self._connected:
            logger.info("DogClient: Already connected, skipping")
            return
            
        if self._go2_connection.isConnected:
            logger.info("DogClient: Go2 connection established, setting up audio")
            
            # Add the audio track to the peer connection AFTER connection is established
            # This is the key difference from the previous implementation
            if self._audio_track and self._go2_connection.pc:
                logger.info("DogClient: Adding audio track to peer connection")
                self._go2_connection.pc.addTrack(self._audio_track)
                logger.info("DogClient: Audio track added successfully")
                
                # Log transceiver info
                transceivers = self._go2_connection.pc.getTransceivers()
                for i, transceiver in enumerate(transceivers):
                    if transceiver.kind == "audio":
                        logger.info(f"DogClient: Audio transceiver {i}: direction={transceiver.direction}, "
                                  f"sender.track={transceiver.sender.track if transceiver.sender else None}")
            elif not self._audio_track:
                logger.warning("DogClient: No audio track available for output")
            elif not self._go2_connection.pc:
                logger.error("DogClient: No peer connection available")
            
            # Enable audio channels
            self._go2_connection.audio.switchAudioChannel(True)
            # Register audio callback
            self._go2_connection.audio.add_track_callback(self._handle_robot_audio)
            self._running = True
            self._connected = True
            # Emit connected event
            await self._callbacks.on_client_connected(self._go2_connection)
        else:
            logger.error("DogClient: Go2 connection not established")
            
    async def disconnect(self):
        """Disconnect from the Go2 robot."""
        if not self._connected:
            return
        self._running = False
        self._connected = False
        await self._callbacks.on_client_disconnected(self._go2_connection)
        
    async def _handle_robot_audio(self, frame):
        """Handle incoming audio from robot."""
        if not self._running:
            return
            
        try:
            # Convert Go2 audio frame to numpy array
            audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
            
            # Go2 provides 48kHz stereo audio
            # Convert stereo to mono by taking left channel
            if len(audio_data) % 2 == 0:
                stereo_data = audio_data.reshape(-1, 2)
                mono_data = stereo_data[:, 0]
            else:
                mono_data = audio_data
            
            # Downsample from 48kHz to 16kHz for better STT performance
            downsample_factor = 3  # 48000 / 16000 = 3
            downsampled_data = mono_data[::downsample_factor]
            
            # Create Pipecat audio frame
            audio_frame = InputAudioRawFrame(
                audio=downsampled_data.tobytes(),
                sample_rate=16000,
                num_channels=1
            )
            
            # Queue the frame for processing
            await self._audio_queue.put(audio_frame)
            
        except Exception as e:
            logger.error(f"DogClient: Error processing robot audio: {e}")
            
    async def read_audio_frame(self):
        """Generator that yields audio frames from the robot."""
        while self._running:
            try:
                audio_frame = await self._audio_queue.get()
                yield audio_frame
            except Exception as e:
                logger.error(f"DogClient: Error reading audio frame: {e}")
                await asyncio.sleep(0.01)
                
    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write audio frame to the robot."""
        if not self._running:
            logger.warning("DogClient: Not running, skipping audio frame")
            return
        if not self._audio_track:
            logger.warning("DogClient: No audio track, skipping audio frame")
            return
            
        try:
            # Get audio data from frame
            input_audio = np.frombuffer(frame.audio, dtype=np.int16)
            logger.debug(f"DogClient: Received audio frame - samples: {len(input_audio)}, rate: {frame.sample_rate}")
            
            # Calculate upsampling factor
            input_rate = frame.sample_rate or 16000
            output_rate = self._out_sample_rate or 48000
            upsample_factor = output_rate // input_rate
            
            # Upsample audio (simple repetition)
            if upsample_factor > 1:
                upsampled = np.repeat(input_audio, upsample_factor)
            else:
                upsampled = input_audio
                
            # Convert mono to stereo by duplicating the channel
            stereo_audio = np.stack([upsampled, upsampled], axis=1)
            
            # Ensure we have the right amount of data for a frame
            # samples_per_frame * 2 channels * 2 bytes per sample
            frame_size_bytes = self._audio_track.samples_per_frame * 2 * 2
            
            # Add to buffer
            self._audio_buffer.extend(stereo_audio.flatten().astype(np.int16).tobytes())
            
            # Send complete frames
            frames_sent = 0
            while len(self._audio_buffer) >= frame_size_bytes:
                frame_data = bytes(self._audio_buffer[:frame_size_bytes])
                self._audio_buffer = self._audio_buffer[frame_size_bytes:]
                
                # Send to audio track
                await self._audio_track.add_audio(frame_data)
                frames_sent += 1
                
            if frames_sent > 0:
                logger.info(f"DogClient: Sent {frames_sent} audio frames to track, buffer remaining: {len(self._audio_buffer)} bytes")
            
            # Log if we're accumulating too much in the buffer
            if len(self._audio_buffer) > frame_size_bytes * 2:
                logger.warning(f"DogClient: Audio buffer growing large: {len(self._audio_buffer)} bytes")
                
        except Exception as e:
            logger.error(f"DogClient: Error writing audio frame: {e}", exc_info=True)
            
    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        """Send a message through the data channel if available."""
        if self._running and hasattr(self._go2_connection, 'datachannel'):
            self._go2_connection.datachannel.send_message(frame.message)
            
    @property
    def is_connected(self) -> bool:
        return self._go2_connection.isConnected


class DogInputTransport(BaseInputTransport):
    """Input transport for receiving audio from the Go2 robot."""
    
    def __init__(self, client: DogClient, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None
        self._initialized = False
        
    async def start(self, frame: StartFrame):
        """Start the input transport."""
        await super().start(frame)
        
        if self._initialized:
            return
            
        self._initialized = True
        
        await self._client.setup(self._params, frame)
        await self._client.connect()
        
        if not self._receive_audio_task and self._params.audio_in_enabled:
            self._receive_audio_task = self.create_task(self._receive_audio())
            
        await self.set_transport_ready(frame)
        
    async def stop(self, frame: EndFrame):
        """Stop the input transport."""
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()
        
    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport."""
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()
        
    async def _stop_tasks(self):
        """Stop all running tasks."""
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
            
    async def _receive_audio(self):
        """Receive audio from the robot."""
        try:
            # Wait a bit to ensure pipeline is ready
            await asyncio.sleep(0.5)
            
            async for audio_frame in self._client.read_audio_frame():
                if audio_frame:
                    await self.push_audio_frame(audio_frame)
                    
        except Exception as e:
            logger.error(f"DogInputTransport: Exception receiving audio: {e}")
            
    async def push_app_message(self, message: Any):
        """Push an app message frame."""
        frame = TransportMessageUrgentFrame(message=message)
        await self.push_frame(frame)


class DogOutputTransport(BaseOutputTransport):
    """Output transport for sending audio to the Go2 robot."""
    
    def __init__(self, client: DogClient, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._initialized = False
        
    async def start(self, frame: StartFrame):
        """Start the output transport."""
        await super().start(frame)
        
        if self._initialized:
            return
            
        self._initialized = True
        
        await self._client.setup(self._params, frame)
        await self._client.connect()
        await self.set_transport_ready(frame)
        
    async def stop(self, frame: EndFrame):
        """Stop the output transport."""
        await super().stop(frame)
        await self._client.disconnect()
        
    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport."""
        await super().cancel(frame)
        await self._client.disconnect()
        
    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        """Send a message through the transport."""
        await self._client.send_message(frame)
        
    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write an audio frame to the robot."""
        logger.debug(f"DogOutputTransport: Writing audio frame")
        await self._client.write_audio_frame(frame)


class DogTransport(BaseTransport):
    """
    Transport for the Unitree Go2 robot that integrates with Pipecat pipelines.
    
    This transport handles bidirectional audio streaming between the robot
    and Pipecat, with automatic format conversion and event handling.
    """
    
    def __init__(
        self,
        go2_connection,
        params: TransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """
        Initialize the DogTransport.
        
        Args:
            go2_connection: An active Go2WebRTCConnection instance
            params: Transport parameters (audio settings, VAD, etc.)
            input_name: Optional name for the input transport
            output_name: Optional name for the output transport
        """
        super().__init__(input_name=input_name, output_name=output_name)
        
        self._params = params
        self._go2_connection = go2_connection
        
        self._callbacks = DogCallbacks(
            on_app_message=self._on_app_message,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_client_closed=self._on_client_closed,
        )
        
        # Create client without audio track (will be created after connection)
        self._client = DogClient(go2_connection, self._callbacks)
        
        self._input: Optional[DogInputTransport] = None
        self._output: Optional[DogOutputTransport] = None
        
        # Register supported handlers
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_client_closed")
        
    def input(self) -> DogInputTransport:
        """Get the input transport."""
        if not self._input:
            self._input = DogInputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._input
        
    def output(self) -> DogOutputTransport:
        """Get the output transport."""
        if not self._output:
            self._output = DogOutputTransport(
                self._client, self._params, name=self._output_name
            )
        return self._output
        
    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send audio frame to robot."""
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)
            
    async def send_image(self, frame: Frame):
        """Send image frame - not implemented for audio-only transport."""
        logger.warning("DogTransport: send_image called but not implemented for audio-only transport")
        
    async def _on_app_message(self, message: Any):
        """Handle app message."""
        if self._input:
            await self._input.push_app_message(message)
        await self._call_event_handler("on_app_message", message)
        
    async def _on_client_connected(self, connection):
        """Handle client connected event."""
        await self._call_event_handler("on_client_connected", self, connection)
        
    async def _on_client_disconnected(self, connection):
        """Handle client disconnected event."""
        await self._call_event_handler("on_client_disconnected", self, connection)
        
    async def _on_client_closed(self, connection):
        """Handle client closed event."""
        await self._call_event_handler("on_client_closed", self, connection)
