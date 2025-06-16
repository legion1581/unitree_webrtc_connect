#!/usr/bin/env python3
"""
Go2 Robot Voice Agent

This script creates a voice-controlled agent for the Unitree Go2 robot using Pipecat AI.
It bridges the robot's WebRTC audio streams with conversational AI capabilities.

Features:
- Bidirectional audio: Robot microphone input and speaker output
- Speech-to-text and text-to-speech processing
- OpenAI LLM for natural conversation
- Real-time audio streaming and processing

Usage:
    python dog-agent.py

Environment Variables Required:
    OPENAI_API_KEY - Your OpenAI API key
    CARTESIA_API_KEY - Your Cartesia API key (for TTS)
    DEEPGRAM_API_KEY - Your Deepgram API key (for STT)
    GO2_SERIAL_NUMBER - Your robot's serial number (or GO2_ROBOT_IP)
"""

import asyncio
import logging
import os
import sys
import numpy as np
import io
import wave
from typing import Optional
from dotenv import load_dotenv
import weave
import fractions
from av import AudioFrame

# Go2 WebRTC imports
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

# aiortc imports for audio streaming
from aiortc import AudioStreamTrack
from aiortc.mediastreams import MediaStreamError

# Pipecat imports
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    Frame,
    StartFrame,
    EndFrame,
    CancelFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService

# Load environment variables
load_dotenv()

# Initialize Weave for audio tracing
weave.init(project_name="dog-agent")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio configuration constants
SAMPLE_RATE = 48000  # Go2 audio sample rate
CHANNELS = 2  # Stereo
FRAMES_PER_BUFFER = 1024


@weave.op()
async def save_audio(audio: bytes, sample_rate: int, num_channels: int, name: str):
    """Save audio data to a buffer for Weave tracking."""
    if len(audio) > 0:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            logger.info(f"Saving {name} audio data ({len(audio)} bytes)")
            buffer.seek(0)
            return wave.open(io.BytesIO(buffer.getvalue()), "rb")
    else:
        logger.debug(f"No {name} audio data to save")


class Go2AudioInput(FrameProcessor):
    """Handles audio input from Go2 robot."""
    
    def __init__(self, go2_connection: Go2WebRTCConnection):
        super().__init__()
        self._go2_connection = go2_connection
        self._running = False
        self._audio_queue = asyncio.Queue()
        self._audio_task = None
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process control frames."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartFrame):
            # Start processing when we receive StartFrame
            await self._start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame) or isinstance(frame, CancelFrame):
            await self._stop()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
            
    async def _start(self):
        """Start receiving audio from robot."""
        if not self._running:
            self._running = True
            self._go2_connection.audio.add_track_callback(self._handle_robot_audio)
            # Create audio processing task
            self._audio_task = self.create_task(self._audio_task_handler())
            logger.info("Started receiving audio from robot")
        
    async def _stop(self):
        """Stop receiving audio."""
        self._running = False
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None
            
    async def _audio_task_handler(self):
        """Process audio frames from the queue."""
        # Wait a bit to ensure StartFrame has propagated through the pipeline
        await asyncio.sleep(0.5)
        
        while self._running:
            try:
                audio_frame = await self._audio_queue.get()
                await self.push_frame(audio_frame)
                self._audio_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audio task handler: {e}")
        
    async def _handle_robot_audio(self, frame):
        """Handle incoming audio from robot microphone."""
        if not self._running:
            return
            
        try:
            # Convert Go2 audio frame to numpy array
            audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
            
            # Convert stereo to mono by taking left channel
            if len(audio_data) % 2 == 0:
                # Reshape to stereo and take left channel
                stereo_data = audio_data.reshape(-1, 2)
                mono_data = stereo_data[:, 0]
            else:
                mono_data = audio_data
            
            # Create Pipecat audio frame (16kHz for better STT performance)
            # Downsample from 48kHz to 16kHz
            downsample_factor = 3  # 48000 / 16000 = 3
            downsampled_data = mono_data[::downsample_factor]
            
            audio_frame = InputAudioRawFrame(
                audio=downsampled_data.tobytes(),
                sample_rate=16000,  # Standard rate for STT
                num_channels=1
            )
            
            # Queue the frame for processing
            await self._audio_queue.put(audio_frame)
            
        except Exception as e:
            logger.error(f"Error processing robot audio input: {e}")


class PipecatAudioStreamTrack(AudioStreamTrack):
    """Custom audio stream track that receives audio from Pipecat and streams to WebRTC."""
    
    def __init__(self):
        super().__init__()
        self.audio_queue = asyncio.Queue()
        self._start_time: Optional[float] = None
        self._timestamp = 0
        # Robot expects 48kHz stereo audio
        self.sample_rate = 48000
        self.channels = 2
        self.samples_per_frame = 960  # 20ms of audio at 48kHz
        
    async def recv(self):
        """Receive the next audio frame."""
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()
            
        # Get audio data from queue
        audio_data = await self.audio_queue.get()
        
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


class Go2AudioOutput(FrameProcessor):
    """Handles audio output to Go2 robot speakers."""
    
    def __init__(self, go2_connection: Go2WebRTCConnection):
        super().__init__()
        self._go2_connection = go2_connection
        self._audio_track = None
        self._audio_buffer = bytearray()
        self._running = False
        
    def setup_audio_track(self):
        """Set up the audio track for output."""
        if not self._audio_track:
            # Create custom audio stream track
            self._audio_track = PipecatAudioStreamTrack()
            logger.info("Audio output track created")
            return self._audio_track
        return None
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and send audio to robot."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartFrame):
            self._running = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame) or isinstance(frame, CancelFrame):
            self._running = False
            await self.push_frame(frame, direction)
        elif isinstance(frame, OutputAudioRawFrame):
            # Handle TTS audio output
            if self._running and self._audio_track:
                try:
                    # Convert audio format from Pipecat to robot format
                    await self._convert_and_send_audio(frame)
                except Exception as e:
                    logger.error(f"Error sending audio to robot: {e}")
            
        await self.push_frame(frame, direction)
        
    async def _convert_and_send_audio(self, frame: OutputAudioRawFrame):
        """Convert Pipecat audio format to robot format and send."""
        if not self._audio_track:
            return
            
        # Get audio data from frame
        input_audio = np.frombuffer(frame.audio, dtype=np.int16)
        
        # Pipecat typically outputs at 16kHz or 24kHz mono
        # Robot expects 48kHz stereo
        
        # Calculate upsampling factor
        input_rate = frame.sample_rate
        output_rate = 48000
        upsample_factor = output_rate // input_rate
        
        # Upsample audio (simple repetition for now)
        if upsample_factor > 1:
            upsampled = np.repeat(input_audio, upsample_factor)
        else:
            upsampled = input_audio
            
        # Convert mono to stereo by duplicating the channel
        stereo_audio = np.stack([upsampled, upsampled], axis=1)
        
        # Ensure we have the right amount of data for a frame
        frame_size = self._audio_track.samples_per_frame * 2  # stereo
        
        # Add to buffer
        self._audio_buffer.extend(stereo_audio.flatten().astype(np.int16).tobytes())
        
        # Send complete frames
        while len(self._audio_buffer) >= frame_size * 2:  # 2 bytes per sample
            frame_data = bytes(self._audio_buffer[:frame_size * 2])
            self._audio_buffer = self._audio_buffer[frame_size * 2:]
            
            # Send to audio track
            await self._audio_track.add_audio(frame_data)


@weave.op()
async def main():
    """Main function to run the voice agent."""
    
    # Check for required environment variables
    required_env_vars = ["OPENAI_API_KEY", "CARTESIA_API_KEY", "DEEPGRAM_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these in your .env file or environment")
        sys.exit(1)
    
    # Configure robot connection using serial number from environment
    serial_number = os.getenv("GO2_SERIAL_NUMBER")
    robot_ip = os.getenv("GO2_ROBOT_IP")
    
    if not serial_number and not robot_ip:
        logger.error("Either GO2_SERIAL_NUMBER or GO2_ROBOT_IP must be set in environment variables")
        sys.exit(1)
    
    # Determine connection method and parameters
    if serial_number:
        # Use serial number for connection (preferred method)
        connection_method = WebRTCConnectionMethod.LocalSTA
        connection_kwargs = {
            "serialNumber": serial_number
        }
        logger.info(f"Connecting to robot with serial number: {serial_number}")
    else:
        # Fallback to IP address
        connection_method = WebRTCConnectionMethod.LocalSTA
        connection_kwargs = {
            "ip": robot_ip
        }
        logger.info(f"Connecting to robot with IP address: {robot_ip}")
    
    # Create Go2 connection
    go2_connection = Go2WebRTCConnection(connection_method, **connection_kwargs)
    
    try:
        # Create audio input/output handlers early so we can set up the audio track
        audio_input = Go2AudioInput(go2_connection)
        audio_output = Go2AudioOutput(go2_connection)
        
        # Set up audio output track before connecting
        output_track = audio_output.setup_audio_track()
        
        # Connect to robot
        logger.info("Connecting to Go2 robot...")
        await go2_connection.connect()
        
        # Add the audio track to the peer connection after connecting
        if output_track and go2_connection.pc:
            go2_connection.pc.addTrack(output_track)
            logger.info("Audio output track added to peer connection")
        
        # Enable audio channels
        go2_connection.audio.switchAudioChannel(True)
        logger.info("Audio channels enabled")
        
        # Initialize services
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        cartesia_key = os.getenv("CARTESIA_API_KEY")
        
        # Type assertions since we already checked these exist
        assert deepgram_key is not None
        assert openai_key is not None
        assert cartesia_key is not None
        
        stt = DeepgramSTTService(
            api_key=deepgram_key
        )
        
        llm = OpenAILLMService(
            api_key=openai_key,
            model="gpt-4o-mini"
        )
        
        tts = CartesiaTTSService(
            api_key=cartesia_key,
            voice_id="c45bc5ec-dc68-4feb-8829-6e6b2748095d"  # Movieman voice
        )
        
        # Set up conversation context
        messages = [
            {
                "role": "system",
                "content": """You are a friendly AI assistant integrated into a Unitree Go2 robot. 
                You can hear through the robot's microphones and speak through its speakers. 
                Keep your responses conversational and brief since they will be spoken aloud. 
                You are here to chat and assist the user. Don't use special characters in your responses 
                since they will be converted to speech. Start by introducing yourself."""
            }
        ]
        
        # Set up conversation context and management
        context = OpenAILLMContext(messages)  # type: ignore
        context_aggregator = llm.create_context_aggregator(context)
        
        # Create audio buffer processor for recording
        audiobuffer = AudioBufferProcessor(enable_turn_audio=True)
        
        # Build the pipeline
        pipeline = Pipeline([
            audio_input,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            audiobuffer,
            audio_output,
            context_aggregator.assistant(),
        ])
        
        # Create and run task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            )
        )
        
        # Set up audio buffer event handlers for Weave tracking
        @audiobuffer.event_handler("on_audio_data")
        @weave.op()
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            await save_audio(audio, sample_rate, num_channels, "full")
        
        @audiobuffer.event_handler("on_user_turn_audio_data")
        @weave.op()
        async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
            logger.info("Recording user turn audio")
            await save_audio(audio, sample_rate, num_channels, "user")
        
        @audiobuffer.event_handler("on_bot_turn_audio_data")
        @weave.op()
        async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
            logger.info("Recording bot turn audio")
            await save_audio(audio, sample_rate, num_channels, "bot")
        
        # Start recording
        await audiobuffer.start_recording()
        
        # Queue initial frame to start the pipeline
        # This will trigger audio_input to start receiving audio
        await task.queue_frame(StartFrame())
        
        # Run the pipeline
        runner = PipelineRunner()
        
        logger.info("Voice agent is running. Speak to the robot!")
        logger.info("Press Ctrl+C to stop...")
        
        await runner.run(task)
        
    except KeyboardInterrupt:
        logger.info("Shutting down voice agent...")
    except Exception as e:
        logger.error(f"Error running voice agent: {e}")
        raise
    finally:
        # Cleanup
        if go2_connection:
            await go2_connection.disconnect()
            logger.info("Disconnected from robot")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nVoice agent stopped by user")
        sys.exit(0)
