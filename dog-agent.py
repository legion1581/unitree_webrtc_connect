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
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Import our custom transport
from go2_webrtc_driver.dog_transport import DogTransport, DogAudioStreamTrack

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
    
    # Create Go2 connection without audio track (will be added after connection)
    go2_connection = Go2WebRTCConnection(connection_method, **connection_kwargs)
    
    try:
        # Connect to robot
        logger.info("Connecting to Go2 robot...")
        await go2_connection.connect()
        
        # Create transport with the Go2 connection
        transport = DogTransport(
            go2_connection=go2_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )
        
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
            transport.input(),              # Receive audio from robot
            stt,                            # Convert speech to text
            context_aggregator.user(),      # Add user messages to context
            llm,                            # Process text with LLM
            tts,                            # Convert text to speech
            audiobuffer,
            transport.output(),             # Send audio responses to robot
            context_aggregator.assistant(), # Add assistant responses to context
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
        
        # Set up transport event handlers
        @transport.event_handler("on_client_connected")
        async def on_client_connected(*args):
            logger.info(f"Robot connected via transport (args: {len(args)})")
            # Queue initial frame to start the pipeline
            await task.queue_frame(StartFrame())
            
        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(*args):
            logger.info("Robot disconnected")
            
        @transport.event_handler("on_client_closed")
        async def on_client_closed(*args):
            logger.info("Robot connection closed")
            await task.cancel()
        
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
