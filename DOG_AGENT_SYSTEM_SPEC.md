# Dog Agent System Specification

## Overview

The `dog-agent.py` file implements a voice-controlled AI agent for the Unitree Go2 robot using the Pipecat AI framework. It creates a bidirectional audio bridge between the robot's WebRTC audio streams and conversational AI services, enabling natural voice interactions with the robot.

## Architecture

### Core Components

1. **WebRTC Connection** (`Go2WebRTCConnection`)
   - Establishes connection to the Go2 robot via serial number or IP address
   - Manages audio channels for bidirectional communication
   - Uses the `go2_webrtc_driver` library for low-level robot communication

2. **Audio Pipeline**
   - **Input**: Robot microphone → WebRTC → Audio processing → STT → LLM
   - **Output**: LLM → TTS → Audio processing → WebRTC → Robot speakers
   - Built using Pipecat's pipeline architecture for modular audio processing

3. **AI Services**
   - **STT (Speech-to-Text)**: Deepgram for converting speech to text
   - **LLM (Language Model)**: OpenAI GPT-4o-mini for conversation
   - **TTS (Text-to-Speech)**: Cartesia for converting text to speech

### Custom Audio Processors

#### Go2AudioInput (FrameProcessor)
- **Purpose**: Handles incoming audio from the robot's microphone
- **Key Features**:
  - Converts WebRTC audio frames to Pipecat-compatible frames
  - Performs audio format conversion (48kHz stereo → 16kHz mono)
  - Uses async queue for thread-safe audio frame handling
  - Decorated with `@weave.op()` for tracing audio processing

**Audio Processing Flow**:
1. Receives 48kHz stereo audio from robot via WebRTC callback
2. Converts to numpy array and extracts left channel for mono
3. Downsamples to 16kHz (optimal for STT)
4. Creates `InputAudioRawFrame` and queues for pipeline processing

#### Go2AudioOutput (FrameProcessor)
- **Purpose**: Handles outgoing audio to robot speakers
- **Status**: Partially implemented (TODO: WebRTC audio sending)
- **Design**: Receives `OutputAudioRawFrame` from TTS and should convert/send to robot

## Weave Integration

### Tracing Implementation

1. **Initialization**
   ```python
   weave.init(project_name="dog-agent")
   ```
   - Creates a Weave project for tracking all operations
   - Enables automatic logging of decorated functions

2. **Audio Tracking**
   - `@weave.op()` decorator on `save_audio()` function
   - Saves audio data as WAV files that Weave can display
   - Three types of audio tracked:
     - **Full**: Complete conversation audio
     - **User**: User speech segments only
     - **Bot**: Bot speech segments only

3. **Function Tracing**
   - `main()`: Tracks entire session lifecycle
   - `_handle_robot_audio()`: Tracks individual audio frame processing
   - Provides performance metrics and debugging insights

### AudioBufferProcessor Integration

The `AudioBufferProcessor` from Pipecat is configured with `enable_turn_audio=True` to capture conversation turns separately. Event handlers are registered for:

- `on_audio_data`: Triggered for all audio passing through
- `on_user_turn_audio_data`: Triggered when user is speaking
- `on_bot_turn_audio_data`: Triggered when bot is speaking

Each handler calls `save_audio()` which creates a WAV file buffer that Weave tracks and displays in its UI.

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
│ Go2AudioInput   │────▶│ STT         │────▶│ Context     │
│ (Robot Mic)     │     │ (Deepgram)  │     │ Aggregator  │
└─────────────────┘     └─────────────┘     └─────────────┘
                                                    │
                                                    ▼
┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
│ Go2AudioOutput  │◀────│ AudioBuffer │◀────│ TTS         │
│ (Robot Speaker) │     │ Processor   │     │ (Cartesia)  │
└─────────────────┘     └─────────────┘     └─────────────┘
                                                    ▲
                                                    │
                                            ┌─────────────┐
                                            │ LLM         │
                                            │ (OpenAI)    │
                                            └─────────────┘
```

## Configuration

### Environment Variables Required
- `OPENAI_API_KEY`: OpenAI API key for LLM
- `CARTESIA_API_KEY`: Cartesia API key for TTS
- `DEEPGRAM_API_KEY`: Deepgram API key for STT
- `GO2_SERIAL_NUMBER` or `GO2_ROBOT_IP`: Robot connection identifier

### Audio Configuration
- **Input Sample Rate**: 48kHz (robot native) → 16kHz (STT optimized)
- **Channels**: Stereo input converted to mono
- **Frame Size**: 1024 samples per buffer

## Key Implementation Details

### Thread Safety
- Uses `asyncio.Queue` for thread-safe audio frame passing between WebRTC callback and async pipeline
- Proper task lifecycle management with cancellation handling

### Audio Format Conversion
```python
# Stereo to mono conversion
stereo_data = audio_data.reshape(-1, 2)
mono_data = stereo_data[:, 0]  # Take left channel

# Downsampling
downsample_factor = 3  # 48000 / 16000
downsampled_data = mono_data[::downsample_factor]
```

### Pipeline Parameters
- `allow_interruptions=True`: Allows user to interrupt bot mid-speech
- `enable_metrics=True`: Tracks performance metrics
- `enable_usage_metrics=True`: Tracks API usage

## Future Development Considerations

### Pending Implementation
1. **Audio Output to Robot**: The `Go2AudioOutput` processor needs WebRTC audio sending implementation
2. **VAD Integration**: Voice Activity Detection is initialized but commented out in pipeline
3. **Error Recovery**: Add reconnection logic for WebRTC disconnections

### Potential Enhancements
1. **Audio Quality**: Implement proper resampling instead of simple decimation
2. **Stereo Processing**: Consider using both channels or proper stereo-to-mono mixing
3. **Buffering**: Add configurable audio buffering for network latency compensation
4. **Metrics**: Expose Weave metrics for audio latency, processing time, etc.

### Integration Points
- **WebRTC Driver**: Uses `go2_webrtc_driver` for robot communication
- **Pipecat Framework**: Leverages pipeline architecture for modular processing
- **Weave Tracking**: All audio data and operations are traced for analysis

## Debugging Tips

1. **Audio Issues**:
   - Check Weave traces for audio frame processing
   - Verify sample rates and channel counts in logs
   - Use saved audio files to debug quality issues

2. **Connection Issues**:
   - Ensure robot is on same network
   - Verify serial number or IP address
   - Check WebRTC connection logs

3. **Pipeline Issues**:
   - Enable debug logging for frame flow visualization
   - Check Weave for operation timing and errors
   - Verify all API keys are set correctly

## Dependencies

- `go2_webrtc_driver`: Robot communication
- `pipecat`: Audio pipeline framework
- `weave`: Operation tracking and visualization
- `numpy`: Audio data manipulation
- `asyncio`: Asynchronous programming
- Various AI service SDKs (OpenAI, Deepgram, Cartesia)
