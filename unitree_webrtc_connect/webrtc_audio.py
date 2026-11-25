
from aiortc import AudioStreamTrack, MediaStreamTrack, RTCRtpSender
import asyncio
import fractions
import logging
import time
import av
import sounddevice as sd
import numpy as np
import wave


class WebRTCAudioChannel:
    def __init__(self, pc, datachannel) -> None:
        self.pc = pc
        self.pc.addTransceiver("audio", direction="sendrecv")
        self.datachannel = datachannel

        # List to hold multiple callbacks
        self.track_callbacks = []
        
    async def frame_handler(self, frame):
        logging.info("Receiving audio frame")

        # Trigger all registered callbacks
        for callback in self.track_callbacks:
            try:
                # Call each callback function and pass the track
                await callback(frame)
            except Exception as e:
                logging.error(f"Error in callback {callback}: {e}")
    
    def add_track_callback(self, callback):
        """
        Adds a callback to be triggered when an audio track is received.
        """
        if callable(callback):
            self.track_callbacks.append(callback)
        else:
            logging.warning(f"Callback {callback} is not callable.")  

    def switchAudioChannel(self, switch: bool):
        self.datachannel.switchAudioChannel(switch)


class QueuedAudioTrack(MediaStreamTrack):
    """Outgoing audio track that plays files one after another on the robot
    speaker over a single WebRTC connection.

    Why not aiortc's MediaPlayer: the robot's audio_hub feeds the incoming
    Opus frames into a GStreamer pipeline whose clock keeps running while no
    audio arrives. When a stream starts after an idle period (a new
    connection, or a new MediaPlayer after the previous file ended), the
    robot discards roughly the first `idle time` of audio, up to ~8 s, so
    short clips come out silent (#28). This track avoids that by:

    * sending `warmup` seconds of silence first, which absorbs the discarded
      lead-in of a new connection, and
    * never stopping: when the queue is empty it sends silence, so files
      queued later play immediately and completely.

    Usage::

        player = QueuedAudioTrack()
        conn.pc.addTrack(player)
        await player.play("bark.wav")      # returns when the file was sent
        await player.play("song.mp3")
        await conn.disconnect()
    """

    kind = "audio"

    SAMPLE_RATE = 48000
    SAMPLES_PER_FRAME = 960          # 20 ms, the Opus frame size
    _BYTES_PER_SAMPLE = 4            # s16 stereo

    def __init__(self, warmup: float = 9.0):
        super().__init__()
        self._buf = bytearray(int(warmup * self.SAMPLE_RATE) * self._BYTES_PER_SAMPLE)
        self._sent = 0               # bytes handed to the encoder so far
        self._queued = len(self._buf)
        self._waiters = []           # (end offset in bytes, future)
        self._pts = 0
        self._start = None

    @classmethod
    def decode(cls, path: str) -> bytes:
        """Decode any audio file PyAV/FFmpeg can read to 48 kHz s16 stereo PCM."""
        pcm = bytearray()
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=cls.SAMPLE_RATE)
        with av.open(path) as container:
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    pcm += bytes(out.planes[0])[: out.samples * cls._BYTES_PER_SAMPLE]
        for out in resampler.resample(None):
            pcm += bytes(out.planes[0])[: out.samples * cls._BYTES_PER_SAMPLE]
        return bytes(pcm)

    def enqueue_pcm(self, pcm: bytes) -> "asyncio.Future":
        """Queue raw 48 kHz s16 stereo PCM. The returned future resolves once
        the last sample has been sent."""
        self._buf += pcm
        self._queued += len(pcm)
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append((self._queued, fut))
        return fut

    async def play(self, path: str) -> None:
        """Queue an audio file (mp3, wav, ...) and wait until it has been sent."""
        pcm = await asyncio.get_event_loop().run_in_executor(None, self.decode, path)
        await self.enqueue_pcm(pcm)

    def clear(self) -> None:
        """Drop everything still queued (stop the current file)."""
        self._queued -= len(self._buf)
        self._buf.clear()

    async def recv(self):
        if self._start is None:
            self._start = time.monotonic()
        wait = self._start + self._pts / self.SAMPLE_RATE - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)

        n = self.SAMPLES_PER_FRAME * self._BYTES_PER_SAMPLE
        chunk = bytes(self._buf[:n])
        del self._buf[:n]
        self._sent += len(chunk)
        while self._waiters and (self._waiters[0][0] <= self._sent or not self._buf):
            _, fut = self._waiters.pop(0)
            if not fut.done():
                fut.set_result(None)

        frame = av.AudioFrame(format="s16", layout="stereo", samples=self.SAMPLES_PER_FRAME)
        frame.planes[0].update(chunk + b"\0" * (n - len(chunk)))
        frame.sample_rate = self.SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.SAMPLE_RATE)
        self._pts += self.SAMPLES_PER_FRAME
        return frame
