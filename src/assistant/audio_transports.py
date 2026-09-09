import asyncio
import logging
from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd
from reachy_mini import ReachyMini

from .utils import AudioChunk, fp32_to_int16, int16_to_fp32

logger = logging.getLogger(__name__)


class AudioTransport(ABC):
    @abstractmethod
    async def start(self) -> None: ...
    @abstractmethod
    async def close(self) -> None: ...
    @abstractmethod
    async def pull_from_mic(self) -> AudioChunk | None: ...
    @abstractmethod
    async def push_to_speaker(self, audio_chunk: AudioChunk, /) -> None: ...


class LocalAudioTransport(AudioTransport):
    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_length_s: float = 0.04,
        prebuffer_target_chunks: int = 3,
        allow_interruptions: bool = False,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_size = int(self._sample_rate * chunk_length_s)
        self._allow_interruptions = allow_interruptions

        # Playback queue and state
        self._output_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._current_audio_chunk: AudioChunk | None = None
        self._chunk_position = 0

        # Jitter buffer
        self._prebuffering = True
        self._prebuffer_target_chunks = prebuffer_target_chunks

        # Interrupt handling
        self._interrupt_event = asyncio.Event()

        # Open streams
        self._audio_stream = sd.InputStream(
            channels=1,
            samplerate=self._sample_rate,
            dtype=np.int16,
        )
        self._audio_player = sd.OutputStream(
            channels=1,
            samplerate=self._sample_rate,
            dtype=np.int16,
            callback=self._output_callback,
            blocksize=self._chunk_size,
        )

    async def start(self) -> None:
        self._audio_stream.start()
        self._audio_player.start()

    async def pull_from_mic(self) -> AudioChunk | None:
        if not self._allow_interruptions and self._is_playing():
            return None  # assistant speaking

        if self._audio_stream.read_available < self._chunk_size:
            return None

        data, _ = self._audio_stream.read(self._chunk_size)
        arr = np.asarray(data).reshape(-1).astype(np.int16)
        return AudioChunk(samples=arr, sample_rate=self._sample_rate)

    async def push_to_speaker(self, audio_chunk: AudioChunk, /) -> None:
        self._output_queue.put_nowait(audio_chunk)

    def _is_playing(self) -> bool:
        return self._current_audio_chunk is not None or not self._output_queue.empty()

    def _output_callback(self, outdata, frames: int, time, status) -> None:
        if status:
            print(f"Output callback status: {status}")

        # Handle interrupt: simply clear output and reset if set
        if self._interrupt_event.is_set():
            outdata.fill(0)
            if self._current_audio_chunk is None:
                while not self._output_queue.empty():
                    try:
                        self._output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self._prebuffering = True
                # clear the asyncio.Event by scheduling from main loop
                self._interrupt_event.clear()
                return

        outdata.fill(0)
        samples_filled = 0

        while samples_filled < len(outdata):
            if self._current_audio_chunk is None:
                try:
                    if (
                        self._prebuffering
                        and self._output_queue.qsize() < self._prebuffer_target_chunks
                    ):
                        break
                    self._prebuffering = False
                    self._current_audio_chunk = self._output_queue.get_nowait()
                    self._chunk_position = 0
                except asyncio.QueueEmpty:
                    break

            samples = self._current_audio_chunk.samples
            remaining_output = len(outdata) - samples_filled
            remaining_chunk = len(samples) - self._chunk_position
            samples_to_copy = min(remaining_output, remaining_chunk)

            if samples_to_copy > 0:
                chunk_data = samples[
                    self._chunk_position : self._chunk_position + samples_to_copy
                ]
                outdata[samples_filled : samples_filled + samples_to_copy, 0] = (
                    chunk_data
                )
                samples_filled += samples_to_copy
                self._chunk_position += samples_to_copy

                if self._chunk_position >= len(samples):
                    self._current_audio_chunk = None
                    self._chunk_position = 0

    async def close(self) -> None:
        self._audio_stream.stop()
        self._audio_stream.close()
        self._audio_player.stop()
        self._audio_player.close()


class ReachyAudioTransport(AudioTransport):
    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_length_s: float = 0.04,
        prebuffer_target_chunks: int = 3,
        allow_interruptions: bool = False,
    ) -> None:
        self._reachy = ReachyMini()

        self._input_sample_rate = self._reachy.media.get_input_audio_samplerate()
        self._output_sample_rate = self._reachy.media.get_output_audio_samplerate()

        self._sample_rate = sample_rate
        self._chunk_size = int(self._sample_rate * chunk_length_s)
        self._allow_interruptions = allow_interruptions

        # Playback queue and state
        self._output_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._current_audio_chunk: AudioChunk | None = None
        self._chunk_position = 0

        # Jitter buffer
        self._prebuffering = True
        self._prebuffer_target_chunks = prebuffer_target_chunks

        # Interrupt handling
        self._interrupt_event = asyncio.Event()

    async def start(self) -> None:
        self._reachy.wake_up()
        self._reachy.media.start_recording()
        self._reachy.media.start_playing()
        self._reachy.enable_wobbling()

    async def close(self) -> None:
        self._reachy.media.stop_recording()
        self._reachy.media.stop_playing()
        self._reachy.goto_sleep()
        self._reachy.disable_wobbling()

    async def pull_from_mic(self) -> AudioChunk | None:
        if not self._allow_interruptions and self._is_playing():
            return  # assistant speaking

        fp32_stereo = self._reachy.media.get_audio_sample()
        if fp32_stereo is None:
            logger.warning("ReachyAudioManager: no audio sample available from Reachy")
            return

        fp32_mono = fp32_stereo.mean(axis=1)
        int16_mono = fp32_to_int16(fp32_mono)

        return AudioChunk(
            samples=int16_mono,
            sample_rate=self._input_sample_rate,
        )

    async def push_to_speaker(self, audio_chunk: AudioChunk, /) -> None:
        resampled = audio_chunk.resample(self._output_sample_rate)
        fp32_samples = int16_to_fp32(resampled.samples)
        self._reachy.media.push_audio_sample(fp32_samples)

    def _is_playing(self) -> bool:
        return self._current_audio_chunk is not None or not self._output_queue.empty()

    def _handle_user_turn(self) -> None:
        self._reachy.media.stop_playing()
        self._reachy.media.start_recording()

    def _handle_model_turn(self) -> None:
        self._reachy.media.stop_recording()
        self._reachy.media.start_playing()
