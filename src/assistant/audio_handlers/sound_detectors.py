import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import whisper
from pydantic import BaseModel

from ..utils import AudioChunk, int16_to_fp32
from .base import AudioHandler

logger = logging.getLogger(__name__)


# different sound detectors
# use separately since each may require a different buffer size/duration,
# and may need to check at different frequencies
# (i.e. music checks should be less often)


class _SoundDetector(AudioHandler, ABC):
    @abstractmethod
    async def _check_for_sound(self, audio_buffer: AudioChunk) -> bool: ...

    def __init__(
        self,
        *,
        max_buffer_duration: float = 5.0,
        delay_between_checks: float = 0.1,
        end_on_detection: bool = True,
    ) -> None:
        self._max_buffer_duration = max_buffer_duration
        self._delay_between_checks = delay_between_checks
        self._end_on_detection = end_on_detection

        self._running = False
        self._audio_buffer: AudioChunk | None = None

    async def _run(self) -> None:
        self._running = True

        while self._running:
            if self._audio_buffer is None or self._audio_buffer.duration() == 0:
                logger.warning("audio buffer is empty")
            elif await self._check_for_sound(self._audio_buffer):
                logger.info("sound detected!")
                if self._end_on_detection:
                    return

            await asyncio.sleep(self._delay_between_checks)

    async def handle_user_audio(self, audio_chunk: AudioChunk, /) -> None:
        if self._audio_buffer is None:
            self._audio_buffer = audio_chunk.clone()

        if audio_chunk.sample_rate != self._audio_buffer.sample_rate:
            raise ValueError(
                "mismatched sample rates - did the audio transport change?"
            )

        self._audio_buffer += audio_chunk

        # trim start of buffer if too long
        buffer_excess = self._audio_buffer.duration() - self._max_buffer_duration
        if buffer_excess > 0:
            self._audio_buffer = self._audio_buffer.slice_duration(
                start_time=buffer_excess
            )

    async def close(self) -> None:
        self._running = False
        self._audio_buffer = None  # reset buffer


class PhraseDetector(_SoundDetector):
    def __init__(
        self,
        wake_phrases: list[str],
        whisper_model: str = "tiny.en",
        max_buffer_duration: float = 2.0,  # depends on length of phrase; shortest is best
        end_on_detection: bool = True,
    ) -> None:
        super().__init__(
            max_buffer_duration=max_buffer_duration,
            end_on_detection=end_on_detection,
        )

        self._wake_phrases = wake_phrases
        self._whisper = whisper.load_model(whisper_model)

    async def _check_for_sound(self, audio_buffer: AudioChunk) -> bool:
        # whisper requires 16khz sample rate
        resampled = audio_buffer.resample(16000)

        fp32_samples = int16_to_fp32(resampled.samples)

        # transcribe fn is sync, so run in a thread to prevent blocking
        output = await asyncio.to_thread(
            self._whisper.transcribe,
            audio=fp32_samples,
        )
        transcription = output.get("text")
        if not transcription or not isinstance(transcription, str):
            return False

        logger.info(f"user said: {transcription}")
        return any(wp.lower() in transcription.lower() for wp in self._wake_phrases)


class DetectedMusicInfo(BaseModel):
    is_music: bool
    bpm: int | None = None
    confidence: float | None = None


class MusicDetector(_SoundDetector):
    def __init__(
        self,
        *,
        music_info_handler: Callable[[DetectedMusicInfo], None],
        end_on_detection: bool = False,
    ) -> None:
        super().__init__(delay_between_checks=5.0, end_on_detection=end_on_detection)

        self._music_info_handler = music_info_handler

    async def _check_for_sound(self, audio_buffer: AudioChunk) -> bool:
        music_info = DetectedMusicInfo(is_music=True, bpm=120)
        self._music_info_handler(music_info)
        return music_info.is_music
