import asyncio
import logging

import whisper

from ..utils import AudioChunk, int16_to_fp32
from .base import ChatClient

logger = logging.getLogger(__name__)


class WakePhraseDetector(ChatClient):
    _SAMPLE_RATE = 16000  # whisper requries 16khz sr

    def __init__(
        self,
        wake_phrases: list[str],
        max_buffer_duration: float = 5.0,
    ) -> None:
        self._wake_phrases = wake_phrases
        self._max_buffer_duration = max_buffer_duration

        self._transcriber = whisper.load_model("tiny.en")
        self._audio_buffer: AudioChunk | None = None

    async def run(self) -> None:
        # reset buffer
        self._audio_buffer = None

        while True:
            await asyncio.sleep(0.2)  # don't run whisper too often since it's slow
            if detected_wake_phrase := await self._check_buffer_for_wake_word():
                logger.info(f"detected wake word: '{detected_wake_phrase}'")
                return

    async def handle_user_audio(self, audio_chunk: AudioChunk, /) -> None:
        resampled = audio_chunk.resample(self._SAMPLE_RATE)
        if self._audio_buffer is None:
            self._audio_buffer = resampled
        else:
            self._audio_buffer += resampled

        # trim start of buffer
        buffer_excess = self._audio_buffer.duration() - self._max_buffer_duration
        if buffer_excess > 0:
            self._audio_buffer = self._audio_buffer.slice_duration(
                start_time=buffer_excess
            )

    async def _check_buffer_for_wake_word(self) -> str | None:
        if not self._audio_buffer or self._audio_buffer.duration() == 0:
            logger.debug("buffer is empty, skipping transcription")
            return None

        fp32_samples = int16_to_fp32(self._audio_buffer.samples)

        # transcribe fn is sync, so run in a thread to prevent blocking
        output = await asyncio.to_thread(
            self._transcriber.transcribe,
            audio=fp32_samples,
        )
        transcription = output.get("text")
        if not transcription or not isinstance(transcription, str):
            return None

        logger.info(f"user said: {transcription}")
        for wp in self._wake_phrases:
            if wp in transcription.lower():
                return wp

        return None
