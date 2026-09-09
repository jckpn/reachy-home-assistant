import asyncio
import logging

import whisper

from .chat_clients import ChatClient
from .utils import AudioChunk, int16_to_fp32

logger = logging.getLogger(__name__)


class WakeWordDetector(ChatClient):
    _SAMPLE_RATE = 16000  # whisper requries 16khz sr

    _audio_buffer: AudioChunk
    _detected: asyncio.Event

    def __init__(
        self,
        wake_phrase: str = "hello",
        max_buffer_duration: float = 10.0,
    ) -> None:
        self._wake_phrase = wake_phrase
        self._max_buffer_duration = max_buffer_duration

        self._transcriber = whisper.load_model("tiny.en")
        self._loop_task: asyncio.Task | None = None

        self._reset()

    def _reset(self) -> None:
        self._detected = asyncio.Event()
        self._audio_buffer = AudioChunk.empty(sample_rate=self._SAMPLE_RATE)

    async def start(self) -> None:
        self._reset()
        self._loop_task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()

    async def _loop(self) -> None:
        while not self._detected.is_set():
            await self._check_buffer()
            await asyncio.sleep(0.1)  # don't run whisper too often since it's slow

        logger.info("wake word detected!")

    async def push_user_audio(self, audio_chunk: AudioChunk, /) -> None:
        if self._detected.is_set():
            logger.warning("still pushing audio after detection!")
            return

        resampled = audio_chunk.resample(self._SAMPLE_RATE)
        self._audio_buffer += resampled

        if self._audio_buffer.duration() > self._max_buffer_duration:
            self._audio_buffer = self._audio_buffer.slice_duration(
                0, self._max_buffer_duration
            )

    async def _check_buffer(self) -> None:
        if self._audio_buffer.duration() == 0:
            logger.warning("buffer is empty, skipping transcription")
            return

        fp32_samples = int16_to_fp32(self._audio_buffer.samples)
        # run in thread to avoid blocking the event loop since original fn is sync
        res = await asyncio.to_thread(self._transcriber.transcribe, audio=fp32_samples)
        transcription = res.get("text")
        if isinstance(transcription, str) and transcription != "":
            logger.info(f"user said: {transcription}")
            if self._wake_phrase in transcription.lower():
                self._detected.set()
        else:
            logger.info(f"no transcription found: {res}")

    async def pull_response_audio(self) -> AudioChunk | None: ...
