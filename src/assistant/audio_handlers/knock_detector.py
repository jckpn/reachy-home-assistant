import asyncio
import logging

from ..utils import AudioChunk, int16_to_fp32
from .base import AudioHandler

logger = logging.getLogger(__name__)


class KnockDetector(AudioHandler):
    _SAMPLE_RATE = 16000  # whisper requries 16khz sr

    def __init__(
        self,
        knock_volume_threshold: float = 0.8,
        max_buffer_duration: float = 5.0,
    ) -> None:
        self._knock_volume_threshold = knock_volume_threshold
        self._max_buffer_duration = max_buffer_duration

        self._audio_buffer: AudioChunk | None = None

    async def run(self) -> None:
        # reset buffer
        self._audio_buffer = None

        while True:
            await asyncio.sleep(0.1)
            if await self._check_buffer_for_knocks():
                logger.info("detected knocking sound!")
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

    async def _check_buffer_for_knocks(self) -> bool:
        if not self._audio_buffer or self._audio_buffer.duration() == 0:
            logger.debug("buffer is empty, skipping detection")
            return False

        min_peak_duration = 0.001
        min_gap_duration = 0.3

        fp32_samples = int16_to_fp32(self._audio_buffer.samples)

        peak_duration = 0.0
        gap_duration = 0.0
        first_peak_found = False
        first_gap_found = False

        for vol in abs(fp32_samples):
            if vol >= self._knock_volume_threshold:
                peak_duration += 1 / self._audio_buffer.sample_rate
                gap_duration = 0.0
            else:
                if peak_duration >= min_peak_duration:
                    if first_peak_found and first_gap_found:
                        return True
                    first_peak_found = True
                else:
                    gap_duration += 1 / self._audio_buffer.sample_rate
                    peak_duration = 0.0
                    if gap_duration >= min_gap_duration:
                        first_gap_found = True

        print(
            f"{peak_duration=:.4f} {gap_duration=:.2f} {first_peak_found=} {first_gap_found=}"
        )

        return False
