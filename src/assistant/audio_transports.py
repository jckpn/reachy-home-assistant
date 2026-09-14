import logging
from abc import ABC, abstractmethod

from reachy_mini import ReachyMini

from .utils import AudioChunk, fp32_to_int16, int16_to_fp32

logger = logging.getLogger(__name__)


class AudioTransport(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def pull_from_mic(self) -> AudioChunk | None: ...

    @abstractmethod
    def push_to_speaker(self, audio_chunk: AudioChunk, /) -> None: ...

    @abstractmethod
    def stop_playback(self) -> None: ...


class ReachyAudioTransport(AudioTransport):
    def __init__(self, reachy: ReachyMini) -> None:
        self._reachy = reachy

        self._input_sample_rate = reachy.media.get_input_audio_samplerate()
        self._output_sample_rate = reachy.media.get_output_audio_samplerate()

    def start(self) -> None:
        self._reachy.media.start_recording()
        self._reachy.media.start_playing()

    def close(self) -> None:
        self._reachy.media.stop_recording()
        self._reachy.media.stop_playing()

    def pull_from_mic(self) -> AudioChunk | None:
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

    def push_to_speaker(self, audio_chunk: AudioChunk, /) -> None:
        resampled = audio_chunk.resample(self._output_sample_rate)
        fp32_samples = int16_to_fp32(resampled.samples)
        self._reachy.media.push_audio_sample(fp32_samples)

    def _handle_user_turn(self) -> None:
        self._reachy.media.stop_playing()
        self._reachy.media.start_recording()

    def _handle_model_turn(self) -> None:
        self._reachy.media.stop_recording()
        self._reachy.media.start_playing()

    def stop_playback(self) -> None:
        if self._reachy.media.audio:
            self._reachy.media.audio.clear_player()
