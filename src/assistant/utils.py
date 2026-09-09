import numpy as np
import samplerate
from pydantic import BaseModel
from pydantic_numpy.typing import Np1DArrayFp32, Np1DArrayInt16


class AudioChunk(BaseModel):
    samples: Np1DArrayInt16
    sample_rate: int

    def rms(self) -> float:
        if self.samples.size == 0:
            return False
        fp32_samples = int16_to_fp32(self.samples)
        return float(np.sqrt(np.mean(fp32_samples**2)))

    def resample(self, new_sample_rate: int) -> "AudioChunk":
        if new_sample_rate == self.sample_rate:
            return self.clone()

        ratio = new_sample_rate / self.sample_rate
        fp32_old = int16_to_fp32(self.samples)
        fp32_new = samplerate.resample(fp32_old, ratio, "sinc_fastest")
        int16_new = fp32_to_int16(fp32_new)
        return AudioChunk(
            samples=int16_new,
            sample_rate=new_sample_rate,
        )

    def slice_samples(self, start_sample_idx: int, end_sample_idx: int) -> "AudioChunk":
        return AudioChunk(
            samples=self.samples[start_sample_idx:end_sample_idx],
            sample_rate=self.sample_rate,
        )

    def slice_duration(self, start_time: float, end_time: float) -> "AudioChunk":
        start = int(start_time * self.sample_rate)
        end = int(end_time * self.sample_rate)
        return self.slice_samples(start, end)

    @classmethod
    def empty(cls, *, sample_rate: int) -> "AudioChunk":
        return AudioChunk(samples=np.array([], dtype=np.int16), sample_rate=sample_rate)

    def __add__(self, other: "AudioChunk") -> "AudioChunk":
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rates must match to concatenate AudioFrames")
        if not (self.samples.shape and self.samples.shape[0] > 0):
            return other
        if not (other.samples.shape and other.samples.shape[0] > 0):
            return self
        new_samples = np.concat([self.samples, other.samples])
        return AudioChunk(samples=new_samples, sample_rate=self.sample_rate)

    def duration(self) -> float:
        return len(self.samples) * self.sample_rate

    def clone(self) -> "AudioChunk":
        return AudioChunk(samples=self.samples.copy(), sample_rate=self.sample_rate)


def fp32_to_int16(samples: Np1DArrayFp32, /) -> Np1DArrayInt16:
    int16 = samples.copy()
    int16 *= 32767.0
    int16 = int16.clip(-32768.0, 32767.0)
    return int16.astype(np.int16)


def int16_to_fp32(samples: Np1DArrayInt16, /) -> Np1DArrayFp32:
    fp32 = samples.copy()
    fp32 = fp32.astype(np.float32)
    return fp32 / 32768.0
