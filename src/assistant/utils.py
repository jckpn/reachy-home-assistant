import base64
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import samplerate
import soundfile
from pydantic import BaseModel
from pydantic_numpy.typing import Np1DArrayFp32, Np1DArrayInt16

type ChatRole = Literal["user", "assistant"]
type AssistantAudioEvent = AudioChunk | PlaybackCancelRequest
type UserAudioEvent = AudioChunk


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GREETINGS_DIR = os.path.join(CURRENT_DIR, "greetings")
TRANSCRIPT_LOG_PATH = os.path.expanduser("~/Desktop/charlie_transcripts.txt")


class AudioChunk(BaseModel):
    type: Literal["audio_chunk"] = "audio_chunk"
    role: ChatRole
    samples: Np1DArrayInt16
    sample_rate: int

    def resample(self, new_sample_rate: int, in_place: bool = False) -> AudioChunk:
        if new_sample_rate == self.sample_rate:
            return self if in_place else self.clone()

        ratio = new_sample_rate / self.sample_rate
        fp32_old = int16_to_fp32(self.samples)
        fp32_new = samplerate.resample(fp32_old, ratio, "sinc_fastest")
        int16_new = fp32_to_int16(fp32_new)
        if in_place:
            self.samples = int16_new
            self.sample_rate = new_sample_rate
            return self
        return AudioChunk(
            role=self.role,
            samples=int16_new,
            sample_rate=new_sample_rate,
        )

    def slice_samples(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
        in_place: bool = False,
    ) -> AudioChunk:
        if end_idx is None:
            end_idx = end_idx or len(self.samples)
        if in_place:
            self.samples = self.samples[start_idx:end_idx]
            return self
        return AudioChunk(
            role=self.role,
            samples=self.samples[start_idx:end_idx],
            sample_rate=self.sample_rate,
        )

    def slice_duration(
        self,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> AudioChunk:
        if end_time is None:
            end_time = self.duration()
        start_idx = int(start_time * self.sample_rate)
        end_idx = int(end_time * self.sample_rate)
        return self.slice_samples(start_idx, end_idx)

    def __add__(self, other: AudioChunk) -> AudioChunk:
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rates must match to add AudioChunks")
        if self.role != other.role:
            raise ValueError("Roles must match to add AudioChunks")
        new_samples = np.concat([self.samples, other.samples])
        return AudioChunk(
            role=self.role,
            samples=new_samples,
            sample_rate=self.sample_rate,
        )

    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def clone(self) -> AudioChunk:
        return AudioChunk(
            role=self.role,
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
        )

    def to_base64(self) -> str:
        pcm = self.samples.tobytes()
        return base64.b64encode(pcm).decode("utf-8")

    @classmethod
    def empty(cls, *, role: ChatRole, sample_rate: int) -> AudioChunk:
        return AudioChunk(
            role=role,
            samples=np.array([], dtype=np.int16),
            sample_rate=sample_rate,
        )

    @classmethod
    def from_file(cls, path: str, role: ChatRole = "assistant") -> AudioChunk:
        if path.endswith(".npy"):
            samples = np.load(path)
            return AudioChunk(
                role="assistant",
                samples=samples,
                sample_rate=24000,  # we loaded these in at 24khz in intro_audio_extractor
            )

        samples_2d, sample_rate = soundfile.read(path, dtype="int16")
        samples = samples_2d.mean(axis=1).astype(np.int16)  # convert to mono
        return AudioChunk(
            role=role,
            samples=samples,  # type: ignore
            sample_rate=sample_rate,
        )

    @classmethod
    def from_base64(cls, b64: str, *, role: ChatRole, sample_rate: int) -> AudioChunk:
        pcm = base64.b64decode(b64)
        samples = np.frombuffer(pcm, dtype=np.int16)
        return AudioChunk(role=role, samples=samples, sample_rate=sample_rate)


class PlaybackCancelRequest(BaseModel):
    type: Literal["playback_cancel_request"] = "playback_cancel_request"
    role: ChatRole = "assistant"


def fp32_to_int16(samples: Np1DArrayFp32, /) -> Np1DArrayInt16:
    int16 = samples.copy()
    int16 *= 32767.0
    int16 = int16.clip(-32768.0, 32767.0)
    return int16.astype(np.int16)


def int16_to_fp32(samples: Np1DArrayInt16, /) -> Np1DArrayFp32:
    fp32 = samples.copy()
    fp32 = fp32.astype(np.float32)
    return fp32 / 32768.0


def load_greeting(*, voice: str, idx: int | None = None) -> AudioChunk:
    if idx is None:
        idx = random.randint(1, 10)
    path = os.path.join(GREETINGS_DIR, f"{voice}_{idx}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    samples = np.load(path)
    return AudioChunk(
        role="assistant",
        samples=samples,
        sample_rate=24000,  # we loaded these in at 24khz in intro_audio_extractor
    )


def get_datetime_str() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def log_transcript(*, role: ChatRole, transcript: str) -> None:
    with open(TRANSCRIPT_LOG_PATH, "a") as f:
        dt_str = get_datetime_str()
        f.write(f"[{dt_str}] {role}: {transcript}\n")


def log_tool_call(*, tool_name: str) -> None:
    with open(TRANSCRIPT_LOG_PATH, "a") as f:
        dt_str = get_datetime_str()
        f.write(f"[{dt_str}] tool call: {tool_name}\n")


def log_chat_started() -> None:
    with open(TRANSCRIPT_LOG_PATH, "a") as f:
        dt_str = get_datetime_str()
        f.write(f"===== NEW CHAT STARTED {dt_str} =====\n")


def log_chat_ended() -> None:
    with open(TRANSCRIPT_LOG_PATH, "a") as f:
        dt_str = get_datetime_str()
        f.write(f"===== CHAT ENDED {dt_str} =====\n\n")
