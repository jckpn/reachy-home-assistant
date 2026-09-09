import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from agents.realtime import (
    RealtimeAgent,
    RealtimeRunner,
    RealtimeSession,
    RealtimeSessionEvent,
)
from agents.realtime.model import RealtimeModelConfig
from agents.tool import function_tool

from .utils import AudioChunk

logger = logging.getLogger(__name__)


class ChatClient(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def push_user_audio(self, audio_chunk: AudioChunk, /) -> None: ...

    @abstractmethod
    async def pull_response_audio(self) -> AudioChunk | None: ...

    def wants_close(self) -> bool: ...


class OpenAIChatClient(ChatClient):
    _SAMPLE_RATE = 24000

    def __init__(
        self,
        voice: str = "ash",
        talking_speed: float = 1.0,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
    ) -> None:
        self._voice = voice
        self._speed = talking_speed
        self._system_prompt = system_prompt
        self._tools = tools

        self._session: RealtimeSession | None = None
        self._assistant_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._run_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        print("RealtimeAudioChatHandler: starting runner")

        tool_schemas = [function_tool(t) for t in self._tools] if self._tools else []
        agent = RealtimeAgent(
            instructions=self._system_prompt,
            name="Assistant",
            tools=tool_schemas,  # type: ignore
        )

        self._run_task = asyncio.create_task(self._run(agent))

    async def _run(self, agent: RealtimeAgent) -> None:
        runner = RealtimeRunner(agent)

        model_config: RealtimeModelConfig = {
            "initial_model_settings": {
                "model_name": "gpt-realtime-2.1",
                "turn_detection": {
                    "type": "semantic_vad",
                    "interrupt_response": True,
                    "create_response": True,
                },
                "voice": self._voice,
                "speed": self._speed,
            },
        }

        async with await runner.run(model_config=model_config) as session:
            print("RealtimeAudioChatHandler: connected to realtime session")
            self._session = session
            self._running = True

            try:
                async for event in session:
                    await self._handle_event(event)
            except asyncio.CancelledError:
                return

            self._running = False

    async def close(self) -> None:
        self._running = False
        if self._run_task:
            self._run_task.cancel()
        if self._session:
            await self._session.close()

    async def push_user_audio(self, audio_chunk: AudioChunk, /) -> None:
        if not self._session:
            return

        if audio_chunk.sample_rate != self._SAMPLE_RATE:
            audio_chunk = audio_chunk.resample(self._SAMPLE_RATE)

        await self._session.send_audio(audio_chunk.samples.tobytes())

    async def pull_response_audio(self) -> AudioChunk | None:
        try:
            return self._assistant_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _handle_event(self, event: RealtimeSessionEvent) -> None:
        logger.info(event)

        if event.type == "audio":
            np_audio = np.frombuffer(event.audio.data, dtype=np.int16)
            chunk = AudioChunk(samples=np_audio, sample_rate=self._SAMPLE_RATE)
            self._assistant_queue.put_nowait(chunk)  # enqueue for pullers
            return
