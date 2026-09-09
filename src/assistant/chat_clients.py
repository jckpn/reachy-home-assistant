import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from agents.realtime import (
    RealtimeAgent,
    RealtimeRunConfig,
    RealtimeRunner,
    RealtimeSession,
    RealtimeSessionEvent,
)
from agents.tool import function_tool

from .utils import AudioChunk

logger = logging.getLogger(__name__)


class ChatClient(ABC):
    @abstractmethod
    async def run(self) -> None: ...

    @abstractmethod
    async def push_user_audio(self, audio_chunk: AudioChunk, /) -> None: ...

    @abstractmethod
    async def pull_response_audio(self) -> AudioChunk | None: ...


class OpenAIChatClient(ChatClient):
    _SAMPLE_RATE = 24000  # required by openai api

    def __init__(
        self,
        voice: str = "ash",
        model_name: str = "gpt-realtime-2.1",
        talking_speed: float = 1.0,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
    ) -> None:
        self._model_name = model_name
        self._voice = voice
        self._talking_speed = talking_speed
        self._tools = tools
        self._system_prompt = system_prompt

        self._session: RealtimeSession | None = None
        self._assistant_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

    async def run(self) -> None:
        tool_schemas = [function_tool(t) for t in self._tools] if self._tools else []
        agent = RealtimeAgent(
            name="Assistant",
            instructions=self._system_prompt,
            tools=tool_schemas,  # type: ignore
        )
        config: RealtimeRunConfig = {
            "model_settings": {
                "model_name": self._model_name,
                "turn_detection": {
                    "type": "semantic_vad",
                    "interrupt_response": True,
                    "create_response": True,
                },
                "voice": self._voice,
                "speed": self._talking_speed,
            }
        }
        runner = RealtimeRunner(agent, config=config)

        async with await runner.run() as session:
            self._session = session  # pointer for audio push/pull functions

            logger.info("started openai realtime session")

            await session.send_message("hello")

            try:
                async for event in session:
                    await self._handle_event(event)
            except asyncio.CancelledError:
                return

    async def push_user_audio(self, audio_chunk: AudioChunk, /) -> None:
        if not self._session:
            logger.warning("no session available, cannot push audio")
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
        logger.info(f"received event: {event.__class__.__name__}")

        if event.type == "audio":
            samples = np.frombuffer(event.audio.data, dtype=np.int16)
            chunk = AudioChunk(samples=samples, sample_rate=self._SAMPLE_RATE)
            self._assistant_queue.put_nowait(chunk)  # enqueue for pullers
