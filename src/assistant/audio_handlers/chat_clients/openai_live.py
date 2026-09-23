import asyncio
import logging
from collections.abc import Callable

from openai import AsyncOpenAI
from openai.resources.live.live import AsyncLiveConnection
from openai.types.live import (
    InputAudioAppendEvent,
    ResponseCreateEvent,
    ResponseItemCreateEvent,
    ServerEvent,
    SessionStartEvent,
)
from openai.types.live.responses_delegation_config import (
    Reasoning,
    ResponsesDelegationConfig,
)
from openai.types.live.session_config import (
    Audio,
    AudioOutput,
    DelegationResponses,
    SessionConfig,
)
from openai.types.responses import EasyInputMessage

from ...utils import (
    AssistantAudioEvent,
    AudioChunk,
    PlaybackCancelRequest,
    UserAudioEvent,
)
from .base import ChatClient

logger = logging.getLogger(__name__)


class OpenAILive(ChatClient):
    _SAMPLE_RATE = 24000

    def __init__(
        self,
        voice_prompt: str | None = None,
        backend_prompt: str | None = None,
        voice: str | None = None,
        tools: list[Callable] | None = None,
    ) -> None:
        self._voice_prompt = voice_prompt
        self._system_prompt = backend_prompt
        self._voice = voice
        self._tools = tools or []

        self._session: AsyncLiveConnection | None = None
        self._assistant_queue: asyncio.Queue[AssistantAudioEvent] = asyncio.Queue()
        self._tools: list[Callable] = []

    async def _run(self) -> None:
        logger.info("connecting...")

        client = AsyncOpenAI()

        session_cfg = SessionConfig(
            model="gpt-live-1",
            instructions=self._voice_prompt,
            audio=Audio(output=AudioOutput(voice=self._voice)),
            delegation=DelegationResponses(
                type="responses",
                responses=ResponsesDelegationConfig(
                    model="gpt-5.6-luna",  # terra is 10x price of luna
                    instructions=self._system_prompt,
                    reasoning=Reasoning(effort="none"),
                ),
            ),
        )

        async with client.live.connect() as session:
            self._session = session  # pointer for push/pull functions

            logger.info("started openai live session")

            start_event = SessionStartEvent(type="session.start", session=session_cfg)
            await session.send(start_event)

            await session.send(
                ResponseItemCreateEvent(
                    type="response.item.create",
                    item=EasyInputMessage(
                        role="user",
                        content="Hello",
                    ),
                )
            )
            await session.send(ResponseCreateEvent(type="response.create"))

            while True:
                event = await session.recv()
                await self._handle_event(event)
                await asyncio.sleep(0.01)

    async def _handle_event(self, event: ServerEvent, /) -> None:
        logger.info(f"received event: {event.__class__.__name__}")

        if event.type == "session.output_audio.delta":
            chunk = AudioChunk.from_base64(
                event.delta,
                role="assistant",
                sample_rate=self._SAMPLE_RATE,
            )
            self._assistant_queue.put_nowait(chunk)

        elif event.type == "error":
            raise RuntimeError(f"session error: {event.error}")

    async def handle_user_audio(self, audio_chunk: UserAudioEvent, /) -> None:
        if not self._session:
            logger.warning("waiting for session to become available...")
            return

        if audio_chunk.sample_rate != self._SAMPLE_RATE:
            audio_chunk = audio_chunk.resample(self._SAMPLE_RATE)

        await self._session.send(
            InputAudioAppendEvent(
                type="session.input_audio.append",
                audio=audio_chunk.to_base64(),
            )
        )

    def pull_response_audio(self) -> AssistantAudioEvent | None:
        try:
            return self._assistant_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def handle_image(self, jpeg: bytes) -> None:
        raise NotImplementedError("OpenAILive does not support image input yet")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
        self._drain_queue()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._assistant_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def register_tools(self, tools: list[Callable], /) -> None:
        if self._session:
            raise RuntimeError("Cannot register tools after session has started")
        self._tools = tools
