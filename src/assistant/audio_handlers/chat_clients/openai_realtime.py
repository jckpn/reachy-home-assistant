import asyncio
import base64
import contextlib
import logging
from collections.abc import Callable

import numpy as np
from agents.realtime import (
    RealtimeAgent,
    RealtimeRunConfig,
    RealtimeRunner,
    RealtimeSession,
    RealtimeSessionEvent,
    RealtimeUserInput,
)
from agents.tool import function_tool

from ...utils import (
    AssistantAudioEvent,
    AudioChunk,
    PlaybackCancelRequest,
    UserAudioEvent,
    load_greeting,
    log_chat_started,
    log_tool_call,
    log_transcript,
)
from .base import ChatClient

logger = logging.getLogger(__name__)


class OpenAIRealtime(ChatClient):
    _SAMPLE_RATE = 24000  # required by openai api

    def __init__(
        self,
        model_name: str = "gpt-realtime-2.1",
        voice: str = "ash",
        talking_speed: float = 1.0,
        system_prompt: str | None = None,
        instant_greeting: bool = True,
    ) -> None:
        self._model_name = model_name
        self._voice = voice
        self._talking_speed = talking_speed
        self._system_prompt = system_prompt
        self._instant_greeting = instant_greeting

        self._session: RealtimeSession | None = None
        self._assistant_queue = asyncio.Queue[AssistantAudioEvent]()
        self._intro_played: bool = False
        self._tools: list[Callable] = []

    def register_tools(self, tools: list[Callable], /) -> None:
        if self._session:
            raise RuntimeError("Cannot register tools after session has started")
        self._tools += tools

    async def _run(self) -> None:
        log_chat_started()

        # play pre-recorded audio while waiting for the session to start
        if self._instant_greeting:
            asyncio.create_task(self._play_intro())

        tool_schemas = [function_tool(t) for t in self._tools]
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
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",
                },
            }
        }
        runner = RealtimeRunner(agent, config=config)

        async with await runner.run() as session:
            self._session = session  # pointer for push/pull functions

            logger.info("started openai realtime session")

            if not self._instant_greeting:
                # prompt the model to speak first
                await session.send_message("Hello")

            async for event in session:
                if not self._session:
                    break
                await self._handle_event(event)

    async def _play_intro(self) -> None:
        intro_audio = load_greeting(voice=self._voice)
        self._assistant_queue.put_nowait(intro_audio)
        await asyncio.sleep(intro_audio.duration())
        self._intro_played = True  # so handle_user_audio knows not to ignore anymore
        # we do this since the API's echo cancellation obviously doesn't know the intro is itself,
        # as it didn't come from the API directly

    async def handle_user_audio(self, audio_chunk: UserAudioEvent, /) -> None:
        if not self._session:
            logger.warning("waiting for session to become available...")
            return

        if self._instant_greeting and not self._intro_played:
            logger.info("waiting for instant greeting to finish...")
            return

        if audio_chunk.sample_rate != self._SAMPLE_RATE:
            audio_chunk = audio_chunk.resample(self._SAMPLE_RATE)

        await self._session.send_audio(audio_chunk.samples.tobytes())

    def pull_response_audio(self) -> AssistantAudioEvent | None:
        try:
            return self._assistant_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _handle_event(self, event: RealtimeSessionEvent, /) -> None:
        logger.info(f"received event: {event.__class__.__name__}")

        if event.type == "audio":
            samples = np.frombuffer(event.audio.data, dtype=np.int16)
            chunk = AudioChunk(
                role="assistant",
                samples=samples,
                sample_rate=self._SAMPLE_RATE,
            )
            self._assistant_queue.put_nowait(chunk)

        elif event.type == "audio_interrupted":
            self._assistant_queue.put_nowait(PlaybackCancelRequest())

        elif event.type == "error":
            raise RuntimeError(f"session error: {event.error}")

        # capture transcripts and tool calls for logs
        elif event.type == "raw_model_event":
            with contextlib.suppress(AttributeError, IndexError):
                if user_transcript := event.data.transcript:  # type: ignore
                    log_transcript(role="user", transcript=user_transcript)
                if assistant_transcript := event.data.item.content[0].transcript:  # type: ignore
                    log_transcript(role="assistant", transcript=assistant_transcript)

        elif event.type == "tool_start":
            log_tool_call(tool_name=event.tool.name)

    async def handle_image(self, jpeg: bytes) -> None:
        if not self._session:
            logger.warning("no session available, cannot push image")
            return

        encoded = base64.b64encode(jpeg).decode("utf-8")
        url = f"data:image/jpeg;base64,{encoded}"

        msg: RealtimeUserInput = {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": url,
                }
            ],
        }
        await self._session.send_message(msg)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._assistant_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        if self._session:
            await self._session.close()
        self._session = None
        self._drain_queue()
        self._intro_played = False
