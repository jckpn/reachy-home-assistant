import asyncio
import base64
import logging
from abc import ABC
from collections.abc import Callable

import numpy as np
import whisper
from agents.realtime import (
    RealtimeAgent,
    RealtimeRunConfig,
    RealtimeRunner,
    RealtimeSession,
    RealtimeSessionEvent,
    RealtimeUserInput,
)
from agents.tool import function_tool

from .utils import (
    AssistantAudioEvent,
    AudioChunk,
    PlaybackCancelRequest,
    UserAudioEvent,
    int16_to_fp32,
    load_greeting,
)

logger = logging.getLogger(__name__)


class ChatClient(ABC):
    async def run(self) -> None:
        """
        Run the chat client, handling events and messages as needed.

        Subclasses should override this method to initiate the chat session and return
        when the session is closed or the client is done.
        """

    async def force_close(self) -> None:
        """
        Used by the end_chat tool to end chats early.
        """

    async def handle_image(self, jpeg: bytes) -> None:
        """
        Used by the use_camera tool to send images to the chat client.
        """

    async def handle_user_audio(self, audio_chunk: UserAudioEvent, /) -> None:
        """
        Used by AudioRouter to push audio from the user to the chat client.

        Subclasses should override this method to handle streamed audio chunks from the
        user's input device.

        Not strictly required, as some chat clients may not require user audio input.
        """

    async def pull_response_audio(self) -> AssistantAudioEvent | None:
        """
        Used by AudioRouter to pull audio from the chat client to play back to the user.

        Subclasses should override this method to return the next chunk of audio from
        the chat client, or None if no audio is available (e.g. during user speech).

        Not strictly required, as some chat clients may not produce audio output (e.g.
        WakeWordDetector).
        """


class OpenAIChatClient(ChatClient):
    _SAMPLE_RATE = 24000  # required by openai api

    def __init__(
        self,
        voice: str = "ash",
        model_name: str = "gpt-realtime-2.1",
        talking_speed: float = 1.0,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        instant_greeting: bool = True,
    ) -> None:
        self._model_name = model_name
        self._voice = voice
        self._talking_speed = talking_speed
        self._system_prompt = system_prompt
        self._tools = tools or []
        self._instant_greeting = instant_greeting

        self._session: RealtimeSession | None = None
        self._assistant_queue: asyncio.Queue[AssistantAudioEvent] = asyncio.Queue()

    def add_tool(self, tool: Callable, /) -> None:
        self._tools.append(tool)

    async def run(self) -> None:
        # reset queue
        self._assistant_queue = asyncio.Queue()

        # play pre-recorded audio while waiting for the session to start
        if self._instant_greeting:
            self._play_intro()

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
            }
        }
        runner = RealtimeRunner(agent, config=config)

        async with await runner.run() as session:
            self._session = session  # pointer for push/pull functions

            logger.info("started openai realtime session")

            if not self._instant_greeting:
                # prompt the model to speak first
                await session.send_message("Hello")

            try:
                async for event in session:
                    await self._handle_event(event)
            except asyncio.CancelledError:
                return

    def _play_intro(self) -> None:
        intro_audio = load_greeting()
        self._assistant_queue.put_nowait(intro_audio)

    async def handle_user_audio(self, audio_chunk: UserAudioEvent, /) -> None:
        if not self._session:
            logger.warning("no session available, cannot push audio")
            return

        if audio_chunk.sample_rate != self._SAMPLE_RATE:
            audio_chunk = audio_chunk.resample(self._SAMPLE_RATE)

        await self._session.send_audio(audio_chunk.samples.tobytes())

    async def pull_response_audio(self) -> AssistantAudioEvent | None:
        try:
            return self._assistant_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _handle_event(self, event: RealtimeSessionEvent) -> None:
        logger.info(f"received event: {event.__class__.__name__}")

        if event.type == "audio":
            samples = np.frombuffer(event.audio.data, dtype=np.int16)
            chunk = AudioChunk(
                role="assistant",
                samples=samples,
                sample_rate=self._SAMPLE_RATE,
            )
            self._assistant_queue.put_nowait(chunk)  # enqueue for pullers

        elif event.type == "audio_interrupted":
            self._assistant_queue.put_nowait(PlaybackCancelRequest())

        elif event.type == "error":
            logger.error(event)

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

    async def force_close(self) -> None:
        if self._session:
            await self._session.close()


class WakeWordDetector(ChatClient):
    _SAMPLE_RATE = 16000  # whisper requries 16khz sr

    def __init__(
        self,
        wake_phrase: str = "hello",
        max_buffer_duration: float = 5.0,
    ) -> None:
        self._wake_phrase = wake_phrase
        self._max_buffer_duration = max_buffer_duration

        self._transcriber = whisper.load_model("tiny.en")
        self._audio_buffer: AudioChunk | None = None

    async def run(self) -> None:
        # reset buffer
        self._audio_buffer = None

        while True:
            await asyncio.sleep(0.2)  # don't run whisper too often since it's slow
            if await self._check_buffer_for_wake_word():
                break

        logger.info("wake word detected!")

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

    async def _check_buffer_for_wake_word(self) -> bool:
        if not self._audio_buffer or self._audio_buffer.duration() == 0:
            logger.debug("buffer is empty, skipping transcription")
            return False

        fp32_samples = int16_to_fp32(self._audio_buffer.samples)

        # transcribe fn is sync, so run in a thread to prevent blocking
        output = await asyncio.to_thread(
            self._transcriber.transcribe,
            audio=fp32_samples,
        )

        transcription = output.get("text")
        if transcription and isinstance(transcription, str):
            logger.info(f"user said: {transcription}")
            if self._wake_phrase in transcription.lower():
                return True

        return False
