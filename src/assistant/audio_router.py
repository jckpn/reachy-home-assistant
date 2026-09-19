import asyncio
import logging

from .audio_transports import AudioTransport
from .chat_clients import ChatClient

logger = logging.getLogger(__name__)


class AudioRouter:
    def __init__(
        self,
        audio_transport: AudioTransport,
        chat_client: ChatClient | None = None,
    ) -> None:
        self._audio_transport = audio_transport
        self._chat_client = chat_client

        self._running = False
        self._loop_tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if not self._chat_client:
            logger.warning("Started AudioRouter with no chat client")

        self._audio_transport.start()
        self._running = True
        self._loop_tasks = [
            asyncio.create_task(self._capture_loop()),
            asyncio.create_task(self._playback_loop()),
        ]

    def close(self) -> None:
        self._running = False
        for task in self._loop_tasks:
            task.cancel()
        self._loop_tasks = []
        self._audio_transport.close()

    async def _capture_loop(self) -> None:
        try:
            while self._running:
                if user_audio := self._audio_transport.pull_from_mic():
                    if self._chat_client:
                        await self._chat_client.handle_user_audio(user_audio)
                    else:
                        logger.warning("no chat client, skipping")
                else:
                    logger.warning("no audio from mic, skipping")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return

    async def _playback_loop(self) -> None:
        try:
            while self._running:
                if self._chat_client:
                    if assistant_audio := self._chat_client.pull_response_audio():
                        if assistant_audio.type == "audio_chunk":
                            self._audio_transport.push_to_speaker(assistant_audio)
                        elif assistant_audio.type == "playback_cancel_request":
                            self._audio_transport.stop_playback()
                else:
                    logger.info("no chat client, skipping playback")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return

    def route_to(self, chat_client: ChatClient, /) -> None:
        print(
            f"Routing {self._audio_transport.__class__.__name__}<->{chat_client.__class__.__name__}..."
        )
        self._chat_client = chat_client
