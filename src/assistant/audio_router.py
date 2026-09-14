import asyncio
import logging

from .audio_transports import AudioTransport
from .chat_clients import ChatClient
from .utils import AudioChunk, PlaybackCancelRequest

logger = logging.getLogger(__name__)


class AudioRouter:
    def __init__(self, audio_transport: AudioTransport) -> None:
        self._audio_transport = audio_transport

        self._running = False
        self._chat_client: ChatClient | None = None
        self._loop_tasks: list[asyncio.Task] = []

    async def _start(self) -> None:
        self._audio_transport.start()

        self._running = True
        self._loop_tasks = [
            asyncio.create_task(self._capture_loop()),
            asyncio.create_task(self._playback_loop()),
        ]

    async def _close(self) -> None:
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
                    event = await self._chat_client.pull_response_audio()
                    if isinstance(event, AudioChunk):
                        self._audio_transport.push_to_speaker(event)
                    elif isinstance(event, PlaybackCancelRequest):
                        self._audio_transport.stop_playback()
                else:
                    logger.info("no chat client, skipping playback")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return

    async def run_until_closed(self, chat_client: ChatClient) -> None:
        print(f"starting {chat_client.__class__.__name__}")
        self._chat_client = chat_client
        await self._start()
        await chat_client.run()
        await self._close()
