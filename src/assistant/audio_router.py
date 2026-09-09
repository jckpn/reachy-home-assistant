import asyncio
import logging

from .audio_transports import AudioTransport
from .chat_clients import ChatClient

logger = logging.getLogger(__name__)


class AudioRouter:
    def __init__(self, audio_transport: AudioTransport) -> None:
        self._audio_transport = audio_transport

        self._chat_client: ChatClient | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._running = True

        await self._audio_transport.start()

        self._tasks = [
            asyncio.create_task(self._capture_loop()),
            asyncio.create_task(self._playback_loop()),
        ]

    async def stop(self) -> None:
        self._running = False

        for t in self._tasks:
            t.cancel()

        if self._chat_client:
            await self._chat_client.close()
        await self._audio_transport.close()

    async def _capture_loop(self) -> None:
        try:
            while self._running:
                chunk = await self._audio_transport.pull_from_mic()
                if chunk and self._chat_client:
                    await self._chat_client.push_user_audio(chunk)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

    async def _playback_loop(self) -> None:
        try:
            while self._running:
                if self._chat_client:
                    chunk = await self._chat_client.pull_response_audio()
                    if chunk:
                        await self._audio_transport.push_to_speaker(chunk)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

    def route_to(self, chat_client: ChatClient | None, /) -> None:
        transport_name = self._audio_transport.__class__.__name__
        client_name = chat_client.__class__.__name__ if chat_client else "None"
        logger.info(f"Audio router: {transport_name}->{client_name}")

        self._chat_client = chat_client
