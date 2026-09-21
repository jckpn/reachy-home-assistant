import asyncio
import logging
from typing import Self

from .audio_handlers import AudioHandler
from .audio_transports import AudioTransport

logger = logging.getLogger(__name__)


class AudioRouter:
    def __init__(
        self,
        transport: AudioTransport,
        handler: AudioHandler | None = None,
    ) -> None:
        self._transport = transport
        self._handler = handler

        self._running = False
        self._tasks: list[asyncio.Task] = []

    def __enter__(self) -> Self:
        if not self._handler:
            logger.warning("Started AudioRouter with no chat client")

        self._transport.start()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._capture_loop()),
            asyncio.create_task(self._playback_loop()),
        ]

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        self._transport.close()

    async def _capture_loop(self) -> None:
        try:
            while self._running:
                try:
                    if self._transport:
                        user_audio = self._transport.pull_from_mic()
                        if user_audio is not None:
                            if self._handler:
                                # logger.info(f"Captured {user_audio.duration()}s audio")
                                await self._handler.handle_user_audio(user_audio)
                            else:
                                logger.warning("no chat client, skipping")
                        else:
                            logger.warning("no audio from mic, skipping")
                except:
                    logger.exception("Error in playback loop")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("Error in capture loop")

    async def _playback_loop(self) -> None:
        try:
            while self._running:
                try:
                    if self._handler:
                        assistant_audio = self._handler.pull_response_audio()
                        if self._transport and assistant_audio is not None:
                            if assistant_audio.type == "audio_chunk":
                                # logger.info(
                                #     f"Playing {assistant_audio.duration()}s audio"
                                # )
                                self._transport.push_to_speaker(assistant_audio)
                            elif assistant_audio.type == "playback_cancel_request":
                                logger.info("Cancelling audio playback")
                                self._transport.stop_playback()
                    else:
                        logger.info("no chat client, skipping playback")
                except:
                    logger.exception("Error in playback loop")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return

    def route_to(self, handler: AudioHandler, /) -> None:
        print(
            f"Routing {self._transport.__class__.__name__} to {handler.__class__.__name__}..."
        )
        self._handler = handler
