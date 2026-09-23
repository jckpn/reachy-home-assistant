import asyncio
import logging

from .audio_handlers import AudioHandler
from .audio_transports import AudioTransport

logger = logging.getLogger(__name__)


class AudioRouter:
    def __init__(
        self,
        transport: AudioTransport,
        handlers: list[AudioHandler] | None = None,
    ) -> None:
        self._transport = transport
        self._handlers = handlers or []

        self._running = False
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if not self._handlers:
            logger.warning("Started AudioRouter with no audio handlers")

        self._transport.start()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._capture_loop()),
            asyncio.create_task(self._playback_loop()),
        ]

    def close(self) -> None:
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
                            if self._handlers:
                                for handler in self._handlers:
                                    await handler.handle_user_audio(user_audio)
                            else:
                                logger.warning("no audio handlers, skipping")
                        else:
                            logger.warning("no audio from mic, skipping")
                except:  # noqa: E722
                    logger.exception("Error in capture loop")  # this outputs exc trace
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return

    async def _playback_loop(self) -> None:
        try:
            while self._running:
                try:
                    if self._handlers:
                        for handler in self._handlers:
                            assistant_audio = handler.pull_response_audio()
                            if self._transport and assistant_audio is not None:
                                if assistant_audio.type == "audio_chunk":
                                    self._transport.push_to_speaker(assistant_audio)
                                elif assistant_audio.type == "playback_cancel_request":
                                    logger.info("cancelling audio playback")
                                    self._transport.stop_playback()
                    else:
                        logger.info("no audio handlers, skipping")
                except:  # noqa: E722
                    logger.exception("Error in playback loop")  # this outputs exc trace
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return

    def route_to(self, handlers: AudioHandler | list[AudioHandler], /) -> None:
        if not self._running:
            raise RuntimeError("AudioRouter is not running. Call start() first.")

        if isinstance(handlers, list):
            self._handlers = handlers
        else:
            self._handlers = [handlers]
