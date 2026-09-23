import asyncio
import logging
from abc import ABC, abstractmethod
from typing import final

from ..utils import AssistantAudioEvent, UserAudioEvent

logger = logging.getLogger(__name__)


class AudioHandler(ABC):
    @final
    async def run(self, *, timeout: float | None = None) -> None:
        try:
            if timeout:
                await asyncio.wait_for(self._run(), timeout=timeout)
            else:
                await self._run()
        except TimeoutError:
            logger.info(f"Ended {self.__class__.__name__} after {timeout}s")
        except:  # noqa: E722
            logger.exception(f"Exception running {self.__class__.__name__}")
        finally:
            await self.close()

    @abstractmethod
    async def _run(self) -> None:
        """
        Run the audio handler, handling events and messages as needed.

        Subclasses should override this method to initiate the chat session and return
        when the session is closed or the client is done.
        """

    async def close(self) -> None:
        """
        Close and tidy up the session.
        """

    async def handle_user_audio(self, audio_chunk: UserAudioEvent, /) -> None:
        """
        Used by AudioRouter to push audio from the user to the chat client.

        Subclasses should override this method to handle streamed audio chunks from the
        user's input device.

        Not strictly required, as some chat clients may not require user audio input.
        """

    def pull_response_audio(self) -> AssistantAudioEvent | None:
        """
        Used by AudioRouter to pull audio from the chat client to play back to the user.

        Subclasses should override this method to return the next chunk of audio from
        the chat client, or None if no audio is available (e.g. during user speech).

        Not strictly required, as some chat clients may not produce audio output (e.g.
        WakeWordDetector).
        """
