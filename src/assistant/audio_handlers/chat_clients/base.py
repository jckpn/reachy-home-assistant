from abc import ABC, abstractmethod
from collections.abc import Callable

from ..base import AudioHandler


class ChatClient(AudioHandler, ABC):
    @abstractmethod
    def register_tools(self, tools: list[Callable], /) -> None:
        """
        Register tools with the chat client.

        Subclasses should override this method to register tools that can be used by the
        chat client to perform actions or provide additional functionality.
        """

    async def handle_image(self, jpeg: bytes) -> None:
        """
        Used by the use_camera tool to send images to the chat client.
        """
