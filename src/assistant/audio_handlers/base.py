from abc import ABC, abstractmethod
from collections.abc import Callable

from ..utils import AssistantAudioEvent, UserAudioEvent


class AudioHandler(ABC):
    @abstractmethod
    async def run(self) -> None:
        """
        Run the chat client, handling events and messages as needed.

        Subclasses should override this method to initiate the chat session and return
        when the session is closed or the client is done.
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

    async def close(self) -> None:
        """
        Used by the end_chat tool to end chats early.
        """


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
