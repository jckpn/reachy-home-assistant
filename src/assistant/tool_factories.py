from collections.abc import Callable

from reachy_mini import ReachyMini

from .chat_clients import ChatClient


def use_camera_tool_factory(
    reachy: ReachyMini,
    chat_client: ChatClient,
    max_attempts: int = 10,
) -> Callable:

    async def use_camera():
        """Take a photo with your camera to look at the real world."""

        # this can take multiple attempts for some reason
        for _ in range(max_attempts):
            jpeg = reachy.media.get_frame_jpeg()
            if jpeg is None:
                raise RuntimeError("Failed to capture image from camera!")
            await chat_client.handle_image(jpeg)
            return

    return use_camera


def end_chat_tool_factory(
    chat_client: ChatClient,
) -> Callable:

    async def end_chat():
        """End the chat."""

        await chat_client.force_close()

    return end_chat
