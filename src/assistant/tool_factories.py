import asyncio
from collections.abc import Callable

import anyio
from reachy_mini import ReachyMini

from .chat_clients import ChatClient
from .utils import get_datetime_str


def use_camera_tool_factory(
    reachy: ReachyMini, chat_client: ChatClient, max_attempts: int = 10
) -> Callable:

    async def use_camera() -> None:
        """
        Take a look at the real world.
        """

        # this can take multiple attempts for some reason
        for _ in range(max_attempts):
            jpeg = reachy.media.get_frame_jpeg()
            if jpeg is not None:
                await chat_client.handle_image(jpeg)

        raise RuntimeError("Failed to capture image from camera!")

    return use_camera


def end_chat_tool_factory(chat_client: ChatClient) -> Callable:

    async def _end_with_delay(delay: float) -> None:
        await asyncio.sleep(delay)
        await chat_client.force_close()

    async def end_chat() -> str:
        """
        End the chat.
        You can use this tool whenever you feel the current chat should end, e.g. if the
        user says bye or otherwise implies the chat should close.
        """

        asyncio.create_task(_end_with_delay(2.0))
        return "Chat ending. Say goodbye to the user if you haven't already."

    return end_chat


def make_note_tool_factory(memories_path: str) -> Callable:

    async def make_note(note: str, is_important: bool = False) -> None:
        """
        Take a note, write some thoughts, or any memories you want to remember for later.

        Args:
            note (str): The note or memory you want to save.
            is_important (bool): If True, the note will be marked as important. This can be
                used to prioritize certain memories over others.
        """

        dt_str = get_datetime_str()

        if is_important:
            # place at top
            async with await anyio.open_file(memories_path, "r") as f:
                notes = await f.read()
            new_notes = f"- [IMPORTANT | {dt_str}]: {note}\n" + notes
            async with await anyio.open_file(memories_path, "w") as f:
                await f.write(new_notes)
        else:
            async with await anyio.open_file(memories_path, "a") as f:
                await f.write(f"- [{dt_str}]: {note}\n")

    return make_note


def get_calendar_tool_factory(calendar_url: str) -> Callable:

    async def get_calendar() -> ...: ...

    return get_calendar
