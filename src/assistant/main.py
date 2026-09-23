import logging
import os

from dotenv import load_dotenv
from reachy_mini import ReachyMini

from .audio_handlers.chat_clients import OpenAIRealtime
from .audio_handlers.sound_detectors import (
    DetectedMusicInfo,
    MusicDetector,
    PhraseDetector,
)
from .audio_router import AudioRouter
from .audio_transports import ReachyAudioTransport
from .movements import MovementManager
from .prompts import load_prompt
from .tool_factories.calendar import (
    GCalClient,
    search_calendar_tool_factory,
    view_upcoming_calendar_events_tool_factory,
)
from .tool_factories.misc import (
    end_chat_tool_factory,
    make_note_tool_factory,
    use_camera_tool_factory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# load env variables
load_dotenv()

MEMORIES_PATH = os.getenv("MEMORIES_PATH")
if not MEMORIES_PATH:
    logger.warning(
        "MEMORIES_PATH environment variable not set. Memories will not be saved."
    )

WAKE_PHRASE = os.getenv("WAKE_PHRASE")
if not WAKE_PHRASE:
    logger.warning(
        "WAKE_PHRASE environment variable not set. Chat will start instantly."
    )


async def main() -> None:
    # connect to reachy daemon
    reachy = ReachyMini()

    # init audio router with connection to reachy's mic + speaker
    audio_router = AudioRouter(transport=ReachyAudioTransport(reachy))

    # init movement manager to handle sleeping, chatting and dancing movements
    movement_manager = MovementManager(reachy)

    # init phrase detector (to start chats) and music detector (to start dances)
    wake_phrase_detector: PhraseDetector | None = None
    if wake_phrase := os.getenv("WAKE_PHRASE"):
        wake_phrase_detector = PhraseDetector(wake_phrases=[wake_phrase])

    def handle_music_info(music_info: DetectedMusicInfo):
        if music_info.is_music and music_info.bpm is not None:
            movement_manager.start_dancing(bpm=music_info.bpm)
        else:
            movement_manager.start_sleeping()

    music_detector = MusicDetector(music_info_handler=handle_music_info)

    # load prompt with previously saved memories
    system_prompt = load_prompt(MEMORIES_PATH)

    # init realtime audio chat client
    chat_client = OpenAIRealtime(
        voice="verse",  # openai recommends cedar (M) or marin (F)
        model_name="gpt-realtime-mini",
        system_prompt=system_prompt,
        talking_speed=1.2,
        instant_greeting=True,
    )

    # build user-specific tools from tool factories and register with chat client
    gcal_client = GCalClient()
    end_chat = end_chat_tool_factory(chat_client)
    use_camera = use_camera_tool_factory(reachy, chat_client)
    view_upcoming_calendar_events = view_upcoming_calendar_events_tool_factory(
        gcal_client
    )
    search_calendar = search_calendar_tool_factory(gcal_client)
    chat_client.register_tools(
        [
            end_chat,
            use_camera,
            view_upcoming_calendar_events,
            search_calendar,
        ]
    )
    if MEMORIES_PATH:
        make_note = make_note_tool_factory(MEMORIES_PATH)
        chat_client.register_tools([make_note])

    # start loop: wait for wake word or music, and start chat or dance accordingly
    audio_router.start()

    while True:
        if wake_phrase_detector:
            movement_manager.start_sleeping()
            music_detector_task = asyncio.create_task(music_detector.run())
            audio_router.route_to([wake_phrase_detector, music_detector])
            await wake_phrase_detector.run()
            music_detector_task.cancel()

        movement_manager.start_chatting()
        audio_router.route_to(chat_client)
        await chat_client.run()

        if not wake_phrase_detector:
            break

    audio_router.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
