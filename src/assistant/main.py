import logging
import os

from dotenv import load_dotenv
from reachy_mini import ReachyMini

from .agent_tools.gcal_client import GCalClient
from .agent_tools.tool_factories import (
    end_chat_tool_factory,
    make_note_tool_factory,
    search_calendar_tool_factory,
    use_camera_tool_factory,
    view_upcoming_calendar_events_tool_factory,
)
from .audio_handlers import (
    KnockDetector,
    OpenAILive,
    OpenAIRealtime,
    WakePhraseDetector,
)
from .audio_router import AudioRouter
from .audio_transports import LocalAudioTransport, ReachyAudioTransport
from .movements import MovementManager
from .prompts import load_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

MEMORIES_PATH = os.path.expanduser("~/Desktop/charlie_memories.txt")


async def main() -> None:
    reachy = ReachyMini(log_level="DEBUG")
    movement_manager = MovementManager(reachy)
    # knock_detector = KnockDetector(knock_volume_threshold=0.8)

    # dancer = ReachyDancer(reachy)
    wake_phrase_detector = WakePhraseDetector(
        wake_phrases=[
            "hey charlie",
            "is that a robot",
            "what is that",
        ],
    )
    system_prompt = load_prompt(MEMORIES_PATH)
    chat_client = OpenAIRealtime(
        voice="cedar",  # openai recommends cedar (M) or marin (F)
        model_name="gpt-realtime-2.1",
        system_prompt=system_prompt,
        talking_speed=1.2,
        instant_greeting=False,
    )

    # chat_client = OpenAILive(voice="ceder", backend_prompt=system_prompt)

    # build tools and register with client
    gcal_client = GCalClient()
    chat_client.register_tools(
        [
            make_note_tool_factory(MEMORIES_PATH),
            end_chat_tool_factory(chat_client),
            use_camera_tool_factory(reachy, chat_client),
            view_upcoming_calendar_events_tool_factory(gcal_client),
            search_calendar_tool_factory(gcal_client),
        ]
    )

    audio_transport = ReachyAudioTransport(reachy)
    with AudioRouter(audio_transport) as audio_router:
        while True:
            with movement_manager.sleeping_movements():
                audio_router.route_to(wake_phrase_detector)
                await wake_phrase_detector.run()

            with movement_manager.chatting_movements():
                audio_router.route_to(chat_client)
                await chat_client.run()

    # await audio_router.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
