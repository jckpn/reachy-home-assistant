import logging
import os

from dotenv import load_dotenv
from reachy_mini import ReachyMini

from .audio_router import AudioRouter
from .audio_transports import LocalAudioTransport, ReachyAudioTransport
from .chat_clients import OpenAILive, OpenAIRealtime, WakePhraseDetector
from .movements import ChattingMovements, SleepingMovements
from .prompts import load_prompt
from .tool_factories import (
    end_chat_tool_factory,
    get_calendar_tool_factory,
    make_note_tool_factory,
    use_camera_tool_factory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

MEMORIES_PATH = os.path.expanduser("~/Desktop/charlie_memories.txt")
system_prompt = load_prompt(MEMORIES_PATH)


async def client_test() -> None:
    audio_transport = LocalAudioTransport()
    chat_client = OpenAILive()
    audio_router = AudioRouter(audio_transport, chat_client)
    audio_router.start()
    await chat_client.run()


async def run_with_reachy() -> None:
    reachy = ReachyMini()
    audio_transport = ReachyAudioTransport(reachy)
    audio_router = AudioRouter(audio_transport)

    # dancer = ReachyDancer(reachy)
    wake_word_detector = WakePhraseDetector(
        wake_phrases=[
            "hey charlie",
            "is that a robot",
        ],
    )

    # chat_client = OpenAIRealtime(
    #     voice="cedar",  # openai recommends cedar (M) or marin (F)
    #     model_name="gpt-realtime-2",
    #     system_prompt=system_prompt,
    #     talking_speed=1.0,
    #     instant_greeting=True,
    # )
    chat_client = OpenAILive(voice="ceder", backend_prompt=system_prompt)
    chat_client.register_tools(
        [
            get_calendar_tool_factory("..."),
            make_note_tool_factory(MEMORIES_PATH),
            use_camera_tool_factory(reachy, chat_client),
            end_chat_tool_factory(chat_client),
        ]
    )

    audio_router.start()

    while True:
        with SleepingMovements(reachy):
            audio_router.route_to(wake_word_detector)
            await wake_word_detector.run()

        with ChattingMovements(reachy, head_tracking_weight=0.8):
            audio_router.route_to(chat_client)
            await chat_client.run()

    # await audio_router.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_with_reachy())
