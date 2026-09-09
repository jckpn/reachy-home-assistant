import asyncio
import logging
import sys

from dotenv import load_dotenv

from .audio_router import AudioRouter
from .audio_transports import ReachyAudioTransport
from .chat_clients import OpenAIChatClient
from .wake_word_detector import WakeWordDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

SYSTEM_PROMPT = """You are Reachy, a robotic household assistant.
Your user's name is Jack.

It is absolutely VITAL that you communicate directly and concisely; one or two sentences maximum.
Longer messages will go ignored."""


async def main() -> None:
    audio_manager = ReachyAudioTransport()
    audio_router = AudioRouter(audio_manager)

    wake_word_detector = WakeWordDetector(wake_phrase="hello")
    openai_chat = OpenAIChatClient(system_prompt=SYSTEM_PROMPT)

    await audio_router.start()
    await audio_router.run_until_closed(wake_word_detector)
    await audio_router.run_until_closed(openai_chat)
    await audio_router.stop()

    # chat_handler = OpenAIChatClient(talking_speed=1.2)
    # await chat_handler.start()
    # audio_router.route_to(chat_handler)
    # await asyncio.sleep(1200.0)
    # await audio_router.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
