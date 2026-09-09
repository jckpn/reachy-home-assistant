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


async def main() -> None:
    audio_manager = ReachyAudioTransport()
    audio_router = AudioRouter(audio_manager)

    wake_word_detector = WakeWordDetector(wake_phrase="hello")
    await wake_word_detector.start()

    await audio_router.start()
    audio_router.route_to(wake_word_detector)

    await asyncio.sleep(60.0)
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
