import logging

from dotenv import load_dotenv
from reachy_mini import ReachyMini

from .audio_router import AudioRouter
from .audio_transports import ReachyAudioTransport
from .chat_clients import OpenAIChatClient, WakeWordDetector
from .movement_manager import ChattingMovements, SleepingMovements
from .tool_factories import end_chat_tool_factory, use_camera_tool_factory
from .tools import get_calendar

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


SYSTEM_PROMPT = """You are Charlie, a voice assistant.

# IMPORTANT: Concise Message Rules
- It is absolutely VITAL that you communicate directly and concisely.
- ALWAYS use one or two sentences maximum.
- In 99% of cases you will only need a few words to communicate your message.
"""


async def main() -> None:
    reachy = ReachyMini()

    # dancer = ReachyDancer(reachy)
    wake_word_detector = WakeWordDetector(wake_phrase="hey charlie")

    openai_chat_client = OpenAIChatClient(
        system_prompt=SYSTEM_PROMPT,
        talking_speed=1.2,
        tools=[get_calendar],
    )
    # add reachy/client-dependent tools tools
    use_camera_tool = use_camera_tool_factory(reachy, openai_chat_client)
    openai_chat_client.add_tool(use_camera_tool)
    end_chat_tool = end_chat_tool_factory(openai_chat_client)
    openai_chat_client.add_tool(end_chat_tool)

    audio_transport = ReachyAudioTransport(reachy)
    audio_router = AudioRouter(audio_transport)

    while True:
        with ChattingMovements(reachy):
            await audio_router.run_until_closed(wake_word_detector)
        # with SleepingMovements(reachy):
        #     await audio_router.run_until_closed(wake_word_detector)

        # with ChattingMovements(reachy):
        #     await audio_router.run_until_closed(openai_chat_client)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
