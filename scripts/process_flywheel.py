import asyncio

from app.core.llm import get_chat_model
from app.db.session import SessionLocal
from app.flywheel.processor import process_pending


async def run():
    report = await process_pending(session_factory=SessionLocal,
        model=get_chat_model(disable_thinking=True, max_retries=0))
    print(report)


if __name__ == "__main__":
    asyncio.run(run())
