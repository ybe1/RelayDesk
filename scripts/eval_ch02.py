"""Run the three labeled Ch02 chat acceptance examples against the configured model."""

import asyncio
import json
import sys
import uuid

from app.api.chat import get_chat_service
from app.db.session import SessionLocal
from app.repositories.messages import load_messages
from app.schemas.chat import ChatRequest


SAMPLES = [
    ("订单 1001 的物流到哪了", "query_logistics", True),
    ("退货政策是什么", "query_faq", True),
    ("邮费是多少", "query_faq", False),
]


async def main() -> None:
    for question, expected_tool, expected_match in SAMPLES:
        service = get_chat_service()
        req = ChatRequest(user_id=f"eval-{uuid.uuid4().hex[:12]}", message=question)
        conversation_id = None
        answer = []
        async for kind, data in service.events(req):
            if kind == "conversation":
                conversation_id = data["conversation_id"]
            elif kind == "delta":
                answer.append(data["delta"])
        with SessionLocal() as session:
            rows = load_messages(session, conversation_id)
        requested = [call["name"] for row in rows if row.role == "assistant" for call in (row.tool_calls or [])]
        tool_rows = [row for row in rows if row.role == "tool"]
        assert requested and requested[0] == expected_tool, (question, requested)
        assert len(tool_rows) == 1, (question, len(tool_rows))
        if expected_tool == "query_faq":
            result = json.loads(json.loads(tool_rows[0].content)["result"])
            assert bool(result["matches"]) is expected_match, (question, result)
        if expected_tool == "query_logistics":
            result = json.loads(json.loads(tool_rows[0].content)["result"])
            assert result["order_id"] == "1001" and result["mock"] is True
        assert answer, question
        print(json.dumps({"question": question, "tool": requested[0], "answer": "".join(answer), "conversation_id": conversation_id}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
