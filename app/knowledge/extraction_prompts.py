"""Prompt used for offline mining of reusable knowledge from chat history."""

from langchain_core.prompts import ChatPromptTemplate


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You extract reusable customer-service knowledge from conversations.
Return only stable, general rules that can safely answer a future customer's
question. Do not infer facts that are not stated. Do not extract private or
case-specific facts, including names, contact details, order numbers, tracking
numbers, logistics events, payment records, or simulated order scenarios.
Ignore tool use and tool results. It is valid to return no items.
Each item must use one supplied conversation_id and preserve the user's real
question wording. Keep different answers when they genuinely conflict.""",
    ),
    ("human", "Conversations to review (JSON):\n{conversations}"),
])
