"""Normalize the current question without history and validate metadata filters."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings
from app.core.rag_prompts import QUERY_UNDERSTANDING_PROMPT


class QueryUnderstanding(BaseModel):
    """Current-question normalization, recall expansion and an optional exact category."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    normalized_query: str = Field(min_length=1, description="当前问题的忠实轻量整理，不扩同义词")
    search_query: str = Field(min_length=1, description="只在此字段追加必要的等价检索词")
    category: str | None = Field(default=None, description="配置白名单的完整分类值或 null")

    @field_validator("category")
    @classmethod
    def category_must_be_configured(cls, value: str | None) -> str | None:
        if value is not None and value not in settings.rag_category_allowlist:
            raise ValueError("category is outside the configured allowlist")
        return value


async def understand_query(model, raw_query: str) -> QueryUnderstanding:
    """Accept one current user question only; model/parse failures propagate to caller."""
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise ValueError("raw_query must be nonempty current-question text")
    messages = QUERY_UNDERSTANDING_PROMPT.format_messages(
        allowed_categories=json.dumps(settings.rag_category_allowlist, ensure_ascii=False),
        payload=json.dumps({"current_question": raw_query}, ensure_ascii=False),
    )
    result = await model.with_structured_output(
        QueryUnderstanding, method="function_calling",
    ).ainvoke(messages, extra_body={"thinking": {"type": "disabled"}})
    return QueryUnderstanding.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)
