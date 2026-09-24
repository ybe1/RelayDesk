"""Check whether retrieved evidence is sufficient for the current question."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from app.core.rag_prompts import EVIDENCE_CHECK_PROMPT


class EvidenceDecision(BaseModel):
    """Whether supplied evidence supports answering all material parts of the question."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    useful: bool
    reason: str = Field(min_length=1, description="简述覆盖依据或缺失信息，不输出推理过程")


async def check_evidence(model, query: str, hits: list[dict]) -> EvidenceDecision:
    """Use retrieved content only; no database/history access or fabricated fallback."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty text")
    if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
        raise ValueError("hits must be a list of evidence objects")
    if not hits:
        return EvidenceDecision(useful=False, reason="没有检索证据")
    evidence = [{key: hit[key] for key in ("id", "question", "answer", "text", "category") if key in hit}
                for hit in hits]
    messages = EVIDENCE_CHECK_PROMPT.format_messages(
        payload=json.dumps({"query": query, "hits": evidence}, ensure_ascii=False),
    )
    result = await model.with_structured_output(
        EvidenceDecision, method="function_calling",
    ).ainvoke(messages, extra_body={"thinking": {"type": "disabled"}})
    return EvidenceDecision.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)
