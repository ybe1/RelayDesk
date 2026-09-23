import hashlib
import json
import unicodedata

from collections.abc import Callable

from sqlalchemy import select

from app.db.models import KnowledgeChunk, QAExtractionStaging


def normalize(text: str) -> str:
    """Return the deterministic comparison form used for every QA source."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def qa_fingerprint(question: str, answer: str) -> str:
    payload = json.dumps(
        [normalize(question), normalize(answer)], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_staging(session_factory: Callable) -> dict[str, int]:
    """Promote the first globally new extracted QA pair in one transaction."""
    with session_factory.begin() as session:
        known = {
            qa_fingerprint(chunk.questions, chunk.answer)
            for chunk in session.scalars(select(KnowledgeChunk)).all()
        }
        kept = 0
        discarded = 0
        imported = 0
        rows = session.scalars(
            select(QAExtractionStaging)
            .where(QAExtractionStaging.status == "extracted")
            .order_by(QAExtractionStaging.id)
        ).all()
        for row in rows:
            fingerprint = qa_fingerprint(row.question, row.answer)
            if fingerprint in known:
                row.status = "discarded"
                discarded += 1
                continue
            known.add(fingerprint)
            row.status = "kept"
            session.add(KnowledgeChunk(
                category="历史对话",
                questions=row.question,
                answer=row.answer,
                section_path=row.source_ref,
                content_type="extracted_qa",
                vectorize_status="pending",
            ))
            kept += 1
            imported += 1
    return {"kept": kept, "discarded": discarded, "imported": imported}
