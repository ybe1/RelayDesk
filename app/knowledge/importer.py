from collections.abc import Callable

from sqlalchemy import select

from app.db.models import FAQ, KnowledgeChunk
from app.knowledge.chunks import ChunkDraft, ImportResult
from app.knowledge.dedup import qa_fingerprint


_LIMITS = {"category": 255, "section_path": 512, "content_type": 32}
_TEXT_LIMIT = 65535


def legacy_faq_draft(faq) -> ChunkDraft:
    """Copy one legacy FAQ without modifying the Ch02 record it came from."""
    return ChunkDraft(
        category=faq.category,
        questions=faq.question,
        answer=faq.answer,
        section_path=f"legacy-faq/{faq.id}",
        content_type="faq",
    )


def import_legacy_faq(session_factory: Callable) -> ImportResult:
    """Read legacy FAQ rows, then copy only unknown QA pairs into Ch03 knowledge."""
    with session_factory() as session:
        drafts = [legacy_faq_draft(faq) for faq in session.scalars(select(FAQ).order_by(FAQ.id))]
    return import_chunks(session_factory, drafts)


def _validate(draft: ChunkDraft) -> None:
    for field, limit in _LIMITS.items():
        if len(getattr(draft, field)) > limit:
            raise ValueError(f"{field} exceeds DDL length {limit}")
    for field in ("questions", "answer"):
        if len(getattr(draft, field).encode("utf-8")) > _TEXT_LIMIT:
            raise ValueError(f"{field} exceeds DDL length {_TEXT_LIMIT}")


def import_chunks(session_factory: Callable, drafts: list[ChunkDraft]) -> ImportResult:
    """Insert previously unseen QA pairs and link only this source's new chunks."""
    for draft in drafts:
        _validate(draft)
    with session_factory() as session:
        try:
            existing = {
                qa_fingerprint(chunk.questions, chunk.answer)
                for chunk in session.scalars(select(KnowledgeChunk)).all()
            }
            inserted: list[KnowledgeChunk] = []
            skipped = 0
            for draft in drafts:
                fingerprint = qa_fingerprint(draft.questions, draft.answer)
                if fingerprint in existing:
                    skipped += 1
                    continue
                existing.add(fingerprint)
                chunk = KnowledgeChunk(
                    category=draft.category,
                    questions=draft.questions,
                    answer=draft.answer,
                    section_path=draft.section_path,
                    content_type=draft.content_type,
                    is_key_clause=int(draft.is_key_clause),
                )
                session.add(chunk)
                session.flush()
                inserted.append(chunk)
            by_source: dict[str, list[KnowledgeChunk]] = {}
            for chunk in inserted:
                source = (chunk.section_path or "").split(" / ", 1)[0]
                by_source.setdefault(source, []).append(chunk)
            for chunks in by_source.values():
                for previous, following in zip(chunks, chunks[1:]):
                    previous.next_chunk_id = following.id
                    following.prev_chunk_id = previous.id
            session.commit()
            return ImportResult(len(inserted), skipped, tuple(chunk.id for chunk in inserted))
        except Exception:
            session.rollback()
            raise
