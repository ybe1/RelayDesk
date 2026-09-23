from __future__ import annotations

from contextlib import contextmanager
from pathlib import PurePath

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.knowledge.chunks import parse_markdown
from app.knowledge.importer import import_chunks
from app.knowledge.indexer import index_pending
from app.knowledge.locks import WriterBusy, writer_lock
from app.knowledge.queries import knowledge_counts, recent_chunks, staging_counts, staging_rows
from app.knowledge.retrieval import search_knowledge
from app.knowledge.sources import SOURCE_TYPES, read_demo_source

router = APIRouter()
_CONTENT_TYPES = frozenset(SOURCE_TYPES.values())
_MAX_MARKDOWN_CHARS = 200_000


def get_session_factory():
    from app.db.session import SessionLocal
    return SessionLocal


def get_database_engine():
    from app.db.session import engine
    return engine


def get_store():
    from app.config import settings
    from app.knowledge.vector_store import MilvusStore
    store = MilvusStore(uri=settings.milvus_uri)
    store.ensure_collection()
    return store


def get_embedder():
    from app.config import settings
    from app.knowledge.embedding import EmbeddingClient
    return EmbeddingClient(base_url=settings.embed_base_url, api_key=settings.embed_api_key, model=settings.embed_model)


class PreviewRequest(BaseModel):
    markdown: str | None = None
    content_type: str | None = None
    source_name: str | None = None
    document: str | None = None


class IngestRequest(BaseModel):
    markdown: str
    content_type: str
    source_name: str
    vectorize: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def _bad(message: str):
    raise HTTPException(400, message)


def _manual_source_valid(source_name: str) -> bool:
    return bool(source_name) and PurePath(source_name).name == source_name and "/" not in source_name and "\\" not in source_name


def _drafts_for_preview(request: PreviewRequest):
    manual = any(value is not None for value in (request.markdown, request.content_type, request.source_name))
    if request.document is not None and manual:
        _bad("provide either document or manual markdown")
    if request.document is not None:
        try:
            text, content_type = read_demo_source(request.document)
        except ValueError:
            _bad("unknown document")
        return parse_markdown(text, request.document, content_type)
    if (not request.markdown or len(request.markdown) > _MAX_MARKDOWN_CHARS or request.content_type not in _CONTENT_TYPES
            or not _manual_source_valid(request.source_name or "")):
        _bad("invalid manual knowledge input")
    return parse_markdown(request.markdown, request.source_name, request.content_type)


def _preview_payload(drafts, session_factory):
    # Preview deliberately has no writer lock and never writes; duplicate is a read-only hint.
    from app.knowledge.dedup import qa_fingerprint
    from app.db.models import KnowledgeChunk
    from sqlalchemy import select
    with session_factory() as session:
        fingerprints = {qa_fingerprint(row.questions, row.answer) for row in session.scalars(select(KnowledgeChunk)).all()}
    chunks = [{"category": draft.category, "questions": draft.questions, "answer": draft.answer,
               "section_path": draft.section_path, "content_type": draft.content_type,
               "characters": len(draft.answer), "is_key_clause": draft.is_key_clause,
               "is_table": draft.is_table, "duplicate": qa_fingerprint(draft.questions, draft.answer) in fingerprints}
              for draft in drafts]
    return {"chunks": chunks, "features": {"heading_split": True, "overlap": len(chunks) > 1,
                                                  "table_header_copied": any(item["is_table"] for item in chunks)}}


@router.get("/api/kb/overview")
def overview():
    factory = get_session_factory()
    dependencies = {"mysql": "available", "milvus": "available"}
    try:
        counts = knowledge_counts(factory)
        staging = staging_counts(factory)
    except Exception:
        counts = {"total": None, "pending": None, "done": None, "key_clause_count": None}
        staging = {"extracted": None, "kept": None, "discarded": None}
        dependencies["mysql"] = "unavailable"
    try:
        milvus_count = get_store().count()
    except Exception:
        milvus_count = None
        dependencies["milvus"] = "unavailable"
    consistent = None if None in (counts["total"], counts["pending"], counts["done"], milvus_count) else (
        counts["total"] == counts["pending"] + counts["done"] == milvus_count
    )
    try:
        from app.config import settings
        config = {"rag_top_k": settings.rag_top_k, "rag_min_score": settings.rag_min_score,
                  "embed_model": settings.embed_model, "milvus_uri": settings.milvus_uri}
    except Exception:
        config = {"rag_top_k": 5, "rag_min_score": 0.55}
    return {**counts, "staging": staging, "milvus_count": milvus_count, "dependencies": dependencies,
            "consistent": consistent, "config": config}


@router.get("/api/kb/documents")
def documents():
    result = []
    for name, content_type in SOURCE_TYPES.items():
        text, _ = read_demo_source(name)
        drafts = parse_markdown(text, name, content_type)
        result.append({"name": name, "content_type": content_type, "chunks": len(drafts),
                       "features": {"table": any(item.is_table for item in drafts), "key_clause": any(item.is_key_clause for item in drafts)}})
    return result


@router.post("/api/kb/preview")
def preview(request: PreviewRequest):
    return _preview_payload(_drafts_for_preview(request), get_session_factory())


@router.post("/api/kb/ingest")
def ingest(request: IngestRequest):
    if (not request.markdown or len(request.markdown) > _MAX_MARKDOWN_CHARS or request.content_type not in _CONTENT_TYPES
            or not _manual_source_valid(request.source_name)):
        _bad("invalid manual knowledge input")
    drafts = parse_markdown(request.markdown, request.source_name, request.content_type)
    if not drafts:
        _bad("markdown contains no knowledge chunks")
    factory = get_session_factory()
    try:
        with writer_lock(get_database_engine()):
            imported = import_chunks(factory, drafts)
            pending = knowledge_counts(factory)["pending"]
            result = {"inserted": imported.inserted, "skipped": imported.skipped, "indexed": 0, "pending": pending}
            if request.vectorize:
                try:
                    indexed = index_pending(factory, get_embedder(), get_store())
                    result["indexed"] = indexed["indexed"]
                    result["pending"] = indexed["pending"]
                except Exception as error:
                    raise HTTPException(502, {**result, "message": "原文已入库，向量待补"}) from error
            return result
    except WriterBusy as error:
        raise HTTPException(503, "knowledge writer is busy") from error


@router.get("/api/kb/staging")
def staging(status: str | None = None, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    if status not in (None, "extracted", "kept", "discarded"):
        raise HTTPException(422, "invalid staging status")
    return staging_rows(get_session_factory(), status=status, limit=limit, offset=offset)


@router.post("/api/kb/vectorize", status_code=202)
def vectorize():
    from app.api.jobs import start_job
    return start_job("kb-vectorize")


@router.post("/api/kb/search")
def search(request: SearchRequest):
    if not request.query.strip() or not 1 <= request.top_k <= 20:
        _bad("query and top_k are invalid")
    try:
        return {"hits": search_knowledge(request.query, get_embedder(), get_store(), top_k=request.top_k)}
    except Exception as error:
        raise HTTPException(502, "dense search unavailable") from error


@router.get("/api/kb/recent")
def recent(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return recent_chunks(get_session_factory(), limit=limit, offset=offset)
