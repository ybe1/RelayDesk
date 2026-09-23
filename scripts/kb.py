"""The cross-platform command line boundary for Ch03 knowledge writes."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import delete, func, select, update

from app.db.models import Conversation, KnowledgeChunk, Message, QAExtractionStaging
from app.knowledge.chunks import parse_markdown
from app.knowledge.importer import import_chunks
from app.knowledge.indexer import index_pending
from app.knowledge.locks import writer_lock
from app.knowledge.sources import SOURCE_TYPES, read_demo_source

# Kept as replaceable module attributes for isolated CLI tests.  Real imports
# stay inside vectorize so preview/build do not require the Milvus package.
EmbeddingClient = None
MilvusStore = None
HybridMilvusStore = None
rebuild_ch04 = None
settings = None
get_chat_model = None
extract_batches = None
deduplicate_staging = None

SYNTHETIC_USER_ID = "ch03-demo-v1"
SYNTHETIC_CONVERSATIONS = (
    ("演示退货期限是多久？", "演示规则：商品签收后七天内可申请退货。"),
    ("演示运费怎么算？", "演示规则：满 99 元包邮；不足 99 元运费 8 元。"),
)


_COUNTERS = ("imported", "skipped", "indexed", "pending", "failed")


def _session_factory_or_default(session_factory: Callable | None) -> Callable:
    if session_factory is not None:
        return session_factory
    from app.db.session import SessionLocal
    return SessionLocal


def _pending_count(session_factory: Callable) -> int:
    with session_factory() as session:
        return int(session.scalar(
            select(func.count()).select_from(KnowledgeChunk).where(
                KnowledgeChunk.vectorize_status == "pending"
            )
        ) or 0)


def _result(**values: int) -> dict[str, int]:
    return {name: int(values.get(name, 0)) for name in _COUNTERS}


def preview(*, session_factory: Callable | None = None, source_root: Path | None = None) -> dict[str, int]:
    """Parse approved documents without creating rows or taking a write lock."""
    session_factory = _session_factory_or_default(session_factory)
    drafts = 0
    for name in SOURCE_TYPES:
        text, content_type = read_demo_source(name, root=source_root)
        drafts += len(parse_markdown(text, name, content_type))
    return _result(skipped=drafts, pending=_pending_count(session_factory))


def build(*, session_factory: Callable | None = None, source_root: Path | None = None) -> dict[str, int]:
    """Import source documents only; vectorization remains a separate operation."""
    session_factory = _session_factory_or_default(session_factory)
    drafts = []
    for name in SOURCE_TYPES:
        text, content_type = read_demo_source(name, root=source_root)
        drafts.extend(parse_markdown(text, name, content_type))
    imported = import_chunks(session_factory, drafts)
    return _result(imported=imported.inserted, skipped=imported.skipped, pending=_pending_count(session_factory))


def vectorize(
    *,
    session_factory: Callable | None = None,
    crash_after_upsert: bool = False,
) -> dict[str, int]:
    session_factory = _session_factory_or_default(session_factory)
    runtime_settings = settings
    if runtime_settings is None:
        from app.config import settings as runtime_settings
    embedding_client = EmbeddingClient
    milvus_store = MilvusStore
    if embedding_client is None:
        from app.knowledge.embedding import EmbeddingClient as embedding_client
    if milvus_store is None:
        from app.knowledge.vector_store import MilvusStore as milvus_store
    store = milvus_store(uri=runtime_settings.milvus_uri)
    store.ensure_collection()
    embedder = embedding_client(
        base_url=runtime_settings.embed_base_url,
        api_key=runtime_settings.embed_api_key,
        model=runtime_settings.embed_model,
    )
    hook = (lambda: os._exit(86)) if crash_after_upsert else None
    return _result(**index_pending(session_factory, embedder, store, after_upsert=hook))


def ch04_rebuild(*, session_factory: Callable | None = None) -> dict:
    session_factory = _session_factory_or_default(session_factory)
    runtime_settings = settings
    if runtime_settings is None:
        from app.config import settings as runtime_settings
    store_class = HybridMilvusStore
    embedding_class = EmbeddingClient
    rebuild = rebuild_ch04
    if store_class is None:
        from app.knowledge.hybrid_store import HybridMilvusStore as store_class
    if embedding_class is None:
        from app.knowledge.embedding import EmbeddingClient as embedding_class
    if rebuild is None:
        from app.knowledge.ch04_indexer import rebuild_ch04 as rebuild
    store = store_class(uri=runtime_settings.milvus_uri)
    embedder = embedding_class(base_url=runtime_settings.embed_base_url,
                               api_key=runtime_settings.embed_api_key, model=runtime_settings.embed_model)
    return rebuild(session_factory, embedder, store)


def mine(*, session_factory: Callable | None = None, model=None) -> dict[str, int]:
    """Extract reusable QA pairs, then promote globally unique pairs to pending."""
    session_factory = _session_factory_or_default(session_factory)
    chat_model_factory = get_chat_model
    extractor = extract_batches
    deduplicator = deduplicate_staging
    if chat_model_factory is None:
        from app.core.llm import get_chat_model as chat_model_factory
    if extractor is None:
        from app.knowledge.extraction import extract_batches as extractor
    if deduplicator is None:
        from app.knowledge.dedup import deduplicate_staging as deduplicator
    if model is None:
        model = chat_model_factory(disable_thinking=True)
    extraction = extractor(session_factory, model)
    deduplication = deduplicator(session_factory)
    return _result(
        imported=deduplication["imported"],
        skipped=extraction["skipped"] + deduplication["discarded"],
        pending=_pending_count(session_factory),
    )


def seed_conv(*, session_factory: Callable | None = None) -> dict[str, int]:
    """Append only missing synthetic conversations; existing Ch02 data is untouched."""
    session_factory = _session_factory_or_default(session_factory)
    added = 0
    with session_factory.begin() as session:
        existing = set(session.scalars(
            select(Message.content).join(Conversation).where(Conversation.user_id == SYNTHETIC_USER_ID, Message.role == "user")
        ).all())
        for question, answer in SYNTHETIC_CONVERSATIONS:
            if question in existing:
                continue
            conversation = Conversation(user_id=SYNTHETIC_USER_ID)
            session.add(conversation)
            session.flush()
            session.add_all([
                Message(conversation_id=conversation.id, role="user", content=question),
                Message(conversation_id=conversation.id, role="assistant", content=answer),
            ])
            added += 1
    return _result(imported=added)


def reset_knowledge(session_factory: Callable, store) -> dict[str, int]:
    """Clear only the two Ch03 SQL tables and the derived knowledge collection."""
    sql_cleared = False
    vector_cleared = False
    try:
        with session_factory.begin() as session:
            session.execute(update(KnowledgeChunk).values(prev_chunk_id=None, next_chunk_id=None))
            session.execute(delete(QAExtractionStaging))
            session.execute(delete(KnowledgeChunk))
        sql_cleared = True
    except Exception:
        return {"sql_cleared": False, "vector_cleared": False, "failed": 1}
    try:
        # The store owns its collection-specific deletion API; it must not target another collection.
        store.clear_collection("knowledge")
        vector_cleared = True
    except Exception:
        pass
    return {"sql_cleared": sql_cleared, "vector_cleared": vector_cleared, "failed": int(not vector_cleared)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MewHelp Ch03 knowledge commands")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("preview")
    subcommands.add_parser("build")
    subcommands.add_parser("ch04-rebuild")
    vector_parser = subcommands.add_parser("vectorize")
    vector_parser.add_argument("--crash-after-upsert", action="store_true", help=argparse.SUPPRESS)
    subcommands.add_parser("mine")
    subcommands.add_parser("seed-conv")
    reset = subcommands.add_parser("reset")
    reset.add_argument("--confirmed", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    session_factory: Callable | None = None,
    database_engine=None,
    source_root: Path | None = None,
    lock=writer_lock,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preview":
            outcome = preview(session_factory=session_factory, source_root=source_root)
        elif args.command == "build":
            if database_engine is None:
                from app.db.session import engine as database_engine
            with lock(database_engine):
                outcome = build(session_factory=session_factory, source_root=source_root)
        elif args.command == "ch04-rebuild":
            if database_engine is None:
                from app.db.session import engine as database_engine
            with lock(database_engine):
                outcome = ch04_rebuild(session_factory=session_factory)
        elif args.command == "vectorize":
            if database_engine is None:
                from app.db.session import engine as database_engine
            with lock(database_engine):
                outcome = vectorize(session_factory=session_factory, crash_after_upsert=args.crash_after_upsert)
        elif args.command == "mine":
            if database_engine is None:
                from app.db.session import engine as database_engine
            with lock(database_engine):
                outcome = mine(session_factory=session_factory)
        elif args.command == "seed-conv":
            if database_engine is None:
                from app.db.session import engine as database_engine
            with lock(database_engine):
                outcome = seed_conv(session_factory=session_factory)
        elif args.command == "reset":
            if not args.confirmed:
                raise RuntimeError("reset requires --confirmed")
            if database_engine is None:
                from app.db.session import engine as database_engine
            runtime_settings = settings
            if runtime_settings is None:
                from app.config import settings as runtime_settings
            store_class = MilvusStore
            if store_class is None:
                from app.knowledge.vector_store import MilvusStore as store_class
            with lock(database_engine):
                outcome = reset_knowledge(session_factory or _session_factory_or_default(None), store_class(uri=runtime_settings.milvus_uri))
        else:
            raise RuntimeError(f"unknown kb command: {args.command}")
    except Exception as error:
        try:
            pending = _pending_count(_session_factory_or_default(session_factory))
        except Exception:
            pending = 0
        print(json.dumps(_result(pending=pending, failed=1), ensure_ascii=False))
        print(f"kb {args.command} failed: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(outcome, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
