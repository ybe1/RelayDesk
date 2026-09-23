"""The cross-platform command line boundary for Ch03 knowledge writes."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import func, select

from app.db.models import KnowledgeChunk
from app.knowledge.chunks import parse_markdown
from app.knowledge.importer import import_chunks
from app.knowledge.indexer import index_pending
from app.knowledge.locks import writer_lock
from app.knowledge.sources import SOURCE_TYPES, read_demo_source

# Kept as replaceable module attributes for isolated CLI tests.  Real imports
# stay inside vectorize so preview/build do not require the Milvus package.
EmbeddingClient = None
MilvusStore = None
settings = None


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


def mine() -> dict[str, int]:
    """Reserve the stable CLI command until Task 5 supplies extraction."""
    raise RuntimeError("kb mine is not available until Task 5")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MewHelp Ch03 knowledge commands")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("preview")
    subcommands.add_parser("build")
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
        elif args.command == "vectorize":
            if database_engine is None:
                from app.db.session import engine as database_engine
            with lock(database_engine):
                outcome = vectorize(session_factory=session_factory, crash_after_upsert=args.crash_after_upsert)
        elif args.command == "mine":
            if database_engine is None:
                from app.db.session import engine as database_engine
            with lock(database_engine):
                outcome = mine()
        else:
            raise RuntimeError(f"kb {args.command} is not available until Task 5")
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
