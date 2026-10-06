from __future__ import annotations

import hashlib
from pathlib import Path

from app.evaluation.confidence_calibration import load_confidence_config


def corpus_hash(kb_dir: str | Path) -> str:
    root=Path(kb_dir)
    return hashlib.sha256(b"".join(path.read_bytes() for path in sorted(root.glob("*.md")))).hexdigest()


def load_runtime_confidence(report_path: str | Path, kb_dir: str | Path, *, rerank_model: str, top_k: int):
    return load_confidence_config(report_path,corpus_hash=corpus_hash(kb_dir),rerank_model=rerank_model,top_k=top_k)


def load_runtime_confidence_if_available(report_path: str | Path, kb_dir: str | Path, *, rerank_model: str, top_k: int):
    path=Path(report_path)
    if not path.is_file():
        return None
    return load_runtime_confidence(path,kb_dir,rerank_model=rerank_model,top_k=top_k)
