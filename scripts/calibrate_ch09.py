from __future__ import annotations

import argparse
import asyncio
import httpx
import hashlib
import json
from pathlib import Path
from typing import Any

from app.evaluation.confidence_calibration import ScoredCase, save_report, select_threshold
from app.evaluation.runtime_config import corpus_hash


DEFAULT_MANIFEST = Path("data/ch09/confidence_cases.jsonl")
DEFAULT_REPORT = Path("data/ch09/reports/confidence-calibration.json")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_scores(manifest, rows):
    expected={str(row["id"]) for row in manifest}
    actual={row.case_id for row in rows}
    if len(rows)!=len(actual) or actual!=expected:
        raise ValueError("score IDs/count must exactly match the frozen manifest")


def _load_progress(path,allowed_ids):
    if not Path(path).is_file(): return {}
    rows={}
    for raw in _jsonl(Path(path)):
        row=ScoredCase.model_validate(raw)
        if row.case_id not in allowed_ids: raise ValueError("live progress case ID is not in manifest")
        rows[row.case_id]=row
    return rows


def _save_progress(path,row):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+"\n"); stream.flush()


async def _live_scores(manifest,progress_path):
    from app.config import settings
    from app.knowledge.retrieval import RetrievalDependencies,search_knowledge_ch04
    completed=_load_progress(progress_path,{case["id"] for case in manifest})
    rows=[]
    async with await RetrievalDependencies.afrom_settings(max_retries=0) as deps:
        for case in manifest:
            if case["id"] in completed:
                rows.append(completed[case["id"]]); continue
            for attempt in range(3):
                try:
                    result=await search_knowledge_ch04(case["query"],top_k=settings.rag_top_k,dependencies=deps)
                    break
                except (httpx.TransportError,TimeoutError):
                    if attempt==2: raise
                    await asyncio.sleep(1*(attempt+1))
            score=_score_result(case["id"],case["query"],result,top_k=settings.rag_top_k)
            scored=ScoredCase(case_id=case["id"],fact_group=case["fact_group"],
                expected_allowed=case["evidence_sufficient"],score=score)
            rows.append(scored); _save_progress(progress_path,scored.model_dump())
            print(json.dumps({"case_id":case["id"],"score":score},ensure_ascii=False),flush=True)
    return rows


def _score_result(case_id,query,result,*,top_k):
    from datetime import UTC,datetime
    from app.flywheel.confidence import compute_confidence
    from app.flywheel.contracts import ConfidenceConfig
    from app.flywheel.evidence import snapshot_hits
    cfg=ConfidenceConfig(threshold=0,top1_floor=0,top1_ceiling=1,gap_floor=0,gap_ceiling=1,
        effective_score_threshold=.5,top_k=top_k,calibration_id="live-scoring")
    hits=[{**hit,"rank":hit.get("rank",i)} for i,hit in enumerate(result.hits,1)]
    snapshot=snapshot_hits(query,"hybrid_rerank",hits,retrieval_id=case_id,created_at=datetime.now(UTC))
    return compute_confidence(snapshot.hits,cfg).score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and freeze the Ch09 evidence threshold")
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores", type=Path, help="JSONL rows with case_id/fact_group/expected_allowed/score")
    source.add_argument("--live",action="store_true",help="Collect current retrieval and rerank scores")
    parser.add_argument("--live-progress",type=Path,default=Path("data/ch09/reports/confidence-live-scores.jsonl"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-hash")
    parser.add_argument("--rerank-model")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    manifest=_jsonl(args.manifest)
    if args.live:
        from app.config import settings
        rows=asyncio.run(_live_scores(manifest,args.live_progress))
        active_corpus=corpus_hash(Path("data/kb")); rerank_model=settings.rerank_model; top_k=settings.rag_top_k
    else:
        if not (args.corpus_hash and args.rerank_model and args.top_k):
            parser.error("--corpus-hash, --rerank-model and --top-k are required with --scores")
        rows = [ScoredCase.model_validate(row) for row in _jsonl(args.scores)]
        active_corpus=args.corpus_hash; rerank_model=args.rerank_model; top_k=args.top_k
    _validate_scores(manifest,rows)
    candidates = sorted({0.0, 1.0, *(row.score for row in rows)})
    report = select_threshold(
        rows,
        candidate_thresholds=candidates,
        metadata={
            "dataset_hash": _sha256(args.manifest),
            "corpus_hash": active_corpus,
            "rerank_model": rerank_model,
            "top_k": top_k,
        },
    )
    save_report(report, args.output)
    print(json.dumps(report.model_dump(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
