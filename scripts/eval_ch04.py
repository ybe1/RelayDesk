"""Four-strategy Ch04 evaluation with atomic checkpoints and separate phases.

Examples (all paths are relative to the repository unless overridden):
  python -m scripts.eval_ch04 --phase retrieval
  python -m scripts.eval_ch04 --phase generation --generation-limit 15
  python -m scripts.eval_ch04 --phase all --retry-skipped

Existing reports are resumed by default. --fresh starts a new run. Failed remote
calls are never automatically replayed across runs; --retry-skipped opts in.
The generation limit is per strategy, stratified across bucket/difficulty cells.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

from app.evaluation.ch04_metrics import answer_point_coverage, refusal_rate, retrieval_metrics
from app.evaluation.report_lock import ReportBusy, report_lock
from app.evaluation.faithfulness import (
    FaithfulnessDecision, GeneratedAnswer, citation_snapshot, generate_answer, judge_faithfulness,
)
from scripts.validate_eval_ch04 import BUCKETS, DEFAULT_DATA, validate_records

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "data/ch04/reports/rag_eval.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(cases):
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def select_generation_ids(cases: list[dict], limit: int | None) -> set[str]:
    cells = defaultdict(list)
    for case in cases:
        cells[case["bucket"], case["difficulty"]].append(case["id"])
    ordered = [values[index] for index in range(max(map(len, cells.values()), default=0))
               for values in cells.values() if index < len(values)]
    return set(ordered if limit is None else ordered[:limit])


def _new_report(cases, judge_model):
    return {"schema_version": 1, "meta": {
        "started_at": _now(), "dataset_sha256": _fingerprint(cases), "case_count": len(cases),
        "strategies": list(STRATEGIES), "top_k": 10, "judge_model": judge_model,
        "generation_requested": False,
        "metric_notes": {
            "retrieval": "original rank; group means per case, then means across A/B/C/E cases; D excluded",
            "evidence_coverage": "whitespace-normalized literal points in allowed sections within Top-10; group means",
            "answer_point_coverage": "whitespace-normalized literal points in answer; group means; not semantic scoring",
            "failures": "skipped calls excluded from metric denominators; counts always reported",
            "refusal": "completed D answers only; D excluded from all other metrics and judging",
        }}, "cases": {strategy: [{**deepcopy(case), "retrieval": {"status": "pending"},
            "generation": {"status": "skipped", "reason": "not_requested"},
            "faithfulness": {"status": "skipped", "reason": "not_requested"}}
            for case in cases] for strategy in STRATEGIES}, "warnings": []}


def _average(values):
    return mean(values) if values else None


def _failure(error):
    result = {"status": "skipped", "reason": type(error).__name__}
    status = getattr(error, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        result["http_status"] = status
    return result


def summarize(report):
    """One source of denominators for saved JSON, text reports and downstream UI."""
    retrieval, evidence, generation = {}, {}, {}
    for strategy, rows in report["cases"].items():
        retrieval[strategy], evidence[strategy], generation[strategy] = {}, {}, {}
        subsets = {"overall": [row for row in rows if row["bucket"] != "D_out_of_scope"],
                   **{bucket: [row for row in rows if row["bucket"] == bucket] for bucket in BUCKETS}}
        for label, subset in subsets.items():
            completed = [row for row in subset if row["retrieval"]["status"] == "ok"]
            generated = [row for row in subset if row["generation"]["status"] == "ok"]
            counts = {"case_count": len(subset), "completed": len(completed), "skipped": len(subset) - len(completed)}
            gen_counts = {"case_count": len(subset), "completed": len(generated), "skipped": len(subset) - len(generated)}
            if label == "D_out_of_scope":
                ret_values = cov_values = {"case_count": len(subset), "excluded": True}
                gen_values = {**gen_counts, "refusal_rate": refusal_rate([row["generation"] for row in subset])}
            else:
                ret_values = {**counts, **{key: _average([row["metrics"][key] for row in completed])
                                          for key in ("recall_at_5", "mrr")}}
                cov_values = {**counts, "coverage": _average([row["metrics"]["evidence_coverage"] for row in completed])}
                gen_values = {**gen_counts, "answer_point_coverage": _average([
                    row["generation"]["answer_point_coverage"] for row in generated])}
            for dest, value in ((retrieval[strategy], ret_values), (evidence[strategy], cov_values),
                                (generation[strategy], gen_values)):
                if label == "overall":
                    dest["overall"] = value
                else:
                    dest.setdefault("buckets", {})[label] = value
    all_rows = [row for rows in report["cases"].values() for row in rows]
    eligible = [row for row in report["cases"]["hybrid_rerank"] if row["bucket"] != "D_out_of_scope"]
    judged = [row for row in eligible if row["faithfulness"]["status"] == "ok"]
    hallucinated = [row["id"] for row in judged if not row["faithfulness"]["faithful"]]
    faithful = {"case_count": len(eligible), "completed": len(judged), "skipped": len(eligible) - len(judged),
                "faithfulness_rate": _average([float(row["faithfulness"]["faithful"]) for row in judged]),
                "hallucination_rate": len(hallucinated) / len(judged) if judged else None,
                "hallucinated_case_ids": hallucinated,
                "buckets": {bucket: {"completed": sum(row["bucket"] == bucket for row in judged),
                    "faithfulness_rate": _average([float(row["faithfulness"]["faithful"])
                        for row in judged if row["bucket"] == bucket])}
                    for bucket in BUCKETS if bucket != "D_out_of_scope"}}
    full_generation = all(row["generation"]["status"] == "ok" for row in all_rows) and len(judged) == len(eligible)
    report.update(retrieval=retrieval, evidence_coverage=evidence,
                  generation={"status": "complete" if full_generation else (
                      "partial" if report["meta"]["generation_requested"] else "not_requested"),
                      "strategies": generation, "faithfulness": faithful})
    meta = report["meta"]
    meta.update(updated_at=_now(), retrieval_complete=all(row["retrieval"]["status"] != "pending" for row in all_rows),
                retrieval_completed=sum(row["retrieval"]["status"] == "ok" for row in all_rows),
                retrieval_skipped=sum(row["retrieval"]["status"] == "skipped" for row in all_rows))
    meta["retrieval_status"] = "complete" if meta["retrieval_completed"] == len(all_rows) else (
        "partial" if meta["retrieval_complete"] else "running")
    comparable = [(name, values["overall"]["mrr"]) for name, values in retrieval.items()
                  if values["overall"]["completed"] == values["overall"]["case_count"]
                  and values["overall"]["mrr"] is not None]
    report["best"] = dict(zip(("strategy", "mrr"), max(comparable, key=lambda item: item[1]))) if comparable else None


def render_text(report):
    def number(value):
        return "N/A" if value is None else f"{value:.6f}"
    lines = ["Ch04 RAG evaluation", f"Cases: {report['meta']['case_count']}",
             f"Retrieval: {report['meta']['retrieval_status']} ({report['meta']['retrieval_completed']} completed, "
             f"{report['meta']['retrieval_skipped']} skipped)", f"Generation: {report['generation']['status']}",
             "strategy | bucket | n/completed/skipped | Recall@5 | MRR | evidence coverage | answer point coverage"]
    for strategy in STRATEGIES:
        for bucket in ("overall", *BUCKETS):
            if bucket == "D_out_of_scope":
                values = report["generation"]["strategies"][strategy]["buckets"][bucket]
                lines.append(f"{strategy} | {bucket} | refusal={number(values['refusal_rate'])} | "
                             f"generated={values['completed']}/{values['case_count']}")
                continue
            def part(section):
                return section[strategy]["overall"] if bucket == "overall" else section[strategy]["buckets"][bucket]
            ret, cov, gen = part(report["retrieval"]), part(report["evidence_coverage"]), part(report["generation"]["strategies"])
            lines.append(f"{strategy} | {bucket} | {ret['case_count']}/{ret['completed']}/{ret['skipped']} | "
                         f"{number(ret['recall_at_5'])} | {number(ret['mrr'])} | {number(cov['coverage'])} | "
                         f"{number(gen['answer_point_coverage'])} (generated={gen['completed']}/{gen['case_count']})")
    faith = report["generation"]["faithfulness"]
    lines.extend([f"hybrid_rerank faithfulness={number(faith['faithfulness_rate'])}; "
                  f"hallucination={number(faith['hallucination_rate'])}; judged={faith['completed']}/{faith['case_count']}",
                  f"Hallucinated IDs this run: {', '.join(faith['hallucinated_case_ids']) or 'none'}",
                  f"Warnings: {len(report['warnings'])}", "Metric definitions:"])
    lines.extend(f"- {key}: {value}" for key, value in report["meta"]["metric_notes"].items())
    return "\n".join(lines) + "\n"


def save_report(report, path):
    summarize(report)
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace a sibling temp file atomically; interruption preserves the prior snapshot.
    for target, content in ((path, json.dumps(report, ensure_ascii=False, separators=(",", ":"))),
                            (path.with_suffix(".txt"), render_text(report))):
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        # Windows readers/antivirus can briefly deny atomic replacement. Keep the
        # old snapshot intact and retry only this local operation, never an LLM call.
        for attempt in range(7):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.05 * (attempt + 1))


def persist_faith_case(*, session_factory=None, **payload):
    from app.repositories.faith_cases import upsert_faith_case
    if session_factory is None:
        from app.db.session import SessionLocal
        session_factory = SessionLocal
    with session_factory() as session:
        with session.begin():
            upsert_faith_case(session, **payload)


async def evaluate(cases, *, retriever, generator=None, judge=None, ledger=None, judge_model=None,
                   report_path=None, resume=None, phase="all", concurrency=2, timeout=90.0,
                   generation_limit=None, retry_skipped=False, metadata=None, progress=None):
    if phase not in {"all", "retrieval", "generation"} or not 1 <= concurrency <= 4 or timeout <= 0:
        raise ValueError("invalid phase, concurrency (1..4), or timeout")
    if generation_limit is not None and generation_limit < 0:
        raise ValueError("generation_limit must be nonnegative")
    report = deepcopy(resume) if resume is not None else _new_report(cases, judge_model)
    if report.get("schema_version") != 1 or report["meta"]["dataset_sha256"] != _fingerprint(cases):
        raise ValueError("checkpoint dataset/schema differs; choose a new report or --fresh")
    if metadata is not None:
        previous = report["meta"].get("configuration")
        if previous is not None and previous != metadata:
            raise ValueError("checkpoint configuration differs; choose a new report or --fresh")
        report["meta"]["configuration"] = metadata
    semaphore = asyncio.Semaphore(concurrency)
    events = 0
    def checkpoint():
        nonlocal events
        save_report(report, report_path)
        events += 1
        if progress is not None and (events % 25 == 0 or events == 1):
            progress({"events": events, "phase": phase, "retrieval_completed": report["meta"]["retrieval_completed"],
                      "retrieval_skipped": report["meta"]["retrieval_skipped"],
                      "generation_completed": sum(row["generation"]["status"] == "ok"
                          for rows in report["cases"].values() for row in rows),
                      "judge_completed": report["generation"]["faithfulness"]["completed"],
                      "generation_status": report["generation"]["status"]})
    def should_run(state):
        return state["status"] == "pending" or (retry_skipped and state["status"] == "skipped")
    async def retrieve_case(index):
        async with semaphore:
            for strategy in STRATEGIES:
                row = report["cases"][strategy][index]
                if not should_run(row["retrieval"]):
                    continue
                try:
                    result = await asyncio.wait_for(retriever(cases[index], strategy), timeout=timeout)
                    metrics = retrieval_metrics(row, result.hits)
                    row["retrieval"] = {"status": "ok", "hits": result.hits, "sufficient": result.sufficient,
                        "reason": result.reason, "understood_query": result.understood_query.model_dump()}
                    row["metrics"] = metrics
                except Exception as error:
                    row["retrieval"] = _failure(error)
                checkpoint()
    if phase != "generation":
        await asyncio.gather(*(retrieve_case(index) for index in range(len(cases))))
    summarize(report)
    if not report["meta"]["retrieval_complete"]:
        raise ValueError("all four retrieval strategies must finish before generation")
    checkpoint()  # The complete retrieval report exists before the first generation.
    if generator is not None and phase != "retrieval":
        report["meta"]["generation_requested"] = True
        blocked_reason = report["meta"].get("generation_blocked_reason") if not retry_skipped else None
        if retry_skipped:
            report["meta"].pop("generation_blocked_reason", None)
        def cloud_failure(error):
            nonlocal blocked_reason
            failure = _failure(error)
            if failure.get("http_status") == 402:
                blocked_reason = "upstream_http_402"
                report["meta"]["generation_blocked_reason"] = blocked_reason
            return failure
        selected = select_generation_ids(cases, generation_limit)
        report["meta"]["generation_selection"] = {
            "limit_per_strategy": generation_limit, "method": "round_robin_bucket_difficulty",
            "selected_case_ids": [case["id"] for case in cases if case["id"] in selected],
        }
        async def generate_case(row, strategy):
            async with semaphore:
                state = row["generation"]
                if row["retrieval"]["status"] != "ok":
                    row["generation"] = {"status": "skipped", "reason": "retrieval_unavailable"}
                    return
                if row["id"] not in selected:
                    if state["status"] != "ok":
                        row["generation"] = {"status": "skipped", "reason": "generation_limit"}
                    return
                if blocked_reason and state["status"] != "ok":
                    row["generation"] = {"status": "skipped", "reason": blocked_reason, "http_status": 402}
                    checkpoint()
                    return
                eligible = should_run(state) or state.get("reason") in {"not_requested", "generation_limit", "retrieval_unavailable"}
                if eligible:
                    try:
                        output = await asyncio.wait_for(generator(row, row["retrieval"]), timeout=timeout)
                        result = GeneratedAnswer.model_validate(output)
                        row["generation"] = {"status": "ok", **result.model_dump(),
                            "answer_point_coverage": answer_point_coverage(row, result.answer)}
                    except Exception as error:
                        row["generation"] = cloud_failure(error)
                    checkpoint()

        await asyncio.gather(*(generate_case(row, strategy) for strategy in STRATEGIES for row in report["cases"][strategy]))
        checkpoint()
        # A smoke/continuation limit cannot relax the full-dataset judge barrier.
        generations_ready = all(row["generation"]["status"] == "ok"
            for rows in report["cases"].values() for row in rows)

        async def judge_case(row, strategy):
            async with semaphore:
                if strategy != "hybrid_rerank" or row["bucket"] == "D_out_of_scope":
                    row["faithfulness"] = {"status": "skipped", "reason": "not_applicable"}
                    return
                if row["id"] not in selected or judge is None:
                    return
                if not generations_ready:
                    if row["faithfulness"]["status"] != "ok":
                        row["faithfulness"] = {"status": "skipped", "reason": "generation_incomplete"}
                    return
                judged = row["faithfulness"]
                citations = citation_snapshot(row["retrieval"]["hits"])
                if blocked_reason and judged["status"] != "ok":
                    row["faithfulness"] = {"status": "skipped", "reason": blocked_reason, "http_status": 402}
                    checkpoint()
                    return
                if should_run(judged) or judged.get("reason") in {"not_requested", "generation_incomplete"}:
                    try:
                        output = await asyncio.wait_for(judge(row, row["generation"]["answer"], citations), timeout=timeout)
                        decision = FaithfulnessDecision.model_validate(output)
                        row["faithfulness"] = {"status": "ok", **decision.model_dump()}
                    except Exception as error:
                        row["faithfulness"] = cloud_failure(error)
                    checkpoint()
                judged = row["faithfulness"]
                if judged["status"] == "ok" and not judged["faithful"] and (
                    row.get("ledger", {}).get("status") is None or
                    (retry_skipped and row.get("ledger", {}).get("status") == "skipped")
                ):
                    try:
                        if ledger is None:
                            raise RuntimeError("ledger is not configured")
                        await asyncio.to_thread(ledger, eval_id=row["id"], bucket=row["bucket"], query=row["query"],
                            strategy=strategy, answer=row["generation"]["answer"], reason=judged["reason"],
                            citations=citations, judge_model=report["meta"]["judge_model"])
                        row["ledger"] = {"status": "ok"}
                    except Exception as error:
                        row["ledger"] = {"status": "skipped", "reason": type(error).__name__}
                        warning = {"stage": "ledger", "id": row["id"], "strategy": strategy, "error": type(error).__name__}
                        if warning not in report["warnings"]:
                            report["warnings"].append(warning)
                    checkpoint()
        await asyncio.gather(*(judge_case(row, strategy) for strategy in STRATEGIES for row in report["cases"][strategy]))
    checkpoint()
    return report


class CachedEmbedder:
    """Reuse identical query embeddings across retrieval strategies within a run."""
    def __init__(self, delegate):
        self.delegate = delegate

    @lru_cache(maxsize=512)
    def _one(self, text):
        return self.delegate.embed([text])[0]

    def embed(self, texts):
        return [self._one(text) for text in texts]


class LiveRetriever:
    def __init__(self, dependencies, resume=None):
        from app.knowledge.query_understanding import QueryUnderstanding
        self.dependencies = dependencies
        self.prepared = {}
        if resume:
            for rows in resume["cases"].values():
                for row in rows:
                    if row["retrieval"]["status"] == "ok":
                        self.prepared.setdefault(row["query"], QueryUnderstanding.model_validate(row["retrieval"]["understood_query"]))
        self.dependencies.embedder = CachedEmbedder(self.dependencies.embedder)

    async def __call__(self, case, strategy):
        from app.knowledge.query_understanding import understand_query
        from app.knowledge.retrieval import search_knowledge_ch04
        query = case["query"]
        if query not in self.prepared:
            self.prepared[query] = await understand_query(self.dependencies.model, query)
        return await search_knowledge_ch04(query, strategy=strategy, top_k=10, dependencies=self.dependencies,
                                           understood_query=self.prepared[query])


async def _live(args, cases, resume):
    from app.config import settings
    from app.core.llm import get_chat_model
    from app.knowledge.retrieval import RetrievalDependencies
    metadata = {"chat_model": settings.chat_model, "embedding_model": settings.embed_model,
        "rerank_model": settings.rerank_model, "rerank_min_score": settings.rerank_min_score,
        "category_allowlist": list(settings.rag_category_allowlist), "query_temperature": 0.3,
        "answer_temperature": 0.3, "judge_temperature": 0,
        "corpus_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in sorted(
            (Path(__file__).resolve().parents[1] / "data/kb").glob("*.md")))).hexdigest()}
    answer_model = get_chat_model(disable_thinking=True)
    judge_model = get_chat_model(disable_thinking=True, temperature=0)
    async def generator(row, result):
        return await generate_answer(answer_model, row["query"], result)
    async def judge(row, answer, citations):
        return await judge_faithfulness(judge_model, row["query"], answer, citations)
    options = dict(generator=generator if args.phase != "retrieval" else None, judge=judge,
        ledger=persist_faith_case, judge_model=settings.chat_model, report_path=args.report,
        resume=resume, phase=args.phase, concurrency=args.concurrency, timeout=args.timeout,
        generation_limit=args.generation_limit, retry_skipped=args.retry_skipped,
        metadata=metadata, progress=lambda value: print(json.dumps(value), flush=True))
    if args.phase == "generation":
        # Replaying saved evidence has no dependency on Milvus or embedding/rerank.
        return await evaluate(cases, retriever=None, **options)
    async with await RetrievalDependencies.afrom_settings() as deps:
        # Outer deadlines bound async calls. Sync embedding requests also need their
        # own deadline because cancelling asyncio.to_thread cannot kill a thread.
        deps.embedder.timeout, deps.embedder.max_retries = min(args.timeout, 20), 0
        report = await evaluate(cases, retriever=LiveRetriever(deps, resume), **options)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--phase", choices=("all", "retrieval", "generation"), default="all")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--generation-limit", type=int)
    parser.add_argument("--retry-skipped", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    try:
        # One owner from checkpoint read through all remote work and final output.
        # CLI and every JobRunner process use this same report-path lock.
        with report_lock(args.report):
            cases = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
            errors = validate_records(cases)
            if errors:
                raise ValueError("; ".join(errors))
            resume = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() and not args.fresh else None
            report = asyncio.run(_live(args, cases, resume))
            print(render_text(report), end="")
            if report["meta"]["retrieval_status"] != "complete":
                return 1
            if args.phase != "retrieval" and report["generation"]["status"] != "complete":
                return 2
            return 0
    except ReportBusy:
        print("Evaluation busy: another run owns this report; retry after it exits.", file=sys.stderr)
        return 3
    except (OSError, ValueError) as error:
        # Configuration, parse, and local I/O errors are actionable CLI failures.
        print(f"Evaluation could not start or save: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
