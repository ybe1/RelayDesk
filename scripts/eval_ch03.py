"""Labeled live acceptance evaluators for the Ch03 dense knowledge system.

The evaluator is intentionally separate from online retrieval: it may use
MySQL after a Milvus hit only to cite the chunk's source document.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select, text

from app.config import settings
from app.db.models import KnowledgeChunk
from app.db.session import SessionLocal
from app.knowledge.dedup import qa_fingerprint
from app.knowledge.extraction import QASet, _safe_messages
from app.knowledge.extraction_prompts import EXTRACTION_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_SAMPLES = PROJECT_ROOT / "tests" / "data" / "ch03_retrieval_samples.json"
MINING_SAMPLES = PROJECT_ROOT / "tests" / "data" / "ch03_extraction_samples.json"
REPORT_SCHEMA = "ch03-eval/v1"


def _source_name(section_path: str | None) -> str | None:
    return None if not section_path else section_path.split(" / ", 1)[0]


def _tool_contract(question: str, hits: list[dict]) -> bool:
    """Exercise the public FAQ tool shape without allowing a database read."""
    from app.tools.business import build_tools

    def forbidden_session():
        raise AssertionError("evaluation FAQ tool opened MySQL")

    tool = {item.name: item for item in build_tools(forbidden_session, 1, retriever=lambda _: hits)}["query_faq"]
    payload = json.loads(tool.invoke({"keyword": question}))
    expected_matches = [
        {"question": hit["question"], "answer": hit["answer"], "category": hit["category"]}
        for hit in hits
    ]
    return payload == {"keyword": question, "matches": expected_matches}


def assess_retrieval_sample(
    sample: dict[str, Any], hits: list[dict], source_lookup: Callable[[int], str | None]
) -> dict[str, Any]:
    """Assess one live retrieval against source, category, answer facts and API shape."""
    annotated_hits = [
        {
            "id": hit["id"],
            "score": hit["score"],
            "source": source_lookup(int(hit["id"])),
            "category": hit["category"],
        }
        for hit in hits
    ]
    tool_contract = _tool_contract(sample["question"], hits)
    if sample.get("expect_no_hit"):
        passed = not hits and tool_contract
        return {
            "id": sample["id"], "question": sample["question"], "passed": passed,
            "expect_no_hit": True, "hits": annotated_hits, "citation": not hits,
            "tool_contract": tool_contract,
        }

    expected_source = sample["expected_source"]
    expected_category = sample["expected_category"]
    expected_facts = sample["expected_facts"]
    match = next(
        (
            hit for hit in hits
            if source_lookup(int(hit["id"])) == expected_source
            and expected_category in hit["category"]
            and all(fact in hit["answer"] for fact in expected_facts)
        ),
        None,
    )
    return {
        "id": sample["id"], "question": sample["question"], "passed": bool(match) and tool_contract,
        "expected_source": expected_source, "expected_category": expected_category,
        "hits": annotated_hits, "citation": bool(match), "tool_contract": tool_contract,
    }


def assess_mining_sample(sample: dict[str, Any], actual: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare structured model output with a labeled sample using QA fingerprints."""
    expected_pairs = {
        qa_fingerprint(item["question"], item["answer"])
        for item in sample["expected"]
    }
    actual_pairs = {
        qa_fingerprint(item["question"], item["answer"])
        for item in actual
    }
    return {
        "id": sample["id"], "passed": expected_pairs == actual_pairs,
        "expected_count": len(expected_pairs), "actual_count": len(actual_pairs),
    }


def safe_conversation_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Apply the production mining privacy filter before any external model call."""
    rows = [
        SimpleNamespace(role=role, content=content, tool_calls=None)
        for role, content in sample["conversation"]
    ]
    return _safe_messages(rows)


def build_report(mode: str, results: list[dict[str, Any]], *, gates: dict[str, bool]) -> dict[str, Any]:
    passed = sum(bool(result.get("passed")) for result in results)
    total = len(results)
    return {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "gates": gates,
        "results": results,
        "summary": {"passed": passed, "total": total, "pass_rate": passed / total if total else 0.0},
    }


def _load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"invalid labeled sample file: {path}")
    return payload


def _docker_healthy() -> bool:
    try:
        completed = subprocess.run(
            ["docker", "compose", "-f", "compose.yaml", "-f", "compose.milvus.yaml", "ps"],
            cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=15, shell=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _retrieval_gates() -> dict[str, bool]:
    gates = {
        "docker_compose": _docker_healthy(),
        "embed_api_key": bool(settings.embed_api_key),
        "chat_api_key": bool(settings.chat_api_key),
        "mysql": False,
        "milvus": False,
    }
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        gates["mysql"] = True
    except Exception:
        pass
    try:
        from app.knowledge.vector_store import MilvusStore
        MilvusStore(uri=settings.milvus_uri).count()
        gates["milvus"] = True
    except Exception:
        pass
    return gates


def _source_lookup() -> Callable[[int], str | None]:
    with SessionLocal() as session:
        rows = session.execute(select(KnowledgeChunk.id, KnowledgeChunk.section_path)).all()
    source_by_id = {int(row.id): _source_name(row.section_path) for row in rows}
    return source_by_id.get


def evaluate_retrieval() -> tuple[dict[str, Any], int]:
    gates = _retrieval_gates()
    required = ("docker_compose", "embed_api_key", "mysql", "milvus")
    if not all(gates[name] for name in required):
        return build_report("retrieval", [], gates=gates), 2

    from app.knowledge.embedding import EmbeddingClient
    from app.knowledge.retrieval import search_knowledge
    from app.knowledge.vector_store import MilvusStore

    embedder = EmbeddingClient(base_url=settings.embed_base_url, api_key=settings.embed_api_key, model=settings.embed_model)
    store = MilvusStore(uri=settings.milvus_uri)
    source_lookup = _source_lookup()
    results = [
        assess_retrieval_sample(
            sample,
            search_knowledge(sample["question"], embedder, store, settings.rag_top_k, settings.rag_min_score),
            source_lookup,
        )
        for sample in _load_samples(RETRIEVAL_SAMPLES)
    ]
    report = build_report("retrieval", results, gates=gates)
    return report, 0 if report["summary"]["passed"] >= 4 and results[0]["passed"] and results[-1]["passed"] else 1


def _extract_one(model: Any, sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = safe_conversation_messages(sample)
    structured = model.with_structured_output(QASet, method="function_calling")
    output = structured.invoke(EXTRACTION_PROMPT.format_messages(
        conversations=json.dumps([{"conversation_id": 1, "messages": messages}], ensure_ascii=False)
    ))
    result = output if isinstance(output, QASet) else QASet.model_validate(output)
    return [item.model_dump() for item in result.items]


def evaluate_mining() -> tuple[dict[str, Any], int]:
    gates = _retrieval_gates()
    required = ("docker_compose", "chat_api_key", "mysql", "milvus")
    if not all(gates[name] for name in required):
        return build_report("mining", [], gates=gates), 2

    from app.core.llm import get_chat_model
    model = get_chat_model(disable_thinking=True)
    results = [assess_mining_sample(sample, _extract_one(model, sample)) for sample in _load_samples(MINING_SAMPLES)]
    report = build_report("mining", results, gates=gates)
    return report, 0 if all(result["passed"] for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Ch03 labeled live acceptance evaluators")
    parser.add_argument("mode", choices=("retrieval", "mining"))
    args = parser.parse_args(argv)
    report, code = evaluate_retrieval() if args.mode == "retrieval" else evaluate_mining()
    for result in report["results"]:
        print(json.dumps(result, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
