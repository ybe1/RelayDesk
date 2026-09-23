"""Labeled live acceptance evaluators for the Ch03 dense knowledge system.

The evaluator is intentionally separate from online retrieval: it may use
MySQL after a Milvus hit only to cite the chunk's source document.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def _find_docker_cli(
    *,
    which: Callable[[str], str | None] = shutil.which,
    environ: dict[str, str] | None = None,
    is_file: Callable[[Path], bool] | None = None,
) -> str | None:
    """Find Docker without depending on a Windows PATH mutation."""
    located = which("docker")
    if located:
        return located
    environ = os.environ if environ is None else environ
    is_file = Path.is_file if is_file is None else is_file
    candidates = [
        Path(environ["LOCALAPPDATA"]) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe"
        for name in ("LOCALAPPDATA",) if environ.get(name)
    ] + [
        Path(environ["ProgramFiles"]) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe"
        for name in ("ProgramFiles",) if environ.get(name)
    ]
    for candidate in candidates:
        if is_file(candidate):
            return str(candidate)
    return None


def _compose_rows(stdout: str) -> list[dict] | None:
    """Accept both Compose's JSON array and v5 newline-delimited JSON records."""
    try:
        parsed = json.loads(stdout)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except (TypeError, json.JSONDecodeError):
        try:
            rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        except (AttributeError, json.JSONDecodeError):
            return None
    return rows if all(isinstance(row, dict) for row in rows) else None


def _compose_services_healthy(run: Callable = subprocess.run, docker_cli: str | None = None) -> bool:
    """Require every standalone dependency to be both running and healthy."""
    docker_cli = docker_cli or _find_docker_cli()
    if docker_cli is None:
        return False
    try:
        completed = run(
            [docker_cli, "compose", "-f", "compose.yaml", "-f", "compose.milvus.yaml", "ps", "--format", "json"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=15, shell=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    rows = _compose_rows(completed.stdout)
    if rows is None:
        return False
    health = {
        row.get("Service"): (str(row.get("State", "")).lower(), str(row.get("Health", "")).lower())
        for row in rows if isinstance(row, dict)
    }
    return all(health.get(name) == ("running", "healthy") for name in ("etcd", "minio", "standalone"))


def _mysql_probe() -> None:
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))


def _milvus_probe() -> None:
    from app.knowledge.vector_store import MilvusStore
    MilvusStore(uri=settings.milvus_uri).count()


def _embedding_probe() -> int:
    from app.knowledge.embedding import EmbeddingClient
    vectors = EmbeddingClient(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    ).embed(["Ch03 evaluator connectivity probe"])
    if len(vectors) != 1:
        raise ValueError("embedding probe returned an unexpected vector count")
    return len(vectors[0])


def _retrieval_gates(
    *,
    compose_healthy: Callable[[], bool] = _compose_services_healthy,
    mysql_probe: Callable[[], None] = _mysql_probe,
    milvus_probe: Callable[[], None] = _milvus_probe,
    embedding_probe: Callable[[], int] = _embedding_probe,
) -> tuple[dict[str, Any], dict[str, str]]:
    gates = {
        "docker_compose": False,
        "embed_api_key": bool(settings.embed_api_key),
        "embedding": False,
        "embedding_dimension": None,
        "chat_api_key": bool(settings.chat_api_key),
        "mysql": False,
        "milvus": False,
    }
    errors: dict[str, str] = {}
    for name, probe in (("docker_compose", compose_healthy), ("mysql", mysql_probe), ("milvus", milvus_probe)):
        try:
            gates[name] = bool(probe()) if name == "docker_compose" else True
        except Exception as error:
            errors[name] = type(error).__name__
    if gates["embed_api_key"]:
        try:
            gates["embedding_dimension"] = embedding_probe()
            gates["embedding"] = True
        except Exception as error:
            errors["embedding"] = type(error).__name__
    return gates, errors


def _mining_gates(model_factory: Callable[..., Any] | None = None) -> tuple[dict[str, bool], dict[str, str], Any | None]:
    """Mining is independent of Docker, MySQL, Milvus, and embedding services."""
    gates = {"chat_api_key": bool(settings.chat_api_key), "chat_model": False}
    errors: dict[str, str] = {}
    if not gates["chat_api_key"]:
        return gates, errors, None
    if model_factory is None:
        from app.core.llm import get_chat_model
        model_factory = get_chat_model
    try:
        model = model_factory(disable_thinking=True)
        gates["chat_model"] = True
        return gates, errors, model
    except Exception as error:
        errors["chat_model"] = type(error).__name__
        return gates, errors, None


def _source_lookup() -> Callable[[int], str | None]:
    with SessionLocal() as session:
        rows = session.execute(select(KnowledgeChunk.id, KnowledgeChunk.section_path)).all()
    source_by_id = {int(row.id): _source_name(row.section_path) for row in rows}
    return source_by_id.get


def evaluate_retrieval() -> tuple[dict[str, Any], int]:
    gates, errors = _retrieval_gates()
    required = ("docker_compose", "embed_api_key", "embedding", "mysql", "milvus")
    if not all(gates[name] for name in required):
        report = build_report("retrieval", [], gates=gates)
        report["errors"] = errors
        return report, 2

    from app.knowledge.embedding import EmbeddingClient
    from app.knowledge.retrieval import search_knowledge
    from app.knowledge.vector_store import MilvusStore

    try:
        embedder = EmbeddingClient(base_url=settings.embed_base_url, api_key=settings.embed_api_key, model=settings.embed_model)
        store = MilvusStore(uri=settings.milvus_uri)
        source_lookup = _source_lookup()
        results = [_run_retrieval_sample(sample, embedder, store, source_lookup) for sample in _load_samples(RETRIEVAL_SAMPLES)]
    except Exception as error:
        report = build_report("retrieval", [], gates=gates)
        report["errors"] = {"evaluation": type(error).__name__}
        return report, 2
    report = build_report("retrieval", results, gates=gates)
    return report, 0 if report["summary"]["passed"] >= 4 and results[0]["passed"] and results[-1]["passed"] else 1


def _extract_one(model: Any, sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = safe_conversation_messages(sample)
    if not messages:
        return []
    structured = model.with_structured_output(QASet, method="function_calling")
    output = structured.invoke(EXTRACTION_PROMPT.format_messages(
        conversations=json.dumps([{"conversation_id": 1, "messages": messages}], ensure_ascii=False)
    ))
    result = output if isinstance(output, QASet) else QASet.model_validate(output)
    return [item.model_dump() for item in result.items]


def _run_retrieval_sample(sample: dict[str, Any], embedder: Any = None, store: Any = None, source_lookup: Callable[[int], str | None] | None = None) -> dict[str, Any]:
    """Run one retrieval sample; optional defaults make isolated error tests simple."""
    if embedder is None or store is None or source_lookup is None:
        raise RuntimeError("retrieval dependencies are required")
    from app.knowledge.retrieval import search_knowledge
    return assess_retrieval_sample(
        sample,
        search_knowledge(sample["question"], embedder, store, settings.rag_top_k, settings.rag_min_score),
        source_lookup,
    )


def evaluate_mining() -> tuple[dict[str, Any], int]:
    gates, errors, model = _mining_gates()
    required = ("chat_api_key", "chat_model")
    if not all(gates[name] for name in required):
        report = build_report("mining", [], gates=gates)
        report["errors"] = errors
        return report, 2

    try:
        results = [assess_mining_sample(sample, _extract_one(model, sample)) for sample in _load_samples(MINING_SAMPLES)]
    except Exception as error:
        report = build_report("mining", [], gates=gates)
        report["errors"] = {"evaluation": type(error).__name__}
        return report, 2
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
