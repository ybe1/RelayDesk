import json
from pathlib import Path
from types import SimpleNamespace


def test_assess_retrieval_requires_source_category_fact_and_tool_contract():
    from scripts.eval_ch03 import assess_retrieval_sample

    sample = {
        "id": "shipping-paraphrase",
        "question": "邮费是多少",
        "expected_source": "shipping.md",
        "expected_category": "配送费用",
        "expected_facts": ["满 99 元包邮", "不足 99 元运费 8 元"],
    }
    hits = [{
        "id": 12, "score": 0.83, "question": "配送费用",
        "answer": "演示规则：满 99 元包邮；不足 99 元运费 8 元。",
        "category": "配送费用",
    }]

    result = assess_retrieval_sample(sample, hits, lambda row_id: "shipping.md")

    assert result["passed"] is True
    assert result["hits"] == [{"id": 12, "score": 0.83, "source": "shipping.md", "category": "配送费用"}]
    assert result["tool_contract"] is True
    assert result["citation"] is True


def test_assess_retrieval_no_hit_requires_empty_results_and_valid_contract():
    from scripts.eval_ch03 import assess_retrieval_sample

    result = assess_retrieval_sample(
        {"id": "astronomy-no-hit", "question": "太阳为什么会发光", "expect_no_hit": True},
        [],
        lambda row_id: None,
    )

    assert result["passed"] is True
    assert result["hits"] == []
    assert result["tool_contract"] is True


def test_assess_mining_compares_normalized_labeled_pairs():
    from scripts.eval_ch03 import assess_mining_sample

    sample = {
        "id": "general-return",
        "expected": [{"question": "退货期限是多久？", "answer": "商品支持七天无理由退货。"}],
    }
    actual = [{"conversation_id": 1, "question": " 退货期限是多久？", "answer": "商品支持七天无理由退货。 "}]

    result = assess_mining_sample(sample, actual)

    assert result == {"id": "general-return", "passed": True, "expected_count": 1, "actual_count": 1}


def test_mining_evaluator_filters_private_order_text_before_model_call():
    from scripts.eval_ch03 import safe_conversation_messages

    messages = safe_conversation_messages({
        "conversation": [
            ["user", "我的订单 A100 到哪了？"],
            ["assistant", "订单 A100 的物流单号为 SF123456，今天已到上海分拨中心。"],
        ]
    })

    assert messages == []


def test_private_only_mining_sample_skips_the_external_model():
    from scripts.eval_ch03 import _extract_one

    class ForbiddenModel:
        def with_structured_output(self, *_args, **_kwargs):
            raise AssertionError("private-only sample reached external model")

    sample = {"conversation": [
        ["user", "我的订单 A100 到哪了？"],
        ["assistant", "订单 A100 的物流单号为 SF123456，今天已到上海分拨中心。"],
    ]}

    assert _extract_one(ForbiddenModel(), sample) == []


def test_compose_health_requires_each_required_service_to_be_running_and_healthy():
    from scripts.eval_ch03 import _compose_services_healthy

    completed = SimpleNamespace(returncode=0, stdout=json.dumps([
        {"Service": "etcd", "State": "running", "Health": "healthy"},
        {"Service": "minio", "State": "running", "Health": "unhealthy"},
        {"Service": "standalone", "State": "running", "Health": "healthy"},
    ]))

    assert _compose_services_healthy(run=lambda *_args, **_kwargs: completed) is False


def test_compose_health_decodes_docker_json_as_utf8():
    from scripts.eval_ch03 import _compose_services_healthy

    received = {}
    completed = SimpleNamespace(returncode=1, stdout="")

    def run(*_args, **kwargs):
        received.update(kwargs)
        return completed

    assert _compose_services_healthy(run=run) is False
    assert received["encoding"] == "utf-8"
    assert received["errors"] == "replace"


def test_docker_cli_discovery_falls_back_to_windows_desktop_location():
    from scripts.eval_ch03 import _find_docker_cli

    candidate = Path(r"C:\Users\DELL\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe")
    found = _find_docker_cli(
        which=lambda _name: None,
        environ={"LOCALAPPDATA": r"C:\Users\DELL\AppData\Local", "ProgramFiles": r"C:\Program Files"},
        is_file=lambda path: path == candidate,
    )

    assert found == str(candidate)


def test_docker_cli_discovery_returns_none_when_path_and_candidates_are_absent():
    from scripts.eval_ch03 import _find_docker_cli

    assert _find_docker_cli(which=lambda _name: None, environ={}, is_file=lambda _path: False) is None


def test_compose_health_accepts_compose_v5_json_lines():
    from scripts.eval_ch03 import _compose_services_healthy

    rows = [
        {"Service": "etcd", "State": "running", "Health": "healthy"},
        {"Service": "minio", "State": "running", "Health": "healthy"},
        {"Service": "standalone", "State": "running", "Health": "healthy"},
    ]
    completed = SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(row) for row in rows))

    assert _compose_services_healthy(run=lambda *_args, **_kwargs: completed, docker_cli="docker") is True


def test_retrieval_gates_report_embedding_failure_without_raising(monkeypatch):
    import scripts.eval_ch03 as evaluator

    monkeypatch.setattr(evaluator, "settings", SimpleNamespace(embed_api_key="supersecret", chat_api_key="", embed_base_url="x", embed_model="m"))
    gates, errors = evaluator._retrieval_gates(
        compose_healthy=lambda: True,
        mysql_probe=lambda: None,
        milvus_probe=lambda: None,
        embedding_probe=lambda: (_ for _ in ()).throw(RuntimeError("network unavailable")),
    )

    assert gates["embedding"] is False
    assert errors == {"embedding": "RuntimeError"}


def test_retrieval_gate_reports_only_the_validated_embedding_dimension(monkeypatch):
    import scripts.eval_ch03 as evaluator

    monkeypatch.setattr(evaluator, "settings", SimpleNamespace(embed_api_key="supersecret", chat_api_key="", embed_base_url="x", embed_model="m"))
    gates, errors = evaluator._retrieval_gates(
        compose_healthy=lambda: True,
        mysql_probe=lambda: None,
        milvus_probe=lambda: None,
        embedding_probe=lambda: 1024,
    )

    assert gates["embedding"] is True
    assert gates["embedding_dimension"] == 1024
    assert "supersecret" not in json.dumps(gates)
    assert errors == {}


def test_mining_gates_do_not_probe_retrieval_dependencies(monkeypatch):
    import scripts.eval_ch03 as evaluator

    monkeypatch.setattr(evaluator, "settings", SimpleNamespace(chat_api_key="key"))
    model = object()
    gates, errors, actual = evaluator._mining_gates(model_factory=lambda **_kwargs: model)

    assert gates == {"chat_api_key": True, "chat_model": True}
    assert errors == {}
    assert actual is model


def test_evaluation_exception_returns_json_report_and_gate_exit_code(monkeypatch):
    import scripts.eval_ch03 as evaluator

    monkeypatch.setattr(evaluator, "_retrieval_gates", lambda: ({
        "docker_compose": True, "embed_api_key": True, "embedding": True,
        "mysql": True, "milvus": True,
    }, {}))
    monkeypatch.setattr(evaluator, "_load_samples", lambda _path: [{"id": "one", "question": "邮费是多少"}])
    monkeypatch.setattr(evaluator, "_run_retrieval_sample", lambda *_args: (_ for _ in ()).throw(OSError("network")))

    report, code = evaluator.evaluate_retrieval()

    assert code == 2
    assert set(report["errors"]) == {"evaluation"}


def test_report_schema_is_json_serializable_and_requires_per_sample_records():
    from scripts.eval_ch03 import build_report

    report = build_report("retrieval", [{"id": "one", "passed": True}], gates={"docker": True})

    assert json.loads(json.dumps(report, ensure_ascii=False)) == report
    assert report["summary"] == {"passed": 1, "total": 1, "pass_rate": 1.0}
