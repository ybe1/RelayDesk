import json


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


def test_report_schema_is_json_serializable_and_requires_per_sample_records():
    from scripts.eval_ch03 import build_report

    report = build_report("retrieval", [{"id": "one", "passed": True}], gates={"docker": True})

    assert json.loads(json.dumps(report, ensure_ascii=False)) == report
    assert report["summary"] == {"passed": 1, "total": 1, "pass_rate": 1.0}
