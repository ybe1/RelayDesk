"""Annotation gates: reject corrupt labels before they can skew evaluation."""

import copy
import json

import pytest


@pytest.fixture
def labeled(tmp_path):
    from scripts.validate_eval_ch04 import BUCKETS
    root = tmp_path / "kb"
    root.mkdir()
    (root / "returns.md").write_text("# 演示\n\n## 甲\n\n满 99 元包邮。\n\n## 乙\n\n七日内退货。\n\n## 替代\n\n满99元包邮。\n", encoding="utf-8")
    first = {"group_id": "fee", "allowed_section_paths": ["returns.md / 演示 / 甲", "returns.md / 演示 / 替代"], "expect_points": ["满99元包邮"]}
    second = {"group_id": "return", "allowed_section_paths": ["returns.md / 演示 / 乙"], "expect_points": ["七日内退货"]}
    rows = []
    for bucket in BUCKETS:
        for i in range(60):
            groups = [] if bucket == "D_out_of_scope" else [copy.deepcopy(first)]
            if bucket == "E_multi":
                groups.append(copy.deepcopy(second))
            rows.append({"id": f"{bucket}-{i}", "bucket": bucket,
                         "difficulty": ("easy", "medium", "hard")[i // 20],
                         "query": f"{bucket}问题{i}", "expected_groups": groups})
    return rows, root


def test_valid_alternatives_and_whitespace_normalized_points(labeled):
    from scripts.validate_eval_ch04 import validate_records
    rows, root = labeled
    assert validate_records(rows, root) == []


@pytest.mark.parametrize("fault", [
    "count", "bucket", "difficulty", "duplicate_id", "duplicate_query", "blank_query",
    "missing_field", "unknown_field", "missing_section", "missing_point", "empty_points",
    "alternative_missing_point", "duplicate_group", "overlapping_groups", "d_ground_truth",
    "e_single_group", "empty_groups", "group_schema", "non_object", "bad_paths",
    "blank_point", "bad_groups", "duplicate_point", "duplicate_path",
])
def test_invalid_labels_are_rejected(labeled, fault):
    from scripts.validate_eval_ch04 import validate_records
    rows, root = labeled
    group = rows[0]["expected_groups"][0]
    if fault == "count": rows.pop()
    elif fault == "bucket": rows[0]["bucket"] = "other"
    elif fault == "difficulty": rows[0]["difficulty"] = 1
    elif fault == "duplicate_id": rows[1]["id"] = rows[0]["id"]
    elif fault == "duplicate_query": rows[1]["query"] = "  " + rows[0]["query"] + "\n"
    elif fault == "blank_query": rows[0]["query"] = "  "
    elif fault == "missing_field": del rows[0]["expected_groups"]
    elif fault == "unknown_field": rows[0]["expect_points"] = ["old schema"]
    elif fault == "missing_section": group["allowed_section_paths"] = ["returns.md / 不存在"]
    elif fault == "missing_point": group["expect_points"] = ["满99元包邮", "无依据承诺"]
    elif fault == "empty_points": group["expect_points"] = []
    elif fault == "alternative_missing_point": group["allowed_section_paths"].append("returns.md / 演示 / 乙")
    elif fault == "duplicate_group": rows[-1]["expected_groups"][1]["group_id"] = "fee"
    elif fault == "overlapping_groups": rows[-1]["expected_groups"][1] = {**copy.deepcopy(group), "group_id": "duplicate_evidence"}
    elif fault == "d_ground_truth": rows[180]["expected_groups"] = [copy.deepcopy(group)]
    elif fault == "e_single_group": rows[-1]["expected_groups"].pop()
    elif fault == "empty_groups": rows[0]["expected_groups"] = []
    elif fault == "group_schema": group["source"] = "old schema"
    elif fault == "non_object": rows[0] = []
    elif fault == "bad_paths": group["allowed_section_paths"] = "returns.md"
    elif fault == "blank_point": group["expect_points"] = [" "]
    elif fault == "bad_groups": rows[0]["expected_groups"] = "invalid"
    elif fault == "duplicate_point": group["expect_points"] *= 2
    elif fault == "duplicate_path": group["allowed_section_paths"] *= 2
    assert validate_records(rows, root)


def test_cli_reports_invalid_json_and_fails(tmp_path, capsys):
    from scripts.validate_eval_ch04 import main
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id":}\n', encoding="utf-8")
    assert main(["--data", str(path)]) == 1
    assert "line 1" in capsys.readouterr().err


def test_repository_dataset_is_exactly_300_valid_cases():
    from scripts.validate_eval_ch04 import DEFAULT_DATA, DEFAULT_KB_ROOT, validate_records
    rows = [json.loads(line) for line in DEFAULT_DATA.read_text(encoding="utf-8").splitlines()]
    assert validate_records(rows, DEFAULT_KB_ROOT) == []
