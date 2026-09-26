"""Validate strategy-independent Ch04 labels against the ingestible corpus.

Section paths inside a group are alternatives. Every group is required, and
every point within a group is required. Each alternative must support all its
group's points. Difficulty is balanced 20/20/20 inside each 60-case bucket.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from app.knowledge.chunks import parse_markdown
from app.knowledge.sources import SOURCE_TYPES

BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_out_of_scope", "E_multi")
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "tests/data/eval_ch04.jsonl"
DEFAULT_KB_ROOT = Path(__file__).resolve().parents[1] / "data/kb"


DIFFICULTIES = ("easy", "medium", "hard")
RECORD_FIELDS = {"id", "bucket", "difficulty", "query", "expected_groups"}
GROUP_FIELDS = {"group_id", "allowed_section_paths", "expect_points"}


def normalize(value):
    return "".join(value.split())


def source_sections(kb_root):
    sections = defaultdict(list)
    for name, content_type in SOURCE_TYPES.items():
        path = Path(kb_root) / name
        if path.exists():
            for chunk in parse_markdown(path.read_text(encoding="utf-8"), name, content_type):
                sections[chunk.section_path].append(chunk.answer)
    return {path: normalize("\n".join(parts)) for path, parts in sections.items()}


def _strings(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and normalize(item) for item in value
    )


def validate_records(records, kb_root=DEFAULT_KB_ROOT):
    errors = []
    if not isinstance(records, list):
        return ["records must be a list"]
    if len(records) != 300:
        errors.append(f"expected 300 cases, got {len(records)}")
    sections = source_sections(kb_root)
    seen = {"id": set(), "query": set()}
    buckets, difficulty = Counter(), Counter()
    for index, record in enumerate(records, 1):
        prefix = f"record {index}"
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            errors.append(f"{prefix}: expected fields {sorted(RECORD_FIELDS)}")
            continue
        for field in ("id", "query"):
            value = record[field]
            if not isinstance(value, str) or not normalize(value):
                errors.append(f"{prefix}: {field} must be nonempty text")
            elif normalize(value) in seen[field]:
                errors.append(f"{prefix}: duplicate {field}")
            else:
                seen[field].add(normalize(value))
        bucket, level = record["bucket"], record["difficulty"]
        if bucket not in BUCKETS:
            errors.append(f"{prefix}: invalid bucket")
            continue
        buckets[bucket] += 1
        if level not in DIFFICULTIES:
            errors.append(f"{prefix}: invalid difficulty")
        else:
            difficulty[bucket, level] += 1
        groups = record["expected_groups"]
        if not isinstance(groups, list):
            errors.append(f"{prefix}: expected_groups must be a list")
            continue
        if bucket == "D_out_of_scope":
            if groups:
                errors.append(f"{prefix}: D must have no ground truth")
            continue
        if (bucket == "E_multi" and len(groups) < 2) or (bucket != "E_multi" and len(groups) != 1):
            errors.append(f"{prefix}: invalid number of evidence groups")
        group_ids, used_paths = set(), set()
        for group in groups:
            if not isinstance(group, dict) or set(group) != GROUP_FIELDS:
                errors.append(f"{prefix}: invalid group schema")
                continue
            group_id = group["group_id"]
            if not isinstance(group_id, str) or not normalize(group_id):
                errors.append(f"{prefix}: invalid group_id")
            elif normalize(group_id) in group_ids:
                errors.append(f"{prefix}: duplicate group_id")
            else:
                group_ids.add(normalize(group_id))
            paths, points = group["allowed_section_paths"], group["expect_points"]
            if not _strings(paths) or not _strings(points):
                errors.append(f"{prefix}: paths and points must be nonempty text lists")
                continue
            if len(set(paths)) != len(paths) or len(set(map(normalize, points))) != len(points):
                errors.append(f"{prefix}: duplicate path or point")
            if used_paths.intersection(paths):
                errors.append(f"{prefix}: independently required groups cannot share section paths")
            used_paths.update(paths)
            for path in paths:
                if path not in sections:
                    errors.append(f"{prefix}: source section does not exist: {path}")
                else:
                    for point in points:
                        if normalize(point) not in sections[path]:
                            errors.append(f"{prefix}: point absent from {path}: {point}")
    for bucket in BUCKETS:
        if buckets[bucket] != 60:
            errors.append(f"{bucket}: expected 60 cases, got {buckets[bucket]}")
        for level in DIFFICULTIES:
            if difficulty[bucket, level] != 20:
                errors.append(f"{bucket}/{level}: expected 20 cases, got {difficulty[bucket, level]}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--kb-root", type=Path, default=DEFAULT_KB_ROOT)
    args = parser.parse_args(argv)
    try:
        records = []
        for number, line in enumerate(args.data.read_text(encoding="utf-8").splitlines(), 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"line {number}: invalid JSON ({error.msg})") from error
        errors = validate_records(records, args.kb_root)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(json.dumps({"valid": True, "cases": len(records),
                      "buckets": dict(Counter(row["bucket"] for row in records)),
                      "difficulty_per_bucket": dict.fromkeys(DIFFICULTIES, 20)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
