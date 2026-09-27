"""Deterministic Ch04 metrics. Each evidence group has equal weight per case.

Ranks belong to retrieval, not the lost-in-the-middle context order. Coverage is
literal after whitespace normalization; it is not a semantic-quality estimate.
"""

from statistics import mean


def normalize(text: str) -> str:
    return "".join(text.split())


def _point_coverage(points: list[str], text: str) -> float:
    normalized = normalize(text)
    return mean(float(normalize(point) in normalized) for point in points)


def retrieval_metrics(case: dict, hits: list[dict]) -> dict | None:
    if case["bucket"] == "D_out_of_scope":
        return None
    ranks = [hit.get("rank") for hit in hits]
    if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(range(1, len(hits) + 1)):
        raise ValueError("hits must retain unique original rank values 1..K")
    recall, reciprocal, coverage = [], [], []
    for group in case["expected_groups"]:
        allowed = set(group["allowed_section_paths"])
        matched = [hit for hit in hits if hit.get("section_path") in allowed]
        first = min((hit["rank"] for hit in matched), default=None)
        recall.append(float(first is not None and first <= 5))
        reciprocal.append(1 / first if first is not None else 0.0)
        coverage.append(_point_coverage(group["expect_points"], "\n".join(hit["answer"] for hit in matched)))
    return {"recall_at_5": mean(recall), "mrr": mean(reciprocal), "evidence_coverage": mean(coverage)}


def answer_point_coverage(case: dict, answer: str) -> float | None:
    if case["bucket"] == "D_out_of_scope":
        return None
    return mean(_point_coverage(group["expect_points"], answer) for group in case["expected_groups"])


def refusal_rate(generations: list[dict]) -> float | None:
    completed = [row for row in generations if row["status"] == "ok"]
    return mean(float(row["refused"]) for row in completed) if completed else None
