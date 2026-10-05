import importlib.util


def test_ch09_migration_module_exists():
    assert importlib.util.find_spec("scripts.migrate_ch09") is not None


def test_partial_migration_resumes_in_dependency_order():
    from scripts.migrate_ch09 import planned_operations

    state = {
        "tables": {"review_queue"},
        "low_confidence_columns": {"retrieved_chunks"},
        "low_confidence_indexes": set(),
        "low_confidence_foreign_keys": set(),
    }
    assert planned_operations(state) == [
        "CREATE TABLE eval_runs",
        "ADD COLUMN matched_review_id",
        "ADD INDEX idx_matched_review_id",
        "ADD CONSTRAINT fk_lcq_review",
    ]


def test_matching_migration_has_no_operations():
    from scripts.migrate_ch09 import planned_operations

    state = {
        "tables": {"review_queue", "eval_runs"},
        "low_confidence_columns": {"retrieved_chunks", "matched_review_id"},
        "low_confidence_indexes": {"idx_matched_review_id"},
        "low_confidence_foreign_keys": {"fk_lcq_review"},
    }
    assert planned_operations(state) == []


def test_migration_never_plans_drop_or_recreate():
    from scripts.migrate_ch09 import planned_operations

    operations = planned_operations({
        "tables": set(), "low_confidence_columns": set(),
        "low_confidence_indexes": set(), "low_confidence_foreign_keys": set(),
    })
    assert operations[:2] == ["CREATE TABLE review_queue", "CREATE TABLE eval_runs"]
    assert all("DROP" not in operation and "RECREATE" not in operation for operation in operations)
