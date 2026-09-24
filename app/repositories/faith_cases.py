"""Faithfulness case ledger. Write operations use the caller's transaction."""

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.db.models import FaithCase


STATUSES = frozenset({"未解决", "已解决", "无需解决"})


def upsert_faith_case(
    session: Session, *, eval_id: str, bucket: str, query: str, strategy: str,
    answer: str, reason: str, citations: list[dict] | None,
    judge_model: str | None,
) -> FaithCase:
    """Record another failed evaluation atomically by its unique eval_id."""
    values = dict(eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                  answer=answer, reason=reason, citations=citations, judge_model=judge_model)
    dialect = session.get_bind().dialect.name
    if dialect == "mysql":
        insert = mysql_insert(FaithCase).values(**values)
    elif dialect == "sqlite":
        insert = sqlite_insert(FaithCase).values(**values)
    else:
        raise NotImplementedError(f"faith-case upsert is unsupported on {dialect}")

    replacements = {name: getattr(insert.inserted if dialect == "mysql" else insert.excluded, name)
                    for name in values if name != "eval_id"}
    replacements.update(
        seen_count=FaithCase.seen_count + 1,
        status="未解决",
        resolution=None,
        last_seen_at=func.current_timestamp(),
    )
    if dialect == "mysql":
        statement = insert.on_duplicate_key_update(**replacements)
    else:
        statement = insert.on_conflict_do_update(
            index_elements=[FaithCase.eval_id], set_=replacements,
        )
    session.execute(statement)
    return session.scalar(
        select(FaithCase).where(FaithCase.eval_id == eval_id)
        .execution_options(populate_existing=True)
    )


def list_faith_cases(
    status: str | None, page: int, size: int, *, session: Session,
) -> tuple[list[FaithCase], int]:
    if status is not None and status not in STATUSES:
        raise ValueError("invalid faith-case status")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("page must be a positive integer")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    predicate = FaithCase.status == status if status is not None else None
    count = select(func.count()).select_from(FaithCase)
    rows = select(FaithCase)
    if predicate is not None:
        count = count.where(predicate)
        rows = rows.where(predicate)
    total = session.scalar(count) or 0
    rows = rows.order_by(FaithCase.last_seen_at.desc(), FaithCase.id.desc())
    rows = rows.offset((page - 1) * size).limit(size)
    return list(session.scalars(rows)), total


def set_faith_case_status(
    id: int, status: str, resolution: str | None, *, session: Session,
) -> FaithCase:
    if status not in STATUSES:
        raise ValueError("invalid faith-case status")
    if status != "未解决" and (not isinstance(resolution, str) or not resolution.strip()):
        raise ValueError("resolution is required for a resolved faith case")
    row = session.get(FaithCase, id)
    if row is None:
        raise LookupError("faith case not found")
    row.status = status
    row.resolution = resolution.strip() if status != "未解决" else None
    if status != "未解决":
        row.resolved_at = func.current_timestamp()
    session.flush()
    return row


def faith_case_status_map(eval_ids: list[str], *, session: Session) -> dict[str, str]:
    if not eval_ids:
        return {}
    rows = session.execute(
        select(FaithCase.eval_id, FaithCase.status).where(FaithCase.eval_id.in_(eval_ids))
    )
    return {eval_id: status for eval_id, status in rows}
