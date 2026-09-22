# Task 1 report — Ch03 schema and standalone foundation

## Static result

- Preserved the user-provided DDL verbatim in `db/ch03.sql`.
- Added exact MySQL SQLAlchemy mappings for `KnowledgeChunk` and
  `QAExtractionStaging`, including columns, comments, defaults, indexes and
  self-referential `ON DELETE SET NULL` foreign keys.
- Added non-secret embed/Milvus/RAG settings. `embed_api_key` defaults to an
  empty string so existing chat startup remains available.
- Added official Milvus v3.0.1 standalone Compose topology and new named
  volumes only. It does not alter the existing MySQL volume.
- Locked `pymilvus==3.0.1`. Milvus documentation recommends the matching
  PyMilvus v3.0.1 for Milvus v3.0.x.

## TDD evidence

| Phase | Command | Result |
|---|---|---|
| Red | `.venv\\Scripts\\python.exe -m pytest tests/test_ch03_schema.py -q` | Collection failed with `ImportError: cannot import name 'KnowledgeChunk'`; verified before ORM implementation. |
| Green | `.venv\\Scripts\\python.exe -m pytest tests/test_ch03_schema.py -q` | `3 passed in 0.81s` |
| Settings | `.venv\\Scripts\\python.exe -c ...` | New non-secret defaults verified. |
| Diff | `git diff --check` | Passed. |

## Context and live-service gates

- Context7 references checked: SQLAlchemy 2.0 mapping style; Milvus official
  v3.0.1 Compose release and PyMilvus compatibility documentation.
- Docker is unavailable while its Windows runtime is repaired. No Compose
  services were started.
- MySQL at `127.0.0.1:3307` is unavailable. No SQL was applied and no
  `SHOW CREATE TABLE` comparison was possible.
- An embedding API key is not configured. Its value was never read or logged.

When Docker and MySQL are restored, the pending live gate is: inspect both
table names first; create only missing tables from `db/ch03.sql`; otherwise
compare `SHOW CREATE TABLE`; then start the Compose overlay and record health.
