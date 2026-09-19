# Ch03 Dense Knowledge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `query_faq` SQL LIKE with BGE-M3/Milvus dense retrieval, backed by a resumable MySQL knowledge pipeline and scheduled conversation QA mining.

**Architecture:** MySQL holds authoritative chunks and staging rows. Offline jobs parse Markdown or extract conversation QA, deduplicate, commit pending chunks, then upsert vectors into Milvus using the MySQL chunk ID. The unchanged `query_faq(keyword)` tool embeds the query, searches Milvus, and reads ranked answers from MySQL.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x, MySQL 8.4, Docker Compose, Milvus standalone, PyMilvus, LangChain, langchain-text-splitters, SiliconFlow `BAAI/bge-m3` embeddings, APScheduler, uv, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-ch03-dense-knowledge-design.md`

## Global Constraints

- Preserve `query_faq(keyword: str)` and JSON `{"keyword": string, "matches": [{"question": string, "answer": string, "category": string}]}`.
- Copy both user DDL tables exactly; add no MySQL knowledge columns or substitute databases. Keep Ch02 tables and chat flow.
- Embed only `category + "\n" + questions + "\n" + answer`; `section_path`, `content_type`, `is_key_clause`, `prev_chunk_id`, `next_chunk_id`, `vector_id`, and status remain metadata in MySQL.
- Use `BAAI/bge-m3` via SiliconFlow, 1024-dimensional dense vectors, and Milvus `knowledge` collection with COSINE. No keyword fallback, hybrid search, or online rerank.
- Stage conversation QA in batches, then globally merge only synonymous questions with factually consistent answers.
- Keep `dev-notes/ch03.md` current after brainstorm, plan approval, **each task**, review, and finish; each entry contains user words, output, corrections, and rework.
- Refresh Context7 for each version-sensitive library API before implementation. If the fixed stack fails on this machine, stop and ask rather than substitute a model or vector database.
- Do not commit `.env` or log API keys. Real cloud acceptance requires the user to set `EMBED_API_KEY` in `.env`.

---

## File map and interfaces

| File | Responsibility |
|---|---|
| `db/ch03.sql`, `compose.milvus.yaml` | Exact two-table DDL and official Milvus standalone services |
| `app/db/models.py` | `KnowledgeChunk`, `QAExtractionStaging` matching DDL |
| `app/knowledge/chunks.py` | `ChunkDraft` and structure-aware Markdown parser |
| `app/knowledge/importer.py` | Atomic, repeatable document/legacy FAQ import |
| `app/knowledge/embedding.py` | SiliconFlow BGE-M3 batch client and vector validation |
| `app/knowledge/vector_store.py` | Milvus collection, upsert and scored-ID search |
| `app/knowledge/indexer.py` | Pending-row two-write state machine |
| `app/knowledge/extraction.py` | Conversation batch selection, QA extraction and staging |
| `app/knowledge/dedup.py` | Global semantic QA merge and knowledge insertion |
| `app/knowledge/retrieval.py` | Query embed, Milvus IDs, ranked MySQL answers |
| `app/jobs/qa_scheduler.py` | Independent daily 02:00 scheduler and one-shot command |
| `knowledge/{policy,faq,manual}/*.md` | Clearly labeled demonstration corpus |
| `scripts/build_knowledge.py`, `scripts/eval_ch03.py` | One-shot build/recovery and labeled evaluation |
| `app/tools/business.py`, `app/core/prompts.py`, `app/config.py`, `.env.example` | Existing tool integration, language and configuration |

The interfaces introduced in order are:

```python
@dataclass(frozen=True)
class ChunkDraft:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool = False

def parse_markdown(path: Path, root: Path, max_chars: int = 800, overlap: int = 100) -> list[ChunkDraft]: ...
def import_source(session: Session, source_key: str, drafts: list[ChunkDraft]) -> list[KnowledgeChunk]: ...

class EmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class MilvusStore:
    def ensure_collection(self) -> None: ...
    def upsert(self, ids: list[int], vectors: list[list[float]]) -> None: ...
    def search(self, vector: list[float], limit: int) -> list[tuple[int, float]]: ...

def index_pending(session_factory, embedder: EmbeddingClient, vector_store: MilvusStore,
                  batch_size: int = 16, after_upsert: Callable | None = None) -> int: ...
def retrieve_faq(keyword: str, session_factory, embedder, vector_store,
                 top_k: int = 5, min_score: float = 0.55) -> dict: ...
```

Task 5 introduces `extract_conversation_batch(session_factory, model, conversation_ids: list[int], batch_no: str) -> int`. Task 6 introduces `deduplicate_staging(session_factory, embedder, judge_model) -> int`. Both are synchronous offline entry points. Their model-specific internals are isolated and tested with labeled examples.

## Task 1: Exact SQLAlchemy mapping and Milvus boot

**Files:** Create `db/ch03.sql`, `compose.milvus.yaml`, `tests/test_ch03_schema.py`; modify `app/db/models.py`, `pyproject.toml`, `uv.lock`.

**Interfaces:** Produces `KnowledgeChunk`, `QAExtractionStaging`, and the running MySQL/Milvus services for later tasks.

- [ ] **Step 1 — refresh docs and record versions.** Recheck Context7 for MySQL dialect `BIGINT(unsigned=True)`, self-FK `ondelete="SET NULL"`, and PyMilvus `MilvusClient.create_collection`. Pin compatible PyMilvus and official Milvus standalone image/config versions in the plan execution notes. Use the official Windows Docker Compose example; keep the existing MySQL service and data volume.
- [ ] **Step 2 — write a failing schema test.** `tests/test_ch03_schema.py` must assert the two new table names, every column's type/nullability/default, indexes, FK names and `ON DELETE SET NULL`, plus exact `status` ENUM members. Run `uv run pytest tests/test_ch03_schema.py -q`; expected red because the models/tables do not exist.
- [ ] **Step 3 — implement the schema.** Save the exact user SQL from the Appendix below as `db/ch03.sql`, run it against the existing `mewhelp` database with UTF-8 client input, and add SQLAlchemy models. The self references must use `ForeignKey("knowledge_chunks.id", name="fk_chunks_prev", ondelete="SET NULL")` and the corresponding next pointer. Use `mysql.BIGINT(unsigned=True)`, dialect ENUM, and `server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")` where applicable.
- [ ] **Step 4 — boot Milvus.** Add the official standalone Docker Compose services (etcd, MinIO, standalone) without changing the existing MySQL volume; expose `19530`. Run `docker compose -f compose.yaml -f compose.milvus.yaml up -d`; confirm health and connect using PyMilvus. If this fixed stack fails for resources, stop and ask the user before any substitute.
- [ ] **Step 5 — verify and commit.** Run the schema test green, `SHOW CREATE TABLE` for both tables, `docker compose ps`, and `git diff --check`. Append the Task 1 four-field entry to `dev-notes/ch03.md`, then commit this task.

## Task 2: Markdown chunker, demonstration corpus and repeatable import

**Files:** Create `app/knowledge/__init__.py`, `app/knowledge/chunks.py`, `app/knowledge/importer.py`, `knowledge/policy/shipping.md`, `knowledge/policy/returns.md`, `knowledge/faq/products.md`, `knowledge/manual/after_sales.md`, `tests/test_knowledge_chunks.py`, `tests/test_knowledge_importer.py`; modify no chat code.

**Interfaces:** Consumes `KnowledgeChunk`; produces `parse_markdown` and `import_source` exactly as in File map.

- [ ] **Step 1 — write failing chunk tests.** Use a small Markdown fixture with nested headings, a long paragraph, a `> [!IMPORTANT]` marker, and a table header plus enough rows to span two chunks. Assert `category` is parent path, `questions` is the current heading for policy/manual or natural question for FAQ, sentence-end overlap, whole rows, and repeated table header. Run `uv run pytest tests/test_knowledge_chunks.py -q`; expected red due missing parser.
- [ ] **Step 2 — implement structure-aware split.** Use `MarkdownHeaderTextSplitter` for heading metadata and `RecursiveCharacterTextSplitter` for oversized prose, then a sentence-boundary pass so each final chunk ends at `。！？.!?` or a complete table row. Oversized indivisible sentences/rows stay whole and log a warning. Exclude metadata from `ChunkDraft.answer` except duplicated table headers needed for comprehension.
- [ ] **Step 3 — write importer red test.** Given two drafts from the same source, first import must insert two `pending` rows with reciprocal prev/next links; identical second import must insert zero; changed ordered sequence must raise `SourceChangedError` without writing. Use a rollback test that raises before commit and confirms no partial source. Run `uv run pytest tests/test_knowledge_importer.py -q` red.
- [ ] **Step 4 — implement importer and legacy FAQ migration.** `section_path` starts with a stable source key such as `knowledge/policy/shipping.md > ...`; compare the full ordered row sequence for repeat imports under a single-writer lock. Import Ch02 `faq` rows with source keys `legacy-faq/{id}`. Commit one source per transaction and fill pointers after row IDs are flushed.
- [ ] **Step 5 — write and evaluate non-code data.** Add clearly labeled demo content with a shipping section explaining when postage is shown and how to confirm it; add returns, product and after-sales examples. Run a labeled data check confirming “邮费是多少” has no literal phrase match in the shipping heading but the shipping answer is appropriate; assert all document types and key-clause/table examples parse. This replaces TDD for corpus copy.
- [ ] **Step 6 — verify and commit.** Run focused tests, record exact chunk counts and stage notes, then commit.

## Task 3: SiliconFlow embeddings and Milvus ID-only adapter

**Files:** Create `app/knowledge/embedding.py`, `app/knowledge/vector_store.py`, `tests/test_embedding.py`, `tests/test_vector_store.py`; modify `app/config.py`, `.env.example`, `pyproject.toml`, `uv.lock`.

**Interfaces:** Produces `EmbeddingClient.embed`, `MilvusStore.ensure_collection/upsert/search` from File map. Input vector length is exactly 1024.

- [ ] **Step 1 — write embedding client red tests.** Use an HTTP fake transport to assert POST to `${EMBED_BASE_URL}/embeddings` with `model="BAAI/bge-m3"`, Bearer key, array input, and `encoding_format="float"`; verify returned `data[index].embedding` order, 1024 dimensions, 429/503 retry, and clear errors for missing key or wrong dimension. Run `uv run pytest tests/test_embedding.py -q` red.
- [ ] **Step 2 — implement embedding client.** Default `EMBED_BASE_URL=https://api.siliconflow.cn/v1`, `EMBED_MODEL=BAAI/bge-m3`; API key has no default. Batch inputs up to 16; bound HTTP timeout and retry only 429/503/504 with short backoff. Never log request auth headers or raw customer conversation text.
- [ ] **Step 3 — write Milvus adapter red tests.** Fake `MilvusClient` should see collection `knowledge`, INT64 explicit ID, FLOAT_VECTOR dimension 1024, COSINE, `auto_id=False`; upsert payload has only `id` and `vector`; search returns ordered `(id, distance)` pairs. Run `uv run pytest tests/test_vector_store.py -q` red.
- [ ] **Step 4 — implement adapter.** Use `MilvusClient.create_collection(..., dimension=1024, primary_field_name="id", id_type="int", vector_field_name="vector", metric_type="COSINE", auto_id=False)` when missing; verify existing schema before reuse. Use `upsert(collection_name="knowledge", data=[{"id": id, "vector": vector}, ...])` and `search(..., data=[vector], limit=limit)`.
- [ ] **Step 5 — verify and commit.** Run focused tests. If `EMBED_API_KEY` is present, make a one-text smoke request and verify 1024 numbers; otherwise record the missing credential as a later acceptance gate, not a reason to change provider. Record Task 3 in dev-notes and commit.

## Task 4: Pending-vector two-write state machine and interruption recovery

**Files:** Create `app/knowledge/indexer.py`, `scripts/build_knowledge.py`, `tests/test_knowledge_indexer.py`; modify `README.md`.

**Interfaces:** Consumes `EmbeddingClient`, `MilvusStore`, `KnowledgeChunk`; produces `index_pending` exactly as in File map.

- [ ] **Step 1 — write red recovery tests.** Insert two `pending` chunks. Fake embedder returns distinct 1024-vectors; fake store records upserts. Inject `after_upsert` failure after first batch: rows must still be pending. Rerun without failure: both become `done`, `vector_id == str(id)`, and store has exactly two IDs. Also assert embed/upsert failures leave rows pending. Run `uv run pytest tests/test_knowledge_indexer.py -q` red.
- [ ] **Step 2 — implement state transitions.** Read a bounded ordered batch of `pending` IDs, close the DB session, embed and upsert, then open a new transaction to mark those exact IDs `done`. An `after_upsert` hook is test-only fault injection at the required crash boundary. Do not hold a MySQL transaction while waiting on the embedding API or Milvus.
- [ ] **Step 3 — one-shot build entry point.** `uv run python -m scripts.build_knowledge` calls the importer for demo files and legacy FAQ, ensures the Milvus collection, and drains pending chunks. `--resume-only` skips import and processes existing pending rows. Report counts: imported, pending, indexed, failed. Guard against concurrent runs with a MySQL advisory lock and release it in `finally`.
- [ ] **Step 4 — verify real recovery.** With services running, trigger the fault hook after Milvus upsert, query MySQL pending and Milvus ID, then run `--resume-only` and verify MySQL `done`/`vector_id` with exactly one Milvus entity per ID. Record the exact commands and IDs without logging secrets.
- [ ] **Step 5 — focused tests, notes, commit.** Run focused tests and append Task 4 outcome to dev-notes.

## Task 5: Batch conversation extraction into staging

**Files:** Create `app/knowledge/extraction.py`, `app/knowledge/extraction_prompts.py`, `tests/test_qa_extraction.py`, `tests/data/ch03_extraction_samples.json`.

**Interfaces:** Produces `extract_conversation_batch(session_factory, model, conversation_ids: list[int], batch_no: str) -> int`; stores `QAExtractionStaging` rows with `status="extracted"`.

- [ ] **Step 1 — write red code tests.** A conversation with user, tool request, tool result and final assistant must send only reusable user/final-answer text to the extraction model; an order-specific/mock answer must not be staged. Re-running the same batch/source/question/answer must add zero rows. Batch size is at most 30 conversation IDs. Run `uv run pytest tests/test_qa_extraction.py -q` red.
- [ ] **Step 2 — implement selection and structured output.** Define Pydantic `ExtractedQA(question: str, answer: str)` and `QASet(items: list[ExtractedQA])`. Use existing `get_chat_model(disable_thinking=True).with_structured_output(QASet, method="function_calling")`; strip obvious order IDs, tool JSON and private facts before model input. Persist each candidate with `batch_no`, `source_ref="conversation/{id}"`, `status="extracted"`.
- [ ] **Step 3 — evaluate the extraction Prompt.** Use at least five labeled sample conversations: a general return rule, same rule in another phrasing, conflict, private order state, and no reusable QA. Run sample evaluation against the configured chat model; record accepted/rejected candidates and revise Prompt only if evidence warrants it. This is the user's evaluation-set replacement for pure Prompt TDD.
- [ ] **Step 4 — verify and commit.** Run focused code tests, record sample outcomes and Task 5 dev-notes, then commit.

## Task 6: Global semantic dedup and daily scheduler

**Files:** Create `app/knowledge/dedup.py`, `app/jobs/__init__.py`, `app/jobs/qa_scheduler.py`, `tests/test_qa_dedup.py`, `tests/test_qa_scheduler.py`; modify `pyproject.toml`, `uv.lock`, `.env.example`.

**Interfaces:** Produces `deduplicate_staging(session_factory, embedder, judge_model) -> int`; command `uv run python -m app.jobs.qa_scheduler --once` or `--schedule`.

- [ ] **Step 1 — write red dedup tests.** Across two different `batch_no` values, normalized duplicates and “邮费怎么算/运费是多少” with equivalent answers become one kept knowledge chunk whose `questions` has both lines; conflicting answers remain separate; failed LLM judgement leaves rows `extracted`. Re-running does not add chunks. Run `uv run pytest tests/test_qa_dedup.py -q` red.
- [ ] **Step 2 — implement global dedup.** Normalize whitespace/punctuation for exact grouping, compute BGE-M3 vectors of remaining questions, shortlist pairs by COSINE >= 0.85, then ask a structured-output LLM judge for `same_intent` and `same_facts`. Only two `True` values merge. Mark representative staging row `kept`, merged rows `discarded`; insert `content_type="extracted_qa"` chunk with newline-separated distinct true questions. Reconcile against existing chunks before insert.
- [ ] **Step 3 — write scheduler red tests.** Freeze time around 02:00 Asia/Shanghai; assert one run invokes extraction, then dedup, then `index_pending`, while an overlapping invocation cannot acquire the advisory lock. `--once` must use the same pipeline. Run `uv run pytest tests/test_qa_scheduler.py -q` red.
- [ ] **Step 4 — implement scheduler.** Before editing this module, query Context7 for the installed APScheduler release and pin that release in `pyproject.toml` and `uv.lock`; use its documented blocking scheduler and Cron trigger APIs. Register one daily trigger at `02:00` in `Asia/Shanghai`, with a single independent worker process. Use a MySQL advisory lock to avoid overlap between scheduled and manual jobs. Make `--once` terminate with nonzero status on errors; `--schedule` stays running. Record the chosen version, import path, and trigger constructor in the Task 6 dev-note.
- [ ] **Step 5 — evaluate non-code dedup Prompt.** Run labeled synonymous, contradictory and unrelated pairs through the configured chat model; do not promote conflicting rows. Log counts only, not raw private conversation text. Run focused tests, append Task 6 notes, commit.

## Task 7: Dense retriever behind the existing `query_faq` contract

**Files:** Create `app/knowledge/retrieval.py`, `tests/test_knowledge_retrieval.py`; modify `app/tools/business.py`, `app/core/prompts.py`, `tests/test_tools.py`, `tests/test_prompts.py`, `app/config.py`, `.env.example`.

**Interfaces:** Produces `retrieve_faq` from File map. `build_tools(session_factory, conversation_id, retriever=None)` adds only an optional injection point; model-visible `query_faq(keyword: str)` and JSON shape stay exactly as in Global Constraints.

- [ ] **Step 1 — write red retriever tests.** Fake embedder/query store returns scored IDs `(12, .83), (9, .72), (15, .40)` and MySQL rows 12/9 as `done`; assert only 12/9 returned in that order, fields map `questions→question`, and metadata/vector ID never appear. Also assert empty hits, missing MySQL row, missing key, and Milvus exception do not trigger SQL LIKE. Run `uv run pytest tests/test_knowledge_retrieval.py -q` red.
- [ ] **Step 2 — implement retriever.** Embed `keyword`, search `knowledge` with `RAG_TOP_K=5`, discard scores below `RAG_MIN_SCORE=0.55`, fetch IDs in one SQL query, filter `vectorize_status="done"`, restore hit order and produce exact JSON. Existing `faq` table is no longer an online data source.
- [ ] **Step 3 — switch tool internals.** Preserve `FaqArgs.keyword`, `query_faq` name, and output shape. Inject a retriever for tests; production constructs BGE/Milvus dependencies lazily. Update tool description and system Prompt to say “知识库语义检索” instead of “字面关键词”. Keep the Ch02 SSE, ticket, order and logistics paths unchanged.
- [ ] **Step 4 — run old/new regressions and commit.** Update the Ch02 test that intentionally expected “邮费” to miss so the new evaluation expects shipping recall. Run `uv run pytest tests/test_tools.py tests/test_chat_api.py tests/test_knowledge_retrieval.py -q`, append Task 7 notes, commit.

## Task 8: Real acceptance, documentation and final review

**Files:** Create `scripts/eval_ch03.py`, `tests/data/ch03_retrieval_samples.json`; modify `README.md`, `dev-notes/ch03.md`.

**Interfaces:** Runs the entire pipeline and prints pass/fail with source chunk IDs, without exposing keys.

- [ ] **Step 1 — prepare labeled retrieval set.** Include “邮费是多少”→shipping explanation, returns, product FAQ, unrelated query, and a paraphrase not appearing literally in headings. Validate policy facts against the demo Markdown source, not against invented model prose.
- [ ] **Step 2 — require the embedding credential.** Confirm `EMBED_API_KEY` exists without printing it. If absent, finish all independent code/tests and request the user to add it to `.env`; do not change provider. When present, verify BGE-M3 returns 1024 values.
- [ ] **Step 3 — run complete build and recovery.** Start MySQL + Milvus, apply exact DDL, import docs and legacy FAQ, run the three labeled extraction checks, inject an upsert/update crash, resume, and confirm Milvus IDs equal MySQL IDs. Run real `query_faq("邮费是多少")` and a browser chat request; assert shipping answer, same tool badge and streamed final answer.
- [ ] **Step 4 — full verification and review.** Run `uv run pytest -q`, the labeled extraction evaluation, `uv run python -m scripts.eval_ch03`, `docker compose ps`, and `git diff --check`. Request backend code review; fix Important issues with red-green tests, repeat checks, and record review conclusion in dev-notes.
- [ ] **Step 5 — finish.** Update README with exact PowerShell/CMD demo commands and explain demo data, API key and scheduler. Append finish entry to `dev-notes/ch03.md` immediately after verification, commit, and report commands, test counts, known resource limits and note path.

## Appendix: exact user-provided Ch03 DDL

The implementer copies this SQL to `db/ch03.sql` without changing column types, keys, defaults or comments:

```sql
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';
```
