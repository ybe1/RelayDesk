# Ch04 Hybrid Retrieval, Reranking, and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `query_faq` from dense-only retrieval to Milvus native BM25 + dense RRF + BGE reranking, add cited refusal-safe generation, and deliver a five-bucket four-strategy evaluation and management workflow.

**Architecture:** Build and validate a parallel `knowledge_ch04` collection, then select it through configuration while retaining `knowledge` for rollback. The existing one-tool chat loop remains; an async FAQ tool runs query understanding, filtered hybrid retrieval, reranking and two evidence gates, then sends citations through the existing SSE stream. Evaluation calls the same retrieval interface and persists reports and faithfulness cases.

**Tech Stack:** FastAPI, SQLAlchemy 2.x, MySQL 8.4, PyMilvus/Milvus 3.0.2 native BM25, SiliconFlow BGE-M3 and `BAAI/bge-reranker-v2-m3`, LangChain tools/structured output, vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-09-23-ch04-hybrid-rag-design.md`

## Global Constraints

- Fixed choices: Milvus native BM25 + `hybrid_search` RRF, BGE-M3 dense, SiliconFlow `BAAI/bge-reranker-v2-m3`; stop and ask if either fail-fast smoke cannot work.
- Online default collection becomes `knowledge_ch04` only after parity gates; retain `knowledge` and never delete it during this plan.
- Online default strategy is `hybrid_rerank`; evaluation uses the same `search_knowledge` function for all four strategies.
- Preserve single model tool-selection call, at most one executed tool, and streamed final answer.
- Do not add keyword fallback, LangGraph, Langfuse, multi-turn rewriting, or coreference resolution.
- Deterministic production code uses TDD; prompt/data tasks use labeled evaluation; Task 13 UI uses the user's Vibe Coding exception.
- Before using a concrete library API, query current official docs through Context7 and record the result in the task report.
- Append `dev-notes/ch04.md` immediately after every task and review, with the user's key words, output, corrections, and rework.
- Run in the existing main checkout because the user's standing preference is direct main development; make one scoped commit per task.
- Preserve the Ch03 synchronous `search_knowledge(query, embedder, store, top_k=5, min_score=0.55)` facade for `/api/kb/search`, `scripts/eval_ch03.py`, and existing tests. Ch04 uses a separately named async orchestrator `search_knowledge_ch04(...)`.
- Keep `MILVUS_COLLECTION=knowledge` through Tasks 1–13. Cut over only in Task 14 after count/ID/schema/live-query parity; rollback is the single config change back to `knowledge` plus application restart.

---

### Task 1: Fail-fast BM25 and rerank probes

**Files:**
- Modify: `app/config.py`, `.env.example`, `pyproject.toml`, `uv.lock`
- Create: `scripts/smoke_ch04_bm25.py`, `scripts/smoke_ch04_rerank.py`
- Test: `tests/test_ch04_smoke.py`

**Interfaces:**
- Produces: `settings.rerank_base_url`, `rerank_api_key`, `rerank_model`, `milvus_collection`, `recall_top_k=50`, `rerank_top_k=10`, `rerank_min_score`; executable smoke functions returning structured summaries.

- [ ] Query Context7 for current PyMilvus BM25 Function/analyzer/hybrid signatures and official SiliconFlow rerank docs; record exact payload/response fields.
- [ ] Add tests that validate config defaults and mock rerank response shape before any network call.
- [ ] Run `uv run pytest tests/test_ch04_smoke.py -q` and confirm red for missing settings/scripts.
- [ ] Implement `smoke_ch04_bm25.main()` using a disposable `smoke_ch04_bm25` collection: chinese analyzer text, BM25 Function, dense/sparse indexes, two documents containing exact Chinese model names, dense + BM25 `AnnSearchRequest`, `RRFRanker`, and category filter. Always drop only the disposable collection in `finally`.
- [ ] Implement `smoke_ch04_rerank.main()` calling `settings.rerank_base_url.rstrip('/') + '/rerank'` with model/query/documents/top_n, validating unique in-range indexes and finite relevance scores. Never print keys or full response bodies.
- [ ] Run unit tests green, then run both real smoke scripts. Expected: Chinese model BM25 hit, hybrid results, filter exclusion, and rerank result with the return-policy document first.
- [ ] If either real smoke fails, add a `⚠️ 红线卡点` dev-note entry and stop. If green, append the four-item Task 1 note and commit.

### Task 2: Ch04 DDL and ORM models

**Files:**
- Create: `db/ch04.sql`
- Modify: `app/db/models.py`, `tests/test_db_models.py`
- Test: `tests/test_ch04_schema.py`

**Interfaces:**
- Produces: `LowConfidenceQuestion`, `FaithCase` ORM models exactly matching the supplied DDL.

**Authoritative DDL source:** the exact SQL is embedded in Appendix A of this plan so implementation does not depend on conversation history or the external course file.

- [ ] Copy the user's two-table DDL verbatim to `db/ch04.sql`, preserving `SET NAMES utf8mb4`, enums, indexes, unique key, defaults, comments and FK.
- [ ] Write schema tests for unsigned IDs, nullable conversation FK, source/status enums, JSON citations, unsigned seen_count, unique eval_id and indexes.
- [ ] Run the schema tests red because models are absent.
- [ ] Add both ORM models using existing MySQL-specific patterns. Update the Ch02 metadata test to require its tables as a subset.
- [ ] Run schema/model tests green. Against live MySQL, check table absence or compare `SHOW CREATE TABLE`; create only missing tables, never drop or replace existing tables.
- [ ] Re-run `SHOW CREATE TABLE` assertions, append task note, commit.

### Task 3: Parallel Ch04 Milvus collection and resumable indexing

**Files:**
- Create: `app/knowledge/hybrid_store.py`, `app/knowledge/ch04_indexer.py`
- Modify: `scripts/kb.py`, `Makefile`, `README.md`
- Test: `tests/test_hybrid_store.py`, `tests/test_ch04_indexer.py`, `tests/test_kb_cli.py`

**Interfaces:**
- Produces: `HybridMilvusStore.ensure_collection()`, `dense_search(vector, limit, filter_expr)`, `bm25_search(text, limit, filter_expr)`, `hybrid_search(vector, text, recall_limit, output_limit, filter_expr)`, `count()`, `ids()`; `rebuild_ch04(session_factory, embedder, store)`.

- [ ] Write fake-client red tests for exact fields, chinese analyzer, BM25 function, dense/sparse indexes, per-leg Top-50, RRFRanker, output metadata, filter propagation and incompatible-schema rejection.
- [ ] Add recovery tests: remote upsert succeeds then hook raises; rerun uses identical SQL IDs and count/IDs stay unique.
- [ ] Implement explicit `knowledge_ch04` schema and byte validation. Input rows include dense/text/question/answer/category/section_path/content_type/is_key_clause; sparse is server-generated.
- [ ] Use these Milvus `VARCHAR max_length` and pre-upsert UTF-8 byte caps: `text=65535`, `question=8192`, `answer=49152`, `category=1024`, `section_path=2048`, `content_type=128`. Build `text` as `分类：...\n问题：...\n答案：...`; reject a source row if any field or the composed text exceeds its cap, reporting the SQL chunk ID.
- [ ] Implement search methods with Strong consistency where current-entity correctness is required. Escape filter values and accept only typed category/content_type/is_key_clause parameters from a builder.
- [ ] Implement `kb ch04-rebuild` under `writer_lock`: read **all** authoritative `knowledge_chunks` rows regardless of `vectorize_status`, embed, idempotent upsert, compare SQL/Milvus counts and IDs. It must not modify Ch03 vectorize_status or delete `knowledge`.
- [ ] Run focused tests; run live rebuild into `knowledge_ch04`; prove count and ID parity plus repeated rebuild idempotency.
- [ ] Append task note, commit.

### Task 4: Rerank client

**Files:**
- Create: `app/knowledge/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Produces: `RerankClient.rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]` async.

- [ ] Write red tests for empty input, payload/model, response reordering, Top-10, duplicate/out-of-range indexes, NaN, missing key, and retries limited to HTTP 429/503/504.
- [ ] Implement with `httpx.AsyncClient`; cap document count and UTF-8 bytes, use configured timeout, and never log documents or credentials on failure.
- [ ] Run focused tests green and the real smoke through this client.
- [ ] Append note and commit.

### Task 5: Query understanding and evidence self-check prompts

**Files:**
- Create: `app/knowledge/query_understanding.py`, `app/knowledge/evidence_check.py`, `app/core/rag_prompts.py`
- Create: `tests/data/ch04_query_samples.json`, `tests/data/ch04_evidence_samples.json`
- Test: `tests/test_query_understanding.py`, `tests/test_evidence_check.py`

**Interfaces:**
- Produces: `QueryUnderstanding(normalized_query, search_query, category)` and `EvidenceDecision(useful, reason)`; async `understand_query(model, raw_query)` and `check_evidence(model, query, hits)`.

- [ ] Define flat Pydantic schemas; category is optional and must be from a configured allowlist. The prompt forbids historical-message input and expands synonyms only into search_query.
- [ ] Add at least 20 labeled query samples spanning policies, model names, colloquial wording, no-filter and invalid-category attempts; add at least 12 sufficient/insufficient evidence samples.
- [ ] Add deterministic transport/schema tests, then run them red.
- [ ] Implement `with_structured_output(..., method='function_calling')` using current repository model conventions and disabled thinking.
- [ ] Run fake transport tests green, then run the labeled samples against the configured model. Require all safety/category cases and at least 90% overall; record mismatches, revise prompt, rerun.
- [ ] Append note and commit.

### Task 6: Four-strategy retrieval orchestration

**Files:**
- Modify: `app/knowledge/retrieval.py`
- Test: `tests/test_knowledge_retrieval.py`, `tests/test_hybrid_retrieval.py`

**Interfaces:**
- Preserves the existing synchronous `search_knowledge(query, embedder, store, top_k=5, min_score=0.55)` unchanged for every Ch03 caller.
- Produces async `search_knowledge_ch04(query, *, strategy='hybrid_rerank', top_k=10, category=None, dependencies=None) -> RetrievalResult`; `RetrievalResult` contains understood query, ordered hits, sufficient flag and reason.

- [ ] Write red tests for every strategy branch, Top-50 legs, category filter, RRF output, rerank Top-10, mechanical threshold, dependency failure and ordering.
- [ ] Implement dense, BM25, hybrid and hybrid_rerank branches using Task 3/4/5 interfaces. For evaluation, strategy selection must not silently add rerank to other branches.
- [ ] Deduplicate candidates by chunk ID after RRF. Rerank text is `问题：{question}\n答案：{answer}`, UTF-8 bounded by Task 4. Apply `rerank_min_score` only to successful rerank output; service/schema failure is an operational error, never a low-confidence refusal.
- [ ] Implement deterministic lost-in-the-middle order: ranks `[0, 2, 4, ..., 5, 3, 1]` so rank 1 is first and rank 2 is last while retaining all items once.
- [ ] Run focused tests and a live model-name/category search against `knowledge_ch04`.
- [ ] Append note and commit.

### Task 7: Low-confidence repository and faith-case ledger

**Files:**
- Create: `app/repositories/low_confidence.py`, `app/repositories/faith_cases.py`
- Test: `tests/test_low_confidence.py`, `tests/test_faith_cases_repository.py`

**Interfaces:**
- Produces: `insert_low_confidence(session, conversation_id, raw_question, source, reason)`; `upsert_faith_case(session, *, eval_id, bucket, query, strategy, answer, reason, citations, judge_model)`; `list_faith_cases(status, page, size)`; `set_faith_case_status(id, status, resolution)`; `faith_case_status_map(eval_ids)`.
- `upsert_faith_case` conflicts on `eval_id`: replace bucket/query/strategy/answer/reason/citations/judge_model, increment `seen_count`, set `status='未解决'`, clear `resolution`, update `last_seen_at`, and retain `first_seen_at` and `resolved_at` so recurrence remains visible.

- [ ] Write transaction tests for nullable conversation, allowed source, exact raw question, rollback, one-row-per-eval, seen_count increments, citation snapshot replacement, resolved recurrence, sorting/pagination, resolution validation and clearing.
- [ ] Run red, implement repositories without internal commits, then run green on SQLite-compatible fixtures and live MySQL integration.
- [ ] Repository callers own transactions. Low-confidence insertion opens a short session only after both network/model gates finish, inserts exactly one row and commits; persistence failure rolls back and surfaces an operational tool error rather than returning a successful refusal.
- [ ] Append note and commit.

### Task 8: Async FAQ tool, citations SSE, and refusal-safe generation

**Files:**
- Modify: `app/tools/business.py`, `app/services/chat.py`, `app/core/prompts.py`, `app/services/tool_executor.py`
- Test: `tests/test_tools.py`, `tests/test_chat_service.py`, `tests/test_chat_api.py`, `tests/test_tool_executor.py`, `tests/test_prompts.py`

**Interfaces:**
- `build_tools(session_factory, conversation_id, retriever=None, evidence_checker=None)` preserves its first two parameters. `query_faq(keyword)` returns keyword/sufficient/reason/evidence/citations.
- ChatService emits `citations` before answer deltas when the executed FAQ result contains citations.
- `retriever` injection accepts sync or async callables. Legacy `list[dict]` results are normalized into a sufficient `RetrievalResult`; native Ch04 results pass through. This preserves current tool/chat tests and external test injectors.
- Persisted citations contain only `n`, `chunk_id`, `section_path`, `question`, and a UTF-8 bounded `answer_excerpt`; the complete serialized ToolMessage must be at most 60,000 bytes before insert. Exceeding data is deterministically truncated by code point and revalidated, never left for MySQL `TEXT` to reject.

- [ ] Write red tests for cited sufficient result, mechanical and self-check refusal inserts, exact citation mapping, no fabricated citation, citations SSE ordering, tool-message persistence and unchanged other tools.
- [ ] Convert only `query_faq` to `async def @tool`; ToolExecutor already uses `ainvoke`. Wire query understanding/retrieval/self-check and repository writes.
- [ ] Extend system prompt with citation rules and forbidden promises. A sufficient tool result tells the final model to answer only from evidence; insufficient tells it to refuse with reason.
- [ ] Parse the nested ToolMessage payload safely and emit citations only for a successful FAQ result. Keep single-tool execution and current retries.
- [ ] Run focused and all Ch01–03 chat/tool regressions. Run live SSE for cited answer and out-of-scope refusal; query MySQL for the new low-confidence row.
- [ ] Append note and commit.

### Task 9: Knowledge corpus and 300-case labeled evaluation set

**Files:**
- Modify/Create: `data/kb/*.md`
- Create: `tests/data/eval_ch04.jsonl`, `scripts/validate_eval_ch04.py`
- Test: `tests/test_eval_ch04_data.py`

**Interfaces:**
- Produces exactly 300 unique cases: 60 each in A_policy/B_model/C_colloquial/D_out_of_scope/E_multi; `make eval-check`.

- [ ] Add demo spec documents with multiple easy-to-confuse real demo model families. Mark them explicitly as demonstration data.
- [ ] Generate 300 labeled records with id, bucket, difficulty, query and strategy-independent ground truth. A/B/C use `expected_groups=[{group_id, allowed_section_paths, expect_points}]`; D has an empty group list; E has two or more independently required groups. Within a group, section paths are alternatives and any one satisfies retrieval; all listed points in that group are jointly required for generation coverage.
- [ ] Implement validator for uniqueness, balanced buckets, valid schema, source section existence, whitespace-normalized point presence and D absence of ground truth.
- [ ] Run validator; fix every data error. Rebuild `knowledge_ch04`, rerun BM25 model-name probes.
- [ ] Append note and commit.

### Task 10: Four-strategy evaluator and report artifacts

**Files:**
- Create: `scripts/eval_ch04.py`, `app/evaluation/ch04_metrics.py`, `app/evaluation/faithfulness.py`
- Modify: `Makefile`, `app/core/llm.py`
- Test: `tests/test_ch04_metrics.py`, `tests/test_eval_ch04_runner.py`, `tests/test_faithfulness.py`, `tests/test_llm.py`

**Interfaces:**
- Produces `data/ch04/reports/rag_eval.json` and `.txt`; `make eval-rag`; `get_chat_model(..., temperature=0.3)` with explicit judge override 0.

- [ ] Write red metric tests including Recall@5, average reciprocal rank across E evidence groups, evidence coverage, answer point coverage, refusal rate and partial generation failure. Per group: recall is 1 if any allowed section occurs in top 5 else 0; RR is reciprocal of the first allowed section rank or 0. Per case averages its groups; overall averages answerable cases (A/B/C/E), while D contributes only refusal metrics.
- [ ] Implement deterministic metrics and report schema first. Retrieval always completes all four strategies and all cases before optional generation.
- [ ] Implement low-concurrency answer generation for four strategies and temperature-zero faithfulness judge for online strategy. Timeouts/errors mark cases skipped; they never erase retrieval results.
- [ ] Upsert each judged hallucination with the complete ordered Top-K citation snapshot; ledger failure only adds a report warning.
- [ ] Run fake tests green, `make eval-check`, then the live 300-case evaluation. Verify four strategy/bucket numbers and B-model BM25 evidence.
- [ ] Append note and commit.

### Task 11: Faith-case APIs and judge regression

**Files:**
- Create: `app/api/rageval.py`, `scripts/judge_check.py`
- Modify: `app/main.py`, `Makefile`
- Test: `tests/test_rageval_api.py`, `tests/test_faith_cases_api.py`, `tests/test_judge_check.py`

**Interfaces:**
- Produces `GET /api/rag-eval/overview`, `GET /api/rag-eval/faith-cases`, `POST /api/rag-eval/faith-cases/{id}/status`, `make judge-check`.
- Produces `GET /api/kb/chunks/{chunk_id}` returning the authoritative MySQL `section_path`, `questions`, and full `answer`. The citation SSE carries only IDs and bounded labels; the browser fetches this endpoint when a citation is opened.

- [ ] Write red API tests for missing/corrupt/partial report, verbatim metric forwarding, best-MRR derivation, status filters/pages, invalid status, required resolution, recurrence and snapshot output.
- [ ] Implement report reader; web layer must never recompute metrics. Implement ledger endpoints with fixed limits and repository transactions.
- [ ] Add the read-only chunk-detail endpoint with 404 for unknown IDs and no Milvus dependency; test full-answer retrieval independently of the bounded ToolMessage citation snapshot.
- [ ] Implement judge replay using only stored answer/citations; map statuses to expected judgment; retry one empty completion, exclude persistent call failures, exit 1 on mismatch.
- [ ] Add six labeled boundary examples to the judge prompt and temperature=0. Run deterministic tests and live `judge-check` after seeding reviewed cases.
- [ ] Append note and commit.

### Task 12: Evaluation job and admin aggregation

**Files:**
- Modify: `app/core/jobs.py`, `app/api/jobs.py`, `app/api/admin.py`, `app/main.py`, `tests/test_jobs_api.py`, `tests/test_admin_api.py`
- Test: `tests/test_rageval_api.py`

**Interfaces:**
- Adds fixed JobSpec `eval-rag` using the current interpreter argv `('-m','scripts.eval_ch04')`, marked heavy for UI confirmation; `/rag-eval` page route.
- `_available()` uses an explicit job-name-to-module map, including `eval-rag -> scripts/eval_ch04.py`. Starting `eval-rag` requires server-side body token `confirmation='RUN_EVAL_RAG'`; omission or mismatch is HTTP 422.

- [ ] Write red tests that arbitrary argv cannot enter, only one eval runs, log tail is redacted, overview is read-only and admin card uses the same report.
- [ ] Register fixed Python argv rather than `make` so Windows works. Extend `StartRequest`/the existing start endpoint so `eval-rag` accepts only `confirmation='RUN_EVAL_RAG'` (parallel to the existing `kb-reset` validation) and returns HTTP 422 otherwise. Expose status/stop/tail through existing APIs.
- [ ] Add route/static serving and admin summary card data.
- [ ] Run focused management regressions, append note, commit.

### Task 13: Chat citations, feedback, and RAG evaluation UI (Vibe Coding)

**Files:**
- Modify: `app/static/index.html`, `app/static/admin.js`, `app/static/admin.html`, `app/static/admin-page.js`
- Create: `app/static/rageval.html`, `app/static/rageval.js`
- Modify: `tests/test_static.py` only for lightweight route/resource smoke.

**Process:** User-authorized Vibe Coding exception. Do not apply brainstorming, TDD, or code review to UI behavior.

- [ ] Extend SSE handling to bind citations to the current assistant bubble. Render only `[n]` values present in that message's citation map as buttons.
- [ ] Add accessible citation dialog: render the bounded SSE label immediately, fetch `/api/kb/chunks/{chunk_id}` on open, then display section_path, question and full answer; close button, Escape and backdrop close; no raw HTML injection. Handle loading, 404 and network failure inside the dialog.
- [ ] Add one-time 👍/👎 bar only after a nonempty final answer. Selection lights, other dims, both lock, “已反馈” appears; no backend call.
- [ ] Build `/rag-eval` using the JSON/API: report meta, four KPI, strategy/bucket charts, generation status, tables, faith ledger filters/actions and fixed job start/stop/log controls. Missing/partial report states remain usable.
- [ ] Add shared navigation. Run JS syntax and static route checks; perform live browser checks for citation, feedback, missing report, completed report and ledger actions. Save screenshots when browser tooling is available.
- [ ] Append note and commit.

### Task 14: End-to-end acceptance and finish

**Files:**
- Modify: `README.md`, `dev-notes/ch04.md`
- Create/update: `data/ch04/reports/rag_eval.json`, `data/ch04/reports/rag_eval.txt` according to repository artifact policy.

**Interfaces:** Final commands and evidence.

- [ ] Verify MySQL DDL and all Docker services healthy. Prove `knowledge` still exists, `knowledge_ch04` count/IDs match MySQL and online configuration points at `knowledge_ch04`.
- [ ] Run full `uv run pytest -q` and `make eval-check`; stop on any failure.
- [ ] Run four-strategy evaluation. Record overall and per-bucket Recall@5/MRR, evidence coverage, answer coverage, faithfulness and refusal rate.
- [ ] CLI prove a concrete model query hits through pure BM25.
- [ ] Browser/SSE prove cited shipping answer and citation source modal; prove feedback locks.
- [ ] Ask an out-of-scope question; verify explicit refusal and a new low_confidence_questions row with expected source/reason.
- [ ] Run `judge-check`; record usable-case count and agreement.
- [ ] Independently review backend/data changes and resolve findings. Append final four-item finish entry, commit, and report demo commands, tests and `dev-notes/ch04.md`.

## Self-review

- Spec coverage: Tasks 1–6 cover BM25/RRF/rerank/filter/query understanding; 7–8 cover DDL, citations, refusal and SSE; 9–10 cover 300 cases and four strategies; 11–12 cover ledger/judge/API/jobs; 13 covers both Vibe UIs; 14 covers every acceptance criterion.
- Placeholder scan: no TBD/TODO or unspecified “handle errors” steps remain. External thresholds use named settings and are calibrated by labeled/live tasks rather than hidden constants.
- Type consistency: `knowledge_ch04`, four strategy names, citation fields, repository methods and report paths are stable across producer and consumer tasks.
- Safety: no task drops `knowledge`, resets Ch02 data, exposes arbitrary commands, logs keys, or exports private conversation text without production redaction.

## Appendix A: Authoritative Ch04 MySQL DDL

```sql
SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';

CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态,人工点按钮改',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)    NULL                    COMMENT '处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';
```
