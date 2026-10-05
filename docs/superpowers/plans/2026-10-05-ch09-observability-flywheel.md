# Ch09 Observability and Data Flywheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有客服 Workflow 增加自托管 Langfuse 全链路观测、正式证据置信度闸、低置信问题飞轮、审核入库闭环和周期评估趋势。

**Architecture:** 聊天 Workflow 在清空或裁剪证据前保存逐轮快照，先走四信号数值闸，再在流式回答前完成证据自评；三个入口幂等写入 MySQL 问题池。后台固定任务把问题标准化并与全部审核状态语义查重，人工核准后以 SQL 为权威源增量写入当前 Milvus 混合集合。Langfuse callback 绑定到每次图执行，手写 observations 覆盖检索和工具；评估及飞轮任务复用 JobRunner，并用 MySQL advisory lock 与本地运行清单实现单机可恢复调度。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy 2、MySQL 8.4、LangGraph 1.2、LangChain 1.4、Langfuse self-hosted、Milvus 2.5+ 原生 BM25、BGE-M3、bge-reranker-v2-m3、原生 HTML/CSS/JavaScript、pytest、Playwright

**Spec:** `docs/superpowers/specs/2026-10-05-ch09-observability-flywheel-design.md`

## Global Constraints

- 严格采用用户提供的 `review_queue`、`eval_runs` 和 `low_confidence_questions` ALTER DDL，不新增业务表或列。
- Langfuse 开源自部署，链路数据只写本机自托管实例；真实密钥不提交仓库、不返回浏览器。
- `category` 固定为“用户问题补充”；审核写回当前 `settings.milvus_collection`，不能写旧 `knowledge` 集合。
- `retrieved_chunks` 只允许三态：SQL NULL、JSON `[]`、Top 3 JSON 数组。
- 置信度固定四信号权重：Top1 0.5、有效证据数 0.2、Top1-Top2 分差 0.2、关键条款命中 0.1。
- 阈值在 60 条可答与 20 条应拒校准集上选择；首要门槛为应拒样例误放率 0%，F1/F0.5 仅作报告指标。
- 查重候选包含待审、通过、驳回；命中通过/驳回只累计次数，状态不复活。
- 每周一 02:00 Asia/Shanghai 跑全量评估；一次只允许一轮，错过时最多补最近一轮。
- Prompt、标注和阈值使用冻结评估集验证；前端按 Vibe Coding 实施，用浏览器验收替代前端 TDD。
- 每个任务开始前用 Context7 核对该任务涉及的官方 API；每个任务完成立即追记 `dev-notes/ch09.md`。
- 真实 DeepSeek、SiliconFlow 与 Langfuse 验收在执行前单独确认第九章数据发送范围，不能沿用其它章节授权。

## Review Focus

- 同一回答重复点 👎 或客户端成功响应丢失：后端返回同一问题池记录，不能重复插入；Task 5 覆盖。
- 检索超时与成功零命中：前者写 NULL 并标运行故障，后者写 `[]` 并进入知识缺口；Task 3、4 覆盖。
- 查重命中已通过或驳回行：只增加 `occurrence_count`，状态和核准答案保持不变；Task 6 覆盖。
- 审核 SQL 已提交、Milvus 写入前进程退出：启动扫描能发现 pending chunk 并补齐，不能重复建知识；Task 7 覆盖。
- Langfuse、Metrics API 或调度器不可用：聊天继续，页面显示暂不可用，不能显示假零或重复发起付费评估；Task 8、9、10 覆盖。

---

### Task 1: Ch09 数据模型与可重跑迁移

**Files:**
- Modify: `app/db/models.py`
- Create: `scripts/migrate_ch09.py`
- Create: `tests/test_ch09_schema.py`
- Create: `tests/test_ch09_migration.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces: ORM `ReviewQueue`, `EvalRun`; extends `LowConfidenceQuestion.retrieved_chunks` and `.matched_review_id`.
- Produces: `scripts.migrate_ch09.main(argv: list[str] | None = None) -> int`, supporting apply and `--check`.

- [ ] **Step 1: Query current SQLAlchemy 2 MySQL JSON, ENUM, foreign-key and inspector APIs in Context7 and record the exact version/API notes in `dev-notes/ch09.md`.**

- [ ] **Step 2: Write schema tests for exact DDL parity.**

```python
def test_ch09_models_match_given_ddl():
    assert {column.name for column in ReviewQueue.__table__.c} >= {
        "id", "normalized_question", "ai_suggested_answer", "occurrence_count",
        "review_status", "approved_answer", "created_at", "updated_at",
    }
    assert str(ReviewQueue.__table__.c.review_status.type.compile(dialect=mysql.dialect())) == "ENUM('待审','通过','驳回')"
    assert EvalRun.__table__.c.metrics.type.__class__.__name__ == "JSON"
    assert LowConfidenceQuestion.__table__.c.retrieved_chunks.nullable is True
    assert LowConfidenceQuestion.__table__.c.matched_review_id.foreign_keys
```

- [ ] **Step 3: Run the focused tests and confirm they fail because the Ch09 models are absent.**

Run: `uv run pytest tests/test_ch09_schema.py -q`

Expected: FAIL importing `ReviewQueue` or finding the new columns.

- [ ] **Step 4: Add the exact user-provided table and column definitions to `app/db/models.py`.**

```python
class ReviewQueue(Base):
    __tablename__ = "review_queue"
    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(String(512), nullable=False)
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(mysql.INTEGER(unsigned=True), nullable=False, server_default=text("1"))
    review_status: Mapped[str] = mapped_column(mysql.ENUM("待审", "通过", "驳回"), nullable=False, server_default=text("'待审'"))
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    triggered_by: Mapped[str] = mapped_column(mysql.ENUM("定时", "手动"), nullable=False, server_default=text("'定时'"))
    dataset_size: Mapped[int] = mapped_column(mysql.INTEGER(unsigned=True), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
```

Include the exact comments, indexes, table options and FK names from the supplied DDL rather than shortening them as in the excerpt.

- [ ] **Step 5: Write migration tests for empty database, partial migration, exact `--check`, and incompatible existing columns.**

```python
def test_partial_migration_resumes_without_recreating_tables(fake_inspector, recorder):
    fake_inspector.tables = {"review_queue"}
    assert migrate_ch09.apply(recorder) == 0
    assert recorder.statements[0].startswith("CREATE TABLE eval_runs")
    assert all("DROP" not in sql.upper() for sql in recorder.statements)
```

- [ ] **Step 6: Implement `scripts/migrate_ch09.py` with `SET NAMES utf8mb4`, ordered create/alter operations, structural comparison, and non-destructive resume.**

```python
OPERATIONS = ("review_queue", "eval_runs", "low_confidence_extensions")

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    with engine.begin() as connection:
        return check(connection) if args.check else apply(connection)
```

- [ ] **Step 7: Run migration and regression tests.**

Run: `uv run pytest tests/test_ch09_schema.py tests/test_ch09_migration.py tests/test_ch04_schema.py tests/test_db_models.py -q`

Expected: PASS.

- [ ] **Step 8: Update `dev-notes/ch09.md` for Task 1 and commit.**

```bash
git add app/db/models.py scripts/migrate_ch09.py tests/test_ch09_schema.py tests/test_ch09_migration.py dev-notes/ch09.md
git commit -m "feat: add ch09 flywheel schema"
```

### Task 2: 证据快照与四信号置信度核心

**Files:**
- Create: `app/flywheel/__init__.py`
- Create: `app/flywheel/contracts.py`
- Create: `app/flywheel/evidence.py`
- Create: `app/flywheel/confidence.py`
- Create: `tests/test_ch09_evidence.py`
- Create: `tests/test_ch09_confidence.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces: `EvidenceSnapshot`, `ConfidenceSignals`, `ConfidenceDecision` Pydantic models.
- Produces: `snapshot_hits(query, strategy, hits, *, retrieval_id, created_at) -> EvidenceSnapshot`.
- Produces: `pool_snapshot(snapshot: EvidenceSnapshot | None, *, retrieval_completed: bool) -> list[dict] | None`.
- Produces: `compute_confidence(hits: Sequence[EvidenceHit | Mapping[str, Any]], config: ConfidenceConfig) -> ConfidenceDecision`.

- [ ] **Step 1: Query Pydantic strict-model APIs and Milvus returned score semantics in Context7; record which field is raw retrieval score and which is rerank score.**

- [ ] **Step 2: Write failing tests for Top 3 ordering and strict three-state serialization.**

```python
def test_pool_snapshot_has_exact_three_state_contract():
    assert pool_snapshot(None, retrieval_completed=False) is None
    assert pool_snapshot(empty_snapshot(), retrieval_completed=True) == []
    saved = pool_snapshot(snapshot_with_four_ranked_hits(), retrieval_completed=True)
    assert [row["rank"] for row in saved] == [1, 2, 3]
    assert all({"chunk_id", "section_path", "answer", "score", "rerank_score", "rank"} <= row.keys() for row in saved)
```

- [ ] **Step 3: Write failing tests for the four fixed signals and invalid scores.**

```python
def test_confidence_uses_fixed_weights_and_key_clause():
    decision = compute_confidence(hits, frozen_config(threshold=0.26))
    assert decision.signals.weights == {"top1": 0.5, "evidence_count": 0.2, "gap": 0.2, "key_clause": 0.1}
    assert decision.score == pytest.approx(
        0.5 * decision.signals.top1 + 0.2 * decision.signals.evidence_count
        + 0.2 * decision.signals.gap + 0.1 * decision.signals.key_clause
    )

def test_invalid_rerank_score_is_operational_error():
    with pytest.raises(InvalidConfidenceInput):
        compute_confidence([{"rerank_score": float("nan")}], frozen_config())
```

- [ ] **Step 4: Implement immutable contracts and snapshot sanitization.**

```python
class EvidenceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    retrieval_id: str
    query: str
    strategy: str
    created_at: datetime
    hits: list[EvidenceHit]

class ConfidenceDecision(BaseModel):
    allowed: bool
    score: float
    threshold: float
    reason: Literal["evidence_ok", "no_hits", "evidence_below_threshold"]
    signals: ConfidenceSignals
```

- [ ] **Step 5: Implement deterministic confidence calculation with exact fixed weights and section-deduplicated effective evidence count.**

```python
WEIGHTS = {"top1": 0.5, "evidence_count": 0.2, "gap": 0.2, "key_clause": 0.1}

def compute_confidence(hits, config):
    ranked = validate_and_sort_hits(hits)
    if not ranked:
        return ConfidenceDecision(allowed=False, score=0.0, threshold=config.threshold,
                                  reason="no_hits", signals=ConfidenceSignals.zero())
    signals = normalize_signals(ranked, config)
    score = sum(WEIGHTS[name] * getattr(signals, name) for name in WEIGHTS)
    return ConfidenceDecision(allowed=score >= config.threshold, score=score,
                              threshold=config.threshold,
                              reason="evidence_ok" if score >= config.threshold else "evidence_below_threshold",
                              signals=signals)
```

- [ ] **Step 6: Run focused tests, update notes and commit.**

Run: `uv run pytest tests/test_ch09_evidence.py tests/test_ch09_confidence.py -q`

```bash
git add app/flywheel tests/test_ch09_evidence.py tests/test_ch09_confidence.py dev-notes/ch09.md
git commit -m "feat: add evidence snapshot and confidence scoring"
```

### Task 3: 校准数据、阈值选择与冻结报告

**Files:**
- Create: `data/ch09/confidence_cases.jsonl`
- Create: `app/evaluation/confidence_calibration.py`
- Create: `scripts/calibrate_ch09.py`
- Create: `tests/test_ch09_calibration.py`
- Modify: `app/config.py`
- Modify: `.env.example`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Consumes: `compute_confidence(...)` from Task 2.
- Produces: `select_threshold(rows: list[CalibrationRow]) -> CalibrationReport`.
- Produces: `data/ch09/reports/confidence-calibration.json` containing data/model/corpus/config hashes.
- Produces: settings `workflow_confidence_report_path` and verified threshold loading; it does not silently trust a bare numeric environment value.

- [ ] **Step 1: Build the frozen 80-case manifest from Ch04: exactly 60 answerable and 20 expected refusals, with `fact_group`, `evidence_sufficient`, and reviewer reason.**

Example row:

```json
{"id":"cal-A-001","query":"签收后多久能退","fact_group":"returns-window","evidence_sufficient":true,"reason":"召回证据覆盖七日时限"}
```

- [ ] **Step 2: Write failing tests that enforce counts, fact-group separation, zero false-pass selection and no all-refuse success.**

```python
def test_selector_prefers_zero_false_pass_then_lowest_false_block():
    report = select_threshold(rows_for_thresholds({0.20: (1, 4), 0.26: (0, 9), 0.30: (0, 12)}))
    assert report.threshold == 0.26
    assert report.false_pass_rate == 0
    assert report.false_block_count == 9

def test_all_refuse_candidate_cannot_pass():
    with pytest.raises(NoSafeUsableThreshold):
        select_threshold(all_refused_rows())
```

- [ ] **Step 3: Implement calibration report generation and atomic save.**

```python
class CalibrationReport(BaseModel):
    threshold: float
    false_pass_rate: float
    false_block_rate: float
    precision: float
    recall: float
    f1: float
    f_0_5: float
    confusion: dict[str, int]
    weights: dict[str, float]
    dataset_hash: str
    corpus_hash: str
    rerank_model: str
    top_k: int
```

- [ ] **Step 4: Implement config loading that rejects report/model/Top-K/corpus mismatch as `ConfidenceConfigurationError`.**

- [ ] **Step 5: Run deterministic local validation without cloud calls.**

Run: `uv run pytest tests/test_ch09_calibration.py tests/test_ch09_confidence.py -q`

Expected: PASS and no report claims threshold 0.26 until live retrieval scores exist.

- [ ] **Step 6: Update notes and commit the manifest, calibrator and tests.**

```bash
git add data/ch09/confidence_cases.jsonl app/evaluation/confidence_calibration.py scripts/calibrate_ch09.py app/config.py .env.example tests/test_ch09_calibration.py dev-notes/ch09.md
git commit -m "feat: calibrate ch09 evidence threshold"
```

### Task 4: Workflow 快照、正式闸与生成前自评

**Files:**
- Modify: `app/workflow/contracts.py`
- Modify: `app/workflow/retrieval.py`
- Modify: `app/workflow/graph.py`
- Modify: `app/workflow/after_sales.py`
- Modify: `app/workflow/agent.py`
- Modify: `app/tools/business.py`
- Modify: `app/workflow/service.py`
- Modify: `app/context/compile.py`
- Create: `tests/test_ch09_workflow_gate.py`
- Create: `tests/test_ch09_self_check.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Adds `WorkflowState.evidence_snapshot: dict`, `.confidence: dict`, `.evidence_useful: bool | None`.
- Adds `TurnContext.confidence_config: ConfidenceConfig` and `TurnContext.record_low_confidence(raw, source, reason, snapshot, event_key) -> Awaitable[int]`.
- Produces final `AIMessage.additional_kwargs["evidence_snapshot"]` for the exact turn; this key is excluded by context compilation.

- [ ] **Step 1: Query LangGraph State reducers/runtime callback propagation and LangChain structured output in Context7; record exact APIs before editing graph code.**

- [ ] **Step 2: Write failing tests proving retrieval snapshot survives a rejected gate and operational errors do not enter the knowledge-gap pool.**

```python
async def test_rejected_gate_persists_snapshot_before_clearing_evidence(workflow):
    state = await workflow.ainvoke(policy_input("猫窝能机洗吗"))
    assert state["gate_allowed"] is False
    assert state["evidence"] == []
    assert state["evidence_snapshot"]["hits"]
    assert pool.calls[0].source == "retrieval_low_conf"

async def test_retrieval_timeout_is_operational_not_gap(workflow):
    await workflow.ainvoke(policy_input("猫窝能机洗吗"))
    assert pool.calls == []
```

- [ ] **Step 3: Write failing tests proving every knowledge path performs useful self-check before the first answer delta.**

```python
@pytest.mark.parametrize("path", ["forced_retrieval", "after_sales", "query_faq_tool"])
async def test_self_check_precedes_streaming(path, harness):
    events = await harness.run(path, useful=False)
    assert next(e for e in events if e.kind == "delta").data["delta"] == FALLBACK_TEXT
    assert harness.answer_model_calls == 0
    assert harness.pool.calls[0].source == "self_check"
```

- [ ] **Step 4: Replace `gate_result` with Task 2 confidence decision and preserve the snapshot before evidence is cleared.**

```python
snapshot = snapshot_hits(state["question"], result.strategy, result.hits,
                         retrieval_id=uuid4().hex, created_at=datetime.now(UTC))
decision = compute_confidence(snapshot.hits, runtime.context.confidence_config)
return {"evidence_snapshot": snapshot.model_dump(mode="json"),
        "confidence": decision.model_dump(),
        "evidence": registry.merge(result.hits) if decision.allowed else [],
        "gate_allowed": decision.allowed, "gate_reason": decision.reason}
```

- [ ] **Step 5: Add one `ensure_useful_evidence` node immediately before knowledge answer streaming and reuse `check_evidence`.**

```python
async def ensure_useful_evidence(state, runtime):
    decision = await check_evidence(runtime.context.agent.model, state["question"], state["evidence"])
    if not decision.useful:
        await runtime.context.record_low_confidence(
            state["raw_question"], "self_check", decision.reason,
            state.get("evidence_snapshot"), f"{state['turn_id']}:self_check",
        )
    return {"evidence_useful": decision.useful, "gate_reason": decision.reason}
```

- [ ] **Step 6: Save the internal snapshot on final/fallback assistant messages and explicitly strip it in `app/context/compile.py` and ordinary history APIs.**

- [ ] **Step 7: Run workflow regressions.**

Run: `uv run pytest tests/test_ch09_workflow_gate.py tests/test_ch09_self_check.py tests/test_ch05_gate.py tests/test_ch06_after_sales.py tests/test_ch07_context_compile.py tests/test_tools.py -q`

- [ ] **Step 8: Update notes and commit.**

```bash
git add app/workflow app/tools/business.py app/context/compile.py tests/test_ch09_workflow_gate.py tests/test_ch09_self_check.py dev-notes/ch09.md
git commit -m "feat: enforce calibrated evidence gate"
```

### Task 5: 问题池幂等写入与 👎 精确回捞

**Files:**
- Modify: `app/repositories/low_confidence.py`
- Create: `app/flywheel/pool.py`
- Create: `app/api/feedback.py`
- Modify: `app/main.py`
- Modify: `app/static/index.html`
- Create: `tests/test_ch09_pool.py`
- Create: `tests/test_ch09_feedback_api.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces: `record_gap(event: GapEvent, session_factory) -> int`.
- Produces: `POST /api/chat/feedback` body `{user_id, conversation_id, assistant_message_id, helpful:false}` and response `{low_confidence_id, accepted:true}`.
- Consumes exact-turn checkpoint snapshot from Task 4; never accepts evidence or raw question from the browser.

- [ ] **Step 1: Query FastAPI dependency/error response and SQLAlchemy advisory-lock transaction patterns in Context7.**

- [ ] **Step 2: Write failing repository tests for duplicate event, same text in a new turn, NULL/[]/array persistence, and concurrent retries.**

```python
def test_same_feedback_event_is_idempotent(session_factory):
    first = record_gap(feedback_event(message_id=91), session_factory)
    second = record_gap(feedback_event(message_id=91), session_factory)
    assert first == second
    assert count_low_confidence(session_factory) == 1
```

- [ ] **Step 3: Implement canonical event digest, MySQL `GET_LOCK`, short transaction recheck and insertion.**

```python
@dataclass(frozen=True)
class GapEvent:
    conversation_id: int | None
    raw_question: str
    source: Literal["retrieval_low_conf", "self_check", "user_feedback"]
    reason: str
    retrieved_chunks: list[dict] | None
    event_key: str
```

- [ ] **Step 4: Write failing API tests for ownership, non-assistant IDs, old-turn snapshot lookup, no-retrieval feedback and repeated POST.**

```python
def test_feedback_uses_the_selected_assistant_turn_not_latest(client, seeded_conversation):
    response = client.post("/api/chat/feedback", json={
        "user_id": "u1", "conversation_id": 7,
        "assistant_message_id": seeded_conversation.first_assistant_id, "helpful": False,
    })
    assert response.status_code == 200
    assert saved_row().retrieved_chunks[0]["chunk_id"] == seeded_conversation.first_chunk_id
```

- [ ] **Step 5: Implement feedback route and state-history lookup by `message_key/turn_id`; return 403/404/409/422 for ownership, missing row, stale checkpoint and invalid payload.**

- [ ] **Step 6: Vibe Coding frontend change: make 👎 POST first, lock and store local feedback only after HTTP success; show a retryable inline error on failure.**

- [ ] **Step 7: Run backend tests plus the existing static feedback regression.**

Run: `uv run pytest tests/test_ch09_pool.py tests/test_ch09_feedback_api.py tests/test_static.py tests/test_ch07_sidebar.py -q`

- [ ] **Step 8: Update notes and commit.**

```bash
git add app/flywheel/pool.py app/repositories/low_confidence.py app/api/feedback.py app/main.py app/static/index.html tests/test_ch09_pool.py tests/test_ch09_feedback_api.py dev-notes/ch09.md
git commit -m "feat: connect negative feedback to flywheel"
```

### Task 6: 标准化、全状态语义查重与待审队列

**Files:**
- Create: `app/flywheel/prompts.py`
- Create: `app/flywheel/processor.py`
- Create: `app/repositories/review_queue.py`
- Create: `scripts/process_flywheel.py`
- Modify: `app/core/jobs.py`
- Create: `tests/data/ch09_flywheel_cases.json`
- Create: `tests/test_ch09_flywheel_processor.py`
- Modify: `tests/test_jobs_api.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces strict `FlywheelDecision(normalized_question, ai_suggested_answer, matched_question_id)`.
- Produces `process_pending(*, session_factory, model, batch_size=20) -> ProcessReport`.
- Adds fixed JobRunner job `flywheel-process`; no caller-provided argv.

- [ ] **Step 1: Query LangChain structured output and SQLAlchemy `with_for_update` APIs in Context7.**

- [ ] **Step 2: Add labeled cases for paraphrase merge, product-model distinction, negation, unresolved reference, operational failure skip, passed match and rejected match.**

- [ ] **Step 3: Write failing tests for one model call, candidate-ID validation and all-status behavior.**

```python
@pytest.mark.parametrize("status", ["待审", "通过", "驳回"])
async def test_match_increments_without_changing_status(status, harness):
    existing = harness.review(status=status, approved_answer="核准答案" if status == "通过" else None)
    report = await harness.process(model_result={"normalized_question": "猫窝能否机洗",
        "ai_suggested_answer": "需看材质", "matched_question_id": existing.id})
    refreshed = harness.get_review(existing.id)
    assert refreshed.occurrence_count == 2
    assert refreshed.review_status == status
    assert refreshed.approved_answer == existing.approved_answer
    assert report.model_calls == 1
```

- [ ] **Step 4: Implement one structured prompt that receives the deidentified resolved question plus every review candidate and returns all three fields. The repository reads candidates in database pages, then builds one stable ID/question/status list for this single model call; reject the task with an explicit budget error rather than silently dropping tail candidates.**

```python
class FlywheelDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    normalized_question: str = Field(min_length=2, max_length=512)
    ai_suggested_answer: str | None
    matched_question_id: int | None
```

- [ ] **Step 5: Implement processor locking and atomic merge.**

Acquire `GET_LOCK('mewhelp:ch09:flywheel', 0)`, make the model call outside a DB transaction, then start a short transaction, lock the low-confidence row, recheck `matched_review_id IS NULL`, create/increment exactly once and set the link. Validate `matched_question_id` against supplied candidates.

- [ ] **Step 6: Mark operational-failure feedback as skipped in `data/runtime/ch09/flywheel-state.json` before any model call; use atomic replace and bounded retry counters.**

- [ ] **Step 7: Register `flywheel-process` in JobRunner and test that arbitrary arguments remain impossible.**

- [ ] **Step 8: Run unit/evaluation cases, update notes and commit.**

Run: `uv run pytest tests/test_ch09_flywheel_processor.py tests/test_jobs_api.py -q`

```bash
git add app/flywheel app/repositories/review_queue.py scripts/process_flywheel.py app/core/jobs.py tests/data/ch09_flywheel_cases.json tests/test_ch09_flywheel_processor.py tests/test_jobs_api.py dev-notes/ch09.md
git commit -m "feat: process flywheel review candidates"
```

### Task 7: 人工审核、增量混合索引与崩溃恢复

**Files:**
- Create: `app/flywheel/review.py`
- Create: `app/knowledge/incremental_indexer.py`
- Create: `app/api/review_queue.py`
- Modify: `app/main.py`
- Modify: `app/core/jobs.py`
- Create: `scripts/recover_ch09_index.py`
- Create: `tests/test_ch09_review_api.py`
- Create: `tests/test_ch09_incremental_indexer.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces `approve_review(review_id, approved_answer, expected_status, deps) -> ReviewResult`.
- Produces `index_pending_review_chunks(session_factory, embedder, store, *, limit=100) -> IndexReport`.
- Produces list/detail/decision APIs from the spec and fixed job `flywheel-index-recover`.

- [ ] **Step 1: Query current pymilvus upsert/BM25 function behavior and SQLAlchemy transaction locking in Context7.**

- [ ] **Step 2: Write failing tests for transactional SQL creation, repeated approval, concurrent reject, UTF-8 field caps and changed-answer conflict.**

```python
def test_approval_creates_one_pending_chunk_in_same_transaction(service):
    result = service.approve(review_id=4, approved_answer="按洗标清洗", expected_status="待审")
    assert result.review_status == "通过"
    chunk = service.chunk_for("数据飞轮 / review_queue/4")
    assert (chunk.category, chunk.content_type, chunk.vectorize_status) == ("用户问题补充", "faq", "pending")
```

- [ ] **Step 3: Implement review repository/API with expected-state conflicts and detail rows joined through `matched_review_id`.**

- [ ] **Step 4: Write failing index recovery tests for SQL-commit crash, remote-upsert/SQL-confirm crash and unrelated legacy pending chunks.**

```python
def test_recovery_only_indexes_approved_review_sources(deps):
    report = index_pending_review_chunks(**deps)
    assert report.indexed_ids == [deps.approved_pending_id]
    assert deps.legacy_pending_id not in report.indexed_ids
```

- [ ] **Step 5: Implement incremental indexer against `settings.milvus_collection`.**

```python
row = {"id": chunk.id, "question": chunk.questions, "answer": chunk.answer,
       "category": chunk.category,
       "text": f"分类：{chunk.category}\n问题：{chunk.questions}\n答案：{chunk.answer}",
       "section_path": chunk.section_path or "", "content_type": chunk.content_type or "",
       "is_key_clause": bool(chunk.is_key_clause), "dense": vector}
store.upsert([row])
```

After successful upsert, update that SQL row to `vector_id=str(id)` and `vectorize_status='done'`; repeated upsert must be safe.

- [ ] **Step 6: Add startup/periodic recovery and fixed JobRunner command without rebuilding the collection.**

- [ ] **Step 7: Run tests, update notes and commit.**

Run: `uv run pytest tests/test_ch09_review_api.py tests/test_ch09_incremental_indexer.py tests/test_ch04_indexer.py tests/test_knowledge_indexer.py -q`

```bash
git add app/flywheel/review.py app/knowledge/incremental_indexer.py app/api/review_queue.py app/main.py app/core/jobs.py scripts/recover_ch09_index.py tests/test_ch09_review_api.py tests/test_ch09_incremental_indexer.py dev-notes/ch09.md
git commit -m "feat: approve and index flywheel knowledge"
```

### Task 8: 自托管 Langfuse 与完整 trace 树

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `app/config.py`
- Modify: `.env.example`
- Create: `compose.langfuse.yaml`
- Create: `app/observability/__init__.py`
- Create: `app/observability/langfuse.py`
- Create: `app/observability/spans.py`
- Modify: `app/main.py`
- Modify: `app/workflow/service.py`
- Modify: `app/workflow/model.py`
- Modify: `app/knowledge/retrieval.py`
- Modify: `app/services/tool_executor.py`
- Create: `tests/test_ch09_observability.py`
- Modify: `Makefile`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces `ObservabilityRuntime.handler`, `trace_config(conversation_id, turn_id) -> RunnableConfig`, `flush(timeout_seconds) -> None`.
- Produces `observe_external(name, input, metadata)` async context manager.
- Adds settings for Langfuse host/public/secret keys and enabled flag.

- [ ] **Step 1: Use Context7 to pin a mutually compatible Langfuse server, Python SDK and LangChain callback API; record exact versions and Metrics API version before changing dependencies.**

- [ ] **Step 2: Write failing tests for one callback binding per execution, session/turn correlation, intent tag update, leaf usage, disabled mode and callback failure isolation.**

```python
async def test_trace_config_is_request_scoped(runtime):
    first = runtime.trace_config(7, "turn-a")
    second = runtime.trace_config(8, "turn-b")
    assert first is not second
    assert first["callbacks"].count(runtime.handler) == 1
    assert first["metadata"]["langfuse_session_id"] == "conversation:7"
```

- [ ] **Step 3: Add pinned SDK dependency and self-hosted Compose with isolated PostgreSQL, ClickHouse, Redis and object-storage volumes; expose Langfuse only on localhost:3000 and disable telemetry.**

- [ ] **Step 4: Implement runtime callback injection via each `graph.astream(..., config=trace_config)` call, not `compile(callbacks=...)`; after intent classification, attach `intent:<intent>` tag.**

- [ ] **Step 5: Instrument embedding, rerank, Milvus and ToolExecutor with child observations containing duration, attempts, result summary and errors; deterministic nodes carry input/output but no fake token usage.**

- [ ] **Step 6: Bound export time, flush on lifespan shutdown and ensure Langfuse failure never breaks chat.**

- [ ] **Step 7: Add `make langfuse-up` and `make langfuse-down` plus documented PowerShell equivalents.**

- [ ] **Step 8: Run tests and a local container smoke that creates one trace and retrieves it from Langfuse API.**

Run: `uv run pytest tests/test_ch09_observability.py tests/test_ch05_checkpoint.py tests/test_ch08_executor.py -q`

Run: `docker compose -f compose.langfuse.yaml config`

Expected: valid Compose; smoke trace contains resolve, intent, retrieval, gate, Agent/model and tool observations.

- [ ] **Step 9: Update notes and commit.**

```bash
git add pyproject.toml uv.lock app/config.py .env.example compose.langfuse.yaml app/observability app/main.py app/workflow app/knowledge/retrieval.py app/services/tool_executor.py tests/test_ch09_observability.py Makefile dev-notes/ch09.md
git commit -m "feat: add self-hosted Langfuse tracing"
```

### Task 9: 按意图聚合 token 与观测 API

**Files:**
- Create: `app/observability/usage.py`
- Create: `app/api/observability.py`
- Modify: `app/main.py`
- Create: `tests/test_ch09_usage.py`
- Create: `tests/test_ch09_observability_api.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces `summarize_usage(start, end, client) -> list[IntentUsage]`.
- Produces `GET /api/observability/intent-usage?days=7`.

- [ ] **Step 1: Verify the pinned Langfuse Metrics API tags grouping and pagination contract in Context7/self-hosted OpenAPI.**

- [ ] **Step 2: Write failing tests that count only model-generation leaf observations, separate online/offline traces, expose missing usage and avoid parent double counting.**

```python
def test_usage_does_not_double_count_parent_trace(fake_client):
    rows = summarize_usage(window(), fake_client.with_parent_total(1000).with_leaf_usage(600, 100))
    assert rows[0].input_tokens == 600
    assert rows[0].output_tokens == 100
```

- [ ] **Step 3: Implement paged tag aggregation with bounded date range, 30-second cache and explicit unavailable response.**

```python
class IntentUsage(BaseModel):
    intent: str
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_missing: int
    cost: float | None
```

- [ ] **Step 4: Run tests, update notes and commit.**

Run: `uv run pytest tests/test_ch09_usage.py tests/test_ch09_observability_api.py -q`

```bash
git add app/observability/usage.py app/api/observability.py app/main.py tests/test_ch09_usage.py tests/test_ch09_observability_api.py dev-notes/ch09.md
git commit -m "feat: report token usage by intent"
```

### Task 10: 周期评估、结果入表与可恢复调度

**Files:**
- Create: `app/evaluation/ch09_runs.py`
- Create: `app/evaluation/scheduler.py`
- Create: `scripts/eval_ch09.py`
- Modify: `app/core/jobs.py`
- Modify: `app/main.py`
- Create: `app/api/eval_runs.py`
- Create: `tests/test_ch09_eval_runs.py`
- Create: `tests/test_ch09_scheduler.py`
- Modify: `Makefile`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces `run_ch09_evaluation(triggered_by, run_id, deps) -> EvalRunResult` by adapting the Ch04 evaluator.
- Produces `FlywheelScheduler.tick(now: datetime) -> SchedulerDecision`.
- Produces `GET /api/eval-runs` and fixed JobRunner job `eval-flywheel`.

- [ ] **Step 1: Query Python scheduling/timezone APIs and SQLAlchemy JSON querying supported by MySQL 8.4; keep the scheduler dependency-free unless Context7 proves an existing dependency is required.**

- [ ] **Step 2: Write failing result tests for two independent runs, same-run idempotence, partial-run exclusion and report-present/row-missing repair.**

```python
def test_same_run_id_is_inserted_once(repository):
    first = repository.save_complete(run_result("run-1"))
    second = repository.save_complete(run_result("run-1"))
    assert first.id == second.id
    assert repository.count() == 1
```

- [ ] **Step 3: Implement Ch04 four-strategy reuse and metrics JSON containing Recall@5, MRR, Faithfulness, refusal rate, bucket scores, real denominators and all hashes.**

```python
metrics = {
    "run_id": run_id, "dataset_hash": dataset_hash, "corpus_hash": corpus_hash,
    "strategies": strategy_metrics, "recall_at_k": 5,
    "faithfulness": faithfulness, "refusal_rate": refusal_rate,
    "completed": completed, "failed": failed, "scheduled_slot": scheduled_slot,
}
```

- [ ] **Step 4: Write failing clock tests for Monday 02:00 Asia/Shanghai, first-enable behavior, one missed slot, multi-worker lock, manual conflict and failed paid-run non-retry.**

```python
def test_startup_catches_up_only_latest_missed_slot(scheduler):
    decision = scheduler.tick(datetime(2026, 10, 19, 9, tzinfo=SHANGHAI))
    assert decision.slot == "2026-10-19T02:00:00+08:00"
    assert decision.run_count == 1
```

- [ ] **Step 5: Implement atomic `data/runtime/ch09/eval-schedule.json`, MySQL advisory execution lock and bounded catch-up.**

- [ ] **Step 6: Register `eval-flywheel`, add `make eval-flywheel`, API listing and lifespan scheduler startup/shutdown.**

- [ ] **Step 7: Run tests, update notes and commit.**

Run: `uv run pytest tests/test_ch09_eval_runs.py tests/test_ch09_scheduler.py tests/test_eval_ch04_runner.py tests/test_jobs_api.py -q`

```bash
git add app/evaluation/ch09_runs.py app/evaluation/scheduler.py scripts/eval_ch09.py app/core/jobs.py app/main.py app/api/eval_runs.py tests/test_ch09_eval_runs.py tests/test_ch09_scheduler.py Makefile dev-notes/ch09.md
git commit -m "feat: schedule flywheel evaluations"
```

### Task 11: 待审队列与观测成本页面（Vibe Coding）

**Files:**
- Create: `app/static/review.html`
- Create: `app/static/review.js`
- Create: `app/static/review.css`
- Create: `app/static/observability.html`
- Create: `app/static/observability.js`
- Create: `app/static/observability.css`
- Modify: `app/static/admin.html`
- Modify: `app/main.py`
- Create: `scripts/verify_ch09_browser.py`
- Modify: `tests/test_static.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Consumes Task 7 review APIs, Task 9 usage API and Task 10 eval-runs API.
- Produces browser routes `/review` and `/observability`.

- [ ] **Step 1: Implement `/review` layout with count-sorted cards, source/time badges, AI answer preview, three audit gates and status filters.**

- [ ] **Step 2: Implement detail expansion showing every merged raw question and exact snapshot states: historical/no retrieval, zero hits, or Top 3 evidence with rerank score and section path. Render all model/database strings with `textContent`.**

- [ ] **Step 3: Implement approve modal prefilled with AI suggestion; require a nonempty edited/confirmed answer and expected status. Implement reject and 409 reload handling.**

- [ ] **Step 4: Implement `/observability` with seven-day intent token/cost table, unavailable/missing-usage states, two-run metric trend, denominator/version labels and manual fixed-job buttons.**

- [ ] **Step 5: Add navigation links and FastAPI file routes.**

```python
@app.get("/review", include_in_schema=False)
async def review_page() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "review.html")

@app.get("/observability", include_in_schema=False)
async def observability_page() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "observability.html")
```

- [ ] **Step 6: Extend static route smoke and create Playwright acceptance for list/detail, NULL/[]/array, approve, reject, unavailable usage and trend display.**

Run: `uv run pytest tests/test_static.py -q`

Run: `uv run python -m scripts.verify_ch09_browser --base-url http://127.0.0.1:8000 --output data/ch09/reports/browser.json`

- [ ] **Step 7: Update notes and commit.**

```bash
git add app/static app/main.py scripts/verify_ch09_browser.py tests/test_static.py dev-notes/ch09.md
git commit -m "feat: add flywheel review and observability pages"
```

### Task 12: 综合回归、真实验收、代码审查与交付

**Files:**
- Create: `scripts/eval_ch09_acceptance.py`
- Create: `data/ch09/reports/acceptance-local.json`
- Create after authorization: `data/ch09/reports/acceptance-live.json`
- Create after runs: `data/ch09/reports/confidence-calibration.json`
- Modify: `README.md`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Produces a machine-readable acceptance report with individual checks and provider call counts.
- Produces final demo commands and evidence paths.

- [ ] **Step 1: Run the full local test suite and save the exact pass/fail count.**

Run: `uv run pytest -q`

Expected: all tests pass; any unrelated pre-existing failure is diagnosed and recorded before proceeding.

- [ ] **Step 2: Run local deterministic Ch09 acceptance covering all three pool sources, three snapshot states, all-status dedup, approval idempotence, crash recovery, trace failure isolation, scheduler locks and two fake evaluation rounds.**

Run: `uv run python -m scripts.eval_ch09_acceptance --mode local --output data/ch09/reports/acceptance-local.json`

- [ ] **Step 3: Apply/check the real migration and start only the infrastructure the user runs locally.**

```powershell
uv run python -m scripts.migrate_ch09
uv run python -m scripts.migrate_ch09 --check
docker compose -f compose.langfuse.yaml up -d
```

- [ ] **Step 4: Before any cloud call, obtain explicit Chapter 9 authorization specifying that demo questions, retrieved chunks, normalized questions and the 80-case calibration/300-case evaluation data may be sent to the configured DeepSeek and SiliconFlow endpoints.**

- [ ] **Step 5: After authorization, run live calibration and verify the actual threshold; do not force 0.26 if scores differ.**

Run: `uv run python -m scripts.calibrate_ch09 --live --output data/ch09/reports/confidence-calibration.json`

Expected: false-pass rate 0%, at least one supported question allowed, report includes the actual threshold and false-block rate.

- [ ] **Step 6: Run two truly independent full evaluation rounds and verify two `eval_runs` rows and a comparable trend.**

```powershell
$env:TRIGGER='手动'; uv run python -m scripts.eval_ch09 --fresh
$env:TRIGGER='定时'; uv run python -m scripts.eval_ch09 --fresh
```

- [ ] **Step 7: Run live browser flywheel loop: ask an unknown question, inspect Langfuse trace, process the pool, inspect snapshots, approve a corrected answer, wait for indexing, ask again, then click 👎 on a separate answer.**

Run: `uv run python -m scripts.eval_ch09_acceptance --mode live --output data/ch09/reports/acceptance-live.json`

- [ ] **Step 8: Request code review, fix every confirmed finding, rerun affected tests, then run the full suite once more.**

Run: `uv run pytest -q`

- [ ] **Step 9: Update README and append separate `code review` and `finish` entries to `dev-notes/ch09.md`, including commands, exact test counts, live report paths, corrections and remaining limitations.**

- [ ] **Step 10: Commit final evidence and documentation.**

```bash
git add scripts/eval_ch09_acceptance.py data/ch09/reports README.md dev-notes/ch09.md
git commit -m "test: verify ch09 flywheel end to end"
```

## Completion Gate

Chapter 9 is complete only when all twelve tasks are checked, full tests pass, the real Langfuse UI contains a navigable trace tree, confidence calibration demonstrates zero false passes without all-refusal, the flywheel closes from unknown question through approval to a searchable cited answer, two independent full evaluation runs are stored, and `dev-notes/ch09.md` contains the brainstorm, plan approval, each task, code review and finish entries.
