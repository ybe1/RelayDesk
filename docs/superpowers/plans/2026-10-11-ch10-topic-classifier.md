# Ch10 多标签主题分类器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 AutoDL 全参微调 `hfl/chinese-roberta-wwm-ext`，在本地批量归类问题池并展示 17 类主题分布。

**Architecture:** 本地准备并脱敏真实问题，DeepSeek 按统一术语表预标，人工抽审后冻结真实验证/测试集；AutoDL 只负责训练和评测；下载完整模型包后，本地独立任务批量推理并幂等写 MySQL。实时聊天不加载分类器。

**Tech Stack:** Python 3.12、PyTorch、Transformers、SQLAlchemy/MySQL、FastAPI、原生 HTML/CSS/JS、AutoDL GPU、DeepSeek 现有模型配置。

**Spec:** `docs/superpowers/specs/2026-10-11-ch10-topic-classifier-design.md`

## Global Constraints

- 底座固定 `hfl/chinese-roberta-wwm-ext`；17 标签、多标签、全参数微调，不用 LoRA。
- 先脱敏，才可发送到 DeepSeek 或 AutoDL；匿名 ID 映射只存本机忽略目录。
- 按真实问题组做目标 80/10/10 切分；模拟问题和增强只进训练集；验证/测试必须真实且人工确认。
- `topic_classifications` 字段、类型、唯一键、外键、默认值、InnoDB/utf8mb4 与用户 DDL 保持一致。
- 用户已批准脱敏数据发往 DeepSeek 与 AutoDL；实际租用 AutoDL 的收费步骤仍需展示配置和成本后执行。
- 每项完成即追加 `dev-notes/ch10.md` 四项记录：用户关键原话、关键产出、纠偏、翻车返工。
- 涉及库 API 先用 Context7 查官方文档；每个代码任务用 TDD，纯数据/训练产物改用标注样例或评估集验证；前端按 Vibe Coding 直接修改。

## Review Focus

1. 带手机号、地址或订单号的真实问题：外发包不得残留可识别信息；Task 2 的脱敏样例覆盖。
2. 一条近似问法被分进训练和测试：必须拒绝导出；Task 3 的组泄漏测试覆盖。
3. 标签次序与模型包不一致：推理必须拒绝加载；Task 6 的模型清单测试覆盖。
4. `其他` 与业务标签同时出现：写库前必须规范成互斥语义；Task 1、Task 6 的测试覆盖。
5. 重跑批量任务或遇到数据库短暂失败：不得重复行或丢进度；Task 6 的 SQLite/MySQL 测试覆盖。

---

## File Map

| 文件 | 单一职责 |
|---|---|
| `app/topic/taxonomy.json`、`app/topic/taxonomy.py` | 17 类权威表及校验 |
| `app/topic/privacy.py` | 脱敏、清洗和不可确认样本拒绝 |
| `app/topic/dataset.py`、`scripts/topic_dataset.py` | 抽取、预标、人工复核文件、分组切分、训练包 |
| `app/topic/train.py`、`scripts/train_topic.py` | AutoDL 全参训练、断点恢复、模型包 |
| `app/topic/report.py`、`scripts/eval_topic.py` | 阈值冻结、逐类测试指标、误例抽审文件 |
| `app/topic/infer.py`、`scripts/classify_topics.py` | 本地批量推理和 MySQL 幂等写入 |
| `app/db/models.py`、`scripts/migrate_ch10.py` | 精确 DDL 模型与可重跑迁移 |
| `app/api/topics.py`、`app/static/topics.*` | 主题统计接口与后台页面 |
| `app/core/jobs.py`、`app/main.py`、`app/static/admin.js` | 固定任务与导航接线 |

## Task 1: 术语表与标签契约

**Files:** Create `app/topic/taxonomy.json`, `app/topic/taxonomy.py`, `tests/test_ch10_taxonomy.py`.

**Interfaces:** `LABELS: tuple[str,...]`; `taxonomy_hash() -> str`; `validate_labels(value: list[str]) -> list[str]`，输出按权威次序排序，未知、重复、空列表及“其他”混合一律抛 `ValueError`。

- [ ] **Step 1:** 先写失败测试：断言 `len(LABELS)==17`，每类有非空边界说明；`validate_labels(["退换货","尺码"])` 保留两类，`validate_labels(["其他","物流"])` 抛错；修、退、运费/物流、价保/优惠边界样例进入测试数据。
- [ ] **Step 2:** 运行 `uv run pytest tests/test_ch10_taxonomy.py -q`，确认红灯源于契约尚不存在。
- [ ] **Step 3:** 写 JSON 术语表与纯函数：`LABELS=tuple(row["name"] for row in taxonomy["labels"])`；`validate_labels` 按 `LABELS` 排序并执行互斥校验；SHA256 对规范 JSON 计算版本哈希。

  ```python
  def validate_labels(value: list[str]) -> list[str]:
      if not value or len(value) != len(set(value)) or set(value) - set(LABELS):
          raise ValueError("invalid labels")
      if "其他" in value and len(value) != 1:
          raise ValueError("其他 must be exclusive")
      return [label for label in LABELS if label in value]
  ```
- [ ] **Step 4:** 跑聚焦测试，追加阶段记录，提交本任务。

## Task 2: 数据抽取、清洗和脱敏

**Files:** Create `app/topic/privacy.py`, `app/topic/dataset.py`, `scripts/topic_dataset.py`, `tests/test_ch10_privacy.py`, `tests/test_ch10_export.py`.

**Interfaces:** `sanitize_question(text: str) -> str | None`，不确定则 `None`；`export_real_questions(session, output_dir: Path) -> dict`，仅导出匿名 `sample_id,text,source=real`，本地另存 ID 映射；`prelabel_batch(rows, llm) -> list[dict]` 使用统一术语表和严格 JSON。

- [ ] **Step 1:** 写失败测试：手机号、地址、订单号替换为占位符；无关空白清理；无法安全识别的个人信息拒绝导出；训练包不得出现 `conversation_id`、`question_id` 或原号；LLM 返回非法标签进入待复核文件。
- [ ] **Step 2:** 运行 `uv run pytest tests/test_ch10_privacy.py tests/test_ch10_export.py -q`，确认红灯。
- [ ] **Step 3:** 以现有 SQLAlchemy session 分页读 `LowConfidenceQuestion`；写本地 `data/ch10/private/` 映射并加 `.gitignore`；外发 JSONL 只含匿名字段。复用项目的 DeepSeek 客户端配置，调用前再次做敏感模式扫描；API 失败不标为已完成。CLI 提供 `export`、`prelabel`、`review-sheet` 子命令；不自动上传。

  ```python
  safe = sanitize_question(row.raw_question)
  if safe is None:
      rejected.append({"local_question_id": row.id, "reason": "privacy_review"})
  else:
      exported.append({"sample_id": uuid.uuid4().hex, "text": safe, "source": "real"})
  ```
- [ ] **Step 4:** 运行脱敏标注样例、聚焦测试；检查生成包没有真实号码；记录并提交。

## Task 3: 人工抽审、分组切分和训练增强

**Files:** Modify `app/topic/dataset.py`, `scripts/topic_dataset.py`; create `tests/test_ch10_split.py`, `data/ch10/label_examples.jsonl`.

**Interfaces:** `split_groups(rows: list[dict], ratios: tuple[float,float,float], seed: int) -> tuple[list[dict],list[dict],list[dict]]`；`freeze_dataset(reviewed_rows: list[dict], seed: int, output_dir: Path) -> dict`，输出 `train/validation/test.jsonl`、分组清单和 SHA256；`augment_train(rows, seed) -> list[dict]` 只能接训练行。

- [ ] **Step 1:** 写失败测试：同组近似问法不能跨份；模拟行只在训练集；增强行不出现在验证/测试；17 类分布与真实比例、稀有类不可评估原因写入 manifest；缺人工确认的行不能进入验证/测试。
- [ ] **Step 2:** 运行 `uv run pytest tests/test_ch10_split.py -q`，确认红灯。
- [ ] **Step 3:** 先对规范化文本做精确分组，再把近似组人工复核或以冻结的相似度规则归组；用多标签迭代分层在组级分配，固定 seed。若组限制使 80/10/10 不能精确满足，保留隔离并报告偏差。模拟问题照术语表生成、抽审后只追加训练；词替换和句式增强仅处理训练行。输出可供人工改标签的审阅 CSV/JSONL，冻结时校验审阅标志。

  ```python
  train_real, validation, test = split_groups(real_rows, ratios=(.8, .1, .1), seed=seed)
  assert all(row["reviewed"] for row in validation + test)
  train = augment_train(train_real + synthetic_rows, seed)
  assert not ({r["group_id"] for r in train_real} & {r["group_id"] for r in test})
  ```
- [ ] **Step 4:** 跑标注样例和切分测试，人工检查清单、记录并提交。

## Task 4: MySQL 模型与迁移

**Files:** Modify `app/db/models.py`; create `scripts/migrate_ch10.py`, `tests/test_ch10_schema.py`, `tests/test_ch10_migration.py`.

**Interfaces:** `TopicClassification` ORM；`scripts.migrate_ch10.inspect_state(connection) -> dict` 与 `planned_operations(state) -> list[str]`；CLI `--check`。

- [ ] **Step 1:** 写失败测试：ORM 编译后的 `BIGINT UNSIGNED`、JSON、DATETIME、`uk_question_id`、`fk_topic_question` 和 `CURRENT_TIMESTAMP` 与 DDL 一致；已有正确表重复迁移为零操作；错列/错索引报错。
- [ ] **Step 2:** 运行 `uv run pytest tests/test_ch10_schema.py tests/test_ch10_migration.py -q`，确认红灯。
- [ ] **Step 3:** 实现 ORM 及显式 `SET NAMES utf8mb4`、建表和结构检查。先确认 `low_confidence_questions` 存在；DDL 非事务中断后可重跑，不删旧表。使用 MySQL 实库 `--check` 和迁移各跑一次，仅在数据库可用时做。

  ```python
  class TopicClassification(Base):
      __tablename__ = "topic_classifications"
      __table_args__ = (Index("uk_question_id", "question_id", unique=True),
                        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})
      id = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
      question_id = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("low_confidence_questions.id", name="fk_topic_question"), nullable=False)
      labels = mapped_column(JSON, nullable=False)
      classified_at = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
  ```

  上面的字段草图不代替用户提供的完整注释和表注释。
- [ ] **Step 4:** 跑聚焦测试和迁移 `--check`，记录并提交。

## Task 5: AutoDL 全参训练和封存测试报告

**Files:** Create `app/topic/train.py`, `app/topic/report.py`, `scripts/train_topic.py`, `scripts/eval_topic.py`, `tests/test_ch10_metrics.py`, `docs/ch10-autodl-runbook.md`; modify `pyproject.toml` for isolated training/inference optional dependencies.

**Interfaces:** `train_model(dataset_dir: Path, output_dir: Path, config: TrainConfig) -> Path`; `calibrate_thresholds(logits, labels) -> list[float]`; `evaluate_holdout(logits, labels, thresholds) -> dict`。模型包含 `manifest.json`、tokenizer、权重、`thresholds.json`、报告。

- [ ] **Step 1:** 用标注评估小样本写指标测试：2×2 混淆矩阵、逐类 P/R/F1、零正例的 `not_evaluable`、阈值只看验证集；多标签例同时命中尺码与退换货。
- [ ] **Step 2:** 跑 `uv run pytest tests/test_ch10_metrics.py -q`，确认红灯。
- [ ] **Step 3:** Context7 核对当前 Transformers `Trainer`、`TrainingArguments`、多标签输出和早停 API，锁定依赖。训练采用 `AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=17, problem_type="multi_label_classification")`；所有底座参数 `requires_grad=True`。训练配置记录 lr、weight decay、dropout、batch、累积、epoch、seed、早停；验证集选最佳 checkpoint/阈值，封存测试集只评一次。报告包含每类四格、P/R/F1、micro/macro、分母、误例抽审文件。runbook 写 AutoDL 上传/训练/下载/停止实例的准确命令及实例规格检查。

  ```python
  model = AutoModelForSequenceClassification.from_pretrained(
      "hfl/chinese-roberta-wwm-ext", num_labels=len(LABELS),
      problem_type="multi_label_classification")
  assert all(parameter.requires_grad for parameter in model.parameters())
  trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                    eval_dataset=validation_ds, processing_class=tokenizer,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
  trainer.train(resume_from_checkpoint=resume_path)
  ```
- [ ] **Step 4:** 用小型本地模型桩验证 CLI/产物结构，不冒称 RoBERTa 真训练；在 AutoDL 跑真实训练后保存终端命令、模型哈希和报告，下载核验；记录并提交代码/报告摘要，权重不入 Git。

## Task 6: 本地批量推理与幂等写库

**Files:** Create `app/topic/infer.py`, `scripts/classify_topics.py`, `tests/test_ch10_infer.py`; modify `app/core/jobs.py`.

**Interfaces:** `load_bundle(path: Path) -> TopicPredictor` 验 manifest/标签次序/阈值；`classify_pending(session, predictor, batch_size: int, reclassify: bool=False) -> dict`；固定 JobRunner 名 `topic-classify`。

- [ ] **Step 1:** 写失败测试：标签次序/术语表哈希不符拒绝加载；`其他` 与业务类互斥；首次写库、重复运行零新增、显式重归类更新原行；中途数据库失败后重跑无重复。
- [ ] **Step 2:** 运行 `uv run pytest tests/test_ch10_infer.py -q`，确认红灯。
- [ ] **Step 3:** 只在子进程加载模型；按页取未分类问题，脱敏后批量 tokenizer + sigmoid + 逐类阈值；零业务标签时置 `其他`；MySQL `INSERT ... ON DUPLICATE KEY UPDATE` 实现显式重归类，普通模式保留旧值。固定任务只接受受控参数，返回处理/跳过/失败数及模型哈希。

  ```python
  labels = [name for name, score, cutoff in zip(LABELS[:-1], scores[:-1], thresholds[:-1]) if score >= cutoff]
  labels = validate_labels(labels or ["其他"])
  statement = mysql_insert(TopicClassification).values(question_id=question_id, labels=labels)
  if reclassify:
      statement = statement.on_duplicate_key_update(labels=statement.inserted.labels,
                                                     classified_at=func.now())
  else:
      statement = statement.on_duplicate_key_update(question_id=statement.inserted.question_id)
  session.execute(statement)
  ```
- [ ] **Step 4:** 聚焦测试、已有 JobRunner 回归、实际模型包烟测及 MySQL 行数核验；记录并提交。

## Task 7: 后台主题分布页

**Files:** Create `app/api/topics.py`, `app/static/topics.html`, `app/static/topics.css`, `app/static/topics.js`; modify `app/main.py`, `app/static/admin.js`; create `tests/test_ch10_topics_api.py`.

**Interfaces:** `GET /api/topics/distribution` 返回 `model_version,total,classified,latest_at,counts[{label,count}]`；`/topics` 页面；固定 `topic-classify` 任务入口复用现有 Jobs API。

- [ ] **Step 1:** 后端先写失败测试：多标签行分别计数、空库返回 17 个零值、未分类计入 total 不计 classified、接口复用现有后台访问控制。
- [ ] **Step 2:** 运行 `uv run pytest tests/test_ch10_topics_api.py -q`，确认红灯；实现聚合与路由，再跑绿灯。

  ```python
  @router.get("/api/topics/distribution")
  def distribution(session: Session = Depends(get_session)) -> dict:
      rows = session.scalars(select(TopicClassification)).all()
      counts = {label: sum(label in row.labels for row in rows) for label in LABELS}
      return {"counts": [{"label": key, "count": value} for key, value in counts.items()]}
  ```

  生产查询需分页/聚合，补齐总量、已归类数、最近时间和模型版本；这里的草图只固定多标签计数语义。
- [ ] **Step 3:** 前端按 Vibe Coding 直接做清晰的条形图、总量与模型信息、刷新/运行按钮、错误提示；用浏览器打开并目视检查空库与多标签统计，移动宽度下检查排版。
- [ ] **Step 4:** 跑 API 与页面烟测，记录并提交。

## Task 8: 端到端验收、审查与交付

**Files:** Modify `dev-notes/ch10.md`; add concise results under `reports/ch10/` excluding原文、密钥、权重。

**Interfaces:** 不新增运行接口；验证 Task 1–7 的成果。

- [ ] **Step 1:** 按官方文档复核锁定 API 与 AutoDL 运行命令；运行全量 `uv run pytest`、DDL `--check`、脱敏包检查、训练报告验证、本地一批真实问题分类和页面统计。
- [ ] **Step 2:** 人工抽审错误样本、核对“买大了想退”多标签、逐类混淆矩阵和真实样本分母；明确未达标或不可评估类，不伪称通过。
- [ ] **Step 3:** 请求代码审查，修复阻断问题后重跑相关验证；逐阶段补 `dev-notes/ch10.md`，最后记录 review 结论与 finish。
- [ ] **Step 4:** 交付 AutoDL 训练、本地分类、网页启动的实际命令，测试结果和 dev-notes 路径；按 finishing-a-development-branch 技能处理分支集成。
