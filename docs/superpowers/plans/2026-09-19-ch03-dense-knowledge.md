# Ch03 Dense Knowledge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (user-selected) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成可在浏览器操作的 dense 知识库，替换 query_faq 内部检索，支持文本去重、手动挖知识及故障重跑。

**Architecture:** 保留同步 SQLAlchemy/MySQL 架构。离线任务先写原文 pending，再按相同 ID upsert Milvus 服务读模型并回填 done。在线仅查询 Milvus；API、CLI 和页面共享服务函数。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy 2.x、MySQL 8.4、Docker Milvus standalone、PyMilvus、LangChain、langchain-text-splitters、硅基流动 BAAI/bge-m3、uv、pytest、原生静态 HTML/JS。

**Spec:** `docs/superpowers/specs/2026-09-19-ch03-dense-knowledge-design.md`

## Global Constraints

- 固定 Docker Milvus standalone（非 Lite）与 BAAI/bge-m3 1024 维 dense/COSINE；在线不回 MySQL，不加关键词 fallback/混合检索/重排。
- 保持 query_faq(keyword: str) 与实际 JSON `{"keyword": string, "matches": [{"question": string, "answer": string, "category": string}]}`；Milvus 必须保存 category。
- 用户两表 DDL 见末尾附录，原样复制；Ch02 四表和数据不改不清。同步 SessionLocal 保留。
- 每条新知识只嵌入 category + "\n" + questions + "\n" + answer，元数据不进入向量。
- 统一规范化 QA 指纹，保留同标题不同正文与相同问题不同答案；不做语义判重、不装 APScheduler。
- 后端 TDD；Prompt/语料走标注 eval；前端 Vibe Coding 与浏览器操作验证。
- 每任务开始前核对涉及的 Context7 版本接口；不得凭记忆照搬课程旧代码。更新 uv.lock。
- 每阶段即时写 dev-notes/ch03.md 四项。密钥只在 .env，日志不输出凭据/完整私有对话。
- 用户已授权修改后执行，并沿用 ch01 已确认的直接 main 与 Subagent-Driven 偏好，无需再问执行模式或创建 worktree；阶段依赖无法满足时完成独立工作并精确报告门槛。每个经过验证的任务单独提交，不推送远端。

## 文件职责与共用接口

| 文件 | 单一职责 |
|---|---|
| db/ch03.sql、compose.milvus.yaml、app/db/models.py | 两表精确映射和独立 standalone 基础设施 |
| app/knowledge/chunks.py、sources.py、dedup.py | ChunkDraft、统一切块、材料白名单、规范化 QA 指纹 |
| app/knowledge/importer.py、locks.py | 原子去重导入、连接级单写者锁 |
| app/knowledge/embedding.py、vector_store.py | 云端嵌入和 Milvus 读模型适配 |
| app/knowledge/indexer.py、extraction.py、extraction_prompts.py | 双写恢复、分批抽取、Prompt |
| app/knowledge/retrieval.py、app/tools/business.py | 在线纯 Milvus 检索及现有工具适配 |
| app/knowledge/queries.py、app/api/kb.py、admin.py | 管理只读账本、KB/聚合 API |
| app/core/jobs.py、app/api/jobs.py | 固定参数作业注册、状态、日志尾和停止 |
| app/static/kb.html、kb.js、admin.html、admin.js、admin.css | 六区录入页、聚合首页、共享导航 |
| scripts/kb.py、scripts/eval_ch03.py | 跨平台唯一 CLI、真实检索/抽取评估 |
| data/kb/*.md、tests/data/ch03_*.json | 演示材料、标注集 |

公共类型放 app/knowledge/chunks.py：

```python
from dataclasses import dataclass
@dataclass(frozen=True)
class ChunkDraft:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool = False
    is_table: bool = False  # 只用于预览，不增加 DDL 字段

@dataclass(frozen=True)
class ImportResult:
    inserted: int
    skipped: int
    ids: tuple[int, ...]
```

统一函数签名（同步，session_factory 使用既有 SessionLocal）：

```python
parse_markdown(text: str, source: str, content_type: str) -> list[ChunkDraft]
qa_fingerprint(question: str, answer: str) -> str
import_chunks(session_factory, drafts: list[ChunkDraft]) -> ImportResult
extract_batches(session_factory, model, batch_size: int = 30) -> dict[str, int]
deduplicate_staging(session_factory) -> dict[str, int]
index_pending(session_factory, embedder, store, batch_size: int = 16,
              after_upsert=None) -> dict[str, int]
# 各写函数内部不抢锁；CLI/API 边界在 writer_lock(engine) 上下文中调用。
# RetrievalHit 定义于 retrieval.py：id:int, score:float, question/answer/category:str。
search_knowledge(query: str, embedder, store, top_k: int = 5,
                 min_score: float = 0.55) -> list[dict]
```

## Task 1：保留现有架构，接入两表与 standalone

**Files:** 新建 db/ch03.sql、compose.milvus.yaml、tests/test_ch03_schema.py；修改 app/db/models.py、app/config.py、.env.example、pyproject.toml、uv.lock、.gitignore。读取 compose.yaml，不改 MySQL 卷。

**Interfaces:** 输出 KnowledgeChunk、QAExtractionStaging，以及 embed_base_url/embed_model/embed_api_key/milvus_uri/rag_top_k/rag_min_score。embed_api_key 允许为空以免阻断旧聊天启动，实际调用时再报缺失。

- [ ] 核对 Context7 和已安装依赖。记录 Docker/MySQL 状态、密钥是否存在，不输出值。下载/启动服务失败单列门槛。
- [ ] 写 schema 红测试，检查完整附录字段集、ENUM、nullability、默认值、索引、FK ondelete；以下是必须断言的关键项：

```python
def test_schema_preserves_user_ddl():
    from app.db.models import KnowledgeChunk, QAExtractionStaging
    chunk = KnowledgeChunk.__table__
    assert chunk.c.id.type.unsigned
    assert chunk.c.vectorize_status.type.enums == ['pending', 'done']
    assert next(iter(chunk.c.prev_chunk_id.foreign_keys)).ondelete == 'SET NULL'
    assert QAExtractionStaging.__table__.c.status.type.enums == ['extracted', 'kept', 'discarded']
    assert {'category', 'questions', 'answer', 'section_path', 'vector_id'} <= set(chunk.c.keys())
```

- [ ] 运行 `uv run pytest tests/test_ch03_schema.py -q` 确认红。DDL 原样复制，ORM 按现有 mapped_column/mysql.BIGINT(unsigned=True) 模式映射。写入示例：

```python
prev_chunk_id = mapped_column(mysql.BIGINT(unsigned=True),
    ForeignKey('knowledge_chunks.id', name='fk_chunks_prev', ondelete='SET NULL'), nullable=True)
vectorize_status = mapped_column(mysql.ENUM('pending', 'done'),
    nullable=False, server_default=text("'pending'"))
```

- [ ] 按官方固定版本 Compose 建 etcd/MinIO/standalone，端口 19530，只添加新命名卷；`docker compose -f compose.yaml -f compose.milvus.yaml up -d`。执行两表 SQL 前检查存在性：不存在才建，存在先 SHOW CREATE TABLE 比对；不能重复执行 CREATE 或 drop 原数据。
- [ ] 重跑 schema 测试，真实 SHOW CREATE TABLE 核对。记录版本、健康结果及未通过门槛到 dev-notes。

## Task 2：统一切块、QA 指纹与原子导入

**Files:** 新建 app/knowledge/__init__.py、chunks.py、sources.py、dedup.py、locks.py、importer.py，data/kb/returns.md、shipping.md、products.md、after_sales.md；tests/test_knowledge_chunks.py、test_knowledge_importer.py、test_qa_dedup.py。

**Interfaces:** 输出文件表中的 ChunkDraft/ImportResult/parse_markdown/qa_fingerprint/import_chunks；sources.py 的 SOURCE_TYPES 是唯一文件白名单。writer_lock(engine) 持有专用连接 GET_LOCK('mewhelp_ch03_writer',0)，失败抛 WriterBusy，finally RELEASE_LOCK。

- [ ] 写红测试，含两节、大表格、IMPORTANT、超长完整句，断言路径映射、表头复用、句界完整、前后块指针、事务回滚和同节多块保留。确定性去重断言：

```python
def test_fingerprint_uses_answer_and_normalization():
    from app.knowledge.dedup import qa_fingerprint
    assert qa_fingerprint('运费？', ' 满 99 包邮。') == qa_fingerprint('运费', '满99包邮')
    assert qa_fingerprint('运费', '满99包邮') != qa_fingerprint('运费', '满199包邮')
    assert qa_fingerprint('邮费多少', '满99包邮') != qa_fingerprint('运费', '满99包邮')
```

- [ ] 运行三个测试文件确认红。实现规范化与指纹：

```python
import hashlib, json, unicodedata
def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))
def qa_fingerprint(question, answer):
    payload = json.dumps([normalize(question), normalize(answer)], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
```

- [ ] 按标题 parser + 递归切分 + 完整句重叠实现 parse_markdown；表格先识别为整行单元，每组复制两行表头。实现 import_chunks：同事务读取已有指纹，逐块判重、flush ID、连接本来源本次新增块，commit；异常 rollback。重复文档 inserted=0；同标题两段不同正文 inserted=2。legacy FAQ 用 source=legacy-faq/{id} 只读复制。
- [ ] 加四份演示文档；运费示例“演示规则：满 99 元包邮；不足 99 元运费 8 元”，标注仅示例。所有字段超 DDL 长度报错不截断。
- [ ] 跑三个测试文件至绿，记录实际块数、表格分割与重录结果；写阶段笔记。

## Task 3：嵌入与 Milvus 全字段读模型

**Files:** 新建 embedding.py、vector_store.py、tests/test_embedding.py、test_vector_store.py。

**Interfaces:** EmbeddingClient.embed(texts:list[str])->list[list[float]]；MilvusStore.ensure_collection()、upsert(rows:list[dict])、search(vector:list[float],top_k:int)->list[dict]、count()->int。search 返回 id/score/question/answer/category。

- [ ] 红测试覆盖 index 乱序还原、错误数量/维度/NaN、缺 key、429/503/504 有限重试、Milvus 字段、检索 output_fields；不得连接真实服务。

```python
def test_vectors_reject_wrong_dimensions():
    import pytest
    from app.knowledge.embedding import validate_vectors
    with pytest.raises(ValueError):
        validate_vectors([[0.0] * 1023], expected_count=1)
```

- [ ] 跑 `uv run pytest tests/test_embedding.py tests/test_vector_store.py -q` 红。实现 OpenAI 兼容 `/embeddings` 同步客户端，model=BAAI/bge-m3、encoding_format=float、batch<=16、超时和有限退避。定义 validate_vectors(vectors, expected_count) 验证数量/长度/finite。
- [ ] 显式 Milvus schema 创建 INT64 id、FLOAT_VECTOR vector(1024)、VARCHAR question/answer/category，UTF-8 字节上限预检。参考 payload：

```python
row = {'id': chunk.id, 'vector': vector, 'question': chunk.questions,
       'answer': chunk.answer, 'category': chunk.category}
client.upsert(collection_name='knowledge', data=[row])
hits = client.search(collection_name='knowledge', data=[vector], limit=top_k,
                     output_fields=['question', 'answer', 'category'])
```

- [ ] 核对真实返回结构和索引 API 后封装，已有字段不兼容则报错。跑测试绿；依赖可用时一次真实 1024 维嵌入和 standalone 连接冒烟，记录门槛结果。

## Task 4：可恢复双写与跨平台 CLI

**Files:** 新建 indexer.py、scripts/kb.py、tests/test_knowledge_indexer.py、tests/test_kb_cli.py；修改 Makefile、README.md。

**Interfaces:** index_pending 签名如前；`uv run python -m scripts.kb preview|build|vectorize|mine|seed-conv|reset`；build 只导入 pending，vectorize 独立补齐。mine 在 Task 5 接入。Makefile 的 kb-preview/kb-build/kb-vectorize/kb-mine 仅包装同一 CLI。

- [ ] 写 fake embed/store + 临时数据库故障测试：upsert 后 hook 抛异常，MySQL 保持 pending；重跑 done、vector_id=str(id)，store 只有同一组 ID；也覆盖云端/写向量失败。

```python
def crash_after_upsert():
    raise RuntimeError('injected-after-upsert')
# 在测试已建立两条 pending 的夹具中：
# pytest.raises(RuntimeError) 包裹 index_pending(..., after_upsert=crash_after_upsert)
# 断言两条 pending；无 hook 重跑后断言 done 且 fake_store 的 ID 集合未增加。
```

- [ ] 跑 `uv run pytest tests/test_knowledge_indexer.py tests/test_kb_cli.py -q` 红。实现如下顺序，关闭读取 session 后才网络调用：

```python
texts = [row.category + '\n' + row.questions + '\n' + row.answer for row in rows]
vectors = embedder.embed(texts)
store.upsert([{'id': r.id, 'vector': v, 'question': r.questions,
               'answer': r.answer, 'category': r.category} for r, v in zip(rows, vectors)])
if after_upsert is not None:
    after_upsert()
# 新 session.begin() 内按此批 ID 更新 done/vector_id；失败不提前标记。
```

- [ ] CLI 使用 writer_lock 包裹 build/vectorize/mine，preview 只读。失败非零退出；输出 imported/skipped/indexed/pending/failed 计数。添加仅 CLI 的 `vectorize --crash-after-upsert` 测试开关，hook 用 os._exit(86) 真退出，默认关闭且不向普通作业按钮暴露。
- [ ] 跑测试至绿；外部服务可用时用新增演示块真实触发退出再运行 vectorize，按 ID 证明唯一及 done；不能靠改错密钥替代这个崩溃窗口。记录结果。

## Task 5：手动历史对话挖掘与整体文本去重

**Files:** 新建 extraction.py、extraction_prompts.py、tests/test_qa_extraction.py、tests/data/ch03_extraction_samples.json；修改 dedup.py、scripts/kb.py、tests/test_qa_dedup.py。

**Interfaces:** extract_batches(session_factory,model,batch_size=30)、deduplicate_staging(session_factory)；mine 顺序为 extract→dedup，留 pending 供独立 vectorize。

- [ ] 写红测试：工具申请/结果不进入输入、候选来源 ID 必须在本批次、重复 source/QA 不再次暂存、跨批重复只保留一条、冲突答案保留、失败 rollback 后可重跑。结构化结果：

```python
from pydantic import BaseModel, Field
class ExtractedQA(BaseModel):
    conversation_id: int
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
class QASet(BaseModel):
    items: list[ExtractedQA]
```

- [ ] 跑 `uv run pytest tests/test_qa_extraction.py tests/test_qa_dedup.py -q` 红。按实际 get_chat_model 的接口使用 with_structured_output(QASet)，调用前核对 Context7。Prompt 明确只抽稳定通用规则，不把模拟订单/物流当事实。
- [ ] 实现 source_ref=conversation/{id}、batch_no 和 extracted；重复判定用来源+QA 指纹。全部批后扫描 extracted 与已有知识指纹，首个新 QA 插入 pending/kept，重复 discarded，事务覆盖这两步。category='历史对话'、content_type='extracted_qa'、questions=真实问法。
- [ ] 用至少五条标注样例实跑抽取：通用退货、文字重复、答案冲突、私有订单、无 QA。记录模型输出分类与预期差异；这是 Prompt 验收，不能用 mock 结果宣称质量通过。跑代码测试绿并写笔记。

## Task 6：在线检索替换且保持真实工具契约

**Files:** 新建 retrieval.py、tests/test_knowledge_retrieval.py；修改 app/tools/business.py、app/core/prompts.py、tests/test_tools.py、tests/test_prompts.py。

**Interfaces:** search_knowledge 无 session_factory 参数；build_tools(session_factory,conversation_id,retriever=None) 保留前两参数，注入 retriever(keyword)->list[dict]。

- [ ] 红测试使用返回分数 .83/.72/.40 的假 store，阈值 .55 保留前两条且按排名；空 hits、依赖异常、不同 category 均覆盖。工具不能打开 session：

```python
def test_faq_contract_has_no_database_read():
    import json
    from app.tools.business import build_tools
    def forbidden_session():
        raise AssertionError('online retrieval opened MySQL')
    retriever = lambda query: [{'id': 12, 'score': .83, 'question': '运费怎么算',
                               'answer': '满99元包邮', 'category': '配送'}]
    tool = {t.name: t for t in build_tools(forbidden_session, 1, retriever=retriever)}['query_faq']
    assert json.loads(tool.invoke({'keyword': '邮费是多少'})) == {
        'keyword': '邮费是多少', 'matches': [{'question': '运费怎么算',
        'answer': '满99元包邮', 'category': '配送'}]}
```

- [ ] 跑 retrieval/tools 测试红。实现 embed(query)→store.search→阈值过滤；工具只映射三个字段，懒加载生产依赖。修改“字面关键词”提示为“知识库语义检索”，模型签名 keyword 不变。
- [ ] 跑 `uv run pytest tests/test_knowledge_retrieval.py tests/test_tools.py tests/test_prompts.py tests/test_chat_api.py -q` 绿；Ch02 邮费无命中旧断言按新语义检索替换，订单/工单/SSE 回归保留。记录阶段笔记。

## Task 7：管理 API、只读账本与白名单作业

**Files:** 新建 queries.py、app/api/kb.py、admin.py、jobs.py、app/core/jobs.py，tests/test_kb_api.py、test_jobs_api.py、test_admin_api.py；修改 app/main.py、.gitignore。

**Interfaces:** 完整路由契约采用 spec 表；JobRunner.start(name)/status(name)/stop(name)/tail(name,limit=200)，返回 job_id/name/state/exit_code/log_tail；state=running/succeeded/failed/stopped。queries 只读统计，recent/staging 有分页上限。

- [ ] 红测试 preview 前后行数不变，ingest 重录 inserted=0，同标题不同表格段都保留；非法类型/路径/空正文 400；向量化异常 502 但原文已提交、pending 可补；overview 任一依赖失败 consistent=null。
- [ ] 作业红测试未知名 404、重入 409、无 shell、固定参数、停止子树、终态日志保留；为 Windows 和 POSIX 分别 fake 进程控制，不使用真实用户 PID。命令注册形态：

```python
import sys
JOBS = {
    'kb-preview': [sys.executable, '-m', 'scripts.kb', 'preview'],
    'kb-build': [sys.executable, '-m', 'scripts.kb', 'build'],
    'kb-mine': [sys.executable, '-m', 'scripts.kb', 'mine'],
    'kb-vectorize': [sys.executable, '-m', 'scripts.kb', 'vectorize'],
    'seed-conv': [sys.executable, '-m', 'scripts.kb', 'seed-conv'],
    'kb-reset': [sys.executable, '-m', 'scripts.kb', 'reset', '--confirmed'],
    'eval-retrieval': [sys.executable, '-m', 'scripts.eval_ch03', 'retrieval'],
    'eval-mining': [sys.executable, '-m', 'scripts.eval_ch03', 'mining'],
}
# subprocess.Popen(JOBS[name], shell=False, cwd=PROJECT_ROOT, ...)
# Windows 创建隐藏进程，停止仅对 runner 持有的子进程 PID 执行固定 taskkill /PID n /T /F。
```

- [ ] 运行三个测试文件确认红。实现 API 复用 parse/import/index/search；def 路由避免同步数据库阻塞事件循环。schema 同时限制正文长度、类型、top_k、分页；非法业务输入 400，依赖失败 502/503。
- [ ] writer_lock 在 API/CLI 边界获取；POST vectorize 启动作业返回 202，不在响应前等待长任务。ingest 可同步可选向量化且返回阶段计数；部分失败异常 payload 保留 inserted/skipped/pending。作业日志限量读尾、脱敏、应用关闭清理，仅本地单 worker 模式。
- [ ] 增加 seed-conv CLI 和 Make 别名：固定 synthetic user_id（如 ch03-demo-v1）及样例 ID 清单，查询存在后只补缺少的合成会话，不更新或删除旧对话。定义 reset_knowledge(session_factory,store) 只在 writer_lock 中清 Ch03 两表（先解除本表邻接指针）和 knowledge 集合；SQL 失败/向量删除失败返回部分状态，不谎报成功。API 的 kb-reset 必须校验请求 confirmation='RESET_CH03' 后才启动带 --confirmed 的固定参数；无确认拒绝，runner 不能被别的接口绕过。tests/test_jobs_api.py 增加确认/无确认、目标表白名单及 Ch02 数据不变用例；CLI reset 缺 --confirmed 非零退出。实现功能不等于授权实际清库，本次验收不执行 destructive reset。
- [ ] 运行测试绿并核对路由清单，overview 增加 key_clause_count；写阶段笔记。

## Task 8：六区 /kb 与 /admin 页面（Vibe Coding）

**Files:** 新建 app/static/kb.html、kb.js、admin.html、admin.js、admin.css、tests/test_static.py；修改 app/static/index.html、app/main.py。

**Interfaces:** `/kb`、`/admin` FileResponse；admin.js 导出 mountAdminNav(active)，IIFE 避免旧页全局名称冲突；旧聊天路径不变。

- [ ] 按 spec 六区直接实现页面：手工编辑和预览/录入、材料清单/逐份预览/作业、暂存三态/逐条查看/seed-conv、pending/done/Milvus/关键条款数量 账本/补向量、dense 自测/Top-K/预设问句、最近入库。kb-reset 标 heavy 并弹明确确认：只清第三章知识，确认后发送 RESET_CH03。只加可达性测试，不为展示细节写镜像单测。
- [ ] 预览逐块显示字段和三种切块触发特性；错误响应也读取 JSON，明确“已入库，向量待补”。计数未知用读数失败，不显示 0。一致性文案写“计数一致”。日志按 job 名缓存在页面状态，完成刷新时恢复日志尾。

```javascript
const logCache = new Map();
function rememberJob(job) {
  logCache.set(job.name, job.log_tail || logCache.get(job.name) || '');
}
// 每次重绘日志窗口从 logCache 取值；轮询终态停止，随后刷新库存。
```

- [ ] 加聚合卡、共享导航与聊天后台入口；不链接不存在的旧模块。增加 static 路由可达性断言后运行 `uv run pytest tests/test_static.py -q`。
- [ ] 浏览器检验所有按钮、表格溢出、busy/error/loading、作业停止、终态日志保留、Milvus 离线单卡失败。截图记录在 dev-notes/generated-images/；如浏览器工具不可用明确留下该验收门槛，不能用静态测试冒充。

## Task 9：真实恢复、标注评估与完整浏览器交付

**Files:** 新建 scripts/eval_ch03.py、tests/data/ch03_retrieval_samples.json；修改 README.md、dev-notes/ch03.md、Makefile。复用 Task 5 抽取样例。

**Interfaces:** `uv run python -m scripts.eval_ch03 retrieval|mining`，逐条 JSON 输出通过/失败、ID和分数，失败非零退出；Makefile eval-retrieval/eval-mining 包装同命令。retrieval 样例以目标来源/类别及预期答案事实标注，不依赖固定自增 ID。

- [ ] 写至少五个检索标注：邮费换说法→运费、退货条件、商品 FAQ、售后流程、无关天文问题→空。预设 top_k/min_score 在真实模型上检验，调整须记录分数与理由；核心运费与无关样例必须过，总通过率至少 4/5，报告每项。
- [ ] 实现评估入口调用 search_knowledge 或 Task 5 结构化抽取；expected_source 用 Milvus 返回 ID 在离线评估阶段查 MySQL 溯源（这不属于在线检索，不改变工具契约）。写 README 的 Windows 可直接运行 uv 命令，Make 只是便利包装。
- [ ] 门槛检查：Docker、MySQL、standalone 健康，聊天与嵌入 key 存在且实调成功；只输出布尔及响应维度。缺失则继续所有独立测试，逐项报告真实验收未完成，不换选型。
- [ ] 浏览器按 Task 8 六区跑材料预览→build→vectorize→自测邮费→聊天相同问句及工具徽章；手工含两节/表格/关键条款的 Markdown 预览→录入→重录零新增→检索新块；mine→暂存三态→补向量。使用与正式数据隔离的新增演示输入，不清 Ch02。
- [ ] 对新增 pending 真执行 `uv run python -m scripts.kb vectorize --crash-after-upsert`，确认退出 86；查询此批 MySQL pending 与 Milvus IDs 后正常重跑，核对 ID 集合、向量数量、done/vector_id，无重复无遗漏。页面刷新日志与账本截图留证。
- [ ] 执行 `uv run pytest -q`、两个真实 eval、`docker compose -f compose.yaml -f compose.milvus.yaml ps`、`git diff --check`。审阅事务/锁、UTF-8 限制、在线零 SQL、进程停止和 Ch02 数据保护；修复有证据的问题后仅重跑受影响检查。记录 code review 和 finish 四字段，交付演示命令、测试/评估结果、截图和 dev-notes 路径；不得把尚有门槛的工作标为全部完成。

## 自审覆盖

Task 1 基础设施/DDL；Task 2 切块/三格嵌入来源/去重；Task 3 向量读模型；Task 4 恢复与 CLI；Task 5 手动挖取/暂存整体去重；Task 6 现有契约和零回表；Task 7 完整 API/作业；Task 8 六区/后台/浏览器；Task 9 真实恢复与评估。作业以 Python argv 执行以适配 Windows，Makefile 保留同名别名，CLI/UI 同一实现。全文不采用课程 Lite、hits、异步 Session 或 POSIX 独占启动方式。末尾 DDL 保留修订前原文。

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

