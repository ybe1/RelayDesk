# MewHelp — 电商智能客服

FastAPI + LangChain,模型上游直连(DeepSeek 官方,OpenAI 协议)。

## Ch04：并行混合检索索引

运行 `uv run python -m scripts.kb ch04-rebuild`（或 `make kb-ch04-rebuild`）可在单写者锁内读取全部 `knowledge_chunks`（含 pending），按原 SQL ID 幂等写入独立的 `knowledge_ch04`。新集合使用 1024 维 COSINE dense 索引、中文 analyzer 与服务端 BM25 sparse 索引。输出包含 `indexed`、`sql_count`、`milvus_count`、`ids_match`、`failed`；只有数量与 ID 集合一致才成功。中断后重新运行同一命令即可重复写入同一 ID。

该命令保留 Ch03 的 `vectorize_status`、`vector_id` 和旧 `knowledge` 集合。Task 14 已验证 MySQL 与 `knowledge_ch04` 均为 35 条且 ID 完全一致，在线默认 `MILVUS_COLLECTION=knowledge_ch04`；现有 `.env` 若显式设置了旧值，需要改为新值并重启应用。生产 FAQ 检索按该配置选择集合；Ch03 同步知识库搜索仍保留原接口与旧集合。现有 Ch04 schema 或索引不兼容时会拒绝重建；远端多余 ID 会导致 parity 检查失败，不会自动删除数据。文本字段在嵌入前和 upsert 前按 UTF-8 字节上限校验，错误会带 SQL chunk ID。

旧 `knowledge` 集合保留 6 条用于恢复旧版本。回滚需要恢复上一稳定的 Ch03 应用版本，将集合配置改回 `knowledge`，然后重启；旧集合没有 BM25/dense 字段，不能只改配置让当前 Ch04 混合检索兼容旧 schema。不要删除旧集合。

切换到 `knowledge_ch04` 后，旧 `kb-reset` 管理入口和 `scripts.kb reset --confirmed` 都会明确拒绝执行，保护共享 SQL 原文和两个集合。本章未提供 Ch04 完整重置动作；需要另行设计完整维护流程。

## Ch04：演示与验收

先确认 Docker Desktop 运行、MySQL 与 Milvus 服务健康，`.env` 已配置 `CHAT_API_KEY`、`EMBED_API_KEY`、独立的 `RERANK_API_KEY` 和 `MILVUS_COLLECTION=knowledge_ch04`。启动已有服务和应用：

```powershell
& "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe" compose -f compose.yaml -f compose.milvus.yaml up -d
uv run uvicorn app.main:app --port 8000
```

打开 [聊天页](http://localhost:8000/)，询问“平台邮费规则是什么？”，点击回答中的引用查看来源章节和全文，再点击一次反馈，确认两按钮锁定并显示“已反馈”。询问“平台是否给火星国际空间站配送猫粮，具体规则是什么？”应明确拒答，并在低置信度问题池留下原问题及原因。聊天会调用已配置的云服务；用户已明确授权将仓库演示知识、检索证据和评估问题发送到 DeepSeek。

打开 [RAG 评估页](http://localhost:8000/rag-eval) 查看已保存报告、策略和分桶指标、编造台账。2026-09-28 完整报告已完成四策略检索 1200/1200、生成 1200/1200 和裁判 240/240；Faithfulness 为 98.75%。没有人工处置案例时，judge-check 一致率仍不可报告为通过。

以下检查不调用云模型，Windows 没有 Make 时直接使用等价 Python 命令：

```powershell
uv run pytest -q
uv run python -m scripts.validate_eval_ch04  # make eval-check
uv run python -m scripts.judge_check --inspect-only
# inspect-only 为明确阻塞状态，退出码 2；不会调用裁判。
```

纯 BM25 型号演示只连接本机 Milvus，不调用 Query 理解、嵌入或重排：

```powershell
@'
import json
from app.config import settings
from app.knowledge.hybrid_store import HybridMilvusStore
store = HybridMilvusStore(uri=settings.milvus_uri, collection_name=settings.milvus_collection)
try:
    hits = store.bm25_search("MH-V10 Pro 额定功率", limit=5)
    print(json.dumps(hits, ensure_ascii=False, indent=2))
finally:
    store.client.close()
'@ | uv run python -
```

完整评估需要将仓库 300 题及演示检索证据发送至已配置的云服务。取得对应目的地授权后，现有断点用以下命令续跑；它保留已成功答案和检索记录，待全部生成成功后再执行裁判：

```powershell
uv run python -m scripts.eval_ch04 --phase generation --retry-skipped --concurrency 2 --timeout 90
# 继续已有断点（不重试失败项）：uv run python -m scripts.eval_ch04（make eval-rag）
# 重试失败并继续断点：uv run python -m scripts.eval_ch04 --retry-skipped
# 新一轮完整评估：uv run python -m scripts.eval_ch04 --fresh
# 有人工处置且含证据快照的案例后：uv run python -m scripts.judge_check（make judge-check）
```

RAG 评估页提供「继续已有断点」「重试失败项」「新一轮完整评估」三种方式，分别映射服务端固定的 `resume`、`retry`、`fresh` 模式。三种方式都需要 `RUN_EVAL_RAG` 确认，不能传任意命令参数。新一轮会重新执行所有项目并替换当前报告；继续断点遇到已经完成的报告时不会再调用模型。

同一报告路径在整个评估运行期间持有跨进程文件锁；其他 CLI 或服务进程会立即报告 `Evaluation busy`（退出码 3），待当前运行退出后再试。旁边的 `.lock` 文件保留，进程退出会释放操作系统锁，无需手动删除文件。

报告位于 `data/ch04/reports/rag_eval.json` 和 `rag_eval.txt`，开发记录位于 `dev-notes/ch04.md`。完整云评估、后端/数据评审和人工个案 `judge-check` 已通过；交付前按上述步骤完成真实浏览器的引用弹层与一次性反馈交互确认。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY
    ./scripts/dev.sh       # :8000 应用

## Ch02：聊天页工具链

先启动 Docker Desktop。Windows PowerShell 如果暂时找不到 `docker` 命令，可用安装程序的完整路径：

```powershell
& "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe" compose up -d
uv run python -m db.seed_ch02
uv run uvicorn app.main:app --port 8000
```

浏览器打开 `http://localhost:8000/`，依次问“订单 1001 的物流到哪了”“退货政策是什么”“邮费是多少”。订单、商品、物流的返回值是随机演示数据；第三句按本章的 FAQ 字面关键词检索应查不到，这是预期漏召回。

另开终端运行真实模型的三条标注样例：

```powershell
uv run python -m scripts.eval_ch02
uv run pytest -q
```

`/api/chat` 请求体使用 `user_id`、可选的数字 `conversation_id` 和 `message`。首次 SSE 返回 `conversation` 帧，工具执行时返回 `tool_status` 帧，最终回答仍以 `delta` 帧流式返回。旧版 `session_id` 请求格式已被替换。

## 验收

    # 1. 流式对话
    curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"user_id": "demo-1", "message": "你们卖猫粮吗?"}'

    # 2. 两轮上下文
    ./scripts/demo_chat.sh

    # 3. 结构化提取
    curl -s http://localhost:8000/api/extract -H 'Content-Type: application/json' \
      -d '{"text": "订单 MH20260701123 的猫爬架散架了,我要退款"}'

## 测试

    uv run pytest -v              # 单测(不打真实模型)
    uv run python scripts/eval_extract.py   # 标注样例评估(需服务运行中)

## Ch03：知识库离线命令

知识原文先写入 MySQL 的 `knowledge_chunks`，再由独立向量化命令按同一 ID 写入 Milvus。这样，Milvus 写入后进程中断时，原文仍保持 `pending`；再次执行向量化会对相同 ID 做幂等 upsert，随后才回填 `done`。

```powershell
uv run python -m scripts.kb preview
uv run python -m scripts.kb build
uv run python -m scripts.kb vectorize
uv run python -m scripts.kb mine
```

`build` 只导入白名单 Markdown，不会调用嵌入服务；`vectorize` 只补齐 `pending` 行。每个命令都会输出 `imported`、`skipped`、`indexed`、`pending`、`failed` 计数。写入命令会取得单写者锁，`preview` 是只读操作。`mine` 将在后续 Task 5 接入历史对话挖掘。

在已启动 MySQL、Milvus 且设置 `EMBED_API_KEY` 的开发环境，可用 `uv run python -m scripts.kb vectorize --crash-after-upsert` 做恢复窗口演示：它会在首次 Milvus upsert 后以退出码 86 立即退出。随后普通 `vectorize` 应按相同 ID 完成回填。该开关只用于 CLI 演示，不属于普通作业入口。

## Ch03：标注评估与真实验收

以下命令可直接在 Windows PowerShell 运行；Make 目标只提供相同的快捷包装。检索评测逐条输出 Milvus 命中 ID、相似度分数、类别和经 MySQL 溯源的来源文件，并检查 `query_faq` 的 `keyword/matches` 契约。挖掘评测会对 `tests/data/ch03_extraction_samples.json` 中的标注会话实际调用结构化模型。

```powershell
uv run python -m scripts.eval_ch03 retrieval
uv run python -m scripts.eval_ch03 mining
# 或：make eval-retrieval / make eval-mining
```

两个评测均先检查 Docker Compose、MySQL、Milvus 和所需密钥；门槛不满足时输出 JSON 报告并以退出码 2 结束，不会把离线测试当作真实验收。检索样本位于 `tests/data/ch03_retrieval_samples.json`，报告 schema 为 `ch03-eval/v1`：包含 `gates`、逐条 `results` 和 `summary`。检索要求至少 4/5 通过，并且“邮费是多少”和无关天文问题必须通过。

## Ch05：Workflow 与多步 Agent

当前 `/api/chat` 已升级为本章Workflow。知识问题强制检索并先过置信度闸，物流/订单/售后直接进入Agent；投诉展示两个独立选项，明确问候不调用模型。订单、商品、物流仍为随机演示数据。

在 VS Code 的 PowerShell 终端、项目根目录运行：

```powershell
& "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe" compose -f compose.yaml -f compose.milvus.yaml up -d
uv sync
uv run python -m scripts.migrate_ch05 --check
# 首次有PENDING时执行；不清除历史数据
uv run python -m scripts.migrate_ch05 --apply
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000/，依次问：

- `退货政策是什么`：终端日志含 retrieval → confidence_gate → decision → final_answer。
- `订单1001的物流到哪了`：Agent调用query_logistics，回答标明演示数据。
- `我要投诉`：独立的转人工/建工单按钮；转人工只模拟问候，工单需编辑并确认才写库。取消或忽略不写工单。
- `你好`：固定话术、model_calls=0。
- `请先查订单1001的状态，再根据订单结果查物流，两步都需要`：两项工具徽章和多步节点日志。

另开终端运行：

```powershell
uv run python -m scripts.demo_agent_loop --offline
uv run python -m scripts.eval_ch05 --offline
uv run python -m scripts.eval_ch05 --live
uv run python -m scripts.eval_ch05_intent --offline
uv run pytest -q
```

`--offline`仍需要本机MySQL/Milvus容器中的MySQL和官方SQLite持久化；聊天模型与检索在该脚本中替换成标明的模拟结果，不发云端请求。`--live`仅跑本章5条演示问题，调用现有CHAT/EMBED/RERANK上游，不运行第四章300题。报告为 `data/ch05/reports/acceptance-offline.json`、`acceptance-live.json`；意图与阈值标注结果位于 `data/ch05/intent_report.json` 和 `gate_report.json`，浏览器交互记录为 `data/ch05/reports/browser.json`。

预算默认每轮5工具执行、6模型调用、12,000累计输入/输出token、120秒；包括意图和检索Query理解。嵌入、重排另记次数和耗时。模型未返回usage时保守估算并标estimated；达到预算用固定话术停止，不额外调用模型总结。配置见 `.env.example` 的 `AGENT_*`、`WORKFLOW_*`。

检查点默认 `data/runtime/ch05-checkpoints.sqlite`，仅支持本机单实例，不能使用多worker/多副本共同处理同一会话。MySQL为聊天/工单账本，SQLite为图状态；跨库不承诺原子提交。断线后停止本轮并闭合已落账工具申请，下一消息开始新轮，不自动重放写动作或续传旧token。匿名user_id沿用现有演示身份约定，不代表已完成生产登录鉴权。

开发过程：`dev-notes/ch05.md`；设计与计划：`docs/superpowers/specs/2026-10-02-ch05-workflow-agent-design.md`、`docs/superpowers/plans/2026-10-02-ch05-workflow-agent.md`。

最终回归：548项通过；真实五题验收5/5、意图标注28/28、浏览器交互7/7。工具重试每次实际调用计入5次执行上限；断连清理等待已开始的数据库写入完成，随后闭合工具申请并释放会话占用。

本章置信度闸仅按小样本校准的重排分数判断。“平台提供火星移民签证吗”实测在闸前拒绝并入池，见`data/ch05/reports/low-confidence-live.json`；“火星移民签证的退款周期是多少”可能因退款词误放行，后续自评虽入池但受预算停止，见`gate-limitation-live.json`。阈值不能替代正式语义充分性检查。另有一个延后的小项：`tool_status`运行与完成帧使用不同字段命名，当前聊天页兼容两者，严格使用旧契约的其他客户端需要适配。

## Ch09：可观测性与数据飞轮

先启动业务基础设施和本机 Langfuse，再检查数据库迁移：

```powershell
docker compose -f compose.yaml -f compose.milvus.yaml up -d
docker compose -f compose.langfuse.yaml up -d
uv run python -m scripts.migrate_ch09 --check
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器入口：

- `http://127.0.0.1:8000/review`：审核标准化后的知识缺口，可查看归并原话和当轮召回证据。
- `http://127.0.0.1:8000/observability`：查看最近七天按意图 token/成本和评估趋势。
- `http://127.0.0.1:3000`：本机自托管 Langfuse trace 树。

本地验收不会调用云端模型：

```powershell
uv run pytest -q
uv run python -m scripts.eval_ch09_acceptance --mode local --output data/ch09/reports/acceptance-local.json
uv run python -m scripts.verify_ch09_browser --base-url http://127.0.0.1:8000 --output data/ch09/reports/browser.json
```

固定任务命令：

```powershell
uv run python -m scripts.process_flywheel
uv run python -m scripts.recover_ch09_index
uv run python -m scripts.eval_ch09
```

`eval_ch09` 是完整付费评估，失败后不会自动重试；确认上游余额和数据发送授权后再手动运行。调度器按 Asia/Shanghai 每周一 02:00 启动，首次启用只登记下一时点，停机错过时最多补最近一轮。开发记录见 `dev-notes/ch09.md`。
