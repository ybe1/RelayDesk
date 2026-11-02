# RelayDesk — 电商智能客服

当前应用使用 Workflow + 多步 Agent 处理聊天：先理解问题与识别意图，再检索知识或调用工具，最后流式返回回答。网页由 Python 后端直接提供，无需单独启动前端。

## Windows 启动

需要 Python 3.12+、uv 和 Docker Desktop。先打开 Docker Desktop 并等待启动完成。

以下以仓库目录名 `RelayDesk` 为例，在其上级目录打开 CMD 后执行：

```cmd
cd /d "RelayDesk"
set PYTHONUTF8=1
uv sync
```

已有 `.env` 时保留原配置。首次使用可以复制 `.env.example` 为 `.env`，填写 `CHAT_API_KEY`、`EMBED_API_KEY` 和 `RERANK_API_KEY`。默认检索集合为 `knowledge_ch04`。

启动 MySQL 和 Milvus：

```cmd
docker compose -f compose.yaml -f compose.milvus.yaml up -d
docker compose -f compose.yaml -f compose.milvus.yaml ps
```

等服务健康后启动应用：

```cmd
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

PowerShell 中进入项目目录用 `cd "RelayDesk"`，设置编码用 `$env:PYTHONUTF8 = "1"`；其余命令相同。运行中的应用终端需要保持打开，按 Ctrl+C 停止。

物流、售后 MCP 工具分别在另外两个终端中启动，终端同样需要进入项目根目录：

```cmd
uv run python -m mcp_servers.logistics
```

```cmd
uv run python -m mcp_servers.after_sales
```

订单、商品、物流包含模拟演示数据，不代表真实电商平台的业务状态。

## 页面入口

| 页面 | 地址 |
| --- | --- |
| 客服聊天 | http://127.0.0.1:8000/ |
| 知识库管理 | http://127.0.0.1:8000/kb |
| 管理首页 | http://127.0.0.1:8000/admin |
| RAG 评估与编造个案复核 | http://127.0.0.1:8000/rag-eval |
| 知识缺口审核 | http://127.0.0.1:8000/review |
| 用量与评估趋势 | http://127.0.0.1:8000/observability |
| 接口文档 | http://127.0.0.1:8000/docs |

先发送“你好”检查基础聊天，再问“退货政策是什么”验证知识检索。知识问答会访问配置的云模型服务。创建工单需要用户确认预览后才写库。

应用启动后还会运行知识缺口整理、索引恢复等后台维护任务，以及下述每周评估调度；其中调用云模型的任务可能产生费用。

## 数据库初始化与升级

现有开发数据库通常已经初始化，先使用检查命令：

```cmd
uv run python -m scripts.migrate_ch05 --check
uv run python -m scripts.migrate_ch06 --check
uv run python -m scripts.migrate_ch07 --check
uv run python -m scripts.migrate_ch08 --check
uv run python -m scripts.migrate_ch09 --check
```

出现 PENDING 时，将对应命令的 `--check` 改为 `--apply`，按顺序执行。这些章节名是现行数据库迁移的名称，仍然需要保留。

全新数据库由 Compose 挂载 `db/ch02.sql` 创建基础表；还需要按顺序导入 `db/ch03.sql`、`db/ch04.sql`，再执行上述迁移。它们不是可重复导入脚本，不要在已有数据库上直接重新导入。

## 代码从哪里读

```text
app/main.py
  → app/api/chat.py
  → app/workflow/service.py
  → app/workflow/graph.py
  → app/knowledge/retrieval.py 或 app/workflow/agent.py
```

| 位置 | 职责 |
| --- | --- |
| `app/config.py` | 读取 `.env` 和环境变量 |
| `app/static/` | 网页和交互 |
| `app/workflow/` | 理解问题、意图分流、售后处理、Agent 和流程状态 |
| `app/services/chat.py` | 数据库聊天存储与仍在使用的历史裁剪，不再包含旧聊天实现 |
| `app/core/prompts.py` | `/api/extract` 使用的结构化提取提示词 |
| `app/context/` | 当前上下文分层、长度预算和历史整理 |
| `app/knowledge/` | 知识导入、嵌入、检索与重排 |
| `app/tooling/`、`app/tools/`、`mcp_servers/` | 工具定义、权限、审计与 MCP 服务 |
| `app/db/`、`app/repositories/` | 数据表和数据读写 |
| `app/evaluation/` | 指标、忠实度裁判与评估调度 |
| `app/flywheel/` | 知识缺口收集、归并和审核 |
| `app/observability/` | 调用追踪与用量统计 |

MySQL 保存业务记录；Milvus 保存检索索引；SQLite 保存工作流检查点。检查点只支持本机单实例，不要使用多 worker 或多副本并发处理同一会话。

`docs/` 和 `dev-notes/` 中的早期设计与验收记录用于追溯，启动方法以本 README 为准。

## 知识库维护

```cmd
uv run python -m scripts.kb preview
uv run python -m scripts.kb build
uv run python -m scripts.kb ch04-rebuild
```

`build` 把白名单 Markdown 导入 MySQL；`ch04-rebuild` 调用嵌入服务，将全部知识按 SQL ID 写入 `knowledge_ch04`，校验数量和 ID 一致性，保留旧集合。

以下早期模块仍由现有知识库管理功能使用，保留：

- `app/knowledge/vector_store.py`、`indexer.py`、`search_knowledge`：知识库管理页的旧集合搜索和向量化。
- `scripts.kb vectorize`、`mine`、`seed-conv`：管理页使用的向量化、对话挖掘和演示数据命令。
- `scripts.eval_ch03`：旧集合的检索与挖掘评估。
- `db/ch02.sql` 和后续 SQL、迁移脚本：创建或升级当前应用所需的数据表。

旧 `knowledge` 集合与 `knowledge_ch04` schema 不同，不能只改集合名互相替代。`knowledge_ch04` 配置下，旧 `kb-reset` 入口会拒绝执行，保护共享 SQL 原文与现有集合。

## 验证与评估

自动化测试不调用真实云模型，但部分集成测试需要本机 MySQL、Milvus 和正确的数据表：

```cmd
uv run pytest -q
uv run python -m scripts.validate_eval_ch04
uv run python -m scripts.eval_ch09_acceptance --mode local --output data/ch09/reports/acceptance-local.json
```

仅验证当前聊天接口且不连接数据库：

```cmd
uv run pytest tests/test_workflow_chat_api.py tests/test_chat_api.py -q
```

`/rag-eval` 展示已保存的评估报告和编造个案。忠实度裁判比较问题、答案和当轮证据，不是对每条在线回答自动执行。付费模型评估命令包括 `scripts.eval_ch04` 与 `scripts.eval_ch09`，运行前确认费用和数据发送范围。

应用启动时也会启用自动评估调度：按 Asia/Shanghai 时区，每周一 02:00 运行 `eval-flywheel`（`scripts.eval_ch09`），会将评估问题和检索证据发送给配置的云模型服务。首次启用只登记下一次运行时点；已有调度状态且停机错过运行时，最多补最近一轮。失败后不会自动重试付费评估，需要人工检查后恢复。即使不手动执行评估命令，保持应用运行也可能触发这些付费调用。

评估报告由跨进程文件锁保护。提示 Evaluation busy 时等待已有评估退出，不需要手动删除 `.lock` 文件。

## 可选：Langfuse

```cmd
docker compose -f compose.langfuse.yaml up -d
```

打开 http://127.0.0.1:3000 创建项目密钥，在 `.env` 中填写 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`，设置 `LANGFUSE_ENABLED=true` 后重启应用。默认关闭。
