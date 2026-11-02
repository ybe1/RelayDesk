# AGENTS.md

适用于整个仓库。以下依据当前代码、`README.md`、`pyproject.toml` 和 `Makefile`；`docs/`、`dev-notes/` 中的早期记录仅供追溯。

## 项目结构

- `app/main.py`：FastAPI 入口，同时提供 `app/static/` 中的 HTML、CSS、JavaScript，无需单独构建或启动前端。
- `app/api/`、`app/schemas/`：HTTP/SSE 接口与请求、响应结构；聊天主链路为 `api/chat.py → workflow/service.py → workflow/graph.py`。
- `app/workflow/`、`app/context/`、`app/knowledge/`：工作流与 Agent、上下文管理、知识导入和 RAG 检索。
- `app/tooling/`、`app/tools/`、`mcp_servers/`、`config/`：工具运行时、业务工具、MCP 服务及工具策略配置。
- `app/services/`、`app/repositories/`、`app/db/`：业务服务、数据读写与 ORM；`app/evaluation/`、`app/flywheel/`、`app/observability/`：评估、知识缺口审核与追踪。
- `db/`、`sql/`、`scripts/`：建表 SQL、升级 SQL 与开发/迁移/评估脚本；`tests/`：pytest 测试；`data/kb/`：知识源；`data/runtime/`：本机运行状态。

## 开发约定

- 沿用现有模块分层和就近文件风格；配置集中在 `app/config.py`，通过环境变量或 `.env` 读取。不要提交 `.env`、密钥或本机运行产物。
- 测试放在 `tests/test_*.py`；已配置 `pytest-asyncio` 的 `asyncio_mode = "auto"`。优先运行相关测试，再按服务条件运行全量测试。
- 保留仍被使用的 `ch02`～`ch09` SQL、迁移及兼容入口，不凭章节名判断代码已废弃。旧 `knowledge` 与 `knowledge_ch04` 集合的 schema 不同，不能仅替换集合名。
- 数据库升级先运行对应迁移的 `--check`，再按 README 顺序决定是否 `--apply`；已有数据库不要重复导入初始化 SQL。
- 工作流 SQLite 检查点仅支持本机单实例；不要增加多 worker 或多副本处理同一会话。
- 未发现格式化、lint、类型检查或 CI 配置；相关工具及团队规范**待确认**，不要假定存在相应命令。

## 启动命令

在项目根目录执行。需要 Python 3.12+、uv、已启动的 Docker Desktop。保留已有 `.env`；仅在缺失时从 `.env.example` 复制，并配置聊天模型、地址、密钥及检索所需的嵌入/重排密钥。

```powershell
$env:PYTHONUTF8 = "1"  # PowerShell；CMD 中使用 set PYTHONUTF8=1
uv sync
docker compose -f compose.yaml -f compose.milvus.yaml up -d
docker compose -f compose.yaml -f compose.milvus.yaml ps
```

首次建库时，Compose 仅自动导入 `db/ch02.sql`；还需按 README 导入 `db/ch03.sql`、`db/ch04.sql`，再依次执行 `scripts.migrate_ch05`～`scripts.migrate_ch09`。确认服务健康、表结构就绪后启动：

```text
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

页面入口为 `http://127.0.0.1:8000/`，接口文档为 `/docs`。使用物流、售后工具时，各另开一个项目根目录终端：

```text
uv run python -m mcp_servers.logistics
uv run python -m mcp_servers.after_sales
```

`make dev` 调用 Bash 脚本 `scripts/dev.sh`；Windows PowerShell/CMD 优先使用以上直接命令。应用启动含后台维护与每周评估调度，可能产生云模型调用费用，详见 README。

## 测试与校验命令

```text
# 聊天接口回归：使用替身，不连接数据库或真实云模型
uv run pytest tests/test_workflow_chat_api.py tests/test_chat_api.py -q

# 全量测试；部分测试需要本机 MySQL、Milvus 和已初始化的数据表
uv run pytest -q

# 本地校验评估样本及知识源，无需云模型
uv run python -m scripts.validate_eval_ch04
```

已安装 Make 时可用 `make test`（对应 `uv run pytest -v`）。`eval-rag`、`eval-flywheel` 等评估目标会调用云模型，不能作为普通离线测试运行。

核验边界：本次起草时，Compose 配置检查与 300 条评估样本校验通过；聊天回归持续无输出后已中止，**未取得测试结果**。完整服务启动、全量测试及真实模型连通性**未验证**，不保证当前本机服务、密钥和数据已就绪。
