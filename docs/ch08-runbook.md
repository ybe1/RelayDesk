# 第八章工具系统运行与演示

## 启动顺序（PowerShell）

先确保 Docker Desktop 正在运行，并且 `.env` 中已有 `CHAT_API_KEY`、数据库连接配置。执行 `--live` 还需可用的 `RERANK_API_KEY`（或 `EMBED_API_KEY`）以及相应服务额度。以下服务分开在三个终端启动；不会自动打开浏览器。

终端一，MySQL：

```powershell
docker compose up -d mysql
uv run python -m scripts.migrate_ch08 --apply
uv run python -m scripts.migrate_ch08 --check
```

终端二，物流 MCP Server：

```powershell
$env:MCP_PORT = "8101"
uv run python -m mcp_servers.logistics
```

终端三，售后 MCP Server：

```powershell
$env:MCP_PORT = "8102"
uv run python -m mcp_servers.after_sales
```

终端四，客服网页后端：

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

在浏览器地址栏手动打开 `http://127.0.0.1:8000/`。客服与两个 MCP Server 是独立进程；新增 MCP 工具只需重启对应 Server。客服每轮重新发现工具。工具是否允许调用，以本地 `config/tool_policy.json` 的完整工具名权限项为准；修改权限文件会在下一次工具调用时生效。未登记默认拒绝。

## 手工验收

- 问“订单1001的物流到哪了”：轨迹徽章应显示 `query_logistics`，答复标明演示数据。实际物流来源由审计表 `tool_source=mcp, mcp_server=logistics` 证明。
- 问“我的订单1001还在保修期内吗”：应使用售后 MCP 的 `query_warranty`。
- 问“帮我建个工单”：应先追问问题描述，不能出现工单号。
- 补充“订单1001包裹显示签收但我没有收到”：回复中出现预览卡，类型和描述来自用户信息。点“取消”后不会写 tickets。
- 新建会话再提交完整问题和建单要求，点“确认提交”：只写一行 tickets，答复显示工单号。连点、刷新后重复确认不会重复创建。
- 切换/重启客服后，待处理预览可通过 `/api/chat/{conversation_id}/pending?user_id=...` 回载；提交接口仅发送用户、会话、轮次、预览ID和确认/取消决定，服务端以预览中固定参数执行。

## 审计查询

```sql
SELECT id, conversation_id, tool_call_id, tool_name, tool_source, mcp_server,
       arguments, result_summary, status, error_message, retry_count, duration_ms, created_at
FROM tool_audit_logs
ORDER BY id DESC
LIMIT 30;
```

`成功`、`失败`、`超时`、`校验拦下`、`权限拒绝` 均会记录。临时网络故障的只读工具最多重试两次；`create_ticket` 超时不会自动重试。审计表故意不设置外键，审计库临时不可用时不阻塞业务工具。

## 本地验收

```powershell
uv run pytest -q
uv run python -m scripts.migrate_ch08 --check
uv run python -m scripts.eval_ch08 --local
uv run python scripts/verify_ch08_browser.py
```

本地验收使用真实独立 MCP HTTP 进程和测试数据库；模型替身覆盖分流/调用。浏览器验证拦截全部 HTTP 请求并使用合成响应，不启动客服网页、不访问云端。截图保存在 `data/ch08/reports/ticket-preview.png`、`ticket-confirmed.png` 和 `ticket-cancelled.png`。

`uv run python -m scripts.eval_ch08 --live` 使用固定的合成订单1001问题和本地 MCP mock 结果：DeepSeek 实际选择并调用物流/售后 MCP，再生成答复；SiliconFlow 的 BGE 重排器对合成工具描述做排序诊断。不会读取或发送客户会话历史，也不会创建真实工单。运行结果写入 `data/ch08/reports/live.json`。发送到这两个外部服务的数据属于本次验收的合成样例。
