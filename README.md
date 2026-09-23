# MewHelp — 电商智能客服

FastAPI + LangChain,模型上游直连(DeepSeek 官方,OpenAI 协议)。

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
