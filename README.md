# MewHelp — 电商智能客服

FastAPI + LangChain,模型上游直连(DeepSeek 官方,OpenAI 协议)。

## Ch04：并行混合检索索引

运行 `uv run python -m scripts.kb ch04-rebuild`（或 `make kb-ch04-rebuild`）可在单写者锁内读取全部 `knowledge_chunks`（含 pending），按原 SQL ID 幂等写入独立的 `knowledge_ch04`。新集合使用 1024 维 COSINE dense 索引、中文 analyzer 与服务端 BM25 sparse 索引。输出包含 `indexed`、`sql_count`、`milvus_count`、`ids_match`、`failed`；只有数量与 ID 集合一致才成功。中断后重新运行同一命令即可重复写入同一 ID。

该命令保留 Ch03 的 `vectorize_status`、`vector_id` 和旧 `knowledge` 集合，在线默认集合仍为 `knowledge`。现有 Ch04 schema 或索引不兼容时会拒绝运行；远端多余 ID 会导致 parity 检查失败，不会自动删除数据。文本字段在嵌入前和 upsert 前按 UTF-8 字节上限校验，错误会带 SQL chunk ID。

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
