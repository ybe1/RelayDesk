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
