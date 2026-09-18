# MewHelp — 电商智能客服(Ch01:纯对话)

FastAPI + LangChain,模型上游直连(DeepSeek 官方,OpenAI 协议)。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY
    ./scripts/dev.sh       # :8000 应用

## 验收

    # 1. 流式对话
    curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"session_id": "s1", "message": "你们卖猫粮吗?"}'

    # 2. 两轮上下文
    ./scripts/demo_chat.sh

    # 3. 结构化提取
    curl -s http://localhost:8000/api/extract -H 'Content-Type: application/json' \
      -d '{"text": "订单 MH20260701123 的猫爬架散架了,我要退款"}'

## 测试

    uv run pytest -v              # 单测(不打真实模型)
    uv run python scripts/eval_extract.py   # 标注样例评估(需服务运行中)
