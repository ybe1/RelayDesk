#!/usr/bin/env bash
# 单入口拉起 FastAPI 应用(:8000)
set -euo pipefail
cd "$(dirname "$0")/.."

# 中文 Windows 默认 GBK,读仓库里带中文的 markdown / jsonl 会抛 UnicodeDecodeError。
# 必须在 Python 启动前生效,写进 .env 太晚。
export PYTHONUTF8=1

if [ ! -f .env ]; then
  echo "缺少 .env,请 cp .env.example .env 并填 CHAT_API_KEY" >&2
  exit 1
fi
set -a; source .env; set +a

uv run uvicorn app.main:app --port 8000
