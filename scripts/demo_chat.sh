#!/usr/bin/env bash
# 验收:两轮流式对话,第二轮须接住第一轮上下文
set -euo pipefail
USER_ID="demo-$$"
TMP1=$(mktemp)
TMP2=$(mktemp)
trap 'rm -f "$TMP1" "$TMP2"' EXIT

# Windows 中文 GBK 下,curl -d 内联中文会乱码(参数经 UTF-16 转码);写 UTF-8 临时文件再 -d @file 规避
printf '{"user_id": "%s", "message": "我叫王小明,昨天买了你们的智能猫砂盆"}' "$USER_ID" > "$TMP1"

echo "=== 第一轮:自报家门+提商品 ==="
FIRST=$(curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' -d @"$TMP1")
printf '%s\n' "$FIRST"
CONVERSATION_ID=$(printf '%s\n' "$FIRST" | sed -n 's/.*"conversation_id": \([0-9][0-9]*\).*/\1/p' | head -n 1)
if [ -z "$CONVERSATION_ID" ]; then
  echo '第一轮未返回 conversation_id' >&2
  exit 1
fi
printf '{"user_id": "%s", "conversation_id": %s, "message": "还记得我叫什么、买了什么吗?"}' "$USER_ID" "$CONVERSATION_ID" > "$TMP2"
echo; echo "=== 第二轮:考上下文(我叫什么?买了什么?) ==="
curl -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' -d @"$TMP2"
echo
