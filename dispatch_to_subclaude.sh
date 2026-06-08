#!/usr/bin/env bash
# ============================================================
#  dispatch_to_subclaude.sh — 非交互式向claude.hk.cn发送任务
#  使用方式: MODEL=claude-opus-4-6 ./dispatch_to_subclaude.sh "你的prompt"
#  或者: MODEL=claude-opus-4-6 ./dispatch_to_subclaude.sh --file prompt.txt
# ============================================================
set -euo pipefail

export MODEL="${MODEL:-claude-opus-4-6}"
export EFFORT="${EFFORT:-high}"
export THINKING="${THINKING:-off}"
export TIMEOUT="${TIMEOUT:-600}"

ORG="9b279708-8d27-463a-bdc8-792a764ed709"
BASE="https://claude.hk.cn/api/organizations/${ORG}"
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'

# ── 读取cookie ──
load_cookie() {
  if [ -f /tmp/claude_hk_cookie.txt ]; then
    COOKIES=$(head -1 /tmp/claude_hk_cookie.txt)
    [ -n "$COOKIES" ] && return 0
  fi
  local CFG="/tmp/claude-hk-config"
  if [ -d "$CFG" ]; then
    git -C "$CFG" pull -q 2>/dev/null || true
  else
    git clone --depth=1 -q https://github.com/dylanyunlon/claude-hk-config.git "$CFG" 2>/dev/null || true
  fi
  if [ -f "$CFG/cookie.txt" ]; then
    COOKIES=$(head -1 "$CFG/cookie.txt")
    echo "$COOKIES" > /tmp/claude_hk_cookie.txt
  elif [ -f "$CFG/raw_curl.txt" ]; then
    COOKIES=$(grep -oP "(?<=-b ')[^']*" "$CFG/raw_curl.txt" | head -1)
    [ -n "$COOKIES" ] && echo "$COOKIES" > /tmp/claude_hk_cookie.txt
  fi
  if [ -z "${COOKIES:-}" ]; then
    echo "ERROR: No cookie found"
    exit 1
  fi
}

COOKIES=""
load_cookie

# ── 读取prompt ──
if [ "${1:-}" = "--file" ] && [ -n "${2:-}" ]; then
  PROMPT=$(cat "$2")
elif [ -n "${1:-}" ]; then
  PROMPT="$1"
else
  echo "Usage: $0 'prompt text' | $0 --file prompt.txt"
  exit 1
fi

echo "═══════════════════════════════════════"
echo "  Dispatching to sub-Claude"
echo "  Model : $MODEL"
echo "  Effort: $EFFORT"
echo "  Prompt: ${#PROMPT} chars"
echo "═══════════════════════════════════════"

# ── 创建对话 ──
CONV_ID=$(curl -sf --max-time 15 "${BASE}/chat_conversations" \
  -H 'content-type: application/json' \
  -b "$COOKIES" -H "user-agent: $UA" \
  -d '{"name":"","model":"'"${MODEL}"'"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["uuid"])')

echo "Conv: $CONV_ID"

# ── 发送prompt ──
H_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')
A_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')

# 写payload（用python避免json转义问题）
python3 << PYEOF
import json

prompt = open("/tmp/_subclaude_prompt.txt").read()
payload = {
    "prompt": prompt,
    "timezone": "Asia/Shanghai",
    "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
        "nameKey":"normal_style_name","prompt":"Normal\n",
        "summary":"Default responses from Claude",
        "summaryKey":"normal_style_summary","isDefault":True}],
    "locale": "en-US", "model": "${MODEL}", "effort": "${EFFORT}",
    "thinking_mode": "${THINKING}",
    "tools": [
        {"type": "web_search_v0", "name": "web_search"},
        {"type": "artifacts_v0", "name": "artifacts"},
        {"type": "repl_v0", "name": "repl"}
    ],
    "turn_message_uuids": {"human_message_uuid": "${H_UUID}", "assistant_message_uuid": "${A_UUID}"},
    "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
    "create_conversation_params":{"name":"","model":"${MODEL}",
        "include_conversation_preferences":True,"paprika_mode":None,"compass_mode":None,
        "tool_search_mode":"auto","is_temporary":False,"enabled_imagine":True}
}
open("/tmp/_chat_payload.json","w").write(json.dumps(payload))
PYEOF

# 先写prompt到文件（避免shell转义）
echo "$PROMPT" > /tmp/_subclaude_prompt.txt

# 重新生成payload
python3 << PYEOF
import json

prompt = open("/tmp/_subclaude_prompt.txt").read()
payload = {
    "prompt": prompt,
    "timezone": "Asia/Shanghai",
    "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
        "nameKey":"normal_style_name","prompt":"Normal\n",
        "summary":"Default responses from Claude",
        "summaryKey":"normal_style_summary","isDefault":True}],
    "locale": "en-US", "model": "${MODEL}", "effort": "${EFFORT}",
    "thinking_mode": "${THINKING}",
    "tools": [
        {"type": "web_search_v0", "name": "web_search"},
        {"type": "artifacts_v0", "name": "artifacts"},
        {"type": "repl_v0", "name": "repl"}
    ],
    "turn_message_uuids": {"human_message_uuid": "${H_UUID}", "assistant_message_uuid": "${A_UUID}"},
    "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
    "create_conversation_params":{"name":"","model":"${MODEL}",
        "include_conversation_preferences":True,"paprika_mode":None,"compass_mode":None,
        "tool_search_mode":"auto","is_temporary":False,"enabled_imagine":True}
}
open("/tmp/_chat_payload.json","w").write(json.dumps(payload))
PYEOF

# ── 发送并解析SSE ──
RESPONSE=$(curl -sf --max-time "$TIMEOUT" \
  "${BASE}/chat_conversations/${CONV_ID}/completion" \
  -H 'accept: text/event-stream' -H 'content-type: application/json' \
  -H 'anthropic-client-platform: web_claude_ai' \
  -b "$COOKIES" -H 'origin: https://claude.hk.cn' \
  -H 'referer: https://claude.hk.cn/new' -H "user-agent: $UA" \
  -d @/tmp/_chat_payload.json 2>&1)

echo "$RESPONSE" | python3 << 'PYEOF'
import sys, json

text_parts = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    try: d = json.loads(line[6:])
    except: continue
    t = d.get("type", "")
    if t == "content_block_delta":
        delta = d.get("delta", {})
        if delta.get("type") == "text_delta":
            text_parts.append(delta["text"])

text = "".join(text_parts)
if text.strip():
    print("═══ Sub-Claude Response ═══")
    print(text)
    # 保存到文件
    open("/tmp/_subclaude_response.txt", "w").write(text)
    print(f"\n═══ Response saved: /tmp/_subclaude_response.txt ({len(text)} chars) ═══")
else:
    print("WARNING: Empty response from sub-Claude")
    # 保存raw用于debug
    open("/tmp/_subclaude_raw.txt", "w").write(sys.stdin.read() if hasattr(sys.stdin, 'read') else "")
PYEOF

echo ""
echo "Conv URL: https://claude.hk.cn/chat/${CONV_ID}"
