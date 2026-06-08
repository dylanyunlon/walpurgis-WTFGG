#!/usr/bin/env bash
# dispatch_subclaude.sh — 向子Claude发送GPU实验任务
# 使用: ./dispatch_subclaude.sh <prompt_file> [model]
# 例如: ./dispatch_subclaude.sh prompts/subclaude_gpu_experiment.md claude-opus-4-6
set -euo pipefail

PROMPT_FILE="${1:?Usage: $0 <prompt_file> [model]}"
MODEL="${2:-claude-opus-4-6}"
EFFORT="${3:-medium}"

cd "$(dirname "$0")"

# 使用claude_hk_chat.sh发送
if [ -f "claude_hk_chat.sh" ]; then
    echo "[dispatch] Sending prompt via claude_hk_chat.sh..."
    echo "[dispatch] Model: ${MODEL}, Effort: ${EFFORT}"
    echo "[dispatch] Prompt: ${PROMPT_FILE}"
    
    # 注入环境变量说明到prompt
    FULL_PROMPT=$(cat "$PROMPT_FILE")
    FULL_PROMPT="${FULL_PROMPT}

## 环境变量
请在服务器上设置:
\`\`\`bash
export GITHUB_TOKEN=\$(cat /tmp/claude-hk-config/github_token.txt 2>/dev/null || echo 'ask_user')
\`\`\`
"
    
    echo "$FULL_PROMPT" > /tmp/_dispatch_prompt.txt
    MODEL="$MODEL" EFFORT="$EFFORT" bash claude_hk_chat.sh /tmp/_dispatch_prompt.txt
else
    echo "[dispatch] claude_hk_chat.sh not found"
    echo "[dispatch] Manual dispatch: paste contents of ${PROMPT_FILE} to Opus 4.6 chat"
    echo ""
    cat "$PROMPT_FILE"
fi
