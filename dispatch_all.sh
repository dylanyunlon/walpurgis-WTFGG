#!/usr/bin/env bash
# dispatch_all.sh — 批量派发Claude-7/8/9任务
# 用法: bash dispatch_all.sh
# 前提: .claude-hk-config/raw_curl.txt 中cookie有效
set -uo pipefail
cd "$(dirname "$0")"

# 同步cookie
cd .claude-hk-config && git pull -q 2>/dev/null; cd ..

dispatch_one() {
    local NAME="$1"
    local TASK_FILE="$2"
    
    echo "=== Dispatching: $NAME ==="
    TASK_FILE="$TASK_FILE" bash claude_hk_chat.sh < /dev/null
    echo ""
}

# Claude-7: TeX引用 (用web search)
dispatch_one "Claude-7 TeX refs" "tasks/task_claude7_M151_M175.md"

# Claude-8: 消融实验
dispatch_one "Claude-8 ablation" "tasks/task_claude8_M176_M200.md"

echo "=== All dispatched ==="
