#!/usr/bin/env bash
# dispatch_all.sh — Phase 2 任务派发器
# 用法: bash dispatch_all.sh [task_number]
# 示例: bash dispatch_all.sh 2  # 派发Claude-2的任务
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TASK_NUM=${1:-2}

case $TASK_NUM in
    2) TASK_FILE="tasks/task_claude2_M326_M350.md" ;;
    3) TASK_FILE="tasks/task_claude3_M351_M375.md" ;;
    4) TASK_FILE="tasks/task_claude4_M376_M400.md" ;;
    5) TASK_FILE="tasks/task_claude5_M401_M425.md" ;;
    6) TASK_FILE="tasks/task_claude6_M426_M450.md" ;;
    *) echo "Unknown task number: $TASK_NUM (valid: 2-6)"; exit 1 ;;
esac

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE not found"; exit 1
fi

echo "=== Dispatching Phase 2 Claude-$TASK_NUM ==="
echo "Task: $TASK_FILE ($(wc -l < "$TASK_FILE") lines)"
echo ""

TASK_FILE="$TASK_FILE" bash claude_hk_chat.sh
