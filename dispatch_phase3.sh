#!/bin/bash
# dispatch_phase3.sh — 串行调度子Claude执行GPU实验
# 每次只派一个Claude, 完成后再派下一个 (避免cookie冲突)
# Usage: bash dispatch_phase3.sh [task_number]
#   task_number: 1-5 (第2-6位Claude的任务)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/claude_hk_chat.sh" --source-only 2>/dev/null || true

# 加载cookie
if [ -f /tmp/claude_hk_cookie.txt ]; then
    COOKIES=$(head -1 /tmp/claude_hk_cookie.txt)
else
    echo "ERROR: /tmp/claude_hk_cookie.txt not found"
    exit 1
fi

ORG="9b279708-8d27-463a-bdc8-792a764ed709"
BASE="https://claude.hk.cn/api/organizations/${ORG}"
MODEL="claude-opus-4-6"
EFFORT="medium"
TIMEOUT=600
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'

TASK=${1:-1}

# ── Prompts: 精简,像第一轮那样只给clone ──
PROMPT_1='你是Walpurgis项目的实验Claude。

第一步: clone仓库并读取任务计划
```
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
cat MULTI_CLAUDE_PLAN.md
cat subclaude_phase3_prompt.md
```

你的任务编号: M110-M115
核心任务: 在GPU服务器上运行cascade变体的METR-LA实验
服务器有H100+A6000, conda环境名walpurgis

规则: 不开新分支, 不加v2/port后缀, git author: dylanyunlon <dogechat@163.com>
被截断就等我发Continue。'

PROMPT_2='你是Walpurgis项目的实验Claude。

第一步: clone仓库并读取任务计划
```
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
cat MULTI_CLAUDE_PLAN.md
cat subclaude_phase3_prompt.md
```

你的任务编号: M116-M121
核心任务: 在GPU服务器上运行nebula和prism变体的METR-LA+PEMS-BAY实验
服务器有H100+A6000, conda环境名walpurgis

规则: 不开新分支, 不加v2/port后缀, git author: dylanyunlon <dogechat@163.com>
被截断就等我发Continue。'

PROMPT_3='你是Walpurgis项目的实验Claude。

第一步: clone仓库并读取任务计划
```
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
cat MULTI_CLAUDE_PLAN.md
cat subclaude_phase3_prompt.md
```

你的任务编号: M122-M127
核心任务: 在GPU服务器上运行flux和reverie变体 + upstream D2STGNN baseline验证
服务器有H100+A6000, conda环境名walpurgis

规则: 不开新分支, 不加v2/port后缀, git author: dylanyunlon <dogechat@163.com>
被截断就等我发Continue。'

PROMPT_4='你是Walpurgis项目的实验Claude。

第一步: clone仓库并读取任务计划
```
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
cat MULTI_CLAUDE_PLAN.md
cat subclaude_phase3_prompt.md
```

你的任务编号: M128-M133
核心任务: 汇总所有实验结果, 生成comparison_table.tex, 确定最优变体并微调超参
git pull获取前面Claude的实验数据

规则: 不开新分支, 不加v2/port后缀, git author: dylanyunlon <dogechat@163.com>
被截断就等我发Continue。'

PROMPT_5='你是Walpurgis项目的实验Claude。

第一步: clone仓库并读取任务计划
```
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
cat MULTI_CLAUDE_PLAN.md
cat subclaude_phase3_prompt.md
```

你的任务编号: M134-M139
核心任务: 验证实验可复现, 更新论文tex Section 5, 仓库最终清理
git pull获取所有实验数据

规则: 不开新分支, 不加v2/port后缀, git author: dylanyunlon <dogechat@163.com>
被截断就等我发Continue。'

# 选择prompt
case $TASK in
    1) PROMPT="$PROMPT_1" ;;
    2) PROMPT="$PROMPT_2" ;;
    3) PROMPT="$PROMPT_3" ;;
    4) PROMPT="$PROMPT_4" ;;
    5) PROMPT="$PROMPT_5" ;;
    *) echo "Invalid task number: $TASK (1-5)"; exit 1 ;;
esac

echo "============================================"
echo "  Dispatching Task $TASK to Opus 4.6 medium"
echo "  Model: $MODEL | Effort: $EFFORT"
echo "============================================"

# 创建新对话
CONV_ID=$(curl -sf --max-time 15 "${BASE}/chat_conversations" \
    -X POST -H 'content-type: application/json' -b "$COOKIES" \
    -H 'origin: https://claude.hk.cn' -H "user-agent: $UA" \
    --data-raw '{"name":"","project_uuid":null,"model":null}' \
    | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["uuid"])')

echo "新对话: $CONV_ID"
echo "链接: https://claude.hk.cn/chat/$CONV_ID"

# 发送prompt
H_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')
A_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')

python3 -c "
import json
payload = {
    'prompt': '''$PROMPT''',
    'timezone': 'Asia/Shanghai',
    'personalized_styles': [{'type':'default','key':'Default','name':'Normal',
        'nameKey':'normal_style_name','prompt':'Normal\n',
        'summary':'Default responses from Claude',
        'summaryKey':'normal_style_summary','isDefault':True}],
    'locale': 'en-US', 'model': '$MODEL', 'effort': '$EFFORT',
    'thinking_mode': 'off',
    'tools': [
        {'type': 'web_search_v0', 'name': 'web_search'},
        {'type': 'artifacts_v0', 'name': 'artifacts'},
        {'type': 'repl_v0', 'name': 'repl'}
    ],
    'turn_message_uuids': {'human_message_uuid': '$H_UUID', 'assistant_message_uuid': '$A_UUID'},
    'attachments':[],'files':[],'sync_sources':[],'rendering_mode':'messages',
    'create_conversation_params':{'name':'','model':'$MODEL',
        'include_conversation_preferences':True,'paprika_mode':None,'compass_mode':None,
        'tool_search_mode':'auto','is_temporary':False,'enabled_imagine':True}
}
open('/tmp/_dispatch_payload.json','w').write(json.dumps(payload))
"

RESPONSE=$(curl -sf --max-time "$TIMEOUT" \
    "${BASE}/chat_conversations/${CONV_ID}/completion" \
    -H 'accept: text/event-stream' -H 'content-type: application/json' \
    -H 'anthropic-client-platform: web_claude_ai' \
    -b "$COOKIES" -H 'origin: https://claude.hk.cn' \
    -H 'referer: https://claude.hk.cn/new' -H "user-agent: $UA" \
    -d @/tmp/_dispatch_payload.json 2>&1)

# 提取回复
echo "$RESPONSE" | python3 -c "
import sys, json
text = []
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: '):
        try:
            d = json.loads(line[6:])
            if d.get('type') == 'content_block_delta':
                delta = d.get('delta', {})
                if delta.get('type') == 'text_delta':
                    text.append(delta.get('text', ''))
        except: pass
print(''.join(text)[:2000])
" 2>/dev/null || echo "[Response parsing failed, check conversation at: https://claude.hk.cn/chat/$CONV_ID]"

echo ""
echo "============================================"
echo "  Task $TASK dispatched."
echo "  对话: https://claude.hk.cn/chat/$CONV_ID"
echo "  如需Continue: MODEL=$MODEL EFFORT=$EFFORT bash claude_hk_chat.sh $CONV_ID"
echo "============================================"
