#!/usr/bin/env python3
"""
dispatch_migrate.py — 向子Claude派发commit迁移任务
用法: python3 dispatch_migrate.py <batch_num>  (0-24)
或:   python3 dispatch_migrate.py all           (派发全部)
"""
import os, sys, re, json, time, uuid
import urllib.request, urllib.error

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)

# ── 读取cookie ──
config_dir = os.path.join(REPO_DIR, ".claude-hk-config")
if not os.path.isdir(config_dir):
    os.system(f"git clone https://github.com/dylanyunlon/claude-hk-config.git {config_dir}")

with open(os.path.join(config_dir, "raw_curl.txt")) as f:
    raw = f.read()

COOKIE = re.search(r"-b '([^']+)'", raw).group(1)
ORG_ID = re.search(r"organizations/([^/]+)", raw).group(1)
m = re.search(r"-H 'origin: ([^']+)'", raw)
ORIGIN = m.group(1) if m else "https://claude.hk.cn"
m2 = re.search(r"-H 'user-agent: ([^']+)'", raw)
UA = m2.group(1) if m2 else "Mozilla/5.0"

HEADERS = {
    "Content-Type": "application/json",
    "origin": ORIGIN,
    "user-agent": UA,
    "referer": f"{ORIGIN}/",
    "accept-language": "zh-CN,zh;q=0.9",
    "anthropic-client-platform": "web_claude_ai",
    "Cookie": COOKIE,
}

def create_conversation():
    data = json.dumps({"name":"","model":"claude-sonnet-4-6","is_temporary":False}).encode()
    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations",
        data=data, headers=HEADERS, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    body = json.loads(resp.read())
    return body["uuid"]

def send_message(conv_id, prompt):
    escaped = json.dumps(prompt)
    human_uuid = str(uuid.uuid4())
    asst_uuid = str(uuid.uuid4())
    payload = json.dumps({
        "prompt": prompt,
        "timezone": "Asia/Shanghai",
        "model": "claude-sonnet-4-6",
        "effort": "medium",
        "thinking_mode": "off",
        "tools": [{"type":"repl_v0","name":"repl"}],
        "turn_message_uuids": {
            "human_message_uuid": human_uuid,
            "assistant_message_uuid": asst_uuid
        },
        "attachments": [], "files": [],
        "rendering_mode": "messages"
    }).encode()

    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations/{conv_id}/completion",
        data=payload, headers={**HEADERS, "accept": "text/event-stream"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=300)

    full_text = []
    for line in resp:
        line = line.decode("utf-8", errors="replace").strip()
        if line.startswith("data: "):
            try:
                d = json.loads(line[6:])
                if d.get("type") == "content_block_delta":
                    t = d.get("delta", {}).get("text", "")
                    if t:
                        full_text.append(t)
                        print(t, end="", flush=True)
            except:
                pass
    print()
    return "".join(full_text)

def build_prompt(batch_num, commits):
    """构建给子Claude的第一轮prompt"""
    commit_list = "\n".join(commits)
    return f"""你是Walpurgis项目的子Claude执行者,负责从cugraph-gnn迁移commit到我们的仓库。

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git clone https://github.com/rapidsai/cugraph-gnn.git /tmp/cugraph-gnn
tree -L 2 --charset ascii src/walpurgis/
cat MULTI_CLAUDE_PLAN.md | head -30
```

## 你的任务: Batch {batch_num:02d}
迁移以下{len(commits)}个commit (从cugraph-gnn):
```
{commit_list}
```

## 操作方法
对每个commit:
1. `cd /tmp/cugraph-gnn && git show <hash> --stat` 看改了什么文件
2. `git diff <hash>^..<hash>` 深入看diff内容
3. 判断: 是否有迁移价值(CI/bot/changelog/merge → SKIP)
4. 有价值的commit: 鲁迅拿法迁移到 walpurgis-WTFGG/src/walpurgis/ 对应位置
   - 改写20%: 加断点调试print、改名适配walpurgis架构、加WALPURGIS_DEBUG输出
   - 不改字符串/docstring表面功夫
   - 不用v2/port/alt后缀
5. 写入MIGRATION_LOG.md末尾(格式同已有条目)

## commit信息
```bash
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
git add -A
git commit -m "migrate batch{batch_num:02d}: <概述>"
```

## 铁律
- 不开新分支
- 不用v2/v3/port/alt/bak后缀
- 改的是算法,不是字符串
- 每个commit的diff都要深入看,不能只看summary
- SKIP的commit也要在MIGRATION_LOG.md记录为什么skip
- 多写断点调试(WALPURGIS_DEBUG=1时print当前数据/结构体状态)
"""

def dispatch_batch(batch_num):
    batch_file = f"/tmp/batch_{batch_num:02d}"
    if not os.path.exists(batch_file):
        print(f"ERROR: {batch_file} not found")
        return
    with open(batch_file) as f:
        commits = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"  Dispatching Batch {batch_num:02d}: {len(commits)} commits")
    print(f"  First: {commits[0][:60]}")
    print(f"  Last:  {commits[-1][:60]}")
    print(f"{'='*60}\n")

    prompt = build_prompt(batch_num, commits)
    print(f"Prompt: {len(prompt)} chars")

    conv_id = create_conversation()
    print(f"Conv: {conv_id}")
    print(f"续传: CONV_ID={conv_id} python3 dispatch_migrate.py continue")

    # Save conv_id for tracking
    tracking_file = os.path.join(REPO_DIR, "dispatch_tracking.json")
    try:
        tracking = json.load(open(tracking_file))
    except:
        tracking = {}
    tracking[f"batch_{batch_num:02d}"] = {
        "conv_id": conv_id,
        "commits": len(commits),
        "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "dispatched"
    }
    with open(tracking_file, "w") as f:
        json.dump(tracking, f, indent=2)

    print(f"\n--- Sending prompt ---")
    response = send_message(conv_id, prompt)

    # Save response
    resp_file = os.path.join(REPO_DIR, f"submodel_batch{batch_num:02d}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(resp_file, "w") as f:
        f.write(response)
    print(f"\nSaved: {resp_file} ({len(response)} chars)")
    return conv_id

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dispatch_migrate.py <batch_num|all>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "all":
        for i in range(25):
            try:
                dispatch_batch(i)
                time.sleep(5)  # avoid rate limit
            except Exception as e:
                print(f"Batch {i} failed: {e}")
    else:
        dispatch_batch(int(arg))
