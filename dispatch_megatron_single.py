#!/usr/bin/env python3
"""
dispatch_megatron_single.py — 一个commit一个Claude
用法: python3 dispatch_megatron_single.py <commit_index>  (0-9061)
或:   python3 dispatch_megatron_single.py loop <start> <end>
每个sub-Claude收到: clone指令 + commit hash + diff内容 + 迁移目标说明
"""
import os, sys, re, json, time, uuid, subprocess
import urllib.request, urllib.error

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)

MEGATRON_DIR = os.path.join(os.path.dirname(REPO_DIR), "Megatron-LM")
TRACKING_FILE = os.path.join(REPO_DIR, "megatron_dispatch_tracking.json")

# ── cookie ──
config_dir = os.path.join(REPO_DIR, ".claude-hk-config")
if not os.path.isdir(config_dir):
    os.system(f"git clone https://github.com/dylanyunlon/claude-hk-config.git {config_dir}")
else:
    subprocess.run(["git", "-C", config_dir, "pull", "-q"], capture_output=True)

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

# ── 动态org解析 ──
def resolve_org():
    global ORG_ID
    try:
        req = urllib.request.Request(
            f"{ORIGIN}/api/organizations",
            headers={**HEADERS, "accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15)
        orgs = json.loads(resp.read())
        if isinstance(orgs, list) and orgs:
            live = orgs[0].get("uuid", "")
            if live and live != ORG_ID:
                print(f"Org updated: {ORG_ID} -> {live}")
                ORG_ID = live
    except Exception as e:
        print(f"Org resolve failed: {e}")

def create_conversation():
    data = json.dumps({
        "name": "", "model": "claude-sonnet-4-6",
        "is_temporary": False
    }).encode()
    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations",
        data=data, headers=HEADERS, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    body = json.loads(resp.read())
    return body["uuid"]

def send_message(conv_id, prompt):
    human_uuid = str(uuid.uuid4())
    asst_uuid = str(uuid.uuid4())
    payload = json.dumps({
        "prompt": prompt,
        "timezone": "Asia/Shanghai",
        "model": "claude-sonnet-4-6",
        "effort": "medium",
        "thinking_mode": "off",
        "tools": [{"type": "repl_v0", "name": "repl"}],
        "turn_message_uuids": {
            "human_message_uuid": human_uuid,
            "assistant_message_uuid": asst_uuid
        },
        "attachments": [], "files": [],
        "rendering_mode": "messages"
    }).encode()

    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations/{conv_id}/completion",
        data=payload,
        headers={**HEADERS, "accept": "text/event-stream"},
        method="POST")
    resp = urllib.request.urlopen(req, timeout=600)

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
            except json.JSONDecodeError:
                pass
    return "".join(full_text)

def get_commit_info(idx):
    """从Megatron-LM获取第idx个commit的完整信息"""
    result = subprocess.run(
        ["git", "-C", MEGATRON_DIR, "log", "--reverse",
         "--format=%H|%an|%ae|%s", "--skip", str(idx), "-1"],
        capture_output=True, text=True)
    parts = result.stdout.strip().split("|", 3)
    if len(parts) < 4:
        return None
    full_hash, author, email, subject = parts

    # 获取diff (限制大小避免prompt过长)
    diff_result = subprocess.run(
        ["git", "-C", MEGATRON_DIR, "diff-tree", "-p",
         "--stat", "--no-commit-id", "-r", full_hash],
        capture_output=True, text=True)
    diff_text = diff_result.stdout
    # 截断过长的diff (保留前8000字符)
    if len(diff_text) > 8000:
        diff_text = diff_text[:8000] + f"\n\n... [TRUNCATED: {len(diff_result.stdout)} chars total, showing first 8000]"

    return {
        "hash": full_hash,
        "short_hash": full_hash[:9],
        "author": author,
        "email": email,
        "subject": subject,
        "diff": diff_text,
        "index": idx,
    }

def build_prompt(commit_info):
    idx = commit_info["index"]
    return f"""你是walpurgis-WTFGG项目的迁移执行者。任务: 迁移Megatron-LM的第{idx}个commit。

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
```

## 你的commit
- Index: {idx} / 9062
- Hash: {commit_info['hash']}
- Author: {commit_info['author']}
- Subject: {commit_info['subject']}

## diff内容 (深入阅读每一行)
```
{commit_info['diff']}
```

## 迁移规则
1. **鲁迅拿法**: 不是全权复制。理解commit的算法/架构思想, 以20%改写迁移到walpurgis对应文件
2. **目标文件**: 根据commit改动的模块, 映射到walpurgis:
   - megatron/core/ → src/walpurgis/core/ 或 src/walpurgis/models/
   - megatron/training/ → src/walpurgis/models/trainer.py 或 train_walpurgis.py
   - megatron/data/ → src/walpurgis/dataloader/
   - megatron/model/ → src/walpurgis/models/
   - CI/docs/scripts → src/walpurgis/core/ (策略抽象)
   - 如果是纯merge/empty commit → 写MIGRATION_LOG.md记录SKIP
3. **断点调试**: 新增代码必须包含 `_dbg()` 或 `dump_struct_state()` 调用
4. **禁止**: v2/v3/port后缀, 新分支, 字符串改动

## commit信息
```bash
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
git add -A
git commit -m "migrate megatron {commit_info['short_hash']}: {commit_info['subject'][:60]}"
git push origin main
```

## MIGRATION_LOG.md格式 (追加到末尾)
```
## migrate megatron {commit_info['short_hash']}: {commit_info['subject'][:60]}
- **Upstream**: Megatron-LM {commit_info['hash']} ({commit_info['author']})
- **Subject**: {commit_info['subject']}
- **迁移位置**: <你迁移到的文件>
- **改写要点**: <20%改写的具体内容>
```
"""

def load_tracking():
    try:
        return json.load(open(TRACKING_FILE))
    except:
        return {}

def save_tracking(data):
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def dispatch_one(idx):
    print(f"\n{'='*60}")
    print(f"  Megatron Commit #{idx} / 9062")
    print(f"{'='*60}")

    commit = get_commit_info(idx)
    if not commit:
        print(f"ERROR: Could not get commit at index {idx}")
        return None

    print(f"  Hash:    {commit['short_hash']}")
    print(f"  Subject: {commit['subject'][:80]}")
    print(f"  Diff:    {len(commit['diff'])} chars")

    # 检查是否已dispatch
    tracking = load_tracking()
    key = f"megatron_{idx:05d}"
    if key in tracking and tracking[key].get("status") == "done":
        print(f"  SKIP: already dispatched and done")
        return None

    resolve_org()
    prompt = build_prompt(commit)
    print(f"  Prompt:  {len(prompt)} chars")

    try:
        conv_id = create_conversation()
    except Exception as e:
        print(f"  ERROR creating conversation: {e}")
        return None
    print(f"  Conv:    {conv_id}")

    tracking[key] = {
        "conv_id": conv_id,
        "hash": commit["short_hash"],
        "subject": commit["subject"][:100],
        "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "dispatched"
    }
    save_tracking(tracking)

    print(f"\n--- Sub-Claude Response ---")
    try:
        response = send_message(conv_id, prompt)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        tracking[key]["status"] = "error"
        tracking[key]["error"] = str(e)
        save_tracking(tracking)
        return conv_id

    # Save response
    resp_dir = os.path.join(REPO_DIR, "megatron_responses")
    os.makedirs(resp_dir, exist_ok=True)
    resp_file = os.path.join(resp_dir, f"commit_{idx:05d}_{commit['short_hash']}.txt")
    with open(resp_file, "w") as f:
        f.write(response)

    tracking[key]["status"] = "done"
    tracking[key]["response_file"] = resp_file
    tracking[key]["response_len"] = len(response)
    save_tracking(tracking)

    print(f"\n\n--- Saved: {resp_file} ({len(response)} chars) ---")
    return conv_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 dispatch_megatron_single.py <index>        # dispatch one commit")
        print("  python3 dispatch_megatron_single.py loop <start> <end>  # dispatch range")
        print("  python3 dispatch_megatron_single.py status         # show progress")
        sys.exit(1)

    if sys.argv[1] == "status":
        tracking = load_tracking()
        total = len(tracking)
        done = sum(1 for v in tracking.values() if v.get("status") == "done")
        errors = sum(1 for v in tracking.values() if v.get("status") == "error")
        print(f"Total dispatched: {total}")
        print(f"  Done:    {done}")
        print(f"  Errors:  {errors}")
        print(f"  Pending: {total - done - errors}")
        print(f"Progress:  {done}/9062 ({done/90.62:.1f}%)")
    elif sys.argv[1] == "loop":
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        end = int(sys.argv[3]) if len(sys.argv) > 3 else 9062
        print(f"Loop: commits {start} to {end}")
        for i in range(start, end):
            try:
                dispatch_one(i)
                time.sleep(3)  # rate limit
            except KeyboardInterrupt:
                print(f"\nStopped at commit {i}")
                break
            except Exception as e:
                print(f"\nCommit {i} failed: {e}")
                time.sleep(5)
    else:
        dispatch_one(int(sys.argv[1]))
