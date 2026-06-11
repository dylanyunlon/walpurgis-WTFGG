#!/usr/bin/env python3
"""
dispatch_single.py — 一个Claude一个commit
用法: python3 dispatch_single.py <commit_hash>
或:   python3 dispatch_single.py batch <start> <count>
"""
import os, sys, re, json, time, uuid
import urllib.request

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)

with open(os.path.join(REPO_DIR, ".claude-hk-config", "raw_curl.txt")) as f:
    raw = f.read()

COOKIE = re.search(r"-b '([^']+)'", raw).group(1)
ORG_ID = re.search(r"organizations/([^/]+)", raw).group(1)
ORIGIN = re.search(r"-H 'origin: ([^']+)'", raw).group(1)
UA = re.search(r"-H 'user-agent: ([^']+)'", raw).group(1)

HEADERS = {
    "Content-Type": "application/json",
    "origin": ORIGIN, "user-agent": UA,
    "referer": f"{ORIGIN}/",
    "accept-language": "zh-CN,zh;q=0.9",
    "anthropic-client-platform": "web_claude_ai",
    "Cookie": COOKIE,
}

GIT_TOKEN = os.environ.get("GIT_TOKEN", "")

def create_conv():
    data = json.dumps({"name":"","model":"claude-sonnet-4-6","is_temporary":False}).encode()
    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations",
        data=data, headers=HEADERS, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())["uuid"]

def send(conv_id, prompt):
    payload = json.dumps({
        "prompt": prompt, "timezone": "Asia/Shanghai",
        "model": "claude-sonnet-4-6", "effort": "medium",
        "thinking_mode": "off",
        "tools": [{"type":"repl_v0","name":"repl"}],
        "turn_message_uuids": {
            "human_message_uuid": str(uuid.uuid4()),
            "assistant_message_uuid": str(uuid.uuid4())
        },
        "attachments": [], "files": [], "rendering_mode": "messages"
    }).encode()
    req = urllib.request.Request(
        f"{ORIGIN}/api/organizations/{ORG_ID}/chat_conversations/{conv_id}/completion",
        data=payload, headers={**HEADERS, "accept": "text/event-stream"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=8)
    except:
        pass  # fire and forget

def dispatch_one(commit_hash, commit_msg=""):
    prompt = f"""apt install -y tree git 2>/dev/null
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git clone https://github.com/rapidsai/cugraph-gnn.git /tmp/cugraph-gnn
cd /tmp/cugraph-gnn && git show {commit_hash} --stat && git diff {commit_hash}^..{commit_hash}

任务: 把这个commit ({commit_hash}) 迁移到 walpurgis-WTFGG/src/walpurgis/ 。
- 鲁迅拿法改写20%: 加WALPURGIS_DEBUG断点print, 适配walpurgis架构
- 如果是CI/merge/docs/changelog → 在MIGRATION_LOG.md记一行SKIP即可
- 不开新分支,不用v2/port后缀
- 完成后:
cd ~/walpurgis-WTFGG
git config user.name "dylanyunlon" && git config user.email "dogechat@163.com"
git remote set-url origin https://x-access-token:{GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git
git add -A && git commit -m "migrate {commit_hash}: 描述"
git pull --rebase origin main && git push origin main"""

    conv_id = create_conv()
    send(conv_id, prompt)
    return conv_id

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dispatch_single.py <hash> | batch <start> <count>")
        sys.exit(1)

    if sys.argv[1] == "batch":
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        commits = open("/tmp/substantive_commits.txt").readlines()

        tracking_file = os.path.join(REPO_DIR, "dispatch_single_tracking.json")
        try:
            tracking = json.load(open(tracking_file))
        except:
            tracking = {}

        for i in range(start, min(start + count, len(commits))):
            line = commits[i].strip()
            hash = line[:7]
            msg = line[8:]
            print(f"[{i+1}/{len(commits)}] {hash} {msg[:50]}...", end=" ")
            try:
                conv_id = dispatch_one(hash, msg)
                print(f"conv={conv_id[:12]}")
                tracking[hash] = {
                    "conv_id": conv_id,
                    "msg": msg,
                    "dispatched": time.strftime("%H:%M:%S"),
                    "idx": i
                }
            except Exception as e:
                print(f"FAIL: {e}")
            time.sleep(2)

        with open(tracking_file, "w") as f:
            json.dump(tracking, f, indent=2)
        print(f"\nDispatched {count} commits starting from index {start}")
    else:
        hash = sys.argv[1]
        conv_id = dispatch_one(hash)
        print(f"Dispatched {hash} -> conv={conv_id}")
