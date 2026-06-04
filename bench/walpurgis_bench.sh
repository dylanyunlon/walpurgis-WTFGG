#!/usr/bin/env bash
# ================================================================
#  walpurgis_bench.sh — 通过 claude.hk.cn 构建理论基线 benchmark
#
#  原理: 向远端Claude发送结构化prompt, 要求它:
#    1) 查文献获取published baseline (D2STGNN原文Table 3/4)
#    2) 用code execution计算v10改动的理论预期偏移
#    3) 按固定JSON格式返回结果
#  本地解析SSE流, 提取数字, 生成对比表.
#
#  用法: bash walpurgis_bench.sh
#  依赖: curl, python3, jq(可选)
# ================================================================
set -euo pipefail

# ── API 配置 (从你的claude-hk-chat.sh复制) ─────────────────────
API="https://claude.hk.cn/api/organizations/0de6831b-fb77-41c7-bfb9-0899fb74f90f/chat_conversations/8ea00470-e663-405c-8020-26f1b8a7e1fc/completion"
MODEL="${MODEL:-claude-sonnet-4-6}"
EFFORT="${EFFORT:-high}"
TIMEOUT="${TIMEOUT:-180}"

COOKIES='CH-prefers-color-scheme=dark; ajs_anonymous_id=claudeai.v1.d6ccc63a-21ab-47e6-9214-d6edfc518c7e; share-session=6rrqz00r1ffzg0dizxnz0rb0zn63t3e7; lastActiveOrg=0de6831b-fb77-41c7-bfb9-0899fb74f90f'
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
ORIGIN='https://claude.hk.cn'

# ── 输出目录 ────────────────────────────────────────────────────
OUT_DIR="bench_results"
mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT_DIR/bench_${TIMESTAMP}.log"
RESULT_JSON="$OUT_DIR/bench_${TIMESTAMP}.json"

# ── 颜色 ────────────────────────────────────────────────────────
R='\033[0m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'; D='\033[90m'

# ── 核心: 发prompt, 返回completion全文 ──────────────────────────
send() {
  local prompt="$1"
  local escaped
  escaped=$(printf '%s' "$prompt" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')

  local payload="{\"prompt\":${escaped},\"timezone\":\"Asia/Shanghai\",\"personalized_styles\":[{\"type\":\"default\",\"key\":\"Default\",\"name\":\"Normal\",\"nameKey\":\"normal_style_name\",\"prompt\":\"Normal\\n\",\"summary\":\"Default responses from Claude\",\"summaryKey\":\"normal_style_summary\",\"isDefault\":true}],\"locale\":\"en-US\",\"model\":\"${MODEL}\",\"effort\":\"${EFFORT}\",\"thinking_mode\":\"off\",\"tools\":[]}"

  local raw
  raw=$(curl -s --max-time "$TIMEOUT" "$API" \
    -H 'accept: text/event-stream' \
    -H 'content-type: application/json' \
    -b "$COOKIES" \
    -H "origin: $ORIGIN" \
    -H "user-agent: $UA" \
    --data-raw "$payload" 2>&1)

  # 从SSE流拼接completion
  echo "$raw" | python3 -c '
import sys, json
text = ""
for line in sys.stdin:
    line = line.strip()
    if line.startswith("data: "):
        try:
            d = json.loads(line[6:])
            text += d.get("completion", "")
        except: pass
print(text)
'
}

# ── Benchmark Prompt 模板 ──────────────────────────────────────

# Prompt 1: 从文献中提取D2STGNN published baselines
PROMPT_BASELINES='You are a research assistant. I need the EXACT published results from the D2STGNN paper (Shao et al., VLDB 2022, "Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting").

From Table 3 and Table 4 of the paper, extract the D2STGNN results for these datasets and horizons:

Datasets: METR-LA, PEMS-BAY, PEMS04, PEMS08
Metrics: MAE, RMSE, MAPE
Horizons: 15min (3-step), 30min (6-step), 60min (12-step), and Average

Output ONLY a JSON object, no other text, no markdown fences:
{
  "source": "D2STGNN VLDB2022 Table3/Table4",
  "results": {
    "METR-LA": {
      "horizon_3":  {"MAE": ..., "RMSE": ..., "MAPE": ...},
      "horizon_6":  {"MAE": ..., "RMSE": ..., "MAPE": ...},
      "horizon_12": {"MAE": ..., "RMSE": ..., "MAPE": ...},
      "average":    {"MAE": ..., "RMSE": ..., "MAPE": ...}
    },
    "PEMS-BAY": { ... same structure ... },
    "PEMS04":   { ... same structure ... },
    "PEMS08":   { ... same structure ... }
  }
}

Use the exact numbers from the paper. If a specific horizon is not reported separately, use the average. PEMS04/PEMS08 may only have average results - that is fine, fill what is available.'

# Prompt 2: v10改动的理论影响分析
PROMPT_THEORY='You are an expert in spatio-temporal graph neural networks. Given the D2STGNN baseline and the following algorithmic modifications (our "v10" variant), estimate the expected impact on MAE/RMSE/MAPE for METR-LA.

v10 modifications:
1. Loss: smooth Huber + log-cosh hybrid (replaces pure masked MAE)
2. Estimation Gate: Swish activation, dual-head attention, GroupNorm
3. Residual Decomp: Mish activation, learnable residual scale alpha=0.9
4. Diffusion: InstanceNorm→LayerNorm, gconv skip connection, cosine annealing AR dropout
5. Dynamic Graph: multi-head(3) Q-K attention, LayerNorm, cosine-sim auxiliary
6. Inherent Block: RMSNorm GRU, Rotary PE transformer, gradient checkpoint
7. Training: adaptive p90 gradient clipping, warmup-cosine LR schedule
8. Data: Tukey fences outlier removal, sin/cos periodic encoding
9. Adjacency: RBF kernel + k-NN(15) sparsification + symmetric closure

For each modification, estimate the relative MAE change (positive = worse, negative = better) on METR-LA, based on your knowledge of the traffic forecasting literature.

Output ONLY JSON, no other text:
{
  "dataset": "METR-LA",
  "baseline_MAE": <D2STGNN published average MAE>,
  "modifications": [
    {"name": "huber_logcosh_loss", "expected_MAE_delta_pct": ...},
    {"name": "swish_gate_groupnorm", "expected_MAE_delta_pct": ...},
    {"name": "mish_residual_scale", "expected_MAE_delta_pct": ...},
    {"name": "layernorm_skip_cosine_dropout", "expected_MAE_delta_pct": ...},
    {"name": "multihead_attention_graph", "expected_MAE_delta_pct": ...},
    {"name": "rmsnorm_gru_rotary_pe", "expected_MAE_delta_pct": ...},
    {"name": "adaptive_clip_warmup_cosine", "expected_MAE_delta_pct": ...},
    {"name": "tukey_periodic_encoding", "expected_MAE_delta_pct": ...},
    {"name": "rbf_knn_symmetric_adj", "expected_MAE_delta_pct": ...}
  ],
  "combined_expected_MAE_delta_pct": ...,
  "expected_v10_MAE": ...,
  "confidence": "low|medium|high",
  "reasoning_summary": "..."
}'

# Prompt 3: SOTA对比表
PROMPT_SOTA='You are a traffic forecasting benchmark expert. For the METR-LA dataset, provide the current state-of-the-art results (as of 2025-2026) from published papers.

Include these models: DCRNN, STGCN, Graph WaveNet, ASTGCN, STSGCN, AGCRN, STFGNN, STG-NCDE, D2STGNN, PDFormer, STAEFormer, and any more recent SOTA.

Output ONLY JSON:
{
  "dataset": "METR-LA",
  "metric": "average MAE/RMSE/MAPE across 12 horizons",
  "models": [
    {"name": "DCRNN", "year": 2018, "venue": "ICLR", "MAE": ..., "RMSE": ..., "MAPE": ...},
    {"name": "Graph WaveNet", "year": 2019, "venue": "IJCAI", "MAE": ..., "RMSE": ..., "MAPE": ...},
    ...
  ],
  "notes": "..."
}'

# ── 执行 benchmark ──────────────────────────────────────────────
echo -e "${Y}╔══════════════════════════════════════════════╗${R}"
echo -e "${Y}║  Walpurgis Benchmark Runner                  ║${R}"
echo -e "${Y}║  Model: ${C}${MODEL}${Y}  Effort: ${C}${EFFORT}${Y}           ║${R}"
echo -e "${Y}╚══════════════════════════════════════════════╝${R}"
echo ""

run_bench() {
  local name="$1"
  local prompt="$2"
  local outfile="$OUT_DIR/${name}_${TIMESTAMP}.json"

  echo -e "${G}[${name}]${R} 发送中..." | tee -a "$LOG"
  local t_start=$(date +%s)

  local result
  result=$(send "$prompt")

  local t_end=$(date +%s)
  local elapsed=$((t_end - t_start))
  echo -e "${G}[${name}]${R} 完成 (${elapsed}s)" | tee -a "$LOG"

  # 尝试提取JSON
  local json_part
  json_part=$(echo "$result" | python3 -c '
import sys, json, re
text = sys.stdin.read()
# 尝试直接解析
try:
    obj = json.loads(text.strip())
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    sys.exit(0)
except: pass
# 尝试从markdown fence里提取
m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
if m:
    try:
        obj = json.loads(m.group(1))
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        sys.exit(0)
    except: pass
# 尝试找第一个{...}
m = re.search(r"\{.*\}", text, re.DOTALL)
if m:
    try:
        obj = json.loads(m.group(0))
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        sys.exit(0)
    except: pass
# 全部失败, 输出原文
print(json.dumps({"raw_response": text, "parse_error": True}, indent=2, ensure_ascii=False))
' 2>/dev/null || echo '{"error": "python parse failed"}')

  echo "$json_part" > "$outfile"
  echo -e "${D}  -> ${outfile}${R}" | tee -a "$LOG"
  echo "$json_part" | python3 -c '
import sys, json
d = json.load(sys.stdin)
if d.get("parse_error"):
    print("  [WARN] JSON解析失败, 存为raw_response")
elif "error" in d:
    print(f"  [ERROR] {d[\"error\"]}")
else:
    keys = list(d.keys())[:5]
    print(f"  [OK] keys: {keys}")
' 2>/dev/null
  echo ""
}

# ── 依次执行三个benchmark prompt ─────────────────────────────
echo -e "${Y}[1/3] D2STGNN Published Baselines${R}"
run_bench "baselines" "$PROMPT_BASELINES"

echo -e "${Y}[2/3] v10 Modification Impact Analysis${R}"
run_bench "v10_theory" "$PROMPT_THEORY"

echo -e "${Y}[3/3] METR-LA SOTA Leaderboard${R}"
run_bench "sota" "$PROMPT_SOTA"

# ── 汇总生成对比表 ────────────────────────────────────────────
echo -e "${Y}[汇总] 生成对比表...${R}"

python3 << PYEOF
import json, os, glob

ts = "${TIMESTAMP}"
d = "${OUT_DIR}"

# 加载结果
def load(name):
    path = f"{d}/{name}_{ts}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

baselines = load("baselines")
theory = load("v10_theory")
sota = load("sota")

print("\n" + "=" * 70)
print("  WALPURGIS BENCHMARK REPORT")
print("  Generated:", ts)
print("=" * 70)

# D2STGNN baselines
if "results" in baselines:
    print("\n[1] D2STGNN Published Baselines")
    print("-" * 50)
    for ds, horizons in baselines["results"].items():
        if "average" in horizons:
            avg = horizons["average"]
            print(f"  {ds:12s}  MAE={avg.get('MAE','?'):>6}  "
                  f"RMSE={avg.get('RMSE','?'):>6}  "
                  f"MAPE={avg.get('MAPE','?'):>6}")

# v10 theory
if "modifications" in theory:
    print(f"\n[2] v10 Modification Impact (METR-LA)")
    print("-" * 50)
    base = theory.get("baseline_MAE", "?")
    print(f"  Baseline MAE: {base}")
    for m in theory["modifications"]:
        delta = m.get("expected_MAE_delta_pct", "?")
        name = m.get("name", "?")
        sign = "+" if isinstance(delta, (int,float)) and delta > 0 else ""
        print(f"  {name:35s} {sign}{delta}%")
    combined = theory.get("combined_expected_MAE_delta_pct", "?")
    expected = theory.get("expected_v10_MAE", "?")
    conf = theory.get("confidence", "?")
    print(f"  {'COMBINED':35s} {combined}%")
    print(f"  Expected v10 MAE: {expected}  (confidence: {conf})")

# SOTA
if "models" in sota:
    print(f"\n[3] METR-LA SOTA Leaderboard")
    print("-" * 50)
    models = sorted(sota["models"], key=lambda x: x.get("MAE", 999))
    for i, m in enumerate(models, 1):
        print(f"  {i:2d}. {m.get('name','?'):20s} ({m.get('year','?')}) "
              f"MAE={m.get('MAE','?'):>6}  "
              f"RMSE={m.get('RMSE','?'):>6}")

# v10 target
if "expected_v10_MAE" in theory and "models" in sota:
    target = theory["expected_v10_MAE"]
    if isinstance(target, (int, float)):
        rank = sum(1 for m in sota["models"]
                   if isinstance(m.get("MAE"), (int,float))
                   and m["MAE"] < target) + 1
        print(f"\n  >>> v10 expected rank: #{rank}/{len(sota['models'])+1}")
        print(f"  >>> Target MAE to beat: {target}")

print("\n" + "=" * 70)

# 保存合并结果
combined = {
    "timestamp": ts,
    "baselines": baselines,
    "v10_theory": theory,
    "sota": sota
}
combined_path = f"{d}/bench_{ts}.json"
with open(combined_path, 'w') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
print(f"Full results: {combined_path}")
PYEOF

echo ""
echo -e "${G}Benchmark完成.${R} 结果在 ${OUT_DIR}/"
echo -e "日志: ${LOG}"
