#!/usr/bin/env python3
"""
walpurgis_eval.py — 10维正交评估
用法: python walpurgis_eval.py --pred output/pred.npz --truth output/truth.npz [--adj adj.pkl] [--params 519280]

输入:
  pred.npz:  y_pred (N, 12, 207) — 模型预测
  truth.npz: y_true (N, 12, 207) — 真实值(反归一化后, mph)
  adj.pkl:   邻接矩阵 (可选, D3/D7需要)
  params:    参数量 (可选, D6需要)

输出:
  10维评估结果JSON + 终端报告 + 雷达图PNG
"""
import argparse
import json
import numpy as np
import os
import sys

# ── 目标值 (来自LLM-as-benchmark) ──────────────────────────────
TARGETS = {
    "D1":  {"name": "Short-Horizon MAE@h3",           "baseline": 2.55, "target": 2.45, "stretch": 2.35, "unit": "mph",  "dir": "lower"},
    "D2":  {"name": "Long-Horizon Ratio h12/h3",      "baseline": 1.18, "target": 1.14, "stretch": 1.10, "unit": "ratio","dir": "lower"},
    "D3":  {"name": "Spatial Equity sparse/dense",     "baseline": 1.35, "target": 1.20, "stretch": 1.10, "unit": "ratio","dir": "lower"},
    "D4":  {"name": "Congestion MAE (speed<30)",       "baseline": 5.80, "target": 5.20, "stretch": 4.60, "unit": "mph",  "dir": "lower"},
    "D5":  {"name": "Tail P95 Error",                  "baseline": 9.50, "target": 8.50, "stretch": 7.50, "unit": "mph",  "dir": "lower"},
    "D6":  {"name": "Efficiency MAE/M-params",         "baseline": 1.95, "target": 1.75, "stretch": 1.55, "unit": "mph/M","dir": "lower"},
    "D7":  {"name": "Graph Sensitivity (adj=I drop)",  "baseline": 0.42, "target": 0.55, "stretch": 0.70, "unit": "mph",  "dir": "higher"},
    "D8":  {"name": "Periodicity WE-WD gap",           "baseline": 0.32, "target": 0.22, "stretch": 0.14, "unit": "mph",  "dir": "lower"},
    "D9":  {"name": "Calibration |bias|",              "baseline": 0.08, "target": 0.04, "stretch": 0.01, "unit": "mph",  "dir": "lower"},
    "D10": {"name": "Stability σ(MAE) across seeds",   "baseline": 0.08, "target": 0.05, "stretch": 0.02, "unit": "mph",  "dir": "lower"},
}


def compute_all(y_pred, y_true, adj=None, param_count=None,
                y_pred_identity=None, seed_maes=None, weekend_mask=None):
    """计算10个维度的指标值."""
    results = {}
    ae = np.abs(y_pred - y_true)

    # D1: Short-Horizon MAE@h3 (horizon index 2, 即15min)
    d1 = np.mean(ae[:, 2, :])
    results["D1"] = round(float(d1), 4)
    print(f"  [D1] MAE@h3 = {d1:.4f}")

    # D2: Long-Horizon Ratio
    mae_h3 = np.mean(ae[:, 2, :])
    mae_h12 = np.mean(ae[:, 11, :])
    d2 = mae_h12 / max(mae_h3, 1e-8)
    results["D2"] = round(float(d2), 4)
    print(f"  [D2] ratio h12/h3 = {d2:.4f}  (h3={mae_h3:.3f}, h12={mae_h12:.3f})")

    # D3: Spatial Equity
    if adj is not None:
        degrees = (adj > 0).sum(axis=1)
        q25 = np.percentile(degrees, 25)
        q75 = np.percentile(degrees, 75)
        sparse_idx = np.where(degrees <= q25)[0]
        dense_idx = np.where(degrees >= q75)[0]
        if len(sparse_idx) > 0 and len(dense_idx) > 0:
            mae_sparse = np.mean(ae[:, :, sparse_idx])
            mae_dense = np.mean(ae[:, :, dense_idx])
            d3 = mae_sparse / max(mae_dense, 1e-8)
            results["D3"] = round(float(d3), 4)
            print(f"  [D3] equity = {d3:.4f}  (sparse={mae_sparse:.3f}[{len(sparse_idx)}nodes], dense={mae_dense:.3f}[{len(dense_idx)}nodes])")
        else:
            results["D3"] = None
            print(f"  [D3] skipped (insufficient nodes in quartiles)")
    else:
        results["D3"] = None
        print(f"  [D3] skipped (no adj)")

    # D4: Congestion MAE
    congestion_mask = y_true < 30.0
    if congestion_mask.sum() > 100:
        d4 = np.mean(ae[congestion_mask])
        pct = congestion_mask.mean() * 100
        results["D4"] = round(float(d4), 4)
        print(f"  [D4] congestion MAE = {d4:.4f}  ({pct:.1f}% samples < 30mph)")
    else:
        results["D4"] = None
        print(f"  [D4] skipped (too few congested samples: {congestion_mask.sum()})")

    # D5: Tail P95
    d5 = np.percentile(ae.ravel(), 95)
    results["D5"] = round(float(d5), 4)
    print(f"  [D5] P95 error = {d5:.4f}")

    # D6: Efficiency
    if param_count and param_count > 0:
        overall_mae = np.mean(ae)
        d6 = overall_mae / (param_count / 1e6)
        results["D6"] = round(float(d6), 4)
        print(f"  [D6] efficiency = {d6:.4f} mph/M-params  (MAE={overall_mae:.3f}, params={param_count:,})")
    else:
        results["D6"] = None
        print(f"  [D6] skipped (no param_count)")

    # D7: Graph Sensitivity
    if y_pred_identity is not None:
        mae_with_graph = np.mean(ae)
        mae_identity = np.mean(np.abs(y_pred_identity - y_true))
        d7 = mae_identity - mae_with_graph
        results["D7"] = round(float(d7), 4)
        print(f"  [D7] graph drop = {d7:.4f}  (identity={mae_identity:.3f}, normal={mae_with_graph:.3f})")
    else:
        results["D7"] = None
        print(f"  [D7] skipped (no identity-adj predictions)")

    # D8: Periodicity
    if weekend_mask is not None:
        weekday_mask = ~weekend_mask
        if weekend_mask.sum() > 100 and weekday_mask.sum() > 100:
            mae_we = np.mean(ae[weekend_mask])
            mae_wd = np.mean(ae[weekday_mask])
            d8 = abs(mae_we - mae_wd)
            results["D8"] = round(float(d8), 4)
            print(f"  [D8] WE-WD gap = {d8:.4f}  (WE={mae_we:.3f}[{weekend_mask.sum()}], WD={mae_wd:.3f}[{weekday_mask.sum()}])")
        else:
            results["D8"] = None
            print(f"  [D8] skipped (insufficient WE/WD samples)")
    else:
        results["D8"] = None
        print(f"  [D8] skipped (no weekend_mask)")

    # D9: Calibration
    d9 = abs(np.mean(y_pred - y_true))
    results["D9"] = round(float(d9), 4)
    print(f"  [D9] |bias| = {d9:.4f}  (raw bias = {np.mean(y_pred - y_true):.4f})")

    # D10: Stability
    if seed_maes is not None and len(seed_maes) > 1:
        d10 = np.std(seed_maes)
        results["D10"] = round(float(d10), 4)
        print(f"  [D10] σ(MAE) = {d10:.4f}  across {len(seed_maes)} seeds: {seed_maes}")
    else:
        results["D10"] = None
        print(f"  [D10] skipped (need multiple seed results)")

    return results


def grade(results):
    """给每个维度打分: PASS/TARGET/STRETCH/MISS."""
    report = []
    for dim_id, info in TARGETS.items():
        val = results.get(dim_id)
        if val is None:
            report.append({"id": dim_id, "name": info["name"], "value": None, "grade": "N/A"})
            continue

        bl = info["baseline"]
        tg = info["target"]
        st = info["stretch"]
        d = info["dir"]

        if d == "lower":
            if val <= st:     grade_str = "STRETCH"
            elif val <= tg:   grade_str = "TARGET"
            elif val <= bl:   grade_str = "PASS"
            else:             grade_str = "MISS"
        elif d == "higher":
            if val >= st:     grade_str = "STRETCH"
            elif val >= tg:   grade_str = "TARGET"
            elif val >= bl:   grade_str = "PASS"
            else:             grade_str = "MISS"
        else:  # near_zero
            if val <= st:     grade_str = "STRETCH"
            elif val <= tg:   grade_str = "TARGET"
            elif val <= bl:   grade_str = "PASS"
            else:             grade_str = "MISS"

        report.append({
            "id": dim_id, "name": info["name"],
            "value": val, "baseline": bl, "target": tg, "stretch": st,
            "grade": grade_str, "unit": info["unit"]
        })

    return report


def print_report(report):
    """终端打印10维报告."""
    symbols = {"STRETCH": "★", "TARGET": "✓", "PASS": "○", "MISS": "✗", "N/A": "─"}
    colors = {"STRETCH": "\033[35m", "TARGET": "\033[32m", "PASS": "\033[33m", "MISS": "\033[31m", "N/A": "\033[90m"}
    R = "\033[0m"

    print("\n" + "=" * 74)
    print("  WALPURGIS 10-DIMENSION EVALUATION REPORT")
    print("=" * 74)
    print(f"  {'ID':4s} {'Dimension':35s} {'Value':>8s} {'Base':>6s} {'Tgt':>6s} {'Grade':>8s}")
    print("  " + "─" * 68)

    n_pass = n_target = n_stretch = n_miss = n_na = 0
    for r in report:
        g = r["grade"]
        sym = symbols[g]
        c = colors[g]
        val_str = f'{r["value"]:.3f}' if r["value"] is not None else "  N/A"
        bl_str = f'{r["baseline"]:.2f}' if "baseline" in r else ""
        tg_str = f'{r["target"]:.2f}' if "target" in r else ""
        print(f'  {r["id"]:4s} {r["name"]:35s} {val_str:>8s} {bl_str:>6s} {tg_str:>6s} {c}{sym} {g:7s}{R}')

        if g == "STRETCH": n_stretch += 1
        elif g == "TARGET": n_target += 1
        elif g == "PASS": n_pass += 1
        elif g == "MISS": n_miss += 1
        else: n_na += 1

    print("  " + "─" * 68)
    print(f"  Summary: {n_stretch}★ STRETCH  {n_target}✓ TARGET  {n_pass}○ PASS  {n_miss}✗ MISS  {n_na}─ N/A")
    print("=" * 74)


def save_radar_data(report, path):
    """保存雷达图数据(可用matplotlib或前端渲染)."""
    radar = {"dimensions": [], "baseline": [], "target": [], "actual": []}
    for r in report:
        if r["value"] is None:
            continue
        radar["dimensions"].append(r["id"])
        bl = r.get("baseline", 1)
        tg = r.get("target", 1)
        val = r["value"]
        # 归一化: baseline=0.5, target=0.75, stretch=1.0
        # 对lower_better: 越低越好, 所以反转
        info = TARGETS[r["id"]]
        if info["dir"] == "lower":
            # 0 = worst(2x baseline), 0.5 = baseline, 1.0 = stretch
            st = info["stretch"]
            worst = bl * 2
            norm_val = 1.0 - (val - st) / max(worst - st, 1e-8)
            norm_bl = 1.0 - (bl - st) / max(worst - st, 1e-8)
            norm_tg = 1.0 - (tg - st) / max(worst - st, 1e-8)
        else:
            # higher_better
            st = info["stretch"]
            norm_val = val / max(st, 1e-8)
            norm_bl = bl / max(st, 1e-8)
            norm_tg = tg / max(st, 1e-8)

        radar["baseline"].append(round(max(0, min(1, norm_bl)), 3))
        radar["target"].append(round(max(0, min(1, norm_tg)), 3))
        radar["actual"].append(round(max(0, min(1, norm_val)), 3))

    with open(path, 'w') as f:
        json.dump(radar, f, indent=2)
    print(f"Radar data: {path}")


def main():
    parser = argparse.ArgumentParser(description="Walpurgis 10-Dim Evaluation")
    parser.add_argument("--pred", required=True, help="y_pred npz file")
    parser.add_argument("--truth", required=True, help="y_true npz file")
    parser.add_argument("--adj", default=None, help="adj_mx pkl file")
    parser.add_argument("--params", type=int, default=None, help="model parameter count")
    parser.add_argument("--pred-identity", default=None, help="predictions with identity adj")
    parser.add_argument("--seed-maes", nargs="+", type=float, default=None)
    parser.add_argument("--output", default="eval_results", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载数据
    y_pred = np.load(args.pred)["y_pred"] if args.pred.endswith(".npz") else np.load(args.pred)
    y_true = np.load(args.truth)["y_true"] if args.truth.endswith(".npz") else np.load(args.truth)
    print(f"Loaded: pred={y_pred.shape}, truth={y_true.shape}")

    adj = None
    if args.adj:
        import pickle
        with open(args.adj, 'rb') as f:
            _, _, adj = pickle.load(f)

    y_pred_id = None
    if args.pred_identity:
        y_pred_id = np.load(args.pred_identity)["y_pred"]

    # 计算
    print("\nComputing 10 dimensions:")
    results = compute_all(
        y_pred, y_true, adj=adj,
        param_count=args.params,
        y_pred_identity=y_pred_id,
        seed_maes=args.seed_maes,
        weekend_mask=None,  # 需要从数据中提取
    )

    # 打分
    report = grade(results)
    print_report(report)

    # 保存
    out_path = os.path.join(args.output, "eval_10dim.json")
    with open(out_path, 'w') as f:
        json.dump({"results": results, "report": report, "targets": TARGETS}, f, indent=2)
    print(f"Saved: {out_path}")

    radar_path = os.path.join(args.output, "radar_data.json")
    save_radar_data(report, radar_path)


if __name__ == "__main__":
    main()
