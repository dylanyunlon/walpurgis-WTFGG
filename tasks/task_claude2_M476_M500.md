# Walpurgis Phase 3 — Claude-2 任务 (M476-M500)

你是 Walpurgis 项目 Phase 3 的第二位 Claude (sonnet 4.6 medium)。
回复若被截断, 会收到 "Continue", 请从断点继续, 不要重复已输出内容。

## 环境准备 (与第一轮 prompt 同样的拿法)
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
git log --oneline -10
cat MULTI_CLAUDE_PLAN.md   # 重点看 Phase 3 章节
```

## 铁律
1. 不开新分支 — 所有改动直接在 main
2. 不用 v2/v3/port/verified/headtohead 等任何奇技淫巧后缀 (Phase 3 已全部清理, 别再引入)
3. 改的是算法 — 不改字符串/docstring 表面功夫
4. 作者信息: dylanyunlon <dogechat@163.com>
5. 服务器 (ags1, conda env: walking3) 只负责运行实验 + push, 不在服务器上派发 Claude
6. 实验运行有延迟, 可能与其他 Claude 的工作冲突: push 前先 `git pull --rebase origin main`

## 你的里程碑 M476-M500: METR-LA 完整实验 (空间自注意力首战)
Phase 3 新算法: 从 STAEformer 移植的空间自注意力 (src/walpurgis/models/model.py:SpatialSelfAttention),
门控残差 (init sigmoid(-3.0)≈0.047) + Pre-LN + 轻量FFN + 时间分块 + 注意力熵诊断。
SYNTH 3ep A/B 已验证: Best Val 5.2510 → 4.4708 (-14.9%), 零 NaN。决定性裁决在 METR-LA (N=207)。

服务器步骤:
```bash
cd /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG
git pull origin main        # 必须! 旧 checkout 会报 GIT_TOKEN: unbound variable (已修)
GPU=2 EPOCHS=200 bash experiments/run_server_experiment.sh
```

运行中盯紧诊断 (开 WALPURGIS_DEBUG=1 时):
- `spatial_attn.gate`: 初始 0.047, 应缓慢上升; 若全程 <0.05 说明模块被骨干抑制
- `spatial_attn.entropy`: log(207)=5.33; 健康区间约 1.6~4.5; ≈5.3=均匀没学到, ≈0=坍缩
- `per_horizon_MAE` / `depth_gates` / `cascade_weights` / `adaptive_emb_gate` 照旧

判据与分支:
- avg MAE < 2.85 → SOTA 达成! 通知 Claude-5 填 tex
- 2.85 ~ 2.90 → 微调: spa_gate 初始 -2.5, 或 num_heads=8 (112/8=14 整除)
- > 2.93 (比无空间注意力的 verified 2.93 还差) → 在 METR-LA.yaml 设 use_spatial_attn: False 回退, 记录消融数据给 Claude-4

结果自动写入 experiments/results/<RUN_ID>/ + summary.json 并 push 到 main。
完成后更新 MULTI_CLAUDE_PLAN.md 的 Phase 3 状态行 (第二位: ✅ + 一行结果)。
