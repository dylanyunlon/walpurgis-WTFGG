# Walpurgis — Workload-Aware GNN Training on Mixed-Generation GPUs

> *"Die Hexen zu dem Brocken ziehn, die Stoppel ist gelb, die Saat ist grün."*
> — Walpurgisnacht, Faust I

Heterogeneous GPU temporal-subgraph engine + D2STGNN traffic forecasting.

## Quick Start (Server)

```bash
# 1. 配置环境
bash experiments/setup_env.sh

# 2. 运行全部实验 (并行: D2STGNN baseline + Walpurgis)
bash experiments/run_all.sh

# 3. 或单独运行
GPU=2 EPOCHS=100 bash experiments/run_metrla.sh --tag myrun
```

## Structure

```
src/walpurgis/          — 统一模型代码 (model/losses/trainer/configs)
upstream/d2stgnn/       — D2STGNN 原始基线
upstream/morphgl/       — MorphGL 参考
experiments/            — 实验脚本和结果
  setup_env.sh          — 环境配置
  run_metrla.sh         — 训练 + 评估 + push
  run_baselines.sh      — 基线对比
  run_all.sh            — 全流程并行
  results/              — 实验结果 (JSON, 自动 push)
bench/                  — 评估框架
src/core/               — C++ 异构内存引擎
src/bridge/             — 时序索引桥接层
src/cuda/               — CUDA 基准测试
```

## Target

METR-LA average MAE < 2.85 (surpassing TITAN 2.88, STAEFormer 2.90).

## Upstream

| Directory | Origin | Role |
|-----------|--------|------|
| `upstream/d2stgnn` | [GestaltCogTeam/D2STGNN](https://github.com/GestaltCogTeam/D2STGNN) | Decoupled ST-GNN baseline |
| `upstream/morphgl` | [initzhang/MorphGL](https://github.com/initzhang/MorphGL) | Collective batching reference |

## Development Plan

See [MULTI_CLAUDE_PLAN.md](MULTI_CLAUDE_PLAN.md) for milestone assignments.
