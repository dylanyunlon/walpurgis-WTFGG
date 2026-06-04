# Walpurgis-WTFGG Claude Development Plan

## Milestone Registry

| Milestone | Description |
|-----------|-------------|
| M001 | Project scaffolding, upstream clone, initial walpurgis copy |
| M002 | walpurgis_ported (first port attempt, minimal changes) |
| M003 | walpurgis_ported_v2 (heavy rewrite, debug flags, common modules) |
| M004 | Review pass M005-M006 |
| M005 | walpurgis_ported_v3: core models (model.py, trainer.py, losses.py) |
| M006 | walpurgis_ported_v3: decouple + diffusion_block + inherent_block |
| M007 | walpurgis_ported_v3: dynamic_graph_conv + utils (cal_adj, train, log) |
| M008 | walpurgis_ported_v3: dataloader + load_data + configs |
| M009 | walpurgis_ported_v3: datasets generators (speed/flow common + wrappers) |
| M010 | walpurgis_ported_v3: main.py entry point + full debug flag system |
| M011 | Integration test: dry-run import check, flag smoke test |
| M012 | C++ bridge: temporal_bridge.hpp ↔ Python interop |
| M013 | CUDA hetero_bench.cu integration with v3 Python |
| M014 | Benchmark suite: philemon_bench against v3 |
| M015 | CI pipeline: Makefile targets for v3 train/test/bench |
| M016 | Documentation: debug flag reference, porting changelog |
| M017 | Final review + merge to main |
| M018 | Post-merge: performance regression tests |

---

## Claude Session Assignment

| Claude # | Milestones | Status |
|----------|------------|--------|
| **第一位 Claude** | M001 – M025 | ✅ 已完成 — 创建 src/walpurgis/ 41py+4yaml |
| **第二位 Claude** | M026 – M040 | ✅ 已完成 — 算法深化 + 断点快照系统 |
| **第三位 Claude** | M041 – M055 | ✅ 已完成 — 标签清除 + 算法增强(诊断/噪声/平坦度/退火) |
| **第四位 Claude** | M056 – M075 | ⏳ 待开发 |
| **第五位 Claude** | M076 – M095 | ⏳ 待开发 |
| **第六位 Claude** | M096 – M115 | ⏳ 待开发 |

---

## Claude 1 Deliverables (M005–M010)

### Files created (45 total)

**Core models (M005-M006):**
- `models/model.py` — D2STGNN with `--debug-model`
- `models/trainer.py` — train/eval/test with `--debug-trainer`
- `models/losses.py` — masked MAE/MSE/RMSE/MAPE with `--debug-loss`
- `models/decouple/estimation_gate.py` — `--debug-gate`
- `models/decouple/residual_decomp.py` — `--debug-resdecomp`
- `models/diffusion_block/dif_block.py` — `--debug-difblk`
- `models/diffusion_block/dif_model.py` — `--debug-stconv`
- `models/diffusion_block/forecast.py` — `--debug-diffc`
- `models/inherent_block/inh_block.py` — `--debug-inhblk`
- `models/inherent_block/inh_model.py` — `--debug-inhmod`
- `models/inherent_block/forecast.py` — `--debug-inhfc`

**Dynamic graph (M007):**
- `models/dynamic_graph_conv/dy_graph_conv.py` — `--debug-dygraph`
- `models/dynamic_graph_conv/utils/distance.py` — `--debug-dist`
- `models/dynamic_graph_conv/utils/mask.py` — `--debug-mask`
- `models/dynamic_graph_conv/utils/normalizer.py` — `--debug-norm`

**Utils (M007-M008):**
- `utils/train.py` — `--debug-train`
- `utils/cal_adj.py` — `--debug-adj`
- `utils/load_data.py` — `--debug-data`
- `utils/log.py` — `--debug-log`
- `dataloader/dataloader.py` — `--debug-loader`

**Datasets (M009):**
- `datasets/raw_data/_gen_speed_common.py` (METR-LA/BAY shared)
- `datasets/raw_data/_gen_flow_common.py` (PEMS04/08 shared)
- `datasets/raw_data/_gen_adj_common.py` (adj builder shared)
- 4× dataset-specific thin wrappers
- `datasets/sensor_graph/describe_adjs.py`

**Entry (M010):**
- `main.py` — `--debug-main`, full flag docstring

**Config (M008):**
- 4× YAML configs (unchanged from upstream)
- 7× `__init__.py` package markers

### Transformation strategy (≈20% delta)

1. Variable renames (`patience→wait_count`, `save_path→checkpoint_path`, etc.)
2. Function signatures reshaped (`set_config(0)→set_config(seed_val=0)`)
3. Structural refactors (shared `_build_mask()`, dispatch-table for adj_type, common modules for dataset generators)
4. Debug instrumentation: 20 independent `--debug-*` flags, each printing tensor shapes, value ranges, norms, NaN counts

### How to use debug flags

```bash
# Full debug storm (every module prints):
python main.py --dataset METR-LA --debug-main --debug-model --debug-trainer \
  --debug-loss --debug-gate --debug-stconv --debug-difblk --debug-inhblk \
  --debug-dygraph --debug-data --debug-adj --debug-train --debug-loader

# Surgical: only inspect the estimation gate and loss:
python main.py --dataset PEMS04 --debug-gate --debug-loss
```
