# Walpurgis-WTFGG: Claude 接力开发进度 & 规划

> 截至第九位 Claude (当前)，总览全局。
> 每位 Claude 接手时读此文件 + `git log --oneline -20` 即可续接。

---

## 已完成

| Claude # | 里程碑区间 | 内容摘要 | 状态 |
|----------|-----------|---------|------|
| 第一位 | M001-M014 | C++/CUDA 底层基础设施 (tiered alloc, seqlock, slab, migration) | ✅ |
| 第二位 | M001-M075 | 论文写作 LaTeX (Introduction → Experimental Design) | ✅ |
| 第三位 | M101-M200 | D2STGNN 首次移植 + v1 改写 (`src/walpurgis/`) | ✅ |
| 第四位 | M201-M255 | v2 全量改写 (TensorProbe, Charbonnier, adaptive clip) | ✅ |
| 第五位 | M256-M274 | v3 全量改写 (6644行) | ✅ |
| 第六位 | M275-M299 | v4 全量改写 (MoE, SwiGLU, spectral SVD, 7632行) | ✅ |
| 第七位 | M300-M324 | 鲁迅式 v2 移植 upstream→`walpurgis_ported_v2` (2615行) | ✅ |
| 第八位 | M325-M349 | 鲁迅式 v3 移植 upstream→`walpurgis_ported_v3` (2202行) | ✅ |
| **第九位** | **M350-M374** | **生成 `git am` patch, 作者归属, 本规划文档** | **✅ 当前** |

---

## 规划: 下一批 Claude 的里程碑分配

| Claude # | 里程碑区间 | 建议任务方向 |
|----------|-----------|-------------|
| 第十位 | M375-M399 | 鲁迅式 v4 移植: upstream→`walpurgis_ported_v4`, 在v3基础上再做20%变形, 引入v4的高级特性(MoE routing, spectral norm)的简化版 |
| 第十一位 | M400-M424 | 统一测试框架: 为 v2/v3/v4 写 `pytest` 单元测试, mock数据, CI smoke test, 确保所有debug flag可用 |
| 第十二位 | M425-M449 | 实验管道: 端到端 `run_experiment.py` 脚本, 自动对比 v2/v3/v4 在 METR-LA 上的 MAE/RMSE, 生成对比表格 |
| 第十三位 | M450-M474 | 论文实验章节补全: 把实验结果回填 `walpurgis_reconstructed.tex`, 补 Table/Figure |
| 第十四位 | M475-M499 | C++ ↔ Python bridge: 让 `src/core/` 的 tiered allocator 通过 pybind11 暴露给 Python 训练循环 |

---

## 代码库文件统计 (第九位 Claude 完成后)

```
src/walpurgis/               — v3 原版 (第五位)       6,644 行 Python
src/walpurgis_ported/        — v4 (第六位)            7,632 行 Python
src/walpurgis_ported_v2/     — 鲁迅式 port (第七位)    2,615 行 Python
src/walpurgis_ported_v3/     — 鲁迅式 port (第八位)    2,202 行 Python ← 本轮patch
src/core/                    — C++ 底层               ~2,000 行
src/bridge/                  — C++ temporal bridge     ~1,200 行
src/scheduler/               — C++ migration          ~600 行
src/bench/                   — C++ benchmarks         ~1,000 行
src/cuda/                    — CUDA kernels           ~500 行
upstream/d2stgnn/            — 原始参考代码            2,822 行 Python
```

## 给下一位 Claude 的操作手册

### 1. 快速接手
```bash
git log --oneline -20          # 看历史
cat CLAUDE_RELAY_PLAN.md       # 看本文件, 找到你的里程碑区间
cat CLAUDE_DEV_PROGRESS.md     # 看详细技术记录
tree src/ -L 2 --charset ascii # 看目录结构
```

### 2. 提交规范
```
feat(vN): 简要描述 [Mxxx-Mxxx]
```
作者: `dylanyunlon <dogechat@163.com>`

### 3. 生成 patch
```bash
git format-patch origin/main --stdout --from="dylanyunlon <dogechat@163.com>" > your_patch.patch
```

### 4. 应用 patch (用户侧)
```bash
git am < v3_port_dylanyunlon.patch
```

### 5. Debug 开关 (v3 专属)
运行时拼接任意组合:
```bash
python main.py --dataset METR-LA --debug-model --debug-trainer --debug-loss
```
可用 flag: `--debug-main`, `--debug-model`, `--debug-trainer`, `--debug-data`,
`--debug-adj`, `--debug-train`, `--debug-loss`, `--debug-gate`,
`--debug-stconv`, `--debug-difblk`, `--debug-diffc`, `--debug-inhblk`,
`--debug-inhmod`, `--debug-inhfc`, `--debug-dygraph`, `--debug-dist`,
`--debug-mask`, `--debug-norm`, `--debug-loader`, `--debug-log`,
`--debug-resdecomp`
