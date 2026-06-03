# Walpurgis-WTFGG: Multi-Claude 开发进度总览

> 每位 Claude 接力完成一段里程碑区间。
> 下一位 Claude 开新对话时，把此文件 + 最新 git log 交给它即可无缝衔接。

---

## 前序开发阶段 (已归档，目录已删除)

以下为历史记录，相关的 `src/walpurgis_ported_v2..v9` 等目录已全部删除合并。

- **C++/CUDA 基础设施**: `src/core/`, `src/bridge/`, `src/scheduler/`, `src/bench/`, `src/cuda/`
- **LaTeX 论文**: `walpurgis_reconstructed.tex`
- **D2STGNN 移植 v1-v9**: 经历13位Claude、9个版本的独立改写，已全部合并删除
- v2/v3 的公共数据模块 (`_gen_adj_common.py`, `_gen_flow_common.py`, `_gen_speed_common.py`) 已合并入当前版本

---

## 当前版本: src/walpurgis/

唯一的 D2STGNN 鲁迅式移植版本。41 个 .py 文件 + 4 个 .yaml 配置，约 4065 行。

### 第一位 Claude — M001-M025: 创建 src/walpurgis/

```
M001-M003  顶层 — __init__.py (全局_dbg调试系统, WALPURGIS_DEBUG环境变量)
           losses.py (Huber+log-cosh 70/30混合δ=5, MAPE floor 5e-6, quantile_loss)
           model.py (Mish输出, softmax层权重+温度聚合, highway gate, kaiming init)
M004-M006  trainer (自适应p90梯度裁剪, warmup-cosine调度, sigmoid CL ramp) +
           decouple (estimation_gate: SiLU+双头投影+GroupNorm4组+可学习温度τ,
           residual_decomp: Mish+可学习α sigmoid+Dropout0.05+LN)
M007-M010  diffusion_block — dif_model (InstanceNorm2d, GELU, gconv残差skip),
           forecast (cosine退火dropout, 线性插值padding, FC前LayerNorm),
           dif_block (3层MLP backcast, sigmoid门控, 0.1*history skip)
M011-M014  dynamic_graph_conv — dy_graph_conv (可学习时间权重softmax, cosine辅助),
           distance (3-head多头QK, InstanceNorm1d, attn dropout0.1),
           mask (softplus+温度soft threshold, 对角线清零),
           normalizer (对称D^{-1/2}AD^{-1/2}, 指数衰减λ^k(0.8), 可学习eps)
M015-M018  inherent_block — inh_model (RMSNorm, gradient checkpoint, pre-norm transformer),
           forecast (可学习步长衰减exp(-γ·step), 简化RoPE),
           inh_block (RoPE位置编码, 2层MLP+Mish backcast, sigmoid门控)
M019-M021  utils — cal_adj (RBF kernel, k-NN(15)稀疏化, 对称闭包, epsilon-smooth),
           load_data (Tukey fences异常剔除, sin/cos周期编码, adj预处理链),
           train (确定性CUBLAS, 相对δ_rel EarlyStopping, SHA256校验),
           log (JSONL+CSV dual dump, git-hash目录)
M022-M023  dataloader (环形wrap, Fisher-Yates shuffle, 3-tuple yield, prefetch buffer) +
           main.py (DataParallel多GPU, AMP GradScaler, ensemble test) +
           4×YAML (dropout 0.1→0.08, v10_adj_preprocess段, v10_training段)
M024-M025  datasets — 数据生成脚本 (stride跳步, cyclic编码, 按周对齐, stats JSON) +
           adj生成 (距离→RBF连续权重, kNN稀疏化, 自适应阈值剪枝, 加权自环) +
           describe_adjs (密度/权重分布/BFS连通/度5数概要/非对称性检测) +
           合并v3公共模块 (_gen_adj_common, _gen_flow_common, _gen_speed_common)
```

**产出**: `src/walpurgis/` — 41 .py + 4 .yaml, 4065行
**改写策略**: upstream骨架 + ≥20%实质算法改动 + 全局_dbg()断点系统

---

## Claude 接力计划

| Claude # | 里程碑 | 内容 | 状态 |
|----------|--------|------|------|
| **第一位** | **M001-M025** | **创建 src/walpurgis/ — 41py+4yaml, 4065行** | **✅ 已完成** |
| 第二位 | M026-M050 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第三位 | M051-M075 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第四位 | M076-M100 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第五位 | M101-M125 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |
| 第六位 | M126-M150 | 在 src/walpurgis/ 上迭代改写 | ⏳ 待开发 |

---

## 文件统计快照

```
src/walpurgis/               4,065 行 Python (41 .py + 4 .yaml, 当前唯一版本)
src/core/                   ~2,000 行 C++ (tiered allocator, seqlock, slab)
src/bridge/                 ~1,200 行 C++ (temporal bridge)
src/scheduler/                ~600 行 C++ (migration scheduler)
src/bench/                  ~1,000 行 C++ (benchmarks)
src/cuda/                     ~500 行 CUDA (device kernels)
walpurgis_reconstructed.tex  ~32KB LaTeX (full paper)
```

---

## 给下一位 Claude 的接手指南

1. `git log --oneline` 查看完整历史
2. 本文件 (`CLAUDE_DEV_PROGRESS.md`) 了解全局进度
3. `upstream/d2stgnn/` = 原始 D2STGNN 参考代码
4. `src/walpurgis/` = 当前唯一的移植版本，直接在此目录迭代
5. 每个 `.py` 文件头部的 docstring 记录了该文件的算法变更
6. 编号规则: `M{三位数}`, 每位 Claude 分配连续 25 个
7. commit 作者: `dylanyunlon <dogechat@163.com>`
8. commit message 格式: `feat: 简述 [Mxxx-Mxxx]`
9. debug: 设置环境变量 `WALPURGIS_DEBUG=1` 开启全局 _dbg() 打印
10. **你是第几位**: 看上面表格，找到你对应的 ⏳ 行
11. **要求**: 算法级改动(≥20%), 不是改字符串/注释/docstring
12. **重要**: 不要创建新目录，直接修改 src/walpurgis/
