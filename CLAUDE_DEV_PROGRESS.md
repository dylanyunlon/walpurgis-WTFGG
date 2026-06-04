# Walpurgis-WTFGG: Multi-Claude 开发进度总览

> 每位 Claude 接力完成一段里程碑区间。
> 下一位 Claude 开新对话时，把此文件 + 最新 git log 交给它即可无缝衔接。

---

## 前序开发阶段 (已归档，目录已删除)

- **C++/CUDA 基础设施**: `src/core/`, `src/bridge/`, `src/scheduler/`, `src/bench/`, `src/cuda/`
- **LaTeX 论文**: `walpurgis_reconstructed.tex`
- **D2STGNN 移植 v1-v9**: 经历13位Claude、9个版本的独立改写，已全部合并删除

---

## 当前版本: src/walpurgis/

唯一的 D2STGNN 鲁迅式移植版本。约 4800 行 Python。

### 第一位 Claude — M001-M025: 创建 src/walpurgis/

```
M001-M003  顶层 — __init__.py (全局_dbg调试系统)
           losses.py (Huber+log-cosh混合, quantile_loss)
           model.py (Mish输出, softmax层权重聚合, highway gate)
M004-M006  trainer (自适应p90梯度裁剪, warmup-cosine, sigmoid CL) +
           decouple (SiLU+双头投影+GroupNorm+可学习温度τ, Mish+可学习α)
M007-M010  diffusion_block — dif_model (InstanceNorm2d, GELU, gconv残差skip),
           forecast (cosine退火dropout, 线性插值padding),
           dif_block (3层MLP backcast, sigmoid门控)
M011-M014  dynamic_graph_conv — dy_graph_conv (可学习时间权重),
           distance (3-head多头QK, InstanceNorm1d),
           mask (softplus soft threshold, 对角线清零),
           normalizer (对称归一化, 指数衰减λ^k)
M015-M018  inherent_block — inh_model (RMSNorm, gradient checkpoint, pre-norm),
           forecast (可学习步长衰减, RoPE),
           inh_block (2层MLP+Mish backcast, sigmoid门控)
M019-M021  utils — cal_adj (RBF kernel, k-NN稀疏化, 对称闭包),
           load_data (Tukey fences, sin/cos编码),
           train (确定性CUBLAS, 相对δ EarlyStopping),
           log (JSONL+CSV dual dump)
M022-M023  dataloader (环形wrap, Knuth shuffle, 3-tuple yield) +
           main.py (DataParallel, AMP, ensemble test) + 4×YAML
M024-M025  datasets — 数据生成脚本 + adj生成 + describe_adjs
```

### 第二位 Claude — M026-M040: 算法深化 + 断点快照系统

```
M026-M028  utils/train.py 算法重写:
           - set_config: Knuth乘法散列派生子seed (torch/numpy/random各自独立)
           - EarlyStopping: 线性回归趋势检测(最近8 epoch斜率), patience//2提前触发
           - EarlyStopping: top-k checkpoint追踪
           - data_reshaper: from_numpy零拷贝, float64→32自动降精度, NaN/Inf预检替换
M029-M031  dy_graph_conv.py 算法强化:
           - DropEdge正则: Bernoulli mask随机丢边 + 1/(1-p) rescale保持期望
           - _raw_cos_alpha: cosine混合系数从硬编码0.1改为sigmoid可学习参数
           - edge_scale: 每节点softplus可学习接收增益, 乘在adj列上
M032-M034  model.py DecoupleLayer:
           - gate通过率监控: gated.norm()/history.norm() 能量比
           - dif/inh分支平衡: 两条预测路径能量比, 检测贡献偏斜
M035-M037  trainer.py 诊断增强:
           - 每N步周期性 snapshot_model + gradient_health_check + weight_diff
           - 验证集loss分布诊断: p50/p90/worst百分位
           - 测试集pred vs real分布对比, 系统性偏差检测(|μ_resid|/σ_real)
M038-M040  __init__.py 断点快照系统:
           - snapshot_model(): 全参数+梯度统计, grad_norm降序, nan检测
           - _ActivationTracker + register_activation_hooks(): forward hook
             记录每层 mean/std/zero_frac, check_dead()检测死神经元(>90%零)
           - gradient_health_check(): 爆炸(>100)/消失(<1e-7)/NaN三类检测
           - weight_diff(): 两个state_dict间参数变化量top-k + 冻结参数检测
           - main.py: 训练前第一个batch activation probe + 初始参数快照
```

**产出**: 14 文件改动, +814/-70 行, 总计 ~4809 行
**验证**: 28个算法文件全部 ≥20% 纯算法改动率 (SequenceMatcher, 去掉注释/import/debug行)

### 第三位 Claude — M041-M055: 标签清除 + 算法增强

```
M041-M043  __init__.py 核心诊断算法:
           - _dbg() NaN/Inf 自动告警: 即使tag未开启也触发 ALERT 级别
           - _dbg() tensor 稀疏度追踪: >95% 自动标记 SPARSE(xx.x%)
           - snapshot_model 参数病态自动诊断:
             COLLAPSED_SCALE (std<1e-6) / MEAN_DRIFT (|μ|>3σ) / GRAD_SPIKE (>50)
M044-M047  trainer.py 训练算法:
           - 梯度噪声注入 σ(t) = η/(1+t)^γ, warmup后启动, 自然衰减
           - loss曲面平坦度探测: 每1000步随机扰动θ, 比较Δloss
           - 平坦度历史记录用于收敛质量判断
M048-M049  model.py highway gate:
           - 可学习温度τ控制sigmoid锐度, log_highway_tau参数
           - embedding后接dropout (rate=dropout*0.5)
M050-M051  losses.py 时序一致性约束:
           - temporal_consistency_penalty: 惩罚pred比real更粗糙的时序跳变
           - α * mean(|pred_diff| - |real_diff|)_+, 只罚超出部分
M052-M053  dy_graph_conv.py DropEdge退火:
           - 余弦退火: rate从base衰减到min(base*0.1)
           - 前期高drop帮助正则化探索, 后期低drop精细收敛
M054-M055  inh_model.py:
           - RMSNorm: 可选affine bias + running_rms指数移动平均(momentum=0.1)
           - 注意力head diversity: 各head attention pattern余弦相似度
           - head坍缩检测(所有head关注相同位置)
全局:      清除34文件中81处v10标签 + 4个yaml配置段重命名
```

**产出**: 35 文件改动, +610/-119 行, 总计 ~4,969 行
**验证**: 41个.py文件全部 AST parse 通过, v10/port 残留 = 0

---

## Claude 接力计划

| Claude # | 里程碑 | 内容 | 状态 |
|----------|--------|------|------|
| **第一位** | **M001-M025** | **创建 src/walpurgis/ — 41py+4yaml** | **✅ 已完成** |
| **第二位** | **M026-M040** | **算法深化 + 断点快照系统** | **✅ 已完成** |
| **第三位** | **M041-M055** | **标签清除 + 算法增强(诊断/噪声/平坦度/退火)** | **✅ 已完成** |
| 第四位 | M056-M075 | 待定 | ⏳ 待开发 |
| 第五位 | M076-M095 | 待定 | ⏳ 待开发 |
| 第六位 | M096-M115 | 待定 | ⏳ 待开发 |

---

## 文件统计快照

```
src/walpurgis/               4,969 行 Python (41 .py + 4 .yaml)
src/core/                   ~2,000 行 C++
src/bridge/                 ~1,200 行 C++
src/scheduler/                ~600 行 C++
src/bench/                  ~1,000 行 C++
src/cuda/                     ~500 行 CUDA
upstream/d2stgnn/            2,822 行 参考代码
```

---

## 给下一位 Claude 的接手指南

1. `git log --oneline` 查看完整历史
2. 本文件了解全局进度
3. `upstream/d2stgnn/` = 原始 D2STGNN 参考代码
4. `src/walpurgis/` = 当前唯一移植版本，直接在此迭代
5. 编号规则: `M{三位数}`, 每位 Claude 分配 15-25 个
6. commit 作者: `dylanyunlon <dogechat@163.com>`
7. commit message 格式: `feat: 简述 [Mxxx-Mxxx]`
8. debug: `WALPURGIS_DEBUG=1` 开启全局 _dbg() 打印
9. **你是第几位**: 看上面表格，找到你对应的 ⏳ 行
10. **要求**: 算法级改动(≥20%), 不是改字符串/注释/docstring
