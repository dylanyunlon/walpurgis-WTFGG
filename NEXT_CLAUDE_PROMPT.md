你是第二十二位Claude (Opus 4.6)，接力开发 walpurgis-WTFGG 项目。

## 你的任务: M647-M664 — nightfall vs walking vs upstream对比

### 项目上下文
仓库: https://github.com/dylanyunlon/walpurgis-WTFGG.git
这是D2STGNN时空图神经网络的"Nightfall"变体。

### 背景
第二十一位Claude已完成:
- train_nightfall.py: 完整训练入口 (SYNTH 3epoch验证通过)
- eval_nightfall.py: 10维评估脚本 (MAE/RMSE/MAPE @15/30/60min + 参数量)
- output/nightfall_eval_results.json: 评估结果JSON
- output/comparison_table.tex: LaTeX对比表 (vs 9个SOTA)

### 你需要做的事

1. **克隆主仓库**:
```bash
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
```

2. **验证已有pipeline**:
```bash
EPOCHS=3 bash run_nightfall.sh   # 确认训练能跑
python eval_nightfall.py --dataset SYNTH  # 确认eval能跑
```

3. **建立对比分析框架**: 创建 `compare_variants.py`
   - 对比 walpurgis_nightfall vs walpurgis_walking vs upstream/d2stgnn
   - 三路对比: 架构差异/训练差异/数据处理差异
   - 生成差异分析表 (markdown + LaTeX)
   - 输出到 output/variant_comparison.json + output/variant_comparison.tex

4. **生成10维×3变体差异矩阵**: 
   - 10维: 模型架构/损失函数/优化器/LR调度/数据增强/归一化/梯度处理/激活函数/注意力机制/残差结构
   - 3变体: upstream baseline / walpurgis_walking / walpurgis_nightfall
   - 每格: 具体改动说明
   - 输出: output/variant_matrix.tex

5. **如果有GPU**: 用METR-LA跑几个epoch对比真实指标
   如果无GPU: 用SYNTH数据对比三个变体的指标

### 关键规则:
- **不开新分支,不加后缀** — 直接改main
- **git push直接到main** — 作者: dylanyunlon <dogechat@163.com>
- **GitHub token**: [由用户在运行时提供]

### 接力信息
- 你是第二十二位Claude, 里程碑M647-M664
- 前置: 第二十一位Claude已完成M611-M646 (训练pipeline + 10维评估)
- 后续: 第二十三位会做 ablation study (M665-M682)

如果太长被截断,发送"Continue"继续执行。
