你是第二十二位Claude (Opus 4.6)，接力开发 walpurgis-WTFGG 项目。

## 你的任务: M629-M646 — nightfall GPU训练 + 10维评估基线

### 项目上下文
仓库: https://github.com/dylanyunlon/walpurgis-WTFGG.git
这是D2STGNN时空图神经网络的"Nightfall"变体，第二十一位Claude已完成训练pipeline。

### 你需要做的事

1. **克隆主仓库**:
```bash
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
```

2. **验证pipeline工作**:
```bash
EPOCHS=3 bash run_nightfall.sh
```

3. **写10维评估脚本**: 创建 `eval_nightfall.py`
   - 加载 output/D2STGNN_SYNTH.pt 训练好的模型
   - 在test集上计算: MAE@1h/3h/6h, RMSE@1h/3h/6h, MAPE@1h/3h/6h, 模型参数量
   - 输出10维基线值(JSON格式)到 output/nightfall_eval_results.json
   - 对比 walpurgis_nightfall vs upstream/d2stgnn (如可能)

4. **如果有GPU**: 运行 DEVICE=cuda EPOCHS=80 bash run_nightfall.sh
   - 如果无GPU: 用SYNTH数据跑10个epoch也可以

5. **生成对比表**: 创建 `compare_nightfall_upstream.py`
   - 加载eval结果 + bench/sota.json
   - 生成LaTeX格式的对比表
   - 输出到 output/comparison_table.tex

### 关键规则:
- **不开新分支,不加后缀** — 直接改main
- **git push直接到main** — 作者: dylanyunlon <dogechat@163.com>
- **GitHub token**: [见git remote config或由用户提供]

### 接力信息
- 你是第二十二位Claude, 里程碑M629-M646
- 前置: 第二十一位Claude已完成M611-M628 (训练pipeline)
  - train_nightfall.py: 完整训练入口
  - run_nightfall.sh: bash pipeline
  - datasets/SYNTH: 10节点合成数据集
  - SYNTH 3epoch验证: MAE 26.27→13.60, 374K参数
- 后续: 第二十三位会做 nightfall vs walking vs upstream对比 (M647-M664)

如果太长被截断,发送"Continue"继续执行。
