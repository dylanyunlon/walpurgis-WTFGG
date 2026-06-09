# Walpurgis 子Claude任务派发

你是Walpurgis项目的子模型执行者。请按以下步骤操作:

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
git log --oneline -10
cat MULTI_CLAUDE_PLAN.md
```

## 查看你的任务
查看 tasks/ 目录下对应你的里程碑编号的文件。

## 铁律
1. **不开新分支** — 所有改动直接在main上
2. **不用v2/v3/port/alt/bak等后缀** — 文件名保持原样
3. **改的是算法** — 不改字符串/docstring/str_replace表面功夫
4. **作者信息**: dylanyunlon <dogechat@163.com>
5. **push方式**:
   ```bash
   git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/walpurgis-WTFGG.git
   git add -A && git commit -m "Claude-N MXXX: 描述" && git push origin main
   ```

## 诊断重点
运行实验时关注以下诊断输出:
- `[DIAG step=N]` — 每200步的完整模型状态
- `per_horizon_MAE` — 每50步的逐horizon MAE
- `adaptive_emb_gate` — 自适应嵌入门控值（应在0.3-0.7）
- `uncertainty_mean` — PSD不确定性均值（epoch>20后生效）
- `depth_gates` — 各层深度门激活值
- `cascade_weights` — 级联残差权重分布

## 当前算法状态 (Phase 2 M301-M325 已完成)
- 自适应时空嵌入: ✅ 从STAEformer移植
- PSD不确定性损失: ✅ 从TITAN移植
- 输出头时序注意力: ✅ TemporalCrossAttention
- 目标: METR-LA MAE < 2.85 (当前最佳: 2.93)
