# Claude-2 Phase 4 任务 (M626-M650)

## 角色: 服务器 METR-LA 200ep 完整实验 (worst-avoidance首战)

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
tree -L 2 --charset ascii
git log --oneline -5
```

## 任务
在 ags1 服务器上运行 METR-LA 完整实验:
```bash
cd /data/jiacheng/system/cache/temp/atc2026/walpurgis-WTFGG
git fetch origin && git reset --hard origin/main
DATASET=METR-LA GPU=2 EPOCHS=200 bash experiments/run_experiment.sh
```

## 重点关注的诊断输出
1. `worst_horizons` — 每batch最差的3个horizon和它们的MAE
2. `spatial_attn: gate=xxx entropy=xxx` — 空间注意力是否在学习
3. `dif_inh_gate_bias` — dif/inh融合偏好方向
4. `grad_top3/bot3` — 梯度健康状况
5. per-horizon MAE 的 h12/h1 比值 — 目标 < 1.45

## 判据
- MAE_avg12 < 2.85 → SOTA突破! 立即push结果
- MAE_avg12 在 2.85-2.88 → 记录, 尝试调整 horizon_scale=0.18
- MAE_avg12 > 2.90 → 检查诊断日志, 回退worst_avoidance参数

## 铁律
- 不开新分支, 不用后缀
- 作者: dylanyunlon <dogechat@163.com>
- 改的是算法, 不改字符串
