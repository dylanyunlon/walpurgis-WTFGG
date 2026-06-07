# Claude 接力开发计划 (Relay Plan)

## 总览

| Claude # | 里程碑 | 变体 | 状态 |
|----------|--------|------|------|
| 第一位Claude | M694-M724 | walpurgis_eclipse | ✅ 已完成并push |
| 第二位Claude | M725-M749 | walpurgis_aurora | ⏳ 待派发 |
| 第三位Claude | M750-M774 | walpurgis_tempest | ⏳ 待派发 |
| 第四位Claude | M775-M799 | 全变体对比实验 | ⏳ 待派发 |
| 第五位Claude | M800-M824 | 性能优化+论文更新 | ⏳ 待派发 |
| 第六位Claude | M825-M849 | 集成测试+部署 | ⏳ 待派发 |

## 已完成变体

### walpurgis_eclipse (M700-M724) ✅
- 47文件, 2122行, 20+算法改写
- Tukey biweight loss, EMA aggregation, GELU+SpectralNorm
- AdamW+ReduceLROnPlateau, adaptive p95 clip
- Ornstein-Uhlenbeck synth data, Fisher-Yates shuffle
- 验证: MAE=12.65 RMSE=17.89 MAPE=17.87%

## 待开发变体

### walpurgis_aurora (M725-M749) ⏳
建议算法方向:
- Loss: Huber loss + uncertainty weighting
- Gate: Mixture-of-Experts gate with top-k routing
- Normalization: GroupNorm throughout
- Attention: Linear attention (O(N) instead of O(N²))
- Optimizer: LAMB / Lookahead
- Graph: Graph attention with edge features

### walpurgis_tempest (M750-M774) ⏳
建议算法方向:
- Loss: Focal loss adapted for regression
- Architecture: Dilated causal convolutions in diffusion block
- Normalization: Weight standardization + GN
- PE: Rotary positional encoding (RoPE)
- Scheduler: Cosine annealing with warm restarts
- Graph: Spectral graph convolution

## 子Claude派发说明

1. 从 dylanyunlon/claude-hk-config 拉取最新cookie
2. 通过 claude_hk_chat.sh 或直接curl创建conversation
3. 发送 SUB_CLAUDE_PROMPT.md (修改变体名+算法描述)
4. 如果响应截断, 发送 "Continue" 继续
5. 完成后git push到main

## 关键规则
- 不开新分支, 直接main
- 不加v10/port/v2/v3后缀
- 改的是算法, 不是字符串/docstring
- 每个文件20%+实质性算法改动
- 全模块debug instrumentation
- git author: dylanyunlon <dogechat@163.com>
