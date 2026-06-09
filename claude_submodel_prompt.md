# 子模型任务: Walpurgis-WTFGG 算法改进 + 断点调试 + 实验运行

## 背景 (来自用户第一轮prompt)

用户要求:
1. 看看这个项目关于代码实验sota与否、目前的数据是否能写入tex的问题
2. 鲁迅那样的拿法 — 在mv的基础上，动态修改算法的20%的内容就能让实验很强了
3. 注意多写一点关于断点调试（或者print当前所有数据、结构体状态）的内容
4. 让我们在运行实验的时候能像现实世界开发一样得到反馈
5. 需要实际执行实验运行（SYNTH数据集，CPU模式，模拟GPU环境）
6. 不允许v10、port等后缀，改的是算法
7. 给出git am格式patch，作者 dylanyunlon <dogechat@163.com>

## 项目现状

仓库: github.com/dylanyunlon/walpurgis-WTFGG
当前最佳MAE: 3.08 (METR-LA), 目标 <2.85, SOTA=2.88 (TITAN)
问题: OOM崩溃(H100 96GB都不够), CL发散, summary.json全是N/A

## 你的任务 (Claude子模型)

### 核心: 对 model.py + trainer.py + losses.py 做算法级改进(~20%)

改进方向:
1. **内存效率**: 减少gconv中的matmul内存峰值 — 分块图卷积 (chunked graph conv)
2. **训练稳定性**: CL sigmoid ramp 的warm-up阶段增加渐进噪声衰减
3. **Forecast精度**: 在inherent forecast中引入残差频率注入 (residual frequency injection)
4. **断点调试增强**: 在每个关键数据流节点加入完整的结构体状态dump

### 具体改动要求

#### 1. dif_model.py — 分块图卷积
在 `gconv` 方法中，将大矩阵乘法分成chunk执行:
```python
def gconv(self, support, X_k, X_0):
    out = [X_0]
    for graph in support:
        if len(graph.shape) == 2:
            pass
        else:
            graph = graph.unsqueeze(1)
        # 分块执行减少内存峰值
        chunk_size = max(X_k.shape[2] // 4, 1)
        H_k_chunks = []
        for i in range(0, X_k.shape[2], chunk_size):
            g_chunk = graph[..., i:i+chunk_size, :]  if len(graph.shape) > 2 else graph[i:i+chunk_size, :]
            x_chunk = X_k
            H_k_chunks.append(torch.matmul(g_chunk, x_chunk) if len(graph.shape) == 2 else torch.matmul(graph, X_k))
        H_k = torch.cat(H_k_chunks, dim=-2) if len(H_k_chunks) > 1 else H_k_chunks[0]
        out.append(H_k)
    out = torch.cat(out, dim=-1)
    out = self.gcn_updt(out)
    out = self.dropout(out)
    return out
```

#### 2. trainer.py — 增强断点调试
在train()方法的forward/loss/backward每个阶段都加入详细的状态dump:
- forward后: 打印output shape、range、NaN检测
- loss计算后: 打印各loss分量、梯度范数预估
- backward后: 打印实际梯度范数、参数更新量

#### 3. losses.py — 自适应温度horizon loss
让LogCoshHorizonLoss的温度参数根据训练进度自适应:
- 早期(epoch<20): 温度高(平滑梯度)
- 中期(20-80): 温度中等
- 后期(>80): 温度低(精确匹配)

#### 4. model.py — cascade_proj引入frequency残差
在DecoupleLayer的cascade_proj后加入frequency-domain残差:
```python
# 在cascade_proj后
cascade_fft = torch.fft.rfft(cascade_feat.squeeze(1), dim=-1)
cascade_fft_mag = cascade_fft.abs()
freq_residual = torch.fft.irfft(cascade_fft, n=cascade_feat.shape[-1], dim=-1)
cascade_residual = cascade_residual + 0.1 * self.freq_gate * freq_residual.unsqueeze(1).expand_as(cascade_residual)
```

### 输出格式

请生成一个完整的 git format-patch 格式文件，包含所有改动。格式:
```
From: dylanyunlon <dogechat@163.com>
Date: <current>
Subject: [PATCH] Claude-5 M101-M110: algorithm refinement — chunked gconv + adaptive temperature + frequency cascade residual + enhanced diagnostics

<commit message body>
---
 <file diffs>
```

注意:
- 只改算法代码，不改字符串/docstring
- 确保所有import正确
- 断点调试用 `_dbg()` 和 `dump_struct_state()` (已在 `__init__.py` 中定义)
- 保持与现有代码风格一致
