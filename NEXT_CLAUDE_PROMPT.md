你是第二十一位Claude (Opus 4.6)，接力开发 walpurgis-WTFGG 项目。

## 你的任务: M611-M628 — nightfall import修复 + smoke test

### 项目上下文
仓库: https://github.com/dylanyunlon/walpurgis-WTFGG.git
目标目录: src/walpurgis_nightfall/
这是D2STGNN时空图神经网络的"Nightfall"变体，由第二十位Claude完成了全部45文件(2516行)的移植。

### 你需要做的事

1. **先拉取cookie**: 
```bash
git clone https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config
```

2. **克隆主仓库**:
```bash
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
```

3. **修复import路径**: src/walpurgis_nightfall/ 中的相对import可能有问题，因为是从upstream机械移植的。需要检查并修复所有import使其正确工作：
   - `from models.xxx` → `from walpurgis_nightfall.models.xxx` 或正确的相对路径
   - `from utils.xxx` → `from walpurgis_nightfall.utils.xxx`
   - `from dataloader` → `from walpurgis_nightfall.dataloader`
   - 确保 `walpurgis_nightfall` 可以作为package被import

4. **写smoke test**: 创建 `tests/test_nightfall_smoke.py`
   - NIGHTFALL_DEBUG=1 环境下import所有模块
   - 用随机tensor做一次forward pass
   - 验证输出shape正确
   - 验证调试打印正常工作

5. **写合成数据生成**: `src/walpurgis_nightfall/generate_synth_data.py`
   - 生成小规模假数据 (10节点, 100时间步)
   - 可以跑通完整训练loop

### 关键规则 (必须遵守):
- **改算法,不改字符串/docstring/str_replace那种** — 如果要修复import,那就修复import,但不要改任何算法逻辑
- **不开新分支,不加port/v2/v3后缀** — 直接改src/walpurgis_nightfall/下的文件
- **git push直接到main** — 作者: dylanyunlon <dogechat@163.com>
- **GitHub token**: [TOKEN已移除,见git remote config]
- **完成后更新CLAUDE_RELAY_PLAN.md** — 把你的M611-M628标记为✅

### 接力信息
- 你是第二十一位Claude, 里程碑M611-M628
- 前置: 第二十位Claude已完成M593-M610 (nightfall移植)
- 后续: 第二十二位会做GPU训练(M629-M646)

如果太长被截断,发送"Continue"继续执行。
