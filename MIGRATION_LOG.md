
## migrate 2bb2e1a: resolve merge conflict

- **Upstream commit**: 2bb2e1a48767fcd4aa3f05ab13503dff6d257c60 (cugraph-gnn, commit #168/452)
- **Commit message**: `resolve merge conflict`
- **Author**: Alexandria Barghi <abarghi@nvidia.com>
- **Date**: 2025-03-21
- **Parents**: 5cfb2e8 (CUDA 12.6 / PyTorch cu126 升级链) × 2d545b9 (TensorDictFeatureStore 废弃)
- **Upstream diff 摘要**: 8 files changed, 28 insertions(+), 22 deletions(-)

  | 文件 | 处置 | 原因 |
  |------|------|------|
  | ci/test_wheel_cugraph-pyg.sh | SKIP | RAPIDS wheel CI 脚本 |
  | conda/environments/all_cuda-126_arch-x86_64.yaml | SKIP | conda 环境矩阵 |
  | dependencies.yaml (cuda 矩阵) | SKIP | RAPIDS 构建矩阵 |
  | dependencies.yaml (depends_on_mkl) | **迁移** | MKL 显式依赖语义（本次独有） |
  | dependencies.yaml (tensordict <=0.6.2) | **迁移** | 版本上界 pin 工程决策（本次独有） |
  | python/*/conda/*.yaml | SKIP | conda dev 环境 |
  | python/*/pyproject.toml | SKIP | 上游包构建配置 |
  | python/cugraph-pyg/cugraph_pyg/data/__init__.py | SKIP | 已由 feature_store_deprecation.py 覆盖 |

- **迁移位置**: `src/walpurgis/core/merge_conflict_resolve.py`（新增，~340 行）

- **鲁迅拿法改写（≥20%）**:
  合并冲突修复如同《野草》里那篇《过客》——走到这里，是两条路合成一条，
  过客不知身后是哪条先走，只知道脚下这一步必须踏实。
  上游 Alexandria 只留一句「resolve merge conflict」，把两个隐性决策埋进了 YAML diff：
  ①新增 `depends_on_mkl`（MKL 显式化，防 conda 不确定性）；
  ②新增 tensordict `<=0.6.2` 上界（防 0.7.x breaking change 破坏 CI）。
  若无人记录，半年后的维护者只能看着版本号猜缘由。
  Walpurgis 将这两个隐性决策提炼为可程序化查询的结构：

  1. **`BranchResolutionStrategy` (Enum)** — 枚举 KEEP_BOTH / PREFER_NEWER / MANUAL_MERGE / EMPTY_DIFF，将上游裸 git 操作外显为策略分类
  2. **`MergeConflictRecord` (frozen dataclass)** — 封装合并元数据：hash、两亲本、策略、files/insertions/deletions、affected_sections、python_diff_files；`is_empty_python_diff()` 精确标记「已前序覆盖」
  3. **`TensorDictVersionPin` (frozen dataclass)** — 将 `>=0.1.2,<=0.6.2` 建模为结构化约束对象，`upper_bound_rationale()` 文档化 0.7.x batch_size API breaking change，`is_compatible(version_str)` 运行时检测，`as_pip_spec()` / `as_conda_spec()` 双格式输出
  4. **`MklDependencyPolicy` (frozen dataclass)** — 记录 depends_on_mkl 新增的工程原因（PyTorch x86_64 MKL 隐性依赖，conda 不确定性防御），`risk_if_missing()` 量化缺失风险（3-5× 性能损失 + symbol conflict）
  5. **`MergeConflictAudit`** — `audit_coverage()` 验证 Python 变更已覆盖，`audit_pin_consistency()` 验证 tensordict 历史（2bb2e1a pin → 78128d9 删除），`audit_mkl_recorded()` 验证 MKL 语义已记录，`summary()` 输出三维审计报告

  全链路 `WALPURGIS_DEBUG=1` 断点 **10 处**：MODULE_LOAD（×2）、MERGE_RECORD_INIT、TENSORDICT_PIN_INIT、MKL_POLICY_INIT、AUDIT_INIT、PIN_COMPAT_CHECK（×5 版本）、AUDIT_COVERAGE_CHECK、AUDIT_PIN_CHECK、AUDIT_MKL_CHECK、SELF_CHECK（×5 步骤）。`self_check()` 5 项断言全部通过（ALL PASS）。

- **三维度审查（Knuth）**:
  - **正确性**: 8 个文件逐行审查，迁移决策矩阵完整；`TensorDictVersionPin.is_compatible()` 经 5 组边界测试（下界、上界、中间值、超上界、低于下界）全部断言通过；`MergeConflictAudit.audit_*()` 三项检查均 PASS；Python 源码变更（data/__init__.py）已确认由 feature_store_deprecation.py 完整覆盖，无重复迁移。
  - **性能**: 纯数据结构与字符串操作，无 I/O，无循环热路径；`is_compatible()` 为 O(1) 正则 + tuple 比较；`audit_coverage()` 为 O(n) 字典遍历（n=1）。
  - **可读性**: 上游 commit 是「无声的合并」，两个隐性决策（MKL 显式化、tensordict 上界）在 git log 中无法检索。Walpurgis 将其结构化，`MergeConflictAudit.summary()` 输出完整审计报告，比 git diff 更具可查询性与可维护性。

---
