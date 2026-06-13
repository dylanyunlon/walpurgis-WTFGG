# 子Claude批量迁移任务 — 剩余26个cugraph-gnn commit

## 你的任务
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git clone https://github.com/rapidsai/cugraph-gnn.git /tmp/cugraph-gnn
```

对你分配到的每个commit:
1. `cd /tmp/cugraph-gnn && git show <hash> --stat` 看改了什么文件
2. `git show <hash> -- "*.py" "*.cu" "*.cpp" "*.hpp"` 逐行读diff
3. 在 walpurgis-WTFGG 中找到对应的迁移位置
4. 鲁迅拿法改写20%: 加结构体/枚举/断点诊断, 不是原样复制
5. 每个commit单独一次 `git commit -m "migrate <hash>: <msg>"`

## 铁律
- 作者: dylanyunlon <dogechat@163.com>
- 不开新分支, 不用v2/port后缀
- 改的是算法, 不改字符串docstring
- push: `git remote set-url origin https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git && git push origin main`

---

## 子Claude-A (5 commits): wholegraph更新 + 基础清理
1. ecc22bf — update code (1 file)
2. b9db217 — Drop Python 3.9 support (1 file)  
3. 8f8b71f — Update flake8 to 7.1.1 (#42) (7 files)
4. 37f8629 — fix import order (4 files)
5. 31ee98f — add PR CI for cugraph-pyg and cugraph-dgl (#59) (18 files)

## 子Claude-B (5 commits): 核心算法
6. a9ab8b4 — [FEA] Support Heterogeneous Sampling in cuGraph-PyG (#82) (11 files)
7. d38b832 — remove dependency on cugraph-ops (#99) (38 files)
8. 0e88280 — Support PyG 2.6 in cuGraph-PyG (#114) (12 files)
9. 87455cf — Remove Build Directory (#107) (31 files)
10. e90d1e6 — Fix of create_node_classification_datasets (#128) (3 files)

## 子Claude-C (5 commits): Dask移除 + DGL清理
11. 431801c — Deprecate the Dask API in cuGraph-PyG (#118) (4 files)
12. 1e91ed7 — Remove Dask API from cuGraph-PyG (#166) (13 files)
13. 05b5791 — cugraph-pyg: remove Dask dependencies (#168) (2 files)
14. 456d5a2 — add deprecation warnings for DGL classes (2 files)
15. adb4006 — fix circular import (1 file)

## 子Claude-D (5 commits): WholeGraph集成 + 示例
16. feffb39 — fix import (1 file)
17. 1b2fce2 — fix bad import (1 file)
18. a57912c — fix references to dask data loader (2 files)
19. e01196b — [IMP] Make WholeGraph a Hard Dependency (#172) (7 files)
20. 70c33af — [IMP] Remove SG and SNMG Examples (#171) (5 files)

## 子Claude-E (6 commits): 依赖解耦 + bug修复
21. 2dd02f9 — feat: add libwholegraph wheel (#182) (4 files)
22. b10f279 — Remove cugraph Python library as a dependency (#271) (29 files)
23. b860220 — [BUG] Fix input type in Taobao example (#301) (1 file)
24. 330b135 — ensure torch CUDA wheels in CI (#425) (36 files)
25. 659a0e1 — [BUG] Fix hashing/node id in disjoint sampling test (#474) (1 file)
