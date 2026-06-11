# Claude-11 任务 (M576-M600)

你是Walpurgis项目的子模型执行者。

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git clone https://github.com/rapidsai/cugraph-gnn.git /tmp/cugraph-gnn
cat tasks/phase3/PHASE3_PLAN.md
```

## 你的commit范围

```
4f250a5 update versions to 24.12
429dbc1 add ops bot
cb6a81f add copy pr bot
7e3182c introduce minimal CI for PRs
0ea17b3 add alpha specs, pre-commit hook
2e0c143 add comment to dependencies.yaml
93e61a7 Merge PR #57
bf64914 add full CI for wholegraph (26 files)
26c7d07 remove docs support in build.sh
712255d remove ci/test_wheel.sh
25d1b55 Merge PR #60
31ee98f add PR CI for cugraph-pyg and cugraph-dgl (43 files)
f7ab898 add nightly builds/tests (6 files)
d260ccb add notebook tests, build.sh args (17 files)
df5bdc4 update wholegraph (64 files)
5a17bbe start publishing packages
2dd3001 enforce wheel size limits
16e614c remove versioning workaround
d56dd66 DOC v25.02 Updates
e1e32bc fix devcontainer builds (3 files)
36c312c Merge branch-24.12
aa099e4 Relax PyTorch upper bound
986cc76 Merge PR #76
2776772 [Feature] Add gather/scatter 1D tensor (3 files)
c5cc3e7 Merge PR #77
4807986 [Bugfix] Dynamic load NVML symbols (already migrated!)
046b2f2 Merge PR #78
23cdecd Add breaking change workflow trigger
01abe44 Require approval for draft PRs
466b5b9 [Bugfix] Add stream sync before scatter (already migrated!)
136e44b Merge PR #83
b3dec8c skip conda on arm64
42c16fe add devcontainers (12 files)
7ec8ace Disable RockyLinux DGL Tests
ce6610d Merge PR #90
ca3ca80 skip CUDA 11.4 conda-python-tests
fa6f125 Update Changelog
b0e0222 merge branch-24.12 into branch-25.02
77206de Merge PR #95
a9ab8b4 [FEA] Support Heterogeneous Sampling (13 files) ← 高价值
```

## 重点commit
- df5bdc4 (64 files): wholegraph更新, 深入C++源码
- 2776772: gather/scatter 1D tensor支持 — 与dist_tensor直接相关
- a9ab8b4 (13 files): 异构图采样支持 — Walpurgis核心功能

## 铁律
同PHASE3_PLAN.md。作者: dylanyunlon <dogechat@163.com>
push: `git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/walpurgis-WTFGG.git`
