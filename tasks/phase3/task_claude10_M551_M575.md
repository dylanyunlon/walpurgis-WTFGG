# Claude-10 任务 (M551-M575)

你是Walpurgis项目的子模型执行者。

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git clone https://github.com/rapidsai/cugraph-gnn.git /tmp/cugraph-gnn
cat tasks/phase3/PHASE3_PLAN.md
cat MIGRATION_LOG.md | tail -30
```

## 你的commit范围

```
44a06e5 Merge PR #28
755c2e3 Merge PR #29
3e5df7c pull in changes from cugraph repo (6 files)
ef8d1e4 Merge PR #31
b9db217 Drop Python 3.9 support (5 files)
d2d9028 update requires-python floor
609f725 Remove NumPy <2 pin (7 files)
98084e8 Merge PR #32
3c59e99 upgrade target Python version for black
e89744d Merge PR #34
305aa8f Add support for Python 3.12 (3 files)
4637f75 Merge PR #41
8f8b71f Update flake8 to 7.1.1 (8 files)
a2e3e2c Fix update-version.sh (4 files)
2798f5e update cmakelists for VERSION file (3 files)
bd8e45b Merge PR #44
74c365d update-version.sh packaging lib
6a93e54 Merge PR #33
b6163b1 pull changes from cugraph repo (4 files)
0c82d1f Merge branch biased-dgl
8a7de9e Merge PR #46
f57ed88 pull in changes from cugraph repo (20 files)
1295d2f update branch
2b6f2cd Merge PR #48 pyg-neg-sampling
37f8629 fix import order (4 files)
2f41ad3 Merge PR #49 fix-rapids-import
```

## 处理方式
同phase3/PHASE3_PLAN.md。重点关注:
- 3e5df7c: 从cugraph拉入的变更, 看是否含采样/图存储改动
- f57ed88 (20 files): 大批代码更新, 深入diff分析

## 铁律
同PHASE3_PLAN.md。作者: dylanyunlon <dogechat@163.com>
push: `git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/walpurgis-WTFGG.git`
