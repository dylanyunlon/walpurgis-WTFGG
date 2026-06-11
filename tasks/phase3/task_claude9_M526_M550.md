# Claude-9 任务 (M526-M550)

你是Walpurgis项目的子模型执行者。

## 环境准备
```bash
apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git
cd walpurgis-WTFGG
git clone https://github.com/rapidsai/cugraph-gnn.git /tmp/cugraph-gnn
tree -L 2 --charset ascii
cat MULTI_CLAUDE_PLAN.md | head -60
cat tasks/phase3/PHASE3_PLAN.md
```

## 你的commit范围

从cugraph-gnn的以下commit开始迁移，按时间顺序逐个处理:

```
4e7a730 remove eggs
2aa7b06 update readme
db74d87 Merge PR #2 copy-from-cugraph
2b2a25e unneeded
acbddf2 Merge PR #3 branch-24.08
baacf73 resolve dependency-file-generator warning
85cb80d [IMP] Limit Test Data Size in gcn_dist_sg.py
996298f skip CMake 3.30.0
5fae539 Merge PR #5
8e6619c Merge PR #8
91b6e85 remove other packages from ci scripts
02c96b9 fix dgl deps
ecc22bf update code
4e2c49e Merge PR #6
f057bdb Merge PR #9
e55b2cd Merge PR #10
d370b0f Merge PR #12
666d114 add codeowners
fca4b79 split CUDA-suffixed deps
90db89a use correct wg communicator
bd703b3 add wholegraph to repo (208 files)
7a8fd29 add wholegraph (224 files)
27b9bcc resolve merge conflict
b8b2e76 update codeowners
43c26b3 fix typo
961fd04 Merge PR #20
0ea4925 refactor (49 files)
8633a54 Merge PR #23
1ec5277 Merge branch
f8625ce Updates for v24.10
3338205 Merge branch
4be0724 update pr
770ddd4 remove whitespace
e6000e5 Merge PR #24
f4ca484 resolve merge conflicts
a600a2a Merge PR #21
3bbdbb5 conda
```

## 每个commit的处理方式

1. `cd /tmp/cugraph-gnn && git show --stat <hash>` 看diff规模
2. 对Merge(0文件)/CI/版本/文档类: 写SKIP条目到MIGRATION_LOG.md
3. 对核心代码commit: 深入`git show <hash>`看diff, 做Knuth审查, 20%改写迁移
4. 特别注意:
   - bd703b3 + 7a8fd29 (wholegraph导入): 评估C++源码哪些部分与Walpurgis异构内存引擎相关
   - 90db89a (wg communicator fix): 与Walpurgis的nvlink_clique.py直接相关
   - 0ea4925 (refactor): cugraph-pyg核心重构, 需要深入看

## 铁律
1. 不开新分支, 不用v2/v3/port后缀
2. 改的是算法, 20%鲁迅拿法
3. 作者: dylanyunlon <dogechat@163.com>
4. push: `git remote set-url origin https://x-access-token:$GIT_TOKEN@github.com/dylanyunlon/walpurgis-WTFGG.git`
5. `git pull --rebase origin main` 再push
