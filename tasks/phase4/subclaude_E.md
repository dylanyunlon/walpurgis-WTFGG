apt install -y tree git
git clone https://github.com/dylanyunlon/walpurgis-WTFGG.git && cd walpurgis-WTFGG
git clone https://github.com/rapidsai/cugraph-gnn.git /tmp/cugraph-gnn
cat tasks/phase4/batch_migrate_remaining.md
你是子Claude-E。读上面的任务文件，找到你(E)分配到的5个commit。
对每个commit: cd /tmp/cugraph-gnn && git show <hash> 逐行读diff，然后在walpurgis中做对应迁移(鲁迅拿法改写20%)。
每个commit单独git commit。全部完成后push。
作者: dylanyunlon <dogechat@163.com>
git remote set-url origin https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/walpurgis-WTFGG.git
git pull --rebase origin main && git push origin main
