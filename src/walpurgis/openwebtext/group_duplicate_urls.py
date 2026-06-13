# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Migrated from Megatron-LM commit d1a10da4a (Rename, #23/9062)
# Walpurgis rewrite: ~20% 鲁迅拿法 — 去芜存菁，字字见骨。

import json
import sys
import time


# ── 调试钩子 ────────────────────────────────────────────────────────────────
def _dbg(tag: str, payload=None) -> None:
    """统一断点桩：生产环境可一键静默，调试时打开 DBG=1 环境变量即可。"""
    import os
    if os.environ.get("DBG"):
        ts = time.strftime("%H:%M:%S")
        print(f"[DBG {ts}] {tag}", flush=True)
        if payload is not None:
            print(f"          {payload!r}", flush=True)
# ────────────────────────────────────────────────────────────────────────────


_SIMILARITY_THRESHOLD = 0.9   # 相似度门槛——低于此值，两 URL 各自为政


def is_similar(jaccard_similarity: float) -> bool:
    """判定两篇文档是否近似重复。"""
    return jaccard_similarity >= _SIMILARITY_THRESHOLD


def _parse_url_cluster(myjson: dict) -> list:
    """
    从单条 JSON 行中抽取主 URL 及其相似邻居。

    原文逻辑散落在主循环，鲁迅曰：'散漫者，思路之贼也。'
    故独立成函数，令意图一目了然。
    """
    urls = []
    for main_url, neighbors in myjson.items():
        urls.append(main_url)
        _dbg("main_url", main_url)
        for neighbor in neighbors:
            for other_url, js in neighbor.items():
                if is_similar(js):
                    urls.append(other_url)
                    _dbg("similar_url", (other_url, js))
    return urls


def _merge_into_index(urls: list,
                      url_to_index: dict,
                      index_to_urls: list) -> None:
    """
    并查集式合并：将一批 URL 纳入同一等价类。

    '合久必分，分久必合'——此处只合不分，将重复者收归一处。
    """
    current_index = -1
    other_indices: set = set()

    # 第一遍：找已有归属
    for url in urls:
        if url in url_to_index:
            idx = url_to_index[url]
            if current_index == -1:
                current_index = idx
            elif current_index != idx:
                other_indices.add(idx)

    # 若全无归属，新开一槽
    if current_index == -1:
        current_index = len(index_to_urls)
        index_to_urls.append(set())
        _dbg("new_cluster", current_index)

    # 写入主槽
    for url in urls:
        url_to_index[url] = current_index
        index_to_urls[current_index].add(url)

    # 吸收旧槽——鲁迅式：'旧的不去，新的不来。'
    for stale_idx in other_indices:
        _dbg("absorb_cluster", (stale_idx, "->", current_index))
        for url in index_to_urls[stale_idx]:
            index_to_urls[current_index].add(url)
            url_to_index[url] = current_index
        index_to_urls[stale_idx] = None   # 墓碑标记


def _tally(index_to_urls: list) -> tuple:
    """统计可保留数与应删除数。"""
    total_remain = total_remove = 0
    for cluster in index_to_urls:
        if cluster is not None and len(cluster) > 1:
            total_remove += len(cluster) - 1
            total_remain += 1
    _dbg("tally", {"remain": total_remain, "remove": total_remove})
    return total_remain, total_remove


def _write_output(index_to_urls: list, output_path: str) -> None:
    """将重复组以 JSON-lines 写出。每行一个等价类，键为组号。"""
    _dbg("write_output", output_path)
    with open(output_path, "wb") as f:
        for i, cluster in enumerate(index_to_urls):
            if cluster is not None and len(cluster) > 1:
                line = json.dumps({str(i): list(cluster)}, ensure_ascii=False)
                f.write(line.encode("utf-8"))
                f.write(b"\n")


if __name__ == "__main__":
    print("grouping duplicate urls ...")
    _dbg("argv", sys.argv)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    url_to_index: dict = {}
    index_to_urls: list = []
    counter = 0
    start_time = time.time()

    with open(input_path, "r") as f:
        for line in f:
            counter += 1
            myjson = json.loads(line)
            _dbg("line", counter)

            urls = _parse_url_cluster(myjson)
            _merge_into_index(urls, url_to_index, index_to_urls)

            if counter % 100_000 == 0:
                elapsed = time.time() - start_time
                print(f" > processed {counter} lines in {elapsed:.1f} seconds ...")
                _dbg("progress", {"counter": counter, "elapsed": elapsed})

    total_remain, total_remove = _tally(index_to_urls)
    print(
        f"out of {total_remove + total_remain} urls, "
        f"only {total_remain} are unique and {total_remove} should be removed"
    )

    _write_output(index_to_urls, output_path)
    _dbg("done", time.time() - start_time)
