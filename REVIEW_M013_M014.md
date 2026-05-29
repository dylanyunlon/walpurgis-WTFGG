# Code Review — M013/M014 Partition-Selection Interval Index

**Reviewer stance:** production-grade, "Knuth standard" — precise complexity,
paranoid about boundary conditions, every invariant proved, every performance
claim backed by a measurement run in this environment. Reviewed from two angles:
(1) **user angle** — could this introduce bugs the caller would hit?
(2) **systems angle** — does it hold up under scale, concurrency, and the
existing architecture?

This document records what was checked, the evidence, and the fixes. It exists
so the next engineer can trust — or re-run — every claim.

---

## 0. What survived review unchanged (the algorithm is correct)

`PartitionSkipList::overlaps()` — the span-max-augmented pruned interval walk:

| Test | Scale | Result |
|------|-------|--------|
| Exhaustive cross-check vs brute force (small endpoint space, covers ~all structural layouts) | 10,000,000 queries | 0 mismatches |
| Random + adversarial (deeply nested, many equal ts_lo, huge spans) | 32,000 queries | 0 mismatches |

A suspected Phase-1 descent under-report was investigated with a constructed
counterexample and **disproven** — the `span_max < lo` skip is provably safe
(every interval in a skipped run ends before `lo`, so none can overlap), and the
landing node plus its base-level successor chain covers the rest.

**Conclusion:** the core data structure is sound. Every issue below is at the
integration / systems / measurement layer.

---

## 1. Issues found, evidence, and resolution

Severity: how badly it bites in production.

### S1 — Streaming rebuild is O(N²·log N) · systems · **severe**
**Evidence (measured):** with a single `PartitionSkipList` rebuilt every flush,
per-flush time grew monotonically — 0.24 ms at P=100, 0.43 ms at P=800.
`build()` is O(P log P) (sort) + O(P) (link + span_max, amortized via the
geometric level structure). N flushes ⇒ Σ O(P_i log P_i) = O(N² log N).
`span_max` is a global per-level fold, so it cannot be locally patched on append.

**Fix:** `SegmentedPartitionIndex` — the log-structured-merge / Lucene-segment
pattern. Each flush builds ONE immutable segment over only the new partitions
(O(M log M), M ≪ P). Queries fan out over segments; a threshold
(`kCompactThreshold = 8`) triggers a merge into one segment, amortizing rebuild
cost the way LSM compaction does.
- **Correctness:** 20,000 three-way cross-checks (segmented vs single whole-list
  vs brute force), including compaction — 0 mismatches.
- **Effect (measured):** cumulative flush 21.76 ms → 13.92 ms; per-flush no
  longer grows with P (0.13 ms at P=800 vs 0.43 ms before); occasional
  0.24–0.28 ms spikes are the compaction flushes, exactly as LSM predicts.

### S2 — Benchmark "validation" could pass falsely · user · **severe**
**Evidence:** `slot_set()` keyed result sets on a hash of
`(ts_lo, ts_hi, edge_count)`. Distinct partitions can share that triple in dense
time slices ⇒ hash collision ⇒ two *different* sets compare equal. The original
commit's "90/90 verified at runtime" rested on this unreliable check. Asserting
correctness with an unproven oracle is precisely the error to avoid.

**Fix:** compare by `alloc_id` — assigned monotonically by `TieredAllocator`,
collision-free. Set equality is now a true assertion. Sampling raised from every
200 steps (90 samples) to every 20 steps (900 samples); **900/900 pass**.

### S3 — seqlock is redundant in query_partitions; "wait-free read" overstated · systems · medium
**Evidence:** a reader holds `shared_lock(part_mu_)` for the whole query; the
index rebuild holds `unique_lock(part_mu_)`. The shared lock already serializes
against the rebuild, so `read_retry` can never fail because of an index rebuild.
M007's "wait-free reads" claim does not hold for the index path — readers are
serialized by `part_mu_`. TSan confirmed no race, i.e. the mutex is doing the
real work.

**Fix:** documented honestly in the concurrency contract. The seqlock is kept
only to preserve M007's protection of partition *metadata* mutated during
migration; no lock-freedom is claimed for the index. (A genuinely lock-free
reader path is left as future work, with `index_epoch_` already in place to
support it.)

### S4 — public rebuild self-took a non-recursive lock · user · medium
**Evidence:** `rebuild_partition_index()` was public and acquired
`unique_lock(part_mu_)` internally. `std::shared_mutex` is non-recursive, so any
future caller holding `part_mu_` that called it would self-deadlock (hang / UB).

**Fix:** split the surface. Public `rebuild_partition_index()` /
`compact_partition_index()` acquire the lock themselves; `_locked` internals
assume the caller holds it. The flush path uses `append_new_partitions`, which
acquires the lock itself (flush holds no lock at that point). Naming and a header
contract make the locking discipline explicit.

### S5 — index_ready_ was a plain bool · systems · medium
**Evidence:** a cross-thread "is the index usable" flag with no synchronization;
safe today only because `part_mu_` happens to cover it — accidental, not
designed.

**Fix:** replaced with `std::atomic<uint64_t> index_epoch_` (acquire/release).
Doubles as a staleness stamp for a future lock-free reader.

### S6 — believed wide queries degrade 5×; added a fallback · user · (withdrawn)
**Evidence chain:**
1. First measurement: "wide 0.2×" — index apparently 5× slower than linear.
2. Added a width-ratio pre-estimate to fall back to linear for wide queries.
3. It never fired: a "wide" window (8 000 wide) is only ~8% of the 100 000 global
   extent, so the ratio guard's >50% condition was never met — and hit rate was
   only ~8% anyway, which an index should win.
4. **Direct profiling of the selection step alone:** at P=8000, widest query —
   **3.7 µs indexed vs 8.0 µs linear**. The index wins.
5. **Root cause:** the indexed query touch()es every hit; the linear *oracle*
   did not. N touch() calls (each a `shared_lock` + registry lookup) were
   miscredited to the index. touch() cost is identical on both real paths.

**Resolution:** the fallback was **removed**, not kept — it solved a non-problem
born of a measurement artifact (adding code for a phantom bug is its own defect).
The benchmark was fixed so both paths touch() equally:

**Corrected, fair numbers at P=8000:** narrow **4.1×**, medium **1.75×**,
wide **0.85×**. The index is at worst ~15% slower (wide, where many segments are
walked and a flat scan is more cache-friendly) and up to 4× faster. No collapse.
A real cost model — for the *intra*-partition scan, the place it actually matters
— stays planned as M047.

---

## 2. Regression suite (all green in this environment)

| # | Check | Result |
|---|-------|--------|
| 1 | Skip-list exhaustive cross-check | 10M queries, 0 mismatch |
| 2 | SegmentedIndex three-way cross-check (+ compaction) | 20k, 0 mismatch |
| 3 | `skiplist_selftest` (make target) | 32k, 0 mismatch |
| 4 | TSan concurrent: 4 readers + flush/compact writer | 2.1M queries, no race/crash |
| 5 | `make philemon_bench pidx_bench skiplist_selftest` | all build |
| 6 | All 8 headers compile standalone (`-Wall`) | OK |
| 7 | All 7 benches compile (`-Wall`) | OK |
| 8 | Main `philemon_bench` end-to-end | runs; queries correct (narrow 1302 … full 1,000,000 edges); migration + slab stats intact |

**Backward compatibility:** `query_partitions` is result-identical to the
pre-index linear version (verified by 900 runtime alloc_id comparisons and by the
main benchmark's unchanged query outputs). `migration_sweep` only mutates tier,
not ts_lo/ts_hi, so the index stays valid across migrations; only flush rebuilds
it.

---

## 3. Honest residual limitations

- **Index vs flat scan at very low selectivity:** ~15% slower (wide). Acceptable
  and bounded; not worth a heuristic given S6's lesson. M047 cost model can
  revisit if a workload proves otherwise.
- **seqlock is not actually wait-free for the index** (S3). Documented, not yet
  redesigned.
- **Compaction is synchronous** inside the flush that crosses the threshold —
  one flush in eight pays an O(P log P) merge. A background-thread compactor
  (true LSM) would smooth this; left as future work.
- **`kCompactThreshold = 8`** is a fixed constant, not tuned per workload.
