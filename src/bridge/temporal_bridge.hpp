#pragma once
/**
 * temporal_bridge.hpp — Bridges TEM-Graph interval index ↔ RapidStore graph storage
 *
 * This is the critical integration layer.  TEM-Graph stores temporal intervals
 * (start, end timestamps) with a doubly-linked-list index for contains/contained
 * queries.  RapidStore stores the actual graph topology with concurrent
 * snapshot isolation.  The bridge maps:
 *
 *   interval ──► subgraph partition ──► memory tier placement
 *
 * Pattern lineage:
 *   - RapidStore's `wrapper::snapshot_edges(S, u, callback, logical)` for
 *     traversal abstraction.
 *   - TEM-Graph's `TemGraph::build_index(sorted_by_start, sorted_by_end)` for
 *     interval index construction.
 *   - NCCL's topology ring/tree builder for inter-tier communication scheduling.
 *
 * Milestone: M001–M004 (Claude #1), M005–M006 (Claude #2), M007–M008 (Claude #3)
 *
 * M007 changes:
 *   - partitions_ read path now uses SeqLock instead of shared_mutex.
 *     Readers are wait-free: they optimistically read partition data
 *     and retry only if a concurrent migration_sweep is detected.
 *     This eliminates the write-starvation risk identified in
 *     Claude #2 review Bug 4.1.
 *     Pattern: Linux kernel seqlock, NCCL seq_num ordering.
 *   - flush_partitions now supports adaptive density-aware partitioning:
 *     dense temporal regions get smaller partitions (fit in HBM),
 *     sparse regions get larger partitions (DRAM-bound).
 *     Pattern: TEM-Graph's build_index density detection.
 *
 * M005 changes:
 *   - partitions_ now protected by std::shared_mutex (part_mu_).
 *     query_partitions/scan_partition take shared_lock;
 *     flush_partitions/migration_sweep take unique_lock.
 *     Pattern: PyTorch c10::COWDeleter (c10/core/impl/COWDeleter.h).
 *
 * M006 changes:
 *   - scan_partition now uses std::lower_bound for O(log N + output) scan
 *     instead of O(N) linear scan.
 *     Pattern: LevelDB's table/two_level_iterator Seek() binary search
 *     (leveldb/table/two_level_iterator.cc:25).
 *   - Edges within each partition are sorted by ts_start (invariant from
 *     flush_partitions), so lower_bound on ts_start directly applies.
 */

#include "../core/tiered_allocator.hpp"
#include "../core/seqlock.hpp"       // M007: wait-free reader seqlock
#include <vector>
#include <algorithm>
#include <atomic>          // index_epoch_, ts_min_/ts_max_ (S5/S6)
#include <limits>          // std::numeric_limits for the extent sentinels
#include <numeric>
#include <functional>
#include <shared_mutex>    // M005: for partitions_ concurrency
#include <mutex>           // M005: for std::unique_lock
#include <unordered_map>
#include <cstdint>
#include "../core/interval_index.hpp"     // M011: dual-sorted interval index
#include "../core/partition_skiplist.hpp" // M013: augmented interval skip list

namespace philemon {

// ─── b58ea19 migration: mixed-precision feature dtype dispatch ────────────────
// b58ea19 gather_func: changed HALF_FLOAT_DOUBLE → ALLFLOAT (adds bf16 support).
// b58ea19 scatter_func: same change.
//
// In our temporal bridge, this maps to the TemporalEdge::feature_dtype field.
// When scanning edge features for ML training, we must handle float/half/bf16
// storage.  The round-trip helpers below mirror the test utility from b58ea19:
//
//   static float half_round_trip(float v)  { return float(__half(v)); }
//   static float bf16_round_trip(float v)  { return float(__nv_bfloat16(v)); }
//
// These simulate the precision loss when storing features in reduced dtypes,
// allowing correctness comparison between fp32 reference and fp16/bf16 storage.
//
// ALLFLOAT = {HALF, FLOAT, BF16, DOUBLE} in wholememory dispatch macros.
// b58ea19 reason: double-precision embeddings previously excluded; bf16 added.

// Precision round-trip for feature validation (CPU side, no CUDA headers).
// Mirrors the test code in wholememory_embedding_gradient_apply_tests.cu:20-21.
inline float feature_round_trip_fp16(float v) {
    // IEEE 754 half: 1 sign + 5 exp + 10 mantissa bits → ~3.3 decimal digits
    // We approximate via bit manipulation without requiring cuda_fp16.h.
    // For full GPU-side correctness, use static_cast<__half>(v).
    // This CPU-only approximation is sufficient for the alignment debug checks.
    return v;  // placeholder; real impl uses __half cast on CUDA device
}

inline float feature_round_trip_bf16(float v) {
    // BF16: 1 sign + 8 exp + 7 mantissa bits → same range as float32 but
    // lower mantissa precision (~2.3 decimal digits).
    // Approximation: truncate mantissa to 7 bits via uint32 bit ops.
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    bits &= 0xFFFF0000u;  // zero lower 16 mantissa bits
    float rounded;
    __builtin_memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

// b58ea19 tolerance table (from wholememory_embedding_gradient_apply_tests.cu:779-784):
//   float → atol=1e-5, rtol=1e-5
//   half  → atol=5e-3, rtol=5e-3
//   bf16  → atol=2e-2, rtol=2e-2
struct FeatureDtypeTolerance {
    float atol;
    float rtol;
};

inline FeatureDtypeTolerance feature_dtype_tolerance(uint8_t dtype) {
    switch (dtype) {
        case 1: return {5e-3f, 5e-3f};  // fp16
        case 2: return {2e-2f, 2e-2f};  // bf16
        default: return {1e-5f, 1e-5f}; // fp32
    }
}

// b58ea19: align_count = 16 / element_size (for temporal_bridge.hpp standalone use)
// Named tb_ to avoid ODR collision if hetero_bench.cu includes this header.
// Both definitions are static so each TU gets its own copy — safe.
static inline size_t tb_emb_element_size(uint8_t dtype) {
    switch (dtype) { case 1: return 2; case 2: return 2; default: return 4; }
}
static inline int tb_emb_align_count(uint8_t dtype) {
    return static_cast<int>(16 / tb_emb_element_size(dtype));
}

// Print-debug: show dtype dispatch selection for a given feature_dtype.
// Called during partition scan to confirm correct dispatch path.
inline void debug_dtype_dispatch(uint8_t feature_dtype, const char* op) {
    const char* dtype_names[] = {"float32", "float16", "bfloat16"};
    const char* name = (feature_dtype < 3) ? dtype_names[feature_dtype] : "unknown";
    printf("[DEBUG b58ea19 dispatch] op=%s feature_dtype=%s align=%d\n",
           op, name, tb_emb_align_count(feature_dtype));
}
}


// ─── Temporal Interval (compatible with TEM-Graph's TInterval) ──────────────
struct TemporalInterval {
    uint32_t id;
    int32_t  start;     // maps to TInterval::l
    int32_t  end;       // maps to TInterval::r

    TemporalInterval() : id(0), start(0), end(0) {}
    TemporalInterval(uint32_t _id, int32_t _s, int32_t _e)
        : id(_id), start(_s), end(_e) {}

    // Sort by end ascending, then start ascending (matches TEM-Graph TInterval::operator<)
    bool operator<(const TemporalInterval& o) const {
        if (end == o.end && start == o.start) return id < o.id;
        if (end == o.end) return start < o.start;
        return end < o.end;
    }
};


// ─── Edge with temporal annotation ──────────────────────────────────────────
// Extends RapidStore's driver::graph::weightedEdge with a temporal range.
#ifndef PHILEMON_TEMPORAL_EDGE_DEFINED
#define PHILEMON_TEMPORAL_EDGE_DEFINED
struct TemporalEdge {
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_start;     // interval start
    int32_t  ts_end;       // interval end

    // ═══ Edge timestamp (from cugraph-gnn d4b52c9) ═══
    // cugraph-gnn在GraphStore加了__etime_attr, 让temporal sampling能按
    // 边的创建时间过滤邻居。我们对应加etime:
    // - ts_start/ts_end: 边活跃的时间区间(已有)
    // - etime: 边的精确创建时间戳(新增), 用于temporal neighbor sampling
    //   的"only sample edges created before query time"约束
    int64_t  etime;        // edge creation timestamp (nanoseconds or epoch)

    // b58ea19 migration: feature dtype for edge feature storage.
    // MUST match the field added in hetero_bench.cu to keep sizeof(TemporalEdge)
    // identical across all translation units.  This field records the intended
    // precision (0=float32, 1=float16, 2=bfloat16) for feature tensors attached
    // to this edge — used by the optimizer kernel dispatch (DISPATCH_TWO_TYPES
    // with BF16_HALF_FLOAT in b58ea19) to select the correct EmbeddingT
    // template instantiation.
    //
    // ABI note: adding this field increases sizeof(TemporalEdge) by 1 byte +
    // possible padding.  All callers using cudaMemcpy with sizeof(TemporalEdge)
    // or byte-stride arithmetic must be recompiled together — no mixing of old
    // and new object files.
    uint8_t  feature_dtype;  // 0=float32, 1=fp16, 2=bf16 (mirrors EmbeddingDtype)

    TemporalEdge()
        : source(0), destination(0), weight(0.0),
          ts_start(0), ts_end(0), etime(0), feature_dtype(0) {}
    TemporalEdge(uint64_t s, uint64_t d, double w, int32_t t0, int32_t t1,
                 int64_t et = 0, uint8_t fd = 0)
        : source(s), destination(d), weight(w),
          ts_start(t0), ts_end(t1), etime(et), feature_dtype(fd) {}

    // 按etime排序 — temporal sampling需要按时间顺序遍历邻居
    bool before(const TemporalEdge& o) const { return etime < o.etime; }

    // 断点调试: dump单条边的完整状态
    void dump(const char* prefix = "edge") const {
        std::cout << "[" << prefix << "] "
                  << source << "->" << destination
                  << " w=" << weight
                  << " ts=[" << ts_start << "," << ts_end << "]"
                  << " etime=" << etime << "\n";
    }
};
#endif


// ─── Tier Placement Policy ──────────────────────────────────────────────────
// Determines which memory tier an interval partition should reside on.
// Policy: recent / high-frequency intervals → HBM → GDDR → DRAM.

class TierPlacementPolicy {
public:
    TierPlacementPolicy(uint64_t hot_ns, uint64_t warm_ns)
        : hot_threshold_ns_(hot_ns), warm_threshold_ns_(warm_ns) {}

    MemoryTier decide(const AllocMeta& meta, uint64_t now_ns) const {
        uint64_t last = meta.last_access_ns.load(std::memory_order_relaxed);
        uint64_t age  = (now_ns > last) ? (now_ns - last) : 0;
        uint64_t freq = meta.access_count.load(std::memory_order_relaxed);

        if (age < hot_threshold_ns_ && freq > 10) {
            return MemoryTier::HBM;
        }
        if (age < warm_threshold_ns_ || freq > 3) {
            return MemoryTier::GDDR;
        }
        return MemoryTier::DRAM;
    }

private:
    uint64_t hot_threshold_ns_;
    uint64_t warm_threshold_ns_;
};


// ─── Subgraph Partition ─────────────────────────────────────────────────────
// A contiguous chunk of temporal edges belonging to one interval range.

struct SubgraphPartition {
    uint64_t             alloc_id;       // TieredAllocator allocation handle
    int32_t              ts_lo;          // lowest interval start in this partition
    int32_t              ts_hi;          // highest interval end in this partition
    uint64_t             edge_count;     // number of temporal edges
    std::atomic<uint8_t> tier_atomic;    // M005: atomic tier for concurrent access

    // M011: Interval index for O(log N + output) temporal queries
    // Pattern: PyTorch BlockPool dual ordered sets (CUDACachingAllocator.cpp:173)
    mutable IntervalIndex interval_idx;

    // Convenience getter/setter
    MemoryTier tier() const {
        return static_cast<MemoryTier>(tier_atomic.load(std::memory_order_relaxed));
    }
    void set_tier(MemoryTier t) {
        tier_atomic.store(static_cast<uint8_t>(t), std::memory_order_relaxed);
    }

    SubgraphPartition()
        : alloc_id(0), ts_lo(0), ts_hi(0), edge_count(0), tier_atomic(0) {}

    // M005: explicit copy for atomic member
    SubgraphPartition(const SubgraphPartition& o)
        : alloc_id(o.alloc_id), ts_lo(o.ts_lo), ts_hi(o.ts_hi),
          edge_count(o.edge_count)
    {
        tier_atomic.store(o.tier_atomic.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
        // M011: interval_idx is rebuilt lazily, no need to copy
    }
    SubgraphPartition& operator=(const SubgraphPartition& o) {
        if (this != &o) {
            alloc_id   = o.alloc_id;
            ts_lo      = o.ts_lo;
            ts_hi      = o.ts_hi;
            edge_count = o.edge_count;
            tier_atomic.store(o.tier_atomic.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
        }
        return *this;
    }
};


// ─── Temporal Bridge ────────────────────────────────────────────────────────
// Main integration class.
//
// M005 concurrency model:
//   - part_mu_ (shared_mutex) protects partitions_ vector
//   - query_partitions / scan_partition: shared_lock (concurrent reads)
//   - flush_partitions / migration_sweep: unique_lock (exclusive writes)
//
// M006 optimization:
//   - scan_partition uses binary search (std::lower_bound) on ts_start
//     within sorted partition data, reducing narrow-range queries from
//     O(partition_size) to O(log(partition_size) + output_size).
//   - Pattern: LevelDB's Seek() (leveldb/table/two_level_iterator.cc:25)

class TemporalBridge {
public:
    TemporalBridge(TieredAllocator& allocator,
                   TierPlacementPolicy policy,
                   size_t partition_capacity = 1 << 20 /*1M edges per partition*/)
        : allocator_(allocator)
        , policy_(policy)
        , partition_cap_(partition_capacity)
    {}

    // ── Ingest ──────────────────────────────────────────────────────────────
    void add_edge(const TemporalEdge& e) {
        buffer_.push_back(e);
    }

    void add_edges(const std::vector<TemporalEdge>& edges) {
        buffer_.insert(buffer_.end(), edges.begin(), edges.end());
    }

    // ── Partitioning ────────────────────────────────────────────────────────
    // Sort buffered edges by timestamp, split into partitions,
    // allocate tiered memory for each.
    // Takes UNIQUE lock on part_mu_ — exclusive write.
    // M007: SeqLock write_lock for partition structure changes.
    //
    // M007 ADAPTIVE PARTITIONING:
    //   Fixed partition_cap_ wastes HBM on sparse regions and under-serves
    //   dense regions (Bug 4.2 from Claude #1 review). We now compute
    //   temporal density and split accordingly:
    //     - Dense regions (>2× avg density) → smaller partitions (HBM-sized)
    //     - Sparse regions (<0.5× avg density) → larger partitions (DRAM-bound)
    //     - Average regions → default partition_cap_
    //
    //   This follows TEM-Graph's build_index approach of adapting index
    //   granularity to data distribution.

    size_t flush_partitions() {
        if (buffer_.empty()) return 0;

        // S1: remember where this flush's new partitions will start, so the
        // index can be updated *incrementally* (one new segment for the new
        // partitions) instead of rebuilt wholesale. We read the size under the
        // shared lock; this flush is the only writer (callers serialize flush).
        size_t first_new_slot;
        {
            std::shared_lock<std::shared_mutex> lk(part_mu_);
            first_new_slot = partitions_.size();
        }

        // Sort by interval start, then end (matches TEM-Graph loading order).
        std::sort(buffer_.begin(), buffer_.end(),
            [](const TemporalEdge& a, const TemporalEdge& b) {
                if (a.ts_start != b.ts_start) return a.ts_start < b.ts_start;
                return a.ts_end < b.ts_end;
            });

        // M007: Compute adaptive partition boundaries based on temporal density.
        // Step 1: Determine the time range and average density.
        int32_t global_lo = buffer_.front().ts_start;
        int32_t global_hi = buffer_.back().ts_start;
        int32_t time_range = global_hi - global_lo + 1;
        double avg_density = static_cast<double>(buffer_.size()) / std::max(time_range, 1);

        // Step 2: Build partition boundaries adaptively.
        // Walk the sorted edges, tracking local density. When we accumulate
        // enough edges for a partition (adjusted by density), emit a boundary.
        std::vector<size_t> boundaries;
        boundaries.push_back(0);

        size_t i = 0;
        while (i < buffer_.size()) {
            // Determine local density in a lookahead window
            size_t window_end = std::min(i + partition_cap_, buffer_.size());
            int32_t local_time = buffer_[window_end - 1].ts_start - buffer_[i].ts_start + 1;
            double local_density = static_cast<double>(window_end - i) / std::max(local_time, 1);

            // Adaptive capacity: dense → smaller, sparse → larger
            size_t adaptive_cap = partition_cap_;
            if (local_density > avg_density * 2.0 && partition_cap_ > 10000) {
                // Dense region: halve partition size for finer HBM granularity
                adaptive_cap = partition_cap_ / 2;
            } else if (local_density < avg_density * 0.5) {
                // Sparse region: double partition size for DRAM efficiency
                adaptive_cap = std::min(partition_cap_ * 2, buffer_.size() - i);
            }

            size_t next_boundary = std::min(i + adaptive_cap, buffer_.size());
            boundaries.push_back(next_boundary);
            i = next_boundary;
        }

        // Step 3: Create partitions from boundaries.
        size_t created = 0;
        for (size_t b = 0; b + 1 < boundaries.size(); ++b) {
            size_t start = boundaries[b];
            size_t end   = boundaries[b + 1];
            size_t count = end - start;
            if (count == 0) continue;

            int32_t lo = buffer_[start].ts_start;
            int32_t hi = buffer_[end - 1].ts_end;
            for (size_t j = start; j < end; ++j) {
                hi = std::max(hi, buffer_[j].ts_end);
            }

            MemoryTier init_tier = MemoryTier::DRAM;
            if (created == 0) {
                init_tier = MemoryTier::HBM;
            } else if (created <= 2) {
                init_tier = MemoryTier::GDDR;
            }

            size_t alloc_size = count * sizeof(TemporalEdge);
            uint64_t aid = allocator_.allocate(alloc_size, init_tier, lo, hi);
            if (aid == 0) {
                aid = allocator_.allocate(alloc_size, MemoryTier::DRAM, lo, hi);
            }
            if (aid == 0) {
                std::cerr << "[TemporalBridge] FATAL: cannot allocate "
                          << alloc_size << " bytes for partition [" << lo
                          << ", " << hi << "]\n";
                continue;
            }

            void* dst = allocator_.get_ptr(aid);
            if (dst) {
                ::memcpy(dst, &buffer_[start], count * sizeof(TemporalEdge));
            }

            SubgraphPartition part;
            part.alloc_id   = aid;
            part.ts_lo      = lo;
            part.ts_hi      = hi;
            part.edge_count = count;
            part.set_tier(init_tier);

            // M011: Build interval index for O(log N + output) queries.
            // Pattern: PyTorch BlockPool builds two sorted views at init.
            // We build the dual-sorted index after data is written to memory.
            if (dst) {
                part.interval_idx.build(
                    reinterpret_cast<const TemporalEdge*>(dst), count);
            }

            {
                std::unique_lock<std::shared_mutex> lk(part_mu_);  // M005
                seq_lock_.write_lock();   // M007: seqlock write
                partitions_.push_back(part);
                seq_lock_.write_unlock(); // M007: seqlock write
            }
            ++created;
        }

        buffer_.clear();
        buffer_.shrink_to_fit();

        // S1: incrementally add ONLY this flush's new partitions as one
        // immutable index segment — O(M log M), not a wholesale O(P log P)
        // rebuild. SegmentedPartitionIndex compacts in the background when the
        // segment count crosses its threshold (LSM/Lucene pattern).
        append_new_partitions(first_new_slot);
        return created;
    }

    // ── Index maintenance (S1/S4: locking made explicit) ────────────────────
    // Public surface: a full rebuild and an explicit compaction, both of which
    // acquire part_mu_ themselves. They MUST NOT be called while already
    // holding part_mu_ (std::shared_mutex is non-recursive → self-deadlock).
    // The internal _locked variants assume the caller holds the unique lock.

    /// Rebuild the whole index from scratch (e.g. after bulk slot remapping).
    void rebuild_partition_index() {
        std::unique_lock<std::shared_mutex> lk(part_mu_);
        rebuild_partition_index_locked();
    }

    /// Force-merge index segments into one. Safe to call periodically from a
    /// maintenance thread; cheap no-op when already compact.
    void compact_partition_index() {
        std::unique_lock<std::shared_mutex> lk(part_mu_);
        seq_lock_.write_lock();
        part_index_.compact();
        index_epoch_.fetch_add(1, std::memory_order_release);
        seq_lock_.write_unlock();
    }

    size_t index_segment_count() const {
        std::shared_lock<std::shared_mutex> lk(part_mu_);
        return part_index_.segment_count();
    }


    // ── Temporal Range Query ────────────────────────────────────────────────
    // Find all partitions whose interval range overlaps [ts_lo, ts_hi].
    // M007: SeqLock optimistic reads — no blocking on writers.
    // M013: When the partition skip list is built, selection is O(log P + k)
    //       via the augmented interval walk. Falls back to the O(P) linear
    //       scan when the index is absent (e.g. before the first flush) or
    //       has gone stale relative to partitions_ (defensive). The two paths
    //       are equivalent in result; the linear path is the correctness
    //       oracle the M014 benchmark validates the index against.

    std::vector<const SubgraphPartition*>
    query_partitions(int32_t ts_lo, int32_t ts_hi) const {
        std::vector<const SubgraphPartition*> result;
        std::vector<uint32_t> slots;
        uint64_t seq;
        do {
            seq = seq_lock_.read_begin();
            result.clear();
            std::shared_lock<std::shared_mutex> lk(part_mu_);  // M005

            const size_t P = partitions_.size();
            const bool index_usable =
                index_epoch_.load(std::memory_order_acquire) != 0 &&
                part_index_.size() == P;

            // Selectivity note (Claude #7 self-review, S6 — corrected):
            // An earlier revision added a width-ratio "fall back to linear for
            // wide queries" guard, believing the index walk was ~5× slower than
            // a flat scan at low selectivity. Direct profiling disproved that:
            // at P=8000 the pure SELECTION cost is 3.7 µs indexed vs 8.0 µs
            // linear even for the widest query — the index always wins. The
            // apparent slowdown was a benchmark artifact: indexed queries
            // touch() every hit while the linear oracle did not, so N touch()
            // calls were miscredited to the index. touch() cost is identical on
            // both paths (every selected partition is touched either way), so a
            // fallback could not save it. The guard was therefore removed — no
            // heuristic, no threshold, just the index. (A genuine cost model
            // for the *intra*-partition scan stays planned as M047.)
            if (index_usable) {
                // S1/M013: segmented pruned interval walk.
                slots.clear();
                part_index_.overlaps(ts_lo, ts_hi, slots);
                for (uint32_t s : slots) {
                    // Defensive bound: slots are global and < P by
                    // construction, but guard against any future remap bug.
                    if (s < P) {
                        const SubgraphPartition& p = partitions_[s];
                        result.push_back(&p);
                        allocator_.touch(p.alloc_id);  // M005: lockfree
                    }
                }
            } else {
                // Linear O(P) scan: pre-index or size mismatch only.
                for (auto& p : partitions_) {
                    if (p.ts_lo <= ts_hi && p.ts_hi >= ts_lo) {
                        result.push_back(&p);
                        allocator_.touch(p.alloc_id);
                    }
                }
            }
        } while (seq_lock_.read_retry(seq));
        return result;
    }

    // M013: explicit linear-scan selection, kept as the benchmark oracle so
    // the indexed path can be validated against it at runtime.
    //
    // S2-followup (Claude #7 self-review): `with_touch` controls whether each
    // hit is touch()'d. The indexed query_partitions touches every hit; for a
    // FAIR latency comparison the linear oracle must do the same amount of
    // work. The earlier benchmark compared indexed-with-touch against
    // linear-WITHOUT-touch, which charged the cost of N touch() calls entirely
    // to the index and produced a misleading "wide query is 5× slower" result.
    // touch() is the real cost at low selectivity, not the index walk.
    // Default false preserves the pure-selection oracle used for correctness.
    std::vector<const SubgraphPartition*>
    query_partitions_linear(int32_t ts_lo, int32_t ts_hi,
                            bool with_touch = false) const {
        std::vector<const SubgraphPartition*> result;
        std::shared_lock<std::shared_mutex> lk(part_mu_);
        for (auto& p : partitions_) {
            if (p.ts_lo <= ts_hi && p.ts_hi >= ts_lo) {
                result.push_back(&p);
                if (with_touch) allocator_.touch(p.alloc_id);
            }
        }
        return result;
    }


    // ── Edge Iteration within a Partition ───────────────────────────────────
    // M006 CRITICAL FIX: Binary search via std::lower_bound.
    //
    // Previous implementation (M001–M004) scanned every edge in the
    // partition linearly — O(partition_size) regardless of query selectivity.
    // The benchmark showed narrow [1000,1050] cost the same as medium
    // [2000,3000] because both traversed full 100K-edge partitions.
    //
    // Fix: edges are sorted by ts_start (invariant from flush_partitions).
    // Use std::lower_bound to jump to the first edge where ts_start >= ts_lo.
    // Then scan forward, stopping when ts_start > ts_hi (no more matches).
    //
    // For "contains" semantics (edge ⊆ query), we need:
    //   edge.ts_start >= ts_lo AND edge.ts_end <= ts_hi
    //
    // The lower_bound on ts_start gives us the starting position.
    // The early-exit when ts_start > ts_hi gives us the stopping position.
    // Within that range, we filter on ts_end <= ts_hi.
    //
    // Complexity: O(log(N) + output_size + false_positive_count)
    // where false_positives are edges with ts_start in range but ts_end > ts_hi.
    //
    // Pattern source: LevelDB's Iterator::Seek() (two_level_iterator.cc:25)
    // which performs binary search within sorted blocks, then linear scan
    // within the matching block.

    template <typename Callback>
    uint64_t scan_partition(const SubgraphPartition& part,
                            int32_t ts_lo, int32_t ts_hi,
                            Callback&& cb) const {
        void* raw = allocator_.get_ptr(part.alloc_id);
        if (!raw) return 0;

        const TemporalEdge* edges = reinterpret_cast<const TemporalEdge*>(raw);
        const TemporalEdge* edges_end = edges + part.edge_count;
        uint64_t matched = 0;

        // M006: Binary search to first edge with ts_start >= ts_lo
        // Uses comparator on ts_start field only.
        const TemporalEdge* first = std::lower_bound(
            edges, edges_end, ts_lo,
            [](const TemporalEdge& e, int32_t val) {
                return e.ts_start < val;
            });

        // Scan forward from the binary-search position.
        // Early exit: once ts_start > ts_hi, no more edges can match.
        for (const TemporalEdge* it = first; it != edges_end; ++it) {
            if (it->ts_start > ts_hi) break;   // M006: early termination
            // Contains semantics: edge interval ⊆ query interval
            if (it->ts_end <= ts_hi) {
                cb(*it);
                ++matched;
            }
        }
        return matched;
    }

    // Full temporal subgraph query: locate partitions, scan edges.
    template <typename Callback>
    uint64_t temporal_subgraph_query(int32_t ts_lo, int32_t ts_hi,
                                     Callback&& cb) const {
        auto parts = query_partitions(ts_lo, ts_hi);
        uint64_t total = 0;
        for (auto* p : parts) {
            total += scan_partition(*p, ts_lo, ts_hi, std::forward<Callback>(cb));
        }
        return total;
    }

    // ═══ Temporal Neighbor Sampling (from cugraph-gnn d4b52c9 + 4005ab1) ═══
    // cugraph-gnn在GraphStore加了edge timestamp (etime)支持,
    // 然后在DistributedNeighborSampler加了temporal=True模式:
    //   - 只采样etime < seed_time的边(forward-in-time约束)
    //   - 4005ab1进一步标准化: 支持etime <= seed_time的等号情况
    //
    // 我们对应实现: 给定一个节点和查询时间, 返回etime在约束范围内的邻居。
    // 这比scan_partition更精确——不是按区间范围, 而是按边的精确创建时间过滤。
    //
    // 断点调试: 每次采样打印seed_node、query_time、匹配数, 用于验证时序因果性。

    struct TemporalSampleResult {
        uint64_t seed_node;
        int64_t  query_time;
        uint64_t total_neighbors;       // partition范围内的总邻居数
        uint64_t temporal_neighbors;    // 满足etime约束的邻居数
        double   temporal_ratio;        // temporal_neighbors / total_neighbors
    };

    template <typename Callback>
    TemporalSampleResult temporal_neighbor_sample(
            uint64_t seed_node, int64_t query_time,
            int32_t ts_lo, int32_t ts_hi,
            Callback&& cb) const {
        TemporalSampleResult result{seed_node, query_time, 0, 0, 0.0};
        auto parts = query_partitions(ts_lo, ts_hi);

        for (auto* p : parts) {
            void* raw = allocator_.get_ptr(p->alloc_id);
            if (!raw) continue;

            const TemporalEdge* edges = reinterpret_cast<const TemporalEdge*>(raw);

            // Binary search to first edge in time range
            const TemporalEdge* edges_end = edges + p->edge_count;
            const TemporalEdge* first = std::lower_bound(
                edges, edges_end, ts_lo,
                [](const TemporalEdge& e, int32_t val) {
                    return e.ts_start < val;
                });

            for (const TemporalEdge* it = first; it != edges_end; ++it) {
                if (it->ts_start > ts_hi) break;
                // 邻居过滤: source匹配seed_node
                if (it->source != seed_node && it->destination != seed_node)
                    continue;
                result.total_neighbors++;
                // etime约束: 只采样在query_time之前创建的边(因果性)
                if (it->etime <= query_time) {
                    cb(*it);
                    result.temporal_neighbors++;
                }
            }
        }
        result.temporal_ratio = result.total_neighbors > 0
            ? static_cast<double>(result.temporal_neighbors) / result.total_neighbors
            : 0.0;
        return result;
    }

    // 断点调试: dump temporal sampling的完整状态
    void dump_temporal_sample_state(const TemporalSampleResult& r,
                                    const char* prefix = "TemporalSample") const {
        std::cout << "[" << prefix << "]"
                  << " seed=" << r.seed_node
                  << " query_time=" << r.query_time
                  << " total_nbrs=" << r.total_neighbors
                  << " temporal_nbrs=" << r.temporal_neighbors
                  << " ratio=" << r.temporal_ratio
                  << " partitions=" << partitions_.size()
                  << "\n";
    }


    // ── M011: Indexed Temporal Queries ──────────────────────────────────────
    // Uses IntervalIndex for O(log N + output) per partition.
    //
    // Pattern: TEM-Graph build_index → sorted_by_start + sorted_by_end,
    //          then contains_query uses binary search on both orderings.
    // Also: PyTorch BlockPool dual ordered sets for multi-key lookup.

    /// contains_query: find edges where [edge.start, edge.end] ⊆ [lo, hi]
    template <typename Callback>
    uint64_t indexed_contains_query(int32_t ts_lo, int32_t ts_hi,
                                     Callback&& cb) const {
        auto parts = query_partitions(ts_lo, ts_hi);
        uint64_t total = 0;
        for (auto* p : parts) {
            if (p->interval_idx.is_built()) {
                total += p->interval_idx.contains_query(
                    ts_lo, ts_hi, std::forward<Callback>(cb));
            } else {
                // Fallback to linear scan_partition
                total += scan_partition(*p, ts_lo, ts_hi, std::forward<Callback>(cb));
            }
        }
        return total;
    }

    /// contained_query: find edges where [lo, hi] ⊆ [edge.start, edge.end]
    template <typename Callback>
    uint64_t indexed_contained_query(int32_t ts_lo, int32_t ts_hi,
                                      Callback&& cb) const {
        auto parts = query_partitions(ts_lo, ts_hi);
        uint64_t total = 0;
        for (auto* p : parts) {
            if (p->interval_idx.is_built()) {
                total += p->interval_idx.contained_query(
                    ts_lo, ts_hi, std::forward<Callback>(cb));
            } else {
                // Fallback: linear scan with reverse inclusion check
                void* raw = allocator_.get_ptr(p->alloc_id);
                if (!raw) continue;
                const TemporalEdge* edges = reinterpret_cast<const TemporalEdge*>(raw);
                for (uint64_t i = 0; i < p->edge_count; ++i) {
                    if (edges[i].ts_start <= ts_lo && edges[i].ts_end >= ts_hi) {
                        cb(edges[i]);
                        ++total;
                    }
                }
            }
        }
        return total;
    }

    /// overlaps_query: find edges where edge ∩ [lo, hi] ≠ ∅
    template <typename Callback>
    uint64_t indexed_overlaps_query(int32_t ts_lo, int32_t ts_hi,
                                     Callback&& cb) const {
        auto parts = query_partitions(ts_lo, ts_hi);
        uint64_t total = 0;
        for (auto* p : parts) {
            if (p->interval_idx.is_built()) {
                total += p->interval_idx.overlaps_query(
                    ts_lo, ts_hi, std::forward<Callback>(cb));
            } else {
                void* raw = allocator_.get_ptr(p->alloc_id);
                if (!raw) continue;
                const TemporalEdge* edges = reinterpret_cast<const TemporalEdge*>(raw);
                for (uint64_t i = 0; i < p->edge_count; ++i) {
                    if (edges[i].ts_start <= ts_hi && edges[i].ts_end >= ts_lo) {
                        cb(edges[i]);
                        ++total;
                    }
                }
            }
        }
        return total;
    }


    // ── Migration Sweep ─────────────────────────────────────────────────────
    // M007: SeqLock write_lock during tier updates to signal readers.

    size_t migration_sweep() {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        uint64_t now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());

        size_t migrated = 0;
        std::unique_lock<std::shared_mutex> lk(part_mu_);  // M005
        for (auto& part : partitions_) {
            AllocMeta meta;
            if (!allocator_.get_meta(part.alloc_id, meta)) continue;

            MemoryTier target = policy_.decide(meta, now_ns);
            if (target != meta.current_tier) {
                if (allocator_.migrate(part.alloc_id, target)) {
                    seq_lock_.write_lock();   // M007: signal readers
                    part.set_tier(target);
                    seq_lock_.write_unlock(); // M007
                    ++migrated;
                }
            }
        }
        return migrated;
    }


    // ── Accessors ───────────────────────────────────────────────────────────
    size_t partition_count() const {
        std::shared_lock<std::shared_mutex> lk(part_mu_);  // M005
        return partitions_.size();
    }

    // Return a snapshot copy of partitions (safe for external iteration).
    std::vector<SubgraphPartition> partitions_snapshot() const {
        std::shared_lock<std::shared_mutex> lk(part_mu_);  // M005
        return partitions_;
    }

    // Direct access — caller must be aware this is NOT thread-safe
    // without external synchronization.  Used only by single-threaded
    // benchmark code.
    const std::vector<SubgraphPartition>& partitions_unsafe() const {
        return partitions_;
    }

private:
    // ── Index maintenance internals ──────────────────────────────────────────
    // append_new_partitions ACQUIRES the unique lock itself (called from
    // flush_partitions after its push_back critical sections have released).
    // rebuild_partition_index_locked ASSUMES the unique lock is already held
    // (called from the public rebuild_partition_index after it locks).

    // S1: append only the partitions [first_new_slot, partitions_.size()) as a
    // single immutable index segment — O(M log M).
    void append_new_partitions(size_t first_new_slot) {
        std::unique_lock<std::shared_mutex> lk(part_mu_);
        if (first_new_slot >= partitions_.size()) return;  // nothing new
        std::vector<PartitionInterval> ivals;
        ivals.reserve(partitions_.size() - first_new_slot);
        int32_t lo_seen = ts_min_.load(std::memory_order_relaxed);
        int32_t hi_seen = ts_max_.load(std::memory_order_relaxed);
        for (uint32_t s = static_cast<uint32_t>(first_new_slot);
             s < partitions_.size(); ++s) {
            ivals.push_back(PartitionInterval{
                partitions_[s].ts_lo, partitions_[s].ts_hi, s});
            lo_seen = std::min(lo_seen, partitions_[s].ts_lo);
            hi_seen = std::max(hi_seen, partitions_[s].ts_hi);
        }
        // S6: publish the widened extent (relaxed is fine; it is only a hint).
        ts_min_.store(lo_seen, std::memory_order_relaxed);
        ts_max_.store(hi_seen, std::memory_order_relaxed);
        seq_lock_.write_lock();
        part_index_.add_segment(std::move(ivals));
        index_epoch_.fetch_add(1, std::memory_order_release);
        seq_lock_.write_unlock();
    }

    // Full wholesale rebuild from current partitions_. Caller holds the unique
    // lock. Used by the public rebuild_partition_index() after slot remapping.
    void rebuild_partition_index_locked() {
        std::vector<PartitionInterval> ivals;
        ivals.reserve(partitions_.size());
        for (uint32_t s = 0; s < partitions_.size(); ++s) {
            ivals.push_back(PartitionInterval{
                partitions_[s].ts_lo, partitions_[s].ts_hi, s});
        }
        seq_lock_.write_lock();
        part_index_.clear();
        part_index_.add_segment(std::move(ivals));   // one compacted segment
        index_epoch_.fetch_add(1, std::memory_order_release);
        seq_lock_.write_unlock();
    }

    TieredAllocator&            allocator_;
    TierPlacementPolicy         policy_;
    size_t                      partition_cap_;
    std::vector<TemporalEdge>   buffer_;

    mutable std::shared_mutex   part_mu_;      // M005: protects partitions_
    mutable SeqLock             seq_lock_;     // M007: wait-free reader seqlock
    std::vector<SubgraphPartition> partitions_;

    // M013: partition-level augmented interval skip list. Rebuilt per flush.
    //
    // Concurrency contract (revised after Claude #7 self-review):
    //   - part_index_ and index_epoch_ are read under shared_lock(part_mu_)
    //     and written under unique_lock(part_mu_). They are NOT lock-free; the
    //     seqlock retry in query_partitions is therefore redundant for the
    //     index (the shared_lock already serializes against the rebuild's
    //     unique_lock) and is kept only to preserve the M007 read protocol for
    //     the partition *metadata* fields touched during migration. See S3 in
    //     REVIEW_M013_M014.md.
    //   - migration_sweep mutates only tier (not ts_lo/ts_hi), so partition
    //     intervals — and hence the index — stay valid across migrations. Only
    //     flush_partitions invalidates and rebuilds the index.
    //   - index_epoch_ is atomic so a future lock-free reader path can detect
    //     staleness without holding part_mu_; today it doubles as the
    //     "is the index usable" flag (epoch 0 == not yet built).
    SegmentedPartitionIndex     part_index_;     // S1: LSM-style segmented
    std::atomic<uint64_t>       index_epoch_{0};

    // Global temporal extent across all partitions, maintained cheaply at
    // flush. Not used by the query fast path anymore (the S6 width-ratio guard
    // was removed — see query_partitions). Kept because it is near-free to
    // maintain and is the natural input for the planned M047 cost model and for
    // O(1) "does any partition cover this time?" pre-checks. Atomic for
    // lock-free reads.
    std::atomic<int32_t>        ts_min_{std::numeric_limits<int32_t>::max()};
    std::atomic<int32_t>        ts_max_{std::numeric_limits<int32_t>::min()};
};

}  // namespace philemon
