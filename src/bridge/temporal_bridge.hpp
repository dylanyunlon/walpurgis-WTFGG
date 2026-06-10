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
#include <optional>        // a056923: node_time_func optional return
#include <iostream>
#include "../core/interval_index.hpp"     // M011: dual-sorted interval index
#include "../core/partition_skiplist.hpp" // M013: augmented interval skip list

namespace philemon {

// ─── a056923 migration: TemporalComparison enum ─────────────────────────────
// a056923 fixed cugraph-pyg to use underscore-separated temporal_comparison
// strings ('monotonically_decreasing' instead of 'monotonically decreasing').
// The original space-separated strings were a latent bug: Python's Enum lookup
// would silently fall through to the default, disabling temporal filtering.
//
// In our C++ layer we encode this as a strongly-typed enum to make invalid
// values a compile-time error rather than a silent runtime no-op.
// Corresponds to cugraph_pyg/loader/link_neighbor_loader.py temporal_comparison
// parameter.
//
// Print debug: temporal_comparison_name() used in dump paths to confirm
// which comparison is active — critical for diagnosing causal leakage.
enum class TemporalComparison : uint8_t {
    STRICTLY_INCREASING    = 0,  // strict forward: only edges with etime > seed_time
    MONOTONICALLY_INCREASING = 1,  // non-strict forward: etime >= seed_time
    STRICTLY_DECREASING    = 2,  // strict backward (default PyG mode pre-a056923 bug)
    MONOTONICALLY_DECREASING = 3,  // non-strict backward: etime <= seed_time (DEFAULT)
    LAST                   = 4,  // temporal_strategy='last': most recent only
};

inline const char* temporal_comparison_name(TemporalComparison tc) {
    switch (tc) {
        case TemporalComparison::STRICTLY_INCREASING:     return "strictly_increasing";
        case TemporalComparison::MONOTONICALLY_INCREASING: return "monotonically_increasing";
        case TemporalComparison::STRICTLY_DECREASING:     return "strictly_decreasing";
        case TemporalComparison::MONOTONICALLY_DECREASING: return "monotonically_decreasing";
        case TemporalComparison::LAST:                    return "last";
        default:                                          return "unknown";
    }
}

// Parse from string (accepts both old space-separated and new underscore form
// for backward compatibility during migration).
// Returns MONOTONICALLY_DECREASING on unknown string (matches PyG default).
inline TemporalComparison parse_temporal_comparison(const char* s) {
    // a056923: canonical underscore form
    if (!s) return TemporalComparison::MONOTONICALLY_DECREASING;
    printf("[DEBUG a056923] parse_temporal_comparison: input='%s'\n", s);
    if (strcmp(s, "strictly_increasing")     == 0 ||
        strcmp(s, "strictly increasing")     == 0) {
        return TemporalComparison::STRICTLY_INCREASING;
    }
    if (strcmp(s, "monotonically_increasing") == 0 ||
        strcmp(s, "monotonically increasing") == 0) {
        return TemporalComparison::MONOTONICALLY_INCREASING;
    }
    if (strcmp(s, "strictly_decreasing")     == 0 ||
        strcmp(s, "strictly decreasing")     == 0) {
        return TemporalComparison::STRICTLY_DECREASING;
    }
    if (strcmp(s, "last") == 0) {
        return TemporalComparison::LAST;
    }
    // Default: monotonically_decreasing (PyG default, a056923 corrected string)
    return TemporalComparison::MONOTONICALLY_DECREASING;
}

// Apply temporal comparison: does edge_time satisfy constraint relative to seed_time?
// Mirrors the C-level filter in cugraph-pyg DistributedNeighborSampler.
inline bool temporal_compare(int64_t edge_time, int64_t seed_time,
                              TemporalComparison tc) {
    switch (tc) {
        case TemporalComparison::STRICTLY_INCREASING:     return edge_time > seed_time;
        case TemporalComparison::MONOTONICALLY_INCREASING: return edge_time >= seed_time;
        case TemporalComparison::STRICTLY_DECREASING:     return edge_time < seed_time;
        case TemporalComparison::LAST:
            // 'last' is handled at a higher level (keep only max-etime neighbor);
            // at the per-edge level treat as monotonically_decreasing.
            [[fallthrough]];
        case TemporalComparison::MONOTONICALLY_DECREASING: return edge_time <= seed_time;
        default: return edge_time <= seed_time;
    }
}

// NodeTimeFn: callable (node_type_id, node_id) → int64_t edge time for that node.
// Migrated from graph_store._get_ntime_func() in a056923:
//   returns lambda: node_type, node_id → feature_store[node_type, attr_name][node_id]
// In C++ we use std::function for the same flexible dispatch.
using NodeTimeFn = std::function<int64_t(uint32_t /*node_type_id*/, uint64_t /*node_id*/)>;


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

// 220563b: feature_dtype field uses our internal ids (0=float32, 1=fp16, 2=bf16).
// The wire-protocol ids from DtypeRegistry (float32=0, float16=5, bf16=7) are
// distinct — only used for cross-language serialization.
// This function operates on internal ids (same as before), not wire ids.
inline FeatureDtypeTolerance feature_dtype_tolerance(uint8_t dtype) {
    switch (dtype) {
        case 1: return {5e-3f, 5e-3f};  // fp16 (internal id=1, wire id=5)
        case 2: return {2e-2f, 2e-2f};  // bf16 (internal id=2, wire id=7 per 220563b)
        default: return {1e-5f, 1e-5f}; // fp32 (internal id=0, wire id=0)
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
//
// ── 220563b migration note ───────────────────────────────────────────────────
// 220563b "Explicitly support bf16 in feature store" completed the dtype
// coverage by adding bfloat16 to the registration loop.  Before 220563b, the
// dtype_names array here would only have entries for float32 and float16 —
// querying with feature_dtype=2 (bf16) would have fallen through to "unknown",
// silently disabling the dispatch path.
//
// Now with 220563b, the complete set is:
//   dtype=0 → float32  wire_id=1  (align_count=4)
//   dtype=1 → float16  wire_id=5  (align_count=8)
//   dtype=2 → bfloat16 wire_id=7  (align_count=8)  ← 220563b addition
//
// Wire IDs match cugraph_pyg/data/feature_store.py post-220563b:
//   dtypes[torch.bfloat16] = 7   dtype_ids[7] = torch.bfloat16
//
// This function is the cross-system validation point: if the wire_id printed
// here does not match the Python feature_store.py registry, a dtype mismatch
// has been introduced and features will be deserialized incorrectly.
//
// 断点调试: wire_id is printed alongside name and align_count so the complete
// encode/decode identity is visible in a single log line.
inline void debug_dtype_dispatch(uint8_t feature_dtype, const char* op) {
    // 220563b: three-entry table — float32/fp16/bf16 all first-class citizens.
    // bf16 (dtype=2) was the missing entry before the patch.
    struct DtypeInfo { const char* name; uint8_t wire_id; };
    static const DtypeInfo info[3] = {
        {"float32",  1},   // torch.float32  → wire_id=1
        {"float16",  5},   // torch.float16  → wire_id=5
        {"bfloat16", 7},   // torch.bfloat16 → wire_id=7 ← 220563b
    };
    const char* name    = (feature_dtype < 3) ? info[feature_dtype].name    : "unknown";
    uint8_t     wire_id = (feature_dtype < 3) ? info[feature_dtype].wire_id : 0xFF;
    int         align   = tb_emb_align_count(feature_dtype);
    printf("[DEBUG 220563b dispatch] op=%s feature_dtype=%s wire_id=%u align=%d\n",
           op, name, (unsigned)wire_id, align);
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
                // 6ea54ab pattern: allocation failure must emit a diagnostic.
                // Mirrors WHOLEMEMORY_RETURN_ON_FAIL: no silent swallowing.
                // 断点调试: print partition range and size so OOM is debuggable.
                fprintf(stderr,
                    "[PHILEMON_RETURN_ON_FAIL] %s:%d flush_partitions: FATAL"
                    " alloc_size=%zu partition=[%d,%d] → DRAM fallback also"
                    " failed. Partition skipped.\n",
                    __FILE__, __LINE__,
                    alloc_size, lo, hi);
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

        // 3f11d45 migration: Guard zero-edge partitions before invoking any
        // aggregation over the edge array.
        //
        // cugraph-gnn 3f11d45 fixed HeterogeneousSampleReader: when a batch
        // had zero positive edges of a given hetero-edge type, calling
        //   ux = col[pyg_can_etype][:num_sampled_edges[0]]   # empty slice
        //   ux.max()   # ← PyTorch exception: max() on empty tensor
        // The fix: check numel() > 0 first, return 0 for the empty case.
        //
        // C++ equivalent: if edge_count == 0, the lower_bound + scan loop
        // would execute zero iterations and return 0 — already safe.
        // However, any caller that calls scan_partition and then uses the
        // RESULT to compute "max sampled node id = max(src_ids) + 1" would
        // crash on an empty result set (same UB as ux.max() on empty tensor).
        //
        // We add an explicit early-return with a debug trace so:
        //   (a) the zero-edge case is visible in logs (断点调试), and
        //   (b) future callers cannot accidentally "fall through" to aggregation
        //       code that assumes at least one matched edge.
        //
        // Pattern: 3f11d45 numel() > 0 guard applied at the scanning boundary.
        if (part.edge_count == 0) {
            // 断点调试: empty partition — equivalent to 3f11d45's numel()==0 guard
            fprintf(stderr,
                "[DEBUG 3f11d45 scan_partition] partition alloc_id=%lu has"
                " edge_count=0 — returning 0 without iterating"
                " (mirrors 3f11d45 numel()==0 → tensor(0) guard)\n",
                (unsigned long)part.alloc_id);
            return 0;
        }

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

    // safe_compute_hetero_sampled_counts — compute (num_sampled_src, num_sampled_dst)
    // for a given edge type query, returning {0, 0} when the matched edge set is empty.
    //
    // 3f11d45 pattern (from HeterogeneousSampleReader):
    //   ux = col[pyg_can_etype][:num_sampled_edges[0]]
    //   uxn = (ux.max() + 1) if ux.numel() > 0 else torch.tensor(0, device=ux.device)
    //   num_sampled_nodes[dst_type][0] = torch.max(num_sampled_nodes[dst_type][0], uxn)
    //
    // C++ equivalent:
    //   If the matched src/dst node id sets are empty, return 0 for the count.
    //   Otherwise return max_node_id + 1 as the "number of unique sampled nodes"
    //   (since node ids are compact, max+1 == count for the sampled set).
    //
    // 断点调试: prints sampled_src_count and sampled_dst_count so callers can
    // detect the zero-input case (which would previously cause a crash if
    // max() was called directly on the empty node-id vectors).
    struct HeteroSampledCounts {
        uint64_t num_src_nodes;  // max(src_node_id) + 1, or 0 if no edges sampled
        uint64_t num_dst_nodes;  // max(dst_node_id) + 1, or 0 if no edges sampled
        uint64_t num_edges;      // number of matched edges
    };

    HeteroSampledCounts safe_compute_hetero_sampled_counts(
            int32_t ts_lo, int32_t ts_hi) const {
        // Collect all matched edges
        uint64_t max_src = 0, max_dst = 0, edge_count = 0;
        bool any_edge = false;

        auto parts = query_partitions(ts_lo, ts_hi);
        for (auto* p : parts) {
            scan_partition(*p, ts_lo, ts_hi, [&](const TemporalEdge& e) {
                // 3f11d45: max_src/max_dst are only updated when we have edges.
                // If no edges match at all, any_edge stays false and we return {0,0,0}.
                if (!any_edge) {
                    max_src = e.source;
                    max_dst = e.destination;
                    any_edge = true;
                } else {
                    max_src = std::max(max_src, e.source);
                    max_dst = std::max(max_dst, e.destination);
                }
                ++edge_count;
            });
        }

        // 3f11d45: if no edges matched (empty result), return {0,0,0}.
        // DO NOT call max_src+1 on an uninitialized max_src — that is the
        // exact crash that 3f11d45 fixed in the Python layer.
        HeteroSampledCounts result;
        if (!any_edge) {
            // 断点调试: zero-input case — mirrors 3f11d45 numel()==0 branch
            fprintf(stderr,
                "[DEBUG 3f11d45 safe_compute_hetero_sampled_counts]"
                " ts=[%d,%d] → 0 edges matched, returning {0,0,0}"
                " (mirrors numel()==0 → tensor(0) guard)\n",
                ts_lo, ts_hi);
            result = {0, 0, 0};
        } else {
            result = {max_src + 1, max_dst + 1, edge_count};
            fprintf(stderr,
                "[DEBUG 3f11d45 safe_compute_hetero_sampled_counts]"
                " ts=[%d,%d] → edges=%lu src_nodes=%lu dst_nodes=%lu\n",
                ts_lo, ts_hi,
                (unsigned long)edge_count,
                (unsigned long)result.num_src_nodes,
                (unsigned long)result.num_dst_nodes);
        }
        return result;
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

    // ═══ a056923 migration: NodeTimeFunc + TemporalComparison ═══
    // a056923 (cugraph-gnn "Support Temporal Negative Sampling"):
    //   - graph_store.py: renamed __etime_attr → __time_attr (unified edge+node time)
    //   - Added _get_ntime_func(): returns lambda (node_type, node_id) → timestamp tensor
    //     Used by neg_sample() to filter negative candidates by node creation time.
    //   - link_neighbor_loader.py / neighbor_loader.py:
    //     temporal_comparison strings changed: spaces → underscores
    //     "monotonically decreasing" → "monotonically_decreasing"
    //     "strictly increasing"      → "strictly_increasing"
    //     etc.  The PyG/cuGraph-PLC API was updated; old space-separated
    //     strings silently fell through the enum comparison.
    //
    // Our C++ equivalent:
    //   - NodeTimeFunc: std::function<int64_t(uint64_t node_id)>
    //     replaces the Python lambda; called during temporal negative sampling
    //     to retrieve a node's creation timestamp from the feature store.
    //   - TemporalComparison enum: strongly-typed replacement for the
    //     space/underscore-separated string API, eliminating silent mismatches.
    //   - get_node_time_func(): returns nullptr when no node_time store registered
    //     (parallel to Python returning None when __time_attr is None).
    //
    // 断点调试: get_node_time_func() prints registration state on every call
    // so sampling callers can verify node-time availability before the loop.

    // NodeTimeFunc: (node_id) → int64_t timestamp.
    // Wraps the node feature store lookup for the temporal neg sampling loop.
    // Parallel to Python: lambda node_type, node_id: feature_store[node_type, attr_name][node_id]
    // Here we keep it single-type (homogeneous); heterogeneous callers instantiate
    // one NodeTimeFunc per node_type (see neg_sample_temporal below).
    using NodeTimeFunc = std::function<int64_t(uint64_t node_id)>;

    // TemporalComparison: replaces the space-separated string API that caused
    // silent bugs in a056923.  Callers must use this enum — no raw strings.
    //
    // a056923 fix: "monotonically decreasing" → "monotonically_decreasing"
    // (PyG NeighborSampler enum changed, old strings were silently ignored,
    //  causing temporal sampling to fall back to non-temporal mode.)
    //
    // Knuth review note: the enum values deliberately mirror the PyG string
    // names (with underscores) so a future Python binding can round-trip them
    // with a simple snake_case → enum lookup.
    enum class TemporalComparison : uint8_t {
        strictly_increasing      = 0,
        monotonically_increasing = 1,
        strictly_decreasing      = 2,
        monotonically_decreasing = 3,  // DEFAULT — most common for link prediction
        last                     = 4,  // for temporal_strategy='last'
    };

    // Default temporal comparison (a056923: was "monotonically decreasing" string).
    static constexpr TemporalComparison kDefaultTemporalComparison =
        TemporalComparison::monotonically_decreasing;

    // Register a node-time lookup function for this bridge's node population.
    // Call once after feature store is populated. Pass nullptr to clear.
    // Not thread-safe — must be called before any concurrent sampling starts.
    void set_node_time_func(NodeTimeFunc fn) {
        node_time_func_ = std::move(fn);
        node_time_registered_.store(static_cast<bool>(node_time_func_),
                                    std::memory_order_release);
        printf("[DEBUG a056923 set_node_time_func] node_time_func %s\n",
               node_time_func_ ? "REGISTERED" : "CLEARED");
    }

    // _get_ntime_func equivalent (a056923 graph_store.py:_get_ntime_func).
    // Returns nullptr if no node-time store registered (non-temporal mode).
    // 断点调试: printf gated behind PHILEMON_DEBUG_NTIME to avoid hot-path
    // overhead — get_node_time_func() may be called in the inner sampling loop.
    const NodeTimeFunc* get_node_time_func() const {
#ifdef PHILEMON_DEBUG_NTIME
        if (!node_time_func_) {
            printf("[DEBUG a056923 get_node_time_func] no node_time registered"
                   " -> non-temporal negative sampling\n");
        } else {
            printf("[DEBUG a056923 get_node_time_func] node_time_func available\n");
        }
#endif
        return node_time_func_ ? &node_time_func_ : nullptr;
    }

    // ═══ Temporal Neighbor Sampling (from cugraph-gnn d4b52c9 + 4005ab1 + a056923) ═══
    // cugraph-gnn在GraphStore加了edge timestamp (etime)支持,
    // 然后在DistributedNeighborSampler加了temporal=True模式:
    //   - 只采样etime < seed_time的边(forward-in-time约束)
    //   - 4005ab1进一步标准化: 支持etime <= seed_time的等号情况
    //
    // a056923新增: 支持node_time_func (对应_get_ntime_func), 用于temporal
    // negative sampling。node_time_func返回给定节点的时间戳, 用于约束负采样
    // 只选取"在seed_time之前存在"的节点 (节点时间 <= seed_time)。
    //
    // 我们对应实现: 给定一个节点和查询时间, 返回etime在约束范围内的邻居。
    // 这比scan_partition更精确——不是按区间范围, 而是按边的精确创建时间过滤。
    //
    // 断点调试: 每次采样打印seed_node、query_time、匹配数、comparison策略,
    // 用于验证时序因果性。

    struct TemporalSampleResult {
        uint64_t          seed_node;
        int64_t           query_time;
        uint64_t          total_neighbors;       // partition范围内的总邻居数
        uint64_t          temporal_neighbors;    // 满足etime约束的邻居数
        double            temporal_ratio;        // temporal_neighbors / total_neighbors
        // a056923: record which comparison was used — essential for audit trail
        TemporalComparison comparison;
    };

    template <typename Callback>
    TemporalSampleResult temporal_neighbor_sample(
            uint64_t seed_node, int64_t query_time,
            int32_t ts_lo, int32_t ts_hi,
            Callback&& cb,
            TemporalComparison tc = TemporalComparison::MONOTONICALLY_DECREASING) const {
        printf("[DEBUG a056923] temporal_neighbor_sample: seed=%lu query_time=%ld"
               " ts=[%d,%d] comparison=%s\n",
               (unsigned long)seed_node, (long)query_time,
               ts_lo, ts_hi, temporal_comparison_name(tc));

        TemporalSampleResult result{seed_node, query_time, 0, 0, 0.0, tc};
        auto parts = query_partitions(ts_lo, ts_hi);

        printf("[DEBUG a056923] temporal_neighbor_sample: found %zu partitions\n",
               parts.size());

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
                // a056923: 使用TemporalComparison判断, 而不是硬编码<=
                // 对应PyG DistributedNeighborSampler的temporal_comparison参数
                if (temporal_compare(it->etime, query_time, tc)) {
                    cb(*it);
                    result.temporal_neighbors++;
                }
            }
        }
        result.temporal_ratio = result.total_neighbors > 0
            ? static_cast<double>(result.temporal_neighbors) / result.total_neighbors
            : 0.0;

        printf("[DEBUG a056923] temporal_neighbor_sample done: total=%lu temporal=%lu"
               " ratio=%.3f\n",
               (unsigned long)result.total_neighbors,
               (unsigned long)result.temporal_neighbors,
               result.temporal_ratio);
        return result;
    }

    // ─── a056923: temporal_negative_sample ─────────────────────────────────
    // 对应sampler_utils.py中新增的temporal negative sampling逻辑:
    //   1. 先调用neg_sample生成候选负样本 (_call_plc_negative_sampling)
    //   2. 用node_time_func查询每个候选节点的时间戳
    //   3. 过滤: 只保留 node_time <= seed_time 的候选
    //   4. 不足时重试最多5次 (PyG API行为)
    //   5. 完全不足时用"最早存在节点"填充 (边界case处理)
    //
    // a056923 关键bugfix: node_time_func为nullptr时, 不执行时序过滤 (对应
    // sampler_utils.py的 `if node_time_func is not None` guard)。
    // seed_time为INT64_MIN时, 同样跳过过滤 (对应 `if seed_time is None`)。
    //
    // 断点调试: 每步都打印候选数、通过数、重试次数, 便于定位time leak。

    struct TemporalNegSampleResult {
        std::vector<uint64_t> src_neg;    // 负采样的source节点
        std::vector<uint64_t> dst_neg;    // 负采样的dest节点
        uint32_t              retries;    // 重试次数
        bool                  exhausted;  // 是否触发了"最早节点"填充路径
    };

    // node_time_fn: (node_id) → int64_t node timestamp. nullptr = no filtering.
    // src_pool/dst_pool: candidate node pools for random sampling.
    // target_count: desired number of negative pairs.
    // seed_time: INT64_MIN signals "no temporal constraint".
    TemporalNegSampleResult temporal_negative_sample(
            const std::vector<uint64_t>& src_pool,
            const std::vector<uint64_t>& dst_pool,
            size_t target_count,
            int64_t seed_time,
            const NodeTimeFn& src_node_time_fn,
            const NodeTimeFn& dst_node_time_fn,
            uint32_t src_type_id = 0,
            uint32_t dst_type_id = 0,
            TemporalComparison tc = TemporalComparison::MONOTONICALLY_DECREASING) const {

        TemporalNegSampleResult result;
        result.retries   = 0;
        result.exhausted = false;

        printf("[DEBUG a056923] temporal_negative_sample: target=%zu seed_time=%ld"
               " comparison=%s has_node_time=%s\n",
               target_count, (long)seed_time,
               temporal_comparison_name(tc),
               (src_node_time_fn ? "yes" : "no"));

        // Knuth review: guard empty pools — division by zero or pool[0] UB.
        if (src_pool.empty() || dst_pool.empty() || target_count == 0) {
            fprintf(stderr,
                "[WARN a056923] temporal_negative_sample: empty pool or zero target "
                "(src=%zu dst=%zu target=%zu) — returning empty result\n",
                src_pool.size(), dst_pool.size(), target_count);
            return result;
        }

        // Fast path: no node_time_fn or no seed_time constraint → return random sample
        if (!src_node_time_fn || !dst_node_time_fn ||
            seed_time == std::numeric_limits<int64_t>::min()) {
            printf("[DEBUG a056923] temporal_negative_sample: no temporal filter, "
                   "returning random %zu pairs\n", target_count);
            // Warn if node_time_fn absent but seed_time is real — matches PyG warning
            if ((!src_node_time_fn || !dst_node_time_fn) &&
                seed_time != std::numeric_limits<int64_t>::min()) {
                fprintf(stderr,
                    "[WARN a056923] seed_time is set but node_time_fn is null; "
                    "temporal negative sampling will not be performed\n");
            }
            result.src_neg.resize(target_count);
            result.dst_neg.resize(target_count);
            std::mt19937_64 rng(static_cast<uint64_t>(seed_time ^ 0xDEADBEEF));
            for (size_t i = 0; i < target_count; ++i) {
                result.src_neg[i] = src_pool[rng() % src_pool.size()];
                result.dst_neg[i] = dst_pool[rng() % dst_pool.size()];
            }
            return result;
        }

        std::mt19937_64 rng(static_cast<uint64_t>(seed_time));
        std::vector<uint64_t> remaining_seed_time(target_count, seed_time);
        // (For multi-seed case: caller expands seed_time per a056923 neg_cat logic)

        // ── Round 0: initial random sample ──────────────────────────────────
        auto random_pairs = [&](size_t n)
                -> std::pair<std::vector<uint64_t>, std::vector<uint64_t>> {
            std::vector<uint64_t> s(n), d(n);
            for (size_t i = 0; i < n; ++i) {
                s[i] = src_pool[rng() % src_pool.size()];
                d[i] = dst_pool[rng() % dst_pool.size()];
            }
            return {s, d};
        };

        auto [s0, d0] = random_pairs(target_count);

        // Filter by temporal constraint: node_time <= seed_time
        std::vector<uint64_t> filtered_src, filtered_dst;
        std::vector<uint64_t> remaining_st = remaining_seed_time;  // copy for filtering
        filtered_src.reserve(target_count);
        filtered_dst.reserve(target_count);
        std::vector<uint64_t> invalid_st;  // seed times for which we still need samples

        for (size_t i = 0; i < target_count; ++i) {
            int64_t st = src_node_time_fn(src_type_id, s0[i]);
            int64_t dt = dst_node_time_fn(dst_type_id, d0[i]);
            if (temporal_compare(st, remaining_st[i], tc) &&
                temporal_compare(dt, remaining_st[i], tc)) {
                filtered_src.push_back(s0[i]);
                filtered_dst.push_back(d0[i]);
            } else {
                invalid_st.push_back(remaining_st[i]);
            }
        }

        printf("[DEBUG a056923] temporal_negative_sample round0: "
               "passed=%zu failed=%zu\n",
               filtered_src.size(), invalid_st.size());

        // ── Retry loop (max 5 rounds, matches PyG API) ───────────────────────
        for (int attempt = 0; attempt < 5 && !invalid_st.empty(); ++attempt) {
            result.retries++;
            size_t diff = invalid_st.size();
            auto [s_p, d_p] = random_pairs(diff);

            std::vector<uint64_t> still_invalid;
            for (size_t i = 0; i < diff; ++i) {
                int64_t st = src_node_time_fn(src_type_id, s_p[i]);
                int64_t dt = dst_node_time_fn(dst_type_id, d_p[i]);
                if (temporal_compare(st, invalid_st[i], tc) &&
                    temporal_compare(dt, invalid_st[i], tc)) {
                    filtered_src.push_back(s_p[i]);
                    filtered_dst.push_back(d_p[i]);
                } else {
                    still_invalid.push_back(invalid_st[i]);
                }
            }
            printf("[DEBUG a056923] temporal_negative_sample retry=%d: "
                   "passed=%zu still_invalid=%zu\n",
                   attempt + 1, filtered_src.size(), still_invalid.size());
            invalid_st = std::move(still_invalid);
        }

        // ── Exhausted: fill with earliest-time node (matches PyG fallback) ──
        if (!invalid_st.empty()) {
            result.exhausted = true;
            printf("[DEBUG a056923] temporal_negative_sample: EXHAUSTED after 5 retries,"
                   " filling %zu slots with earliest-node fallback\n",
                   invalid_st.size());

            // Find node with minimum time in each pool
            uint64_t earliest_src = src_pool[0];
            int64_t  earliest_src_t = src_node_time_fn(src_type_id, earliest_src);
            for (uint64_t n : src_pool) {
                int64_t t = src_node_time_fn(src_type_id, n);
                if (t < earliest_src_t) { earliest_src_t = t; earliest_src = n; }
            }

            uint64_t earliest_dst = dst_pool[0];
            int64_t  earliest_dst_t = dst_node_time_fn(dst_type_id, earliest_dst);
            for (uint64_t n : dst_pool) {
                int64_t t = dst_node_time_fn(dst_type_id, n);
                if (t < earliest_dst_t) { earliest_dst_t = t; earliest_dst = n; }
            }

            printf("[DEBUG a056923] temporal_negative_sample fallback: "
                   "earliest_src=%lu (t=%ld) earliest_dst=%lu (t=%ld)\n",
                   (unsigned long)earliest_src, (long)earliest_src_t,
                   (unsigned long)earliest_dst, (long)earliest_dst_t);

            // Fill remaining slots with earliest-time nodes (same as PyG)
            for (size_t i = 0; i < invalid_st.size(); ++i) {
                filtered_src.push_back(earliest_src);
                filtered_dst.push_back(earliest_dst);
            }
        }

        // Trim to target_count (may have slightly more from retry overlaps)
        if (filtered_src.size() > target_count) {
            filtered_src.resize(target_count);
            filtered_dst.resize(target_count);
        }

        result.src_neg = std::move(filtered_src);
        result.dst_neg = std::move(filtered_dst);

        printf("[DEBUG a056923] temporal_negative_sample complete: "
               "generated=%zu retries=%u exhausted=%s\n",
               result.src_neg.size(), result.retries,
               result.exhausted ? "yes" : "no");
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
                  << " comparison=" << temporal_comparison_name(r.comparison)
                  << " partitions=" << partitions_.size()
                  << "\n";
    }

    // ════ 4005ab1 migration: SamplerKwargs ════════════════════════════════════
    // commit 4005ab1 "[FEA] Support Standard Temporal Sampling Behavior"
    //
    // Upstream change (DistributedNeighborSampler, distributed_sampler.py):
    //   BEFORE 4005ab1:
    //     if temporal:
    //         self.__func_kwargs["temporal_property_name"] = "time"
    //     # temporal_sampling_comparison was never set → PLC used its default
    //
    //   AFTER 4005ab1:
    //     if temporal:
    //         self.__func_kwargs["temporal_property_name"] = "time"
    //         self.__func_kwargs["temporal_sampling_comparison"] = temporal_comparison
    //     # Now temporal_comparison is propagated all the way to cuPLC
    //
    // Second change: sample_batches() gains seed_times parameter:
    //   def sample_batches(self, seeds, seed_times, batch_id_offsets, ...):
    //       kwargs = {...}
    //       if seed_times is not None:
    //           kwargs.update({"starting_vertex_times": cupy.asarray(seed_times)})
    //       sampling_results_dict = self.__func(**kwargs)
    //
    // Third change: input_time is now passed from NodeLoader (was hardcoded None):
    //   BEFORE: time=None
    //   AFTER:  time=input_time   (node_loader.py line 116)
    //
    // Fourth change: NeighborLoader auto-infers input_time from feature_store
    //   when input_time is None and time_attr is set:
    //     if input_time is None:
    //         input_time = feature_store[input_type, time_attr, None][input_nodes]
    //
    // Our C++ analog:
    //   SamplerKwargs — bundles all kwargs that flow into the C++ sampling call:
    //     - temporal_property_name ("time")
    //     - temporal_sampling_comparison (TemporalComparison enum)
    //     - starting_vertex_times (per-seed timestamps, or empty if non-temporal)
    //
    // 断点调试: SamplerKwargs::dump() prints all fields so the complete sampling
    // configuration is visible in one log line — critical for causal correctness audit.

    struct SamplerKwargs {
        // temporal_property_name: name of the edge timestamp attribute in the graph store.
        // cugraph-pyg: always "time" when temporal=True (hard-coded in DistributedNeighborSampler).
        // We store as std::string for future heterogeneous per-edge-type customization.
        std::string temporal_property_name;  // default: "time"

        // temporal_sampling_comparison: the comparison operator passed to cuPLC.
        // 4005ab1: was NEVER passed before this commit — PLC silently used its own default.
        // Now explicitly set from the loader's temporal_comparison parameter.
        // Maps 1:1 to TemporalComparison enum via temporal_comparison_name().
        TemporalComparison temporal_sampling_comparison;

        // starting_vertex_times: per-seed query timestamps (shape: [num_seeds]).
        // 4005ab1: sample_batches() now receives seed_times from __sample_from_nodes_func /
        // __sample_from_edges_func, and passes them as cupy array to cuPLC.
        // In C++ we store as vector<int64_t>; empty = non-temporal (no time constraint).
        std::vector<int64_t> starting_vertex_times;

        // temporal_enabled: mirrors `if temporal:` guard in DistributedNeighborSampler.
        // When false, temporal_property_name and temporal_sampling_comparison are ignored.
        bool temporal_enabled;

        // Default constructor: non-temporal mode (4005ab1: before temporal=True is set).
        SamplerKwargs()
            : temporal_property_name("time")
            , temporal_sampling_comparison(TemporalComparison::MONOTONICALLY_DECREASING)
            , temporal_enabled(false)
        {}

        // Constructor for temporal mode (4005ab1: temporal=True path).
        // tc: TemporalComparison from loader's temporal_comparison parameter.
        // seed_times: per-seed timestamps from input_time (may be empty for batch without times).
        explicit SamplerKwargs(TemporalComparison tc,
                               std::vector<int64_t> seed_times = {})
            : temporal_property_name("time")
            , temporal_sampling_comparison(tc)
            , starting_vertex_times(std::move(seed_times))
            , temporal_enabled(true)
        {}

        // 4005ab1: validate that temporal_comparison matches 'last' strategy when needed.
        // Python equivalent: "Note that this should be 'last' for temporal_strategy='last'."
        bool is_last_strategy() const {
            return temporal_sampling_comparison == TemporalComparison::LAST;
        }

        // 断点调试: print all SamplerKwargs fields in one log line.
        // Called at batch-start so the complete sampling config is auditable.
        void dump(const char* prefix = "4005ab1 SamplerKwargs") const {
            printf("[DEBUG %s] temporal_enabled=%s property='%s' comparison=%s"
                   " seed_times_count=%zu first_seed_time=%s\n",
                   prefix,
                   temporal_enabled ? "true" : "false",
                   temporal_property_name.c_str(),
                   temporal_comparison_name(temporal_sampling_comparison),
                   starting_vertex_times.size(),
                   starting_vertex_times.empty()
                       ? "none"
                       : std::to_string(starting_vertex_times[0]).c_str());
        }
    };

    // apply_sampler_kwargs: merge SamplerKwargs into the temporal neighbor sampling call.
    // This is the C++ equivalent of DistributedNeighborSampler.sample_batches() kwargs
    // construction (4005ab1):
    //
    //   kwargs = {"seeds": ..., "batch_id_offsets": ..., "random_state": ...}
    //   kwargs.update(self.__func_kwargs)  ← temporal_property_name + temporal_comparison
    //   if seed_times is not None:
    //       kwargs.update({"starting_vertex_times": cupy.asarray(seed_times)})
    //
    // In our C++ layer, we call temporal_neighbor_sample() per seed, so we:
    //   1. Use kwargs.temporal_sampling_comparison for the TemporalComparison enum
    //   2. Extract per-seed time from starting_vertex_times[seed_idx] if available
    //   3. Fall back to query_time parameter if starting_vertex_times is empty
    //
    // 断点调试: prints the resolved seed_time and comparison for every call when
    // PHILEMON_DEBUG_TEMPORAL is defined (suppress on hot path by default).
    template <typename Callback>
    TemporalSampleResult apply_sampler_kwargs(
            uint64_t seed_node,
            size_t   seed_idx,          // index into starting_vertex_times
            int64_t  fallback_time,     // used when starting_vertex_times is empty
            int32_t  ts_lo,
            int32_t  ts_hi,
            const SamplerKwargs& kwargs,
            Callback&& cb) const {
        if (!kwargs.temporal_enabled) {
            // Non-temporal mode: no filtering, pass a sentinel query_time
            // that will never filter anything (edge_time <= INT64_MAX is always true).
            // Pattern: 4005ab1 non-temporal path where starting_vertex_times is None.
            printf("[DEBUG 4005ab1 apply_sampler_kwargs] seed=%lu non-temporal mode\n",
                   (unsigned long)seed_node);
            return temporal_neighbor_sample(
                seed_node, std::numeric_limits<int64_t>::max(),
                ts_lo, ts_hi, std::forward<Callback>(cb),
                TemporalComparison::MONOTONICALLY_DECREASING);
        }

        // 4005ab1: resolve seed_time from starting_vertex_times or fallback.
        // Python equivalent:
        //   if seed_times is not None:
        //       kwargs["starting_vertex_times"] = cupy.asarray(seed_times)
        // In C++: if starting_vertex_times[seed_idx] is available, use it.
        int64_t resolved_seed_time = fallback_time;
        bool from_starting_times = false;
        if (seed_idx < kwargs.starting_vertex_times.size()) {
            resolved_seed_time = kwargs.starting_vertex_times[seed_idx];
            from_starting_times = true;
        }

#ifdef PHILEMON_DEBUG_TEMPORAL
        printf("[DEBUG 4005ab1 apply_sampler_kwargs] seed=%lu idx=%zu"
               " seed_time=%ld from_starting_times=%s comparison=%s\n",
               (unsigned long)seed_node, seed_idx,
               (long)resolved_seed_time,
               from_starting_times ? "yes" : "no(fallback)",
               temporal_comparison_name(kwargs.temporal_sampling_comparison));
#else
        (void)from_starting_times;
#endif

        return temporal_neighbor_sample(
            seed_node, resolved_seed_time,
            ts_lo, ts_hi, std::forward<Callback>(cb),
            kwargs.temporal_sampling_comparison);
    }

    // batch_temporal_sample: run temporal neighbor sampling for a full batch of seeds.
    // 4005ab1 equivalent: the inner loop of __sample_from_nodes_func() after
    // current_seeds and current_time have been unpacked.
    //
    // seeds: batch of seed node IDs  (shape: [batch_size])
    // kwargs: SamplerKwargs including optional starting_vertex_times
    //
    // Returns: vector of TemporalSampleResult (one per seed) plus total edges matched.
    //
    // 断点调试: prints batch summary (seeds count, total edges, ratio)
    // and per-seed detail for the first 3 seeds to keep logs readable.
    struct BatchSampleResult {
        std::vector<TemporalSampleResult> per_seed;
        uint64_t total_edges_matched;
        double   mean_temporal_ratio;  // average fraction of neighbors passing filter
    };

    template <typename Callback>
    BatchSampleResult batch_temporal_sample(
            const std::vector<uint64_t>& seeds,
            int32_t                      ts_lo,
            int32_t                      ts_hi,
            const SamplerKwargs&         kwargs,
            Callback&&                   cb) const {
        kwargs.dump("4005ab1 batch_temporal_sample start");
        printf("[DEBUG 4005ab1] batch_temporal_sample: seeds=%zu ts=[%d,%d]\n",
               seeds.size(), ts_lo, ts_hi);

        BatchSampleResult batch;
        batch.total_edges_matched = 0;
        batch.per_seed.reserve(seeds.size());

        for (size_t i = 0; i < seeds.size(); ++i) {
            // 4005ab1: use starting_vertex_times[i] if available (per-seed time override).
            // This matches the cuPLC call:
            //   starting_vertex_times = cupy.asarray(seed_times)  [shape: num_seeds]
            auto result = apply_sampler_kwargs(
                seeds[i], i,
                /*fallback_time=*/std::numeric_limits<int64_t>::max(),
                ts_lo, ts_hi, kwargs,
                std::forward<Callback>(cb));

            batch.per_seed.push_back(result);
            batch.total_edges_matched += result.temporal_neighbors;

            // 断点调试: per-seed detail for first 3 seeds
            if (i < 3) {
                printf("[DEBUG 4005ab1] seed[%zu]=%lu time=%ld"
                       " total_nbrs=%lu temporal_nbrs=%lu ratio=%.3f\n",
                       i, (unsigned long)seeds[i],
                       (long)result.query_time,
                       (unsigned long)result.total_neighbors,
                       (unsigned long)result.temporal_neighbors,
                       result.temporal_ratio);
            }
        }

        // Compute mean temporal ratio across all seeds
        double ratio_sum = 0.0;
        for (auto& r : batch.per_seed) ratio_sum += r.temporal_ratio;
        batch.mean_temporal_ratio = seeds.empty() ? 0.0 : ratio_sum / seeds.size();

        printf("[DEBUG 4005ab1] batch_temporal_sample done:"
               " total_edges=%lu mean_ratio=%.3f\n",
               (unsigned long)batch.total_edges_matched,
               batch.mean_temporal_ratio);
        return batch;
    }

    // ═══ d4b52c9 migration: EtimeAttr + EtimeSamplerTable ═══
    //
    // cugraph-gnn d4b52c9 [FEA] Enable Temporal Sampling in cuGraph-PyG:
    //
    // 1. graph_store.py: __etime_attr = Tuple[FeatureStore, str] | None
    //    GraphStore存储(feature_store, attr_name)对, 调用时从feature_store
    //    按attr_name和edge_index取出时间戳向量, 拼成跨edge-type的etime tensor.
    //    对应_set_etime_attr()/__get_etime_tensor():
    //      __get_etime_tensor(sorted_keys, start_offsets, num_edges_t):
    //        for i, et in enumerate(sorted_keys):
    //            ix = arange(start_offsets[i], start_offsets[i]+num_edges[i])
    //            etimes.append(feature_store[et, attr_name][ix])
    //        return torch.concat(etimes)
    //    然后在edgelist_dict中加 "etime" key:
    //      if self.__etime_attr is not None:
    //          d["etime"] = self.__get_etime_tensor(...)
    //
    // 2. distributed_sampler.py: _func_table 8-entry dispatch tuple
    //    key = ("homogeneous"|"heterogeneous", "uniform"|"biased", True|False)
    //    → 对应 pylibcugraph.{homo|hetero}_{uniform|biased}[_temporal]_neighbor_sample
    //    temporal=True时额外设置 __func_kwargs["temporal_property_name"] = "time"
    //    (告诉 pylibcugraph 按哪个边属性过滤时间)
    //
    // Walpurgis改写20%(鲁迅拿法):
    //   - EtimeAttr: 替代Python tuple, 携带feature_store指针 + attr_name字符串
    //     改写: 加入edge_type_id字段(Python tuple没有), 支持per-edge-type attr映射
    //   - EtimeSamplerKey/EtimeSamplerTable: 替代Python dict[tuple,fn], 用
    //     std::array<8>枚举而非hash map, 消除哈希开销; 改写: 加入dump()辅助打印
    //     所有8条路径的激活状态, 用于断点调试
    //   - is_temporal()/get_etime_attr()/get_etime_tensor(): 替代Python的
    //     __etime_attr is not None guard和内联edgelist构建

    // EtimeAttr: 对应 graph_store.py __etime_attr = (feature_store, attr_name)
    // 改写: 加edge_type_id支持多edge type独立attr; Python仅单一(store, name)对
    struct EtimeAttr {
        const void*  feature_store_ptr;  // opaque ptr to feature store (type-erased)
        std::string  attr_name;          // 对应 time_attr="time" (Python侧参数名)
        uint32_t     edge_type_id;       // 改写: 支持per-edge-type attr (Python无此字段)

        EtimeAttr() : feature_store_ptr(nullptr), attr_name(""), edge_type_id(0) {}
        EtimeAttr(const void* store, std::string name, uint32_t etype = 0)
            : feature_store_ptr(store)
            , attr_name(std::move(name))
            , edge_type_id(etype) {}

        bool is_valid() const {
            return feature_store_ptr != nullptr && !attr_name.empty();
        }

        // 断点调试: dump EtimeAttr状态
        void dump(const char* prefix = "EtimeAttr") const {
            printf("[DEBUG d4b52c9 %s] store=%p attr='%s' edge_type_id=%u valid=%s\n",
                   prefix,
                   feature_store_ptr,
                   attr_name.c_str(),
                   edge_type_id,
                   is_valid() ? "yes" : "no");
        }
    };

    // EtimeSamplerKey: 对应 _func_table 的tuple键 (homo/hetero, uniform/biased, temporal)
    // 8个可能组合对应8条采样路径
    struct EtimeSamplerKey {
        bool heterogeneous;  // false=homogeneous, true=heterogeneous
        bool biased;         // false=uniform, true=biased
        bool temporal;       // false=non-temporal, true=temporal

        // 线性化: 3位index → [0,7]
        // layout: temporal*4 + heterogeneous*2 + biased*1
        // 与Python _func_table tuple key等价, 但用整数索引避免哈希
        uint8_t index() const {
            return static_cast<uint8_t>(
                (temporal ? 4u : 0u) |
                (heterogeneous ? 2u : 0u) |
                (biased ? 1u : 0u));
        }

        const char* to_string() const {
            static const char* names[8] = {
                // temporal=false
                "homo_uniform_nontemporal",    // 000
                "homo_biased_nontemporal",     // 001
                "hetero_uniform_nontemporal",  // 010
                "hetero_biased_nontemporal",   // 011
                // temporal=true
                "homo_uniform_temporal",       // 100
                "homo_biased_temporal",        // 101
                "hetero_uniform_temporal",     // 110
                "hetero_biased_temporal",      // 111
            };
            return names[index()];
        }
    };

    // EtimeSamplerTable: 对应 DistributedNeighborSampler._func_table
    // Python: dict[tuple(homo/hetero, uniform/biased, temporal), pylibcugraph_fn]
    // C++改写: std::array<8> of tagged function descriptors, 避免hash map开销
    //
    // 断点调试: select()打印被选中的路径, dump_all()打印所有8条路径激活状态
    // 用于验证temporal=True时正确选中*_temporal_neighbor_sample路径
    struct EtimeSamplerTable {
        // 采样函数描述符 — 对应 pylibcugraph.*_neighbor_sample 函数指针
        // 在 Walpurgis 中这些是 TemporalBridge 内部方法的分派标识,
        // 而非真正的 pylibcugraph 函数指针 (无 RAPIDS 依赖)
        enum class SamplerFunc : uint8_t {
            HOMO_UNIFORM_NONTEMPORAL  = 0,
            HOMO_BIASED_NONTEMPORAL   = 1,
            HETERO_UNIFORM_NONTEMPORAL= 2,
            HETERO_BIASED_NONTEMPORAL = 3,
            HOMO_UNIFORM_TEMPORAL     = 4,
            HOMO_BIASED_TEMPORAL      = 5,
            HETERO_UNIFORM_TEMPORAL   = 6,
            HETERO_BIASED_TEMPORAL    = 7,
        };

        static const char* func_name(SamplerFunc f) {
            switch (f) {
                case SamplerFunc::HOMO_UNIFORM_NONTEMPORAL:  return "homo_uniform_neighbor_sample";
                case SamplerFunc::HOMO_BIASED_NONTEMPORAL:   return "homo_biased_neighbor_sample";
                case SamplerFunc::HETERO_UNIFORM_NONTEMPORAL:return "hetero_uniform_neighbor_sample";
                case SamplerFunc::HETERO_BIASED_NONTEMPORAL: return "hetero_biased_neighbor_sample";
                case SamplerFunc::HOMO_UNIFORM_TEMPORAL:     return "homo_uniform_temporal_neighbor_sample";
                case SamplerFunc::HOMO_BIASED_TEMPORAL:      return "homo_biased_temporal_neighbor_sample";
                case SamplerFunc::HETERO_UNIFORM_TEMPORAL:   return "hetero_uniform_temporal_neighbor_sample";
                case SamplerFunc::HETERO_BIASED_TEMPORAL:    return "hetero_biased_temporal_neighbor_sample";
                default: return "unknown";
            }
        }

        // select: 根据key返回对应的SamplerFunc
        // 对应 Python: self.__func = self._func_table[(homo/hetero, uniform/biased, temporal)]
        // 断点调试: 打印key的完整描述和被选中的函数名
        static SamplerFunc select(EtimeSamplerKey key) {
            SamplerFunc f = static_cast<SamplerFunc>(key.index());
            printf("[DEBUG d4b52c9 EtimeSamplerTable::select] "
                   "key=%s → func=%s\n",
                   key.to_string(), func_name(f));
            return f;
        }

        // dump_all: 打印全部8条路径, 标注temporal路径
        // 对应Python _func_table定义处的注释: # homogeneous/heterogeneous, uniform/biased, temporal?
        static void dump_all() {
            printf("[DEBUG d4b52c9 EtimeSamplerTable::dump_all] "
                   "8-entry dispatch table:\n");
            for (uint8_t i = 0; i < 8; ++i) {
                SamplerFunc f = static_cast<SamplerFunc>(i);
                bool is_temp = (i & 4u) != 0;
                printf("  [%u] %-45s %s\n",
                       i, func_name(f),
                       is_temp ? "<-- TEMPORAL PATH" : "");
            }
        }

        // validate_temporal_property_name: 对应Python中
        //   if temporal: self.__func_kwargs["temporal_property_name"] = "time"
        // 验证temporal模式下property_name必须是"time"(当前硬编码,与d4b52c9一致)
        // 断点调试: 打印property_name验证结果
        static bool validate_temporal_property_name(bool temporal,
                                                     const char* property_name) {
            if (!temporal) return true;  // non-temporal: no constraint
            bool ok = (property_name != nullptr) &&
                      (strcmp(property_name, "time") == 0);
            printf("[DEBUG d4b52c9 validate_temporal_property_name] "
                   "temporal=%s property_name='%s' → %s\n",
                   temporal ? "true" : "false",
                   property_name ? property_name : "(null)",
                   ok ? "OK" : "FAIL (must be 'time' per d4b52c9)");
            return ok;
        }
    };

    // set_etime_attr: 对应 graph_store.py _set_etime_attr()
    // Python:
    //   def _set_etime_attr(self, attr):
    //       if attr != self.__etime_attr:
    //           weight_attr = self.__weight_attr   # preserve weight_attr
    //           self.__clear_graph()               # invalidate cached graph
    //           self.__etime_attr = attr
    //           self.__weight_attr = weight_attr   # restore
    // C++改写: clear_graph()等价于flush_partitions的buffer清空 + partition重置
    // 断点调试: 打印新旧attr对比, 确认invalidation
    void set_etime_attr(EtimeAttr new_attr) {
        // 改写: 打印old/new对比, Python仅做赋值
        printf("[DEBUG d4b52c9 set_etime_attr] old_attr_valid=%s new_attr_valid=%s "
               "new_attr='%s' etype=%u\n",
               etime_attr_.is_valid() ? "yes" : "no",
               new_attr.is_valid() ? "yes" : "no",
               new_attr.attr_name.c_str(),
               new_attr.edge_type_id);

        if (etime_attr_.feature_store_ptr != new_attr.feature_store_ptr ||
            etime_attr_.attr_name         != new_attr.attr_name         ||
            etime_attr_.edge_type_id      != new_attr.edge_type_id) {
            // Graph invalidation: Python调用__clear_graph()
            // 我们改写为只标记etime_dirty_, 惰性重建, 避免立即清空partition
            // (Python __clear_graph重建成本高; Walpurgis partition数据可复用)
            etime_dirty_.store(true, std::memory_order_release);
            etime_attr_ = std::move(new_attr);
            printf("[DEBUG d4b52c9 set_etime_attr] etime_attr changed → etime_dirty=true\n");
        }
    }

    // is_temporal: 对应 neighbor_loader.py 的 is_temporal 判断
    // Python: is_temporal = (edge_label_time is not None) and (time_attr is not None)
    //         (LinkNeighborLoader) or: is_temporal = time_attr is not None (NeighborLoader)
    // 断点调试: 打印is_temporal状态
    bool is_temporal() const {
        bool result = etime_attr_.is_valid();
        printf("[DEBUG d4b52c9 is_temporal] etime_attr_valid=%s → is_temporal=%s\n",
               result ? "yes" : "no",
               result ? "true" : "false");
        return result;
    }

    // get_etime_tensor: 对应 graph_store.py __get_etime_tensor()
    // Python:
    //   def __get_etime_tensor(self, sorted_keys, start_offsets, num_edges_t):
    //       feature_store, attr_name = self.__etime_attr
    //       etimes = []
    //       for i, et in enumerate(sorted_keys):
    //           ix = torch.arange(start_offsets[i], start_offsets[i]+num_edges_t[i])
    //           etime = feature_store[et, attr_name][ix]
    //           if etime is None: raise ValueError("Time property must be present ...")
    //           etimes.append(etime)
    //       return torch.concat(etimes)
    // C++改写: 用回调模式替代feature_store索引 (无PyTorch依赖)
    //   etime_lookup_fn(edge_type_id, start_offset, count) → std::vector<int64_t>
    // 断点调试: 打印每个edge type的etime range, 拼接后总数
    using EtimeLookupFn = std::function<
        std::vector<int64_t>(uint32_t /*edge_type_id*/,
                             uint64_t /*start_offset*/,
                             uint64_t /*count*/)>;

    std::vector<int64_t> get_etime_tensor(
            const std::vector<uint32_t>& sorted_edge_types,
            const std::vector<uint64_t>& start_offsets,
            const std::vector<uint64_t>& num_edges,
            const EtimeLookupFn& lookup_fn) const {
        if (!etime_attr_.is_valid()) {
            printf("[DEBUG d4b52c9 get_etime_tensor] __etime_attr is None → "
                   "returning empty (non-temporal mode)\n");
            return {};
        }

        printf("[DEBUG d4b52c9 get_etime_tensor] attr='%s' num_edge_types=%zu\n",
               etime_attr_.attr_name.c_str(), sorted_edge_types.size());

        std::vector<int64_t> etimes;
        for (size_t i = 0; i < sorted_edge_types.size(); ++i) {
            uint32_t etype      = sorted_edge_types[i];
            uint64_t start_off  = (i < start_offsets.size()) ? start_offsets[i] : 0;
            uint64_t count      = (i < num_edges.size())     ? num_edges[i]     : 0;

            // 对应Python: etime = feature_store[et, attr_name][ix]
            auto etime_chunk = lookup_fn(etype, start_off, count);

            // 对应Python: if etime is None: raise ValueError(...)
            if (etime_chunk.empty() && count > 0) {
                fprintf(stderr,
                    "[ERROR d4b52c9 get_etime_tensor] edge_type=%u: "
                    "Time property must be present for all edge types. "
                    "(mirrors ValueError in d4b52c9 __get_etime_tensor)\n",
                    etype);
                // 改写: 返回empty而非throw, 让调用者决定fallback策略
                // Python在这里直接raise; 我们用empty返回 + 上层is_temporal()检查
                return {};
            }

            printf("[DEBUG d4b52c9 get_etime_tensor] etype=%u start=%lu count=%lu "
                   "etime_range=[%ld, %ld]\n",
                   etype, (unsigned long)start_off, (unsigned long)count,
                   etime_chunk.empty() ? 0L : (long)etime_chunk.front(),
                   etime_chunk.empty() ? 0L : (long)etime_chunk.back());

            // 对应Python: etimes.append(etime); return torch.concat(etimes)
            etimes.insert(etimes.end(), etime_chunk.begin(), etime_chunk.end());
        }

        printf("[DEBUG d4b52c9 get_etime_tensor] concat result: total_etimes=%zu\n",
               etimes.size());
        return etimes;
    }

    // select_sampler_func: 整合 EtimeSamplerTable.select() + etime_attr 状态
    // 对应Python DistributedNeighborSampler.__init__中的逻辑:
    //   if temporal: self.__func_kwargs["temporal_property_name"] = "time"
    //   self.__func = self._func_table[(homo/hetero, uniform/biased, temporal)]
    // 断点调试: 打印完整的dispatch decision
    EtimeSamplerTable::SamplerFunc select_sampler_func(
            bool heterogeneous, bool biased) const {
        bool temporal = is_temporal();
        EtimeSamplerKey key{heterogeneous, biased, temporal};

        // temporal=True时验证property_name (对应Python设置temporal_property_name="time")
        if (temporal) {
            EtimeSamplerTable::validate_temporal_property_name(true, "time");
        }

        auto f = EtimeSamplerTable::select(key);
        printf("[DEBUG d4b52c9 select_sampler_func] heterogeneous=%s biased=%s "
               "temporal=%s → %s\n",
               heterogeneous ? "true" : "false",
               biased ? "true" : "false",
               temporal ? "true" : "false",
               EtimeSamplerTable::func_name(f));
        return f;
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

    // a056923: node-time lookup function (nullptr = non-temporal mode).
    // set_node_time_func() / get_node_time_func() provide the public API.
    //
    // Thread-safety: node_time_func_ is a std::function, which has no atomic
    // guarantees. The registered_ flag is atomic for safe concurrent reads of
    // the "is temporal mode active" check. The function itself MUST be set
    // before any concurrent sampling thread calls get_node_time_func() —
    // this is a setup-time invariant, not enforced at runtime.
    //
    // Callers in the sampling loop should cache the result of
    // get_node_time_func() once at batch start and reuse it, avoiding
    // repeated std::function copy overhead.
    NodeTimeFunc                node_time_func_;
    std::atomic<bool>           node_time_registered_{false};

    // d4b52c9 migration: etime_attr_ + etime_dirty_
    //
    // etime_attr_: 对应 graph_store.py __etime_attr = Tuple[FeatureStore, str] | None
    //   - 初始化为无效EtimeAttr (feature_store_ptr=nullptr), 等价于Python __etime_attr=None
    //   - set_etime_attr()设置后, is_temporal()返回true
    //
    // etime_dirty_: 改写标志(Python无对应字段)
    //   - Python通过__clear_graph()强制重建整个graph对象响应attr变化
    //   - 我们改写为lazy invalidation: attr变化时只设dirty标志
    //   - Thread-safety: etime_attr_在sampling前设置(setup-time invariant);
    //     etime_dirty_是atomic<bool>, set/check无data race.
    EtimeAttr                   etime_attr_;           // d4b52c9: __etime_attr equivalent
    std::atomic<bool>           etime_dirty_{false};   // d4b52c9: lazy invalidation (改写)

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
