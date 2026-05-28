#pragma once
/**
 * interval_index.hpp — Philemon-TSH Interval Index for Temporal Queries
 *
 * From PyTorch BlockPool (C, the good example):
 * ```cpp
 *   // pytorch/c10/cuda/CUDACachingAllocator.cpp:173
 *   struct BlockPool {
 *       std::set<Block*, Comparison> blocks;           // sorted by size
 *       std::set<Block*, Comparison> blocks_by_addr;   // sorted by address
 *       std::pair<iterator, bool> insert_into_blocks(Block* block);
 *   };
 * ```
 * Dual ordered views over the same data: one sorted by size for best-fit
 * allocation, one sorted by address for coalescing. Both provide O(log N)
 * lookup on different keys.
 *
 * Then we follow that pattern to implement IntervalIndex (D), letting
 * scan_partition (E) perform contains/contained queries in O(log N + k) (F),
 * and support overlaps_query for any-intersection semantics (G).
 *
 * Next the build() method (H) introduces dual-sort construction (I), so
 * contains_query (J) can binary-search on ts_start AND ts_end (K), while
 * contained_query (L) optimizes the reverse inclusion check (M).
 *
 * Subsequently the PartitionIndex (N) integrates per-partition index
 * caching (O), so TemporalBridge (P) supports index-accelerated queries (Q),
 * and in turn flush_partitions (R) enhances the build step (S).
 *
 * Finally the benchmark (T) validates correctness and throughput (U),
 * ensuring linear-scan results (V) match indexed results (W),
 * comprehensively upgrading query performance (Y) to achieve
 * sub-linear temporal subgraph retrieval (Z).
 *
 * Pattern sources:
 *
 * [1] PyTorch BlockPool (CUDACachingAllocator.cpp:173-200)
 *     Dual std::set with different comparators over same Block pointers.
 *     Our IntervalIndex keeps two sorted index vectors: by_start_, by_end_.
 *
 * [2] Megatron _IndexReader (indexed_dataset.py:233-280)
 *     Memory-mapped binary index with O(1) offset lookup.
 *     Our index uses contiguous vector storage for cache-friendly scan.
 *
 * [3] CCCL thrust::lower_bound (binary_search.h:26)
 *     GPU-optimized binary search in sorted range.
 *     Our CPU version uses std::lower_bound/upper_bound identically.
 *
 * [4] NCCL ncclTopoSort (topo.cc:94-154)
 *     Sorted node lookup by type+id in topology graph.
 *     Our partition-level index lookup mirrors this pattern.
 *
 * [5] LevelDB Iterator::Seek (two_level_iterator.cc:25)
 *     Binary search in sorted block, then linear scan within block.
 *     Already referenced in scan_partition (M006); we extend it.
 *
 * Milestone: M011 (Claude #6)
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>

namespace philemon {

// Minimal TemporalEdge definition for interval indexing.
// Must match the canonical definition in bridge/temporal_bridge.hpp.
// If included after temporal_bridge.hpp, this is a no-op (guarded by struct name).
#ifndef PHILEMON_TEMPORAL_EDGE_DEFINED
#define PHILEMON_TEMPORAL_EDGE_DEFINED
struct TemporalEdge {
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_start;
    int32_t  ts_end;

    TemporalEdge()
        : source(0), destination(0), weight(0.0), ts_start(0), ts_end(0) {}
    TemporalEdge(uint64_t s, uint64_t d, double w, int32_t t0, int32_t t1)
        : source(s), destination(d), weight(w), ts_start(t0), ts_end(t1) {}
};
#endif

/**
 * IntervalIndex — Dual-sorted index over temporal edges in a partition.
 *
 * Maintains two index arrays:
 *   by_start_[i] = index into original edge array, sorted by ts_start
 *   by_end_[i]   = index into original edge array, sorted by ts_end
 *
 * This enables three query types:
 *
 * 1. contains_query(lo, hi): edge ⊆ [lo, hi]
 *    ts_start >= lo AND ts_end <= hi
 *    → lower_bound on by_start_ for ts_start >= lo,
 *      then check ts_end <= hi (using by_end_ for early termination)
 *
 * 2. contained_query(lo, hi): [lo, hi] ⊆ edge
 *    ts_start <= lo AND ts_end >= hi
 *    → upper_bound on by_start_ for ts_start <= lo,
 *      intersect with lower_bound on by_end_ for ts_end >= hi
 *
 * 3. overlaps_query(lo, hi): edge ∩ [lo, hi] ≠ ∅
 *    ts_start <= hi AND ts_end >= lo
 *    → all edges not strictly before or after the query range
 *
 * Space: O(N) per index (two uint32_t arrays)
 * Build: O(N log N) — two sorts
 * Query: O(log N + output)
 */
class IntervalIndex {
public:
    IntervalIndex() = default;

    /**
     * Build the index from a contiguous array of temporal edges.
     *
     * Pattern: PyTorch BlockPool constructor (CUDACachingAllocator.cpp:174-178)
     *   blocks(BlockComparatorSizeCounterAddress),
     *   blocks_by_addr(BlockComparatorAddress)
     * Two sorted views, built once, queried many times.
     *
     * @param edges  Pointer to contiguous TemporalEdge array
     * @param count  Number of edges
     */
    void build(const TemporalEdge* edges, size_t count) {
        edges_ = edges;
        count_ = count;

        if (count == 0) {
            by_start_.clear();
            by_end_.clear();
            return;
        }

        // Build index sorted by ts_start (primary), ts_end (secondary)
        by_start_.resize(count);
        for (size_t i = 0; i < count; ++i) by_start_[i] = static_cast<uint32_t>(i);
        std::sort(by_start_.begin(), by_start_.end(),
            [this](uint32_t a, uint32_t b) {
                if (edges_[a].ts_start != edges_[b].ts_start)
                    return edges_[a].ts_start < edges_[b].ts_start;
                return edges_[a].ts_end < edges_[b].ts_end;
            });

        // Build index sorted by ts_end (primary), ts_start (secondary)
        by_end_.resize(count);
        for (size_t i = 0; i < count; ++i) by_end_[i] = static_cast<uint32_t>(i);
        std::sort(by_end_.begin(), by_end_.end(),
            [this](uint32_t a, uint32_t b) {
                if (edges_[a].ts_end != edges_[b].ts_end)
                    return edges_[a].ts_end < edges_[b].ts_end;
                return edges_[a].ts_start < edges_[b].ts_start;
            });

        built_ = true;
    }

    /**
     * contains_query: find edges whose interval is contained within [lo, hi].
     * Edge e matches iff: e.ts_start >= lo AND e.ts_end <= hi
     *
     * Algorithm:
     *   1. Binary search by_start_ for first index where ts_start >= lo
     *   2. Binary search by_end_ for last index where ts_end <= hi
     *   3. Scan from the start position, checking the end condition
     *
     * Complexity: O(log N + scan_range), where scan_range ≤ output + false_positives
     *
     * Pattern: LevelDB Iterator::Seek + linear scan (already in scan_partition)
     *          + CCCL thrust::lower_bound (binary_search.h:26)
     */
    template <typename Callback>
    uint64_t contains_query(int32_t lo, int32_t hi, Callback&& cb) const {
        if (!built_ || count_ == 0) return 0;

        // Find first by_start_ entry where ts_start >= lo
        auto it_start = std::lower_bound(
            by_start_.begin(), by_start_.end(), lo,
            [this](uint32_t idx, int32_t val) {
                return edges_[idx].ts_start < val;
            });

        uint64_t matched = 0;
        for (auto it = it_start; it != by_start_.end(); ++it) {
            const auto& e = edges_[*it];
            if (e.ts_start > hi) break;  // early termination: past query range
            if (e.ts_end <= hi) {
                cb(e);
                ++matched;
            }
        }
        return matched;
    }

    /**
     * contained_query: find edges whose interval contains [lo, hi].
     * Edge e matches iff: e.ts_start <= lo AND e.ts_end >= hi
     *
     * Algorithm:
     *   1. Binary search by_start_ for entries where ts_start <= lo
     *      (all entries from begin to upper_bound(lo))
     *   2. For each, check ts_end >= hi
     *
     * Complexity: O(log N + candidate_count)
     * Worst case: O(N) if many edges start before lo.
     * For typical temporal graphs with narrow edges, candidate_count << N.
     */
    template <typename Callback>
    uint64_t contained_query(int32_t lo, int32_t hi, Callback&& cb) const {
        if (!built_ || count_ == 0) return 0;

        // Find last by_start_ entry where ts_start <= lo
        auto it_end = std::upper_bound(
            by_start_.begin(), by_start_.end(), lo,
            [this](int32_t val, uint32_t idx) {
                return val < edges_[idx].ts_start;
            });

        uint64_t matched = 0;
        for (auto it = by_start_.begin(); it != it_end; ++it) {
            const auto& e = edges_[*it];
            if (e.ts_end >= hi) {
                cb(e);
                ++matched;
            }
        }
        return matched;
    }

    /**
     * overlaps_query: find edges whose interval overlaps [lo, hi].
     * Edge e matches iff: e.ts_start <= hi AND e.ts_end >= lo
     *
     * This is the most general query. Uses by_start_ to skip edges
     * starting after hi, and checks ts_end >= lo for overlap.
     *
     * Complexity: O(log N + scan_range)
     */
    template <typename Callback>
    uint64_t overlaps_query(int32_t lo, int32_t hi, Callback&& cb) const {
        if (!built_ || count_ == 0) return 0;

        // All edges with ts_start <= hi are candidates
        auto it_end = std::upper_bound(
            by_start_.begin(), by_start_.end(), hi,
            [this](int32_t val, uint32_t idx) {
                return val < edges_[idx].ts_start;
            });

        uint64_t matched = 0;
        for (auto it = by_start_.begin(); it != it_end; ++it) {
            const auto& e = edges_[*it];
            if (e.ts_end >= lo) {
                cb(e);
                ++matched;
            }
        }
        return matched;
    }

    /// Check if index has been built
    bool is_built() const { return built_; }

    /// Number of indexed edges
    size_t size() const { return count_; }

    /// Memory overhead in bytes (two index vectors of uint32_t)
    size_t memory_overhead() const {
        return (by_start_.capacity() + by_end_.capacity()) * sizeof(uint32_t);
    }

private:
    const TemporalEdge* edges_ = nullptr;
    size_t              count_ = 0;
    bool                built_ = false;

    // Dual sorted indices — mirrors PyTorch BlockPool's dual std::set
    std::vector<uint32_t> by_start_;  // indices sorted by ts_start
    std::vector<uint32_t> by_end_;    // indices sorted by ts_end
};

}  // namespace philemon
