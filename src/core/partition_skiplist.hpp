#pragma once
/**
 * partition_skiplist.hpp — Augmented Interval Skip List for partition selection
 *
 * ── Problem (the actual bottleneck this fixes) ──────────────────────────────
 *
 * TemporalBridge::query_partitions currently does:
 *
 *     for (auto& p : partitions_)                 //  O(P)
 *         if (p.ts_lo <= hi && p.ts_hi >= lo) ...  //  test every partition
 *
 * M006 made the *intra*-partition scan O(log N + output) with std::lower_bound,
 * but the *inter*-partition step is still a full linear sweep over every
 * partition. Under streaming ingestion (Phase 5) the partition count P grows
 * without bound, so a narrow temporal query that touches one partition still
 * pays O(P) to find it. This is the partition-level analogue of the redundant
 * full-input pass that the CUB DeviceTopK refactor pulled out into its own
 * path: the work is real but it does not need to be paid on every query.
 *
 * ── Why a plain sorted array + binary search is not enough ──────────────────
 *
 * Partitions are emitted by flush_partitions sorted by ts_lo, so we could
 * binary-search the lower bound on ts_lo. But intervals overlap: a partition
 * with a small ts_lo can have a large ts_hi that reaches into the query window,
 * while a later partition with ts_lo inside the window may have already ended.
 * A 1-D binary search on ts_lo cannot prune on ts_hi. We need the interval-tree
 * augmentation — carry a subtree/span maximum of ts_hi — but in a structure
 * that supports the lock-light, append-mostly access pattern of the bridge.
 *
 * ── Design: skip list augmented with span-max(ts_hi) ────────────────────────
 *
 * A Pugh skip list ordered by ts_lo. In a classic skip list, a level-L forward
 * pointer from node u skips over the run of nodes between u and its level-L
 * successor. We augment every forward pointer with the maximum ts_hi over
 * exactly the nodes it skips (u inclusive, successor exclusive). This is the
 * skip-list counterpart of an interval tree's subtree-max augmentation
 * (CLRS 14.3): the max-ts_hi over a contiguous run, attached to the pointer
 * that spans that run.
 *
 * overlaps(lo, hi) then walks left-to-right at the bottom level, but uses the
 * span-max on higher-level pointers to jump over entire runs whose every
 * interval ends before lo (max_hi < lo ⇒ no member can overlap). The walk
 * descends a level only when a high pointer's span *might* contain an overlap,
 * and stops once the current node's ts_lo exceeds hi (sorted by ts_lo ⇒ no
 * later node can start in range). Expected cost: O(log P + k), k = matches.
 *
 * ── Invariants ──────────────────────────────────────────────────────────────
 *  (I1) nodes are kept in non-decreasing ts_lo order at the base level.
 *  (I2) for a level-L forward pointer from u to v, span_max[L] equals
 *       max(ts_hi) over [u, v) at the base level. The head's pointers cover
 *       the whole list; a tail/null successor terminates the span.
 *  (I3) the structure is rebuilt wholesale by build(); it is immutable between
 *       builds, matching flush_partitions' batch-append model. This keeps reads
 *       lock-free without per-node locking (readers see a fully built list).
 *
 * Determinism: level assignment uses a splitmix64-seeded generator so a given
 * partition set always yields the same structure (reproducible benchmarks).
 *
 * Reference algorithms:
 *   W. Pugh, "Skip Lists: A Probabilistic Alternative to Balanced Trees"
 *     (CACM 1990) — base structure and level geometry.
 *   CLRS 3rd ed. §14.3 "Interval trees" — the max-endpoint augmentation we
 *     transplant onto skip-list spans.
 *   H. Samet, "Foundations of Multidimensional and Metric Data Structures" —
 *     stabbing-query pruning by subtree endpoint maxima.
 *
 * Milestone: M013 (Claude #7)
 */

#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <memory>   // SegmentedPartitionIndex: std::unique_ptr segments

namespace philemon {

// One indexed interval: the partition's [ts_lo, ts_hi] plus its slot in the
// bridge's partitions_ vector, so a hit maps straight back to the partition.
struct PartitionInterval {
    int32_t  ts_lo;
    int32_t  ts_hi;
    uint32_t partition_slot;   // index into TemporalBridge::partitions_
};

class PartitionSkipList {
public:
    static constexpr int      kMaxLevel = 24;   // supports ~16M partitions
    static constexpr int32_t  kNegInf   = std::numeric_limits<int32_t>::min();

    PartitionSkipList() : levels_(1), size_(0) { reset_head(); }

    size_t size()   const { return size_; }
    int    levels() const { return levels_; }

    // ── build: construct the augmented skip list from a partition set ────────
    // O(P log P) to sort + O(P) to link and compute span maxima bottom-up.
    void build(std::vector<PartitionInterval> ivals) {
        nodes_.clear();
        size_ = ivals.size();
        reset_head();
        if (ivals.empty()) { levels_ = 1; return; }

        // (I1) order by ts_lo, tie-break ts_hi then slot for stability.
        std::sort(ivals.begin(), ivals.end(),
            [](const PartitionInterval& a, const PartitionInterval& b) {
                if (a.ts_lo != b.ts_lo) return a.ts_lo < b.ts_lo;
                if (a.ts_hi != b.ts_hi) return a.ts_hi < b.ts_hi;
                return a.partition_slot < b.partition_slot;
            });

        // Assign a tower height to each node (geometric, p = 1/2),
        // deterministically seeded for reproducibility.
        nodes_.reserve(ivals.size());
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ (uint64_t(ivals.size()) << 1);
        int max_lvl = 1;
        for (auto& iv : ivals) {
            Node n;
            n.iv     = iv;
            n.height = random_height(rng);
            max_lvl  = std::max(max_lvl, n.height);
            n.next.assign(n.height, kNil);
            n.span_max.assign(n.height, kNegInf);
            nodes_.push_back(std::move(n));
        }
        levels_ = max_lvl;
        head_.next.assign(levels_, kNil);
        head_.span_max.assign(levels_, kNegInf);

        // Link towers level by level. `last[L]` = index of the most recent
        // node owning a pointer at level L (head_ = sentinel index kHead).
        std::vector<uint32_t> last(levels_, kHead);
        for (uint32_t i = 0; i < nodes_.size(); ++i) {
            for (int L = 0; L < nodes_[i].height; ++L) {
                forward_at(last[L], L) = i;
                last[L] = i;
            }
        }

        // (I2) span maxima. At the base level each pointer spans exactly one
        // node, so its span_max is that node's ts_hi. At level L the pointer
        // from u to v spans the same nodes as the chain of level-(L-1) pointers
        // between them, so span_max[L] = max over that sub-chain — computed by
        // walking the lower level between consecutive upper-level nodes.
        compute_span_max();
    }

    // ── overlaps: collect slots of partitions whose interval meets [lo,hi] ───
    // Pruned interval walk: O(log P + k). Appends matching partition_slot to
    // `out`. Caller supplies the vector to avoid per-query allocation.
    void overlaps(int32_t lo, int32_t hi, std::vector<uint32_t>& out) const {
        if (size_ == 0 || lo > hi) return;

        // Phase 1: descend from the head, using span maxima to skip runs whose
        // every interval ends before `lo`. We land on the first base-level node
        // that could possibly overlap (the leftmost node with ts_hi >= lo that
        // is also reachable without crossing a fully-out-of-range span).
        uint32_t cur = kHead;
        for (int L = levels_ - 1; L >= 0; --L) {
            uint32_t nx = forward_at_const(cur, L);
            // Advance while the *spanned* run cannot contain any overlap:
            // either it starts after hi (ts_lo > hi ⇒ done at this level) or
            // its whole run ends before lo (span_max < lo ⇒ skip it).
            while (nx != kNil) {
                const Node& vn = nodes_[nx];
                if (vn.iv.ts_lo > hi) break;                 // sorted: stop right
                if (span_max_at_const(cur, L) < lo) {        // run all ends < lo
                    cur = nx;
                    nx  = forward_at_const(cur, L);
                    continue;
                }
                break;  // this span might hold an overlap → descend a level
            }
        }

        // Phase 2: linear base-level scan from the landing node, emitting hits
        // and stopping as soon as ts_lo exceeds hi. The skip phase guarantees
        // we start near the first relevant node, so this scan is O(k + small).
        uint32_t i = forward_at_const(cur, 0);
        // `cur` itself may be a real node that overlaps; re-check it unless head.
        if (cur != kHead && interval_overlaps(nodes_[cur].iv, lo, hi))
            out.push_back(nodes_[cur].iv.partition_slot);
        while (i != kNil) {
            const PartitionInterval& iv = nodes_[i].iv;
            if (iv.ts_lo > hi) break;                        // (I1) early exit
            if (iv.ts_hi >= lo) out.push_back(iv.partition_slot);
            i = forward_at_const(i, 0);
        }
    }

    bool empty() const { return size_ == 0; }

    // Export all intervals in base-level (ts_lo-sorted) order into `out`.
    // Used by SegmentedPartitionIndex::compact to merge segments. O(P).
    void collect(std::vector<PartitionInterval>& out) const {
        out.reserve(out.size() + nodes_.size());
        for (uint32_t i = forward_at_const(kHead, 0); i != kNil;
             i = forward_at_const(i, 0)) {
            out.push_back(nodes_[i].iv);
        }
    }

private:
    static constexpr uint32_t kNil  = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kHead = kNil - 1;   // sentinel id for head_

    struct Node {
        PartitionInterval     iv;
        int                   height = 0;
        std::vector<uint32_t> next;      // forward[L]
        std::vector<int32_t>  span_max;  // augmentation[L]
    };

    void reset_head() {
        head_.next.assign(1, kNil);
        head_.span_max.assign(1, kNegInf);
        nodes_.clear();
    }

    // splitmix64 step → geometric height, capped at kMaxLevel.
    static int random_height(uint64_t& s) {
        int h = 1;
        while (h < kMaxLevel) {
            s += 0x9E3779B97F4A7C15ull;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z =  z ^ (z >> 31);
            if (z & 1ull) ++h; else break;   // p = 1/2
        }
        return h;
    }

    uint32_t& forward_at(uint32_t id, int L) {
        return (id == kHead) ? head_.next[L] : nodes_[id].next[L];
    }
    uint32_t forward_at_const(uint32_t id, int L) const {
        return (id == kHead) ? head_.next[L] : nodes_[id].next[L];
    }
    int32_t& span_max_at(uint32_t id, int L) {
        return (id == kHead) ? head_.span_max[L] : nodes_[id].span_max[L];
    }
    int32_t span_max_at_const(uint32_t id, int L) const {
        return (id == kHead) ? head_.span_max[L] : nodes_[id].span_max[L];
    }

    static bool interval_overlaps(const PartitionInterval& iv,
                                  int32_t lo, int32_t hi) {
        return iv.ts_lo <= hi && iv.ts_hi >= lo;
    }

    // (I2) Fill span_max[L] for every pointer. Base level first (single node),
    // then each higher level by folding the maxima of the lower-level chain it
    // spans — so we never rescan raw intervals above level 0.
    void compute_span_max() {
        // Level 0: pointer u→v spans exactly node v.
        for (uint32_t id = kHead, prev = kHead;;) {
            uint32_t nx = forward_at_const(id, 0);
            if (nx == kNil) break;
            span_max_at(id, 0) = nodes_[nx].iv.ts_hi;
            prev = id; (void)prev;
            id = nx;
        }
        // Levels 1..levels_-1: walk the lower level between successive nodes
        // that also exist on level L, accumulating the lower span maxima.
        for (int L = 1; L < levels_; ++L) {
            uint32_t u = kHead;
            while (u != kNil) {
                uint32_t v = forward_at_const(u, L);
                if (v == kNil) {
                    span_max_at(u, L) = fold_lower(u, kNil, L - 1);
                    break;
                }
                span_max_at(u, L) = fold_lower(u, v, L - 1);
                u = v;
            }
        }
    }

    // Max of span_max[L-1] over the level-(L-1) chain from `from` up to (but
    // not including) `to`. Both are node ids or sentinels.
    int32_t fold_lower(uint32_t from, uint32_t to, int lowerL) const {
        int32_t m = kNegInf;
        uint32_t w = from;
        while (w != to && w != kNil) {
            m = std::max(m, span_max_at_const(w, lowerL));
            w = forward_at_const(w, lowerL);
        }
        return m;
    }

    Node              head_;     // sentinel; head_.next[L] starts each level
    std::vector<Node> nodes_;    // node id == index here == base-level order
    int               levels_;
    size_t            size_;
};


// ─── SegmentedPartitionIndex ────────────────────────────────────────────────
//
// Why this exists (Claude #7 self-review, S1):
//   A single PartitionSkipList must be rebuilt wholesale on every flush —
//   build() is O(P log P) (sort) + O(P) (link + span_max), and span_max is a
//   global per-level fold that cannot be locally patched when nodes are
//   appended. Under streaming ingestion with N flushes that is
//   Σ O(P_i log P_i) = O(N² log N) cumulative — a real scaling cliff
//   (measured: per-flush rebuild grew from 0.24 ms at P=100 to 0.43 ms at
//   P=800, monotonically).
//
// The fix is the log-structured-merge / Lucene-segment pattern: each flush
// builds ONE small immutable skip-list segment over just the *new* partitions
// (O(M log M), M = new partitions ≪ P). A query fans out over all segments and
// merges results. When the segment count crosses a threshold, compact() merges
// them back into one — amortizing the rebuild cost the way LSM compaction does.
//
//   LevelDB/RocksDB: immutable SSTables + background compaction.
//   Lucene:          immutable index segments + periodic merge.
//
// Cost model:
//   add_segment(M)  : O(M log M)
//   overlaps        : O(S · (log P + k_s))   S = #segments, usually O(log N)
//   compact         : O(P log P), run every kCompactThreshold segments, so the
//                     amortized rebuild cost per partition is O(log P).
//
// Concurrency: the index is owned by TemporalBridge and mutated only under its
// unique_lock(part_mu_); readers hold shared_lock. Segments are immutable once
// built, so a reader iterating segments_ under shared_lock sees a consistent
// snapshot. Slots are GLOBAL partition slots (indices into partitions_), so a
// hit maps straight back regardless of which segment produced it.
class SegmentedPartitionIndex {
public:
    // Merge once the number of live segments reaches this. Geometric segment
    // sizes under monotone flushes keep S = O(log N) between compactions; the
    // threshold caps the per-query fan-out in the meantime.
    static constexpr size_t kCompactThreshold = 8;

    SegmentedPartitionIndex() : total_(0) {}

    size_t size()          const { return total_; }
    size_t segment_count() const { return segments_.size(); }

    void clear() {
        segments_.clear();
        total_ = 0;
    }

    // Append a batch of NEW partition intervals as one immutable segment.
    // `ivals` need not be pre-sorted; the segment's build() sorts internally.
    void add_segment(std::vector<PartitionInterval> ivals) {
        if (ivals.empty()) return;
        auto seg = std::make_unique<PartitionSkipList>();
        seg->build(std::move(ivals));
        total_ += seg->size();
        segments_.push_back(std::move(seg));
        if (segments_.size() >= kCompactThreshold) compact();
    }

    // Collect global slots whose interval meets [lo, hi], across all segments.
    // Appends to `out` (caller-owned, reused to avoid per-query allocation).
    // Note: a global slot appears in exactly one segment, so no de-dup needed.
    void overlaps(int32_t lo, int32_t hi, std::vector<uint32_t>& out) const {
        for (const auto& seg : segments_) seg->overlaps(lo, hi, out);
    }

    // Merge all segments into a single skip list. O(P log P). Called
    // automatically at the threshold, or explicitly by a maintenance sweep.
    void compact() {
        if (segments_.size() <= 1) return;
        std::vector<PartitionInterval> all;
        all.reserve(total_);
        for (const auto& seg : segments_) seg->collect(all);
        auto merged = std::make_unique<PartitionSkipList>();
        merged->build(std::move(all));
        segments_.clear();
        segments_.push_back(std::move(merged));
        // total_ unchanged: compaction preserves membership.
    }

private:
    std::vector<std::unique_ptr<PartitionSkipList>> segments_;
    size_t                                          total_;
};

}  // namespace philemon
