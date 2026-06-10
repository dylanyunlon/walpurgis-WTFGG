#pragma once
/**
 * migration_scheduler.hpp — Background thread for cross-tier data migration
 *
 * Follows NCCL's ncclTopoGraph ring/tree scheduling pattern:
 * periodically evaluate partition hotness, issue migrations between
 * HBM ↔ GDDR ↔ DRAM to keep hot temporal subgraphs close to compute.
 *
 * Also draws from Megatron's DistributedDataParallel._make_param_hook
 * pattern of deferred, batched operations.
 *
 * Milestone: M001–M004 (Claude #1), M005–M006 (Claude #2)
 *
 * 220563b migration: Explicitly support bf16 in migration bandwidth accounting
 *
 * cugraph-gnn commit 220563b added bfloat16 (id=7) to the feature store's
 * dtype↔id bidirectional table.  The key insight is that bf16 tensors have
 * element_size=2 (same as float16), so cross-tier migration bandwidth
 * accounting must use element_size=2 for bf16 — not fall back to float32=4.
 *
 * Before 220563b: bfloat16 had no id, so feature store partitions stored as
 * bf16 would be misidentified or silently treated as float32, doubling the
 * byte-count estimate and under-scheduling migrations.
 *
 * Our MigrationBandwidthBudget now explicitly handles all 8 registered ids
 * (220563b table: float32=0, int64=1, float64=2, int32=3, int16=4,
 *  float16=5, int8=6, bfloat16=7).  Byte counting uses the correct
 * element_size per dtype, preventing incorrect budget exhaustion for bf16.
 *
 * 断点调试: MigrationBandwidthBudget::record() prints dtype_id and byte delta
 * so any mismatch between estimated and actual migration bytes is immediately
 * visible in the scheduler's log output.
 */

#include "../core/tiered_allocator.hpp"
#include "../bridge/temporal_bridge.hpp"
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

namespace philemon {

// ─── 220563b migration: design commentary ───────────────────────────────────
// cugraph-gnn commit 220563b "Explicitly support bf16 in feature store":
//
//   The scheduler is not directly modified by 220563b, but is downstream-
//   affected: migrations now carry partitions that may store bfloat16 features
//   (wire_id=7 in the feature_store.py dtype registry).
//
//   Before 220563b: if a caller tried to store bf16 features and then trigger
//   a migration sweep, the migration would succeed at the memory level
//   (bytes are bytes), but the feature store decode on retrieval would fail
//   because dtype_ids had no entry for wire_id=7.
//
//   After 220563b: bf16 (wire_id=7) is a first-class registered dtype.
//   Migrations now support the full {float32, float16, bfloat16} dtype set,
//   matching the three EmbeddingDtype enum values in tiered_allocator.hpp.
//
//   MigrationScheduler::sweep_once() is agnostic to dtype (it calls
//   bridge_.migration_sweep() which calls allocator_.migrate()).  The dtype
//   metadata lives inside each SubgraphPartition's TemporalEdge::feature_dtype
//   field.  Migration preserves this field exactly (memcpy-based transfer),
//   so no special handling is needed here — bf16 support is inherited.
//
//   Diagnostic implication:
//   - stats_.total_migrations counts migrations across all dtypes.
//   - For per-dtype breakdown, see AsyncMigrationEngine::Stats::bytes_bfloat16
//     (async_migration.hpp), which was updated in 220563b to explicitly track
//     bf16 migration volume with wire_id=7 as the canonical identifier.
// ─────────────────────────────────────────────────────────────────────────────

struct MigrationStats {
    std::atomic<uint64_t> total_migrations{0};
    std::atomic<uint64_t> hbm_to_gddr{0};
    std::atomic<uint64_t> hbm_to_dram{0};
    std::atomic<uint64_t> gddr_to_hbm{0};
    std::atomic<uint64_t> gddr_to_dram{0};
    std::atomic<uint64_t> dram_to_hbm{0};
    std::atomic<uint64_t> dram_to_gddr{0};
    std::atomic<uint64_t> sweep_count{0};

    // 220563b: per-dtype migration byte counters.
    // Before 220563b, bf16 partitions were invisible here (no id registered).
    // Now id=7 has its own counter so monitoring can detect bf16 migration load.
    std::atomic<uint64_t> bytes_migrated_by_dtype[8]{};  // indexed by DtypeRegistry::ID_*

    void print() const {
        std::cout << "[MigrationStats] sweeps=" << sweep_count.load()
                  << " total_migrations=" << total_migrations.load()
                  << " HBM→GDDR=" << hbm_to_gddr.load()
                  << " HBM→DRAM=" << hbm_to_dram.load()
                  << " GDDR→HBM=" << gddr_to_hbm.load()
                  << " GDDR→DRAM=" << gddr_to_dram.load()
                  << " DRAM→HBM=" << dram_to_hbm.load()
                  << " DRAM→GDDR=" << dram_to_gddr.load()
                  << "\n";
        // 220563b: print per-dtype byte breakdown
        const char* id_names[] = {
            "float32", "int64", "float64", "int32",
            "int16",   "float16", "int8", "bfloat16"  // id=7 explicitly named
        };
        for (int i = 0; i < 8; ++i) {
            uint64_t b = bytes_migrated_by_dtype[i].load(std::memory_order_relaxed);
            if (b > 0) {
                std::cout << "  dtype[" << i << "]=" << id_names[i]
                          << " bytes_migrated=" << b << "\n";
            }
        }
    }

    // Record a migration, updating direction counters and dtype byte counter.
    // dtype_id: wire-protocol id from DtypeRegistry (bf16=7 is now valid).
    void record_migration(MemoryTier from, MemoryTier to,
                          size_t bytes, uint8_t dtype_id = DtypeRegistry::ID_FLOAT32) {
        total_migrations.fetch_add(1, std::memory_order_relaxed);
        if (dtype_id < 8) {
            bytes_migrated_by_dtype[dtype_id].fetch_add(bytes, std::memory_order_relaxed);
        }
        if (from == MemoryTier::HBM  && to == MemoryTier::GDDR) hbm_to_gddr.fetch_add(1, std::memory_order_relaxed);
        if (from == MemoryTier::HBM  && to == MemoryTier::DRAM) hbm_to_dram.fetch_add(1, std::memory_order_relaxed);
        if (from == MemoryTier::GDDR && to == MemoryTier::HBM)  gddr_to_hbm.fetch_add(1, std::memory_order_relaxed);
        if (from == MemoryTier::GDDR && to == MemoryTier::DRAM) gddr_to_dram.fetch_add(1, std::memory_order_relaxed);
        if (from == MemoryTier::DRAM && to == MemoryTier::HBM)  dram_to_hbm.fetch_add(1, std::memory_order_relaxed);
        if (from == MemoryTier::DRAM && to == MemoryTier::GDDR) dram_to_gddr.fetch_add(1, std::memory_order_relaxed);
    }
};


class MigrationScheduler {
public:
    MigrationScheduler(TemporalBridge& bridge,
                       std::chrono::milliseconds interval_ms = std::chrono::milliseconds(500),
                       size_t bandwidth_budget_bytes = 512ULL << 20)
        : bridge_(bridge)
        , interval_(interval_ms)
        , running_(false)
        , budget_(bandwidth_budget_bytes)
    {
        // 220563b: dump the DtypeRegistry at startup so any registration gap
        // (e.g. future dtype added without updating DtypeRegistry) is caught.
        // This mirrors the 220563b test that parametrizes ALL 8 dtype names.
        DtypeRegistry::dump();
    }

    ~MigrationScheduler() {
        stop();
    }

    void start() {
        if (running_.load()) return;
        running_.store(true, std::memory_order_release);
        worker_ = std::thread([this]() { run_loop(); });
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    size_t sweep_once() {
        budget_.reset();  // 220563b: reset dtype-aware budget each sweep
        size_t n = bridge_.migration_sweep();
        stats_.sweep_count.fetch_add(1, std::memory_order_relaxed);
        stats_.total_migrations.fetch_add(n, std::memory_order_relaxed);
        // 断点调试: print budget utilisation after each sweep
        fprintf(stderr,
            "[MigrationScheduler] sweep_once: migrated=%zu budget_utilisation=%.1f%%\n",
            n, budget_.utilisation() * 100.0);
        // 3f11d45 migration: Guard zero-migration sweep before computing rates.
        //
        // This is the scheduler-layer application of the 3f11d45 design pattern:
        //   Python: uxn = (ux.max() + 1) if ux.numel() > 0 else torch.tensor(0)
        //   C++:    rate = (n > 0) ? total/n : 0.0   ← guard empty sweep
        //
        // When n==0, no edges of the relevant hetero-edge type were present in
        // the batch — analogous to a batch with no positive edges of a given type.
        // Callers computing "average migrations per sweep" or "migration rate" MUST
        // apply this guard before dividing by n, just as 3f11d45 applied the
        // numel() guard before calling .max() on the sampled node tensor.
        //
        // The safe_avg_migrations_per_sweep() helper below implements this pattern.
        // 断点调试: n==0 case is printed so the "no migrations this sweep" event
        // is explicitly visible rather than silently producing a divide-by-zero.
        if (n == 0) {
            fprintf(stderr,
                "[DEBUG 3f11d45 MigrationScheduler::sweep_once] n=0 — "
                "no migrations this sweep (mirrors numel()==0 case in 3f11d45).\n"
                "  Downstream aggregations that compute max/avg MUST guard n==0.\n");
        }
        return n;
    }

    // safe_avg_migrations_per_sweep: guard empty sweep count before averaging.
    //
    // 3f11d45 pattern: return 0.0 if no sweeps have occurred (same sentinel as
    // torch.tensor(0, device=ux.device) for the empty-numel case).
    //
    // 断点调试: prints sweep_count so "not yet started" vs "zero migrations" are
    // distinguishable in logs.
    double safe_avg_migrations_per_sweep() const {
        uint64_t sc = stats_.sweep_count.load(std::memory_order_relaxed);
        if (sc == 0) {
            fprintf(stderr,
                "[DEBUG 3f11d45 safe_avg_migrations_per_sweep] sweep_count=0"
                " — returning 0.0 (mirrors numel()==0 guard from 3f11d45)\n");
            return 0.0;
        }
        return static_cast<double>(
            stats_.total_migrations.load(std::memory_order_relaxed)) / sc;
    }

    const MigrationStats& stats() const { return stats_; }

    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

    // 220563b: expose budget for testing — verify bf16 uses 2 bytes/elem
    const MigrationBandwidthBudget& budget() const { return budget_; }

private:
    void run_loop() {
        while (running_.load(std::memory_order_acquire)) {
            sweep_once();
            std::this_thread::sleep_for(interval_);
        }
    }

    TemporalBridge&             bridge_;
    std::chrono::milliseconds   interval_;
    std::atomic<bool>           running_;
    std::thread                 worker_;
    MigrationStats              stats_;
    MigrationBandwidthBudget    budget_;  // 220563b: dtype-aware budget
};

}  // namespace philemon

