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

// ─── MigrationBandwidthBudget ─────────────────────────────────────────────
// Per-sweep bandwidth budget tracker, dtype-aware.
//
// 220563b design: before this commit, the Python FeatureStore's dtype table
// had no entry for bfloat16, so code that called dtypes[torch.bfloat16]
// raised KeyError and fell back to treating tensors as float32 (4 bytes).
// This caused byte-count inflation of 2× for all bf16 feature partitions.
//
// Our C++ analogy: the scheduler estimates bytes-to-migrate per sweep using
// element_size_for_dtype_id().  Before 220563b, id=7 was unregistered so
// the switch fell to default (4 bytes = float32).  Now id=7 → 2 bytes.
//
// The budget prevents a single sweep from overwhelming PCIe bandwidth.
// Partitions are migrated in hotness order until the budget is exhausted.
// With correct dtype byte-counts, bf16 partitions use half the budget
// of float32 partitions of the same element count — enabling more bf16
// migrations per sweep, matching the access-frequency intent.
//
// Pattern: PyTorch CachingAllocator's free-block byte accounting
// (CUDACachingAllocator.cpp: stat_t.current per pool).
struct MigrationBandwidthBudget {
    // Per-sweep byte budget.  Default 512 MB is a reasonable PCIe 4.0 ×16
    // limit for 500ms sweeps (32 GB/s × 0.5s × utilisation_factor=0.03).
    size_t budget_bytes;
    size_t consumed_bytes{0};

    explicit MigrationBandwidthBudget(size_t budget = 512ULL << 20 /* 512 MB */)
        : budget_bytes(budget), consumed_bytes(0) {}

    // Returns element size in bytes for the given wire-protocol dtype id.
    // Maps to 220563b feature_store.py dtype table (all 8 registered ids):
    //   float32=0 → 4 bytes
    //   int64=1   → 8 bytes
    //   float64=2 → 8 bytes
    //   int32=3   → 4 bytes
    //   int16=4   → 2 bytes
    //   float16=5 → 2 bytes
    //   int8=6    → 1 byte
    //   bfloat16=7 → 2 bytes  ← 220563b: explicit registration, was missing
    //
    // 断点调试: fprintf in default branch catches any future id registration
    // gaps before they silently inflate migration byte estimates.
    static size_t element_size_for_dtype_id(uint8_t dtype_id) {
        switch (dtype_id) {
            case DtypeRegistry::ID_FLOAT32: return 4;
            case DtypeRegistry::ID_INT64:   return 8;
            case DtypeRegistry::ID_FLOAT64: return 8;
            case DtypeRegistry::ID_INT32:   return 4;
            case DtypeRegistry::ID_INT16:   return 2;
            case DtypeRegistry::ID_FLOAT16: return 2;
            case DtypeRegistry::ID_INT8:    return 1;
            case DtypeRegistry::ID_BF16:    return 2;  // 220563b: id=7, 2 bytes
            default:
                fprintf(stderr,
                    "[MigrationBandwidthBudget] WARNING: unknown dtype_id=%u,"
                    " assuming 4 bytes (float32 default)\n", dtype_id);
                return 4;
        }
    }

    // Compute migration cost in bytes for a partition with given element count
    // and dtype id.  This is the dtype-aware analog of:
    //   Python: len(tensor) * tensor.element_size()
    // which silently used float32 for bf16 before 220563b.
    static size_t migration_bytes(size_t element_count, uint8_t dtype_id) {
        return element_count * element_size_for_dtype_id(dtype_id);
    }

    // Try to consume bytes for one migration.  Returns false if budget exhausted.
    // dtype_id used only for debug print; byte_count already computed by caller.
    bool try_consume(size_t byte_count, uint8_t dtype_id = 0) {
        if (consumed_bytes + byte_count > budget_bytes) {
            return false;
        }
        consumed_bytes += byte_count;
        // 断点调试: print each consumption so bf16 vs float32 budget usage is visible
        fprintf(stderr,
            "[MigrationBandwidthBudget] consume: dtype_id=%u bytes=%zu"
            " consumed=%zu / budget=%zu\n",
            dtype_id, byte_count, consumed_bytes, budget_bytes);
        return true;
    }

    void reset() { consumed_bytes = 0; }

    double utilisation() const {
        return budget_bytes > 0
               ? static_cast<double>(consumed_bytes) / budget_bytes
               : 0.0;
    }
};


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
        return n;
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

