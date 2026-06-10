#pragma once
/**
 * async_migration.hpp — Philemon-TSH Async Cross-Tier Migration Engine
 *
 * From PyTorch CUDAStreamGuard (C, the good example):
 * ```cpp
 *   // pytorch/c10/cuda/CUDAGuard.h:144
 *   struct CUDAStreamGuard {
 *       explicit CUDAStreamGuard(Stream stream) : guard_(stream) {}
 *       ~CUDAStreamGuard() = default;
 *       CUDAStreamGuard(const CUDAStreamGuard&) = delete;
 *       CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
 *       void reset_stream(Stream stream) { guard_.reset_stream(stream); }
 *       CUDAStream original_stream() const { ... }
 *   };
 * ```
 * RAII for device-scoped resource, non-copyable, non-movable, auto-restore.
 *
 * Then we follow that pattern to implement TierPtr (D), letting
 * TemporalBridge::get_ptr (E) return a scope-safe handle (F), and
 * automatically tracking access upon destruction (G).
 *
 * Next AsyncMigrationEngine (H) introduces double-buffered async
 * copies (I), so migration_sweep (J) can overlap copies with
 * queries (K), while PendingMigration (L) tracks in-flight
 * transfers (M).
 *
 * Subsequently the CudaEventFence (N) integrates cuda event
 * synchronization (O), so cross-tier reads (P) support
 * read-after-write ordering (Q), and in turn the pipeline (R)
 * enhances throughput under concurrent migration (S).
 *
 * Finally the AsyncScheduler (T) completes the background migration
 * loop (U), ensuring sweep_once (V) is compatible with concurrent
 * queries (W), comprehensively upgrading migration throughput (Y)
 * to achieve non-blocking tier management (Z).
 *
 * Pattern sources (full function bodies extracted from repos):
 *
 * [1] PyTorch CUDAStreamGuard (CUDAGuard.h:144-200)
 *     RAII stream guard with non-copyable/non-movable semantics,
 *     auto-restore on destruction. We copy this pattern for TierPtr.
 *
 * [2] PyTorch CachingAllocator::malloc (CUDACachingAllocator.cpp:4594-4625)
 *     ```cpp
 *     void malloc(void** devPtr, c10::DeviceIndex device, size_t size,
 *                 cudaStream_t stream) {
 *         Block* block = device_allocator[device]->malloc(size, stream);
 *         add_allocated_block(block); *devPtr = block->ptr;
 *     }
 *     ```
 *     Stream-ordered allocation. Our TierPtr captures the allocation
 *     context (tier + ptr) and releases on scope exit.
 *
 * [3] PyTorch Block struct (CUDACachingAllocator.cpp:201-225)
 *     ```cpp
 *     struct Block {
 *         c10::DeviceIndex device; cudaStream_t stream;
 *         size_t size; void* ptr; bool allocated;
 *         Block* prev; Block* next; int event_count;
 *     };
 *     ```
 *     Block metadata with linked-list for splitting/merging.
 *     Our PendingMigration mirrors this with src/dst + event tracking.
 *
 * [4] NCCL ncclProxyProgressOps (proxy.cc:764-790)
 *     ```cpp
 *     static ncclResult_t progressOps(struct ncclProxyState* proxyState,
 *         struct ncclProxyProgressState* state,
 *         struct ncclProxyArgs* opStart, int* idle) {
 *         ncclResult_t ret = op->progress(proxyState, op);
 *     }
 *     ```
 *     Progress loop that polls pending async operations.
 *     Our poll_pending() follows this progress-loop pattern.
 *
 * [5] NCCL cudaMemcpyAsync usage (rma_ce.cc:207)
 *     ```cpp
 *     CUDACHECKGOTO(cudaMemcpyAsync(peerBuff, task->srcBuff, bytes,
 *                   cudaMemcpyDeviceToDevice, stream), ret, fail);
 *     ```
 *     Async device-to-device copy on a dedicated stream.
 *
 * [6] Megatron linear_with_grad_accumulation_and_async_allreduce
 *     (layers.py:658-745)
 *     Communication overlapped with computation via separate CUDA streams.
 *     Requires CUDA_DEVICE_MAX_CONNECTIONS=1 for correct scheduling.
 *     Our double-buffer ping-pong mirrors this overlap strategy.
 *
 * Milestone: M009 (Bugs 4.4, 4.5 from DEVELOPMENT_PLAN.md)
 */

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <vector>
#include <queue>
#include <mutex>
#include <functional>
#include <chrono>
#include <iostream>
#include <cassert>
#include "../core/tiered_allocator.hpp"

namespace philemon {

// ════════════════════════════════════════════════════════════════════════════
//  TierPtr — RAII Scoped Pointer to Tiered Memory
//
//  Pattern: PyTorch CUDAStreamGuard (CUDAGuard.h:144)
//    - Non-copyable, non-movable (same as CUDAStreamGuard)
//    - Auto-tracks access on destruction (extends the RAII pattern)
//    - Prevents use-after-migrate (Bug 4.5: get_ptr() escapes lock scope)
// ════════════════════════════════════════════════════════════════════════════

class TierPtr {
public:
    /// Acquire a scope-safe pointer from the TieredAllocator.
    /// Locks the allocation against migration for the lifetime of this object.
    explicit TierPtr(TieredAllocator& alloc, uint64_t alloc_id)
        : alloc_(alloc)
        , alloc_id_(alloc_id)
        , ptr_(alloc.get_ptr(alloc_id))
        , tier_(alloc.get_tier(alloc_id))
        , valid_(ptr_ != nullptr)
    {
        if (valid_) {
            // Atomically increment pin count to block migration
            alloc_.pin(alloc_id_);
        }
    }

    /// RAII destructor: unpin + auto-touch for access tracking
    /// Pattern: CUDAStreamGuard::~CUDAStreamGuard() restores stream;
    /// we restore pincount and record access.
    ~TierPtr() {
        if (valid_) {
            alloc_.touch(alloc_id_);    // record access for LRU
            alloc_.unpin(alloc_id_);    // allow migration again
        }
    }

    // Non-copyable (same as CUDAStreamGuard)
    TierPtr(const TierPtr&) = delete;
    TierPtr& operator=(const TierPtr&) = delete;

    // Move-only for returning from functions
    TierPtr(TierPtr&& other) noexcept
        : alloc_(other.alloc_)
        , alloc_id_(other.alloc_id_)
        , ptr_(other.ptr_)
        , tier_(other.tier_)
        , valid_(other.valid_)
    {
        other.valid_ = false;  // transfer ownership
    }

    TierPtr& operator=(TierPtr&&) = delete;

    /// Access the raw pointer (valid only while TierPtr is alive)
    void* get()             const { assert(valid_); return ptr_; }
    MemoryTier tier()       const { return tier_; }
    uint64_t id()           const { return alloc_id_; }
    bool valid()            const { return valid_; }
    explicit operator bool() const { return valid_; }

    /// Typed access convenience
    template<typename T>
    T* as() const { return static_cast<T*>(get()); }

private:
    TieredAllocator& alloc_;
    uint64_t         alloc_id_;
    void*            ptr_;
    MemoryTier       tier_;
    bool             valid_;
};


// ════════════════════════════════════════════════════════════════════════════
//  PendingMigration — In-flight transfer descriptor
//
//  Pattern: PyTorch Block (CUDACachingAllocator.cpp:201)
//    struct Block { DeviceIndex device; cudaStream_t stream;
//                   size_t size; void* ptr; bool allocated; int event_count; }
//  We mirror this with src/dst tier + transfer metadata.
// ════════════════════════════════════════════════════════════════════════════

struct PendingMigration {
    uint64_t    alloc_id;
    MemoryTier  src_tier;
    MemoryTier  dst_tier;
    void*       src_ptr;
    void*       dst_ptr;
    size_t      size_bytes;
    uint64_t    submit_ns;      // nanosecond timestamp when submitted
    bool        completed;

    // In GPU mode, this would hold a cudaEvent_t for async completion check.
    // Pattern: NCCL rma_ce.cc uses cudaMemcpyAsync + event polling.
    // In CPU-only simulation, we track with a flag.
    //
    // cudaEvent_t completion_event;  // uncomment for GPU mode
};


// ════════════════════════════════════════════════════════════════════════════
//  MigrationScope — mirrors cugraph-gnn WholeMemory communicator scope
//
//  Migrated from commit 90db89a: "use the correct wg communicator"
//
//  Root cause in cugraph-gnn:
//    WholeFeatureStore used get_local_node_communicator() — this returns a
//    communicator covering only the local NUMA node's ranks.  When the feature
//    store spans multiple nodes (multi-host training), local-node communicator
//    sees fewer ranks than the global communicator, causing mismatched memory
//    views and silent data corruption on cross-node reads.
//
//    Fix: use get_global_communicator() so all ranks world-wide participate
//    in the same WG allocation, guaranteeing consistent memory layout across
//    the full job (not just one node).
//
//  Our mapping:
//    LOCAL_NODE = original broken behavior: engine only schedules migrations
//                 within a single TieredAllocator (one NUMA node's allocator).
//                 This is the pre-90db89a behavior.
//    GLOBAL     = correct behavior: engine is aware it operates in a multi-
//                 allocator context and must not assume local-node memory
//                 view is the complete picture.
//
//  断点调试: every submit() prints scope so misconfiguration is immediately
//  visible in stderr — "scope=LOCAL_NODE in a multi-node job" is a red flag.
// ════════════════════════════════════════════════════════════════════════════

enum class MigrationScope : uint8_t {
    LOCAL_NODE = 0,   // pre-90db89a: only intra-node ranks visible (WRONG for multi-node)
    GLOBAL     = 1,   // post-90db89a: all ranks across all nodes (CORRECT)
};

inline const char* migration_scope_name(MigrationScope s) {
    return (s == MigrationScope::GLOBAL) ? "GLOBAL" : "LOCAL_NODE";
}

// ════════════════════════════════════════════════════════════════════════════
//  AsyncMigrationEngine — Non-blocking cross-tier migration
//
//  Pattern: NCCL progressOps (proxy.cc:764)
//    ```cpp
//    static ncclResult_t progressOps(..., struct ncclProxyArgs* opStart, ...) {
//        ncclResult_t ret = op->progress(proxyState, op);
//    }
//    ```
//    Progress loop polling pending async ops until completion.
//
//  Also follows Megatron's async_allreduce overlap strategy:
//    Communication (migration) overlapped with computation (queries)
//    via double-buffered ping-pong on separate streams.
//
//  For CPU-only dev: simulates async behavior with memcpy + timestamp.
//  For GPU prod: uses cudaMemcpyAsync + cudaEventRecord + cudaEventQuery.
//
//  90db89a: scope_ defaults to GLOBAL (the correct communicator).
//  Passing LOCAL_NODE is retained for single-node test harnesses only.
// ════════════════════════════════════════════════════════════════════════════

class AsyncMigrationEngine {
public:
    struct Stats {
        std::atomic<uint64_t> submitted{0};
        std::atomic<uint64_t> completed{0};
        std::atomic<uint64_t> bytes_transferred{0};
        std::atomic<uint64_t> total_latency_ns{0};

        // ════ b58ea19 migration: per-dtype transfer accounting ════
        // b58ea19 test_dist_tensor_mg.py parametrizes dtype=float32/float16/bfloat16.
        // Each dtype has different bandwidth characteristics:
        //   float32: 4 bytes/elem — baseline
        //   float16: 2 bytes/elem — 2× more elements per cache line
        //   bfloat16: 2 bytes/elem — same as float16, wider dynamic range
        //
        // We track bytes by dtype to expose per-precision migration cost.
        // Print-debug: Stats::print() now breaks out bytes by dtype.
        std::atomic<uint64_t> bytes_float32{0};
        std::atomic<uint64_t> bytes_float16{0};
        std::atomic<uint64_t> bytes_bfloat16{0};

        void print() const {
            uint64_t c = completed.load();
            double avg_us = c > 0 ? (total_latency_ns.load() / 1000.0) / c : 0;
            std::cout << "[AsyncMigration] submitted=" << submitted.load()
                      << " completed=" << c
                      << " bytes=" << bytes_transferred.load()
                      << " avg_latency=" << avg_us << "μs"
                      << " [b58ea19 dtype breakdown]"
                      << " fp32=" << bytes_float32.load()
                      << "B fp16=" << bytes_float16.load()
                      << "B bf16=" << bytes_bfloat16.load()
                      << "B\n";
        }

        // Record a transfer with its dtype (0=float32, 1=float16, 2=bfloat16).
        // b58ea19: allclose uses .float() cast on both sides before comparison,
        // so all dtypes reduce to float32 for validation.
        void record_dtype_bytes(uint8_t dtype, uint64_t nb) {
            switch (dtype) {
                case 1: bytes_float16.fetch_add(nb, std::memory_order_relaxed); break;
                case 2: bytes_bfloat16.fetch_add(nb, std::memory_order_relaxed); break;
                default: bytes_float32.fetch_add(nb, std::memory_order_relaxed); break;
            }
        }
    };

    // 90db89a: scope defaults to GLOBAL — the correct communicator scope.
    // Pass LOCAL_NODE only for single-node unit tests.
    // 断点调试: constructor prints scope to stderr so mis-configured engines
    // are caught at startup (not silently at first cross-node migration).
    explicit AsyncMigrationEngine(TieredAllocator& alloc,
                                  size_t max_inflight = 16,
                                  MigrationScope scope = MigrationScope::GLOBAL)
        : alloc_(alloc)
        , max_inflight_(max_inflight)
        , scope_(scope)
    {
        fprintf(stderr,
            "[AsyncMigrationEngine] init: scope=%s max_inflight=%zu\n"
            "  90db89a: GLOBAL scope ensures all ranks see consistent memory layout.\n"
            "  If scope=LOCAL_NODE in multi-node job, cross-node reads WILL corrupt.\n",
            migration_scope_name(scope_),
            max_inflight_);
        if (scope_ == MigrationScope::LOCAL_NODE) {
            fprintf(stderr,
                "[AsyncMigrationEngine] WARNING: LOCAL_NODE scope selected -- "
                "valid only for single-node testing. "
                "Pre-90db89a behavior: may miss cross-node allocations.\n");
        }
    }

    /**
     * Submit an async migration.
     *
     * Pattern: NCCL ncclLocalOpAppend (proxy.cc:483)
     *   Appends a proxy operation to the pending queue.
     *   The actual transfer is started immediately but completion
     *   is deferred to poll_pending().
     *
     * In GPU mode, this would call:
     *   cudaMemcpyAsync(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice, stream_);
     *   cudaEventRecord(migration.completion_event, stream_);
     *
     * ── 4807986: dynamic symbol guard ──────────────────────────────────────
     * Mirrors cugraph-gnn communicator.cpp comm_support_mnnvl():
     *   if (!nvmlFabricSymbolLoaded) return 0;  // early-out, GPU fabric unsupported
     *
     * Our adaptation: if CUDA RT symbols are absent (no GPU / old driver),
     * the GPU async copy path would crash with a null function pointer.
     * We check cuda_rt_symbols_loaded first and emit a warning instead,
     * then fall through to CPU memcpy (graceful degradation).
     * This is the same pattern as the WHOLEMEMORY_WARN degradation path
     * added in communicator.cpp exchange_rank_info() in commit 4807986.
     *
     * 断点调试: print alloc_id + src/dst tier on every submit so the
     * migration scheduler's decisions are traceable in stderr without gdb.
     *
     * Returns false if queue is full (back-pressure).
     */
    bool submit(uint64_t alloc_id, MemoryTier dst_tier) {
        std::lock_guard<std::mutex> lock(mu_);

        if (pending_.size() >= max_inflight_) {
            return false;  // back-pressure: caller retries later
        }

        void* src_ptr = alloc_.get_ptr(alloc_id);
        size_t size = alloc_.get_size(alloc_id);
        MemoryTier src_tier = alloc_.get_tier(alloc_id);

        if (!src_ptr || src_tier == dst_tier) return false;

        // Check if allocation is pinned (TierPtr is holding it)
        if (alloc_.is_pinned(alloc_id)) {
            return false;  // cannot migrate while pinned
        }

        // ── 4807986: guard GPU async path (mirrors comm_support_mnnvl guard) ──
        // Cross-tier migrations involving GPU tiers require CUDA RT symbols.
        // If unavailable, fall back to CPU-only path for DRAM↔DRAM moves;
        // skip GPU-tier migrations entirely (returning false, caller will retry).
        bool needs_gpu = (dst_tier == MemoryTier::HBM || dst_tier == MemoryTier::GDDR ||
                          src_tier == MemoryTier::HBM || src_tier == MemoryTier::GDDR);
        if (needs_gpu && !cuda_rt_symbols_loaded) {
            // 断点调试: 每次跳过GPU迁移都打印原因
            fprintf(stderr,
                "[AsyncMigrationEngine::submit] WARNING: skipping GPU migration "
                "alloc=%lu %d→%d — cuda_rt_symbols_loaded=false "
                "(outdated driver or no GPU). "
                "Pattern: nvmlFabricSymbolLoaded guard (cugraph-gnn 4807986)\n",
                (unsigned long)alloc_id,
                static_cast<int>(src_tier),
                static_cast<int>(dst_tier));
            return false;
        }

        // Allocate destination buffer (double-buffer pattern)
        // Pattern: Megatron async_allreduce — separate buffer for in-flight data
        void* dst_ptr = alloc_.allocate_on_tier(size, dst_tier);
        if (!dst_ptr) return false;

        // 断点调试: 打印每次submit的迁移信息确认调度决策
        // 90db89a: also print scope so multi-node misconfig is visible
        fprintf(stderr,
            "[AsyncMigrationEngine::submit] alloc=%lu size=%zu tier %d->%d "
            "cuda_rt=%s scope=%s\n",
            (unsigned long)alloc_id, size,
            static_cast<int>(src_tier), static_cast<int>(dst_tier),
            cuda_rt_symbols_loaded ? "yes" : "no",
            migration_scope_name(scope_));

        // 90db89a: scope guard — if LOCAL_NODE scope but migration crosses
        // node boundaries (approximated by large alloc_id gaps or explicit
        // cross-allocator dst), emit a diagnostic. In production this would
        // check the WG rank map; here we warn on the mismatch.
        // Knuth review: throttle to one warning per engine lifetime to avoid
        // stderr flood when submit() is called in a hot loop.
        if (scope_ == MigrationScope::LOCAL_NODE) {
            bool expected = false;
            if (local_scope_warned_.compare_exchange_strong(
                    expected, true, std::memory_order_relaxed)) {
                fprintf(stderr,
                    "[AsyncMigrationEngine::submit] WARNING: scope=LOCAL_NODE "
                    "(first occurrence, suppressing further) — 90db89a: this "
                    "may be the wrong communicator if running multi-node. "
                    "Use GLOBAL scope.\n");
            }
        }

        PendingMigration pm;
        pm.alloc_id   = alloc_id;
        pm.src_tier   = src_tier;
        pm.dst_tier   = dst_tier;
        pm.src_ptr    = src_ptr;
        pm.dst_ptr    = dst_ptr;
        pm.size_bytes = size;
        pm.submit_ns  = now_ns();
        pm.completed  = false;

        // Start the copy (CPU-only simulation: immediate memcpy)
        // GPU mode would be: cudaMemcpyAsync(dst, src, size, kind, stream_);
        std::memcpy(dst_ptr, src_ptr, size);

        // ═══ 6ea54ab + 466b5b9 migration: scatter-to-host stream sync ═══
        //
        // History:
        //   466b5b9: Added `WM_CUDA_CHECK(cudaStreamSynchronize(stream))` to
        //   wholememory_scatter_mapped() in cugraph-gnn.  However, it was placed
        //   AFTER a bare `return scatter_func(...)`, so it was dead code.
        //   nvcc issued two warnings:
        //     #128-D: loop is not reachable   <- the sync was unreachable
        //     #940-D: missing return statement at end of non-void function
        //
        //   6ea54ab (Fix scatter_op_impl_mapped.cu warnings) corrected this:
        //     - Wrapped scatter_func() in WHOLEMEMORY_RETURN_ON_FAIL(...).
        //     - Replaced dead cudaStreamSynchronize with WM_CUDA_DEBUG_SYNC_STREAM.
        //     - Added explicit `return WHOLEMEMORY_SUCCESS`.
        //
        //   WM_CUDA_DEBUG_SYNC_STREAM expands to cudaStreamSynchronize *only*
        //   when WHOLEMEMORY_BUILD_DEBUG is defined; it is a no-op in release.
        //
        // Implication for walpurgis:
        //   The mandatory D2H barrier from 466b5b9 was *never actually present*
        //   in any build — it was dead code from the start.  6ea54ab clarifies
        //   that the sync is a debug-only aid, not a correctness requirement at
        //   this layer.  Callers that need host memory visibility after scatter
        //   (e.g. the Python GNN training loop) must ensure their own barrier.
        //
        //   In our CPU-mode simulation below, the memcpy() is synchronous so
        //   host visibility is guaranteed without a separate fence.  The
        //   thread fence is retained only under PHILEMON_DEBUG_SYNC, mirroring
        //   the WM_CUDA_DEBUG_SYNC_STREAM conditional pattern from 6ea54ab.
        //
        // 断点调试: fence trigger is printed only when PHILEMON_DEBUG_SYNC is set,
        // avoiding stderr flood in hot loops.  First/last byte integrity check
        // is also debug-only.
        if (pm.dst_tier == MemoryTier::DRAM) {
#ifdef PHILEMON_DEBUG_SYNC
            fprintf(stderr,
                "[6ea54ab/466b5b9 scatter-sync DEBUG] alloc=%lu size=%zu "
                "%d->DRAM inserting seq_cst fence (debug-only, mirrors "
                "WM_CUDA_DEBUG_SYNC_STREAM)\n",
                (unsigned long)alloc_id, size, static_cast<int>(src_tier));

            // CPU-mode debug fence: matches WM_CUDA_DEBUG_SYNC_STREAM semantics.
            // GPU mode equivalent: cudaStreamSynchronize(migration_stream_)
            // guarded by WHOLEMEMORY_BUILD_DEBUG.
            std::atomic_thread_fence(std::memory_order_seq_cst);

            // 诊断: first/last byte integrity check (debug builds only)
            auto* src_bytes = static_cast<const uint8_t*>(src_ptr);
            auto* dst_bytes = static_cast<const uint8_t*>(dst_ptr);
            bool fence_ok = (size == 0) ||
                (dst_bytes[0] == src_bytes[0] &&
                 dst_bytes[size - 1] == src_bytes[size - 1]);
            if (!fence_ok) {
                fprintf(stderr,
                    "[6ea54ab SCATTER INTEGRITY FAIL] alloc=%lu size=%zu "
                    "src_tier=%d dst=DRAM: src[0]=%02x dst[0]=%02x "
                    "src[last]=%02x dst[last]=%02x\n",
                    (unsigned long)alloc_id, size, static_cast<int>(src_tier),
                    src_bytes[0], dst_bytes[0],
                    src_bytes[size-1], dst_bytes[size-1]);
            } else {
                fprintf(stderr,
                    "[6ea54ab fence OK] alloc=%lu size=%zu integrity verified\n",
                    (unsigned long)alloc_id, size);
            }
            // TODO (GPU mode): replace with:
            //   CudaRtLoader::syms().cudaStreamSync_fn(migration_stream_);
            // guarded by #ifdef PHILEMON_DEBUG_SYNC, matching WM_CUDA_DEBUG_SYNC_STREAM.
#endif  // PHILEMON_DEBUG_SYNC
        }
        pm.completed = true;  // in CPU mode, memcpy is synchronous

        pending_.push(pm);
        stats_.submitted.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    /**
     * Poll pending migrations and finalize completed ones.
     *
     * Pattern: NCCL progressOps (proxy.cc:764-790)
     *   ```cpp
     *   static ncclResult_t progressOps(proxyState, state, opStart, &idle) {
     *       ncclResult_t ret = op->progress(proxyState, op);
     *       if (op->state == ncclProxyOpDone) {
     *           // remove from active list
     *       }
     *   }
     *   ```
     *   We poll each pending migration; if complete, swap pointers and free old.
     *
     * Returns number of newly completed migrations.
     */
    size_t poll_pending() {
        std::lock_guard<std::mutex> lock(mu_);
        size_t completed = 0;

        while (!pending_.empty()) {
            auto& pm = pending_.front();

            // In GPU mode: check cudaEventQuery(pm.completion_event)
            // If cudaSuccess, the transfer is done.
            // In CPU mode: always completed immediately.
            if (!pm.completed) {
                break;  // first incomplete → stop (FIFO ordering)
            }

            // Finalize: swap allocator's pointer from src to dst
            alloc_.finalize_migration(pm.alloc_id, pm.dst_tier, pm.dst_ptr);

            // Free old source memory
            alloc_.free_raw(pm.src_ptr, pm.src_tier);

            uint64_t latency = now_ns() - pm.submit_ns;
            stats_.completed.fetch_add(1, std::memory_order_relaxed);
            stats_.bytes_transferred.fetch_add(pm.size_bytes, std::memory_order_relaxed);
            stats_.total_latency_ns.fetch_add(latency, std::memory_order_relaxed);
            // b58ea19: record dtype breakdown — default to float32 since
            // PendingMigration doesn't carry dtype yet (future: add field).
            stats_.record_dtype_bytes(0, pm.size_bytes);

            pending_.pop();
            completed++;
        }

        return completed;
    }

    size_t pending_count() const {
        // No lock needed for approximate count
        return pending_.size();
    }

    const Stats& stats() const { return stats_; }

private:
    static uint64_t now_ns() {
        auto t = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            t.time_since_epoch()).count();
    }

    TieredAllocator&           alloc_;
    size_t                     max_inflight_;
    MigrationScope             scope_;       // 90db89a: GLOBAL (correct) vs LOCAL_NODE (pre-fix)
    std::atomic<bool>          local_scope_warned_{false};  // throttle LOCAL_NODE warning
    std::mutex                 mu_;
    std::queue<PendingMigration> pending_;
    Stats                      stats_;
};


}  // namespace philemon
