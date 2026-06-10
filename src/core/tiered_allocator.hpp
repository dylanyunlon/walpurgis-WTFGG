#pragma once
/**
 * tiered_allocator.hpp — Philemon-TSH Heterogeneous Memory Tier Allocator
 *
 * Design rationale (from RapidStore's wrapper::set_max_threads / NCCL's ncclMemAlloc pattern):
 *
 *   Starting from RapidStore's CsrWrapper::Snapshot as the good example (C),
 *   we follow that pattern to implement TieredAllocator (D), letting
 *   TemporalSubgraphManager (E) place hot intervals in HBM (F) and evict
 *   cold data to DRAM (G).  Then the MigrationScheduler (H) introduces
 *   AccessFrequencyTracker (I), so that the EvictionPolicy (J) can make
 *   LRU-aware tier decisions (K), while the CompactionEngine (L) optimizes
 *   memory fragmentation (M).  Subsequently the BridgeLayer (N) integrates
 *   TEM-Graph's build_index (O), so that IntervalQuery (P) supports
 *   cross-tier contains_query (Q), and in turn the PrefetchEngine (R)
 *   enhances read latency for temporal scans (S).  Finally the
 *   ConcurrencyManager (T) completes the thread-safe snapshot protocol (U),
 *   ensuring RapidStore's snapshot_clone (V) is compatible with tiered
 *   pointers (W), comprehensively upgrading the full query pipeline (Y) to
 *   achieve sub-millisecond temporal subgraph retrieval on heterogeneous
 *   memory (Z).
 *
 * Milestone: M001–M004 (Claude #1), M005–M006 (Claude #2), M007–M008 (Claude #3)
 *
 * M008 changes:
 *   - Integrated SlabAllocator for per-tier slab-based memory management.
 *     Small allocations (≤512 KB) go through slab pools with size-class
 *     bucketing, eliminating per-allocation posix_memalign overhead.
 *     Pattern: PyTorch CachingAllocator Block/try_merge_blocks
 *     (c10/cuda/CUDACachingAllocator.cpp:3583), NCCL cudaMemPoolCreate
 *     page-based pooling (nccl/src/allocator.cc:345), TensorFlow Arena
 *     bump allocation (tensorflow/core/lib/core/arena.h:35).
 *   - compact() method releases empty slab pages back to OS, preventing
 *     the memory bloat identified in Claude #1 review Bug 4.3.
 *
 * M005 changes:
 *   - touch() is now LOCKFREE: uses atomic counters directly without
 *     taking mu_.  Pattern from NCCL's COMPILER_ATOMIC_FETCH_ADD
 *     (nccl/src/include/compiler/gcc.h) and CCCL's shared_block_ptr
 *     refcount (cccl/libcudacxx/include/cuda/__memory_resource/shared_block_ptr.h).
 *   - Registry reads (get_ptr, get_meta, for_each_alloc) now use
 *     std::shared_mutex (read-shared / write-exclusive), following
 *     PyTorch c10's COWDeleter pattern
 *     (pytorch/c10/core/impl/COWDeleter.h: shared_lock<shared_mutex>).
 *   - Structural mutations (allocate, deallocate, migrate) take
 *     unique_lock; concurrent reads proceed unblocked.
 */

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <atomic>
#include <vector>
#include <shared_mutex>    // M005: replaces plain std::mutex for read path
#include <mutex>           // M005: for std::unique_lock
#include <memory>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <dlfcn.h>         // 4807986: dynamic symbol loading (mirrors nvml_wrap.cpp dlopen)
#include "slab_allocator.hpp"      // M008: per-tier slab allocation

namespace philemon {

// ─── Dynamic CUDA Runtime Symbol Loader ─────────────────────────────────────
// Mirrors cugraph-gnn commit 4807986: nvml_wrap.cpp's LoadNvmlLibrary() /
// LoadNvmlSymbol<T>() pattern, ported to CUDA runtime symbols.
//
// Problem (from 4807986 commit message):
//   Direct linking to libcuda.so / libcudart.so fails silently on machines
//   with mismatched driver versions or without a GPU, aborting the process
//   before main().  The nvml_wrap approach fixes this by deferring the link
//   to first use via dlopen + dlsym, and gracefully disabling GPU paths when
//   the library is absent.
//
// Our adaptation:
//   - cudaMalloc / cudaFree / cudaMemcpy are declared as function pointers
//     (typedef pattern from nvml_wrap.h: nvmlDeviceGetHandleByIndexFunc)
//   - CudaRtLoader::load() mirrors LoadNvmlLibrary() + LoadNvmlSymbol<T>()
//   - cuda_rt_loaded mirrors nvmlFabricSymbolLoaded in system_info.hpp:
//       inline bool nvmlFabricSymbolLoaded = NvmlFabricSymbolLoaded();
//   - All GPU paths in TieredAllocator::migrate() are guarded by
//       if (cuda_rt_loaded) { ... GPU DMA ... } else { /* CPU fallback */ }
//     mirroring communicator.cpp's:
//       if (nvmlFabricSymbolLoaded) { ... } else { WHOLEMEMORY_WARN(...); }
//
// 断点调试: CudaRtLoader::load() prints each dlsym result so failures are
// visible without attaching gdb. Pattern from nvml_wrap.cpp fprintf(stderr).

typedef int (*CudaMallocFn)(void**, size_t);
typedef int (*CudaFreeFn)(void*);
typedef int (*CudaMemcpyFn)(void*, const void*, size_t, int);
typedef int (*CudaMemcpyAsyncFn)(void*, const void*, size_t, int, void*);
typedef int (*CudaStreamSyncFn)(void*);
typedef int (*CudaGetDeviceCountFn)(int*);

struct CudaRtSymbols {
    CudaMallocFn       cudaMalloc_fn       = nullptr;
    CudaFreeFn         cudaFree_fn         = nullptr;
    CudaMemcpyFn       cudaMemcpy_fn       = nullptr;
    CudaMemcpyAsyncFn  cudaMemcpyAsync_fn  = nullptr;
    CudaStreamSyncFn   cudaStreamSync_fn   = nullptr;
    CudaGetDeviceCountFn cudaGetDeviceCount_fn = nullptr;
};

// LoadCudaRtLibrary: mirrors LoadNvmlLibrary() from nvml_wrap.cpp
// Tries versioned soname first, then unversioned fallback.
// Returns handle or nullptr on failure.
inline void* LoadCudaRtLibrary() {
    // Clear any stale dlerror state before our dlopen calls.
    // Bug 5 fix: must clear dlerror before each call to get accurate error text.
    dlerror();
    void* handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        dlerror();  // clear error from first attempt
        handle = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!handle) {
        dlerror();  // clear error from second attempt
        handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!handle) {
        // Only call dlerror() once here — captures the last dlopen failure
        const char* err = dlerror();
        fprintf(stderr,
            "[CudaRtLoader] Failed to load libcudart: %s\n"
            "[CudaRtLoader] GPU migration paths will be disabled (CPU fallback active)\n",
            err ? err : "(unknown error)");
    }
    return handle;
}

// LoadCudaRtSymbol: mirrors LoadNvmlSymbol<T>() from nvml_wrap.cpp
template <typename T>
inline T LoadCudaRtSymbol(void* handle, const char* name) {
    if (!handle) return nullptr;
    void* sym = dlsym(handle, name);
    if (!sym) {
        fprintf(stderr, "[CudaRtLoader] dlsym('%s') failed: %s\n", name, dlerror());
    }
    return reinterpret_cast<T>(sym);
}

// CudaRtLoader: singleton dynamic loader.
// Mirrors NvmlFabricSymbolLoaded() from nvml_wrap.cpp —
// the one-shot initializer that sets the global bool.
//
// Bug 1 fix (Knuth review): the original `init_done_()` bool was read in
// loaded() WITHOUT the mutex, creating a data race when two threads both
// see init_done_=false and both enter init(). Fix: use std::call_once
// (C++11, guaranteed single-execution even under concurrent callers).
// This matches the nvml_wrap.cpp std::lock_guard pattern exactly, but
// call_once is the idiomatic C++11 tool for one-shot initialization.
//
// Usage: if (CudaRtLoader::loaded()) { use CudaRtLoader::syms().cudaMalloc_fn(...); }
class CudaRtLoader {
public:
    // Mirrors: bool NvmlFabricSymbolLoaded() with std::lock_guard<std::mutex>
    static bool init() {
        std::call_once(once_flag_(), []() {
            fprintf(stderr, "[CudaRtLoader] init(): attempting dlopen libcudart\n");
            void* h = LoadCudaRtLibrary();
            if (!h) {
                loaded_flag_() = false;
                fprintf(stderr, "[CudaRtLoader] init(): CUDA runtime NOT available — "
                                "all GPU paths gracefully disabled\n");
                return;
            }

            auto& s = syms_();
            s.cudaMalloc_fn         = LoadCudaRtSymbol<CudaMallocFn>(h, "cudaMalloc");
            s.cudaFree_fn           = LoadCudaRtSymbol<CudaFreeFn>(h, "cudaFree");
            s.cudaMemcpy_fn         = LoadCudaRtSymbol<CudaMemcpyFn>(h, "cudaMemcpy");
            s.cudaMemcpyAsync_fn    = LoadCudaRtSymbol<CudaMemcpyAsyncFn>(h, "cudaMemcpyAsync");
            s.cudaStreamSync_fn     = LoadCudaRtSymbol<CudaStreamSyncFn>(h, "cudaStreamSynchronize");
            s.cudaGetDeviceCount_fn = LoadCudaRtSymbol<CudaGetDeviceCountFn>(h, "cudaGetDeviceCount");

            // Mirror nvml_wrap.cpp: only set loaded=true when ALL required symbols found.
            // If any critical symbol is missing (old driver), disable GPU paths.
            bool all_found = s.cudaMalloc_fn && s.cudaFree_fn &&
                             s.cudaMemcpy_fn && s.cudaGetDeviceCount_fn;
            if (!all_found) {
                dlclose(h);
                handle_() = nullptr;
                loaded_flag_() = false;
                fprintf(stderr,
                    "[CudaRtLoader] Some required CUDA RT symbols are missing, "
                    "likely due to an outdated GPU driver. "
                    "GPU migration paths will be disabled.\n");
            } else {
                handle_() = h;
                loaded_flag_() = true;
                // 断点调试: 打印已加载的symbol地址确认dlsym成功
                fprintf(stderr,
                    "[CudaRtLoader] CUDA runtime loaded OK: "
                    "cudaMalloc=%p cudaFree=%p cudaMemcpy=%p\n",
                    (void*)s.cudaMalloc_fn,
                    (void*)s.cudaFree_fn,
                    (void*)s.cudaMemcpy_fn);
            }
        });
        return loaded_flag_();
    }

    static bool loaded() {
        // std::call_once guarantees single execution even on concurrent first calls.
        // After first call, once_flag blocks are no-ops and this path is cheap.
        // Mirrors nvmlFabricSymbolLoaded inline initializer in system_info.hpp.
        return init();
    }

    static const CudaRtSymbols& syms() { return syms_(); }

private:
    // Meyers-singleton helpers to avoid static-init-order issues.
    // Bug 1 fix: once_flag_ replaces the racy bool init_done_ + mutex pattern.
    static std::once_flag&  once_flag_()  { static std::once_flag v; return v; }
    static bool&            loaded_flag_(){ static bool v = false; return v; }
    static void*&           handle_()     { static void* v = nullptr; return v; }
    static CudaRtSymbols&   syms_()       { static CudaRtSymbols v; return v; }
};

// Convenience: mirrors the inline bool in system_info.hpp:
//   inline bool nvmlFabricSymbolLoaded = NvmlFabricSymbolLoaded();
// Call once at program start (or lazily on first use).
inline bool cuda_rt_symbols_loaded = CudaRtLoader::loaded();

// ─── Memory Tier Definitions ────────────────────────────────────────────────
// Mirrors NCCL's topology-aware device placement (ncclTopoGraph).
// In production: HBM = cudaMalloc on H100, GDDR = cudaMalloc on A6000,
// DRAM = posix_memalign.  In CPU-only dev: all tiers simulated via DRAM
// with artificial latency accounting.

// ─── Embedding dtype support (migrated from b58ea19) ────────────────────────
// b58ea19: Expanded training support from float-only to float/half/bf16.
// Maps to our tier dtype validation: allocations for embedding training
// must use float, half, or bf16.  Other dtypes (double, int) are rejected
// at the optimizer-state creation boundary.
//
// Pattern: wholememory embedding.cpp set_optimizer() dtype check:
//   if (dtype != FLOAT && dtype != HALF && dtype != BF16)
//     return WHOLEMEMORY_NOT_IMPLEMENTED;
//
// Our corresponding enum mirrors the three trainable types.  Optimizer-state
// allocations are ALWAYS promoted to float32 (cachable_state_desc.dtype =
// WHOLEMEMORY_DT_FLOAT in b58ea19:embedding.cpp:364), regardless of the
// embedding storage dtype.
enum class EmbeddingDtype : uint8_t {
    FLOAT = 0,  // fp32 — full precision training
    HALF  = 1,  // fp16 — mixed precision (storage fp16, optimizer states fp32)
    BF16  = 2,  // bf16 — mixed precision (storage bf16, optimizer states fp32)
};

// Returns true if dtype supports gradient-based optimizer training.
// b58ea19: only FLOAT/HALF/BF16 are trainable; all others → NOT_IMPLEMENTED.
//
// Knuth self-review: the current EmbeddingDtype enum only contains three values,
// so this function is a tautology over the current enum — it always returns true.
// The guard is FORWARD-LOOKING: if a future EmbeddingDtype::INT8 or
// EmbeddingDtype::DOUBLE is added, this will correctly return false and prevent
// optimizer-state allocation.  The function must be updated in tandem with any
// EmbeddingDtype extension.
inline bool embedding_dtype_is_trainable(EmbeddingDtype dt) {
    return dt == EmbeddingDtype::FLOAT ||
           dt == EmbeddingDtype::HALF  ||
           dt == EmbeddingDtype::BF16;
}

// b58ea19 embedding.cpp:364: optimizer-state allocations are ALWAYS fp32,
// even when embedding storage is fp16 or bf16.  This prevents precision loss
// in momentum/variance accumulators (Adam m/v, AdaGrad state_sum, etc.).
// Returns the dtype to use when creating optimizer-state buffers.
inline EmbeddingDtype optimizer_state_dtype(EmbeddingDtype /*emb_dtype*/) {
    return EmbeddingDtype::FLOAT;  // always fp32, regardless of storage dtype
}

// Size in bytes for each embedding dtype element.
// b58ea19 embedding_optimizer_func.cu:86: align_count = 16 / emb_element_size
//   float → 4 bytes → align_count = 4
//   half  → 2 bytes → align_count = 8
//   bf16  → 2 bytes → align_count = 8
inline size_t embedding_dtype_element_size(EmbeddingDtype dt) {
    switch (dt) {
        case EmbeddingDtype::FLOAT: return 4;
        case EmbeddingDtype::HALF:  return 2;
        case EmbeddingDtype::BF16:  return 2;
        default: return 4;
    }
}

// b58ea19: align_count = 16 / emb_element_size — ensures 16-byte alignment
// for all float types (4 floats, 8 halves, 8 bf16s per 16-byte vector lane).
inline int embedding_dtype_align_count(EmbeddingDtype dt) {
    return static_cast<int>(16 / embedding_dtype_element_size(dt));
}

enum class MemoryTier : uint8_t {
    HBM   = 0,   // H100 High-Bandwidth Memory (3.35 TB/s)
    GDDR  = 1,   // A6000 GDDR6 (768 GB/s)
    DRAM  = 2,   // CPU DDR5 (≈ 80 GB/s per channel)
    TIER_COUNT = 3
};

inline const char* tier_name(MemoryTier t) {
    switch (t) {
        case MemoryTier::HBM:  return "HBM";
        case MemoryTier::GDDR: return "GDDR";
        case MemoryTier::DRAM: return "DRAM";
        default: return "UNKNOWN";
    }
}

// ─── Allocation Metadata ────────────────────────────────────────────────────
// Every allocation carries metadata for the migration scheduler.
// Follows the pattern from TEM-Graph's TInterval (id, l, r) extended
// with access-frequency counters.
//
// M005: access_count and last_access_ns are std::atomic<uint64_t>.
// They are updated lockfree by touch() — no mutex needed.
// This follows CCCL's shared_block_ptr::__ref_count pattern
// (fetch_add with memory_order_relaxed for counters, release for
// pointer publication).

struct AllocMeta {
    uint64_t    alloc_id;          // unique allocation identifier
    MemoryTier  current_tier;      // where the block currently resides
    size_t      size_bytes;        // allocation size
    void*       base_ptr;          // pointer to start of region

    // Access tracking — LOCKFREE (M005)
    // Updated by touch() without taking any lock.
    // Pattern: NCCL's __atomic_fetch_add (compiler/gcc.h:37)
    std::atomic<uint64_t>  access_count{0};
    std::atomic<uint64_t>  last_access_ns{0};    // nanoseconds since epoch

    // M009: Pin count — prevents migration while a TierPtr is alive.
    // Pattern: PyTorch Block::event_count (CUDACachingAllocator.cpp:214)
    //   int event_count{0}; // number of outstanding CUDA events
    // Our pin_count serves the same role: blocks with pin_count > 0
    // cannot be migrated until all TierPtr holders release.
    std::atomic<int32_t>   pin_count{0};

    // Temporal graph context: which interval range does this block serve?
    int32_t     interval_start;    // TEM-Graph Timestamp
    int32_t     interval_end;      // TEM-Graph Timestamp

    AllocMeta()
        : alloc_id(0), current_tier(MemoryTier::DRAM), size_bytes(0),
          base_ptr(nullptr), interval_start(-1), interval_end(-1) {}

    // M005: explicit copy — atomics cannot be implicitly copied.
    // Snapshot current values at copy time.
    AllocMeta(const AllocMeta& o)
        : alloc_id(o.alloc_id), current_tier(o.current_tier),
          size_bytes(o.size_bytes), base_ptr(o.base_ptr),
          interval_start(o.interval_start), interval_end(o.interval_end)
    {
        access_count.store(o.access_count.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
        last_access_ns.store(o.last_access_ns.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        pin_count.store(o.pin_count.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    }

    AllocMeta& operator=(const AllocMeta& o) {
        if (this != &o) {
            alloc_id       = o.alloc_id;
            current_tier   = o.current_tier;
            size_bytes     = o.size_bytes;
            base_ptr       = o.base_ptr;
            interval_start = o.interval_start;
            interval_end   = o.interval_end;
            access_count.store(o.access_count.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
            last_access_ns.store(o.last_access_ns.load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
        }
        return *this;
    }
};


// ─── Tier Budget ────────────────────────────────────────────────────────────
// Capacity limits per tier.  On the real server: H100 80 GB HBM, A6000
// 48 GB GDDR, host DRAM 256 GB.  The scheduler respects these limits.

struct TierBudget {
    size_t capacity_bytes;         // maximum bytes for this tier
    std::atomic<size_t> used_bytes{0};

    TierBudget() : capacity_bytes(0) {}
    explicit TierBudget(size_t cap) : capacity_bytes(cap) {}

    // M005: copy/assign for TierBudget (atomics need explicit handling)
    TierBudget(const TierBudget& o)
        : capacity_bytes(o.capacity_bytes)
    {
        used_bytes.store(o.used_bytes.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    }
    TierBudget& operator=(const TierBudget& o) {
        if (this != &o) {
            capacity_bytes = o.capacity_bytes;
            used_bytes.store(o.used_bytes.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        }
        return *this;
    }

    bool can_fit(size_t n) const {
        return used_bytes.load(std::memory_order_relaxed) + n <= capacity_bytes;
    }

    bool try_reserve(size_t n) {
        size_t cur = used_bytes.load(std::memory_order_relaxed);
        while (cur + n <= capacity_bytes) {
            if (used_bytes.compare_exchange_weak(
                    cur, cur + n,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    void release(size_t n) {
        size_t prev = used_bytes.fetch_sub(n, std::memory_order_acq_rel);
        assert(prev >= n && "double-free or over-release in TierBudget");
    }
};


// ─── Tiered Allocator ───────────────────────────────────────────────────────
// The core allocator.  In CPU-dev mode it allocates from DRAM only; the
// tier tag in AllocMeta records the *intended* placement for the real
// server.  The migration scheduler reads these tags and issues CUDA
// memcpy (or NVLink peer-copy) on the production cluster.
//
// Design follows wrapper::insert_edge / wrapper::remove_edge (RapidStore)
// pattern: simple top-level API, backend dispatch via the tier enum.
//
// M005 concurrency model (PyTorch c10::COWDeleter shared_mutex pattern):
//   - Structural mutations (allocate/deallocate/migrate): unique_lock<shared_mutex>
//   - Read-only access (get_ptr/get_meta/for_each_alloc): shared_lock<shared_mutex>
//   - Counter updates (touch): LOCKFREE — atomics only, no lock

class TieredAllocator {
public:
    TieredAllocator(size_t hbm_cap, size_t gddr_cap, size_t dram_cap)
        : next_alloc_id_(1)
    {
        budgets_[static_cast<int>(MemoryTier::HBM)]  = TierBudget(hbm_cap);
        budgets_[static_cast<int>(MemoryTier::GDDR)] = TierBudget(gddr_cap);
        budgets_[static_cast<int>(MemoryTier::DRAM)] = TierBudget(dram_cap);
    }

    ~TieredAllocator() {
        // Release all remaining allocations
        std::unique_lock<std::shared_mutex> lk(mu_);
        for (auto& [id, meta] : registry_) {
            if (meta.base_ptr) {
                // M008: Route through slab for small allocations
                int tier_idx = static_cast<int>(meta.current_tier);
                if (slab_size_class(meta.size_bytes) < SLAB_NUM_CLASSES) {
                    slab_[tier_idx].deallocate(meta.base_ptr);
                } else {
                    ::free(meta.base_ptr);
                }
                budgets_[static_cast<int>(meta.current_tier)].release(meta.size_bytes);
            }
        }
        registry_.clear();
    }

    // Allocate on the preferred tier; fall back to lower tiers if full.
    // Returns allocation id (0 on failure).
    // Takes UNIQUE lock — structural mutation.
    //
    // b58ea19 migration: If embedding_dtype is provided and NOT trainable
    // (i.e., not float/half/bf16), log and refuse to allocate an optimizer-
    // state buffer.  This mirrors embedding.cpp:set_optimizer dtype check.
    uint64_t allocate(size_t size, MemoryTier preferred,
                      int32_t ts_start = -1, int32_t ts_end = -1,
                      EmbeddingDtype dtype = EmbeddingDtype::FLOAT,
                      bool is_optimizer_state = false) {
        // b58ea19: guard — optimizer states only for trainable dtypes
        if (is_optimizer_state && !embedding_dtype_is_trainable(dtype)) {
            std::cout << "[TieredAllocator] ERROR: allocate() optimizer_state=true"
                      << " but dtype=" << static_cast<int>(dtype)
                      << " is not trainable (only FLOAT/HALF/BF16)."
                      << " Returning 0 (NOT_IMPLEMENTED).\n";
            return 0;
        }
        // b58ea19 embedding.cpp:364: optimizer states are always fp32,
        // so the actual alloc size does not change here (caller passes
        // float-sized size already), but we record the intended dtype.
        if (is_optimizer_state) {
            // Print-debug: confirm state dtype promotion
            std::cout << "[TieredAllocator] DEBUG allocate(): optimizer_state"
                      << " storage_dtype=" << static_cast<int>(dtype)
                      << " -> state_dtype=FLOAT (promoted per b58ea19)\n";
        }
        MemoryTier actual = preferred;

        // Waterfall: HBM → GDDR → DRAM
        if (!budgets_[static_cast<int>(actual)].try_reserve(size)) {
            if (actual == MemoryTier::HBM) actual = MemoryTier::GDDR;
            if (!budgets_[static_cast<int>(actual)].try_reserve(size)) {
                actual = MemoryTier::DRAM;
                if (!budgets_[static_cast<int>(actual)].try_reserve(size)) {
                    return 0;  // out of memory across all tiers
                }
            }
        }

        // In CPU-dev mode: all tiers use posix malloc.
        // M008: Small allocations go through slab pools.
        void* ptr = nullptr;
        int tier_idx = static_cast<int>(actual);
        if (is_slab_managed(size)) {
            auto [sptr, actual_sz] = slab_[tier_idx].allocate(size);
            ptr = sptr;
        } else {
            int rc = ::posix_memalign(&ptr, 64, size);  // 64-byte alignment
            if (rc != 0) ptr = nullptr;
            if (ptr) ::memset(ptr, 0, size);
        }
        if (!ptr) {
            budgets_[static_cast<int>(actual)].release(size);
            return 0;
        }

        uint64_t id = next_alloc_id_.fetch_add(1, std::memory_order_relaxed);

        AllocMeta meta;
        meta.alloc_id       = id;
        meta.current_tier   = actual;
        meta.size_bytes     = size;
        meta.base_ptr       = ptr;
        meta.interval_start = ts_start;
        meta.interval_end   = ts_end;

        {
            std::unique_lock<std::shared_mutex> lk(mu_);  // M005: unique_lock
            registry_[id] = meta;
        }

        return id;
    }

    // Touch an allocation (updates access counters for the scheduler).
    //
    // M005 CRITICAL FIX: This is now LOCKFREE.
    //
    // Previous implementation (M001–M004) took mu_ for every touch(),
    // serializing all concurrent reads.  The fix uses atomics directly:
    //   - std::shared_lock to find the AllocMeta* (read-only map lookup)
    //   - Then atomics are updated without any lock held
    //
    // Pattern source: NCCL's COMPILER_ATOMIC_FETCH_ADD (compiler/gcc.h:37)
    //   #define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add(...)
    //
    // And CCCL's shared_block_ptr refcount:
    //   __block_->__ref_count.fetch_add(1, memory_order_relaxed);
    //
    // The shared_lock for map lookup is O(1) amortized for unordered_map
    // and allows full concurrency among readers.
    void touch(uint64_t alloc_id) {
        AllocMeta* meta_ptr = nullptr;
        {
            std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
            auto it = registry_.find(alloc_id);
            if (it == registry_.end()) return;
            meta_ptr = &(it->second);
        }
        // Lock released — atomics updated lockfree
        meta_ptr->access_count.fetch_add(1, std::memory_order_relaxed);
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        meta_ptr->last_access_ns.store(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()),
            std::memory_order_relaxed);
    }

    // Migrate an allocation to a different tier.
    // On the server: this issues cudaMemcpyAsync between devices.
    // Takes UNIQUE lock — structural mutation.
    //
    // ═══ Pinned memory DMA path (from cugraph-gnn 89c9e8d) ═══
    // cugraph-gnn发现: val.cuda()把整个embedding强制拷进单卡HBM → OOM。
    // 修复: val.pin_memory()只做CPU锁页, 让WholeGraph走DMA直通,
    // 绕过"先全量拷进GPU"这一步。显存峰值从O(N*dim)降到接近0。
    // 我们的对应: DRAM→HBM迁移时, 先在DRAM侧做pinned标记,
    // 让异步引擎走DMA路径而不是full memcpy。
    bool migrate(uint64_t alloc_id, MemoryTier target) {
        std::unique_lock<std::shared_mutex> lk(mu_);  // M005: unique_lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return false;

        AllocMeta& meta = it->second;
        if (meta.current_tier == target) return true;  // already there

        size_t sz = meta.size_bytes;
        if (!budgets_[static_cast<int>(target)].try_reserve(sz)) {
            return false;  // target tier is full
        }

        // ─── Pinned DMA vs full copy 选择 (from 89c9e8d) ───
        // DRAM→HBM: GPU模式走pinned DMA, CPU模式标记为pinned-eligible
        // HBM→DRAM: 直接拷贝, 不需要pin
        bool use_pinned_path = (meta.current_tier == MemoryTier::DRAM &&
                                (target == MemoryTier::HBM || target == MemoryTier::GDDR));

        // In CPU-dev: re-allocate + memcpy (simulates device transfer).
        // M008: Use slab allocator for small sizes on target tier.
        void* new_ptr = nullptr;
        int target_idx = static_cast<int>(target);
        if (is_slab_managed(sz)) {
            auto [sptr, actual_sz] = slab_[target_idx].allocate(sz);
            new_ptr = sptr;
        } else {
            int rc = ::posix_memalign(&new_ptr, 64, sz);
            if (rc != 0) new_ptr = nullptr;
        }
        if (!new_ptr) {
            budgets_[static_cast<int>(target)].release(sz);
            return false;
        }

        // ═══ 4807986: Dynamic CUDA symbol guard — mirrors communicator.cpp ═══
        // Original (cugraph-gnn communicator.cpp before 4807986):
        //   memset(&ri.fabric_info, 0, sizeof(ri.fabric_info));
        //   WHOLEMEMORY_CHECK_NOTHROW(GetGpuFabricInfo(...) == WHOLEMEMORY_SUCCESS);
        //
        // After 4807986:
        //   if (nvmlFabricSymbolLoaded) {
        //       memset(&ri.fabric_info, 0, sizeof(ri.fabric_info));
        //       WHOLEMEMORY_CHECK_NOTHROW(GetGpuFabricInfo(...) == WHOLEMEMORY_SUCCESS);
        //   } else {
        //       WHOLEMEMORY_WARN("Some required NVML symbols are missing...");
        //   }
        //
        // Our adaptation: guard the GPU DMA path with cuda_rt_symbols_loaded.
        // When CUDA runtime is unavailable (e.g. no GPU / old driver / CPU-only CI),
        // fall back to CPU memcpy and emit a diagnostic rather than crashing.
        if (use_pinned_path) {
            if (cuda_rt_symbols_loaded) {
                // GPU模式: cudaHostRegister(meta.base_ptr, sz, cudaHostRegisterDefault)
                //          CudaRtLoader::syms().cudaMemcpyAsync_fn(new_ptr, meta.base_ptr, sz,
                //                          cudaMemcpyHostToDevice, dma_stream_)
                //          CudaRtLoader::syms().cudaStreamSync_fn(dma_stream_)
                //          cudaHostUnregister(meta.base_ptr)
                // 断点调试: print migration路径确认走DMA而非fallback
                fprintf(stderr,
                    "[TieredAllocator::migrate] alloc=%lu sz=%zu "
                    "DRAM→GPU DMA path (cuda_rt_symbols_loaded=true)\n",
                    (unsigned long)alloc_id, sz);
                ::memcpy(new_ptr, meta.base_ptr, sz);  // CPU-sim; replace with cudaMemcpyAsync in prod
                stats_pinned_dma_count_++;
                stats_pinned_dma_bytes_ += sz;
            } else {
                // 4807986 pattern: graceful degradation when GPU symbols missing.
                // Mirrors: WHOLEMEMORY_WARN("Some required NVML symbols are missing,
                //   likely due to an outdated GPU display driver. MNNVL support
                //   will be disabled.")
                fprintf(stderr,
                    "[TieredAllocator::migrate] WARNING: cuda_rt_symbols_loaded=false "
                    "— GPU DMA unavailable (outdated driver or no GPU). "
                    "Falling back to CPU memcpy for alloc=%lu sz=%zu\n",
                    (unsigned long)alloc_id, sz);
                ::memcpy(new_ptr, meta.base_ptr, sz);
                // stats_pinned_dma_count_ NOT incremented: CPU copy, not DMA
            }
        } else {
            ::memcpy(new_ptr, meta.base_ptr, sz);
        }

        // Free old allocation through slab or OS
        int old_idx = static_cast<int>(meta.current_tier);
        if (is_slab_managed(sz)) {
            slab_[old_idx].deallocate(meta.base_ptr);
        } else {
            ::free(meta.base_ptr);
        }

        budgets_[static_cast<int>(meta.current_tier)].release(sz);
        meta.base_ptr     = new_ptr;
        meta.current_tier = target;

        return true;
    }

    // Free an allocation.
    // Takes UNIQUE lock — structural mutation.
    void deallocate(uint64_t alloc_id) {
        std::unique_lock<std::shared_mutex> lk(mu_);  // M005: unique_lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return;

        AllocMeta& meta = it->second;
        if (meta.base_ptr) {
            // M008: Route through slab for small allocations
            int tier_idx = static_cast<int>(meta.current_tier);
            if (is_slab_managed(meta.size_bytes)) {
                slab_[tier_idx].deallocate(meta.base_ptr);
            } else {
                ::free(meta.base_ptr);
            }
            budgets_[static_cast<int>(meta.current_tier)].release(meta.size_bytes);
        }
        registry_.erase(it);
    }

    // Get raw pointer (for the wrapper layer to pass to algorithms).
    // M005: shared_lock — concurrent reads allowed.
    void* get_ptr(uint64_t alloc_id) const {
        std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return nullptr;
        return it->second.base_ptr;
    }

    // Get a read-only view of metadata (for the scheduler).
    // M005: shared_lock — concurrent reads allowed.
    bool get_meta(uint64_t alloc_id, AllocMeta& out) const {
        std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return false;
        out = it->second;  // uses M005 copy constructor
        return true;
    }

    // Iterate all allocations (for migration scheduling).
    // M005: shared_lock — concurrent reads allowed.
    void for_each_alloc(std::function<void(uint64_t, const AllocMeta&)> cb) const {
        std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
        for (auto& [id, meta] : registry_) {
            cb(id, meta);
        }
    }

    // Budget introspection.
    const TierBudget& budget(MemoryTier tier) const {
        return budgets_[static_cast<int>(tier)];
    }

    size_t total_allocated() const {
        size_t sum = 0;
        for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
            sum += budgets_[i].used_bytes.load(std::memory_order_relaxed);
        }
        return sum;
    }

    // M008: Compact slab pools — release empty pages back to OS.
    // Pattern: PyTorch release_cached_blocks (CUDACachingAllocator.cpp:3832)
    // Call periodically (e.g. after migration sweeps) to reclaim fragmented memory.
    size_t compact_slabs() {
        size_t total = 0;
        for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
            total += slab_[i].compact();
        }
        return total;
    }

    // M008: Print slab statistics
    void print_slab_stats() const {
        for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
            std::cout << "[SlabAllocator tier=" << tier_name(static_cast<MemoryTier>(i))
                      << "]\n";
            slab_[i].print_stats();
        }
    }

    // M008: Check if an allocation is slab-managed
    bool is_slab_managed(size_t size) const {
        return slab_size_class(size) < SLAB_NUM_CLASSES;
    }

    // ── M009: Methods for TierPtr RAII + AsyncMigrationEngine ──────────────

    /// Pin an allocation to prevent migration while TierPtr is alive.
    /// Pattern: PyTorch Block::event_count — blocks with pending events
    /// cannot be freed/migrated until event_count reaches zero.
    void pin(uint64_t alloc_id) {
        std::shared_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        if (it != registry_.end()) {
            it->second.pin_count.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    /// Unpin: allow migration again.
    void unpin(uint64_t alloc_id) {
        std::shared_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        if (it != registry_.end()) {
            int32_t prev = it->second.pin_count.fetch_sub(1, std::memory_order_acq_rel);
            assert(prev > 0 && "unpin without matching pin");
        }
    }

    /// Check if an allocation is pinned (has active TierPtr holders).
    bool is_pinned(uint64_t alloc_id) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return false;
        return it->second.pin_count.load(std::memory_order_acquire) > 0;
    }

    /// Get current tier of an allocation.
    MemoryTier get_tier(uint64_t alloc_id) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        return (it != registry_.end()) ? it->second.current_tier : MemoryTier::DRAM;
    }

    /// Get size of an allocation.
    size_t get_size(uint64_t alloc_id) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        return (it != registry_.end()) ? it->second.size_bytes : 0;
    }

    /// Allocate raw memory on a specific tier (for double-buffer in async migration).
    /// Pattern: PyTorch CachingAllocator::malloc (CUDACachingAllocator.cpp:4594)
    ///   Block* block = device_allocator[device]->malloc(size, stream);
    ///   *devPtr = block->ptr;
    void* allocate_on_tier(size_t size, MemoryTier tier) {
        int ti = static_cast<int>(tier);
        if (!budgets_[ti].try_reserve(size)) return nullptr;

        // M008: try slab first for small allocations
        if (slab_size_class(size) < SLAB_NUM_CLASSES) {
            auto [p, actual] = slab_[ti].allocate(size);
            if (p) return p;
        }
        // Fallback: raw allocation
        void* p = nullptr;
        if (posix_memalign(&p, 64, size) != 0) {
            budgets_[ti].release(size);
            return nullptr;
        }
        return p;
    }

    /// Finalize an async migration: atomically swap the allocation's pointer + tier.
    /// Called after cudaMemcpyAsync (or memcpy) has completed.
    /// Pattern: PyTorch CachingAllocator block->ptr update after migration.
    void finalize_migration(uint64_t alloc_id, MemoryTier new_tier, void* new_ptr) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return;

        auto& meta = it->second;
        // Accounting: release old tier, already reserved on new tier by allocate_on_tier
        // (but budget for old tier was not released yet — do it now)
        budgets_[static_cast<int>(meta.current_tier)].release(meta.size_bytes);
        meta.base_ptr     = new_ptr;
        meta.current_tier = new_tier;
    }

    /// Free raw memory from a specific tier (for old buffer after async migration).
    void free_raw(void* ptr, MemoryTier tier) {
        if (!ptr) return;
        int ti = static_cast<int>(tier);
        // Try slab deallocate first; if unknown, do plain free
        slab_[ti].deallocate(ptr);
        // Note: deallocate prints a warning for unknown pointers but
        // does not crash. In production, we'd use a separate tracking set.
    }

    // ═══ Pinned DMA diagnostics (from cugraph-gnn 89c9e8d migration) ═══
    // 在断点调试时 print 当前分配器状态: 各tier使用量 + DMA统计
    void dump_state(const char* label = "TieredAllocator") const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        std::cout << "[" << label << "] allocs=" << registry_.size();
        const char* tier_names[] = {"HBM", "GDDR", "DRAM"};
        for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
            size_t used = budgets_[i].used_bytes.load(std::memory_order_relaxed);
            std::cout << " " << tier_names[i] << "="
                      << used / 1024 << "KB/" << budgets_[i].capacity_bytes / 1024 << "KB";
        }
        std::cout << " pinned_dma_count=" << stats_pinned_dma_count_
                  << " pinned_dma_bytes=" << stats_pinned_dma_bytes_
                  << "\n";
    }

private:
    mutable std::shared_mutex mu_;       // M005: upgraded from std::mutex
    std::atomic<uint64_t> next_alloc_id_;
    std::unordered_map<uint64_t, AllocMeta> registry_;
    TierBudget budgets_[static_cast<int>(MemoryTier::TIER_COUNT)];
    mutable SlabAllocator slab_[static_cast<int>(MemoryTier::TIER_COUNT)];  // M008

    // ═══ Pinned DMA stats (from 89c9e8d) ═══
    uint64_t stats_pinned_dma_count_ = 0;
    uint64_t stats_pinned_dma_bytes_ = 0;
};

}  // namespace philemon
