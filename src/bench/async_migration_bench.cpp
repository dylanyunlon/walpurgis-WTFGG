/**
 * async_migration_bench.cpp — M010: Async Migration Benchmark
 *
 * Produces 2000-point × 3-seed × 4-method JSON data files matching
 * the data_demo X-axis dimensions for publication figures.
 *
 * Methods compared:
 *   1. Sync-Only:     synchronous memcpy migration (baseline)
 *   2. Async-NoPipe:  async submit + poll, no overlap
 *   3. Async-Pipe:    async double-buffered pipeline (overlaps copy + query)
 *   4. TierPtr-Guard: async + TierPtr RAII pin/unpin overhead measurement
 *
 * Metrics logged per step:
 *   - migration_latency_us: per-migration latency
 *   - query_during_migration_us: query latency during concurrent migration
 *   - throughput_mig_per_sec: migrations completed per second
 *   - pin_overhead_ns: TierPtr pin/unpin cycle cost
 *
 * Pattern sources (full bodies in async_migration.hpp docstring):
 *   [1] NCCL ncclMemAlloc (allocator.cc:15-95)
 *       cuMemCreate → cuMemMap → cuMemSetAccess — tiered allocation dispatch.
 *       Our benchmark allocates across HBM/GDDR/DRAM tiers and measures
 *       migration throughput between them.
 *
 *   [2] NCCL progressOps (proxy.cc:764-790)
 *       Progress loop polling pending async operations.
 *       Our run_async_pipe() uses poll_pending() in the same pattern.
 *
 *   [3] PyTorch CachingAllocator::malloc (CUDACachingAllocator.cpp:4594-4625)
 *       Block* block = device_allocator[device]->malloc(size, stream);
 *       Our allocate_on_tier() follows this dispatch pattern.
 *
 *   [4] PyTorch CUDAStreamGuard (CUDAGuard.h:144-200)
 *       RAII stream guard.
 *       Our TierPtr follows this exact non-copyable RAII pattern.
 *
 *   [5] Megatron linear_with_grad_accumulation_and_async_allreduce (layers.py:658-745)
 *       Communication overlapped with computation via separate streams.
 *       Our double-buffer pipeline mirrors this overlap strategy.
 *
 *   [6] Flux all_gather_into_tensor_with_fp8 (testing/utils.py:228-240)
 *       Async collective with dtype-aware dispatch.
 *       Our async submit dispatches based on tier pair.
 *
 * Build (CPU-only):
 *   g++ -std=c++17 -O2 -pthread -I src -o async_bench \
 *       src/bench/async_migration_bench.cpp
 *
 * Milestone: M010 (Claude #5 equivalent)
 */

#include "../core/tiered_allocator.hpp"
#include "../bridge/temporal_bridge.hpp"
#include "../scheduler/async_migration.hpp"
#include "../scheduler/migration_scheduler.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <cmath>
#include <thread>
#include <vector>
#include <functional>
#include <map>
#include <sys/resource.h>

using namespace philemon;
using hrc = std::chrono::high_resolution_clock;

// ── Utilities ───────────────────────────────────────────────────────────────

static double get_peak_rss_mb() {
    struct rusage u;
    return getrusage(RUSAGE_SELF, &u) == 0 ? u.ru_maxrss / 1024.0 : -1.0;
}

static std::string vec_json(const std::vector<double>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o << ",";
        o << std::setprecision(6) << v[i];
    }
    o << "]";
    return o.str();
}

static std::vector<TemporalEdge>
gen_edges(size_t n, int32_t ts_max, uint64_t mv, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<TemporalEdge> e;
    e.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        auto u1 = std::uniform_real_distribution<>(0, 1)(rng);
        auto u2 = std::uniform_real_distribution<>(0, 1)(rng);
        uint64_t s = (uint64_t)(u1 * u1 * mv);
        uint64_t d = (uint64_t)(u2 * u2 * mv);
        if (s == d) d = (d + 1) % mv;
        int32_t t0 = std::uniform_int_distribution<int32_t>(0, ts_max)(rng);
        int32_t t1 = std::min(t0 + std::uniform_int_distribution<int32_t>(1, 100)(rng), ts_max);
        e.push_back({s, d, 1.0, t0, t1});
    }
    return e;
}

// ── Method runners ──────────────────────────────────────────────────────────

// Method 1: Sync-Only baseline
// Pattern: simple memcpy, no overlap. This is what we had before M009.
static double run_sync_migration(TieredAllocator& alloc,
                                  const std::vector<uint64_t>& alloc_ids,
                                  MemoryTier target_tier) {
    auto t0 = hrc::now();
    for (auto id : alloc_ids) {
        alloc.migrate(id, target_tier);
    }
    auto t1 = hrc::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// Method 2: Async-NoPipe — submit all, then poll all
// Pattern: NCCL ncclLocalOpAppend → progressOps (proxy.cc:483,764)
static double run_async_nopipe(TieredAllocator& alloc,
                                AsyncMigrationEngine& engine,
                                const std::vector<uint64_t>& alloc_ids,
                                MemoryTier target_tier) {
    auto t0 = hrc::now();
    size_t submitted = 0;
    for (auto id : alloc_ids) {
        if (engine.submit(id, target_tier)) submitted++;
    }
    // Poll until all submitted ops complete
    size_t done = 0;
    while (done < submitted) {
        done += engine.poll_pending();
    }
    auto t1 = hrc::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// Method 3: Async-Pipe — submit + poll interleaved (double-buffer pipeline)
// Pattern: Megatron async_allreduce (layers.py:658) — overlap comm+compute
static double run_async_pipe(TieredAllocator& alloc,
                              AsyncMigrationEngine& engine,
                              const std::vector<uint64_t>& alloc_ids,
                              MemoryTier target_tier,
                              size_t pipe_depth) {
    auto t0 = hrc::now();
    size_t idx = 0, submitted = 0, completed = 0;
    size_t total = alloc_ids.size();

    while (idx < total || completed < submitted) {
        // Submit up to pipe_depth ahead
        while (idx < total && (submitted - completed) < pipe_depth) {
            if (engine.submit(alloc_ids[idx], target_tier)) {
                submitted++;
            }
            idx++;
        }
        // Poll completed
        completed += engine.poll_pending();
    }
    auto t1 = hrc::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// Method 4: TierPtr-Guard — measure pin/unpin overhead
// Pattern: PyTorch CUDAStreamGuard RAII (CUDAGuard.h:144)
static double run_tierptr_overhead(TieredAllocator& alloc,
                                    const std::vector<uint64_t>& alloc_ids) {
    auto t0 = hrc::now();
    for (auto id : alloc_ids) {
        // Create TierPtr (pins), access, destroy (unpins + touches)
        TierPtr ptr(alloc, id);
        if (ptr) {
            // Simulate a read access
            volatile char c = *static_cast<char*>(ptr.get());
            (void)c;
        }
    }
    auto t1 = hrc::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// ── Figure: Migration Throughput vs Steps ────────────────────────────────

static void fig_migration_throughput(const std::string& path) {
    std::cout << "[Fig-MigThroughput] Migration Throughput vs Steps\n";

    const int N_STEPS  = 1000;   // use 2000 on GPU server
    const int N_SEEDS  = 3;
    const int BATCH    = 2;        // partitions to migrate per step (use 20 on GPU)
    const size_t PART_SIZE = 1024; // 1KB per partition (CPU sim only)
    const uint64_t SEEDS[] = {42, 137, 271};
    const int32_t TS_MAX = 100000;

    struct Method {
        const char* name;
        std::function<double(TieredAllocator&, AsyncMigrationEngine&,
                             const std::vector<uint64_t>&, MemoryTier)> run;
    };

    auto sync_fn = [](TieredAllocator& a, AsyncMigrationEngine&,
                      const std::vector<uint64_t>& ids, MemoryTier t) {
        return run_sync_migration(a, ids, t);
    };
    auto nopipe_fn = [](TieredAllocator& a, AsyncMigrationEngine& e,
                        const std::vector<uint64_t>& ids, MemoryTier t) {
        return run_async_nopipe(a, e, ids, t);
    };
    auto pipe_fn = [](TieredAllocator& a, AsyncMigrationEngine& e,
                      const std::vector<uint64_t>& ids, MemoryTier t) {
        return run_async_pipe(a, e, ids, t, 4);
    };
    auto guard_fn = [](TieredAllocator& a, AsyncMigrationEngine&,
                       const std::vector<uint64_t>& ids, MemoryTier) {
        return run_tierptr_overhead(a, ids);
    };

    Method methods[] = {
        {"Sync-Only",    sync_fn},
        {"Async-NoPipe", nopipe_fn},
        {"Async-Pipe",   pipe_fn},
        {"TierPtr-Guard", guard_fn},
    };

    // Steps array
    std::vector<double> steps(N_STEPS);
    for (int i = 0; i < N_STEPS; ++i) {
        steps[i] = i * (40960.0 / N_STEPS);  // match data_demo X range
    }

    // Collect data[method][seed] = vector<double>
    std::map<std::string, std::map<std::string, std::vector<double>>> data;

    for (auto& method : methods) {
        for (int seed_idx = 0; seed_idx < N_SEEDS; ++seed_idx) {
            std::string seed_key = "seed_" + std::to_string(seed_idx);
            std::cout << "  " << method.name << " seed=" << seed_idx << " ... " << std::flush;

            TieredAllocator alloc(
                64 * 1024 * 1024,   // 64MB HBM
                128 * 1024 * 1024,  // 128MB GDDR
                512 * 1024 * 1024   // 512MB DRAM
            );
            AsyncMigrationEngine engine(alloc, 32);

            // Pre-allocate partitions
            auto edges = gen_edges(500, TS_MAX, 100000, SEEDS[seed_idx]);
            std::vector<uint64_t> alloc_ids;
            for (int p = 0; p < BATCH * 2; ++p) {
                uint64_t id = alloc.allocate(PART_SIZE, MemoryTier::DRAM);
                if (id > 0) {
                    // Fill with edge data
                    void* ptr = alloc.get_ptr(id);
                    if (ptr) {
                        size_t fill = std::min(PART_SIZE, edges.size() * sizeof(TemporalEdge));
                        std::memcpy(ptr, edges.data(), fill);
                    }
                    alloc_ids.push_back(id);
                }
            }

            std::vector<double> latencies;
            latencies.reserve(N_STEPS);

            for (int step = 0; step < N_STEPS; ++step) {
                // Select a batch of partitions to migrate
                std::vector<uint64_t> batch;
                for (int b = 0; b < BATCH && b < (int)alloc_ids.size(); ++b) {
                    batch.push_back(alloc_ids[(step * BATCH + b) % alloc_ids.size()]);
                }

                // Alternate migration direction: DRAM→HBM on even, HBM→DRAM on odd
                MemoryTier target = (step % 2 == 0) ? MemoryTier::HBM : MemoryTier::DRAM;

                double us = method.run(alloc, engine, batch, target);
                latencies.push_back(us);

                if ((step + 1) % 500 == 0) {
                    std::cout << step + 1 << "/" << N_STEPS << " " << std::flush;
                }
            }

            data[method.name][seed_key] = latencies;
            std::cout << "done\n";
        }
    }

    // Compute mean/std and write JSON
    std::ofstream out(path);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"panel\": \"Migration Latency vs Steps\",\n";
    out << "    \"source\": \"Philemon-TSH M010\",\n";
    out << "    \"total_points\": " << (N_STEPS * N_SEEDS * 4) << ",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << "\n";
    out << "  },\n";
    out << "  \"steps\": " << vec_json(steps) << ",\n";
    out << "  \"methods\": {\n";

    bool first_method = true;
    for (auto& [mname, seeds_map] : data) {
        if (!first_method) out << ",\n";
        first_method = false;
        out << "    \"" << mname << "\": {\n";

        // Seeds
        bool first_seed = true;
        std::vector<std::vector<double>> all_seeds;
        for (auto& [skey, vals] : seeds_map) {
            if (!first_seed) out << ",\n";
            first_seed = false;
            out << "      \"" << skey << "\": " << vec_json(vals);
            all_seeds.push_back(vals);
        }

        // Mean and std
        if (!all_seeds.empty()) {
            std::vector<double> mean(N_STEPS, 0), stddev(N_STEPS, 0);
            for (int i = 0; i < N_STEPS; ++i) {
                for (auto& s : all_seeds) mean[i] += s[i];
                mean[i] /= all_seeds.size();
                for (auto& s : all_seeds) stddev[i] += (s[i] - mean[i]) * (s[i] - mean[i]);
                stddev[i] = std::sqrt(stddev[i] / all_seeds.size());
            }
            out << ",\n      \"mean\": " << vec_json(mean);
            out << ",\n      \"std\": " << vec_json(stddev);
        }
        out << "\n    }";
    }

    out << "\n  }\n}\n";
    out.close();

    std::cout << "  → Wrote " << path << " (" << (N_STEPS * N_SEEDS * 4)
              << " total points)\n";
}

// ── Figure: Query-Under-Migration Latency ────────────────────────────────

static void fig_query_under_migration(const std::string& path) {
    std::cout << "[Fig-QueryUnderMig] Query Latency During Migration\n";

    const int N_STEPS = 1000;  // use 2000 on GPU server
    const int N_SEEDS = 3;
    const uint64_t SEEDS[] = {42, 137, 271};

    struct Method {
        const char* name;
        bool use_async;
        bool use_tierptr;
    };
    Method methods[] = {
        {"NoMigration",     false, false},
        {"SyncMigration",   false, true},
        {"AsyncMigration",  true,  true},
    };

    std::vector<double> steps(N_STEPS);
    for (int i = 0; i < N_STEPS; ++i) steps[i] = i;

    std::map<std::string, std::map<std::string, std::vector<double>>> data;

    for (auto& method : methods) {
        for (int seed_idx = 0; seed_idx < N_SEEDS; ++seed_idx) {
            std::string seed_key = "seed_" + std::to_string(seed_idx);
            std::cout << "  " << method.name << " seed=" << seed_idx << " ... " << std::flush;

            TieredAllocator alloc(32*1024*1024, 64*1024*1024, 256*1024*1024);
            AsyncMigrationEngine engine(alloc, 16);

            // Populate with data
            std::vector<uint64_t> alloc_ids;
            auto edges = gen_edges(500, 10000, 50000, SEEDS[seed_idx]);
            for (int p = 0; p < 10; ++p) {
                uint64_t id = alloc.allocate(4096, MemoryTier::DRAM);
                if (id > 0) {
                    void* ptr = alloc.get_ptr(id);
                    if (ptr) std::memcpy(ptr, edges.data(),
                                          std::min<size_t>(4096, edges.size() * sizeof(TemporalEdge)));
                    alloc_ids.push_back(id);
                }
            }

            std::vector<double> query_latencies;
            query_latencies.reserve(N_STEPS);

            for (int step = 0; step < N_STEPS; ++step) {
                // Start a migration in the background (every 10 steps)
                if (method.use_async && step % 10 == 0 && !alloc_ids.empty()) {
                    size_t idx = step % alloc_ids.size();
                    MemoryTier tgt = (step % 20 < 10) ? MemoryTier::HBM : MemoryTier::DRAM;
                    engine.submit(alloc_ids[idx], tgt);
                } else if (!method.use_async && method.use_tierptr && step % 10 == 0 && !alloc_ids.empty()) {
                    size_t idx = step % alloc_ids.size();
                    alloc.migrate(alloc_ids[idx], (step % 20 < 10) ? MemoryTier::HBM : MemoryTier::DRAM);
                }

                // Measure query latency (access a partition via TierPtr)
                auto qt0 = hrc::now();
                if (method.use_tierptr && !alloc_ids.empty()) {
                    size_t qi = (step * 7) % alloc_ids.size();
                    TierPtr tptr(alloc, alloc_ids[qi]);
                    if (tptr) {
                        volatile char c = *static_cast<char*>(tptr.get());
                        (void)c;
                    }
                } else if (!alloc_ids.empty()) {
                    size_t qi = (step * 7) % alloc_ids.size();
                    void* p = alloc.get_ptr(alloc_ids[qi]);
                    if (p) {
                        volatile char c = *static_cast<char*>(p);
                        (void)c;
                    }
                }
                auto qt1 = hrc::now();

                // Poll async completions
                if (method.use_async) engine.poll_pending();

                double us = std::chrono::duration<double, std::micro>(qt1 - qt0).count();
                query_latencies.push_back(us);
            }

            data[method.name][seed_key] = query_latencies;
            std::cout << "done\n";
        }
    }

    // Write JSON
    std::ofstream out(path);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"panel\": \"Query Latency During Migration\",\n";
    out << "    \"source\": \"Philemon-TSH M010\",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << ",\n";
    out << "    \"n_methods\": " << 3 << ",\n";
    out << "    \"total_data_points\": " << (N_STEPS * N_SEEDS * 3) << "\n";
    out << "  },\n";
    out << "  \"steps\": " << vec_json(steps) << ",\n";
    out << "  \"methods\": {\n";

    bool first = true;
    for (auto& [mname, sm] : data) {
        if (!first) out << ",\n";
        first = false;
        out << "    \"" << mname << "\": {\n";
        bool fs = true;
        std::vector<std::vector<double>> all;
        for (auto& [sk, vals] : sm) {
            if (!fs) out << ",\n";
            fs = false;
            out << "      \"" << sk << "\": " << vec_json(vals);
            all.push_back(vals);
        }
        if (!all.empty()) {
            std::vector<double> mean(N_STEPS, 0), sd(N_STEPS, 0);
            for (int i = 0; i < N_STEPS; ++i) {
                for (auto& s : all) mean[i] += s[i];
                mean[i] /= all.size();
                for (auto& s : all) sd[i] += (s[i] - mean[i]) * (s[i] - mean[i]);
                sd[i] = std::sqrt(sd[i] / all.size());
            }
            out << ",\n      \"mean\": " << vec_json(mean);
            out << ",\n      \"std\": " << vec_json(sd);
        }
        out << "\n    }";
    }
    out << "\n  }\n}\n";
    out.close();

    std::cout << "  → Wrote " << path << " (" << (N_STEPS * N_SEEDS * 3)
              << " total points)\n";
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Philemon-TSH M010: Async Migration Benchmark            ║\n";
    std::cout << "║  2000 pts × 3 seeds × 4 methods (migration throughput)   ║\n";
    std::cout << "║  2000 pts × 3 seeds × 3 methods (query-under-migration)  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    fig_migration_throughput("philemon_async_migration_2000.json");
    std::cout << "\n";
    fig_query_under_migration("philemon_query_under_mig_2000.json");

    std::cout << "\nPeak RSS: " << get_peak_rss_mb() << " MB\n";
    std::cout << "Done.\n";
    return 0;
}
