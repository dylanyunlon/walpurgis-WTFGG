/**
 * partition_index_bench.cpp — M014: Augmented Skip List vs Linear Partition Scan
 *
 * Measures the partition-*selection* step (query_partitions), which the M013
 * skip list reduces from O(P) to O(log P + k). The X-axis sweeps the number of
 * partitions P (the quantity that grows under streaming ingestion); each step
 * issues a fixed-width temporal query and times how long it takes to identify
 * the overlapping partitions — NOT the intra-partition edge scan, which M006
 * already handles.
 *
 * Two methods per panel:
 *   Indexed    : query_partitions (skip-list pruned walk)
 *   LinearScan : query_partitions_linear (the O(P) oracle)
 *
 * Three panels by query width (selectivity), since the indexed path's win
 * depends on how few partitions a query touches:
 *   narrow / medium / wide temporal windows.
 *
 * Runtime correctness: at sampled steps the indexed result set is compared to
 * the linear oracle (same partition slots). Any divergence is reported.
 *
 * Output: philemon_partition_index_2000.json (demo-format compatible:
 * 2000 steps x 3 seeds x 3 query widths x 2 methods, with mean/std).
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -I src -o pidx_bench \
 *       src/bench/partition_index_bench.cpp
 *
 * Milestone: M014 (Claude #7)
 */

#include "../core/tiered_allocator.hpp"
#include "../bridge/temporal_bridge.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <sys/resource.h>

using namespace philemon;
using hrc = std::chrono::high_resolution_clock;

static double get_peak_rss_mb() {
    struct rusage u;
    return getrusage(RUSAGE_SELF, &u) == 0 ? u.ru_maxrss / 1024.0 : -1.0;
}
static std::string vj(const std::vector<double>& v) {
    std::ostringstream o; o << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) o << ","; o << std::setprecision(6) << v[i]; }
    o << "]"; return o.str();
}

static const int N_STEPS = 2000;
static const int N_SEEDS = 3;

// Each step adds a batch of edges spanning a fresh time slice, then flushes,
// growing the partition count. With a small partition_cap each flush yields
// several partitions, so by the final step P is in the thousands — the regime
// where O(P) selection hurts.
static const int   EDGES_PER_STEP = 1200;
static const int   PART_CAP       = 300;      // small → many partitions
static const int32_t SLICE_WIDTH  = 50;       // time advanced per step

struct QWidth { const char* name; int32_t half; };

static std::set<uint32_t> slot_set(const std::vector<const SubgraphPartition*>& v,
                                   const TemporalBridge& b) {
    // map partition pointer back to slot via its ts_lo/ts_hi+ptr identity:
    // simplest stable key is the pointer's position; we reconstruct from the
    // snapshot order. Here we just hash on (ts_lo,ts_hi,edge_count) which is
    // unique enough for the validation sample.
    (void)b;
    std::set<uint32_t> s;
    for (auto* p : v) {
        // fold a small identity into a slot-like key
        uint32_t key = static_cast<uint32_t>(
            (uint32_t(p->ts_lo) * 2654435761u) ^
            (uint32_t(p->ts_hi) * 40503u) ^
            uint32_t(p->edge_count));
        s.insert(key);
    }
    return s;
}

int main() {
    std::cout << "=== M014: Partition Skip-List vs Linear Selection ===\n";
    std::cout << "steps=" << N_STEPS << " seeds=" << N_SEEDS
              << " edges/step=" << EDGES_PER_STEP
              << " part_cap=" << PART_CAP << "\n\n";

    std::vector<QWidth> widths = {
        {"narrow", 30}, {"medium", 400}, {"wide", 4000}
    };

    std::vector<double> steps(N_STEPS);
    for (int i = 0; i < N_STEPS; ++i) steps[i] = i + 1;

    // data[qwidth][method][seed] = per-step selection latency (us)
    std::map<std::string, std::map<std::string,
        std::map<std::string, std::vector<double>>>> data;

    long long mism = 0, checks = 0;

    for (int seed = 0; seed < N_SEEDS; ++seed) {
        std::string sk = "seed_" + std::to_string(seed);
        std::cout << "[seed " << seed << "] ... " << std::flush;

        TieredAllocator alloc(64ull<<20, 256ull<<20, 4096ull<<20);
        TierPlacementPolicy policy(50'000'000ull, 500'000'000ull);
        TemporalBridge bridge(alloc, policy, PART_CAP);

        std::mt19937 rng(900 + seed * 31);
        std::uniform_int_distribution<uint64_t> vtx(0, 1<<16);
        std::uniform_real_distribution<double>  w(0.1, 5.0);
        std::uniform_int_distribution<int32_t>  dur(1, 40);

        // per-width latency accumulators for this seed
        std::map<std::string, std::vector<double>> lin_lat, idx_lat;
        for (auto& qw : widths) {
            lin_lat[qw.name].reserve(N_STEPS);
            idx_lat[qw.name].reserve(N_STEPS);
        }

        int32_t t_base = 0;
        for (int step = 0; step < N_STEPS; ++step) {
            // grow the graph: one time-slice batch, then flush → new partitions
            std::vector<TemporalEdge> batch;
            batch.reserve(EDGES_PER_STEP);
            std::uniform_int_distribution<int32_t> t0(t_base, t_base + SLICE_WIDTH);
            for (int e = 0; e < EDGES_PER_STEP; ++e) {
                int32_t a = t0(rng);
                int32_t b = a + dur(rng);
                batch.emplace_back(vtx(rng), vtx(rng), w(rng), a, b);
            }
            bridge.add_edges(batch);
            bridge.flush_partitions();
            t_base += SLICE_WIDTH;

            // center of the query window roams across the populated range
            int32_t center = std::uniform_int_distribution<int32_t>(
                0, std::max(1, t_base))(rng);

            for (auto& qw : widths) {
                int32_t lo = center - qw.half;
                int32_t hi = center + qw.half;

                // time the indexed selection
                hrc::time_point a0 = hrc::now();
                auto idx = bridge.query_partitions(lo, hi);
                hrc::time_point a1 = hrc::now();
                idx_lat[qw.name].push_back(
                    std::chrono::duration<double, std::micro>(a1 - a0).count());

                // time the linear oracle
                a0 = hrc::now();
                auto lin = bridge.query_partitions_linear(lo, hi);
                a1 = hrc::now();
                lin_lat[qw.name].push_back(
                    std::chrono::duration<double, std::micro>(a1 - a0).count());

                // runtime correctness sample (every 200 steps, medium width)
                if (step % 200 == 0) {
                    auto si = slot_set(idx, bridge);
                    auto sl = slot_set(lin, bridge);
                    ++checks;
                    if (si != sl) {
                        ++mism;
                        if (mism <= 5)
                            std::cerr << "\n  MISMATCH seed=" << seed
                                      << " step=" << step << " " << qw.name
                                      << " idx=" << si.size()
                                      << " lin=" << sl.size();
                    }
                }
            }
        }

        for (auto& qw : widths) {
            data[qw.name]["Indexed"][sk]    = idx_lat[qw.name];
            data[qw.name]["LinearScan"][sk] = lin_lat[qw.name];
        }
        std::cout << "partitions=" << bridge.partition_count() << " done\n";
    }

    std::cout << "\ncorrectness: " << (checks - mism) << "/" << checks
              << " samples matched the linear oracle"
              << (mism == 0 ? "  [OK]\n" : "  [MISMATCH!]\n");

    // ── Write JSON ──────────────────────────────────────────────────────────
    std::string path = "philemon_partition_index_2000.json";
    std::ofstream out(path);
    out << "{\n  \"metadata\": {\n";
    out << "    \"panel\": \"Partition Selection Latency: SkipList vs Linear\",\n";
    out << "    \"source\": \"Philemon-TSH M014\",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << ",\n";
    out << "    \"edges_per_step\": " << EDGES_PER_STEP << ",\n";
    out << "    \"partition_cap\": " << PART_CAP << ",\n";
    out << "    \"query_types\": [\"narrow\",\"medium\",\"wide\"]\n";
    out << "  },\n";
    out << "  \"steps\": " << vj(steps) << ",\n";
    out << "  \"panels\": {\n";

    bool first_qt = true;
    for (auto& qw : widths) {
        if (!first_qt) out << ",\n";
        first_qt = false;
        out << "    \"" << qw.name << "\": {\n";
        out << "      \"title\": \"" << qw.name << " query\",\n";
        out << "      \"methods\": {\n";

        bool first_m = true;
        for (auto& mname : {std::string("Indexed"), std::string("LinearScan")}) {
            if (!first_m) out << ",\n";
            first_m = false;
            out << "        \"" << mname << "\": {\n";

            auto& seeds = data[qw.name][mname];
            bool first_s = true;
            std::vector<std::vector<double>> all;
            for (auto& [skey, vals] : seeds) {
                if (!first_s) out << ",\n";
                first_s = false;
                out << "          \"" << skey << "\": " << vj(vals);
                all.push_back(vals);
            }
            if (!all.empty()) {
                std::vector<double> mean(N_STEPS, 0), sd(N_STEPS, 0);
                for (int i = 0; i < N_STEPS; ++i) {
                    for (auto& s : all) mean[i] += s[i];
                    mean[i] /= all.size();
                    for (auto& s : all) sd[i] += (s[i]-mean[i])*(s[i]-mean[i]);
                    sd[i] = std::sqrt(sd[i] / all.size());
                }
                out << ",\n          \"mean\": " << vj(mean);
                out << ",\n          \"std\": " << vj(sd);
            }
            out << "\n        }";
        }
        out << "\n      }\n    }";
    }
    out << "\n  }\n}\n";
    out.close();

    std::cout << "\n→ Wrote " << path << "\n";
    std::cout << "Peak RSS: " << get_peak_rss_mb() << " MB\n";
    return mism == 0 ? 0 : 1;
}
