/**
 * interval_index_bench.cpp — M012: IntervalIndex vs Linear Scan Benchmark
 *
 * Produces 1000-point × 3-seed × 3-query-type × 2-method JSON data matching
 * the data_demo X-axis dimensions.
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -I src -o idx_bench \
 *       src/bench/interval_index_bench.cpp
 *
 * Milestone: M012 (Claude #6)
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

static std::vector<TemporalEdge>
gen_edges(size_t n, int32_t ts_max, uint64_t mv, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<TemporalEdge> e; e.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        auto u1 = std::uniform_real_distribution<>(0,1)(rng);
        auto u2 = std::uniform_real_distribution<>(0,1)(rng);
        uint64_t s = (uint64_t)(u1*u1*mv), d = (uint64_t)(u2*u2*mv);
        if (s==d) d=(d+1)%mv;
        int32_t t0 = std::uniform_int_distribution<int32_t>(0,ts_max)(rng);
        int32_t dur = std::uniform_int_distribution<int32_t>(1,200)(rng);
        int32_t t1 = std::min(t0+dur, ts_max);
        e.emplace_back(s,d,1.0,t0,t1);
    }
    return e;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Philemon-TSH M012: IntervalIndex Benchmark              ║\n";
    std::cout << "║  Indexed vs Linear Scan × 3 query types                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    const int N_STEPS = 1000;
    const int N_SEEDS = 3;
    const int32_t TS_MAX = 100000;
    const size_t N_EDGES = 500000;   // edges per bridge
    const uint64_t SEEDS[] = {42, 137, 271};

    std::vector<double> steps(N_STEPS);
    for (int i = 0; i < N_STEPS; ++i) steps[i] = i * (40960.0 / N_STEPS);

    // query_type → method → seed → latency_vec
    std::map<std::string, std::map<std::string, std::map<std::string, std::vector<double>>>> data;

    struct QueryType { const char* name; int32_t width; };
    QueryType qtypes[] = {
        {"narrow",  50},
        {"medium", 500},
        {"wide",  5000},
    };

    for (auto& qt : qtypes) {
        std::cout << "[" << qt.name << " query, width=" << qt.width << "]\n";

        for (int seed_idx = 0; seed_idx < N_SEEDS; ++seed_idx) {
            std::string sk = "seed_" + std::to_string(seed_idx);
            std::cout << "  seed=" << seed_idx << " building bridge... " << std::flush;

            TieredAllocator alloc(32*1024*1024, 64*1024*1024, 256*1024*1024);
            TierPlacementPolicy policy(1000000000ULL, 5000000000ULL);
            TemporalBridge bridge(alloc, policy, 100000);

            auto edges = gen_edges(N_EDGES, TS_MAX, 500000, SEEDS[seed_idx]);
            bridge.add_edges(edges);
            bridge.flush_partitions();
            edges.clear();

            std::cout << "querying... " << std::flush;
            std::mt19937_64 qrng(SEEDS[seed_idx] + 1000);

            std::vector<double> linear_lat, indexed_lat;
            linear_lat.reserve(N_STEPS);
            indexed_lat.reserve(N_STEPS);

            for (int step = 0; step < N_STEPS; ++step) {
                int32_t lo = std::uniform_int_distribution<int32_t>(0, TS_MAX - qt.width)(qrng);
                int32_t hi = lo + qt.width;

                // Linear scan (original scan_partition path)
                uint64_t count_lin = 0;
                auto t0 = hrc::now();
                bridge.temporal_subgraph_query(lo, hi,
                    [&](const TemporalEdge&) { ++count_lin; });
                auto t1 = hrc::now();
                double us_lin = std::chrono::duration<double, std::micro>(t1 - t0).count();
                linear_lat.push_back(us_lin);

                // Indexed contains_query (M011)
                uint64_t count_idx = 0;
                t0 = hrc::now();
                bridge.indexed_contains_query(lo, hi,
                    [&](const TemporalEdge&) { ++count_idx; });
                t1 = hrc::now();
                double us_idx = std::chrono::duration<double, std::micro>(t1 - t0).count();
                indexed_lat.push_back(us_idx);

                // Correctness check (spot-check every 100 steps)
                if (step % 100 == 0 && count_lin != count_idx) {
                    std::cerr << "  WARNING: mismatch at step " << step
                              << " linear=" << count_lin << " indexed=" << count_idx << "\n";
                }
            }

            data[qt.name]["LinearScan"][sk] = linear_lat;
            data[qt.name]["Indexed"][sk] = indexed_lat;
            std::cout << "done\n";
        }
    }

    // Write JSON
    std::string path = "philemon_interval_index_2000.json";
    std::ofstream out(path);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"panel\": \"Interval Query Latency: Indexed vs Linear\",\n";
    out << "    \"source\": \"Philemon-TSH M012\",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << ",\n";
    out << "    \"n_edges\": " << N_EDGES << ",\n";
    out << "    \"query_types\": [\"narrow\",\"medium\",\"wide\"]\n";
    out << "  },\n";
    out << "  \"steps\": " << vj(steps) << ",\n";
    out << "  \"panels\": {\n";

    bool first_qt = true;
    for (auto& [qtname, methods] : data) {
        if (!first_qt) out << ",\n";
        first_qt = false;
        out << "    \"" << qtname << "\": {\n";
        out << "      \"title\": \"" << qtname << " query\",\n";
        out << "      \"methods\": {\n";

        bool first_m = true;
        for (auto& [mname, seeds] : methods) {
            if (!first_m) out << ",\n";
            first_m = false;
            out << "        \"" << mname << "\": {\n";

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
    return 0;
}
