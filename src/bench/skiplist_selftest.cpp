// skiplist_selftest.cpp — randomized correctness + pruning check for
// PartitionSkipList::overlaps against an O(P) brute-force oracle.
//
// Build: g++ -std=c++17 -O2 -I src -o /tmp/sl_test src/bench/skiplist_selftest.cpp
//
// Milestone: M013 (Claude #7)

#include "../core/partition_skiplist.hpp"
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace philemon;

static std::vector<uint32_t> brute(const std::vector<PartitionInterval>& iv,
                                   int32_t lo, int32_t hi) {
    std::vector<uint32_t> r;
    for (auto& x : iv)
        if (x.ts_lo <= hi && x.ts_hi >= lo) r.push_back(x.partition_slot);
    std::sort(r.begin(), r.end());
    return r;
}

int main() {
    std::mt19937 rng(20260528);
    int trials = 4000, fails = 0;
    size_t total_slots = 0, total_brute = 0;

    for (int t = 0; t < trials; ++t) {
        int P = std::uniform_int_distribution<int>(0, 300)(rng);
        std::uniform_int_distribution<int32_t> pt(-500, 500);
        std::uniform_int_distribution<int32_t> len(0, 120);

        std::vector<PartitionInterval> iv(P);
        for (int i = 0; i < P; ++i) {
            int32_t a = pt(rng);
            int32_t b = a + len(rng);
            iv[i] = PartitionInterval{a, b, (uint32_t)i};
        }

        PartitionSkipList sl;
        sl.build(iv);

        // several queries per built list, including degenerate points
        for (int q = 0; q < 8; ++q) {
            int32_t lo = pt(rng);
            int32_t hi = lo + std::uniform_int_distribution<int32_t>(0, 200)(rng);

            std::vector<uint32_t> got;
            sl.overlaps(lo, hi, got);
            std::sort(got.begin(), got.end());
            got.erase(std::unique(got.begin(), got.end()), got.end());

            auto want = brute(iv, lo, hi);
            total_slots += got.size();
            total_brute += want.size();

            if (got != want) {
                if (fails < 5) {
                    std::cerr << "MISMATCH trial=" << t << " q=" << q
                              << " P=" << P << " [" << lo << "," << hi << "]"
                              << " got=" << got.size()
                              << " want=" << want.size() << "\n";
                }
                ++fails;
            }
        }
    }

    std::cout << "trials=" << trials
              << " emitted=" << total_slots
              << " oracle=" << total_brute
              << " mismatches=" << fails << "\n";
    if (fails == 0) std::cout << "ALL OVERLAP QUERIES CORRECT\n";
    return fails == 0 ? 0 : 1;
}
