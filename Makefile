# Makefile — Philemon-TSH: Temporal Subgraph on Heterogeneous Memory
#
# Targets:
#   make cpu          — CPU-only benchmark (dev VM, no GPU needed)
#   make cuda         — CUDA heterogeneous benchmark (ags1: A6000×2 + H100)
#   make all          — both
#   make clean

CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O2 -pthread -Wall
NVFLAGS  := -std=c++14 -O2 -Xcompiler "-pthread -fopenmp -Wall" -lineinfo

# Detect CUDA version for arch flags and compression tuning
# b89f57d migration: detect nvcc version to gate -Xfatbin=--compress-level=3
#   (only valid for CUDA 12.9.x; base -Xfatbin=-compress-all applies to all
#    recent nvcc versions — mirrors the RAPIDS WHOLEGRAPH pattern)
CUDA_VER := $(shell nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//')
CUDA_MAJOR := $(shell echo $(CUDA_VER) | cut -d. -f1)
CUDA_MINOR := $(shell echo $(CUDA_VER) | cut -d. -f2)

# b89f57d: Enable device code compression to reduce binary sizes.
# Base flag: always append -Xfatbin=-compress-all when nvcc is present.
# Tune flag: -Xfatbin=--compress-level=3 is only available on CUDA 12.9.x.
# WALPURGIS_DEBUG=1 → print resolved compression flags before each nvcc build.
FATBIN_COMPRESS_BASE := $(shell \
  nvcc --version >/dev/null 2>&1 && echo "-Xfatbin=-compress-all" || echo "")
FATBIN_COMPRESS_TUNE := $(shell \
  [ "$(CUDA_MAJOR)" = "12" ] && [ "$(CUDA_MINOR)" = "9" ] && \
  echo "-Xfatbin=--compress-level=3" || echo "")
FATBIN_FLAGS := $(FATBIN_COMPRESS_BASE) $(FATBIN_COMPRESS_TUNE)
NVFLAGS      += $(FATBIN_FLAGS)

# sm_86 for A6000, compute_80 PTX for H100 JIT (CUDA 11.5 compatible)
ARCH_FLAGS := -arch=sm_86 -gencode=arch=compute_80,code=compute_80

.PHONY: all cpu cuda clean

all: cpu cuda

cpu: philemon_bench

cuda: hetero_bench

philemon_bench: src/bench/philemon_bench.cpp src/core/tiered_allocator.hpp \
                src/bridge/temporal_bridge.hpp src/scheduler/migration_scheduler.hpp
	$(CXX) $(CXXFLAGS) -I src -o $@ src/bench/philemon_bench.cpp

# M013/M014: partition skip-list self-test + selection benchmark
skiplist_selftest: src/bench/skiplist_selftest.cpp src/core/partition_skiplist.hpp
	$(CXX) $(CXXFLAGS) -I src -o $@ src/bench/skiplist_selftest.cpp

pidx_bench: src/bench/partition_index_bench.cpp src/core/partition_skiplist.hpp \
            src/bridge/temporal_bridge.hpp src/core/tiered_allocator.hpp
	$(CXX) $(CXXFLAGS) -I src -o $@ src/bench/partition_index_bench.cpp

hetero_bench: src/cuda/hetero_bench.cu
	@# b89f57d WALPURGIS_DEBUG: print resolved fatbin compression flags before compile
	@if [ "$$WALPURGIS_DEBUG" = "1" ]; then \
	  echo "[WALPURGIS_DEBUG b89f57d fatbin] CUDA_VER=$(CUDA_VER) CUDA_MAJOR=$(CUDA_MAJOR) CUDA_MINOR=$(CUDA_MINOR)"; \
	  echo "[WALPURGIS_DEBUG b89f57d fatbin] FATBIN_COMPRESS_BASE=$(FATBIN_COMPRESS_BASE)"; \
	  echo "[WALPURGIS_DEBUG b89f57d fatbin] FATBIN_COMPRESS_TUNE=$(FATBIN_COMPRESS_TUNE)"; \
	  echo "[WALPURGIS_DEBUG b89f57d fatbin] FATBIN_FLAGS=$(FATBIN_FLAGS)"; \
	  echo "[WALPURGIS_DEBUG b89f57d fatbin] full NVFLAGS=$(NVFLAGS)"; \
	fi
	$(NVCC) $(NVFLAGS) $(ARCH_FLAGS) -o $@ $<

clean:
	rm -f philemon_bench hetero_bench skiplist_selftest pidx_bench
