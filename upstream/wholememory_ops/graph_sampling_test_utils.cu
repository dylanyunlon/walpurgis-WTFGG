/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/* [migrate 7e914aa] fix to difference in cpu and gpu precision in sample
 * 沉默呵，不在沉默中爆发，就在沉默中消亡——
 * CPU与GPU之间的精度裂缝，便是那无声的消亡。
 * exp2f(-one_bit) 代替 pow(2, -one_bit)，单精度一以贯之。
 */
#include "graph_sampling_test_utils.hpp"

#include <algorithm>
#include <experimental/random>
#include <gtest/gtest.h>
#include <iterator>
#include <queue>
#include <random>
#include <vector>

#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <wholememory_ops/register.hpp>

namespace wholegraph_ops {
namespace testing {

// [DEBUG] 全局调试开关，设为1则打印详细断点信息
static int g_walpurgis_debug = 1;

template <typename DataType>
void host_get_csr_graph(int64_t graph_node_count,
                        int64_t graph_edge_count,
                        void* host_csr_row_ptr,
                        wholememory_array_description_t graph_csr_row_ptr_desc,
                        void* host_csr_col_ptr,
                        wholememory_array_description_t graph_csr_col_ptr_desc)
{
  // [DEBUG BP-01] 进入host_get_csr_graph，检验图结构参数
  if (g_walpurgis_debug) {
    printf("[BP-01][host_get_csr_graph] node_count=%ld edge_count=%ld\n",
           (long)graph_node_count, (long)graph_edge_count);
  }

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  DataType* csr_col_ptr         = static_cast<DataType*>(host_csr_col_ptr);
  int64_t average_edge_per_node = graph_edge_count / graph_node_count;

  std::default_random_engine generator;
  std::binomial_distribution<int64_t> distribution(average_edge_per_node, 1);

  int total_edge = 0;

  for (int64_t i = 0; i < graph_node_count; i++) {
    while (true) {
      int64_t random_num = distribution(generator);
      if (random_num >= 0 && random_num <= graph_node_count) {
        csr_row_ptr[i] = random_num;
        total_edge += random_num;
        break;
      }
    }
  }

  // 边数偏差修正——削多补少，不偏不倚
  int64_t adjust_edge = std::abs(total_edge - graph_edge_count);
  std::random_device rand_dev;
  std::mt19937 gen(rand_dev());
  std::uniform_int_distribution<int64_t> distr(0, graph_node_count - 1);
  if (total_edge > graph_edge_count) {
    for (int64_t i = 0; i < adjust_edge; i++) {
      while (true) {
        int64_t random_row_id = distr(gen);
        if (csr_row_ptr[random_row_id] > 0) {
          csr_row_ptr[random_row_id]--;
          break;
        }
      }
    }
  }
  if (total_edge < graph_edge_count) {
    for (int64_t i = 0; i < adjust_edge; i++) {
      while (true) {
        int64_t random_row_id = distr(gen);
        if (csr_row_ptr[random_row_id] < graph_node_count) {
          csr_row_ptr[random_row_id]++;
          break;
        }
      }
    }
  }

  host_prefix_sum_array(host_csr_row_ptr, graph_csr_row_ptr_desc);

  EXPECT_TRUE(csr_row_ptr[graph_node_count] == graph_edge_count);

  for (int64_t i = 0; i < graph_node_count; i++) {
    int64_t start      = csr_row_ptr[i];
    int64_t end        = csr_row_ptr[i + 1];
    int64_t edge_count = end - start;
    if (edge_count == 0) continue;
    std::vector<int64_t> array_out(edge_count);
    std::vector<int64_t> array_in(graph_node_count);
    for (int64_t i = 0; i < graph_node_count; i++) {
      array_in[i] = i;
    }

    std::sample(array_in.begin(), array_in.end(), array_out.begin(), edge_count, gen);
    for (int j = 0; j < edge_count; j++) {
      csr_col_ptr[start + j] = (DataType)array_out[j];
    }
  }
}

template <typename DataType>
void host_get_csr_weight_graph(void* host_csr_weight_ptr,
                               wholememory_array_description_t graph_csr_weight_ptr_desc)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DataType> dis(1.0, 20.0);
  for (int64_t i = 0; i < graph_csr_weight_ptr_desc.size; i++) {
    static_cast<DataType*>(host_csr_weight_ptr)[i] = (DataType)dis(gen);
  }
}

void gen_csr_graph(int64_t graph_node_count,
                   int64_t graph_edge_count,
                   void* host_csr_row_ptr,
                   wholememory_array_description_t graph_csr_row_ptr_desc,
                   void* host_csr_col_ptr,
                   wholememory_array_description_t graph_csr_col_ptr_desc,
                   void* host_csr_weight_ptr,
                   wholememory_array_description_t graph_csr_weight_ptr_desc)
{
  EXPECT_TRUE(graph_csr_row_ptr_desc.dtype == WHOLEMEMORY_DT_INT64);

  if (graph_csr_col_ptr_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_csr_graph<int64_t>(graph_node_count,
                                graph_edge_count,
                                host_csr_row_ptr,
                                graph_csr_row_ptr_desc,
                                host_csr_col_ptr,
                                graph_csr_col_ptr_desc);

  } else if (graph_csr_col_ptr_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_csr_graph<int>(graph_node_count,
                            graph_edge_count,
                            host_csr_row_ptr,
                            graph_csr_row_ptr_desc,
                            host_csr_col_ptr,
                            graph_csr_col_ptr_desc);
  }
  if (host_csr_weight_ptr != nullptr) {
    if (graph_csr_weight_ptr_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
      host_get_csr_weight_graph<float>(host_csr_weight_ptr, graph_csr_weight_ptr_desc);
    } else if (graph_csr_weight_ptr_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
      host_get_csr_weight_graph<double>(host_csr_weight_ptr, graph_csr_weight_ptr_desc);
    }
  }
}

template <typename DataType>
void host_get_random_array(void* array,
                           wholememory_array_description_t array_desc,
                           int64_t low,
                           int64_t high)
{
  DataType* array_ptr = static_cast<DataType*>(array);
  std::experimental::reseed();
  for (int64_t i = 0; i < array_desc.size; i++) {
    DataType random_num                      = std::experimental::randint<DataType>(low, high);
    array_ptr[i + array_desc.storage_offset] = random_num;
  }
}

void host_random_init_array(void* array,
                            wholememory_array_description_t array_desc,
                            int64_t low,
                            int64_t high)
{
  EXPECT_TRUE(array_desc.dtype == WHOLEMEMORY_DT_INT || array_desc.dtype == WHOLEMEMORY_DT_INT64);
  if (array_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_random_array<int>(array, array_desc, low, high);
  } else {
    host_get_random_array<int64_t>(array, array_desc, low, high);
  }
}

template <typename DataType>
void host_get_prefix_sum_array(void* array, wholememory_array_description_t array_desc)
{
  DataType* array_ptr = static_cast<DataType*>(array);
  if (array_desc.size <= 0) return;
  DataType old_value = array_ptr[0];
  array_ptr[0]       = 0;
  for (int64_t i = 1; i < array_desc.size; i++) {
    DataType tmp = array_ptr[i];
    array_ptr[i] = array_ptr[i - 1] + old_value;
    old_value    = tmp;
  }
}

void host_prefix_sum_array(void* array, wholememory_array_description_t array_desc)
{
  EXPECT_TRUE(array_desc.dtype == WHOLEMEMORY_DT_INT || array_desc.dtype == WHOLEMEMORY_DT_INT64);
  if (array_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_prefix_sum_array<int>(array, array_desc);
  } else {
    host_get_prefix_sum_array<int64_t>(array, array_desc);
  }
}

void copy_host_array_to_wholememory(void* host_array,
                                    wholememory_handle_t array_handle,
                                    wholememory_array_description_t array_desc,
                                    cudaStream_t stream)
{
  void* local_array_ptr;
  size_t local_array_size, local_array_offset;
  EXPECT_EQ(wholememory_get_local_memory(
              &local_array_ptr, &local_array_size, &local_array_offset, array_handle),
            WHOLEMEMORY_SUCCESS);
  int64_t array_ele_size = wholememory_dtype_get_element_size(array_desc.dtype);
  EXPECT_EQ(local_array_size % array_ele_size, 0);
  EXPECT_EQ(local_array_offset % array_ele_size, 0);
  wholememory_comm_t wm_comm;
  EXPECT_EQ(wholememory_get_communicator(&wm_comm, array_handle), WHOLEMEMORY_SUCCESS);

  if (local_array_size) {
    EXPECT_EQ(cudaMemcpyAsync(local_array_ptr,
                              static_cast<char*>(host_array) + local_array_offset,
                              local_array_size,
                              cudaMemcpyHostToDevice,
                              stream),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
  wholememory_communicator_barrier(wm_comm);
}

template <typename DataType>
void host_get_sample_offset(void* host_csr_row_ptr,
                            wholememory_array_description_t csr_row_ptr_desc,
                            void* host_center_nodes,
                            wholememory_array_description_t center_node_desc,
                            int max_sample_count,
                            void* host_ref_output_sample_offset,
                            wholememory_array_description_t output_sample_offset_desc)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  DataType* center_nodes_ptr    = static_cast<DataType*>(host_center_nodes);
  int* output_sample_offset_ptr = static_cast<int*>(host_ref_output_sample_offset);

  for (int64_t i = 0; i < center_node_desc.size; i++) {
    DataType center_node_id = center_nodes_ptr[i];
    int neighbor_node_count = csr_row_ptr[center_node_id + 1] - csr_row_ptr[center_node_id];
    if (max_sample_count > 0) {
      neighbor_node_count = std::min(neighbor_node_count, max_sample_count);
    }
    output_sample_offset_ptr[i] = neighbor_node_count;
  }
}

template <typename IdType, typename ColIdType>
void host_sample_all(void* host_csr_row_ptr,
                     wholememory_array_description_t csr_row_ptr_desc,
                     void* host_csr_col_ptr,
                     wholememory_array_description_t csr_col_ptr_desc,
                     void* host_center_nodes,
                     wholememory_array_description_t center_node_desc,
                     int max_sample_count,
                     void* host_ref_output_sample_offset,
                     wholememory_array_description_t output_sample_offset_desc,
                     void* host_ref_output_dest_nodes,
                     void* host_ref_output_center_nodes_local_id,
                     void* host_ref_output_global_edge_id)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  ColIdType* csr_col_ptr        = static_cast<ColIdType*>(host_csr_col_ptr);
  IdType* center_nodes_ptr      = static_cast<IdType*>(host_center_nodes);
  int* output_sample_offset_ptr = static_cast<int*>(host_ref_output_sample_offset);

  ColIdType* output_dest_nodes_ptr      = static_cast<ColIdType*>(host_ref_output_dest_nodes);
  int* output_center_nodes_local_id_ptr = static_cast<int*>(host_ref_output_center_nodes_local_id);
  int64_t* output_global_edge_id_ptr    = static_cast<int64_t*>(host_ref_output_global_edge_id);

  int64_t center_nodes_count = center_node_desc.size;

  for (int64_t i = 0; i < center_nodes_count; i++) {
    int output_id         = output_sample_offset_ptr[i];
    int output_local_id   = 0;
    IdType center_node_id = center_nodes_ptr[i];
    for (int64_t j = csr_row_ptr[center_node_id]; j < csr_row_ptr[center_node_id + 1]; j++) {
      output_dest_nodes_ptr[output_id + output_local_id]            = csr_col_ptr[j];
      output_center_nodes_local_id_ptr[output_id + output_local_id] = (int)i;
      output_global_edge_id_ptr[output_id + output_local_id]        = j;
      output_local_id++;
    }
  }
}

REGISTER_DISPATCH_TWO_TYPES(HOSTSAMPLEALL, host_sample_all, SINT3264, SINT3264)

template <int Offset = 0>
void random_sample_without_replacement_cpu_base(std::vector<int>* a,
                                                const std::vector<int32_t>& r,
                                                int M,
                                                int N)
{
  a->resize(M + Offset);
  std::vector<int> Q(N + Offset);
  for (int i = Offset; i < N + Offset; ++i) {
    Q[i] = i;
  }
  for (int i = Offset; i < M + Offset; ++i) {
    a->at(i) = Q[r[i]];
    Q[r[i]]  = Q[N - i + 2 * Offset - 1];
  }
}

template <typename IdType, typename ColIdType>
void host_unweighted_sample_without_replacement(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_center_nodes,
  wholememory_array_description_t center_node_desc,
  int max_sample_count,
  void* host_ref_output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void* host_ref_output_dest_nodes,
  void* host_ref_output_center_nodes_local_id,
  void* host_ref_output_global_edge_id,
  unsigned long long random_seed)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  ColIdType* csr_col_ptr        = static_cast<ColIdType*>(host_csr_col_ptr);
  IdType* center_nodes_ptr      = static_cast<IdType*>(host_center_nodes);
  int* output_sample_offset_ptr = static_cast<int*>(host_ref_output_sample_offset);

  ColIdType* output_dest_nodes_ptr      = static_cast<ColIdType*>(host_ref_output_dest_nodes);
  int* output_center_nodes_local_id_ptr = static_cast<int*>(host_ref_output_center_nodes_local_id);
  int64_t* output_global_edge_id_ptr    = static_cast<int64_t*>(host_ref_output_global_edge_id);

  int64_t center_nodes_count = center_node_desc.size;
  int M                      = max_sample_count;

  // [DEBUG BP-02] 无权重采样入口：种子、采样数、节点数
  if (g_walpurgis_debug) {
    printf("[BP-02][unweighted_sample] seed=%llu max_sample=%d center_count=%ld\n",
           random_seed, max_sample_count, (long)center_nodes_count);
  }

  static const int warp_count_array[32]       = {1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8,
                                                 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
  static const int items_per_thread_array[32] = {1, 2, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  int func_idx         = (max_sample_count - 1) / 32;
  int device_num_threads = warp_count_array[func_idx] * 32;
  int items_per_thread   = items_per_thread_array[func_idx];

  for (int64_t i = 0; i < center_nodes_count; i++) {
    int output_id          = output_sample_offset_ptr[i];
    int output_local_id    = 0;
    IdType center_node_id  = center_nodes_ptr[i];
    int64_t start          = csr_row_ptr[center_node_id];
    int64_t end            = csr_row_ptr[center_node_id + 1];
    int64_t neighbor_count = end - start;
    int N                  = neighbor_count;
    int blockidx           = i;
    int gidx               = blockidx * device_num_threads;

    if (neighbor_count <= 0) continue;

    if (neighbor_count <= max_sample_count) {
      // 邻居数不足采样数，照单全收
      for (int64_t j = start; j < end; j++) {
        output_dest_nodes_ptr[output_id + output_local_id]            = csr_col_ptr[j];
        output_center_nodes_local_id_ptr[output_id + output_local_id] = (int)i;
        output_global_edge_id_ptr[output_id + output_local_id]        = j;
        output_local_id++;
      }
    } else {
      std::vector<int32_t> r(neighbor_count);
      for (int j = 0; j < device_num_threads; j++) {
        int local_gidx = gidx + j;
        raft::random::RngState _rngstate(random_seed, 0, raft::random::GeneratorType::GenPC);
        raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate(_rngstate);
        raft::random::detail::PCGenerator rng(rngstate, (uint64_t)local_gidx);
        raft::random::detail::UniformDistParams<int32_t> params;
        params.start = 0;
        params.end   = 1;

        for (int k = 0; k < items_per_thread; k++) {
          int id = k * device_num_threads + j;
          int32_t random_num;
          raft::random::detail::custom_next(rng, &random_num, params, 0, 0);
          if (id < neighbor_count) { r[id] = id < M ? (random_num % (N - id)) : N; }
        }
      }

      std::vector<int> random_sample_id(max_sample_count, 0);
      random_sample_without_replacement_cpu_base(&random_sample_id, r, M, N);
      for (int sample_id = 0; sample_id < M; sample_id++) {
        output_dest_nodes_ptr[output_id + sample_id] =
          csr_col_ptr[start + random_sample_id[sample_id]];
        output_center_nodes_local_id_ptr[output_id + sample_id] = i;
        output_global_edge_id_ptr[output_id + sample_id] = start + random_sample_id[sample_id];
      }
    }
  }
}

REGISTER_DISPATCH_TWO_TYPES(HOSTUNWEIGHTEDSAMPLEWITHOUTREPLACEMENT,
                            host_unweighted_sample_without_replacement,
                            SINT3264,
                            SINT3264)

void wholegraph_csr_unweighted_sample_without_replacement_cpu(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_center_nodes,
  wholememory_array_description_t center_node_desc,
  int max_sample_count,
  void** host_ref_output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void** host_ref_output_dest_nodes,
  void** host_ref_output_center_nodes_local_id,
  void** host_ref_output_global_edge_id,
  int* output_sample_dest_nodes_count,
  unsigned long long random_seed)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_sample_offset_desc.size, center_node_desc.size + 1);

  *host_ref_output_sample_offset =
    (void*)malloc(wholememory_get_memory_size_from_array(&output_sample_offset_desc));

  if (center_node_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_sample_offset<int64_t>(host_csr_row_ptr,
                                    csr_row_ptr_desc,
                                    host_center_nodes,
                                    center_node_desc,
                                    max_sample_count,
                                    *host_ref_output_sample_offset,
                                    output_sample_offset_desc);
  } else if (center_node_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_sample_offset<int>(host_csr_row_ptr,
                                csr_row_ptr_desc,
                                host_center_nodes,
                                center_node_desc,
                                max_sample_count,
                                *host_ref_output_sample_offset,
                                output_sample_offset_desc);
  }
  host_prefix_sum_array(*host_ref_output_sample_offset, output_sample_offset_desc);
  *output_sample_dest_nodes_count =
    static_cast<int*>(*host_ref_output_sample_offset)[center_node_desc.size];

  *host_ref_output_dest_nodes            = malloc((*output_sample_dest_nodes_count) *
                                       wholememory_dtype_get_element_size(csr_col_ptr_desc.dtype));
  *host_ref_output_center_nodes_local_id = malloc((*output_sample_dest_nodes_count) * sizeof(int));
  *host_ref_output_global_edge_id = malloc((*output_sample_dest_nodes_count) * sizeof(int64_t));

  if (max_sample_count <= 0) {
    DISPATCH_TWO_TYPES(center_node_desc.dtype,
                       csr_col_ptr_desc.dtype,
                       HOSTSAMPLEALL,
                       host_csr_row_ptr,
                       csr_row_ptr_desc,
                       host_csr_col_ptr,
                       csr_col_ptr_desc,
                       host_center_nodes,
                       center_node_desc,
                       max_sample_count,
                       *host_ref_output_sample_offset,
                       output_sample_offset_desc,
                       *host_ref_output_dest_nodes,
                       *host_ref_output_center_nodes_local_id,
                       *host_ref_output_global_edge_id);
    return;
  }
  if (max_sample_count > 1024) { return; }

  DISPATCH_TWO_TYPES(center_node_desc.dtype,
                     csr_col_ptr_desc.dtype,
                     HOSTUNWEIGHTEDSAMPLEWITHOUTREPLACEMENT,
                     host_csr_row_ptr,
                     csr_row_ptr_desc,
                     host_csr_col_ptr,
                     csr_col_ptr_desc,
                     host_center_nodes,
                     center_node_desc,
                     max_sample_count,
                     *host_ref_output_sample_offset,
                     output_sample_offset_desc,
                     *host_ref_output_dest_nodes,
                     *host_ref_output_center_nodes_local_id,
                     *host_ref_output_global_edge_id,
                     random_seed);
}

template <typename DataType>
void check_value_same(void* value, void* ref, int64_t size)
{
  int64_t diff_count = 0;

  DataType* value_ptr = static_cast<DataType*>(value);
  DataType* ref_ptr   = static_cast<DataType*>(ref);

  for (int i = 0; i < size; i++) {
    if (value_ptr[i] != ref_ptr[i]) {
      if (diff_count < 10 * 1000 * 1000) {
        printf("i=%d, value = %ld, ref = %ld\n",
               i,
               static_cast<int64_t>(value_ptr[i]),
               static_cast<int64_t>(ref_ptr[i]));
        EXPECT_EQ(value_ptr[i], ref_ptr[i]);
      }
      diff_count++;
    }
  }
}

REGISTER_DISPATCH_ONE_TYPE(CHECKVALUESAME, check_value_same, SINT3264)

void host_check_two_array_same(void* host_array,
                               wholememory_array_description_t host_array_desc,
                               void* host_ref,
                               wholememory_array_description_t host_ref_desc)
{
  EXPECT_EQ(host_array_desc.dtype, host_ref_desc.dtype);
  EXPECT_EQ(host_array_desc.size, host_ref_desc.size);
  DISPATCH_ONE_TYPE(
    host_array_desc.dtype, CHECKVALUESAME, host_array, host_ref, host_array_desc.size);
}

// 数one_bit：计算前导零个数，知晓随机数的"骨相"
inline int count_one(unsigned long long num)
{
  int c = 0;
  while (num) {
    num >>= 1;
    c++;
  }
  return 64 - c;
}

/*
 * [7e914aa] host_gen_key_from_weight —— CPU/GPU精度统一
 *
 * 旧实现之患：
 *   u *= pow(2, -one_bit);           // double精度，与GPU不符
 *   float logk = (1/weight) * (log1p(u)/log(2.0));  // double中间量，差之毫厘
 *
 * 今修正之法：
 *   u *= exp2f(-one_bit);            // 单精度exp2f，GPU亦如此
 *   float logk = (log1pf(u)/logf(2.0)) * (1.0f/(float)weight);  // 纯float路径
 *
 * 此乃精度之战，非性能之战。CPU测试若以double算，则与GPU结果相差，
 * 测试自然失败——不是GPU错了，是CPU假装自己更精确。
 */
template <typename WeightType>
float host_gen_key_from_weight(const WeightType weight, raft::random::detail::PCGenerator& rng)
{
  float u = 0.0;
  rng.next(u);
  u = -(0.5 + 0.5 * u);

  uint64_t random_num2 = 0;
  int seed_count       = -1;
  do {
    rng.next(random_num2);
    seed_count++;
  } while (!random_num2);

  int one_bit = count_one(random_num2) + seed_count * 64;

  // [DEBUG BP-03] [7e914aa核心] 检查精度修正前后的中间值
  // 用exp2f(-one_bit)而非pow(2,-one_bit)，保持单精度一致性
  float u_before = u;
  u *= exp2f(-one_bit);  // [FIX 7e914aa] 单精度exp2f，对齐GPU精度
  if (g_walpurgis_debug) {
    printf("[BP-03][host_gen_key_from_weight] weight=%f one_bit=%d u_before=%f u_after=%f\n",
           (float)weight, one_bit, u_before, u);
  }

  // [FIX 7e914aa] 全程float运算：log1pf + logf，不再混入double
  float logk = (log1pf(u) / logf(2.0)) * (1.0f / (float)weight);

  if (g_walpurgis_debug) {
    printf("[BP-03][host_gen_key_from_weight] logk=%f\n", logk);
  }
  // u = random_uniform(0,1), logk = 1/weight *logf(u)
  return logk;
}

template <typename IdType, typename ColIdType, typename WeightType>
void host_weighted_sample_without_replacement(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_csr_weight_ptr,
  wholememory_array_description_t csr_weight_ptr_desc,
  void* host_center_nodes,
  wholememory_array_description_t center_node_desc,
  int max_sample_count,
  void* host_ref_output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void* host_ref_output_dest_nodes,
  void* host_ref_output_center_nodes_local_id,
  void* host_ref_output_global_edge_id,
  unsigned long long random_seed)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  ColIdType* csr_col_ptr        = static_cast<ColIdType*>(host_csr_col_ptr);
  WeightType* csr_weight_ptr    = static_cast<WeightType*>(host_csr_weight_ptr);
  IdType* center_nodes_ptr      = static_cast<IdType*>(host_center_nodes);
  int* output_sample_offset_ptr = static_cast<int*>(host_ref_output_sample_offset);

  ColIdType* output_dest_nodes_ptr      = static_cast<ColIdType*>(host_ref_output_dest_nodes);
  int* output_center_nodes_local_id_ptr = static_cast<int*>(host_ref_output_center_nodes_local_id);
  int64_t* output_global_edge_id_ptr    = static_cast<int64_t*>(host_ref_output_global_edge_id);

  int64_t center_nodes_count = center_node_desc.size;

  // [DEBUG BP-04] 有权重采样入口：统计采样规模
  if (g_walpurgis_debug) {
    printf("[BP-04][weighted_sample] seed=%llu max_sample=%d center_count=%ld\n",
           random_seed, max_sample_count, (long)center_nodes_count);
  }

  int block_size = 128;
  if (max_sample_count > 256) { block_size = 256; }

  for (int64_t i = 0; i < center_nodes_count; i++) {
    int output_id          = output_sample_offset_ptr[i];
    int output_local_id    = 0;
    IdType center_node_id  = center_nodes_ptr[i];
    int64_t start          = csr_row_ptr[center_node_id];
    int64_t end            = csr_row_ptr[center_node_id + 1];
    int64_t neighbor_count = end - start;
    int blockidx           = i;
    int gidx               = blockidx * block_size;

    if (neighbor_count <= 0) continue;

    if (neighbor_count <= max_sample_count) {
      // 邻居稀少，不必动用堆，尽数取来
      for (int64_t j = start; j < end; j++) {
        output_dest_nodes_ptr[output_id + output_local_id]            = csr_col_ptr[j];
        output_center_nodes_local_id_ptr[output_id + output_local_id] = (int)i;
        output_global_edge_id_ptr[output_id + output_local_id]        = j;
        output_local_id++;
      }
    } else {
      // 邻居众多，以priority_queue维护top-K，优胜劣汰
      int process_count = 0;
      struct cmp {
        bool operator()(std::pair<int, WeightType> left, std::pair<int, WeightType> right)
        {
          return (left.second) > (right.second);
        }
      };
      std::priority_queue<std::pair<int, WeightType>, std::vector<std::pair<int, WeightType>>, cmp>
        small_heap;

      auto consume_fun = [&](int id, raft::random::detail::PCGenerator& rng) {
        WeightType edge_weight = csr_weight_ptr[start + id];
        WeightType weight      = host_gen_key_from_weight(edge_weight, rng);
        process_count++;
        if (process_count <= max_sample_count) {
          small_heap.push(std::make_pair(id, weight));
        } else {
          std::pair<int, WeightType> small_heap_top_ele = small_heap.top();
          if (small_heap_top_ele.second < weight) {
            small_heap.pop();
            small_heap.push(std::make_pair(id, weight));
          }
        }
      };

      for (int j = 0; j < block_size; j++) {
        int local_gidx = gidx + j;
        raft::random::RngState _rngstate(random_seed, 0, raft::random::GeneratorType::GenPC);
        raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate(_rngstate);
        raft::random::detail::PCGenerator rng(rngstate, (uint64_t)local_gidx);
        for (int id = j; id < neighbor_count; id += block_size) {
          if (id < neighbor_count) { consume_fun(id, rng); }
        }
      }

      // [DEBUG BP-05] 堆满，输出最终采样结果概况
      if (g_walpurgis_debug && i == 0) {
        printf("[BP-05][weighted_sample] center_node=%ld heap_size=%zu max_sample=%d\n",
               (long)i, small_heap.size(), max_sample_count);
      }

      for (int sample_id = 0; sample_id < max_sample_count; sample_id++) {
        output_dest_nodes_ptr[output_id + sample_id] = csr_col_ptr[start + small_heap.top().first];
        output_center_nodes_local_id_ptr[output_id + sample_id] = i;
        output_global_edge_id_ptr[output_id + sample_id]        = start + small_heap.top().first;
        small_heap.pop();
      }
    }
  }
}

REGISTER_DISPATCH_THREE_TYPES(HOSTWEIGHTEDSAMPLEWITHOUTREPLACEMENT,
                              host_weighted_sample_without_replacement,
                              SINT3264,
                              SINT3264,
                              FLOAT_DOUBLE)

void wholegraph_csr_weighted_sample_without_replacement_cpu(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_csr_weight_ptr,
  wholememory_array_description_t csr_weight_ptr_desc,
  void* host_center_nodes,
  wholememory_array_description_t center_node_desc,
  int max_sample_count,
  void** host_ref_output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void** host_ref_output_dest_nodes,
  void** host_ref_output_center_nodes_local_id,
  void** host_ref_output_global_edge_id,
  int* output_sample_dest_nodes_count,
  unsigned long long random_seed)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_sample_offset_desc.size, center_node_desc.size + 1);

  *host_ref_output_sample_offset =
    (void*)malloc(wholememory_get_memory_size_from_array(&output_sample_offset_desc));

  if (center_node_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_sample_offset<int64_t>(host_csr_row_ptr,
                                    csr_row_ptr_desc,
                                    host_center_nodes,
                                    center_node_desc,
                                    max_sample_count,
                                    *host_ref_output_sample_offset,
                                    output_sample_offset_desc);
  } else if (center_node_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_sample_offset<int>(host_csr_row_ptr,
                                csr_row_ptr_desc,
                                host_center_nodes,
                                center_node_desc,
                                max_sample_count,
                                *host_ref_output_sample_offset,
                                output_sample_offset_desc);
  }
  host_prefix_sum_array(*host_ref_output_sample_offset, output_sample_offset_desc);
  *output_sample_dest_nodes_count =
    static_cast<int*>(*host_ref_output_sample_offset)[center_node_desc.size];

  *host_ref_output_dest_nodes            = malloc((*output_sample_dest_nodes_count) *
                                       wholememory_dtype_get_element_size(csr_col_ptr_desc.dtype));
  *host_ref_output_center_nodes_local_id = malloc((*output_sample_dest_nodes_count) * sizeof(int));
  *host_ref_output_global_edge_id = malloc((*output_sample_dest_nodes_count) * sizeof(int64_t));

  if (max_sample_count <= 0) {
    DISPATCH_TWO_TYPES(center_node_desc.dtype,
                       csr_col_ptr_desc.dtype,
                       HOSTSAMPLEALL,
                       host_csr_row_ptr,
                       csr_row_ptr_desc,
                       host_csr_col_ptr,
                       csr_col_ptr_desc,
                       host_center_nodes,
                       center_node_desc,
                       max_sample_count,
                       *host_ref_output_sample_offset,
                       output_sample_offset_desc,
                       *host_ref_output_dest_nodes,
                       *host_ref_output_center_nodes_local_id,
                       *host_ref_output_global_edge_id);
    return;
  }

  if (max_sample_count > 1024) { return; }

  DISPATCH_THREE_TYPES(center_node_desc.dtype,
                       csr_col_ptr_desc.dtype,
                       csr_weight_ptr_desc.dtype,
                       HOSTWEIGHTEDSAMPLEWITHOUTREPLACEMENT,
                       host_csr_row_ptr,
                       csr_row_ptr_desc,
                       host_csr_col_ptr,
                       csr_col_ptr_desc,
                       host_csr_weight_ptr,
                       csr_weight_ptr_desc,
                       host_center_nodes,
                       center_node_desc,
                       max_sample_count,
                       *host_ref_output_sample_offset,
                       output_sample_offset_desc,
                       *host_ref_output_dest_nodes,
                       *host_ref_output_center_nodes_local_id,
                       *host_ref_output_global_edge_id,
                       random_seed);
}

template <typename DataType>
void host_get_segment_sort(void* host_output_sample_offset,
                           wholememory_array_description_t output_sample_offset_desc,
                           void* host_output_dest_nodes,
                           wholememory_array_description_t output_dest_nodes_desc,
                           void* host_output_global_edge_id,
                           wholememory_array_description_t output_global_edge_id_desc)
{
  int* output_sample_offset_ptr      = static_cast<int*>(host_output_sample_offset);
  DataType* output_dest_nodes_ptr    = static_cast<DataType*>(host_output_dest_nodes);
  int64_t* output_global_edge_id_ptr = static_cast<int64_t*>(host_output_global_edge_id);

  for (int64_t i = 0; i < output_sample_offset_desc.size - 1; i++) {
    int start = output_sample_offset_ptr[i];
    int end   = output_sample_offset_ptr[i + 1];
    std::sort(output_dest_nodes_ptr + start, output_dest_nodes_ptr + end);
    std::sort(output_global_edge_id_ptr + start, output_global_edge_id_ptr + end);
  }
}

void segment_sort_output(void* host_output_sample_offset,
                         wholememory_array_description_t output_sample_offset_desc,
                         void* host_output_dest_nodes,
                         wholememory_array_description_t output_dest_nodes_desc,
                         void* host_output_global_edge_id,
                         wholememory_array_description_t output_global_edge_id_desc)
{
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_global_edge_id_desc.dtype, WHOLEMEMORY_DT_INT64);

  if (output_dest_nodes_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_segment_sort<int>(host_output_sample_offset,
                               output_sample_offset_desc,
                               host_output_dest_nodes,
                               output_dest_nodes_desc,
                               host_output_global_edge_id,
                               output_global_edge_id_desc);
  } else if (output_dest_nodes_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_segment_sort<int64_t>(host_output_sample_offset,
                                   output_sample_offset_desc,
                                   host_output_dest_nodes,
                                   output_dest_nodes_desc,
                                   host_output_global_edge_id,
                                   output_global_edge_id_desc);
  }
}

}  // namespace testing
}  // namespace wholegraph_ops
