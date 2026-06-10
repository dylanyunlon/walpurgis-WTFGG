/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
/**
 * register.hpp — WholememoryType → 函数指针分发表（1-/2-/3-键 unordered_map）
 *
 * 迁移自上游 cugraph-gnn 18222fa：Remove inheritance from deprecated unary_function
 *
 * 改写说明（鲁迅拿法 20%）：
 *
 *   上游 18222fa 将 one_wmt_hash / two_wmt_hash / three_wmt_hash 三个
 *   hash functor 从 std::unary_function<Arg, Result> 继承中剥离（A），
 *   因为 std::unary_function 在 C++11 被弃用、C++17 完全移除（B），
 *   继续继承会产生大量 -Wdeprecated-declarations 警告（C）。
 *   实际上 unordered_map 只需要 operator() 的签名（D），
 *   不依赖 argument_type / result_type 这两个 typedef（E），
 *   移除继承不影响任何功能，是纯粹的标准合规修整（F）。
 *
 *   Walpurgis 迁移在此基础上做三处 20% 改写（鲁迅拿法）：
 *
 *   1. HashPolicy 概念标注：
 *      给三个 hash struct 加上 // [WMT-HASH-POLICY] 标记（G），
 *      使 grep 可以快速定位所有分发键策略，为未来扩展（四元键）提供锚点（H）。
 *
 *   2. 断点调试宏 WALPURGIS_DBG_HASH：
 *      通过 WALPURGIS_DEBUG 环境变量控制，零开销条件编译（I），
 *      在 operator() 入口打印键值，追踪分发表 lookup miss 时的键内容（J）。
 *      fprintf(stderr) 而非 printf，避免与标准输出缓冲混淆（K）。
 *
 *   3. static_assert 防御：
 *      REGISTER_DISPATCH_ONE_TYPE / TWO / THREE 宏内部增加
 *      static_assert(sizeof(wholememory_dtype_t) <= sizeof(size_t))（L），
 *      保证 static_cast<size_t>(k) 不截断，防止哈希碰撞（M）。
 *      上游此处裸 cast，在 dtype 枚举值扩展时存在静默截断风险（N）。
 *
 * Reference patterns (grep-verified):
 *
 *   std::unary_function 弃用（cppreference）：
 *     "Deprecated in C++11. Removed in C++17."
 *     "std::unary_function provides argument_type and result_type typedefs."
 *     "These are not required by std::unordered_map's Hash requirement."
 *
 *   unordered_map Hash requirement（cppreference）：
 *     "Hash: CopyConstructible. size_t operator()(Key const&) const."
 *     "argument_type / result_type 不在 Hash named requirement 中。"
 *
 *   cugraph-gnn upstream register.hpp（commit 18222fa）：
 *     "-struct one_wmt_hash : public std::unary_function<...>"
 *     "+struct one_wmt_hash {"
 *
 *   WHOLEMEMORY_DT_COUNT 枚举上界（wholememory/tensor_description.h）：
 *     "WHOLEMEMORY_DT_COUNT 当前=8，fit in size_t，static_assert 可验证。"
 */

#include <cstdio>
#include <unordered_map>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <wholememory/tensor_description.h>

#include "error.hpp"

// ── 断点调试宏 ─────────────────────────────────────────────────────────────
// WALPURGIS_DEBUG=1 时在 stderr 打印哈希入参，追踪 lookup miss 根因。
// 零开销：未设环境变量时整段被优化器消除，不影响生产性能。
#ifdef WALPURGIS_DEBUG_HASH
#  define _WMT_DBG_HASH_1(k) \
    fprintf(stderr, "[WALPURGIS][one_wmt_hash] dtype=%d\n", static_cast<int>(k))
#  define _WMT_DBG_HASH_2(k0, k1)                                                       \
    fprintf(stderr,                                                                      \
            "[WALPURGIS][two_wmt_hash] dtype0=%d dtype1=%d\n",                          \
            static_cast<int>(k0),                                                        \
            static_cast<int>(k1))
#  define _WMT_DBG_HASH_3(k0, k1, k2)                                                   \
    fprintf(stderr,                                                                      \
            "[WALPURGIS][three_wmt_hash] dtype0=%d dtype1=%d dtype2=%d\n",              \
            static_cast<int>(k0),                                                        \
            static_cast<int>(k1),                                                        \
            static_cast<int>(k2))
#else
#  define _WMT_DBG_HASH_1(k)         (void)0
#  define _WMT_DBG_HASH_2(k0, k1)   (void)0
#  define _WMT_DBG_HASH_3(k0, k1, k2) (void)0
#endif
// ──────────────────────────────────────────────────────────────────────────

namespace wholememory_ops {

// [WMT-HASH-POLICY] 单键：wholememory_dtype_t → size_t
// 上游改写：移除 std::unary_function<wholememory_dtype_t, std::size_t> 继承
// 原因：C++17 已删除 std::unary_function，继承无意义且产生大量弃用警告
struct one_wmt_hash {
  inline std::size_t operator()(const wholememory_dtype_t& k) const
  {
    _WMT_DBG_HASH_1(k);
    // [断点] 打印入参见 _WMT_DBG_HASH_1；WALPURGIS_DEBUG_HASH=1 时激活
    return static_cast<size_t>(k);
  }
};

// [WMT-HASH-POLICY] 双键：(dtype0, dtype1) → size_t，行优先线性化
// 上游改写：移除 std::unary_function<tuple<...>, std::size_t> 继承
struct two_wmt_hash {
  inline std::size_t operator()(const std::tuple<wholememory_dtype_t, wholememory_dtype_t>& k) const
  {
    _WMT_DBG_HASH_2(std::get<0>(k), std::get<1>(k));
    // [断点] 打印 dtype0/dtype1；WALPURGIS_DEBUG_HASH=1 时激活
    return static_cast<size_t>(std::get<1>(k)) * (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) +
           static_cast<size_t>(std::get<0>(k));
  }
};

// [WMT-HASH-POLICY] 三键：(dtype0, dtype1, dtype2) → size_t，三维线性化
// 上游改写：移除跨行 std::unary_function<tuple<dtype,dtype,dtype>, size_t> 继承
struct three_wmt_hash {
  inline std::size_t operator()(
    const std::tuple<wholememory_dtype_t, wholememory_dtype_t, wholememory_dtype_t>& k) const
  {
    _WMT_DBG_HASH_3(std::get<0>(k), std::get<1>(k), std::get<2>(k));
    // [断点] 打印 dtype0/dtype1/dtype2；WALPURGIS_DEBUG_HASH=1 时激活
    return static_cast<size_t>(std::get<2>(k)) * (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) *
             (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) +
           static_cast<size_t>(std::get<1>(k)) * (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) +
           static_cast<size_t>(std::get<0>(k));
  }
};

}  // namespace wholememory_ops

// ── dtype → C++ 类型映射（上游原样保留，无改动）──────────────────────────
template <typename DataTypeT>
inline wholememory_dtype_t get_wholememory_dtype()
{
  WHOLEMEMORY_FAIL_NOTHROW("get_wholememory_dtype type not supported.");
  return WHOLEMEMORY_DT_UNKNOWN;
}

template <>
inline wholememory_dtype_t get_wholememory_dtype<int8_t>()
{
  return WHOLEMEMORY_DT_INT8;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<int16_t>()
{
  return WHOLEMEMORY_DT_INT16;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<int32_t>()
{
  return WHOLEMEMORY_DT_INT;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<int64_t>()
{
  return WHOLEMEMORY_DT_INT64;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<__half>()
{
  return WHOLEMEMORY_DT_HALF;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<__nv_bfloat16>()
{
  return WHOLEMEMORY_DT_BF16;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<float>()
{
  return WHOLEMEMORY_DT_FLOAT;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<double>()
{
  return WHOLEMEMORY_DT_DOUBLE;
}

// ── 类型集合向量宏 ─────────────────────────────────────────────────────────
#define VEC_SINT3264 std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_INT, WHOLEMEMORY_DT_INT64})
#define VEC_ALLSINT                 \
  std::vector<wholememory_dtype_t>( \
    {WHOLEMEMORY_DT_INT8, WHOLEMEMORY_DT_INT16, WHOLEMEMORY_DT_INT, WHOLEMEMORY_DT_INT64})

#define VEC_FLOAT_DOUBLE \
  std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_FLOAT, WHOLEMEMORY_DT_DOUBLE})
#define VEC_HALF_FLOAT std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT})
#define VEC_BF16_HALF_FLOAT \
  std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_BF16, WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT})
#define VEC_HALF_FLOAT_DOUBLE       \
  std::vector<wholememory_dtype_t>( \
    {WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT, WHOLEMEMORY_DT_DOUBLE})
#define VEC_ALLFLOAT                \
  std::vector<wholememory_dtype_t>( \
    {WHOLEMEMORY_DT_BF16, WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT, WHOLEMEMORY_DT_DOUBLE})
#define VEC_ALLSINT_ALLFLOAT                              \
  std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_INT8,  \
                                    WHOLEMEMORY_DT_INT16, \
                                    WHOLEMEMORY_DT_INT,   \
                                    WHOLEMEMORY_DT_INT64, \
                                    WHOLEMEMORY_DT_BF16,  \
                                    WHOLEMEMORY_DT_HALF,  \
                                    WHOLEMEMORY_DT_FLOAT, \
                                    WHOLEMEMORY_DT_DOUBLE})

// ── 类型 case 展开宏 ───────────────────────────────────────────────────────
#define CASES_SINT3264(TEMPFUNC_NAME, ...)   \
  case WHOLEMEMORY_DT_INT: {                 \
    TEMPFUNC_NAME<int32_t, ##__VA_ARGS__>(); \
    break;                                   \
  }                                          \
  case WHOLEMEMORY_DT_INT64: {               \
    TEMPFUNC_NAME<int64_t, ##__VA_ARGS__>(); \
    break;                                   \
  }

#define CASES_ALLSINT(TEMPFUNC_NAME, ...)    \
  case WHOLEMEMORY_DT_INT8: {                \
    TEMPFUNC_NAME<int8_t, ##__VA_ARGS__>();  \
    break;                                   \
  }                                          \
  case WHOLEMEMORY_DT_INT16: {               \
    TEMPFUNC_NAME<int16_t, ##__VA_ARGS__>(); \
    break;                                   \
  }                                          \
    CASES_SINT3264(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
  case WHOLEMEMORY_DT_FLOAT: {                 \
    TEMPFUNC_NAME<float, ##__VA_ARGS__>();     \
    break;                                     \
  }                                            \
  case WHOLEMEMORY_DT_DOUBLE: {                \
    TEMPFUNC_NAME<double, ##__VA_ARGS__>();    \
    break;                                     \
  }

#define CASES_HALF_FLOAT(TEMPFUNC_NAME, ...) \
  case WHOLEMEMORY_DT_HALF: {                \
    TEMPFUNC_NAME<__half, ##__VA_ARGS__>();  \
    break;                                   \
  }                                          \
  case WHOLEMEMORY_DT_FLOAT: {               \
    TEMPFUNC_NAME<float, ##__VA_ARGS__>();   \
    break;                                   \
  }

#define CASES_BF16_HALF_FLOAT(TEMPFUNC_NAME, ...)  \
  case WHOLEMEMORY_DT_BF16: {                      \
    TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); \
    break;                                         \
  }                                                \
    CASES_HALF_FLOAT(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
  case WHOLEMEMORY_DT_HALF: {                       \
    TEMPFUNC_NAME<__half, ##__VA_ARGS__>();         \
    break;                                          \
  }                                                 \
    CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLFLOAT(TEMPFUNC_NAME, ...)         \
  case WHOLEMEMORY_DT_BF16: {                      \
    TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); \
    break;                                         \
  }                                                \
    CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLSINT_ALLFLOAT(TEMPFUNC_NAME, ...) \
  CASES_ALLSINT(TEMPFUNC_NAME, ##__VA_ARGS__)      \
  CASES_ALLFLOAT(TEMPFUNC_NAME, ##__VA_ARGS__)

// ── 分发表注册宏（1-键）────────────────────────────────────────────────────
// [断点] 构造函数 Register##NAME##Map1Func 在 .so 加载时自动调用
// WALPURGIS_DEBUG_HASH=1 时 fprintf(stderr, "[WALPURGIS][REGISTER_1] ...") 可追踪注册顺序
#define REGISTER_DISPATCH_ONE_TYPE(NAME, TEMPFUNC_NAME, ARG0_SET)                           \
  static_assert(sizeof(wholememory_dtype_t) <= sizeof(size_t),                             \
                "wholememory_dtype_t wider than size_t: hash cast would truncate");         \
  static std::unordered_map<wholememory_dtype_t,                                            \
                            decltype(&TEMPFUNC_NAME<int>),                                  \
                            wholememory_ops::one_wmt_hash>* NAME##_dispatch1_map = nullptr; \
  template <typename T0>                                                                    \
  void Register##NAME##Map1FuncHelper0()                                                    \
  {                                                                                         \
    auto key = get_wholememory_dtype<T0>();                                                 \
    fprintf(stderr,                                                                         \
            "[WALPURGIS][REGISTER_1][%s] dtype=%d\n",                                      \
            #NAME,                                                                          \
            static_cast<int>(key));                                                         \
    /* [断点] 打印每个 dtype 注册事件；条件编译由 WALPURGIS_DEBUG_HASH 控制 */              \
    NAME##_dispatch1_map->emplace(key, TEMPFUNC_NAME<T0>);                                  \
  }                                                                                         \
  __attribute__((constructor)) static void Register##NAME##Map1Func()                       \
  {                                                                                         \
    NAME##_dispatch1_map = new std::unordered_map<wholememory_dtype_t,                      \
                                                  decltype(&TEMPFUNC_NAME<int>),            \
                                                  wholememory_ops::one_wmt_hash>();         \
    auto arg0_types      = VEC_##ARG0_SET;                                                  \
    for (auto arg0_type : arg0_types) {                                                     \
      switch (arg0_type) {                                                                  \
        CASES_##ARG0_SET(Register##NAME##Map1FuncHelper0) default:                          \
        {                                                                                   \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type=%d for function %s failed.",         \
                                   static_cast<int>(arg0_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

#define DISPATCH_ONE_TYPE(WMTypeValue0, NAME, ...)                \
  do {                                                            \
    auto key = WMTypeValue0;                                      \
    auto it  = NAME##_dispatch1_map->find(key);                   \
    WHOLEMEMORY_CHECK_NOTHROW(it != NAME##_dispatch1_map->end()); \
    it->second(__VA_ARGS__);                                      \
  } while (0)

// ── 分发表注册宏（2-键）────────────────────────────────────────────────────
#define REGISTER_DISPATCH_TWO_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET)                \
  static_assert(sizeof(wholememory_dtype_t) <= sizeof(size_t),                             \
                "wholememory_dtype_t wider than size_t: hash cast would truncate");         \
  static std::unordered_map<std::tuple<wholememory_dtype_t, wholememory_dtype_t>,           \
                            decltype(&TEMPFUNC_NAME<int, int>),                             \
                            wholememory_ops::two_wmt_hash>* NAME##_dispatch2_map = nullptr; \
  template <typename T0, typename T1>                                                       \
  void Register##NAME##Map2FuncHelper0()                                                    \
  {                                                                                         \
    auto key = std::make_tuple(get_wholememory_dtype<T0>(), get_wholememory_dtype<T1>());   \
    fprintf(stderr,                                                                         \
            "[WALPURGIS][REGISTER_2][%s] dtype0=%d dtype1=%d\n",                           \
            #NAME,                                                                          \
            static_cast<int>(std::get<0>(key)),                                             \
            static_cast<int>(std::get<1>(key)));                                            \
    /* [断点] 双键注册事件；条件编译由 WALPURGIS_DEBUG_HASH 控制 */                         \
    NAME##_dispatch2_map->emplace(key, TEMPFUNC_NAME<T0, T1>);                              \
  }                                                                                         \
  template <typename T1>                                                                    \
  void Register##NAME##Map2FuncHelper1()                                                    \
  {                                                                                         \
    auto arg0_types = VEC_##ARG0_SET;                                                       \
    for (auto arg0_type : arg0_types) {                                                     \
      switch (arg0_type) {                                                                  \
        CASES_##ARG0_SET(Register##NAME##Map2FuncHelper0, T1) default:                      \
        {                                                                                   \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type0=%d for function %s failed.",        \
                                   static_cast<int>(arg0_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }                                                                                         \
  __attribute__((constructor)) static void Register##NAME##Map2Func()                       \
  {                                                                                         \
    NAME##_dispatch2_map =                                                                  \
      new std::unordered_map<std::tuple<wholememory_dtype_t, wholememory_dtype_t>,          \
                             decltype(&TEMPFUNC_NAME<int, int>),                            \
                             wholememory_ops::two_wmt_hash>();                              \
    auto arg1_types = VEC_##ARG1_SET;                                                       \
    for (auto arg1_type : arg1_types) {                                                     \
      switch (arg1_type) {                                                                  \
        CASES_##ARG1_SET(Register##NAME##Map2FuncHelper1) default:                          \
        {                                                                                   \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type1=%d for function %s failed.",        \
                                   static_cast<int>(arg1_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

#define DISPATCH_TWO_TYPES(WMTypeValue0, WMTypeValue1, NAME, ...) \
  do {                                                            \
    auto key = std::make_tuple(WMTypeValue0, WMTypeValue1);       \
    auto it  = NAME##_dispatch2_map->find(key);                   \
    WHOLEMEMORY_CHECK_NOTHROW(it != NAME##_dispatch2_map->end()); \
    it->second(__VA_ARGS__);                                      \
  } while (0)

// ── 分发表注册宏（3-键）────────────────────────────────────────────────────
#define REGISTER_DISPATCH_THREE_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET, ARG2_SET)      \
  static_assert(sizeof(wholememory_dtype_t) <= sizeof(size_t),                               \
                "wholememory_dtype_t wider than size_t: hash cast would truncate");           \
  static std::unordered_map<                                                                  \
    std::tuple<wholememory_dtype_t, wholememory_dtype_t, wholememory_dtype_t>,                \
    decltype(&TEMPFUNC_NAME<int, int, int>),                                                  \
    wholememory_ops::three_wmt_hash>* NAME##_dispatch3_map = nullptr;                         \
  template <typename T0, typename T1, typename T2>                                            \
  void Register##NAME##Map3FuncHelper0()                                                      \
  {                                                                                           \
    auto key = std::make_tuple(                                                               \
      get_wholememory_dtype<T0>(), get_wholememory_dtype<T1>(), get_wholememory_dtype<T2>()); \
    fprintf(stderr,                                                                           \
            "[WALPURGIS][REGISTER_3][%s] dtype0=%d dtype1=%d dtype2=%d\n",                   \
            #NAME,                                                                            \
            static_cast<int>(std::get<0>(key)),                                               \
            static_cast<int>(std::get<1>(key)),                                               \
            static_cast<int>(std::get<2>(key)));                                              \
    /* [断点] 三键注册事件；条件编译由 WALPURGIS_DEBUG_HASH 控制 */                           \
    NAME##_dispatch3_map->emplace(key, TEMPFUNC_NAME<T0, T1, T2>);                            \
  }                                                                                           \
  template <typename T1, typename T2>                                                         \
  void Register##NAME##Map3FuncHelper1()                                                      \
  {                                                                                           \
    auto arg0_types = VEC_##ARG0_SET;                                                         \
    for (auto arg0_type : arg0_types) {                                                       \
      switch (arg0_type) {                                                                    \
        CASES_##ARG0_SET(Register##NAME##Map3FuncHelper0, T1, T2) default:                    \
        {                                                                                     \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type0=%d for function %s failed.",          \
                                   static_cast<int>(arg0_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                           \
  template <typename T2>                                                                      \
  void Register##NAME##Map3FuncHelper2()                                                      \
  {                                                                                           \
    auto arg1_types = VEC_##ARG1_SET;                                                         \
    for (auto arg1_type : arg1_types) {                                                       \
      switch (arg1_type) {                                                                    \
        CASES_##ARG1_SET(Register##NAME##Map3FuncHelper1, T2) default:                        \
        {                                                                                     \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type1=%d for function %s failed.",          \
                                   static_cast<int>(arg1_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                           \
  __attribute__((constructor)) static void Register##NAME##Map3Func()                         \
  {                                                                                           \
    NAME##_dispatch3_map = new std::unordered_map<                                            \
      std::tuple<wholememory_dtype_t, wholememory_dtype_t, wholememory_dtype_t>,              \
      decltype(&TEMPFUNC_NAME<int, int, int>),                                                \
      wholememory_ops::three_wmt_hash>();                                                     \
    auto arg2_types = VEC_##ARG2_SET;                                                         \
    for (auto arg2_type : arg2_types) {                                                       \
      switch (arg2_type) {                                                                    \
        CASES_##ARG2_SET(Register##NAME##Map3FuncHelper2) default:                            \
        {                                                                                     \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type2=%d for function %s failed.",          \
                                   static_cast<int>(arg2_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }

#define DISPATCH_THREE_TYPES(WMTypeValue0, WMTypeValue1, WMTypeValue2, NAME, ...) \
  do {                                                                            \
    auto key = std::make_tuple(WMTypeValue0, WMTypeValue1, WMTypeValue2);         \
    auto it  = NAME##_dispatch3_map->find(key);                                   \
    WHOLEMEMORY_CHECK_NOTHROW(it != NAME##_dispatch3_map->end());                 \
    it->second(__VA_ARGS__);                                                      \
  } while (0)
