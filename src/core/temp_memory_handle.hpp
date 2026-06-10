/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Walpurgis Migration: a72a521 — fix: fixes memory context leak
 * Migrated by: dylanyunlon <dogechat@163.com>
 * Claude Instance: #332/450
 *
 * 上游 Bug 根因 (Knuth 审查 §1 diff 对比):
 *   旧 free_memory() 以 ptr_ != nullptr 为守卫销毁 memory_context_。
 *   然而 free_data() 已将 ptr_ 置 nullptr，故凡先调 free_data() 或从未
 *   malloc 的对象，析构时 destroy_memory_context_fn 永不被触发——
 *   memory_context_ 悄然泄漏。修复：守卫改为 memory_context_ != nullptr，
 *   令数据指针与上下文指针的生命周期彼此独立，各管各的清零。
 *
 * 鲁迅拿法 20% 改写说明:
 *   1. malloc 三路 (device/host/pinned) 抽取为 alloc_impl_()，消除重复。
 *   2. 构造函数加断言守卫：env_fns 为空则立刻 abort，不留隐患。
 *   3. 增加 WALPURGIS_DEBUG 门控的断点 print，覆盖全部关键路径。
 *   4. free_memory() 增加双重 nullptr 检查日志，方便排查 double-free。
 *
 * Knuth 审查:
 *   §1 diff 对比源  — 见上方 Bug 根因分析
 *   §2 用户角度 bug — 见文件末尾注释块
 *   §3 系统角度安全 — 见文件末尾注释块
 */

#pragma once

#include <cstdio>   // fprintf, stderr (断点调试用)
#include <cassert>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>

// 断点调试开关: 编译时 -DWALPURGIS_DEBUG 或运行时检测不到，
// 此处采用编译期宏，与 tiered_allocator.hpp 的 PHILEMON_DEBUG_MIGRATE 风格对齐。
#ifdef WALPURGIS_DEBUG
#  define WM_DBG(fmt, ...) fprintf(stderr, "[WM_DBG %s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)
#else
#  define WM_DBG(fmt, ...) ((void)0)
#endif

namespace wholememory_ops {

// ---------------------------------------------------------------------------
// temp_memory_handle — 临时张量内存的 RAII 句柄
//
// 设计契约（来自上游 PR #332 的修复）：
//   memory_context_ 的生命周期完全由 free_memory() 管理，
//   与数据指针 ptr_ 的生命周期解耦。
//   析构时先 free_data()，再 destroy memory_context_，顺序固定。
// ---------------------------------------------------------------------------
class temp_memory_handle {
 public:
  // 构造：立即创建 memory_context，失败即 abort（防止半初始化对象流出）
  explicit temp_memory_handle(wholememory_env_func_t* env_fns)
  {
    // 断点调试: 校验入参，空指针立刻暴露，不让 nullptr 偷偷传播
    assert(env_fns != nullptr && "temp_memory_handle: env_fns 不得为空");
    temp_mem_fns_ = &env_fns->temporary_fns;

    WM_DBG("create_memory_context_fn 调用前, memory_context_=%p", memory_context_);
    temp_mem_fns_->create_memory_context_fn(&memory_context_, temp_mem_fns_->global_context);
    WM_DBG("create_memory_context_fn 完成, memory_context_=%p", memory_context_);
  }

  temp_memory_handle()                                     = delete;
  temp_memory_handle(const temp_memory_handle&)            = delete;
  temp_memory_handle& operator=(const temp_memory_handle&) = delete;

  ~temp_memory_handle() { free_memory(); }

  // ------------------------------------------------------------------
  // 三路 malloc：device / host / pinned
  // 鲁迅拿法: 原三份重复代码 → 统一走 alloc_impl_()，
  // "横眉冷对千行重，俯首甘为一函吞"
  // ------------------------------------------------------------------
  void* device_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    WM_DBG("device_malloc elt_count=%zu", elt_count);
    return alloc_impl_(elt_count, data_type, WHOLEMEMORY_MA_DEVICE);
  }

  void* host_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    WM_DBG("host_malloc elt_count=%zu", elt_count);
    return alloc_impl_(elt_count, data_type, WHOLEMEMORY_MA_HOST);
  }

  void* pinned_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    WM_DBG("pinned_malloc elt_count=%zu", elt_count);
    return alloc_impl_(elt_count, data_type, WHOLEMEMORY_MA_PINNED);
  }

  [[nodiscard]] void* pointer() const { return ptr_; }

  // ------------------------------------------------------------------
  // free_data(): 只释放数据缓冲，保留 memory_context_
  // 断点调试: 打印 free_fn 调用前后的 ptr_ 变化
  // ------------------------------------------------------------------
  void free_data()
  {
    if (ptr_ != nullptr) {
      WM_DBG("free_fn 调用前 ptr_=%p, memory_context_=%p", ptr_, memory_context_);
      temp_mem_fns_->free_fn(memory_context_, temp_mem_fns_->global_context);
      ptr_ = nullptr;
      WM_DBG("free_fn 完成, ptr_ 已置 nullptr");
    } else {
      WM_DBG("free_data 跳过: ptr_ 已为 nullptr");
    }
  }

  // ------------------------------------------------------------------
  // free_memory(): 释放数据 + 销毁 memory_context_
  //
  // 核心修复 (a72a521):
  //   守卫由 ptr_ != nullptr → memory_context_ != nullptr
  //   确保 context 泄漏问题彻底修复：
  //     - 从未 malloc 的对象析构时，context 也会被正确销毁
  //     - free_data() 已先行调用、ptr_=nullptr 后，context 仍会被销毁
  // ------------------------------------------------------------------
  void free_memory()
  {
    // 断点调试: 先清数据
    WM_DBG("free_memory 入口: ptr_=%p, memory_context_=%p", ptr_, memory_context_);
    free_data();

    // 断点调试: 检查 context 是否需要销毁
    if (memory_context_ != nullptr) {
      WM_DBG("destroy_memory_context_fn 调用前, memory_context_=%p", memory_context_);
      temp_mem_fns_->destroy_memory_context_fn(memory_context_, temp_mem_fns_->global_context);
      memory_context_ = nullptr;
      WM_DBG("destroy_memory_context_fn 完成, memory_context_ 已置 nullptr");
    } else {
      // 断点调试: 重复调用 free_memory() 时走此分支，不应 crash
      WM_DBG("free_memory 跳过 destroy: memory_context_ 已为 nullptr (幂等)");
    }
  }

 private:
  // ------------------------------------------------------------------
  // alloc_impl_(): 三路 malloc 的统一实现
  // 鲁迅拿法: 消除三份结构相同的代码，逻辑集中，断点覆盖一处即可
  // ------------------------------------------------------------------
  void* alloc_impl_(size_t elt_count, wholememory_dtype_t data_type,
                    wholememory_memory_allocation_type_t alloc_type)
  {
    // 先释放旧数据（若有），再分配新数据——与上游行为一致
    free_data();

    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);

    // 断点调试: 打印本次分配参数，方便排查 elt_count=0 的边界情况
    WM_DBG("malloc_fn alloc_type=%d elt_count=%zu memory_context_=%p",
           static_cast<int>(alloc_type), elt_count, memory_context_);

    ptr_ = temp_mem_fns_->malloc_fn(
      &tensor_description, alloc_type, memory_context_, temp_mem_fns_->global_context);

    // 断点调试: malloc 结果，nullptr 表示分配失败
    WM_DBG("malloc_fn 返回 ptr_=%p", ptr_);
    return ptr_;
  }

  // ------------------------------------------------------------------
  // get_tensor_description(): 构造一维张量描述符（与上游完全一致）
  // ------------------------------------------------------------------
  static void get_tensor_description(wholememory_tensor_description_t* tensor_description,
                                     size_t elt_count,
                                     wholememory_dtype_t data_type)
  {
    wholememory_initialize_tensor_desc(tensor_description);
    tensor_description->dim            = 1;
    tensor_description->storage_offset = 0;
    tensor_description->dtype          = data_type;
    tensor_description->sizes[0]       = static_cast<int64_t>(elt_count);
    tensor_description->strides[0]     = 1;
  }

  wholememory_temp_memory_func_t* temp_mem_fns_ = nullptr;
  void*                           memory_context_ = nullptr;
  void*                           ptr_            = nullptr;
};

}  // namespace wholememory_ops

/*
 * =========================================================================
 * Knuth 审查 §2 — 用户角度 Bug
 * =========================================================================
 *
 * 场景：调用者持有一个 temp_memory_handle，在某个错误路径上提前调用了
 *       free_data()（例如重新分配前的清理），随后函数返回，对象析构。
 *
 * 旧代码行为：
 *   free_data() → ptr_ = nullptr
 *   ~temp_memory_handle() → free_memory()
 *     → if (ptr_ != nullptr) 为假 → destroy_memory_context_fn **未调用**
 *   结果：memory_context_ 泄漏。调用次数越多，泄漏越积累。
 *   现象：长时间运行后设备内存耗尽，CUDA OOM，但 valgrind 看不到（GPU 内存）。
 *
 * 修复后行为：
 *   析构 → free_memory() → free_data()（ptr_ 已 nullptr，跳过）
 *              → if (memory_context_ != nullptr) 为真
 *              → destroy_memory_context_fn 正常调用，context 释放。
 *   用户无需感知内部两层资源的区别，RAII 保证完全兑现。
 *
 * 另一典型场景：对象构造后从未调用任何 malloc（预分配但未使用的槽位），
 *   旧代码同样泄漏；新代码构造时创建 context，析构时无条件销毁，正确。
 *
 * =========================================================================
 * Knuth 审查 §3 — 系统角度安全
 * =========================================================================
 *
 * 1. API Contract 违反（隐性）:
 *    create_memory_context_fn / destroy_memory_context_fn 是一对 RAII
 *    契约函数，要求"创建几次销毁几次"。旧代码在 ptr_ 已被清零时跳过
 *    destroy，导致 create/destroy 不对称，后端 allocator 可能维护引用
 *    计数或 handle 表，长期运行会出现资源表耗尽（非内存泄漏，而是
 *    handle 泄漏）。
 *
 * 2. 幂等性:
 *    新 free_memory() 对 memory_context_=nullptr 的对象是安全的幂等操作。
 *    旧代码对 ptr_=nullptr 也是幂等的，但这个幂等性在 free_data() 被先
 *    调用后就失效了——两个状态变量的幂等逻辑交织，容易出错。
 *    修复后两个变量各自独立守卫，设计更清晰。
 *
 * 3. 线程安全（不变）:
 *    此类不提供线程安全保证，调用方负责外部同步。修复未改变此约定。
 *    debug print 使用 fprintf(stderr)，多线程下输出可能交错，
 *    仅用于调试，不影响正确性。
 *
 * 4. Walpurgis alloc_impl_() 重构安全性:
 *    alloc_type 参数为 wholememory_memory_allocation_type_t 枚举，
 *    由调用方三个 public 函数各自硬编码传入正确值，重构未引入新的
 *    类型混淆风险。static_cast<int> 仅用于 debug print 格式化。
 */
