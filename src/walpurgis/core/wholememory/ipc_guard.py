"""
ipc_guard.py — migrate 3626464: C++ buffer overflow 修复的 Python 防护层

上游来源: cpp/src/wholememory/communicator.cpp + memory_handle.cpp
commit: 36264648f3247a2a09f14364193e34618a862380
forward-merge: release/25.12 → main (PR #358, Jake Awe, 2025-12-10)
核心 bug 修复来自 PR #367 (linhu-nv): fix some potential buffer overflow problems

上游 C++ 修复摘要 (PR #367):
  1. communicator.cpp get_host_name():
     - 旧: strncpy(hostname, "unknown", maxlen); WHOLEMEMORY_FATAL(...)
           先写 hostname 再 fatal，逻辑多余；fatal 后代码不可达
     - 新: 直接 WHOLEMEMORY_FATAL，去掉死代码写入

  2. communicator.cpp get_boot_id():
     - 旧: strncpy(host_id, env_host_id, len-1) 未正确限制 copy 长度
           strncpy(host_id+offset, p, len-offset-1) 同样未用 min 保护
           fclose(file) 在 if (file != nullptr) 外，可能 NULL 解引用
           #undef HOSTID_FILE 宏名写错（应为 BOOTID_FILE）
     - 新: size_t copy_len = std::min(strlen(src), max_len); memcpy(...)
           fclose(file) 移入 if (file != nullptr) 块内
           #undef BOOTID_FILE (正确宏名)

  3. memory_handle.cpp continuous_device_wholememory_impl::exchange_handle():
     - 旧: strcpy(cliaddr.sun_path, dst_name.c_str()) — 无长度检查，
           若 dst_name.length() >= sizeof(sockaddr_un.sun_path)=108 会溢出
     - 新: if (dst_name.length() >= sizeof(cliaddr.sun_path)) WHOLEMEMORY_FATAL(...)
           guard 先于 strcpy，保证不溢出

迁移策略:
  C++ 层修复无法直接在 Python 中重现（sockaddr_un 由 C 库管理），
  但 Python 端所有涉及 IPC socket 路径、hostname、boot_id 的构造逻辑
  均可在传递前做防护性检查，让错误在 Python 层早于 C++ fatal 被发现。

  Walpurgis 鲁迅拿法 20% 改写（相对上游 C++ 补丁）:
  1. HostnameGuard dataclass:
     - 上游: get_host_name() 直接写 char[maxlen]，无 Python 层感知
     - 本版: Python 级 validate_hostname(hostname, maxlen) 提前检查 + WALPURGIS_DEBUG 断点
  2. BootIdGuard dataclass:
     - 上游: get_boot_id() 内联 min 保护；无路径抽象
     - 本版: BootIdBuilder，将「env → file → truncate → sentinel」四段逻辑
             封装为可测试的有状态构建器，步骤 API 化
  3. IpcPathGuard dataclass:
     - 上游: memory_handle.cpp 加了 if (len>=108) fatal
     - 本版: Python 层 IpcPathGuard 在调用 WM 初始化前做等价检查，
             UNIX_SUN_PATH_LEN=108 常量显式记录 sockaddr_un 约束
  4. WALPURGIS_DEBUG 断点:
     - 断点1: validate_hostname 入口 (检查 maxlen 约束)
     - 断点2: BootIdBuilder.build() 路径选择 (env / file / empty)
     - 断点3: IpcPathGuard.validate() 长度边界 (>= 108 提前 raise)
  5. 类型注解 + docstring:
     - 上游 C++ 无文档；本版补充各守卫类的 Invariant 说明

鲁迅: 「我向来是不惮以最坏的恶意来推测人的——
        但尚未确知真相之前，不妨存而不论。」
应用: 对 dst_name / hostname / boot_id 的来源不做信任假设，
      在传入 C 层前一律经过守卫验证。
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────────────────────────────────────

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点打印: WALPURGIS_DEBUG=1 时输出到 stderr。"""
    if _DEBUG:
        print(
            f"[WALPURGIS wholememory/ipc_guard|{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 常量: UNIX domain socket 路径长度上限
#
# 上游 memory_handle.cpp 修复:
#   if (dst_name.length() >= sizeof(cliaddr.sun_path)) WHOLEMEMORY_FATAL(...)
#   sizeof(sockaddr_un.sun_path) == 108 (Linux 上 UNIX_PATH_MAX)
#
# 断点3 所用边界值。
# ──────────────────────────────────────────────────────────────────────────────

UNIX_SUN_PATH_LEN: int = 108  # sockaddr_un.sun_path 的字节长度 (含 NUL)
HOSTNAME_MAXLEN: int = 256     # gethostname 通常支持的最大长度 (HOST_NAME_MAX+1)
BOOTID_MAXLEN: int = 256       # /proc/sys/kernel/random/boot_id 典型长度


# ──────────────────────────────────────────────────────────────────────────────
# HostnameGuard
#
# 上游 communicator.cpp get_host_name():
#   char hostname[MAXLEN]; gethostname(hostname, MAXLEN);
#   → 若 gethostname 失败则 WHOLEMEMORY_FATAL
#
# Python 防护: 在创建 WM communicator 前提前 validate hostname 获取能力，
#               并检查返回值是否在 maxlen 内。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HostnameGuard:
    """
    Python 层 hostname 获取守卫。

    Invariant: hostname 字节长度（含 NUL）< maxlen

    migrate 3626464 对应:
        communicator.cpp get_host_name() 的 fatal 路径前置到 Python 层,
        在调用 WM communicator 构造之前暴露 hostname 获取失败。

    断点1: validate 入口打印 hostname + 长度
    """
    maxlen: int = HOSTNAME_MAXLEN

    def get_hostname(self) -> str:
        """
        获取当前主机名，并验证其长度满足 C 层约束。

        上游等价:
            char hostname[maxlen];
            if (gethostname(hostname, maxlen) != 0) { WHOLEMEMORY_FATAL(...); }
        """
        # 断点1: hostname validate 入口
        _dbg("hostname", f"获取 hostname (maxlen={self.maxlen})")

        try:
            hostname = socket.gethostname()
        except OSError as exc:
            # 上游: WHOLEMEMORY_FATAL("gethostname failed.")
            # Python 层: 提前以 RuntimeError 通报，附带 errno 信息
            _dbg("hostname", f"gethostname 失败: {exc}")
            raise RuntimeError(
                f"[ipc_guard] gethostname() 失败: {exc}。"
                "WholeMemory communicator 无法确定本机标识。"
            ) from exc

        # 验证长度约束 (含 NUL 字节)
        encoded_len = len(hostname.encode("utf-8")) + 1  # +1 for NUL
        _dbg("hostname", f"hostname={hostname!r} encoded_len={encoded_len} maxlen={self.maxlen}")

        if encoded_len > self.maxlen:
            raise ValueError(
                f"[ipc_guard] hostname 长度 {encoded_len} 超过 maxlen={self.maxlen}。"
                f"hostname={hostname!r}。"
                "请检查系统 hostname 配置或增大 maxlen。"
            )

        _dbg("hostname", f"✓ hostname 验证通过: {hostname!r}")
        return hostname

    def get_delimited(self, delim: str = ".") -> str:
        """
        获取 hostname 并截断至第一个 delim 字符（与 C++ get_host_name 逻辑等价）。

        上游等价:
            int i = 0;
            while ((hostname[i] != delim) && (hostname[i] != '\\0') && (i < maxlen-1)) i++;
            hostname[i] = '\\0';
        """
        hostname = self.get_hostname()
        idx = hostname.find(delim)
        if idx >= 0:
            truncated = hostname[:idx]
            _dbg("hostname", f"delimited: {hostname!r} → {truncated!r} (delim={delim!r})")
            return truncated
        return hostname


# ──────────────────────────────────────────────────────────────────────────────
# BootIdBuilder
#
# 上游 communicator.cpp get_boot_id():
#   1. 若 WHOLEMEMORY_BOOTID 环境变量存在: strncpy(host_id, env_val, len-1)
#   2. 否则: fopen(BOOTID_FILE, "r"); fscanf(file, "%ms", &p); strncpy(...)
#   3. host_id[offset] = '\0'
#
# PR #367 修复:
#   - strncpy → memcpy(copy_len=min(src_len, remaining)) 防止 overflow
#   - fclose 移入 if (file != nullptr) 块
#   - #undef HOSTID_FILE → #undef BOOTID_FILE（宏名拼写纠正）
#
# Python 防护: 封装四段逻辑，每步做 min 截断，fclose 等价 → with 语句。
# ──────────────────────────────────────────────────────────────────────────────

_BOOTID_FILE = "/proc/sys/kernel/random/boot_id"


@dataclass
class BootIdBuilder:
    """
    Python 层 boot_id 构建器。

    Invariant:
        build() 返回的字符串长度 <= maxlen-1（不含 NUL）。
        fclose 等价保证: 文件句柄通过 with 语句管理，不会泄漏。

    migrate 3626464 对应:
        communicator.cpp get_boot_id() 的四段逻辑 Python 化，
        每个截断点显式使用 min 保护（等价 PR #367 的 memcpy(min) 修复）。

    断点2: build() 路径选择打印
    """
    maxlen: int = BOOTID_MAXLEN
    bootid_file: str = _BOOTID_FILE
    env_var: str = "WHOLEMEMORY_BOOTID"

    def build(self) -> str:
        """
        构建 boot_id 字符串。

        优先顺序:
          1. WHOLEMEMORY_BOOTID 环境变量
          2. /proc/sys/kernel/random/boot_id（Linux）
          3. 空字符串（fallback，非 Linux 平台）

        断点2: 打印路径选择
        """
        max_chars = self.maxlen - 1  # 预留 NUL

        # 路径 1: 环境变量
        env_val = os.environ.get(self.env_var)
        if env_val is not None:
            _dbg("bootid", f"路径1: 环境变量 {self.env_var}={env_val!r}")
            # 等价 C++: copy_len = min(strlen(env_val), len-1); memcpy(...)
            result = env_val[:max_chars]
            _dbg("bootid", f"env boot_id (截断到 {max_chars}): {result!r}")
            return result

        # 路径 2: 文件读取 (with 语句等价 fclose 防泄漏)
        try:
            with open(self.bootid_file, "r") as f:
                _dbg("bootid", f"路径2: 读取 {self.bootid_file}")
                raw = f.read().strip()
                # 等价 C++: copy_len = min(strlen(p), len-offset-1); memcpy(...)
                result = raw[:max_chars]
                _dbg("bootid", f"file boot_id (截断到 {max_chars}): {result!r}")
                return result
        except OSError:
            # 路径 3: fallback (非 Linux 或权限不足)
            _dbg("bootid", "路径3: 文件不可读，返回空字符串")
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# IpcPathGuard
#
# 上游 memory_handle.cpp exchange_handle():
#   strcpy(cliaddr.sun_path, dst_name.c_str());
#   → PR #367 在 strcpy 前加:
#       if (dst_name.length() >= sizeof(cliaddr.sun_path)) {
#           WHOLEMEMORY_FATAL("IPC socket path length (%zu) larger than ...", ...);
#       }
#
# Python 防护: 在将 IPC 路径传入 WM tensor 创建前做等价长度检查。
# UNIX_SUN_PATH_LEN=108 是 Linux 上 sockaddr_un.sun_path 的硬上限。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IpcPathGuard:
    """
    IPC socket 路径长度守卫。

    Invariant:
        len(path) < UNIX_SUN_PATH_LEN (108)
        否则底层 strcpy(cliaddr.sun_path, ...) 会发生 buffer overflow。

    migrate 3626464 对应:
        memory_handle.cpp exchange_handle() 中 PR #367 新增的长度检查
        前置到 Python 层，在路径传入 WM 之前提前发现问题。

    断点3: validate 入口打印路径长度 vs 上限
    """
    sun_path_len: int = UNIX_SUN_PATH_LEN  # 含 NUL: 108

    def validate(self, ipc_path: str) -> None:
        """
        验证 IPC socket 路径满足 sockaddr_un.sun_path 长度约束。

        上游等价:
            if (dst_name.length() >= sizeof(cliaddr.sun_path)) {
                WHOLEMEMORY_FATAL("IPC socket path length (%zu) larger than "
                                  "sockaddr_un.sun_path length (%lu), full_path: %s",
                                  dst_name.length(), sizeof(cliaddr.sun_path),
                                  dst_name.c_str());
            }

        断点3: 打印路径长度 vs 上限 (108)
        """
        # 断点3: IPC 路径长度边界检查
        path_len = len(ipc_path.encode("utf-8"))  # 不含 NUL
        _dbg(
            "ipc_path",
            f"validate: path={ipc_path!r} path_len={path_len} "
            f"sun_path_len={self.sun_path_len} (含NUL上限={self.sun_path_len})"
        )

        # 上游条件: dst_name.length() >= sizeof(cliaddr.sun_path)
        # dst_name.length() 是不含 NUL 的字节数;
        # sizeof(sun_path)=108 包含 NUL，所以有效路径最长 107 字节
        max_path_bytes = self.sun_path_len - 1  # 107

        if path_len >= self.sun_path_len:
            raise ValueError(
                f"[ipc_guard] IPC socket 路径长度 ({path_len}) >= "
                f"sockaddr_un.sun_path 上限 ({self.sun_path_len})。"
                f"\n路径: {ipc_path!r}"
                f"\n上游 C++ 等价: WHOLEMEMORY_FATAL(\"IPC socket path length (%zu) "
                f"larger than sockaddr_un.sun_path length (%lu), full_path: %s\", "
                f"{path_len}, {self.sun_path_len}, \"{ipc_path}\")"
                f"\n提示: 请缩短 WholeMemory IPC 临时目录路径，"
                f"确保路径字节数 < {self.sun_path_len}。"
                f"常见原因: 容器内长临时目录 + 长作业 ID。"
            )

        _dbg("ipc_path", f"✓ 路径长度 {path_len} < {self.sun_path_len}，通过")

    def safe_path(self, ipc_path: str) -> str:
        """
        验证并返回通过验证的路径，便于链式调用。

        用法:
            safe = IpcPathGuard().safe_path(my_path)
            wm.create_tensor(..., ipc_path=safe)
        """
        self.validate(ipc_path)
        return ipc_path


# ──────────────────────────────────────────────────────────────────────────────
# 模块级便捷实例（单例，避免重复构造）
# ──────────────────────────────────────────────────────────────────────────────

_hostname_guard: HostnameGuard = HostnameGuard()
_boot_id_builder: BootIdBuilder = BootIdBuilder()
_ipc_path_guard: IpcPathGuard = IpcPathGuard()


def get_safe_hostname(delim: str = ".") -> str:
    """便捷函数: 获取并验证 hostname（截断至第一个 delim）。"""
    return _hostname_guard.get_delimited(delim)


def build_boot_id() -> str:
    """便捷函数: 构建 boot_id（自动选 env / file / empty 路径）。"""
    return _boot_id_builder.build()


def validate_ipc_path(path: str) -> str:
    """便捷函数: 验证并返回合法的 IPC socket 路径。"""
    return _ipc_path_guard.safe_path(path)


# ──────────────────────────────────────────────────────────────────────────────
# 自测 (WALPURGIS_DEBUG=1 时运行)
# ──────────────────────────────────────────────────────────────────────────────

if _DEBUG:
    _dbg("selftest", "=== ipc_guard 自测开始 ===")

    # 断点1: hostname
    try:
        hn = get_safe_hostname()
        _dbg("selftest", f"✓ hostname: {hn!r}")
    except Exception as exc:
        _dbg("selftest", f"✗ hostname 异常: {exc}")

    # 断点2: boot_id
    bid = build_boot_id()
    _dbg("selftest", f"✓ boot_id: {bid!r}")

    # 断点3: IPC 路径边界 — 正常路径
    try:
        ok_path = "/tmp/wm_ipc_test"
        validate_ipc_path(ok_path)
        _dbg("selftest", f"✓ IPC path OK: {ok_path!r}")
    except ValueError as exc:
        _dbg("selftest", f"✗ IPC path 意外失败: {exc}")

    # 断点3: IPC 路径边界 — 超长路径（预期 ValueError）
    try:
        long_path = "/tmp/" + "a" * 110
        validate_ipc_path(long_path)
        _dbg("selftest", "✗ 超长路径应抛 ValueError 但未抛！")
    except ValueError:
        _dbg("selftest", "✓ 超长路径正确抛出 ValueError")

    _dbg("selftest", "=== ipc_guard 自测完成 ===")
