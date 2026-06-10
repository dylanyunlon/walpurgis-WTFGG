# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 03292cf: Migrate cugraph gnn packages to cugraph-pyg
# Walpurgis 迁移: sampler.io 子包 — 采样 I/O 工具

from .reader import BufferedSampleReader

__all__ = ["BufferedSampleReader"]
