"""
Walpurgis migrate: Megatron-LM 93ab4bea5 — added missing valid-data line (#9)

原 Megatron 脚本将 GPT-2 评估命令拼接成字符串后抛给 subprocess——
如传令兵，收了将令便跑，不问命令是否完整、参数是否遗漏。
93ab4bea5 补上了 PPL 评估分支中遗失的 ``--valid-data`` 参数，
令牌缺位之处，方得补全——鲁迅称之为"遗漏的那一行"：
明明白白的缺口，偏偏无人察觉，直到有人摔了跤才想起来填坑。

鲁迅见此脚本，曰：把命令拼成一根绳子，再用绳子绑住自己，
一旦绳子少了一段，便要跌进数据的深渊——
``--valid-data`` 不在，evaluate_gpt2.py 茫然四顾，不知向谁要数据。

Walpurgis 将此脚本结构化为三层：
  1. EvalConfig   — 封装所有命令行参数与评估模式判断，_dbg() 暴露决策路径
  2. CmdBuilder   — 负责分模式拼装命令字符串，fix 93ab4bea5 在此生效
  3. run_eval()   — 主入口，负责参数解析、命令构建与 subprocess 执行

Fix 93ab4bea5 位置：CmdBuilder.build_ppl_cmd()，
  在 ``CMD = 'evaluate_gpt2.py' + CMD`` 之前插入：
  ``CMD += ' --valid-data {} '.format(args.data_path)``

Usage:
    python src/walpurgis/scripts/run_gpt2_eval.py [args...]
"""

import os
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# _dbg(): 全链路调试断点，WALPURGIS_DEBUG=1 时激活
# ---------------------------------------------------------------------------

def _dbg(tag: str, **kwargs) -> None:
    """Walpurgis debug breakpoint — 鲁迅所谓"在场的见证"。

    只有设置环境变量 WALPURGIS_DEBUG=1 时才输出，
    生产环境静默——如鲁迅笔下的看客，见而不言。
    """
    if os.environ.get("WALPURGIS_DEBUG", "0") == "1":
        parts = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        print(f"[DBG:{tag}] {parts}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# EvalConfig: 封装评估模式与参数的结构化容器
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """封装 GPT-2 评估任务所需的全部参数与模式判断。

    原脚本将模式判断（lambada / webtext / ppl）直接散落在 if/elif/else 链中，
    参数通过 argparse namespace 裸传——如鲁迅所言"一盘散沙"，
    谁是谁的参数，谁决定了哪个命令，读代码的人要从头捋到尾。

    Walpurgis 将模式判断收拢至 EvalConfig，从 argparse 结果构造，
    决策路径由 _dbg() 暴露，不再是黑箱。
    """

    # 数据路径（PPL 评估必需，fix 93ab4bea5 所修补的字段）
    data_path: Optional[str] = None

    # 评估模式标志（三选一）
    lambada_eval: bool = False
    webtext_eval: bool = False
    # ppl_eval 是默认分支，当 lambada_eval 和 webtext_eval 均为 False 时生效

    # 其余透传给 Megatron 的参数（原样拼入命令字符串）
    extra_args: str = ""

    @classmethod
    def from_args(cls, args) -> "EvalConfig":
        """从 argparse Namespace 构造 EvalConfig。"""
        _dbg("EVAL_CONFIG_FROM_ARGS",
             lambada=getattr(args, "lambada_eval", False),
             webtext=getattr(args, "webtext_eval", False),
             data_path=getattr(args, "data_path", None))

        cfg = cls(
            data_path=getattr(args, "data_path", None),
            lambada_eval=getattr(args, "lambada_eval", False),
            webtext_eval=getattr(args, "webtext_eval", False),
        )

        _dbg("EVAL_CONFIG_READY",
             mode=cfg.mode_name,
             data_path=cfg.data_path)
        return cfg

    @property
    def mode_name(self) -> str:
        """返回当前评估模式名称，便于日志与 _dbg() 输出。"""
        if self.lambada_eval:
            return "lambada"
        if self.webtext_eval:
            return "webtext"
        return "ppl"


# ---------------------------------------------------------------------------
# CmdBuilder: 分模式拼装评估命令
# ---------------------------------------------------------------------------

class CmdBuilder:
    """负责将 EvalConfig 转换为可执行的命令字符串。

    原脚本将命令拼接逻辑直接写在 if/elif/else 分支里，
    且 PPL 分支遗漏了 ``--valid-data`` 参数（fix 93ab4bea5 所修补）——
    如流水线工人忘了装某个零件，下游才发现机器转不起来。

    Walpurgis 将三个分支封装为独立方法，fix 明确标注于 build_ppl_cmd()。
    """

    def __init__(self, cfg: EvalConfig, base_cmd: str = "") -> None:
        self.cfg = cfg
        self.base_cmd = base_cmd
        _dbg("CMD_BUILDER_INIT", mode=cfg.mode_name, base_cmd=base_cmd)

    def build(self) -> tuple[str, str]:
        """根据评估模式构建完整命令字符串，返回 (cmd, label)。"""
        _dbg("CMD_BUILDER_BUILD_START", mode=self.cfg.mode_name)

        if self.cfg.lambada_eval:
            cmd, label = self._build_lambada_cmd()
        elif self.cfg.webtext_eval:
            cmd, label = self._build_webtext_cmd()
        else:
            cmd, label = self._build_ppl_cmd()

        _dbg("CMD_BUILDER_BUILD_DONE", label=label, cmd=cmd)
        return cmd, label

    def _build_lambada_cmd(self) -> tuple[str, str]:
        """构建 Lambada 评估命令。"""
        cmd = self.base_cmd
        cmd = "pretrain_gpt2.py" + cmd
        label = "Running Lambada Eval Command:"
        _dbg("CMD_BUILDER_LAMBADA", cmd=cmd)
        return cmd, label

    def _build_webtext_cmd(self) -> tuple[str, str]:
        """构建 Webtext 评估命令。"""
        cmd = self.base_cmd
        cmd = "pretrain_gpt2.py" + cmd
        label = "Running Webtext Eval Command:"
        _dbg("CMD_BUILDER_WEBTEXT", cmd=cmd)
        return cmd, label

    def _build_ppl_cmd(self) -> tuple[str, str]:
        """构建 PPL 评估命令。

        Fix 93ab4bea5: 在拼入 'evaluate_gpt2.py' 前，
        先将 --valid-data 追加进 CMD——
        原脚本遗漏此行，导致 evaluate_gpt2.py 拿不到验证集路径，
        静默失败或报错，如鲁迅所言"明摆着的坑，偏偏无人填"。
        """
        cmd = self.base_cmd

        # [fix 93ab4bea5] --valid-data 必须在 evaluate_gpt2.py 之前注入
        if self.cfg.data_path:
            cmd += " --valid-data {} ".format(self.cfg.data_path)
            _dbg("CMD_BUILDER_PPL_VALID_DATA_INJECTED",
                 data_path=self.cfg.data_path)
        else:
            _dbg("CMD_BUILDER_PPL_VALID_DATA_MISSING",
                 warning="data_path is None; --valid-data will not be set")

        cmd = "evaluate_gpt2.py" + cmd
        label = "Running PPL Eval Command:"
        _dbg("CMD_BUILDER_PPL_DONE", cmd=cmd)
        return cmd, label


# ---------------------------------------------------------------------------
# run_eval(): 主入口
# ---------------------------------------------------------------------------

def _build_base_args(args) -> str:
    """从 argparse Namespace 拼出基础参数字符串（非模式相关部分）。

    原脚本将参数逐一拼接散落在模块顶层，
    如鲁迅所言"零件堆满一桌，却没人知道怎么装"。
    Walpurgis 将此收拢为单一函数，便于 _dbg() 监控。
    """
    parts = []

    # 通用参数映射表：(argparse attr, cmd flag, value_fn)
    arg_map = [
        ("model_parallel_size",  "--model-parallel-size",   str),
        ("num_layers",           "--num-layers",            str),
        ("hidden_size",          "--hidden-size",           str),
        ("num_attention_heads",  "--num-attention-heads",   str),
        ("max_position_embeddings", "--max-position-embeddings", str),
        ("tokenizer_type",       "--tokenizer-type",        str),
        ("vocab_file",           "--vocab-file",            str),
        ("merge_file",           "--merge-file",            str),
        ("load",                 "--load",                  str),
        ("batch_size",           "--batch-size",            str),
        ("seq_length",           "--seq-length",            str),
        ("log_interval",         "--log-interval",          str),
    ]

    for attr, flag, fn in arg_map:
        val = getattr(args, attr, None)
        if val is not None:
            parts.append(f" {flag} {fn(val)} ")
            _dbg("BASE_ARGS_APPEND", flag=flag, val=val)

    # 布尔标志
    bool_flags = [
        ("fp16",          "--fp16"),
        ("no_load_optim", "--no-load-optim"),
    ]
    for attr, flag in bool_flags:
        if getattr(args, attr, False):
            parts.append(f" {flag} ")
            _dbg("BASE_ARGS_BOOL_FLAG", flag=flag)

    result = "".join(parts)
    _dbg("BASE_ARGS_BUILT", result=result)
    return result


def _make_parser():
    """构造 argparse.ArgumentParser，镜像原 Megatron 脚本参数集。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Walpurgis GPT-2 evaluation runner "
                    "(migrated from Megatron-LM 93ab4bea5)"
    )

    # 评估模式（互斥）
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--lambada-eval",  action="store_true",
                      help="Run Lambada evaluation")
    mode.add_argument("--webtext-eval",  action="store_true",
                      help="Run Webtext evaluation")

    # 数据路径（PPL 评估必需，fix 93ab4bea5 所修补）
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to evaluation data (required for PPL eval)")

    # 模型结构
    parser.add_argument("--model-parallel-size",    type=int,  default=None)
    parser.add_argument("--num-layers",              type=int,  default=None)
    parser.add_argument("--hidden-size",             type=int,  default=None)
    parser.add_argument("--num-attention-heads",     type=int,  default=None)
    parser.add_argument("--max-position-embeddings", type=int,  default=None)

    # 分词器
    parser.add_argument("--tokenizer-type",          type=str,  default=None)
    parser.add_argument("--vocab-file",              type=str,  default=None)
    parser.add_argument("--merge-file",              type=str,  default=None)

    # 训练/评估超参
    parser.add_argument("--load",                    type=str,  default=None)
    parser.add_argument("--batch-size",              type=int,  default=None)
    parser.add_argument("--seq-length",              type=int,  default=None)
    parser.add_argument("--log-interval",            type=int,  default=None)
    parser.add_argument("--fp16",                    action="store_true")
    parser.add_argument("--no-load-optim",           action="store_true")

    return parser


def run_eval(argv=None) -> int:
    """主评估入口：解析参数 → 构建命令 → 执行。

    原脚本将所有逻辑平铺于模块顶层，
    如鲁迅所言"平铺直叙，毫无章法，
    读者跟着跑完全程才知道终点在哪"。
    Walpurgis 将其收拢为可测试的函数，返回退出码。
    """
    _dbg("MAIN_ENTRY", argv=argv)

    parser = _make_parser()
    args = parser.parse_args(argv)

    _dbg("MAIN_ARGS_PARSED",
         mode="lambada" if args.lambada_eval else
              "webtext" if args.webtext_eval else "ppl",
         data_path=args.data_path)

    # 构造 EvalConfig
    cfg = EvalConfig.from_args(args)

    # 构造基础参数字符串
    base_cmd = _build_base_args(args)

    # 构建完整命令
    builder = CmdBuilder(cfg, base_cmd)
    cmd, label = builder.build()

    # 打印并执行
    print(label, flush=True)
    print(cmd, flush=True)

    _dbg("MAIN_EXEC_START", cmd=cmd)
    ret = subprocess.call(["python", "-u"] + cmd.split(), env=os.environ)
    _dbg("MAIN_EXEC_DONE", returncode=ret)

    return ret


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_eval())
