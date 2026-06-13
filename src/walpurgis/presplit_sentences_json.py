"""
Walpurgis migrate: Megatron-LM d08783338 — added missing presplit_sentences_json.py

原 Megatron 脚本将松散 JSON 语料逐行读入，
用 NLTK 按句切分后重写 'text' 字段，
再逐行写出——如工厂流水线，进料、切割、出料，
一气呵成，不留痕迹，不问语料何来、句意何归。

鲁迅见之，曰：字字皆刀，刀刀割断上下文；
所谓"句子"，不过是流水线上的零件，
装箱出厂，谁也不认识谁了。

Walpurgis 将此流程结构化为三层：
  1. DocRecord   — 封装单条语料的原文、切句结果及诊断信息
  2. SentSplitter — 包装 NLTK，暴露 split_paragraph() 接口，带 _dbg() 断点
  3. presplit_file() — 主处理函数，负责逐行 IO 与异常上报

Usage:
    python src/walpurgis/presplit_sentences_json.py <input.jsonl> <output.jsonl>
"""

import sys
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import nltk

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
# NLTK 资源初始化
# ---------------------------------------------------------------------------

def _ensure_punkt() -> None:
    """确保 NLTK punkt 分词资源已下载。

    原脚本在模块顶层直接调用 nltk.download('punkt')，
    每次 import 都触发网络请求——鲁迅称之为"逢人便磕头，
    头破了也不知道为什么磕"。
    Walpurgis 改为惰性检查，已有则跳过。
    """
    _dbg("PUNKT_INIT", action="check")
    try:
        nltk.data.find("tokenizers/punkt")
        _dbg("PUNKT_INIT", status="already_present")
    except LookupError:
        _dbg("PUNKT_INIT", status="downloading")
        nltk.download("punkt")
        _dbg("PUNKT_INIT", status="done")


# ---------------------------------------------------------------------------
# DocRecord: 单条语料的结构化容器
# ---------------------------------------------------------------------------

@dataclass
class DocRecord:
    """封装一条 JSON 语料行的解析结果与切句输出。

    原脚本直接在循环体内 in-place 修改 parsed dict，
    没有中间状态，出错时无从追溯——如案卷被焚，
    只剩灰烬，不知原文是何模样。
    """

    raw_json: str                        # 原始 JSON 字符串
    parsed: dict = field(default_factory=dict)   # 解析后的 dict
    sentences: List[str] = field(default_factory=list)  # 切分后的句子列表
    error: Optional[str] = None          # 解析或切句异常信息
    line_no: int = 0                     # 行号，便于错误定位

    @classmethod
    def from_line(cls, raw: str, line_no: int = 0) -> "DocRecord":
        """从原始 JSON 行构造 DocRecord。"""
        rec = cls(raw_json=raw.strip(), line_no=line_no)
        _dbg("DOC_RECORD_PARSE", line_no=line_no, raw_len=len(raw))
        try:
            rec.parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            rec.error = f"JSONDecodeError at line {line_no}: {exc}"
            _dbg("DOC_RECORD_PARSE_ERR", line_no=line_no, error=rec.error)
        return rec

    def to_json_line(self) -> str:
        """将处理后的 parsed dict 序列化为 JSON 行（含换行符）。"""
        out = json.dumps(self.parsed) + "\n"
        _dbg("DOC_RECORD_SERIALIZE", line_no=self.line_no, out_len=len(out))
        return out


# ---------------------------------------------------------------------------
# SentSplitter: NLTK 句子切分器
# ---------------------------------------------------------------------------

class SentSplitter:
    """包装 NLTK sent_tokenize，提供段落级切分接口。

    原脚本将切分逻辑散落在循环体内，
    段落遍历、空行过滤、句列展开三者混写——
    如鲁迅所言："一锅乱炖，自己也不清楚炖的是什么。"
    Walpurgis 将其收拢为单一职责类。
    """

    LINE_SEP: str = "\n"  # 原脚本 line_seperator（原文拼写保留，含错）

    def __init__(self) -> None:
        _ensure_punkt()
        _dbg("SENT_SPLITTER_INIT", sep=repr(self.LINE_SEP))

    def split_paragraph(self, text: str) -> List[str]:
        """将多段文本按段落→句子两层切分，返回句子列表。

        - 空段落跳过（原脚本：`if line != '\\n'`，此处统一为 strip 判空）
        - 每段内调用 nltk.tokenize.sent_tokenize
        """
        _dbg("SPLIT_PARA_IN", text_len=len(text))
        sent_list: List[str] = []
        for para in text.split("\n"):
            if not para.strip():          # 跳过空行/纯空白行
                continue
            sents = nltk.tokenize.sent_tokenize(para)
            _dbg("SPLIT_PARA_PARA", para_len=len(para), n_sents=len(sents))
            sent_list.extend(sents)
        _dbg("SPLIT_PARA_OUT", total_sents=len(sent_list))
        return sent_list

    def join(self, sentences: List[str]) -> str:
        """将句子列表用 LINE_SEP 连接，对应原脚本 line_seperator.join(sent_list)。"""
        return self.LINE_SEP.join(sentences)


# ---------------------------------------------------------------------------
# presplit_file(): 主处理函数
# ---------------------------------------------------------------------------

def presplit_file(input_path: str, output_path: str) -> None:
    """逐行读取松散 JSONL，切句后写出。

    原脚本双层嵌套 with open，内外各一——
    鲁迅见之，曰：两扇门，进了外门，还有内门，
    出了内门，还要出外门，繁文缛节，毫无必要。
    Walpurgis 改为单层 with 同时打开两文件。
    """
    _dbg("PRESPLIT_FILE_START", input=input_path, output=output_path)

    splitter = SentSplitter()
    _dbg("PRESPLIT_FILE_SPLITTER_READY")

    ok_count = 0
    err_count = 0

    with open(input_path, "r", encoding="utf-8") as ifile, \
         open(output_path, "w", encoding="utf-8") as ofile:

        for line_no, raw_line in enumerate(ifile, start=1):
            rec = DocRecord.from_line(raw_line, line_no=line_no)

            if rec.error:
                # 记录错误但继续处理，不因一行坏料中断全局——
                # 鲁迅曰："沉默不是金，是铁，压在心口。"
                # Walpurgis 选择发声而非静默。
                print(f"[WARN] {rec.error}", file=sys.stderr)
                err_count += 1
                continue

            original_text = rec.parsed.get("text", "")
            _dbg("PRESPLIT_FILE_LINE", line_no=line_no, text_len=len(original_text))

            rec.sentences = splitter.split_paragraph(original_text)
            rec.parsed["text"] = splitter.join(rec.sentences)

            ofile.write(rec.to_json_line())
            ok_count += 1

    _dbg(
        "PRESPLIT_FILE_DONE",
        ok=ok_count,
        errors=err_count,
        input=input_path,
        output=output_path,
    )
    print(
        f"[presplit] done: {ok_count} docs written, {err_count} errors. "
        f"output → {output_path}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """命令行入口，兼容原 Megatron 脚本调用方式。

    Usage:
        python src/walpurgis/presplit_sentences_json.py <input.jsonl> <output.jsonl>
    """
    _dbg("MAIN_ENTRY", argv=sys.argv)
    if len(sys.argv) != 3:
        print(
            "Usage: python presplit_sentences_json.py <input.jsonl> <output.jsonl>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    _dbg("MAIN_ARGS", input=input_file, output=output_file)

    presplit_file(input_file, output_file)


if __name__ == "__main__":
    main()
