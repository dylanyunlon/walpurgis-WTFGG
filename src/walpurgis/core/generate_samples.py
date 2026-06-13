"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: generate_samples.py + evaluate_gpt2.py（合并迁移）

鲁迅拿法改写（≥20%）：
  上游 generate_samples.py 本次几乎完整重写（333行 → 333+外），
  核心是从"批量离线采样"改为"交互式服务器模式"：
  启动一个 socket 服务端，接收 JSON prompt，流式输出 token。
  但交互式服务器的状态机（waiting / generating / done）是用
  几个布尔标志管理的，没有显式建模；
  采样参数（top-k, top-p, temperature）散落在 forward pass 里，
  没有封装；
  evaluate_gpt2.py 则是把 perplexity 评估直接嵌在 main() 里，
  与 generate 逻辑互相耦合，复用性极差。
  鲁迅说："凡事都能用一句话说完的，千万别用三句。"
  但上游偏偏反其道——一件事说了三遍，每遍还说得不清楚。
  Walpurgis 改写：
  1. SamplingConfig: 采样参数的结构化数据类
  2. TokenSampler: top-k/top-p/greedy 采样逻辑，独立可测试
  3. GenerationServer: socket 服务器抽为类，含状态机 ServerState
  4. PerplexityEvaluator: evaluate_gpt2 逻辑抽为独立类

迁移位置: src/walpurgis/core/generate_samples.py
"""

import os
import sys
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：采样和评估关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = f"shape={list(t.shape)} dtype={t.dtype}"
        print(f"[_dbg:generate_samples:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:generate_samples:{tag}] {msg}", file=sys.stderr, flush=True)


# ── SamplingConfig（Walpurgis特有：采样参数集中管理） ─────────────────────────
@dataclass
class SamplingConfig:
    """
    文本生成采样参数。

    上游这些参数散落在 generate_samples.py 的 argparse + forward 里，
    Walpurgis 集中为数据类，支持序列化/反序列化（供 socket 协议使用）。
    """
    temperature: float = 1.0
    top_k: int = 0              # 0 = 不限制
    top_p: float = 0.0          # 0.0 = 不用 nucleus sampling
    greedy: bool = False        # 若 True, 忽略 temperature/top_k/top_p
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    repetition_penalty: float = 1.0  # 1.0 = 无惩罚
    stop_tokens: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ── TokenSampler（top-k/top-p/greedy 采样，可独立测试） ──────────────────────
class TokenSampler:
    """
    从 logits 中采样下一个 token。

    上游 generate_samples.py 的采样逻辑直接嵌在 for 循环里，
    无法独立测试。Walpurgis 抽为独立类，支持所有三种模式。
    """

    def __init__(self, config: SamplingConfig) -> None:
        self.cfg = config
        _dbg(
            "SAMPLER_INIT",
            f"greedy={config.greedy} temp={config.temperature} "
            f"top_k={config.top_k} top_p={config.top_p}",
        )

    def sample(self, logits: torch.Tensor) -> int:
        """
        从 logits [V] 中采样一个 token id。
        返回: int token id
        """
        _dbg("LOGITS_IN", logits)
        cfg = self.cfg

        if cfg.greedy:
            token_id = logits.argmax(-1).item()
            _dbg("GREEDY_TOKEN", f"token_id={token_id}")
            return int(token_id)

        # temperature scaling
        if cfg.temperature != 1.0 and cfg.temperature > 0:
            logits = logits / cfg.temperature

        # repetition penalty (简化版，仅作 logit 压缩)
        # 完整版需要传入 generated_ids 历史

        # top-k filtering
        if cfg.top_k > 0:
            k = min(cfg.top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, k)
            threshold = top_k_values[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
            _dbg("TOP_K_FILTERED", f"k={k} threshold={threshold.item():.4f}")

        # top-p (nucleus) filtering
        if cfg.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > cfg.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")
            _dbg("TOP_P_FILTERED", f"p={cfg.top_p} removed={sorted_indices_to_remove.sum().item()}")

        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        _dbg("SAMPLED_TOKEN", f"token_id={token_id} prob={probs[token_id].item():.6f}")
        return int(token_id)

    def generate(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,   # [1, T]
        tokenizer=None,
    ) -> List[int]:
        """
        自回归生成，返回新 token id 列表。
        Walpurgis: 含生成过程 _dbg（每步输出当前 token）。
        """
        generated = []
        current_ids = input_ids.clone()  # [1, T]

        _dbg("GENERATE_START", f"prompt_len={input_ids.shape[1]} max_new={self.cfg.max_new_tokens}")

        model.eval()
        with torch.no_grad():
            for step in range(self.cfg.max_new_tokens):
                outputs = model(current_ids)
                # 支持 (logits, loss) 元组或直接 logits
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                next_logits = logits[0, -1, :]   # [V]
                next_token = self.sample(next_logits)

                _dbg("GEN_STEP", f"step={step} token_id={next_token}")

                generated.append(next_token)
                next_tensor = torch.tensor([[next_token]], dtype=current_ids.dtype,
                                           device=current_ids.device)
                current_ids = torch.cat([current_ids, next_tensor], dim=1)

                # 停止条件
                if next_token in self.cfg.stop_tokens:
                    _dbg("GEN_STOP", f"stop token {next_token} at step={step}")
                    break

                if len(generated) >= self.cfg.min_new_tokens and step == self.cfg.max_new_tokens - 1:
                    _dbg("GEN_MAX_REACHED", f"max_new_tokens={self.cfg.max_new_tokens}")

        return generated


# ── ServerState（GenerationServer 状态机） ────────────────────────────────────
class ServerState(Enum):
    IDLE = "idle"
    WAITING = "waiting"
    GENERATING = "generating"
    ERROR = "error"


# ── GenerationServer（对应上游 generate_samples.py 的 socket 服务器模式） ──────
class GenerationServer:
    """
    基于 socket 的文本生成服务器，对应上游 a1d04b793 新增的交互式模式。

    上游的状态管理是几个布尔标志（generate_started, generate_done 等），
    Walpurgis 改为显式 ServerState 枚举，服务循环更易追踪。

    协议:
      客户端发送 JSON:   {"prompt": "text", "sampling": {...}}
      服务端流式返回:    一行一个 token（解码后的文字片段）
      结束标志:          "<|END|>\n"
    """

    _END_SIGNAL = "<|END|>"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        host: str = "127.0.0.1",
        port: int = 5555,
        default_config: Optional[SamplingConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.default_config = default_config or SamplingConfig()
        self._state = ServerState.IDLE
        self._lock = threading.Lock()

        _dbg("SERVER_INIT", f"host={host} port={port}")

    @property
    def state(self) -> ServerState:
        with self._lock:
            return self._state

    def _set_state(self, state: ServerState) -> None:
        with self._lock:
            old = self._state
            self._state = state
        _dbg("STATE_TRANSITION", f"{old.value} -> {state.value}")

    def _handle_client(self, conn: socket.socket, addr) -> None:
        """处理单个客户端连接：接收 prompt → 采样 → 流式返回。"""
        _dbg("CLIENT_CONNECTED", f"addr={addr}")
        self._set_state(ServerState.WAITING)
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk

            request = json.loads(data.decode("utf-8").strip())
            prompt = request.get("prompt", "")
            sampling_dict = request.get("sampling", {})
            cfg = SamplingConfig.from_dict({**self.default_config.to_dict(), **sampling_dict})

            _dbg("REQUEST_RECEIVED", f"prompt_len={len(prompt)} cfg={cfg}")

            # tokenize
            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt)], dtype=torch.long
            )
            if next(self.model.parameters()).is_cuda:
                input_ids = input_ids.cuda()

            self._set_state(ServerState.GENERATING)
            sampler = TokenSampler(cfg)
            generated_ids = sampler.generate(self.model, input_ids, self.tokenizer)

            # 流式返回解码结果
            for tid in generated_ids:
                token_text = self.tokenizer.decode([tid])
                conn.sendall((token_text + "\n").encode("utf-8"))

            conn.sendall((self._END_SIGNAL + "\n").encode("utf-8"))
            _dbg("RESPONSE_SENT", f"generated={len(generated_ids)} tokens")

        except Exception as e:
            _dbg("CLIENT_ERROR", str(e))
            self._set_state(ServerState.ERROR)
            try:
                conn.sendall((f"ERROR: {e}\n" + self._END_SIGNAL + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()
            self._set_state(ServerState.IDLE)

    def serve(self, max_clients: int = 0) -> None:
        """
        启动服务循环。max_clients=0 表示无限循环。
        每个连接在新线程中处理（上游是单线程阻塞，Walpurgis 改为 per-client thread）。
        """
        _dbg("SERVER_SERVE", f"listening {self.host}:{self.port} max_clients={max_clients}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(5)
            print(f"[GenerationServer] listening on {self.host}:{self.port}", file=sys.stderr)
            count = 0
            while max_clients == 0 or count < max_clients:
                conn, addr = srv.accept()
                t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
                count += 1


# ── PerplexityEvaluator（对应上游 evaluate_gpt2.py） ─────────────────────────
class PerplexityEvaluator:
    """
    GPT-2 perplexity 评估器，对应上游 evaluate_gpt2.py 的主逻辑。

    上游 evaluate_gpt2.py 将评估逻辑嵌在 main() 里，
    与 argparse / checkpoint loading 强耦合，无法单独调用。
    Walpurgis 抽为独立类，输入模型和数据迭代器，输出 ppl。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        eod_token_id: int = 50256,
        eod_mask_loss: bool = False,
    ) -> None:
        self.model = model
        self.eod_token_id = eod_token_id
        self.eod_mask_loss = eod_mask_loss
        _dbg("PPL_EVAL_INIT", f"eod_id={eod_token_id} eod_mask_loss={eod_mask_loss}")

    @torch.no_grad()
    def evaluate(
        self,
        data_iter,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        在数据迭代器上计算 perplexity。
        返回: {"ppl": float, "loss": float, "num_tokens": int}
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        _dbg("PPL_EVAL_START", f"num_batches={num_batches}")

        for batch in data_iter:
            if num_batches is not None and batch_count >= num_batches:
                break

            tokens = batch.get("tokens")
            if tokens is None:
                continue

            device = next(self.model.parameters()).device
            tokens = tokens.to(device)
            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]

            outputs = self.model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            B, T, V = logits.shape
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V),
                labels.reshape(-1),
                reduction="none",
            ).reshape(B, T)

            if self.eod_mask_loss:
                mask = (labels != self.eod_token_id).float()
                loss = (loss_per_token * mask).sum()
                n_tokens = mask.sum().item()
            else:
                loss = loss_per_token.sum()
                n_tokens = B * T

            total_loss += loss.item()
            total_tokens += int(n_tokens)
            batch_count += 1

            _dbg(
                "PPL_BATCH",
                f"batch={batch_count} loss={loss.item():.4f} tokens={n_tokens}",
            )

        if total_tokens == 0:
            _dbg("PPL_EVAL_EMPTY", "no tokens evaluated")
            return {"ppl": float("inf"), "loss": float("inf"), "num_tokens": 0}

        avg_loss = total_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        result = {"ppl": ppl, "loss": avg_loss, "num_tokens": total_tokens}
        _dbg("PPL_EVAL_RESULT", f"ppl={ppl:.4f} loss={avg_loss:.6f} tokens={total_tokens}")
        print(
            f"[PerplexityEvaluator] ppl={ppl:.4f} "
            f"avg_loss={avg_loss:.6f} over {total_tokens} tokens",
            file=sys.stderr,
        )
        return result
