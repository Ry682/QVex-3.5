"""Shared pieces of the quantization pipeline: where weights live, how a layer is assembled
from arbitrary linear objects, and the quality metrics."""

import json
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from engine.model import Config, DenseLinear, KVCache, Layer, RecurrentCache

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BF16_DIR = ROOT / "models/qwen3.5-9b-bf16"          # original Qwen/Qwen3.5-9B checkpoint
MLX_DIR = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"  # config, tokenizer, small tensors
CACHE_DIR = ROOT / "models/ours-cache"               # quantized tensors per bit width, reused by every variant
TOPK = 64


def layer_linears(cfg: Config, i: int) -> list[str]:
    pre = f"model.layers.{i}"
    names = [f"{pre}.mlp.{n}" for n in ("gate_proj", "up_proj", "down_proj")]
    if cfg.is_linear(i):
        names += [f"{pre}.linear_attn.{n}" for n in
                  ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")]
    else:
        names += [f"{pre}.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
    return names


def units(cfg: Config) -> dict[str, list[str]]:
    """Groups of linears that always get the same bit width (fused at load time or sharing an input)."""
    out = {}
    for i in range(cfg.num_hidden_layers):
        pre = f"model.layers.{i}"
        if cfg.is_linear(i):
            out[f"{pre}.mixer_in"] = [f"{pre}.linear_attn.{n}" for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
            out[f"{pre}.mixer_out"] = [f"{pre}.linear_attn.out_proj"]
        else:
            out[f"{pre}.mixer_in"] = [f"{pre}.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj")]
            out[f"{pre}.mixer_out"] = [f"{pre}.self_attn.o_proj"]
        out[f"{pre}.mlp_in"] = [f"{pre}.mlp.gate_proj", f"{pre}.mlp.up_proj"]
        out[f"{pre}.mlp_out"] = [f"{pre}.mlp.down_proj"]
    out["lm_head"] = ["lm_head"]
    return out


class Source:
    """Original bf16 weights (lazy, mmap'ed) + the small unquantized tensors of the MLX checkpoint
    (norms, conv, A_log, dt_bias: already in the layout the engine expects)."""

    def __init__(self):
        self.cfg = Config.from_file(MLX_DIR / "config.json")
        # name -> shard file. Tensors are read on demand and not kept: holding the loaded arrays
        # would pin all 17 GB of bf16 weights in memory as the pipeline walks through the layers.
        self.bf16 = {}
        for f in sorted(BF16_DIR.glob("model.safetensors-*.safetensors")):
            with open(f, "rb") as fh:  # safetensors: u64 header length, then a JSON header
                header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
            self.bf16.update({k: f for k in header if k != "__metadata__"})
        self.small = {}
        for f in sorted(MLX_DIR.glob("*.safetensors")):
            for k, v in mx.load(str(f)).items():
                if k.startswith("language_model.") and not k.endswith((".scales", ".biases")) and v.dtype != mx.uint32:
                    self.small[k[len("language_model."):]] = v

    def weight(self, name: str) -> mx.array:
        """bf16 [out, in] weight of a linear (or the embedding) by engine module name."""
        key = "lm_head.weight" if name == "lm_head" else "model.language_model." + name[len("model."):] + ".weight"
        return mx.load(str(self.bf16[key]))[key]

    def layer(self, i: int, make_linear) -> Layer:
        cfg, s = self.cfg, self.small
        pre = f"model.layers.{i}"
        p = {
            "input_layernorm": s[f"{pre}.input_layernorm.weight"],
            "post_attention_layernorm": s[f"{pre}.post_attention_layernorm.weight"],
            "mlp": {n: make_linear(f"{pre}.mlp.{n}") for n in ("gate_proj", "up_proj", "down_proj")},
        }
        if cfg.is_linear(i):
            la = f"{pre}.linear_attn"
            p["linear_attn"] = {
                **{n: make_linear(f"{la}.{n}") for n in
                   ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")},
                "conv1d": s[f"{la}.conv1d.weight"],
                "A_log": s[f"{la}.A_log"].astype(mx.float32),
                "dt_bias": s[f"{la}.dt_bias"],
                "norm": s[f"{la}.norm.weight"],
            }
        else:
            sa = f"{pre}.self_attn"
            p["self_attn"] = {
                **{n: make_linear(f"{sa}.{n}") for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
                "q_norm": s[f"{sa}.q_norm.weight"],
                "k_norm": s[f"{sa}.k_norm.weight"],
            }
        return Layer(cfg, i, p)

    def dense_layer(self, i: int) -> Layer:
        return self.layer(i, lambda n: DenseLinear(self.weight(n)))


def fresh_cache(cfg: Config, i: int):
    return RecurrentCache() if cfg.is_linear(i) else KVCache()


def log_softmax(logits: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def topk(lp: mx.array, k: int = TOPK):
    idx = mx.argpartition(-lp, k - 1, axis=-1)[..., :k]
    return idx.astype(mx.int32), mx.take_along_axis(lp, idx, axis=-1)


def kl_vs_ref(lp: mx.array, ref_idx: mx.array, ref_lp: mx.array) -> mx.array:
    """Per-token KL(ref || model) using the reference's top-k; the rest of the vocabulary is one bucket."""
    p = mx.exp(ref_lp)
    q_lp = mx.take_along_axis(lp, ref_idx, axis=-1)
    head = (p * (ref_lp - q_lp)).sum(-1)
    p_tail = mx.maximum(1.0 - p.sum(-1), 1e-10)
    q_tail = mx.maximum(1.0 - mx.exp(q_lp).sum(-1), 1e-10)
    return head + p_tail * (mx.log(p_tail) - mx.log(q_tail))


def wait_cool(monitor, hot=85.0, cool=72.0):
    """Long builds run the GPU flat out: pause whenever the fanless chassis gets too hot."""
    if monitor is None or monitor.current_temp() < hot:
        return
    print(f"  [{monitor.current_temp():.0f} °C: pausa per raffreddare]", flush=True)
    while monitor.current_temp() > cool:
        time.sleep(2)


def load_tokens():
    d = np.load(DATA / "tokens.npz")
    return d["calib"], d["eval"], d["calib_src"], d["eval_src"]
