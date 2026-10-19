"""Qwen3.5 (dense, text-only) forward pass on raw MLX ops.

Hybrid stack: every 4th layer is gated softmax attention, the other three are
Gated DeltaNet (linear attention with a recurrent state). Weights are MLX affine
4-bit (group 64), used directly by mx.quantized_matmul.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from .kernels import gated_delta


@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    full_attention_interval: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    rotary_dim: int
    group_size: int
    bits: int
    quant: dict = field(default_factory=dict)  # per-module overrides: {"model.layers.3.mlp.down_proj": {"bits": 3, ...}}

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text())
        t = raw.get("text_config", raw)
        rope = t.get("rope_parameters") or {}
        q = raw.get("quantization") or raw.get("quantization_config") or {}
        head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        return cls(
            hidden_size=t["hidden_size"],
            intermediate_size=t["intermediate_size"],
            num_hidden_layers=t["num_hidden_layers"],
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            head_dim=head_dim,
            linear_num_key_heads=t["linear_num_key_heads"],
            linear_num_value_heads=t["linear_num_value_heads"],
            linear_key_head_dim=t["linear_key_head_dim"],
            linear_value_head_dim=t["linear_value_head_dim"],
            linear_conv_kernel_dim=t["linear_conv_kernel_dim"],
            full_attention_interval=t.get("full_attention_interval", 4),
            rms_norm_eps=t["rms_norm_eps"],
            vocab_size=t["vocab_size"],
            rope_theta=rope.get("rope_theta", t.get("rope_theta", 10_000_000)),
            rotary_dim=int(head_dim * rope.get("partial_rotary_factor", 0.25)),
            group_size=q.get("group_size", 64),
            bits=q.get("bits", 4),
            quant={k.removeprefix("language_model."): v for k, v in q.items() if isinstance(v, dict)},
        )

    def qparams(self, module: str) -> tuple[int, int]:
        """(group_size, bits) of one quantized module: mixed-precision models override per module."""
        o = self.quant.get(module, {})
        return o.get("group_size", self.group_size), o.get("bits", self.bits)

    def is_linear(self, i: int) -> bool:
        return (i + 1) % self.full_attention_interval != 0


# ---------------------------------------------------------------- building blocks


class QLinear:
    """Affine-quantized linear layer (y = x @ W^T)."""

    def __init__(self, w, scales, biases, group_size, bits):
        self.w, self.scales, self.biases = w, scales, biases
        self.group_size, self.bits = group_size, bits

    def __call__(self, x):
        return mx.quantized_matmul(
            x, self.w, self.scales, self.biases,
            transpose=True, group_size=self.group_size, bits=self.bits,
        )

    @staticmethod
    def fuse(layers: list["QLinear"]) -> "QLinear":
        """Stack output rows of several layers sharing the same input (lossless)."""
        l0 = layers[0]
        assert all((l.group_size, l.bits) == (l0.group_size, l0.bits) for l in layers), \
            "fused layers must share the quantization (bits, group size)"
        return QLinear(
            mx.concatenate([l.w for l in layers]),
            mx.concatenate([l.scales for l in layers]),
            mx.concatenate([l.biases for l in layers]),
            l0.group_size, l0.bits,
        )


class DenseLinear:
    """Unquantized linear layer, same interface as QLinear (used for bf16 reference runs)."""

    def __init__(self, w):
        self.w = w

    def __call__(self, x):
        return x @ self.w.T

    @staticmethod
    def fuse(layers: list["DenseLinear"]) -> "DenseLinear":
        return DenseLinear(mx.concatenate([l.w for l in layers]))


class QEmbedding:
    def __init__(self, w, scales, biases, group_size, bits):
        self.w, self.scales, self.biases = w, scales, biases
        self.group_size, self.bits = group_size, bits

    def __call__(self, ids):
        return mx.dequantize(
            self.w[ids], self.scales[ids], self.biases[ids],
            group_size=self.group_size, bits=self.bits,
        )


@mx.compile
def _swiglu(gate, up):
    return gate * mx.sigmoid(gate) * up


@mx.compile
def _gated_rmsnorm(x, z, w, eps):
    # rms_norm(x) * silu(z), computed in float32 like the reference
    n = mx.fast.rms_norm(x, w, eps).astype(mx.float32)
    zf = z.astype(mx.float32)
    return (zf * mx.sigmoid(zf) * n).astype(x.dtype)


# ---------------------------------------------------------------- caches


class KVCache:
    step = 256

    def __init__(self):
        self.k = self.v = None
        self.offset = 0

    def update(self, k, v):
        B, H, L, D = k.shape
        prev = self.offset
        if self.k is None or prev + L > self.k.shape[2]:
            n = ((prev + L + self.step - 1) // self.step) * self.step
            nk = mx.zeros((B, H, n, D), k.dtype)
            nv = mx.zeros((B, H, n, v.shape[-1]), v.dtype)
            if self.k is not None:
                nk[..., :prev, :] = self.k[..., :prev, :]
                nv[..., :prev, :] = self.v[..., :prev, :]
            self.k, self.v = nk, nv
        self.k[..., prev:prev + L, :] = k
        self.v[..., prev:prev + L, :] = v
        self.offset += L
        return self.k[..., :self.offset, :], self.v[..., :self.offset, :]

    def state(self):
        return [self.k, self.v] if self.k is not None else []

    def rollback(self, n=1):
        self.offset -= n  # entries past offset are simply overwritten later


class RecurrentCache:
    def __init__(self):
        self.conv = None   # [B, K-1, conv_dim]
        self.ssm = None    # [B, Hv, Dv, Dk] float32
        self.offset = 0
        self.conv_prev = self.ssm_prev = None  # state one token back (speculative decoding)

    def state(self):
        return [x for x in (self.conv, self.ssm) if x is not None]

    def rollback(self, n=1):
        assert n == 1 and self.ssm_prev is not None, "can only roll back the last token"
        self.conv, self.ssm = self.conv_prev, self.ssm_prev
        self.conv_prev = self.ssm_prev = None
        self.offset -= 1


# ---------------------------------------------------------------- layers


class Attention:
    def __init__(self, cfg: Config, p: dict):
        self.cfg = cfg
        self.n_heads, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (
            p["q_proj"], p["k_proj"], p["v_proj"], p["o_proj"]
        )
        self.q_norm, self.k_norm = p["q_norm"], p["k_norm"]

    def __call__(self, x, cache: KVCache):
        B, L, _ = x.shape
        cfg = self.cfg
        qg = self.q_proj(x).reshape(B, L, self.n_heads, 2 * self.head_dim)
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim:]
        k = self.k_proj(x).reshape(B, L, self.n_kv, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.head_dim)

        q = mx.fast.rms_norm(q, self.q_norm, cfg.rms_norm_eps).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(k, self.k_norm, cfg.rms_norm_eps).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        rope = dict(dims=cfg.rotary_dim, traditional=False, base=cfg.rope_theta,
                    scale=1.0, offset=cache.offset)
        q = mx.fast.rope(q, **rope)
        k = mx.fast.rope(k, **rope)
        k, v = cache.update(k, v)

        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal" if L > 1 else None
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o * mx.sigmoid(gate.reshape(B, L, -1)))


class GatedDeltaNet:
    def __init__(self, cfg: Config, p: dict):
        self.cfg = cfg
        self.hk, self.hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.hk * self.dk, self.hv * self.dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.K = cfg.linear_conv_kernel_dim
        # qkv | z | b | a share the input -> one fused quantized matmul
        self.in_proj = type(p["in_proj_qkv"]).fuse(
            [p["in_proj_qkv"], p["in_proj_z"], p["in_proj_b"], p["in_proj_a"]]
        )
        self.splits = [
            self.conv_dim,
            self.conv_dim + self.value_dim,
            self.conv_dim + self.value_dim + self.hv,
        ]
        self.conv_w = p["conv1d"]            # [conv_dim, K, 1]
        self.A_log, self.dt_bias = p["A_log"], p["dt_bias"]
        self.norm_w = p["norm"]
        self.out_proj = p["out_proj"]

    def __call__(self, x, cache: RecurrentCache, keep_prev=False):
        B, S, _ = x.shape
        qkv, z, b, a = mx.split(self.in_proj(x), self.splits, axis=-1)

        conv_state = cache.conv
        if conv_state is None:
            conv_state = mx.zeros((B, self.K - 1, self.conv_dim), x.dtype)
        conv_in = mx.concatenate([conv_state, qkv], axis=1)
        cache.conv = mx.contiguous(conv_in[:, -(self.K - 1):, :])
        if keep_prev:
            cache.conv_prev = conv_in[:, -self.K:-1, :]
        conv_out = mx.conv1d(conv_in, self.conv_w, groups=self.conv_dim)
        conv_out = conv_out * mx.sigmoid(conv_out)

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.hk, self.dk)
        k = k.reshape(B, S, self.hk, self.dk)
        v = v.reshape(B, S, self.hv, self.dv)
        inv = self.dk ** -0.5
        q = (inv * inv) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv * mx.fast.rms_norm(k, None, 1e-6)

        state = cache.ssm
        if state is None:
            state = mx.zeros((B, self.hv, self.dv, self.dk), mx.float32)
        if keep_prev:
            y, cache.ssm, cache.ssm_prev = gated_delta(
                q, k, v, a, b, self.A_log, self.dt_bias, state, keep_mid=True)
        else:
            y, cache.ssm = gated_delta(q, k, v, a, b, self.A_log, self.dt_bias, state)
        cache.offset += S

        y = _gated_rmsnorm(y, z.reshape(B, S, self.hv, self.dv), self.norm_w,
                           self.cfg.rms_norm_eps)
        return self.out_proj(y.reshape(B, S, -1))


class MLP:
    def __init__(self, p: dict):
        self.gate_up = type(p["gate_proj"]).fuse([p["gate_proj"], p["up_proj"]])
        self.down = p["down_proj"]
        self.hidden = p["gate_proj"].w.shape[0]

    def __call__(self, x):
        gu = self.gate_up(x)
        return self.down(_swiglu(gu[..., : self.hidden], gu[..., self.hidden:]))


class Layer:
    def __init__(self, cfg: Config, i: int, p: dict):
        self.linear = cfg.is_linear(i)
        self.mixer = GatedDeltaNet(cfg, p["linear_attn"]) if self.linear else Attention(cfg, p["self_attn"])
        self.mlp = MLP(p["mlp"])
        self.ln1, self.ln2 = p["input_layernorm"], p["post_attention_layernorm"]
        self.eps = cfg.rms_norm_eps

    def __call__(self, x, cache, keep_prev=False):
        n = mx.fast.rms_norm(x, self.ln1, self.eps)
        h = x + (self.mixer(n, cache, keep_prev) if self.linear else self.mixer(n, cache))
        return h + self.mlp(mx.fast.rms_norm(h, self.ln2, self.eps))


class Qwen35:
    def __init__(self, cfg: Config, embed: QEmbedding, layers: list[Layer], norm, lm_head: QLinear):
        self.cfg = cfg
        self.embed, self.layers, self.norm, self.lm_head = embed, layers, norm, lm_head

    def make_cache(self):
        return [RecurrentCache() if l.linear else KVCache() for l in self.layers]

    def __call__(self, ids, cache, last_only=True, return_hidden=False, keep_prev=False):
        """ids: [B, L] int32 -> logits [B, L or 1, vocab] (and pre-norm hidden [B, L, H])"""
        x = self.embed(ids)
        for layer, c in zip(self.layers, cache):
            x = layer(x, c, keep_prev)
        h = x
        if last_only:
            x = x[:, -1:, :]
        logits = self.lm_head(mx.fast.rms_norm(x, self.norm, self.cfg.rms_norm_eps))
        return (logits, h) if return_hidden else logits


# ---------------------------------------------------------------- loading


def load_model(model_dir: str | Path) -> Qwen35:
    model_dir = Path(model_dir)
    cfg = Config.from_file(model_dir / "config.json")

    weights = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            if k.startswith("language_model."):
                weights[k[len("language_model."):]] = v
            elif k.startswith("model.language_model."):
                weights["model." + k[len("model.language_model."):]] = v
    if any("mtp." in k for k in weights):
        raise ValueError("unsanitized checkpoint (mtp weights present): convert with mlx first")

    def qlin(prefix):
        return QLinear(weights.pop(f"{prefix}.weight"), weights.pop(f"{prefix}.scales"),
                       weights.pop(f"{prefix}.biases"), *cfg.qparams(prefix))

    layers = []
    for i in range(cfg.num_hidden_layers):
        pre = f"model.layers.{i}"
        p = {
            "input_layernorm": weights.pop(f"{pre}.input_layernorm.weight"),
            "post_attention_layernorm": weights.pop(f"{pre}.post_attention_layernorm.weight"),
            "mlp": {n: qlin(f"{pre}.mlp.{n}") for n in ("gate_proj", "up_proj", "down_proj")},
        }
        if cfg.is_linear(i):
            la = f"{pre}.linear_attn"
            p["linear_attn"] = {
                **{n: qlin(f"{la}.{n}") for n in
                   ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")},
                "conv1d": weights.pop(f"{la}.conv1d.weight"),
                "A_log": weights.pop(f"{la}.A_log").astype(mx.float32),
                "dt_bias": weights.pop(f"{la}.dt_bias"),
                "norm": weights.pop(f"{la}.norm.weight"),
            }
        else:
            sa = f"{pre}.self_attn"
            p["self_attn"] = {
                **{n: qlin(f"{sa}.{n}") for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
                "q_norm": weights.pop(f"{sa}.q_norm.weight"),
                "k_norm": weights.pop(f"{sa}.k_norm.weight"),
            }
        layer = Layer(cfg, i, p)
        mx.eval(_layer_params(layer))
        layers.append(layer)

    e = "model.embed_tokens"
    embed = QEmbedding(weights.pop(f"{e}.weight"), weights.pop(f"{e}.scales"), weights.pop(f"{e}.biases"),
                       *cfg.qparams(e))
    model = Qwen35(cfg, embed, layers, weights.pop("model.norm.weight"), qlin("lm_head"))

    mx.eval([embed.w, embed.scales, embed.biases, model.norm,
             model.lm_head.w, model.lm_head.scales, model.lm_head.biases])
    del weights  # vision tower etc. are never materialized
    return model


def _layer_params(l: Layer) -> list:
    out = [l.ln1, l.ln2]
    objs = [l.mlp.gate_up, l.mlp.down]
    m = l.mixer
    if l.linear:
        objs += [m.in_proj, m.out_proj]
        out += [m.conv_w, m.A_log, m.dt_bias, m.norm_w]
    else:
        objs += [m.q_proj, m.k_proj, m.v_proj, m.o_proj]
        out += [m.q_norm, m.k_norm]
    for q in objs:
        out += [q.w, q.scales, q.biases]
    return out
