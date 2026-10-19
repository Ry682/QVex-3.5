"""Multi-token prediction (MTP) head of Qwen3.5, used as a built-in draft model.

Given the main model's hidden state at position i and the token at i+1, the
head predicts token i+2:

    x = fc([rms(embed(tok_{i+1})), rms(h_i)])  ->  attention+MLP block  ->  norm  ->  lm_head

It shares embed_tokens and lm_head with the main model; its own attention
layer keeps a KV cache over every confirmed position.
"""

from pathlib import Path

import mlx.core as mx

from .model import Attention, KVCache, Layer, MLP, QLinear, Qwen35


class MTPHead:
    def __init__(self, model: Qwen35, w: dict, group_size: int, bits: int, hidden_mode: str = "pre"):
        cfg = model.cfg
        self.model = model
        self.eps = cfg.rms_norm_eps
        self.hidden_mode = hidden_mode  # "pre": raw last-layer output, "post": after final norm

        def qlin(prefix):
            return QLinear(w[f"{prefix}.weight"], w[f"{prefix}.scales"], w[f"{prefix}.biases"],
                           group_size, bits)

        self.fc = qlin("mtp.fc")
        self.norm_emb = w["mtp.pre_fc_norm_embedding.weight"]
        self.norm_hid = w["mtp.pre_fc_norm_hidden.weight"]
        self.norm_out = w["mtp.norm.weight"]

        layer = Layer.__new__(Layer)  # same block as a main full-attention layer
        layer.linear = False
        layer.eps = cfg.rms_norm_eps
        layer.ln1 = w["mtp.layers.0.input_layernorm.weight"]
        layer.ln2 = w["mtp.layers.0.post_attention_layernorm.weight"]
        sa = "mtp.layers.0.self_attn"
        layer.mixer = Attention(cfg, {
            **{n: qlin(f"{sa}.{n}") for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
            "q_norm": w[f"{sa}.q_norm.weight"], "k_norm": w[f"{sa}.k_norm.weight"],
        })
        layer.mlp = MLP({n: qlin(f"mtp.layers.0.mlp.{n}") for n in ("gate_proj", "up_proj", "down_proj")})
        self.layer = layer
        mx.eval(self._params())

    def _params(self):
        m, l = self.layer.mixer, self.layer
        qs = [self.fc, m.q_proj, m.k_proj, m.v_proj, m.o_proj, l.mlp.gate_up, l.mlp.down]
        return [self.norm_emb, self.norm_hid, self.norm_out, l.ln1, l.ln2, m.q_norm, m.k_norm] + \
               [x for q in qs for x in (q.w, q.scales, q.biases)]

    def make_cache(self) -> KVCache:
        return KVCache()

    def __call__(self, hidden, next_ids, cache: KVCache, last_only=True):
        """hidden [B, T, H] (main model, pre-norm), next_ids [B, T] -> logits [B, 1 or T, V]"""
        if self.hidden_mode == "post":
            hidden = mx.fast.rms_norm(hidden, self.model.norm, self.eps)
        e = mx.fast.rms_norm(self.model.embed(next_ids), self.norm_emb, self.eps)
        h = mx.fast.rms_norm(hidden, self.norm_hid, self.eps)
        x = self.fc(mx.concatenate([e, h], axis=-1))
        x = self.layer(x, cache)
        if last_only:
            x = x[:, -1:, :]
        return self.model.lm_head(mx.fast.rms_norm(x, self.norm_out, self.eps))


def load_mtp(model: Qwen35, path: str | Path, hidden_mode: str = "pre") -> MTPHead:
    w, meta = mx.load(str(path), return_metadata=True)
    return MTPHead(model, w, int(meta.get("group_size", 64)), int(meta.get("bits", 4)), hidden_mode)
