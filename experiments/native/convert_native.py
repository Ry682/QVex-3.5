"""Convert the MLX 4-bit Qwen3.5 checkpoint into the native engine's .q35 file.

    .venv/bin/python experiments/native/convert_native.py [--model DIR] [--out models/qwen3.5-9b-native/model.q35]

Layout (little endian):
    "Q35W" u32 version
    config: 16 x i32, 2 x f32
    u32 n_tensors, then n_tensors x {char name[64]; u32 dtype; u32 ndim; u64 shape[4]; u64 offset; u64 nbytes}
    tensor data, each tensor 256-byte aligned, data section 16 KB aligned, file padded to 16 KB
    so the whole file can be mmap'ed and wrapped in one zero-copy MTLBuffer.

dtype: 0 f32, 1 bf16 (raw bits), 2 u32
Projections that share an input are fused here (rows concatenated, lossless):
    linear attention: in_proj_qkv | in_proj_z | in_proj_b | in_proj_a  -> L{i}.in
    full attention:   q_proj | k_proj | v_proj                          -> L{i}.qkv
    MLP:              gate_proj | up_proj                               -> L{i}.gu
"""

import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "models/qwen3.5-9b-native/model.q35"

PAGE = 16384
ALIGN = 256
F32, BF16, U32 = 0, 1, 2


def to_numpy(a: mx.array):
    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16)), BF16
    if a.dtype == mx.uint32:
        return np.array(a), U32
    return np.array(a.astype(mx.float32)), F32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    model_dir, out = Path(args.model), Path(args.out)

    raw = json.loads((model_dir / "config.json").read_text())
    t = raw.get("text_config", raw)
    q = raw.get("quantization", {})
    rope = t.get("rope_parameters", {})
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    assert q.get("bits") == 4 and q.get("group_size") == 64, "native kernels expect 4-bit / group 64"
    cfg_i = [
        t["hidden_size"], t["intermediate_size"], t["num_hidden_layers"], t["num_attention_heads"],
        t["num_key_value_heads"], head_dim, t["linear_num_key_heads"], t["linear_num_value_heads"],
        t["linear_key_head_dim"], t["linear_value_head_dim"], t["linear_conv_kernel_dim"],
        t.get("full_attention_interval", 4), t["vocab_size"],
        int(head_dim * rope.get("partial_rotary_factor", 0.25)), q["group_size"], q["bits"],
    ]
    cfg_f = [t["rms_norm_eps"], float(rope.get("rope_theta", 10_000_000))]

    w = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            if k.startswith("language_model."):
                w[k[len("language_model."):]] = v
    assert not any("mtp." in k for k in w)

    tensors: list[tuple[str, np.ndarray, int]] = []

    def add(name, arr: mx.array, as_f32=False):
        if as_f32:
            arr = arr.astype(mx.float32)
        data, dt = to_numpy(arr)
        tensors.append((name, np.ascontiguousarray(data), dt))

    def add_q(name, prefixes):
        for suffix, short in (("weight", "w"), ("scales", "s"), ("biases", "b")):
            add(f"{name}.{short}", mx.concatenate([w[f"{p}.{suffix}"] for p in prefixes]))

    n_layers, interval = cfg_i[2], cfg_i[11]
    add_q("embed", ["model.embed_tokens"])
    add_q("lm_head", ["lm_head"])
    add("norm", w["model.norm.weight"], True)
    for i in range(n_layers):
        p = f"model.layers.{i}"
        add(f"L{i}.ln1", w[f"{p}.input_layernorm.weight"], True)
        add(f"L{i}.ln2", w[f"{p}.post_attention_layernorm.weight"], True)
        add_q(f"L{i}.gu", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"])
        add_q(f"L{i}.down", [f"{p}.mlp.down_proj"])
        if (i + 1) % interval != 0:
            la = f"{p}.linear_attn"
            add_q(f"L{i}.in", [f"{la}.in_proj_qkv", f"{la}.in_proj_z", f"{la}.in_proj_b", f"{la}.in_proj_a"])
            add_q(f"L{i}.out", [f"{la}.out_proj"])
            add(f"L{i}.conv", w[f"{la}.conv1d.weight"].reshape(-1, cfg_i[10]), True)  # [conv_dim, K]
            add(f"L{i}.A_log", w[f"{la}.A_log"], True)
            add(f"L{i}.dt_bias", w[f"{la}.dt_bias"], True)
            add(f"L{i}.gnorm", w[f"{la}.norm.weight"], True)
        else:
            sa = f"{p}.self_attn"
            add_q(f"L{i}.qkv", [f"{sa}.q_proj", f"{sa}.k_proj", f"{sa}.v_proj"])
            add_q(f"L{i}.o", [f"{sa}.o_proj"])
            add(f"L{i}.qnorm", w[f"{sa}.q_norm.weight"], True)
            add(f"L{i}.knorm", w[f"{sa}.k_norm.weight"], True)

    # ---- layout
    header = b"Q35W" + struct.pack("<I", 1) + struct.pack("<16i", *cfg_i) + struct.pack("<2f", *cfg_f)
    header += struct.pack("<I", len(tensors))
    entry_size = 64 + 4 + 4 + 8 * 4 + 8 + 8
    data_start = (len(header) + entry_size * len(tensors) + PAGE - 1) // PAGE * PAGE
    offset, table = data_start, b""
    placed = []
    for name, data, dt in tensors:
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        shape = list(data.shape) + [0] * (4 - data.ndim)
        assert len(name) < 64
        table += struct.pack("<64sII4QQQ", name.encode(), dt, data.ndim, *shape, offset, data.nbytes)
        placed.append((offset, data))
        offset += data.nbytes
    total = (offset + PAGE - 1) // PAGE * PAGE

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(header + table)
        for off, data in placed:
            f.seek(off)
            f.write(data.tobytes())
        f.truncate(total)
    print(f"wrote {out}: {len(tensors)} tensors, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
