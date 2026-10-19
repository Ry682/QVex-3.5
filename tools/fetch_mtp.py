"""Download only the MTP tensors of Qwen/Qwen3.5-9B via HTTP Range requests,
then quantize them to the same MLX affine format as the main model.

    .venv/bin/python tools/fetch_mtp.py [--bits 4]

Output: models/qwen3.5-9b-mtp/mtp-{bits}bit.safetensors (+ mtp-bf16.safetensors)
"""

import argparse
import json
import struct
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = "Qwen/Qwen3.5-9B"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "qwen3.5-9b-mtp"
INDEX = Path.home() / ".lmstudio/models/Qwen/Qwen3.5-9B/model.safetensors.index.json"

# zero-centered RMSNorm weights in the HF checkpoint: effective weight is 1 + w
SHIFT_NORMS = (
    "input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight",
    "k_norm.weight", "mtp.norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
)
QUANTIZE = ("fc.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def url(shard):
    return f"https://huggingface.co/{REPO}/resolve/main/{shard}"


def get_range(shard, start, end):
    req = urllib.request.Request(url(shard), headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise RuntimeError(f"server ignored Range request ({r.status})")
        data = r.read()
    if len(data) != end - start:
        raise RuntimeError(f"short read {len(data)} != {end - start}")
    return data


def fetch(index_path):
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    by_shard = defaultdict(list)
    for k, shard in weight_map.items():
        if k.startswith("mtp."):
            by_shard[shard].append(k)

    tensors = {}
    total = 0
    for shard, keys in sorted(by_shard.items()):
        n = struct.unpack("<Q", get_range(shard, 0, 8))[0]
        header = json.loads(get_range(shard, 8, 8 + n))
        base = 8 + n
        for k in sorted(keys):
            meta = header[k]
            assert meta["dtype"] == "BF16", meta
            s, e = meta["data_offsets"]
            raw = get_range(shard, base + s, base + e)
            total += len(raw)
            # bf16 -> float32 via bit shift (numpy has no bf16)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
            tensors[k] = mx.array(f32).astype(mx.bfloat16)
            print(f"  {k:55s} {tuple(meta['shape'])}  {len(raw) / 1e6:7.1f} MB", flush=True)
    print(f"downloaded {total / 1e6:.0f} MB")
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "mtp-bf16.safetensors"
    if raw_path.exists():
        tensors = mx.load(str(raw_path))
        print(f"using cached {raw_path}")
    else:
        if not INDEX.exists():
            sys.exit(f"missing {INDEX}")
        tensors = fetch(INDEX)
        mx.save_safetensors(str(raw_path), tensors)

    out = {}
    for k, v in tensors.items():
        if any(k.endswith(s) for s in SHIFT_NORMS) and v.ndim == 1:
            out[k] = (v.astype(mx.float32) + 1.0).astype(mx.bfloat16)
        elif any(k.endswith(s) for s in QUANTIZE):
            w, scales, biases = mx.quantize(v, group_size=args.group_size, bits=args.bits)
            base = k[: -len(".weight")]
            out[f"{base}.weight"], out[f"{base}.scales"], out[f"{base}.biases"] = w, scales, biases
        else:
            out[k] = v
    dst = OUT / f"mtp-{args.bits}bit.safetensors"
    mx.save_safetensors(str(dst), out, metadata={"group_size": str(args.group_size), "bits": str(args.bits)})
    size = sum(x.nbytes for x in out.values())
    print(f"saved {dst} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
