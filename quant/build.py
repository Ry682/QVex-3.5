"""Assemble one of our models from the quantized cache and write it as a normal MLX model folder.

    .venv/bin/python -m quant.build --name qvex-q4  --all 4
    .venv/bin/python -m quant.build --name qvex-mix --bpw 4.0     # 3/4-bit mix chosen by sensitivity

Mixed models: units are moved from 4 to 3 bits starting from the one that costs the least KL per
byte saved (data/sensitivity_4to3.json), until the linears average the requested bits per weight
(scales and offsets included: 4-bit = 4.5 bpw, 3-bit = 3.5 bpw).
"""

import argparse
import json
import shutil

import mlx.core as mx
import numpy as np

from .common import DATA, MLX_DIR, ROOT, Source, units
from .quantize import GROUP, Cache

BPW = {3: 3.5, 4: 4.5, 5: 5.5, 6: 6.5}


def plan(src: Source, bpw: float | None, all_bits: int | None) -> dict[str, int]:
    u = units(src.cfg)
    if all_bits:
        return {name: all_bits for names in u.values() for name in names}
    sens = json.loads((DATA / "sensitivity_4to3.json").read_text())["units"]
    bits = {k: 4 for k in u}
    total = sum(v["params"] for v in sens.values())
    order = sorted(sens, key=lambda k: max(sens[k]["delta"], 0) / sens[k]["params"])
    avg = BPW[4]
    for k in order:
        if avg <= bpw:
            break
        bits[k] = 3
        avg -= (BPW[4] - BPW[3]) * sens[k]["params"] / total
    return {name: bits[k] for k, names in u.items() for name in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bpw", type=float)
    g.add_argument("--all", type=int, choices=[3, 4, 5, 6])
    args = ap.parse_args()

    src = Source()
    cache = Cache()
    bits = plan(src, args.bpw, args.all)
    out = ROOT / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)

    tensors = {f"language_model.{k}": v for k, v in src.small.items()}
    for n, b in bits.items():
        w, s, z = cache.get(n, b)
        tensors.update({f"language_model.{n}.weight": w, f"language_model.{n}.scales": s,
                        f"language_model.{n}.biases": z})
    w, s, z = cache.get("model.embed_tokens", 4)
    tensors.update({"language_model.model.embed_tokens.weight": w,
                    "language_model.model.embed_tokens.scales": s,
                    "language_model.model.embed_tokens.biases": z})
    mx.save_safetensors(str(out / "model.safetensors"), tensors, metadata={"format": "mlx"})

    cfg = json.loads((MLX_DIR / "config.json").read_text())
    q = {"group_size": GROUP, "bits": 4, "mode": "affine"}
    q.update({f"language_model.{n}": {"group_size": GROUP, "bits": b} for n, b in bits.items() if b != 4})
    cfg["quantization"] = cfg["quantization_config"] = q
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in MLX_DIR.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)

    params = {n: int(np.prod(src.weight(n).shape)) for n in bits}
    avg = sum(BPW[bits[n]] * p for n, p in params.items()) / sum(params.values())
    n3 = sum(p for n, p in params.items() if bits[n] == 3) / sum(params.values())
    size = (out / "model.safetensors").stat().st_size
    meta = {"name": args.name, "bpw_linears": avg, "frac_3bit": n3, "bytes": size,
            "bits": {k: bits[names[0]] for k, names in units(src.cfg).items()}}
    (out / "ours.json").write_text(json.dumps(meta, indent=1))
    print(f"{args.name}: {avg:.2f} bpw sui pesi lineari, {100 * n3:.0f}% a 3 bit, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
