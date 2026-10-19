"""Our quantizer: imatrix-weighted affine quantization in MLX's format, then per-unit sensitivity.

    .venv/bin/python -m quant.quantize cache --bits 4 3      # quantize every linear at these widths
    .venv/bin/python -m quant.quantize sensitivity            # KL cost of moving each unit 4 -> 3 bit

Plain MLX quantization takes each group of 64 weights, maps [min, max] onto the 2^b levels and
rounds. That treats every weight as equally important. Here each group's range is searched
(shrinking min/max to clip outliers) and the scale/offset are re-fitted by weighted least squares,
where the weight of input channel j is its imatrix value: how strongly that channel is driven
by real text. Errors on channels that carry large activations cost more, exactly as they do at
inference time. The output is bit-for-bit the MLX layout, so the engine's kernels run it as is.
"""

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from engine.model import QEmbedding, QLinear, Qwen35
from engine.telemetry import PowerMonitor

from .common import (CACHE_DIR, DATA, Source, kl_vs_ref, layer_linears, load_tokens, log_softmax,
                     units, wait_cool)

GROUP = 64
SHRINK = (0.0, 0.02, 0.05, 0.09, 0.14)   # fraction of the group range clipped at each end
ROW_CHUNK = 2048   # rows per GPU pass: keeps the float32 temporaries ~1 GB on a 16 GB Mac


def _quantize_rows(W, w, bits):
    """W [N, G, gs] float32, w [1|N, G, gs] channel weights -> q uint8, scale, bias (bf16-exact)."""
    maxq = 2 ** bits - 1
    mn, mx_ = W.min(-1, keepdims=True), W.max(-1, keepdims=True)
    rng = mx_ - mn
    Sw = w.sum(-1, keepdims=True) * mx.ones_like(mn)
    Sx = (w * W).sum(-1, keepdims=True)
    best_err = best_s = best_z = None
    for a in SHRINK:
        for b in SHRINK:
            lo, hi = mn + a * rng, mx_ - b * rng
            s = mx.maximum((hi - lo) / maxq, 1e-10)
            z = lo
            for _ in range(2):  # round, then re-fit scale/offset by weighted least squares
                q = mx.clip(mx.round((W - z) / s), 0, maxq)
                Sq, Sqq, Sqx = (w * q).sum(-1, keepdims=True), (w * q * q).sum(-1, keepdims=True), (w * q * W).sum(-1, keepdims=True)
                det = Sw * Sqq - Sq * Sq
                ok = det > 1e-20
                safe = mx.where(ok, det, 1.0)
                s_new = (Sw * Sqx - Sq * Sx) / safe
                z_new = (Sqq * Sx - Sq * Sqx) / safe
                good = ok & (s_new > 1e-10)
                s, z = mx.where(good, s_new, s), mx.where(good, z_new, z)
            # the engine stores scale/offset in bf16: score the candidate with what will be stored
            s, z = s.astype(mx.bfloat16).astype(mx.float32), z.astype(mx.bfloat16).astype(mx.float32)
            s = mx.maximum(s, 1e-10)
            q = mx.clip(mx.round((W - z) / s), 0, maxq)
            err = (w * (s * q + z - W) ** 2).sum(-1, keepdims=True)
            if best_err is None:
                best_err, best_s, best_z = err, s, z
            else:
                better = err < best_err
                best_err = mx.where(better, err, best_err)
                best_s, best_z = mx.where(better, s, best_s), mx.where(better, z, best_z)
            mx.eval(best_err, best_s, best_z)
    q = mx.clip(mx.round((W - best_z) / best_s), 0, maxq).astype(mx.uint8)
    return q, best_s[..., 0].astype(mx.bfloat16), best_z[..., 0].astype(mx.bfloat16)


def pack(q: np.ndarray, bits: int) -> np.ndarray:
    """uint8 codes [N, K] -> MLX packed uint32 [N, K*bits/32] (little-endian bit stream per row)."""
    N, K = q.shape
    bitarr = np.unpackbits(q[..., None], axis=-1, bitorder="little")[..., :bits].reshape(N, K * bits)
    return np.packbits(bitarr, axis=-1, bitorder="little").view(np.uint32)


def quantize(W: mx.array, imp: np.ndarray | None, bits: int, gs: int = GROUP):
    """bf16 [N, K] weight -> (packed uint32, scales bf16, biases bf16) in MLX affine format."""
    N, K = W.shape
    G = K // gs
    if imp is None:
        w = mx.ones((1, G, gs), mx.float32)
    else:
        imp = imp / imp.mean()
        w = mx.array(imp + 0.01, dtype=mx.float32).reshape(1, G, gs)  # floor: no channel is free
    packed, scales, biases = [], [], []
    for r in range(0, N, ROW_CHUNK):
        Wc = W[r:r + ROW_CHUNK].astype(mx.float32).reshape(-1, G, gs)
        q, s, z = _quantize_rows(Wc, w, bits)
        packed.append(pack(np.array(q.reshape(-1, K)), bits))
        scales.append(s); biases.append(z)
    return mx.array(np.concatenate(packed)), mx.concatenate(scales), mx.concatenate(biases)


def weighted_rel_err(W, packed, s, z, imp, bits):
    """Relative imatrix-weighted error, row block by row block (lm_head alone is 4 GB in float32)."""
    w = mx.array(imp, dtype=mx.float32)
    num = den = 0.0
    for r in range(0, W.shape[0], ROW_CHUNK):
        d = mx.dequantize(packed[r:r + ROW_CHUNK], s[r:r + ROW_CHUNK], z[r:r + ROW_CHUNK],
                          group_size=GROUP, bits=bits).astype(mx.float32)
        Wc = W[r:r + ROW_CHUNK].astype(mx.float32)
        num += (((d - Wc) ** 2) * w).sum().item()
        den += ((Wc ** 2) * w).sum().item()
    return num / den


# ---------------------------------------------------------------- cache


def build_cache(bits_list):
    mx.set_cache_limit(512 << 20)  # don't hoard freed GPU buffers: the bf16 source already fills the page cache
    src = Source()
    cfg = src.cfg
    imatrix = dict(np.load(DATA / "imatrix.npz"))
    for bits in bits_list:
        out = CACHE_DIR / f"b{bits}"
        out.mkdir(parents=True, exist_ok=True)
        report = {}
        t0 = time.perf_counter()
        groups = [(f"layer_{i:02d}", layer_linears(cfg, i)) for i in range(cfg.num_hidden_layers)]
        groups += [("lm_head", ["lm_head"]), ("embed", ["model.embed_tokens"])]
        for fname, names in groups:
            path = out / f"{fname}.safetensors"
            if path.exists():
                continue
            tensors = {}
            for n in names:
                W = src.weight(n)
                imp = imatrix.get(n)  # the embedding is a lookup: no input statistics
                packed, s, z = quantize(W, imp, bits)
                tensors.update({f"{n}.weight": packed, f"{n}.scales": s, f"{n}.biases": z})
                if imp is not None and W.shape[0] <= 32768:  # report only; skip the 248k-row lm_head
                    ours = weighted_rel_err(W, packed, s, z, imp, bits)
                    ref = mx.quantize(W, group_size=GROUP, bits=bits)  # packed: ~1/4 of bf16 size
                    plain = weighted_rel_err(W, *ref, imp, bits)
                    del ref
                    report[n] = {"ours": ours, "plain": plain}
            tmp = path.with_suffix(".tmp.safetensors")
            mx.save_safetensors(str(tmp), tensors)
            tmp.rename(path)  # an interrupted run never leaves a half-written file that looks done
            mx.clear_cache()
            gain = [r["plain"] / r["ours"] for k, r in report.items() if any(k == n for n in names)]
            print(f"{bits}-bit {fname:9s} {time.perf_counter() - t0:6.1f}s"
                  + (f"  errore {np.mean(gain):.2f}x più basso di MLX standard" if gain else ""), flush=True)
        if report:
            old = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
            (out / "report.json").write_text(json.dumps({**old, **report}, indent=1))


class Cache:
    """Quantized tensors from CACHE_DIR, loaded lazily per bit width."""

    def __init__(self):
        self.files = {}

    def get(self, name: str, bits: int):
        if name == "lm_head":
            fname = "lm_head"
        elif name == "model.embed_tokens":
            fname = "embed"
        else:
            fname = f"layer_{int(name.split('.')[2]):02d}"
        key = (bits, fname)
        if key not in self.files:
            self.files[key] = mx.load(str(CACHE_DIR / f"b{bits}" / f"{fname}.safetensors"))
        f = self.files[key]
        return f[f"{name}.weight"], f[f"{name}.scales"], f[f"{name}.biases"]


def assemble(src: Source, cache: Cache, bits_of) -> Qwen35:
    """Build an engine model from cached tensors; bits_of(name) -> bit width of that linear."""
    cfg = src.cfg
    lin = lambda n: QLinear(*cache.get(n, bits_of(n)), GROUP, bits_of(n))
    layers = [src.layer(i, lin) for i in range(cfg.num_hidden_layers)]
    embed = QEmbedding(*cache.get("model.embed_tokens", 4), GROUP, 4)
    model = Qwen35(cfg, embed, layers, src.small["model.norm.weight"], lin("lm_head"))
    return model


# ---------------------------------------------------------------- sensitivity


def sensitivity(n_windows: int, low: int, base_bits: int):
    """For each unit: mean KL vs bf16 when only that unit drops from base_bits to low bits."""
    src = Source()
    cfg = src.cfg
    cache = Cache()
    calib, _, calib_src, _ = load_tokens()
    ref = np.load(DATA / "reference.npz")
    # a few windows from every source, the same ones for every measurement (paired comparison)
    pick = []
    for s in dict.fromkeys(calib_src):
        idx = np.where(calib_src == s)[0]
        pick += list(idx[np.linspace(0, len(idx) - 1, max(1, n_windows * len(idx) // len(calib))).astype(int)])
    pick = sorted(set(pick))
    print(f"{len(pick)} finestre di calibrazione: {[str(calib_src[i]) for i in pick]}")
    ref_idx = [mx.array(ref["idx"][i]) for i in pick]
    ref_lp = [mx.array(ref["lp"][i].astype(np.float32)) for i in pick]

    model = assemble(src, cache, lambda n: base_bits)
    monitor = PowerMonitor(interval_ms=1000).start()

    # inputs of every layer for the base model, so a change in layer i only reruns layers i..end
    inputs = [[] for _ in range(cfg.num_hidden_layers + 1)]
    for j in pick:
        x = model.embed(mx.array(calib[j])[None])
        for i, layer in enumerate(model.layers):
            inputs[i].append(x)
            x = layer(x, model.make_cache()[i])
            mx.eval(x)
        inputs[-1].append(x)

    def kl_from(i, layers_override=None, head=None):
        tot = 0.0
        for w in range(len(pick)):
            x = inputs[i][w]
            for k in range(i, cfg.num_hidden_layers):
                layer = layers_override if (layers_override is not None and k == i) else model.layers[k]
                x = layer(x, model.make_cache()[k])
            h = mx.fast.rms_norm(x, model.norm, cfg.rms_norm_eps)
            lp = log_softmax((head or model.lm_head)(h)[0])
            kl = kl_vs_ref(lp, ref_idx[w], ref_lp[w]).mean()
            tot += kl.item()
        return tot / len(pick)

    base = kl_from(0)
    print(f"base {base_bits}-bit: KL {base:.5f}", flush=True)
    result = {"base_bits": base_bits, "low_bits": low, "base_kl": base, "windows": [int(i) for i in pick], "units": {}}
    t0 = time.perf_counter()
    for u, names in units(cfg).items():
        params = sum(int(np.prod(src.weight(n).shape)) for n in names)
        bits_of = lambda n: low if n in names else base_bits
        if u == "lm_head":
            kl = kl_from(cfg.num_hidden_layers, head=QLinear(*cache.get("lm_head", low), GROUP, low))
        else:
            i = int(u.split(".")[2])
            layer = src.layer(i, lambda n: QLinear(*cache.get(n, bits_of(n)), GROUP, bits_of(n)))
            kl = kl_from(i, layers_override=layer)
        result["units"][u] = {"kl": kl, "delta": kl - base, "params": params}
        print(f"  {u:28s} ΔKL {kl - base:+.5f}  ({params / 1e6:5.0f}M param)  {time.perf_counter() - t0:6.0f}s  "
              f"{monitor.current_temp():.0f} °C", flush=True)
        mx.clear_cache()
        wait_cool(monitor)
    (DATA / f"sensitivity_{base_bits}to{low}.json").write_text(json.dumps(result, indent=1))
    monitor.stop()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    s = sub.add_parser("sensitivity")
    s.add_argument("--windows", type=int, default=12)
    s.add_argument("--low", type=int, default=3)
    s.add_argument("--base", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "cache":
        build_cache(args.bits)
    else:
        sensitivity(args.windows, args.low, args.base)


if __name__ == "__main__":
    main()
