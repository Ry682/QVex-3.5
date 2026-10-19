"""Run the original bf16 model once, layer by layer, over the calibration and evaluation windows.

    .venv/bin/python -m quant.reference

The full bf16 model (17 GB) does not fit in 16 GB, but one layer at a time does: every window's
hidden state is pushed through layer i, then layer i is dropped and layer i+1 is read.

Writes:
    data/imatrix.npz    importance of every input channel of every linear: mean of x^2 over the
                        calibration tokens (the "imatrix" of llama.cpp / ds4)
    data/reference.npz  bf16 top-64 log-probabilities at every position (calibration + evaluation)
                        and the log-probability of the real next token, for KL / perplexity
"""

import time

import mlx.core as mx
import numpy as np

from engine.model import DenseLinear
from engine.telemetry import PowerMonitor

from .common import DATA, Source, fresh_cache, load_tokens, log_softmax, topk, wait_cool


class RecDense(DenseLinear):
    """Dense linear that accumulates sum(x^2) per input channel while `active`."""
    active = False
    stats: dict = {}

    def __init__(self, w, names):
        super().__init__(w)
        self.names = names

    def __call__(self, x):
        if RecDense.active:
            key = self.names[0]
            sq = (x.astype(mx.float32) ** 2).reshape(-1, x.shape[-1]).sum(0)
            s, n = RecDense.stats.get(key, (0, 0))
            RecDense.stats[key] = (s + sq, n + x.shape[0] * x.shape[1])
        return super().__call__(x)

    @staticmethod
    def fuse(layers):
        # fused projections share their input: one statistic serves all of them
        return RecDense(mx.concatenate([l.w for l in layers]), [n for l in layers for n in l.names])


def main():
    calib, evals, _, _ = load_tokens()
    seqs = np.concatenate([calib, evals])
    n_cal = len(calib)
    src = Source()
    cfg = src.cfg
    monitor = PowerMonitor(interval_ms=1000).start()
    t0 = time.perf_counter()

    emb = src.weight("model.embed_tokens")
    xs = [emb[mx.array(s)][None] for s in seqs]
    mx.eval(xs)
    del emb

    groups = {}  # first name of a fused group -> all names
    for i in range(cfg.num_hidden_layers):
        layer = src.layer(i, lambda n: RecDense(src.weight(n), [n]))
        for j in range(len(xs)):
            RecDense.active = j < n_cal
            xs[j] = layer(xs[j], fresh_cache(cfg, i))
            mx.eval(xs[j], [v[0] for v in RecDense.stats.values()])
        for obj in (layer.mlp.gate_up, layer.mlp.down, *vars(layer.mixer).values()):
            if isinstance(obj, RecDense):
                groups[obj.names[0]] = obj.names
        del layer
        mx.clear_cache()
        print(f"layer {i:2d}  {time.perf_counter() - t0:6.1f}s  {monitor.current_temp():.0f} °C", flush=True)
        wait_cool(monitor)

    norm = src.small["model.norm.weight"]
    head = RecDense(src.weight("lm_head"), ["lm_head"])
    groups["lm_head"] = ["lm_head"]
    all_idx, all_lp, tok_lp = [], [], []
    for j, x in enumerate(xs):
        RecDense.active = j < n_cal
        lp = log_softmax(head(mx.fast.rms_norm(x, norm, cfg.rms_norm_eps))[0])
        idx, val = topk(lp)
        nxt = mx.take_along_axis(lp[:-1], mx.array(seqs[j][1:, None]), axis=-1)[:, 0]
        mx.eval(idx, val, nxt)
        all_idx.append(np.array(idx)); all_lp.append(np.array(val.astype(mx.float16))); tok_lp.append(np.array(nxt))
    print(f"head done  {time.perf_counter() - t0:6.1f}s")

    imatrix = {}
    for key, names in groups.items():
        s, n = RecDense.stats[key]
        v = np.array(s / n, dtype=np.float32)
        for name in names:
            imatrix[name] = v
    np.savez(DATA / "imatrix.npz", **imatrix)
    np.savez(DATA / "reference.npz", idx=np.stack(all_idx), lp=np.stack(all_lp),
             tok_lp=np.stack(tok_lp), n_calib=n_cal)
    tl = np.stack(tok_lp)
    print(f"bf16 perplexity: calib {np.exp(-tl[:n_cal].mean()):.3f}  eval {np.exp(-tl[n_cal:].mean()):.3f}")
    print(f"wrote imatrix ({len(imatrix)} linears) and reference ({len(seqs)} windows)")
    monitor.stop()


if __name__ == "__main__":
    main()
