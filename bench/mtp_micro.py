"""Per-component time and power: main T=1, main T=2 (+keep_prev), MTP head, lm_head."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
from engine.model import load_model
from engine.mtp import load_mtp
from engine.telemetry import PowerMonitor

MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
m = load_model(MODEL)
mtp = load_mtp(m, Path(__file__).resolve().parent.parent / "models/qwen3.5-9b-mtp/mtp-4bit.safetensors")
cache = m.make_cache(); m(mx.array([list(range(100, 400))]), cache)
mcache = mtp.make_cache(); h = mx.random.normal((1, 300, 4096)).astype(mx.bfloat16)
mx.eval(mtp(h, mx.array([list(range(100, 400))]), mcache))
x1 = mx.array([[123]]); x2 = mx.array([[123, 456]])

def bench(name, fn, n=60):
    for _ in range(3): mx.eval(fn())
    time.sleep(2)
    t0 = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    dt = (time.perf_counter() - t0) / n
    pw = mon.window(t0, time.perf_counter())
    print(f"{name:34s} {dt*1000:6.2f} ms  {pw.get('soc_w',0):5.1f} W  {pw.get('gpu_mhz',0):5.0f} MHz  {pw.get('soc_w',0)*dt*1000:5.0f} mJ")

def main_step(ids, keep=False):
    out = m(ids, cache, last_only=False, keep_prev=keep)
    for c in cache:
        if keep: c.rollback(ids.shape[1] - 1) if ids.shape[1] > 1 else None
    for c in cache:
        if hasattr(c, "k"): c.offset -= 1 if ids.shape[1] == 1 or keep else 2
    return out

def step_t1():
    out = m(x1, cache)
    for c in cache: c.offset -= 1  # keep context length fixed (state drift irrelevant for timing)
    return out

def step_t2():
    out = m(x2, cache, last_only=False, keep_prev=True)
    for c in cache: c.offset -= 2
    return out

def step_t2_nokeep():
    out = m(x2, cache, last_only=False)
    for c in cache: c.offset -= 2
    return out

def mtp_step():
    out = mtp(h[:, :1], x1, mcache)
    mcache.offset -= 1
    return out

hx = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
with PowerMonitor(200) as mon:
    bench("main T=1", step_t1)
    bench("main T=2 (keep_prev)", step_t2)
    bench("main T=2", step_t2_nokeep)
    bench("MTP head T=1", mtp_step)
    bench("lm_head only", lambda: m.lm_head(hx))
    bench("qmm 4096->12288 M=1", lambda: m.layers[0].mlp.gate_up(hx))
    bench("qmm 4096->12288 M=2", lambda: m.layers[0].mlp.gate_up(mx.concatenate([hx, hx], 1)))
