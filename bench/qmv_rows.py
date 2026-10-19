"""Does MLX's quantized matvec cost scale with rows? lm_head (248320x4096, 4-bit)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
from engine.telemetry import PowerMonitor

w = mx.random.normal((248320, 4096)).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4); del w
mx.eval(wq, s, b)
with PowerMonitor(200) as mon:
    for M in (1, 2, 4, 8):
        x = mx.random.normal((M, 4096)).astype(mx.bfloat16)
        f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        for _ in range(5): mx.eval(f())
        time.sleep(1.5)
        n = 200; t0 = time.perf_counter()
        outs = [f() for _ in range(n)]; mx.eval(outs)  # queued back-to-back: no CPU bubbles
        dt = (time.perf_counter() - t0) / n
        pw = mon.window(t0, time.perf_counter())
        gbs = wq.nbytes * 1.06 / dt / 1e9
        print(f"M={M}: {dt*1000:5.2f} ms  {gbs:5.0f} GB/s  {pw.get('soc_w',0):5.1f} W  {pw.get('gpu_mhz',0):5.0f} MHz  {pw.get('soc_w',0)*dt*1000:5.1f} mJ/call")
