"""Custom multi-row qmv vs mx.quantized_matmul: correctness, time, energy."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
from engine.kernels import qmv_multi
from engine.telemetry import PowerMonitor

shapes = {"lm_head 248320x4096": (248320, 4096), "mlp gate_up 24576x4096": (24576, 4096),
          "down 4096x12288": (4096, 12288)}
with PowerMonitor(200) as mon:
    for name, (N, K) in shapes.items():
        w = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.02
        wq, s, b = mx.quantize(w, group_size=64, bits=4); del w; mx.eval(wq, s, b)
        reps = max(50, int(2.0 / (wq.nbytes / 110e9)))
        for M in (1, 2):
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
            got = qmv_multi(x, wq, s, b)
            err = (mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max() /
                   mx.abs(ref.astype(mx.float32)).max()).item()
            row = []
            for label, f in (("mlx", lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)),
                             ("ours", lambda: qmv_multi(x, wq, s, b))):
                for _ in range(5): mx.eval(f())
                time.sleep(2.0)
                t0 = time.perf_counter(); outs = [f() for _ in range(reps)]; mx.eval(outs)
                dt = (time.perf_counter() - t0) / reps
                pw = mon.window(t0, time.perf_counter())
                row.append(f"{label} {dt*1000:6.3f} ms {pw.get('soc_w',0) * dt * 1000:6.2f} mJ")
            print(f"{name:24s} M={M}  rel.err {err:.1e}  |  " + "  |  ".join(row), flush=True)
