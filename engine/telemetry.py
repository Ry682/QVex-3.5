"""Background power/temperature sampler built on `macmon pipe` (no sudo needed)."""

import json
import shutil
import subprocess
import threading
import time
from collections import deque


class PowerMonitor:
    def __init__(self, interval_ms: int = 250, history: int = 2400):
        self.interval_ms = interval_ms
        self.samples: deque[dict] = deque(maxlen=history)
        self.available = shutil.which("macmon") is not None
        self._proc = None
        self._thread = None

    def start(self):
        if not self.available or self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            ["macmon", "pipe", "-i", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(0.6)  # let the first samples arrive
        return self

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _read(self):
        for line in self._proc.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            temp = d.get("temp", {})
            mem = d.get("memory", {})
            self.samples.append(
                {
                    "t": time.perf_counter(),
                    "gpu_w": d.get("gpu_power", 0.0),
                    "cpu_w": d.get("cpu_power", 0.0),
                    "sys_w": d.get("sys_power", 0.0),
                    "gpu_mhz": d.get("gpu_freq_mhz", 0),
                    "cpu_c": temp.get("cpu_temp_avg", 0.0),
                    "gpu_c": temp.get("gpu_temp_avg", 0.0),
                    "gpu_pct": 100 * d.get("gpu_active_ratio", 0.0),
                    "cpu_pct": 100 * d.get("cpu_active_ratio", 0.0),
                    "ram_used_gb": mem.get("ram_usage", 0) / 2**30,
                    "ram_total_gb": mem.get("ram_total", 0) / 2**30,
                }
            )

    def window(self, t0: float, t1: float) -> dict:
        """Aggregate samples whose timestamp falls in [t0, t1]."""
        samples = list(self.samples)
        s = [x for x in samples if t0 <= x["t"] <= t1]
        if not s:
            # window shorter than the sampling interval: take the nearest sample after t0
            s = [x for x in samples if x["t"] >= t0][:1]
        if not s:
            return {}
        n = len(s)
        return {
            "gpu_w": sum(x["gpu_w"] for x in s) / n,
            "cpu_w": sum(x["cpu_w"] for x in s) / n,
            "soc_w": sum(x["gpu_w"] + x["cpu_w"] for x in s) / n,
            "sys_w": sum(x["sys_w"] for x in s) / n,
            "gpu_mhz": sum(x["gpu_mhz"] for x in s) / n,
            "max_gpu_c": max(x["gpu_c"] for x in s),
            "max_cpu_c": max(x["cpu_c"] for x in s),
        }

    def latest(self) -> dict:
        if not self.samples:
            return {}
        x = self.samples[-1]
        return {**x, "soc_w": x["gpu_w"] + x["cpu_w"]}

    def current_temp(self) -> float:
        """Hottest of GPU/CPU die average: the number that matters for a fanless chassis."""
        x = self.latest()
        return max(x.get("gpu_c", 0.0), x.get("cpu_c", 0.0))
