"""Thermal governor: keeps the die under a target temperature by pacing work.

Decode is memory-bound, so average power scales with tokens per second. The
governor holds a tokens/s cap that shrinks while the chip is above target and
grows back when it cools, and inserts short pauses between prefill chunks.
"""

import time
from dataclasses import dataclass

from .telemetry import PowerMonitor

MODES = {
    # name: (target °C, label)
    "eco": (70.0, "Eco"),
    "balanced": (80.0, "Balanced"),
    "max": (95.0, "Max"),
}


@dataclass
class GovernorState:
    mode: str
    target_c: float
    temp_c: float
    tps_cap: float | None   # None = unthrottled
    throttled_s: float


class ThermalGovernor:
    MIN_TPS = 3.0
    MAX_TPS = 200.0

    def __init__(self, monitor: PowerMonitor, mode: str = "balanced"):
        self.monitor = monitor
        self.set_mode(mode)
        self.tps_cap: float | None = None
        self.throttled_s = 0.0
        self._last_adjust = time.perf_counter()
        self._last_token = None

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.target_c = MODES[mode][0]

    def begin(self):
        self._last_token = None
        self.throttled_s = 0.0

    def _adjust(self, observed_tps: float | None):
        now = time.perf_counter()
        dt = now - self._last_adjust
        if dt < 0.5:
            return
        self._last_adjust = now
        temp = self.monitor.current_temp()
        if not temp:
            return
        err = temp - self.target_c
        if err > 0:
            base = self.tps_cap or observed_tps or 25.0
            # proportional slowdown: ~4% per °C over target, per second
            factor = max(0.5, 1.0 - 0.04 * err * dt)
            self.tps_cap = max(self.MIN_TPS, base * factor)
        elif self.tps_cap is not None and err < -2.0:
            self.tps_cap *= 1.0 + min(0.15, 0.02 * -err) * dt
            if self.tps_cap >= self.MAX_TPS or (observed_tps and self.tps_cap > observed_tps * 1.5):
                self.tps_cap = None  # cool enough: release the brake

    def pace_token(self, observed_tps: float | None = None):
        """Call once per generated token; sleeps if a cap is active."""
        self._adjust(observed_tps)
        now = time.perf_counter()
        if self.tps_cap and self._last_token is not None:
            wait = 1.0 / self.tps_cap - (now - self._last_token)
            if wait > 0:
                time.sleep(wait)
                self.throttled_s += wait
        self._last_token = time.perf_counter()

    def pace_prefill(self, chunk_seconds: float):
        """Call after each prefill chunk; pauses while the chip is over target."""
        temp = self.monitor.current_temp()
        over = temp - self.target_c if temp else 0.0
        if over > 0:
            # duty cycle: 1 °C over target -> pause 25% of the chunk time, capped
            wait = min(2.0, chunk_seconds * 0.25 * over)
            time.sleep(wait)
            self.throttled_s += wait

    def state(self) -> GovernorState:
        return GovernorState(self.mode, self.target_c, self.monitor.current_temp(),
                             self.tps_cap, self.throttled_s)
