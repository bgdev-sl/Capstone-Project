"""Resource and inference-performance measurement helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Generic, TypeVar

import psutil

T = TypeVar("T")


@dataclass
class ResourceMetrics:
    """Measurements recorded during a model operation."""

    elapsed_ms: float
    latency_ms_per_sample: float
    cpu_percent: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    throughput_samples_per_second: float
    sample_count: int

    def as_dict(self) -> dict:
        return asdict(self)


class ResourceMonitor:
    """Measure process-level resource use for an operation.

    Measurements are intentionally lightweight so they can run on an edge-like
    VM. They describe the process, not the entire host, and should therefore be
    reported as prototype measurements rather than hardware-independent truths.
    """

    def __init__(self):
        self.process = psutil.Process()

    def measure(self, operation: Callable[[], T], sample_count: int = 1) -> tuple[T, ResourceMetrics]:
        rss_before = self.process.memory_info().rss
        cpu_before = self.process.cpu_times()
        start = perf_counter()

        result = operation()

        elapsed = perf_counter() - start
        cpu_after = self.process.cpu_times()
        rss_after = self.process.memory_info().rss

        elapsed_ms = elapsed * 1000.0
        latency_ms_per_sample = elapsed_ms / max(sample_count, 1)
        # Process CPU time divided by wall time is more interpretable for short
        # operations than psutil's non-blocking instantaneous percentage. It can
        # exceed 100% when a multi-threaded estimator uses multiple CPU cores.
        cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
        cpu_percent = (cpu_seconds / elapsed * 100.0) if elapsed > 0 else 0.0
        throughput = sample_count / elapsed if elapsed > 0 and sample_count > 0 else 0.0
        metrics = ResourceMetrics(
            elapsed_ms=elapsed_ms,
            latency_ms_per_sample=latency_ms_per_sample,
            cpu_percent=cpu_percent,
            rss_before_mb=rss_before / (1024 * 1024),
            rss_after_mb=rss_after / (1024 * 1024),
            rss_delta_mb=(rss_after - rss_before) / (1024 * 1024),
            throughput_samples_per_second=throughput,
            sample_count=sample_count,
        )
        return result, metrics
