# Resource and inference-performance measurements

from __future__ import annotations
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Generic, TypeVar
import psutil

T = TypeVar('T')

@dataclass
class ResourceMetrics:
    elapsed_ms: float
    latency_ms_per_sample: float 
    cpu_percent: float 
    rss_before_mb: float 
    rss_after_mb: float 
    rss_delta_mb: float
    sample_count: int 

    def as_dict(self) -> dict:
        return asdict(self) 

# Measure process-level resource use for an operation
# Process based, not hardware-independent
class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()

    def measure(self, operation: Callable[[], T], sample_count: int = 1) -> tuple[T, ResourceMetrics]:
        rss_before = self.process.memory_info().rss 
        self.process.cpu_percent(interval=None)
        start = perf_counter()
        result = operation()

        elapsed = perf_counter() - start 
        cpu_percent = self.process.cpu_percent(interval=None)
        rss_after = self.process.memory_info().rss 

        elapsed_ms = elapsed * 1000.0
        latency_ms_per_sample = elapsed_ms / max(sample_count, 1)
        metrics = ResourceMetrics(
            elapsed_ms=elapsed_ms,
            latency_ms_per_sample=latency_ms_per_sample,
            cpu_percent=cpu_percent,
            rss_before_mb=rss_before / (1024 * 1024),
            rss_after_mb=rss_after / (1024 * 1024),
            rss_delta_mb=(rss_after - rss_before) / (1024 * 1024),
            sample_count=sample_count
        )
        return result, metrics 

    