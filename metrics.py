"""Lightweight instrumentation for the ASR -> LLM -> TTS pipeline.

Drop `measure()` around any stage to get its wall-clock latency plus CPU
(process + whole-machine) and memory usage, and GPU utilization/memory if a
supported backend (NVIDIA via pynvml/torch, or Apple MPS via torch) is
detected on the machine:

    from metrics import measure

    with measure("asr.transcribe"):
        transcript = asr.transcribe(audio_path)

Each call logs an immediate one-line reading and appends to a shared
`default_recorder`, so `default_recorder.summary()` prints an aggregated
table (count / avg / max per stage) at the end of a run.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

_process = psutil.Process()
# Prime cpu_percent(): the first call always returns a meaningless 0.0 (or an
# average since process start), so warm it up once at import time.
_process.cpu_percent(interval=None)
psutil.cpu_percent(interval=None)


@dataclass
class StageMetrics:
    name: str
    duration_s: float
    cpu_percent_proc: float  # this process's CPU use, avg over the block (100% = 1 core)
    cpu_percent_system: float  # whole-machine CPU use, avg over the block
    memory_mb: float  # this process's peak RSS during the block
    gpu_percent: float | None = None
    gpu_memory_mb: float | None = None
    gpu_backend: str | None = None

    def format(self) -> str:
        parts = [
            self.name,
            f"{self.duration_s * 1000:.0f}ms",
            f"cpu(proc)={self.cpu_percent_proc:.0f}%",
            f"cpu(sys)={self.cpu_percent_system:.0f}%",
            f"mem={self.memory_mb:.0f}MB",
        ]
        if self.gpu_percent is not None:
            parts.append(f"gpu={self.gpu_percent:.0f}%")
        if self.gpu_memory_mb is not None:
            parts.append(f"gpu_mem={self.gpu_memory_mb:.0f}MB")
        elif self.gpu_backend is None:
            parts.append("gpu=n/a")
        return " | ".join(parts)


# Sentinel so "not detected yet" (False) is distinguishable from "detected: no
# GPU" (None).
_gpu_backend_cache: str | None | bool = False


def gpu_backend() -> str | None:
    """Detect which GPU-stats backend is usable on this machine, once, and
    cache the result. Returns None if no GPU metrics are available (e.g. no
    NVIDIA GPU and no torch MPS backend), in which case only CPU metrics are
    collected."""
    global _gpu_backend_cache
    if _gpu_backend_cache is not False:
        return _gpu_backend_cache

    try:
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetHandleByIndex(0)
        _gpu_backend_cache = "nvml"
        return _gpu_backend_cache
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            _gpu_backend_cache = "torch_cuda"
            return _gpu_backend_cache
        if torch.backends.mps.is_available():
            _gpu_backend_cache = "torch_mps"
            return _gpu_backend_cache
    except Exception:
        pass

    _gpu_backend_cache = None
    return _gpu_backend_cache


def _sample_gpu() -> tuple[float | None, float | None]:
    """Best-effort single (utilization_percent, memory_mb) GPU sample.
    Returns (None, None) if no GPU backend is available on this machine."""
    backend = gpu_backend()
    if backend == "nvml":
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
            return float(util), float(mem)
        except Exception:
            return None, None
    if backend == "torch_cuda":
        try:
            import torch

            return float(torch.cuda.utilization()), torch.cuda.memory_allocated() / (1024**2)
        except Exception:
            return None, None
    if backend == "torch_mps":
        try:
            import torch

            # MPS exposes no utilization counter, only memory.
            return None, torch.mps.current_allocated_memory() / (1024**2)
        except Exception:
            return None, None
    return None, None


class _ResourceSampler:
    """Polls process/system CPU, memory, and GPU stats on a background
    thread for the duration of a `measure()` block, so even sub-second
    stages get an averaged/peak reading instead of one noisy sample."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.cpu_proc_samples: list[float] = []
        self.cpu_sys_samples: list[float] = []
        self.mem_samples: list[float] = []
        self.gpu_util_samples: list[float] = []
        self.gpu_mem_samples: list[float] = []

    def start(self) -> None:
        _process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        self.cpu_proc_samples.append(_process.cpu_percent(interval=None))
        self.cpu_sys_samples.append(psutil.cpu_percent(interval=None))
        self.mem_samples.append(_process.memory_info().rss / (1024**2))
        gpu_util, gpu_mem = _sample_gpu()
        if gpu_util is not None:
            self.gpu_util_samples.append(gpu_util)
        if gpu_mem is not None:
            self.gpu_mem_samples.append(gpu_mem)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        # Guarantee at least one reading for blocks shorter than interval_s.
        self._sample_once()

    @staticmethod
    def _avg(samples: list[float]) -> float:
        return sum(samples) / len(samples) if samples else 0.0

    def result(self, name: str, duration_s: float) -> StageMetrics:
        return StageMetrics(
            name=name,
            duration_s=duration_s,
            cpu_percent_proc=self._avg(self.cpu_proc_samples),
            cpu_percent_system=self._avg(self.cpu_sys_samples),
            memory_mb=max(self.mem_samples) if self.mem_samples else 0.0,
            gpu_percent=self._avg(self.gpu_util_samples) if self.gpu_util_samples else None,
            gpu_memory_mb=max(self.gpu_mem_samples) if self.gpu_mem_samples else None,
            gpu_backend=gpu_backend(),
        )


class MetricsRecorder:
    """Accumulates StageMetrics across a run for a final aggregated
    summary (see `summary()`), grouped by stage name."""

    def __init__(self):
        self.records: list[StageMetrics] = []
        self._lock = threading.Lock()

    def add(self, metric: StageMetrics) -> None:
        with self._lock:
            self.records.append(metric)

    def summary(self) -> str:
        if not self.records:
            return "No metrics recorded."

        by_name: dict[str, list[StageMetrics]] = {}
        for record in self.records:
            by_name.setdefault(record.name, []).append(record)

        header = (
            f"{'stage':<28} {'n':>3} {'avg_ms':>8} {'max_ms':>8} "
            f"{'avg_cpu%':>9} {'avg_mem_mb':>11} {'avg_gpu%':>9}"
        )
        lines = [header]
        for name, records in by_name.items():
            durations = [r.duration_s * 1000 for r in records]
            cpus = [r.cpu_percent_proc for r in records]
            mems = [r.memory_mb for r in records]
            gpus = [r.gpu_percent for r in records if r.gpu_percent is not None]
            avg_gpu = f"{sum(gpus) / len(gpus):>9.1f}" if gpus else f"{'n/a':>9}"
            lines.append(
                f"{name:<28} {len(records):>3} {sum(durations) / len(durations):>8.0f} "
                f"{max(durations):>8.0f} {sum(cpus) / len(cpus):>9.1f} "
                f"{sum(mems) / len(mems):>11.0f} {avg_gpu}"
            )
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps([asdict(r) for r in self.records], indent=2), encoding="utf-8")
        return path


default_recorder = MetricsRecorder()


@contextlib.contextmanager
def measure(name: str, recorder: MetricsRecorder | None = None, *, log: bool = True):
    """Context manager: measure wall time, CPU (process + system), memory,
    and GPU usage (if a supported backend is detected) for the wrapped
    block. Metrics are logged immediately and appended to `recorder` (the
    shared `default_recorder` unless one is passed) for later aggregation
    via `recorder.summary()`."""
    recorder = recorder or default_recorder
    sampler = _ResourceSampler()
    sampler.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        sampler.stop()
        metric = sampler.result(name, duration)
        recorder.add(metric)
        if log:
            logger.info("[metrics] %s", metric.format())


def record_instant(
    name: str, duration_s: float, recorder: MetricsRecorder | None = None, *, log: bool = True
) -> StageMetrics:
    """Record a single point-in-time latency (e.g. time-to-first-chunk) with
    no resource sampling attached -- for measurements that don't map onto a
    whole `measure()` block."""
    recorder = recorder or default_recorder
    metric = StageMetrics(
        name=name,
        duration_s=duration_s,
        cpu_percent_proc=0.0,
        cpu_percent_system=0.0,
        memory_mb=0.0,
    )
    recorder.add(metric)
    if log:
        logger.info("[metrics] %s: %.0fms", name, duration_s * 1000)
    return metric
