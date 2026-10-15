"""Per-process memory guards and synchronized GPU duty pacing on macOS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import plistlib
import re
import resource
import subprocess
import sys
import time
from typing import Callable

GIB = 1024 ** 3
PACING_METHODS = ("full-sleep", "idle-credit")


class ResourceLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourcePolicy:
    gpu_duty_cycle: float = 0.8
    memory_limit_bytes: int = 8 * GIB
    rss_limit_bytes: int = 8 * GIB
    min_available_bytes: int = 6 * GIB
    gpu_sample_interval_seconds: float = 1.0
    pacing_method: str = "idle-credit"

    def __post_init__(self):
        if (isinstance(self.gpu_duty_cycle, bool)
                or not isinstance(self.gpu_duty_cycle, (int, float))
                or not math.isfinite(self.gpu_duty_cycle)
                or not 0 < self.gpu_duty_cycle <= 0.8):
            raise ValueError("GPU duty cycle must be finite, above zero, and at most 0.8.")
        for name in ("memory_limit_bytes", "rss_limit_bytes", "min_available_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "min_available_bytes" else 1):
                raise ValueError(f"{name} must be an integer byte count with a positive limit.")
        interval = self.gpu_sample_interval_seconds
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or interval <= 0):
            raise ValueError("GPU sampling interval must be finite and positive.")
        if self.pacing_method not in PACING_METHODS:
            raise ValueError("Pacing method must be full-sleep or idle-credit.")


@dataclass(frozen=True)
class MemorySnapshot:
    physical_bytes: int
    available_bytes: int
    rss_bytes: int
    pressure_level: int
    swap_used_bytes: int
    swapouts: int
    pageouts: int
    page_size_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ResourceLimitError("Memory telemetry contains an invalid counter.")
        if (self.physical_bytes <= 0 or self.page_size_bytes <= 0
                or self.available_bytes > self.physical_bytes):
            raise ResourceLimitError("Memory telemetry contains invalid physical memory sizes.")


def _command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=2).stdout


def read_memory_snapshot() -> MemorySnapshot:
    if sys.platform != "darwin":
        raise ResourceLimitError("The memory guard requires macOS memory telemetry.")
    try:
        values = _command(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level",
                           "hw.memsize", "vm.swapusage"]).strip().splitlines()
        pressure, physical = int(values[0]), int(values[1])
        swap = re.search(r"\bused\s*=\s*([\d.]+)\s*([BKMGT])\b", values[2])
        if swap is None:
            raise ValueError("missing swap usage")
        swap_bytes = int(float(swap[1]) * 1024 ** "BKMGT".index(swap[2]))
        vm = _command(["/usr/bin/vm_stat"])
        page_size = int(re.search(r"page size of (\d+) bytes", vm)[1])
        counts = {}
        for line in vm.splitlines()[1:]:
            label, separator, count = line.partition(":")
            if separator:
                counts[label.strip('" ')] = int(count.strip().removesuffix("."))
        available = sum(counts[key] for key in ("Pages free", "Pages inactive", "Pages speculative"))
        return MemorySnapshot(physical, available * page_size,
                              int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), pressure,
                              swap_bytes, counts["Swapouts"], counts["Pageouts"], page_size)
    except ResourceLimitError:
        raise
    except Exception as error:
        raise ResourceLimitError(f"Cannot read required macOS memory telemetry. {error}") from error


def read_gpu_fraction() -> float | None:
    """Read optional global GPU activity, rather than a per-process allocation limit."""
    try:
        result = subprocess.run(["/usr/sbin/ioreg", "-a", "-r", "-c", "AGXAccelerator"],
                                check=True, capture_output=True, timeout=2)
        devices = plistlib.loads(result.stdout)
        values = [float(device["PerformanceStatistics"]["Device Utilization %"]) / 100
                  for device in devices if "PerformanceStatistics" in device
                  and "Device Utilization %" in device["PerformanceStatistics"]]
        return max(values) if values and all(math.isfinite(value) and 0 <= value <= 1 for value in values) else None
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, plistlib.InvalidFileException):
        return None


class ResourceController:
    def __init__(self, policy: ResourcePolicy = ResourcePolicy(), *, backend: str = "cpu",
                 synchronize: Callable[[], None] = lambda: None,
                 gpu_memory_bytes: Callable[[], int] = lambda: 0,
                 sample_memory: Callable[[], MemorySnapshot] = read_memory_snapshot,
                 sample_gpu: Callable[[], float | None] = read_gpu_fraction,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.policy, self.backend = policy, backend
        self._synchronize, self._gpu_memory_bytes = synchronize, gpu_memory_bytes
        self._sample_memory, self._sample_gpu = sample_memory, sample_gpu
        self._clock, self._sleep = clock, sleep
        self._baseline = self._latest = None
        self._unit_started = None
        self._last_gpu_sample = -math.inf
        self._gpu_samples = []
        self._active_seconds = self._idle_seconds = 0.0
        self._post_sync_idle_seconds = self._sleep_seconds = 0.0
        self._work_units = self._peak_gpu_bytes = self._peak_rss_bytes = 0
        self._stop_reason = None

    def _check_memory(self) -> str | None:
        snapshot = self._sample_memory()
        if not isinstance(snapshot, MemorySnapshot):
            raise ResourceLimitError("Memory sampler did not return a MemorySnapshot.")
        if self._baseline is None:
            self._baseline = snapshot
        self._latest = snapshot
        self._peak_rss_bytes = max(self._peak_rss_bytes, snapshot.rss_bytes)
        gpu = self._gpu_memory_bytes()
        if type(gpu) is not int or gpu < 0:
            raise ResourceLimitError("GPU allocation telemetry is unavailable or invalid.")
        self._peak_gpu_bytes = max(self._peak_gpu_bytes, gpu)
        if snapshot.pressure_level != 1:
            return f"macOS memory pressure is not normal. Level {snapshot.pressure_level}."
        if snapshot.available_bytes < self.policy.min_available_bytes:
            return f"Available RAM fell below the {self.policy.min_available_bytes}-byte reserve."
        if snapshot.rss_bytes > self.policy.rss_limit_bytes:
            return f"Process peak RSS exceeded the {self.policy.rss_limit_bytes}-byte limit."
        if gpu > self.policy.memory_limit_bytes:
            return f"GPU allocations exceeded the {self.policy.memory_limit_bytes}-byte limit."
        if (snapshot.swapouts < self._baseline.swapouts or snapshot.pageouts < self._baseline.pageouts
                or snapshot.page_size_bytes != self._baseline.page_size_bytes):
            return "macOS swap or pageout telemetry reset during this run."
        if snapshot.swapouts > self._baseline.swapouts or snapshot.pageouts > self._baseline.pageouts:
            return "macOS recorded new swap writes or pageouts during this run."
        return None

    def check_startup(self) -> None:
        try:
            reason = self._check_memory()
        except Exception as error:
            reason = f"Memory guard could not verify safe startup. {error}"
        if reason:
            self._stop_reason = reason
            raise ResourceLimitError(reason)

    def begin_unit(self) -> None:
        if self._stop_reason:
            raise ResourceLimitError(self._stop_reason)
        if self._unit_started is None:
            self.check_startup()
            self._unit_started = self._clock()

    def finish_unit(self) -> str | None:
        """Return a stop request so callers can first record and save an applied update."""
        if self._unit_started is None:
            self._stop_reason = self._stop_reason or "Resource controller finished a unit that was not started."
            return self._stop_reason
        started, self._unit_started = self._unit_started, None
        self._work_units += 1
        synchronized = sleeping = None
        try:
            self._synchronize()
            synchronized = self._clock()
            reason = self._check_memory()
            now = self._clock()
            if reason:
                self._stop_reason = reason
                return reason
            if now - self._last_gpu_sample >= self.policy.gpu_sample_interval_seconds:
                try:
                    value = self._sample_gpu()
                except Exception:
                    value = None
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and 0 <= value <= 1):
                    self._gpu_samples.append(value)
                self._last_gpu_sample = now
            sleeping = self._clock()
            active = max(0.0, synchronized - started)
            post_sync_idle = max(0.0, sleeping - synchronized)
            if self.policy.pacing_method == "idle-credit":
                delay = max(0.0, active * (1 / self.policy.gpu_duty_cycle - 1) - post_sync_idle)
            else:
                delay = max(0.0, sleeping - started) * (1 / self.policy.gpu_duty_cycle - 1)
            if delay:
                self._sleep(delay)
        except BaseException as error:
            self._stop_reason = f"Resource controller requested a safe stop. {type(error).__name__}. {error}"
        finally:
            ended = self._clock()
            if synchronized is None:
                self._active_seconds += max(0.0, ended - started)
            else:
                post_sync_idle = max(0.0, (sleeping if sleeping is not None else ended) - synchronized)
                slept = max(0.0, ended - sleeping) if sleeping is not None else 0.0
                self._post_sync_idle_seconds += post_sync_idle
                self._sleep_seconds += slept
                if self.policy.pacing_method == "idle-credit":
                    self._active_seconds += max(0.0, synchronized - started)
                    self._idle_seconds += post_sync_idle + slept
                else:
                    self._active_seconds += max(0.0, (sleeping if sleeping is not None else ended) - started)
                    self._idle_seconds += slept
        return self._stop_reason

    def report(self) -> dict:
        total = self._active_seconds + self._idle_seconds
        baseline, latest = self._baseline, self._latest
        return {"format": "ncm-resource-report-v1", "backend": self.backend, "policy": asdict(self.policy),
                "work_units": self._work_units, "active_seconds": self._active_seconds,
                "idle_seconds": self._idle_seconds,
                "pacing_method": self.policy.pacing_method,
                "post_sync_idle_seconds": self._post_sync_idle_seconds,
                "sleep_seconds": self._sleep_seconds,
                "work_duty_cycle": self._active_seconds / total if total else None,
                "peak_gpu_allocated_bytes": self._peak_gpu_bytes, "peak_rss_bytes": self._peak_rss_bytes,
                "gpu_memory_measurement": (
                    "Maximum synchronized endpoint active-plus-cache allocation or MLX allocator active-memory peak."
                    if self.backend == "mlx" else
                    "Maximum synchronized endpoint allocation sample. Temporary between-sample allocations are not measured."),
                "initial_memory": asdict(baseline) if baseline else None,
                "latest_memory": asdict(latest) if latest else None,
                "new_swapout_bytes": (latest.swapouts - baseline.swapouts) * latest.page_size_bytes if latest else None,
                "new_pageout_bytes": (latest.pageouts - baseline.pageouts) * latest.page_size_bytes if latest else None,
                "global_gpu_activity": {"samples": len(self._gpu_samples),
                                        "mean": sum(self._gpu_samples) / len(self._gpu_samples) if self._gpu_samples else None,
                                        "minimum": min(self._gpu_samples) if self._gpu_samples else None,
                                        "maximum": max(self._gpu_samples) if self._gpu_samples else None},
                "stop_reason": self._stop_reason}

    def close(self) -> dict:
        try:
            self._synchronize()
            reason = self._check_memory()
        except BaseException as error:
            reason = f"Final memory guard could not verify safe completion. {type(error).__name__}. {error}"
        self._stop_reason = self._stop_reason or reason
        return self.report()


def make_torch_controller(torch, policy: ResourcePolicy = ResourcePolicy(), **options) -> ResourceController:
    controller = ResourceController(policy, backend="mps", **options)
    controller.check_startup()
    try:
        recommended = torch.mps.recommended_max_memory()
        if type(recommended) is not int or recommended <= 0:
            raise ValueError("Invalid recommended MPS working set size.")
        fraction = policy.memory_limit_bytes / recommended
        if not 0 < fraction <= 2:
            raise ValueError("Requested MPS memory budget exceeds the supported allocation fraction.")
        torch.mps.set_per_process_memory_fraction(float(fraction))
        controller._synchronize = torch.mps.synchronize
        controller._gpu_memory_bytes = torch.mps.driver_allocated_memory
        controller.check_startup()
    except Exception as error:
        raise ResourceLimitError(f"Cannot configure the MPS allocation limit. {error}") from error
    return controller


def make_mlx_controller(mx, policy: ResourcePolicy = ResourcePolicy(), **options) -> ResourceController:
    controller = ResourceController(policy, backend="mlx", **options)
    controller.check_startup()
    try:
        mx.set_memory_limit(policy.memory_limit_bytes)
        mx.set_cache_limit(min(256 * 1024 ** 2, policy.memory_limit_bytes))
        controller._synchronize = mx.synchronize

        def memory_bytes():
            values = [mx.get_active_memory(), mx.get_cache_memory(), mx.get_peak_memory()]
            if any(type(value) is not int or value < 0 for value in values):
                raise ResourceLimitError("MLX allocation telemetry is unavailable or invalid.")
            return max(values[0] + values[1], values[2])

        controller._gpu_memory_bytes = memory_bytes
        controller.check_startup()
    except Exception as error:
        raise ResourceLimitError(f"Cannot configure the MLX memory guideline and cache limit. {error}") from error
    return controller
