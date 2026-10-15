from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import signal
import tempfile
import threading
import time
from typing import Callable, Mapping, Protocol, Sequence

from ncm_resource import ResourceController, ResourceLimitError, ResourcePolicy


@dataclass(frozen=True)
class NumericalPolicy:
    precision: str = "fp32"
    attention: str = "reference"
    checkpoint_copies: str = "batched"
    frozen_pooling: str = "metal"

    def __post_init__(self):
        options = {"precision": ("fp32", "bf16"), "attention": ("reference", "bf16-math"),
                   "checkpoint_copies": ("blocking", "batched"),
                   "frozen_pooling": ("reference", "metal")}
        for name, choices in options.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}.")
        if self.attention == "bf16-math" and self.precision != "bf16":
            raise ValueError("Reduced-math attention requires explicit bf16 precision.")


class PuzzleTask(Protocol):
    name: str
    metadata: Mapping
    source_paths: Sequence[Path]

    def run(self, runtime: "PuzzleRuntime") -> Mapping: ...


def _atomic_json(path: Path, value):
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".puzzle-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _source_hashes(paths):
    root = Path(__file__).resolve().parent
    hashes = {}
    for supplied in paths:
        path = Path(supplied).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Task source is not a file. {path}")
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def _dependency_versions(device):
    versions = {"python": platform.python_version(), "system": platform.system(),
                "release": platform.release(), "machine": platform.machine()}
    for package in ("numpy", "torch" if device != "mlx" else "mlx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _make_controller(device, policy):
    if device == "cpu":
        controller = ResourceController(policy, backend="cpu", sample_gpu=lambda: None)
        controller.check_startup()
        return controller
    if device == "mps":
        import torch
        from ncm_resource import make_torch_controller

        if not torch.backends.mps.is_available():
            raise ValueError("The requested MPS backend is unavailable.")
        return make_torch_controller(torch, policy)
    import mlx.core as mx
    from ncm_resource import make_mlx_controller

    return make_mlx_controller(mx, policy)


class PuzzleRuntime:
    def __init__(self, *, device, run_directory, resuming, descriptor, controller,
                 numerical, seconds, clock=time.monotonic):
        self.device = device
        self.run_directory = Path(run_directory)
        self.resuming = resuming
        self.descriptor = _json_copy(descriptor)
        self.numerical = numerical
        self._controller = controller
        self._deadline = clock() + seconds
        self._clock = clock
        self._stop_reason = None
        self._resource_stop_reason = None
        self._in_unit = False

    @property
    def stop_reason(self):
        if self._stop_reason is None and self._clock() >= self._deadline:
            self._stop_reason = "The invocation reached its cooperative time limit."
        return self._stop_reason

    @property
    def stop_requested(self):
        return self.stop_reason is not None

    @contextmanager
    def work_unit(self):
        if self._in_unit:
            raise RuntimeError("Puzzle work units cannot be nested.")
        if self.stop_requested:
            raise ResourceLimitError(self.stop_reason)
        self._controller.begin_unit()
        self._in_unit = True
        primary = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            self._in_unit = False
            try:
                reason = self._controller.finish_unit()
                self._stop_reason = self._stop_reason or reason
                self._resource_stop_reason = self._resource_stop_reason or reason
            except BaseException as error:
                self._stop_reason = self._stop_reason or f"Work-unit cleanup failed. {type(error).__name__}. {error}"
                if primary is None:
                    raise

    @contextmanager
    def forward_context(self):
        if self.device == "mlx":
            yield
            return
        from ncm_attention import select_generator_attention

        with select_generator_attention(self.numerical.attention):
            if self.numerical.precision == "bf16":
                import torch

                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    yield
            else:
                yield

    def cpu_tree(self, payload):
        if self.device == "mlx":
            raise TypeError("MLX tasks own MLX serialization. Torch checkpoint conversion is unavailable.")
        if self._in_unit:
            raise RuntimeError("Save checkpoints after the completed work-unit boundary.")
        if self.numerical.checkpoint_copies == "batched":
            from ncm_checkpoint import batched_cpu_tree

            return batched_cpu_tree(payload)
        import torch

        def visit(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: visit(item) for key, item in value.items()}
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, tuple):
                return tuple(visit(item) for item in value)
            return value

        return visit(payload)


@contextmanager
def _signals(runtime):
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def stop(signum, frame):
        runtime._stop_reason = runtime._stop_reason or f"Received {signal.Signals(signum).name}."

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_task(task_module: str, arguments: Sequence[str], *, run_directory: Path,
             device: str, resume: bool = False, policy: ResourcePolicy = ResourcePolicy(),
             numerical: NumericalPolicy = NumericalPolicy(), seconds: float = 600,
             _controller_factory: Callable | None = None) -> dict:
    """Run a task after startup guards, preserving primary exceptions and final reports.

    The private factory seam receives ``(device, policy)`` and returns a configured
    controller. The caller must still provide macOS memory telemetry in production.
    """
    run = Path(run_directory).expanduser().resolve()
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent / f".{run.name}.puzzle.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another puzzle runner owns this run directory.") from error
        run.mkdir(parents=True, exist_ok=True)
        admitted = False
        controller = runtime = None
        signal_scope = ExitStack()
        result = None
        primary = cleanup_error = None
        resources = {"backend": device, "policy": asdict(policy), "stop_reason": None}
        status, code = "failed", 1
        try:
            if device not in ("cpu", "mps", "mlx"):
                raise ValueError("Device must be explicitly cpu, mps, or mlx.")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 600:
                raise ValueError("The invocation limit must be positive and at most 600 seconds.")
            if device == "mlx" and (numerical.precision != "fp32" or numerical.attention != "reference" or numerical.frozen_pooling != "reference"):
                raise ValueError("MLX tasks cannot use Torch precision, attention, or Metal pooling options.")
            if device == "cpu" and numerical.precision != "fp32":
                raise ValueError("The CPU reference runtime supports fp32 precision.")
            controller = (_make_controller if _controller_factory is None else _controller_factory)(device, policy)
            controller.check_startup()
            runtime = PuzzleRuntime(device=device, run_directory=run, resuming=resume,
                                    descriptor={}, controller=controller,
                                    numerical=numerical, seconds=seconds)
            signal_scope.enter_context(_signals(runtime))
            module = importlib.import_module(task_module)
            task = module.build_task(list(arguments))
            if not isinstance(task.name, str) or not task.name or not isinstance(task.metadata, Mapping):
                raise ValueError("A task needs a nonempty name and JSON metadata mapping.")
            source_paths = [Path(__file__), Path(__file__).with_name("puzzle_runner.py"),
                            Path(__file__).with_name("puzzle_ncm.py"),
                            Path(__file__).with_name("ncm_resource.py"),
                            Path(__file__).with_name("ncm_attention.py"),
                            Path(__file__).with_name("ncm_checkpoint.py"),
                            Path(__file__).with_name("ncm_fast_targets.py"),
                            Path(__file__).with_name("ncm_area_pool.py"),
                            Path(__file__).with_name("ncm_torch_ops.py"),
                            Path(module.__file__), *task.source_paths]
            descriptor = {"format": "puzzle-run-identity-v1", "task_module": task_module,
                          "task_name": task.name, "task": _json_copy(dict(task.metadata)),
                          "device": device, "versions": _dependency_versions(device), "resources": asdict(policy),
                          "numerical": asdict(numerical), "sources_sha256": _source_hashes(source_paths)}
            if device == "mlx":
                descriptor["numerical"]["checkpoint_copies"] = "not-applicable"
            manifest = run / "run-manifest.json"
            if resume:
                if not manifest.is_file() or json.loads(manifest.read_text()) != descriptor:
                    raise ValueError("Resume requires the same task, configuration, backend, numerical policy, resources, and source hashes.")
            else:
                if any(run.iterdir()):
                    raise ValueError("Choose an empty run directory, or use --resume for an existing run.")
                _atomic_json(manifest, descriptor)
            admitted = True
            runtime.descriptor = _json_copy(descriptor)
            result = _json_copy(task.run(runtime))
            status, code = ("stopped", 0) if runtime.stop_requested else ("completed", 0)
        except BaseException as error:
            primary = error
        finally:
            if controller is not None:
                try:
                    resources = controller.close()
                except BaseException as error:
                    cleanup_error = error
            if cleanup_error is not None or resources.get("stop_reason") or (runtime and runtime._resource_stop_reason):
                status, code = "failed", 1
            error = primary or cleanup_error
            report = {"format": "puzzle-run-result-v1", "status": status,
                      "returncode": code, "error": ({"type": type(error).__name__, "message": str(error)} if error else None),
                      "stop_reason": (runtime.stop_reason if runtime else None) or resources.get("stop_reason"),
                      "resources": resources, "task_result": result}
            try:
                if not admitted:
                    _atomic_json(run.parent / f".{run.name}.admission-error-report.json", report)
                else:
                    _atomic_json(run / "resource-report.json", {**resources, "runtime_status": status,
                                                                "runtime_error": report["error"],
                                                                "runtime_stop_reason": report["stop_reason"]})
                    _atomic_json(run / "result.json", report)
            except BaseException:
                if primary is None:
                    raise
            finally:
                signal_scope.close()
        if primary is not None:
            raise primary.with_traceback(primary.__traceback__)
        return report
