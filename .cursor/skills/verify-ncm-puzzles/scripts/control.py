#!/usr/bin/env python3
"""Run the real NCM command-line tools in isolated verification sessions."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[4]
SESSIONS = ROOT / "runs/verification"
PYTHON = ROOT / ".venv/bin/python"
ENTRY_POINTS = {
    "evaluate_latin.py": "--output",
    "prepare_latin.py": "--output",
    "train_latin.py": "--run",
    "resource_runner.py": None,
    "run_latin_10min.py": "--run",
    "continue_latin.py": "--run",
    "scripts/verify_mlx.py": "--output",
    "puzzle_runner.py": "--run",
    "scripts/verify_puzzle_runtime.py": "--output",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_hashes():
    paths = [*ROOT.glob("*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("examples/*.py"),
             *ROOT.glob("configs/*.yaml"), ROOT / "requirements.txt"]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def session(directory, *, allow_cleaned=False):
    directory = Path(directory).resolve()
    if directory.parent != SESSIONS.resolve():
        raise ValueError("Use the session directory printed by launch.")
    record = json.loads((directory / "session.json").read_text())
    if record["repository"] != str(ROOT) or record["directory"] != str(directory):
        raise ValueError("This session belongs to another checkout.")
    if record.get("cleaned_at") and not allow_cleaned:
        raise ValueError("This session was cleaned. Launch a new session.")
    return directory, record


@contextmanager
def exclusive(directory):
    with (directory / ".control.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another command owns this session. Wait for it before cleanup.") from error
        if (directory / "active.json").exists():
            raise RuntimeError(f"Inspect {directory / 'active.json'} for an abandoned child before continuing.")
        yield


def launch():
    if not PYTHON.is_file():
        raise RuntimeError("Prepare .venv with the fresh-checkout instructions in README.md first.")
    SESSIONS.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("ncm-puzzles-%Y%m%dT%H%M%SZ-")
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=SESSIONS))
    for name in ("work", "evidence/actions", "evidence/artifacts"):
        (directory / name).mkdir(parents=True)
    (directory / ".control.lock").touch()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(directory / "session.json", {
        "repository": str(ROOT), "directory": str(directory), "created_at": now(),
        "git_head": revision, "source_sha256": source_hashes(),
    })
    print(directory)


def doctor(directory, device):
    directory, record = session(directory)
    with exclusive(directory):
        if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
            raise RuntimeError("Run this helper with .venv/bin/python, using Python 3.12.")
        if source_hashes() != record["source_sha256"]:
            raise RuntimeError("App source changed after launch. Clean up and launch a new session.")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(ROOT))
        import evaluate_latin as app
        provenance = app.verify_setup(ROOT / "artifacts/sudoku/hard", device)
        import torch
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable.")
        if device == "mlx":
            import mlx.core as mx
            if not mx.metal.is_available():
                raise RuntimeError("MLX Metal is unavailable.")
        print(json.dumps({"ready": True, "session": str(directory), "device": device,
                          "python": sys.version, "source_sha256": record["source_sha256"],
                          "provenance": provenance}, indent=2, sort_keys=True))


def values(arguments, flag):
    found = []
    for index, argument in enumerate(arguments):
        if argument == flag:
            if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {flag}.")
            found.append(arguments[index + 1])
        elif argument.startswith(flag + "="):
            found.append(argument.split("=", 1)[1])
    return found


def isolated_arguments(directory, arguments):
    entry, *options = arguments
    if entry not in ENTRY_POINTS:
        raise ValueError(f"Choose a real app entry point: {', '.join(ENTRY_POINTS)}")
    required = ENTRY_POINTS[entry]
    if entry == "resource_runner.py":
        if not options or options[0] not in ("train", "evaluate"):
            raise ValueError("resource_runner.py requires train or evaluate.")
        required = "--run" if options[0] == "train" else "--output"
    if not values(options, required):
        raise ValueError(f"Supply an explicit {required} inside this session.")
    needs_cache = entry == "prepare_latin.py" or (entry == "evaluate_latin.py" and not values(options, "--data"))
    if entry == "resource_runner.py" and options[0] == "evaluate" and not values(options, "--data"):
        needs_cache = True
    if needs_cache and not values(options, "--cache-dir"):
        raise ValueError("Supply an isolated --cache-dir under this session's work directory.")
    writable = ("--output", "--run", "--cache-dir", "--resource-report")
    for argument in options:
        option = argument.split("=", 1)[0]
        if option != "--" and option.startswith("--") and option not in writable and any(flag.startswith(option) for flag in writable):
            raise ValueError("Use full names for writable destination flags.")
    for flag in writable:
        for value in values(options, flag):
            path = (ROOT / value).resolve()
            base = directory / ("work" if flag == "--cache-dir" else "evidence/artifacts")
            if not path.is_relative_to(base) or path == base:
                raise ValueError(f"{flag} must name a child of {base}.")
    return [str(PYTHON), "-B", "-u", str(ROOT / entry), *options]


def drive(directory, label, arguments):
    directory, _ = session(directory)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", label):
        raise ValueError("Use a unique lowercase action label with letters, digits, and hyphens.")
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise ValueError("Supply an app script after --.")
    command = isolated_arguments(directory, arguments)
    with exclusive(directory):
        action = directory / "evidence/actions" / label
        action.mkdir()
        write_json(action / "command.json", {"feature_action": label, "argv": command,
                                              "cwd": str(ROOT), "started_at": now()})
        master, slave = pty.openpty()
        started = time.monotonic()
        process = None
        previous = {}
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdin=slave, stdout=slave,
                                       stderr=subprocess.PIPE, start_new_session=True)
            os.close(slave)
            slave = None
            write_json(directory / "active.json", {"pid": process.pid, "argv": command, "action": label})

            def forward(signum, frame):
                if process.poll() is None:
                    process.send_signal(signum)

            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, forward)
            with selectors.DefaultSelector() as selector, \
                    (action / "stdout.txt").open("wb") as stdout, \
                    (action / "stderr.txt").open("wb") as stderr, \
                    (action / "transcript.jsonl").open("w") as transcript:
                selector.register(master, selectors.EVENT_READ, ("stdout", stdout, sys.stdout.buffer))
                selector.register(process.stderr, selectors.EVENT_READ, ("stderr", stderr, sys.stderr.buffer))
                while selector.get_map():
                    for key, _ in selector.select(timeout=0.5):
                        try:
                            payload = os.read(key.fd, 65536)
                        except OSError as error:
                            if error.errno != errno.EIO:
                                raise
                            payload = b""
                        if not payload:
                            selector.unregister(key.fileobj)
                            continue
                        name, log, console = key.data
                        log.write(payload)
                        log.flush()
                        console.write(payload)
                        console.flush()
                        transcript.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                                                     "stream": name, "text": payload.decode(errors="replace")}) + "\n")
                        transcript.flush()
            process.wait()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.wait()
                write_json(action / "result.json", {"returncode": process.returncode, "pid": process.pid,
                                                     "finished_at": now(), "wall_seconds": time.monotonic() - started})
                process.stderr.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            os.close(master)
            if slave is not None:
                os.close(slave)
            (directory / "active.json").unlink(missing_ok=True)
        return process.returncode if process.returncode >= 0 else 128 - process.returncode


def cleanup(directory):
    directory, record = session(directory, allow_cleaned=True)
    with exclusive(directory):
        work = directory / "work"
        if work.exists():
            shutil.rmtree(work)
        record["cleaned_at"] = now()
        write_json(directory / "session.json", record)
        evidence = directory / "evidence"
        files = {str(path.relative_to(evidence)): {"bytes": path.stat().st_size, "sha256": digest(path)}
                 for path in sorted(evidence.rglob("*")) if path.is_file() and path != evidence / "manifest.json"}
        write_json(evidence / "manifest.json", {"files": files, "scratch_removed": not work.exists()})
        print(json.dumps({"cleaned": True, "evidence": str(evidence), "files_preserved": len(files)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("launch")
    check = sub.add_parser("doctor")
    check.add_argument("session")
    check.add_argument("--device", choices=("cpu", "mlx", "mps"), default="mlx")
    run = sub.add_parser("run")
    run.add_argument("session")
    run.add_argument("label")
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    clean = sub.add_parser("cleanup")
    clean.add_argument("session")
    args = parser.parse_args()
    try:
        if args.mode == "launch":
            launch()
        elif args.mode == "doctor":
            doctor(args.session, args.device)
        elif args.mode == "run":
            return drive(args.session, args.label, args.arguments)
        else:
            cleanup(args.session)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
