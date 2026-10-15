#!/usr/bin/env python3
"""Verify real NCM task execution and exact saved resume through the puzzle CLI."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_state(left, right, counts, path="state"):
    import torch

    if type(left) is not type(right):
        raise AssertionError(f"Checkpoint type differs at {path}.")
    if isinstance(left, torch.Tensor):
        same = (left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                right.contiguous().reshape(-1).view(torch.uint8)))
        if not same:
            raise AssertionError(f"Checkpoint tensor bytes differ at {path}.")
        counts["tensors"] += 1
    elif isinstance(left, dict):
        if list(left) != list(right):
            raise AssertionError(f"Checkpoint keys differ at {path}.")
        for key in left:
            compare_state(left[key], right[key], counts, f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Checkpoint sequence length differs at {path}.")
        for index, (first, second) in enumerate(zip(left, right)):
            compare_state(first, second, counts, f"{path}[{index}]")
    else:
        same = struct.pack("!d", left) == struct.pack("!d", right) if isinstance(left, float) else left == right
        if not same:
            raise AssertionError(f"Checkpoint value differs at {path}.")
        counts["values"] += 1


def check_resources(report, device):
    if report["returncode"] != 0 or report["status"] != "completed":
        raise AssertionError("The invocation did not complete successfully.")
    resources = report["resources"]
    if (resources["backend"] != device or resources["stop_reason"] is not None
            or resources["new_swapout_bytes"] != 0 or resources["new_pageout_bytes"] != 0
            or resources["work_duty_cycle"] > 0.800001
            or resources["peak_rss_bytes"] > 8 * 1024**3
            or resources["peak_gpu_allocated_bytes"] > 8 * 1024**3):
        raise AssertionError("An invocation failed its recorded resource limits.")


def verify(output, device):
    import torch

    sys.path.insert(0, str(ROOT))
    from examples.sudoku4_task import score_board, solutions

    records = []
    common = ["--updates", "2", "--tokenizer-updates", "2", "--batch-size", "4",
              "--checkpoint-every", "1", "--evaluate"]

    def drive(label, run, task_arguments, *, resume=False, expected=0):
        command = [sys.executable, "-B", "-u", str(ROOT / "puzzle_runner.py"),
                   "--task", "examples.sudoku4_task", "--device", device,
                   "--run", str(run), "--seconds", "120"]
        if resume:
            command.append("--resume")
        command += ["--", *task_arguments]
        with (output / f"{label}.stdout.txt").open("w") as stdout, \
                (output / f"{label}.stderr.txt").open("w") as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        records.append({"label": label, "argv": command, "returncode": result.returncode})
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode != expected:
            raise AssertionError(f"{label} returned {result.returncode}, expected {expected}. Read its stderr log.")
        if expected == 0:
            report = read_json(run / "result.json")
            check_resources(report, device)
            return report
        return read_json(run.parent / f".{run.name}.admission-error-report.json")

    whole, split = output / "whole", output / "split"
    full_report = drive("whole", whole, common)
    partial_report = drive("split-first", split, [*common, "--stop-after", "3"])
    if partial_report["task_result"]["completed"] != {"tokenizer": 2, "generator": 1}:
        raise AssertionError("The split run did not save the expected applied boundary.")
    resumed_report = drive("split-resume", split, common, resume=True)
    for report in (full_report, resumed_report):
        task = report["task_result"]
        if task["completed"] != {"tokenizer": 2, "generator": 2}:
            raise AssertionError("The task did not reach both fixed update targets.")
        if task["ncm_loss"]["implementation"] != "reference":
            raise AssertionError("A 16-cell task selected an inapplicable specialized kernel.")
        if task["evaluation"]["count"] != 32:
            raise AssertionError("The example did not evaluate all 32 heldout puzzles.")
    first = torch.load(whole / "checkpoint.pt", map_location="cpu", weights_only=True)
    second = torch.load(split / "checkpoint.pt", map_location="cpu", weights_only=True)
    counts = {"tensors": 0, "values": 0}
    compare_state(first, second, counts)

    unchanged = {name: digest(split / name) for name in
                 ("checkpoint.pt", "run-manifest.json", "result.json", "resource-report.json")}
    changed = [*common]
    changed[changed.index("--batch-size") + 1] = "8"
    rejected = drive("changed-resume", split, changed, resume=True, expected=1)
    if rejected["returncode"] != 1 or rejected["error"]["type"] != "ValueError":
        raise AssertionError("The changed resume was not rejected by identity validation.")
    if unchanged != {name: digest(split / name) for name in unchanged}:
        raise AssertionError("Rejected admission changed the checkpoint or prior successful reports.")

    boards = solutions()
    if len(boards) != 288 or not all(score_board([0] * 16, board)["solved"] for board in boards):
        raise AssertionError("The source contains an invalid Sudoku board.")
    latin_without_boxes = (1, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 2, 4, 1, 2, 3)
    if score_board([0] * 16, latin_without_boxes)["solved"]:
        raise AssertionError("The independent Sudoku scorer accepted invalid boxes.")
    return {"verified": True, "device": device, "exact_saved_resume": counts,
            "completed": second["completed"], "reference_fallback": True,
            "changed_resume_preserved_state_and_reports": True,
            "independent_scorer_checked": True,
            "evaluation": resumed_report["task_result"]["evaluation"],
            "scope": "Execution and exact resume smoke test. No speedup or solve-quality claim.",
            "commands": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="CPU verifies exact bytes. MPS can fail strict equality because of backend embedding-gradient nondeterminism.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = verify(output, args.device)
    except Exception as error:
        report = {"verified": False, "error": {"type": type(error).__name__, "message": str(error)}}
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
