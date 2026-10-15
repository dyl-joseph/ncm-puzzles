#!/usr/bin/env python3
"""Independently check the evaluator's saved user-visible results without inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(puzzle, raw):
    shape = len(raw) == 89 and all(type(raw[index]) is int and raw[index] == 10 for index in range(9, 80, 10))
    total = sum(value != 0 for value in puzzle)
    if shape:
        cells = [value for index, value in enumerate(raw) if index % 10 != 9]
        digits = all(type(value) is int and 1 <= value <= 9 for value in cells)
        rows = digits and all(set(cells[start:start + 9]) == set(range(1, 10)) for start in range(0, 81, 9))
        columns = digits and all(set(cells[column::9]) == set(range(1, 10)) for column in range(9))
        correct = sum(type(value) is int and value == given for value, given in zip(cells, puzzle) if given)
        boxes = digits and all(len({cells[(row + dr) * 9 + column + dc]
                                    for dr in range(3) for dc in range(3)}) == 9
                               for row in (0, 3, 6) for column in (0, 3, 6))
    else:
        digits = rows = columns = boxes = False
        correct = 0
    clues = shape and correct == total
    return {"shape_valid": shape, "digits_valid": digits, "rows_valid": rows, "columns_valid": columns,
            "clues_preserved": clues, "sudoku_boxes_valid": boxes, "given_cells_correct": correct,
            "given_cells_total": total, "solved": shape and digits and rows and columns and clues}


def totals(scores):
    count = len(scores)
    correct = sum(item["given_cells_correct"] for item in scores)
    given = sum(item["given_cells_total"] for item in scores)
    solved = sum(item["solved"] for item in scores)
    return {"scored": count, "solved": solved, "solve_rate": solved / count if count else None,
            "validity_rates": {key: sum(item[key] for item in scores) / count if count else None
                               for key in ("shape_valid", "digits_valid", "rows_valid", "columns_valid",
                                           "clues_preserved", "sudoku_boxes_valid")},
            "given_cells_correct": correct, "given_cells_total": given,
            "given_cell_agreement": correct / given if given else None}


def check(directory, count, device):
    summary = json.loads((directory / "summary.json").read_text())
    cases_path = directory / "cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    require(summary["status"] == "complete", "Evaluation is incomplete.")
    require(summary["device"] == device, f"Expected {device} evaluation, found {summary['device']}.")
    expected_backend = "mlx-metal" if device == "mlx" else f"pytorch-{device}"
    require(summary["backend"] == expected_backend, "The recorded inference backend differs from the requested device.")
    if device == "mlx":
        require(summary["device_info"].get("device_name") and summary["mlx_enable_tf32"] == "0",
                "Missing Metal GPU identity or FP32 inference metadata.")
    require(summary["requested"] == summary["attempted"] == summary["completed"] == len(cases) == len(predictions) == count,
            "Requested, attempted, completed, case, or prediction count differs.")
    require(len({case["id"] for case in cases}) == count, "Repeated case IDs.")
    require(summary["cases_sha256"] == hashlib.sha256(cases_path.read_bytes()).hexdigest(), "Cases checksum mismatch.")
    bins = Counter(case["clue_count"] for case in cases)
    require(summary["clue_bin_counts"] == {str(key): value for key, value in bins.items()}, "Clue bins differ.")
    require(set(bins).issubset(summary["clue_counts"]), "Saved cases use undeclared clue counts.")
    scores = []
    for case, prediction in zip(cases, predictions):
        require(case["id"] == prediction["id"], "Case and prediction IDs differ.")
        require(len(case["puzzle"]) == len(case["witness"]) == 81, "Invalid saved board shape.")
        witness = case["witness"]
        require(all(set(witness[start:start + 9]) == set(range(1, 10)) for start in range(0, 81, 9))
                and all(set(witness[column::9]) == set(range(1, 10)) for column in range(9)), "Invalid witness.")
        require(all(given == 0 or given == value for given, value in zip(case["puzzle"], witness)), "Witness changes a clue.")
        expected = score(case["puzzle"], prediction["raw_ids"])
        require(case["clue_count"] == prediction["clue_count"] == expected["given_cells_total"], "Clue counts differ.")
        require(prediction["score"] == expected, f"Independent score differs for {case['id']}.")
        scores.append(expected)
    metrics = totals(scores)
    require(summary["metrics"] == metrics, "Summary metrics differ from the raw predictions.")
    require(set(summary["per_clue_count"]) == {str(value) for value in summary["clue_counts"]},
            "Per-clue groups differ from the declared clue counts.")
    for clues, reported in summary["per_clue_count"].items():
        group = [item for case, item in zip(cases, scores) if case["clue_count"] == int(clues)]
        require(reported == totals(group), f"Per-clue metrics differ for {clues}.")
    return {"verified": True, "directory": str(directory.resolve()), "count": count,
            "device": summary["device"], "backend": summary["backend"], "cases_sha256": summary["cases_sha256"],
            "metrics": metrics, "argv": sys.argv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-device", choices=("mlx", "cpu", "mps", "cuda"), default="mlx")
    args = parser.parse_args()
    try:
        require(args.expected_count > 0, "Expected count must be positive.")
        print(json.dumps(check(args.directory, args.expected_count, args.expected_device), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Evaluation proof failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
