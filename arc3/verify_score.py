"""Independently verify saved gameplay scores, action traces and GPU use."""

import argparse
from collections import Counter
import hashlib
import json
import lzma
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def verify_weights(checkpoint):
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    for name, expected in provenance["weight_sha256"].items():
        actual = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        require(actual == expected, f"Frozen checkpoint hash differs: {name}")
    return provenance["weight_sha256"]


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def close(actual, expected):
    require(abs(actual - expected) <= 1e-8, f"Score mismatch: {actual} != {expected}")


def score_levels(actions, baselines, completed):
    require(len(actions) == len(baselines) > 0, "Level vectors differ or are empty")
    require(0 <= completed <= len(actions), "Completed level count is invalid")
    require(all(type(value) is int and value >= 0 for value in actions), "Invalid action count")
    require(all(type(value) is int and value > 0 for value in baselines), "Invalid human baseline")
    scores = [min(115.0, 100.0 * (baseline / action) ** 2)
              if index < completed and action > 0 else 0.0
              for index, (baseline, action) in enumerate(zip(baselines, actions))]
    total_weight = sum(range(1, len(scores) + 1))
    weighted = sum(weight * value for weight, value in enumerate(scores, 1)) / total_weight
    completion_cap = 100.0 * sum(index for index, value in enumerate(scores, 1) if value > 0) / total_weight
    return scores, min(weighted, completion_cap)


def frame_check(view):
    frame = view["frame"]
    require(len(frame) == 64 and all(len(row) == 64 for row in frame), "Frame shape differs from 64x64")
    require(all(type(value) is int and 0 <= value < 16 for row in frame for value in row),
            "Frame palette differs from 0..15")
    raw = bytes(value for row in frame for value in row)
    require(hashlib.sha256(raw).hexdigest() == view["frame_sha256"], "Frame hash differs")


def verify_trace(directory, summary):
    games = {row["game_id"]: row for row in summary["episodes"]}
    last, histories = {}, {}
    counts, calls, failures, ended, level_used = (Counter() for _ in range(5))
    plain_trace = directory / "trace.jsonl"
    stream = plain_trace.open() if plain_trace.exists() else lzma.open(directory / "trace.jsonl.xz", "rt")
    with stream:
        for line in stream:
            row = json.loads(line)
            game = row["game_id"]
            require(game in games, f"Unexpected traced game: {game}")
            event = row["event"]
            if event == "initial_reset":
                require(game not in last, "Duplicate initial reset")
                frame_check(row["frame_after"])
                require(row["harness_actions"] == row["level_budget_used"] == 0, "Initial reset consumed budget")
                require(row["action"] == {"action_id": 0, "data": {}}, "Malformed initial reset")
                last[game] = row["frame_after"]
                histories[game] = []
            elif event in ("action", "policy_failure"):
                before = row["frame_before"]
                require(before == last[game], "Observation differs from previous real game response")
                require(not ended[game] and not failures[game], "Policy continued after termination")
                if "history" in row:
                    expected = [[0, 0, 0]] * max(0, 4 - len(histories[game])) + histories[game][-4:]
                    require(row["history"] == expected, "Policy history includes actions it did not take")
                if "inference" in row:
                    calls[game] += 1
                if event == "policy_failure":
                    failures[game] += 1
                    if row["failure"]["reason"] == "prefix_capacity_overflow":
                        require(row["prefix_length"] > row["capacity"] == 1142, "Incorrect capacity failure")
                    continue
                action = row["action"]
                action_id = action["action_id"]
                require(action_id == 0 or action_id in before["available_actions"], "Unavailable action executed")
                if action_id == 6:
                    require(set(action["data"]) == {"x", "y"} and
                            all(type(value) is int and 0 <= value < 64 for value in action["data"].values()),
                            "Malformed click executed")
                    packet = [30, 32 + action["data"]["x"], 32 + action["data"]["y"]]
                else:
                    require(type(action_id) is int and 0 <= action_id < 8 and action["data"] == {},
                            "Malformed nonclick action executed")
                    packet = [24 + action_id, 1, 1]
                if "inference" in row:
                    require(row["inference"]["tokens"][:3] == packet, "Executed action differs from model output")
                after = row["frame_after"]
                frame_check(after)
                counts[game] += 1
                require(row["harness_actions"] == counts[game], "Trace action counter differs")
                old_level = before["levels_completed"]
                require(level_used[game] < games[game]["level_budgets"][old_level], "Action exceeded level budget")
                level_used[game] += 1
                if after["levels_completed"] > old_level:
                    level_used[game] = 0
                require(row["level_budget_used"] == level_used[game], "Level budget counter differs")
                if action_id == 0 or after["levels_completed"] != old_level:
                    histories[game] = []
                else:
                    histories[game] = (histories[game] + [packet])[-4:]
                last[game] = after
            elif event == "episode_end":
                ended[game] += 1
                require(ended[game] == 1, "Duplicate episode end")
                for key in ("termination", "steps", "harness_actions", "policy_calls", "forced_resets"):
                    require(row[key] == games[game][key], f"Episode end differs: {key}")
            elif event == "exception":
                require(games[game]["termination"] == "sdk_or_runner_failure", "Unreported runner failure")
            else:
                raise AssertionError(f"Unknown trace event: {event}")
    require(set(last) == set(games), "An evaluated game has no initial reset")
    for game, result in games.items():
        require(ended[game] == 1, "An evaluated game has no episode end")
        require(counts[game] == result["sdk_actions"] == result["harness_actions"] == result["steps"],
                f"SDK and trace action counts differ: {game}")
        require(last[game]["levels_completed"] == result["completed_levels"], "Completion count differs")
        if result["termination"] == "level_action_budget":
            require(level_used[game] == result["level_budgets"][result["completed_levels"]],
                    "Stopped before declared action budget")
    stats = summary.get("worker_stats")
    if stats:
        controls = json.loads((directory / "adapter_checks.json").read_text())
        require(controls["status"] == "PASS", "Replay controls did not pass")
        require(stats["calls"] == sum(calls.values()) + len(controls["checks"]), "Inference count differs")
        require(stats["device"] == "Device(gpu, 0)", "Neural inference did not run on MLX GPU")
        require(stats["cpu_neural_inference"] is False and stats["torch_imported"] is False,
                "Unexpected neural runtime")
        require(stats["duty_target"] == 0.8 and stats["duty_is_soft_pacing"] is True, "GPU pacing differs")
    return {"actions": sum(counts.values()), "frame_chains_verified": True,
            "own_action_history_verified": True, "budgets_verified": True,
            "policy_failure_games": sum(failures.values())}


def inspect_run(directory, baselines):
    summary = json.loads((directory / "summary.json").read_text())
    card = json.loads((directory / "scorecard.json").read_text())
    episodes = {row["game_id"]: row for row in summary["episodes"]}
    actual_ids = {environment["id"] for environment in card["environments"]}
    expected_ids = set(episodes) if summary["subset_smoke"] else set(baselines)
    require(expected_ids == actual_ids == set(episodes), "Score denominator differs from pinned game set")
    require(len(actual_ids) == len(card["environments"]) == len(summary["episodes"]), "Duplicate games")
    require(not summary["sdk_missing_games"], "SDK omitted games")
    require(summary["initial_reset_budget_units"] == summary["initial_reset_sdk_actions"] == 0,
            "Initial reset accounting differs")
    require(summary["network_disabled"] is True and summary["new_training"] is False,
            "Evaluation was not frozen and offline")
    rows = []
    for environment in card["environments"]:
        game = environment["id"]
        require(len(environment["runs"]) == 1, "Expected one run per game")
        run = environment["runs"][0]
        taken = run["level_actions"]
        values = baselines[game]
        expected = values["baseline_actions"] if isinstance(values, dict) else values
        require(run["level_baseline_actions"] == expected, f"Human baseline differs: {game}")
        require(sum(taken) == run["actions"], "Per-level actions do not sum to total")
        scores, value = score_levels(taken, expected, run["levels_completed"])
        require(len(run["level_scores"]) == len(scores), "Per-level score count differs")
        for actual, predicted in zip(run["level_scores"], scores):
            close(actual, predicted)
        close(run["score"], value)
        close(environment["score"], value)
        close(episodes[game]["score"], value)
        require(episodes[game]["level_budgets"] == [math.ceil(number * summary["multiplier"]) for number in expected],
                "Declared level budgets differ from pinned human baselines")
        require(environment["actions"] == episodes[game]["sdk_actions"] == run["actions"], "SDK action counts differ")
        require(environment["levels_completed"] == episodes[game]["completed_levels"] == run["levels_completed"],
                "SDK completion counts differ")
        rows.append({"game_id": game, "score": value, "actions": run["actions"],
                     "levels_completed": run["levels_completed"], "level_count": len(scores),
                     "completed": run["completed"]})
    mean = sum(row["score"] for row in rows) / len(rows)
    close(card["score"], mean)
    close(summary["sdk_scorecard_rhae_percent"], mean)
    close(summary["all_games"]["rhae_percent"], mean)
    require(card["total_environments"] == summary["all_games"]["games"] == len(rows), "Total game count differs")
    require(card["total_actions"] == sum(row["actions"] for row in rows), "Total actions differ")
    require(card["total_levels_completed"] == sum(row["levels_completed"] for row in rows), "Total completed levels differ")
    require(card["total_levels"] == sum(row["level_count"] for row in rows), "Total levels differ")
    transfers = [row for row in rows if not row["game_id"].startswith("ls20-")]
    transfer_score = sum(row["score"] for row in transfers) / len(transfers) if transfers else None
    if transfer_score is not None:
        close(summary["excluding_training_game"]["rhae_percent"], transfer_score)
    trace = verify_trace(directory, summary)
    return {"directory": str(directory), "policy": summary["policy"], "games": len(rows),
            "rhae_percent": mean, "excluding_ls20_rhae_percent": transfer_score,
            "completed_levels": card["total_levels_completed"], "levels": card["total_levels"],
            "actions": card["total_actions"], "trace": trace, "per_game": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--baselines", type=Path, default=HERE / "public-baselines.json")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints/ls20-10min")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baselines = json.loads(args.baselines.read_text())
    baselines = baselines.get("games", baselines)
    weights = verify_weights(args.checkpoint)
    runs = [inspect_run(directory, baselines) for directory in args.directories]
    for directory in args.directories:
        summary = json.loads((directory / "summary.json").read_text())
        if summary["worker_stats"]:
            require(summary["worker_stats"]["weight_sha256"] == weights, "Evaluated weights differ from checkpoint")
    report = {"status": "passed", "frozen_weight_sha256": weights, "runs": runs}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
