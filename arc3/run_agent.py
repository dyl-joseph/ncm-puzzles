#!/usr/bin/env python3
"""Evaluate frozen policies on pinned public ARC-AGI-3 games offline."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BENCHMARK_REFERENCE_COMMIT = "d7f6961bf19738e1e8ffd5d3a982f4e0de138f79"

if __package__:
    from .encoding import command_tokens, encode_prefix, decode_prediction
else:
    from encoding import command_tokens, encode_prefix, decode_prediction


def no_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled during this local evaluation")


socket.socket.connect = no_network
requests.sessions.Session.request = no_network

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

MAX_RUNTIME_SECONDS = 12 * 60 * 60


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()
                if key.lower() not in ("api_key", "arc_api_key", "x-api-key")}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def state_name(observation):
    return observation.state.name


def observation_view(observation):
    frame = np.asarray(observation.frame[-1], dtype=np.uint8)
    return {"frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "levels_completed": int(observation.levels_completed),
            "state": state_name(observation),
            "available_actions": sorted({int(action) for action in observation.available_actions}),
            "frame": frame.tolist()}



class MLXWorker:
    def __init__(self, output, mlx_python, checkpoint):
        self.error_file = (output / "worker.stderr.log").open("w")
        self.process = subprocess.Popen(
            [str(mlx_python), str(HERE / "mlx_worker.py"), "--checkpoint", str(checkpoint.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_file,
            text=True, bufsize=1, cwd=ROOT)
        self.ready = self.read()
        if self.ready.get("type") != "ready":
            raise RuntimeError(f"MLX worker did not initialize: {self.ready}")

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"MLX worker exited with status {self.process.poll()}")
        return json.loads(line)

    def infer(self, prefix):
        self.process.stdin.write(json.dumps({"type": "infer", "prefix": prefix}) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        try:
            self.process.stdin.write('{"type":"close"}\n')
            self.process.stdin.flush()
            result = self.read()
            self.process.wait(timeout=30)
            if self.process.returncode != 0:
                raise RuntimeError(f"MLX worker exit status {self.process.returncode}")
            return result
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.error_file.close()


def verify_adapter(worker):
    from types import SimpleNamespace
    fixture = json.loads((HERE / "fixtures/replay_controls.json").read_text())
    checks = []
    for control in fixture["controls"]:
        observation = SimpleNamespace(frame=[control["frame"]],
                                      levels_completed=control["levels_completed"],
                                      available_actions=control["available_actions"])
        prefix, info = encode_prefix(observation, control["history"])
        if prefix != control["prefix"] or info["prefix_sha256"] != control["prefix_sha256"]:
            raise AssertionError(f"Prefix differs at replay control {control['index']}")
        result = worker.infer(prefix)
        if result.get("tokens") != control["expected_tokens"]:
            raise AssertionError(f"Prediction differs at replay control {control['index']}")
        checks.append({"index": control["index"], "event_guid": control["event_guid"],
                       "prefix_exact": True, "prediction_exact": True})
    return {"status": "PASS", "checks": checks, "cold_start_history": [[0, 0, 0]] * 4,
            "fixture_sha256": digest(HERE / "fixtures/replay_controls.json")}


def baseline_action(policy, observation, rng, previous):
    available = sorted({int(action) for action in observation.available_actions} - {0})
    if not available:
        return {"type": "policy_failure", "reason": "no_nonreset_action_available"}
    if policy == "random":
        action = rng.choice(available)
        payload = {"x": rng.randrange(64), "y": rng.randrange(64)} if action == 6 else {}
        return {"type": "action", "action_id": action, "data": payload}
    if previous is None:
        action = available[0]
        return {"type": "action", "action_id": action,
                "data": {"x": 32, "y": 32} if action == 6 else {}}
    if previous["action_id"] not in available:
        return {"type": "policy_failure", "reason": "repeated_action_unavailable"}
    return previous.copy()


def play(arcade, info, card_id, args, worker, emit):
    start = time.monotonic()
    episode = {"game_id": info.game_id, "termination": "unknown", "steps": 0,
               "harness_actions": 0, "sdk_actions": None, "completed_levels": 0,
               "score": None, "policy_calls": 0, "forced_resets": 0,
               "level_budgets": [math.ceil(value * args.multiplier) for value in info.baseline_actions]}
    try:
        environment = arcade.make(info.game_id, seed=args.seed, scorecard_id=card_id)
        if environment is None or environment.observation_space is None:
            raise RuntimeError("SDK failed to initialize/reset the environment")
        observation = environment.observation_space
        before = observation_view(observation)
        emit({"event": "initial_reset", "game_id": info.game_id, "action": {"action_id": 0, "data": {}},
              "frame_before": None, "frame_after": before,
              "harness_actions": 0, "level_budget_used": 0})
        level = int(observation.levels_completed)
        level_actions = 0
        history = []
        previous = None
        rng = random.Random(f"{args.seed}:{info.game_id}")
        while True:
            if state_name(observation) == "WIN":
                episode["termination"] = "win"
                break
            if time.monotonic() - start >= MAX_RUNTIME_SECONDS:
                episode["termination"] = "time_budget"
                break
            if level < len(episode["level_budgets"]) and level_actions >= episode["level_budgets"][level]:
                episode["termination"] = "level_action_budget"
                break
            before = observation_view(observation)
            decision_start = time.monotonic()
            info_row = {}
            forced_reset = state_name(observation) in ("GAME_OVER", "NOT_PLAYED")
            if forced_reset:
                result = {"type": "action", "action_id": 0, "data": {}}
                episode["forced_resets"] += 1
            elif args.policy == "ncm":
                prefix, info_row = encode_prefix(observation, history)
                if prefix is None:
                    result = info_row
                else:
                    prediction = worker.infer(prefix)
                    episode["policy_calls"] += 1
                    info_row = {**info_row, "inference": prediction}
                    result = (decode_prediction(prediction.get("tokens"), before["available_actions"])
                              if prediction.get("type") == "prediction" else prediction)
            else:
                result = baseline_action(args.policy, observation, rng, previous)
                episode["policy_calls"] += 1
            decision_seconds = time.monotonic() - decision_start
            if result["type"] != "action":
                episode["termination"] = "policy_failure"
                episode["failure"] = result
                emit({"event": "policy_failure", "game_id": info.game_id,
                      "frame_before": before, **info_row, "failure": result,
                      "decision_seconds": decision_seconds})
                break
            action = {"action_id": result["action_id"], "data": result["data"]}
            step_start = time.monotonic()
            after = environment.step(GameAction.from_id(action["action_id"]), data=action["data"])
            step_seconds = time.monotonic() - step_start
            if after is None:
                raise RuntimeError("SDK step returned no observation")
            episode["steps"] += 1
            episode["harness_actions"] += 1
            level_actions += 1
            next_level = int(after.levels_completed)
            if action["action_id"] == 0 or next_level != level:
                history = []
                previous = None
            else:
                history.append(command_tokens({"id": action["action_id"], "data": action["data"]}))
                history = history[-4:]
                previous = {"type": "action", **action}
            if next_level > level:
                level_actions = 0
            level = next_level
            observation = after
            episode["completed_levels"] = level
            emit({"event": "action", "game_id": info.game_id, "frame_before": before,
                  "frame_after": observation_view(after), "action": action,
                  "forced_reset": forced_reset, **info_row,
                  "harness_actions": episode["harness_actions"], "level_budget_used": level_actions,
                  "decision_seconds": decision_seconds, "step_seconds": step_seconds})
        episode["final_state"] = state_name(observation)
    except Exception as error:
        episode["termination"] = "sdk_or_runner_failure"
        episode["failure"] = {"type": type(error).__name__, "error": str(error)}
        emit({"event": "exception", "game_id": info.game_id, "failure": episode["failure"]})
    episode["wall_seconds"] = time.monotonic() - start
    emit({"event": "episode_end", **episode})
    return episode


def totals(episodes):
    return {"games": len(episodes), "rhae_percent": sum(row["score"] or 0 for row in episodes) / len(episodes) if episodes else None,
            "completed_games": sum(row.get("final_state") == "WIN" for row in episodes),
            "completed_levels": sum(row["completed_levels"] for row in episodes),
            "sdk_actions": sum(row["sdk_actions"] or 0 for row in episodes),
            "termination_counts": dict(Counter(row["termination"] for row in episodes))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=("ncm", "random", "repeat"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multiplier", type=float, default=5)
    parser.add_argument("--games", nargs="*", help="Explicit smoke subset, e.g. --games ls20")
    parser.add_argument("--baselines", type=Path, default=HERE / "public-baselines.json",
                        help="Pinned scorer-only public human action baselines")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints/ls20-10min")
    parser.add_argument("--mlx-python", type=Path, default=Path(sys.executable),
                        help="Python interpreter with MLX installed; used only for the NCM policy")
    parser.add_argument("--environments-dir", type=Path,
                        default=ROOT / "data/arcagi3/evaluation_public/environment_files",
                        help="Local public game sources; versions and file hashes are recorded")
    args = parser.parse_args()
    if args.multiplier <= 0:
        parser.error("--multiplier must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists() or (args.output / "trace.jsonl").exists():
        parser.error("Output already contains a run; use a new directory")
    logger = logging.getLogger("arc3-local-evaluation")
    logger.setLevel(logging.WARNING)
    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=str(args.environments_dir.resolve()),
                    recordings_dir=str(args.output / "recordings"), logger=logger)
    all_games = sorted(arcade.get_environments(), key=lambda item: item.game_id)
    games = [item for item in all_games if args.games is None or
             any(item.game_id == game or item.game_id.split("-")[0] == game for game in args.games)]
    if not games or (args.games is None and len(games) != 25):
        raise RuntimeError(f"Expected pinned game set, found {len(games)}")
    baseline_path = args.baselines.resolve()
    baseline_provenance = json.loads(baseline_path.read_text())
    overrides = baseline_provenance.get("games", baseline_provenance)
    if args.games is None and {game.game_id for game in games} != set(overrides):
        raise RuntimeError("Downloaded game versions differ from the pinned public baseline set")
    for game in games:
        override = overrides[game.game_id]
        values = override["baseline_actions"] if isinstance(override, dict) else override
        if not isinstance(values, list) or not values or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError(f"Invalid baseline override for {game.game_id}")
        game.baseline_actions = values
    card_id = arcade.create_scorecard(tags=["local", "frozen-ncm-pilot", args.policy])
    start = time.monotonic()
    worker = MLXWorker(args.output, args.mlx_python, args.checkpoint) if args.policy == "ncm" else None
    worker_stats = None
    episodes = []
    with (args.output / "trace.jsonl").open("w") as stream:
        def emit(row):
            stream.write(json.dumps(sanitize(row), separators=(",", ":")) + "\n")
            stream.flush()
        try:
            if worker:
                write_json(args.output / "adapter_checks.json", verify_adapter(worker))
            for game in games:
                episode = play(arcade, game, card_id, args, worker, emit)
                episodes.append(episode)
                print(json.dumps({key: episode[key] for key in ("game_id", "termination", "steps", "completed_levels")}), flush=True)
        finally:
            if worker:
                worker_stats = worker.close()
    scorecard = arcade.close_scorecard(card_id)
    if scorecard is None:
        raise RuntimeError("SDK did not close a scorecard")
    card = sanitize(scorecard.model_dump(mode="json"))
    write_json(args.output / "scorecard.json", card)
    by_game = {row["id"]: row for row in card["environments"]}
    missing_games = []
    for episode in episodes:
        official = by_game.get(episode["game_id"])
        if official is None:
            missing_games.append(episode["game_id"])
            episode["score"] = 0.0
            continue
        episode["score"] = official["score"]
        episode["sdk_actions"] = official["actions"]
        episode["completed_levels"] = official["levels_completed"]
        episode["official_runs"] = official["runs"]
    def source_name(path):
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path.resolve())
    hashes = {}
    for game in games:
        for path in sorted(Path(game.local_dir).glob("*")):
            if path.suffix in (".json", ".py"):
                hashes[source_name(path)] = digest(path)
    for path in [Path(__file__), HERE / "mlx_worker.py", HERE / "mlx_ncm.py", HERE / "encoding.py",
                 HERE / "verify_score.py", HERE / "download_games.py", HERE / "fixtures/replay_controls.json",
                 baseline_path]:
        hashes[source_name(path)] = digest(path)
    if worker:
        for name in ("tokenizer.safetensors", "generator.safetensors", "metadata.json", "provenance.json"):
            path = args.checkpoint / name
            hashes[source_name(path)] = digest(path)
    summary = {"policy": args.policy,
               "policy_description": {"ncm": "frozen greedy NCM, execute first command and replan",
                                      "random": "uniform available nonreset action; uniform x/y for clicks",
                                      "repeat": "repeat own command; initialize smallest legal action, center click"}[args.policy],
               "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
               "subset_smoke": args.games is not None, "network_disabled": True,
               "multiplier": args.multiplier, "max_runtime_seconds_per_game": MAX_RUNTIME_SECONDS,
               "baseline_source": {"path": str(baseline_path), "sha256": digest(baseline_path),
                                   "provenance": baseline_provenance},
               "benchmark_reference_commit": BENCHMARK_REFERENCE_COMMIT,
               "initial_reset_budget_units": 0, "initial_reset_sdk_actions": 0,
               "no_progress_stop": False, "new_training": False,
               "training_game": "ls20", "cold_start_history": "four PAD command packets; clear on resets and level changes",
               "runtime_versions": {name: importlib.metadata.version(name) for name in ("arc-agi", "arcengine", "numpy")},
               "episodes": episodes, "all_games": totals(episodes),
               "excluding_training_game": totals([row for row in episodes if not row["game_id"].startswith("ls20-")]),
               "sdk_scorecard_rhae_percent": card["score"], "sdk_missing_games": missing_games,
               "worker_stats": worker_stats, "wall_seconds": time.monotonic() - start,
               "source_and_weight_sha256": hashes,
               "environments_dir": str(args.environments_dir.resolve()),
               "comparison_limits": ["Public game versions are declared and hashed; this is not the semi-private leaderboard set",
                                     "LS20 was the training game; other 24 games are separately reported",
                                     "A frozen policy pilot; no claim of official leaderboard verification"]}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["all_games"], sort_keys=True), flush=True)
    if missing_games or card["total_environments"] != len(games):
        raise RuntimeError("SDK scorecard denominator mismatch; inspect explicit failures")


if __name__ == "__main__":
    main()
