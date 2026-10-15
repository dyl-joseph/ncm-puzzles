#!/usr/bin/env python3
"""One bounded MLX policy-gradient pilot driven by actual public game transitions."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from .encoding import command_tokens, decode_prediction, encode_prefix
else:
    from encoding import command_tokens, decode_prediction, encode_prefix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HORIZON = 128
GAMMA = 0.99
TEMPERATURE = 1.5
BASELINE_DECAY = 0.9
SAVE_RESERVE_SECONDS = 25


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interrupt_on_signal(*_):
    raise KeyboardInterrupt


def transition_reward(before, after=None, *, invalid=False):
    if invalid and after is not None:
        raise ValueError("An invalid action cannot have a game response")
    completion = max(0, after["levels_completed"] - before["levels_completed"]) if after else 0
    components = {"level_completion": float(completion),
                  "invalid_action": -1.0 if invalid else 0.0,
                  "game_over": -1.0 if after and after["state"] == "GAME_OVER" else 0.0}
    return {**components, "total": sum(components.values())}


def discounted_weights(rewards, baseline_previous, gamma=GAMMA):
    if not math.isfinite(baseline_previous) or not 0 < gamma <= 1:
        raise ValueError("Invalid previous baseline or discount")
    if len(rewards) > HORIZON or any(type(r) not in (int, float) or not math.isfinite(r) for r in rewards):
        raise ValueError("Invalid episode reward vector")
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    weights = [gamma ** index * (value - baseline_previous) for index, value in enumerate(returns)]
    return returns, weights


def objective_from_log_probs(log_probs, weights):
    if len(log_probs) != len(weights):
        raise ValueError("Log probabilities and weights differ")
    return -sum(weight * value for weight, value in zip(weights, log_probs)) / HORIZON


@dataclass
class EpisodeBuffer:
    policy_version: int
    episode_index: int
    prefixes: list = field(default_factory=list)
    codes: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)

    def check_version(self, version):
        if type(version) is not int or version != self.policy_version:
            raise ValueError("Stale or malformed policy version")

    def append(self, prefix, codes, log_prob, version):
        self.check_version(version)
        if len(self.prefixes) >= HORIZON:
            raise ValueError("Episode exceeded the fixed decision horizon")
        self.prefixes.append(prefix)
        self.codes.append(codes)
        self.log_probs.append(log_prob)

    def validate_rewards(self, rewards, version):
        self.check_version(version)
        if not isinstance(rewards, list) or len(rewards) != len(self.prefixes):
            raise ValueError("Rewards do not match sampled decisions")
        return discounted_weights(rewards, 0.0)[0]


def create_training_environment(arcade, game_id, seed, card_id):
    environment = arcade.make(game_id, seed=seed, scorecard_id=card_id)
    if environment is None or environment.observation_space is None:
        raise RuntimeError("SDK failed to create the training environment")
    if environment.renderer is not None:
        raise RuntimeError("Training environment unexpectedly enabled rendering")
    return environment


class GameplayWorker:
    def __init__(self, args):
        self.error_file = (args.output / "worker.stderr.log").open("w")
        self.process = subprocess.Popen(
            [str(args.mlx_python), str(HERE / "gameplay_worker.py"),
             "--checkpoint", str(args.checkpoint.resolve()), "--output", str(args.output.resolve()),
             "--deadline", str(args.deadline), "--seed", str(args.seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_file,
            text=True, bufsize=1, cwd=ROOT)
        self.ready = self.read()
        if self.ready.get("type") != "ready":
            raise RuntimeError(f"MLX worker did not initialize: {self.ready}")

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"MLX worker exited with status {self.process.poll()}")
        response = json.loads(line)
        if response.get("type") == "error":
            raise RuntimeError(response["error"])
        return response

    def request(self, kind, **payload):
        self.process.stdin.write(json.dumps({"type": kind, **payload}) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        try:
            result = self.request("close")
            self.process.wait(timeout=10)
            if self.process.returncode:
                raise RuntimeError(f"MLX worker exit status {self.process.returncode}")
            return result
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.error_file.close()


def coordinator(args):
    import logging
    import socket
    import numpy as np
    import requests

    def offline_only(*unused_args, **unused_kwargs):
        raise RuntimeError("Network access is disabled during gameplay training")

    socket.socket.connect = offline_only
    requests.sessions.Session.request = offline_only
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    def view(observation):
        frame = np.asarray(observation.frame[-1], dtype=np.uint8)
        return {"frame": frame.tolist(), "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                "levels_completed": int(observation.levels_completed), "state": observation.state.name,
                "available_actions": sorted({int(action) for action in observation.available_actions})}

    logger = logging.getLogger("arc3-gameplay-training")
    logger.setLevel(logging.WARNING)
    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=str(args.environments_dir.resolve()),
                    recordings_dir=str(args.output / "recordings"), logger=logger)
    games = [game for game in arcade.get_environments() if game.game_id.split("-")[0] == "ls20"]
    if len(games) != 1:
        raise RuntimeError("Expected exactly one local LS20 training game")
    game = games[0]
    card_id = arcade.create_scorecard(tags=["local", "training-only", "on-policy"])
    worker = GameplayWorker(args)
    episodes, counts = [], Counter()
    last_save = time.monotonic()
    policy_version = worker.ready["policy_version"]
    worker_stats = None
    failure = None
    with (args.output / "training.jsonl").open("w") as training, (args.output / "game-trace.jsonl").open("w") as trace:
        def emit(stream, row):
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()

        emit(training, {"event": "worker_ready", **worker.ready})
        try:
            worker.request("save")
            while time.monotonic() < args.deadline - SAVE_RESERVE_SECONDS:
                index = len(episodes)
                seed = args.seed + index
                start = time.monotonic()
                environment = create_training_environment(arcade, game.game_id, seed, card_id)
                observation = environment.observation_space
                history, rewards = [], []
                initial = view(observation)
                worker.request("begin", episode_index=index, policy_version=policy_version)
                emit(trace, {"event": "initial_reset", "episode_index": index, "game_id": game.game_id,
                             "seed": seed, "policy_version": policy_version, "frame_after": initial,
                             "controller_reset": True, "renderer_is_none": True})
                termination = "horizon"
                executed = 0
                for step in range(HORIZON):
                    if time.monotonic() >= args.deadline - SAVE_RESERVE_SECONDS:
                        termination = "training_deadline"
                        break
                    before = view(observation)
                    prefix, info = encode_prefix(observation, history)
                    if prefix is None:
                        termination = "untrainable_prefix_capacity_overflow"
                        emit(trace, {"event": "untrainable_observation", "episode_index": index,
                                     "policy_version": policy_version, "frame_before": before, **info})
                        break
                    prediction = worker.request("act", prefix=prefix, policy_version=policy_version)
                    if prediction["type"] == "deadline":
                        termination = "training_deadline"
                        break
                    if prediction["type"] != "prediction":
                        raise RuntimeError(f"Unexpected worker prediction: {prediction}")
                    decision = decode_prediction(prediction["tokens"], before["available_actions"])
                    row = {"episode_index": index, "step": step, "game_id": game.game_id,
                           "seed": seed, "policy_version": policy_version,
                           "frame_before": before, "inference": prediction, **info}
                    if decision["type"] != "action":
                        reward = transition_reward(before, invalid=True)
                        rewards.append(reward["total"])
                        counts["invalid_actions"] += 1
                        termination = "invalid_action"
                        emit(trace, {"event": "invalid_action", **row, "failure": decision,
                                     "reward": reward, "frame_after": None})
                        break
                    action = {"action_id": decision["action_id"], "data": decision["data"]}
                    after = environment.step(GameAction.from_id(action["action_id"]), data=action["data"])
                    if after is None:
                        raise RuntimeError("SDK step returned no observation")
                    executed += 1
                    after_view = view(after)
                    reward = transition_reward(before, after_view)
                    rewards.append(reward["total"])
                    counts["actual_game_actions"] += 1
                    counts["level_completion_rewards"] += int(reward["level_completion"])
                    counts["game_over_penalties"] += int(reward["game_over"] < 0)
                    emit(trace, {"event": "action", **row, "action": action,
                                 "frame_after": after_view, "reward": reward})
                    if action["action_id"] == 0 or after_view["levels_completed"] != before["levels_completed"]:
                        history = []
                    else:
                        history = (history + [command_tokens({"id": action["action_id"], "data": action["data"]})])[-4:]
                    observation = after
                    if after_view["state"] in ("WIN", "GAME_OVER"):
                        termination = after_view["state"].lower()
                        break
                update = worker.request("update", policy_version=policy_version, rewards=rewards,
                                        termination=termination)
                if update["type"] != "updated":
                    raise RuntimeError(f"Unexpected worker update: {update}")
                policy_version = update["policy_version"]
                episode = {"episode_index": index, "seed": seed, "game_id": game.game_id,
                           "termination": termination, "executed_actions": executed,
                           "sampled_decisions": len(rewards), "reward_sum": sum(rewards),
                           "final_levels_completed": int(observation.levels_completed),
                           "wall_seconds": time.monotonic() - start, "update": update}
                episodes.append(episode)
                emit(trace, {"event": "episode_end", **episode})
                emit(training, {"event": "episode_end", **episode})
                write_json(args.output / "status.json", {"status": "training", "episodes": len(episodes),
                                                         "policy_version": policy_version, **counts})
                print(json.dumps({"episode": index, "termination": termination, "steps": executed,
                                  "reward": sum(rewards), "updates": update["optimizer_updates"]}), flush=True)
                if time.monotonic() - last_save >= 60 and time.monotonic() < args.deadline - SAVE_RESERVE_SECONDS:
                    worker.request("save")
                    last_save = time.monotonic()
                if termination in ("training_deadline", "untrainable_prefix_capacity_overflow"):
                    break
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            emit(training, {"event": "failure", "error": failure})
            if worker.process.poll() is None:
                worker.request("discard")
        finally:
            try:
                try:
                    if time.monotonic() < args.deadline - 3:
                        saved = worker.request("save")
                        emit(training, {"event": "checkpoint_saved", **saved})
                finally:
                    worker_stats = worker.close()
            finally:
                scorecard = arcade.close_scorecard(card_id)
                if scorecard is not None:
                    write_json(args.output / "training-scorecard.json", scorecard.model_dump(mode="json", exclude={"api_key"}))
    summary = {"status": "failed" if failure else "completed", "error": failure,
               "training_game": game.game_id, "training_only_not_benchmark": True,
               "network_disabled": True, "renderer_is_none": True, "episodes": episodes,
               "counts": dict(counts), "worker_stats": worker_stats,
               "policy_version": policy_version, "max_seconds": args.max_seconds,
               "checkpoint": str(args.output / "checkpoint"), "initial_checkpoint": str(args.checkpoint.resolve()),
               "game_source_sha256": {path.name: digest(path) for path in Path(game.local_dir).glob("*")
                                      if path.suffix in (".json", ".py")}}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "status.json", {"status": summary["status"], "episodes": len(episodes),
                                             "policy_version": policy_version, **counts})
    if failure:
        raise RuntimeError(failure)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints/ls20-10min")
    result.add_argument("--environments-dir", type=Path, default=ROOT / "data/arcagi3/evaluation_public/environment_files")
    result.add_argument("--mlx-python", type=Path, required=True)
    result.add_argument("--max-seconds", type=float, default=600)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    return result


def parse_args(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not math.isfinite(args.max_seconds) or not 30 <= args.max_seconds <= 600:
        cli.error("--max-seconds must be between 30 and 600 inclusive")
    if args.coordinator and (args.deadline is None or not math.isfinite(args.deadline)):
        cli.error("The coordinator needs its parent's absolute deadline")
    return args


def main():
    args = parse_args()
    if args.coordinator:
        coordinator(args)
        return
    start = time.monotonic()
    deadline = start + args.max_seconds
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", {
        "max_seconds": args.max_seconds, "seed": args.seed, "horizon": HORIZON,
        "gamma": GAMMA, "temperature": TEMPERATURE, "learning_rate": 1e-4,
        "microbatch_size": 4, "gradient_clip_norm": 1.0, "baseline_decay": BASELINE_DECAY,
        "fixed_loss_normalizer": HORIZON, "baseline": "EWMA of previous episode returns only",
        "objective": "-sum_t(gamma^t*(G_t-baseline_previous)*sum_22_code_log_probabilities)/128",
        "reward": {"visible_level_completion": 1, "invalid_action": -1, "visible_game_over": -1},
        "training_backend": "MLX GPU", "cpu_neural_training": False, "duty_target": 0.8,
        "duty_is_soft_pacing": True, "frozen_tokenizer": True, "checkpoint": str(args.checkpoint.resolve()),
        "environments_dir": str(args.environments_dir.resolve())})
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
               "--coordinator", "--deadline", str(deadline)]
    process = subprocess.Popen(command, start_new_session=True)
    timed_out = False
    interrupted = False
    previous_handler = signal.signal(signal.SIGTERM, interrupt_on_signal)
    try:
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    except BaseException:
        interrupted = True
        raise
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait()
        signal.signal(signal.SIGTERM, previous_handler)
        elapsed = time.monotonic() - start
        write_json(args.output / "deadline-result.json", {
            "max_seconds": args.max_seconds, "wall_seconds": elapsed, "hard_timeout": timed_out,
            "interrupted": interrupted, "training_process_group_stopped": True, "exit_code": process.returncode,
            "includes_initialization_sampling_gradient_work_save_shutdown": True})
    if process.returncode:
        raise SystemExit(process.returncode if process.returncode > 0 else 1)


if __name__ == "__main__":
    main()
