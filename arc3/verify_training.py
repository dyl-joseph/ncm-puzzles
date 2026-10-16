"""Verify real-game rewards, on-policy updates, checkpoint changes and the time limit."""

import argparse
from collections import Counter
import hashlib
import json
import lzma
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frame(view):
    frame = view["frame"]
    require(len(frame) == 64 and all(len(row) == 64 for row in frame), "Invalid frame shape")
    require(all(type(value) is int and 0 <= value < 16 for row in frame for value in row), "Invalid palette")
    require(hashlib.sha256(bytes(value for row in frame for value in row)).hexdigest() == view["frame_sha256"],
            "Frame checksum differs")


def verify(directory, initial_checkpoint):
    summary = json.loads((directory / "summary.json").read_text())
    deadline = json.loads((directory / "deadline-result.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    checkpoint = directory / "checkpoint"
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    require(summary["status"] == "completed" and summary["error"] is None, "Training failed")
    require(not deadline["hard_timeout"] and deadline["exit_code"] == 0, "Training did not shut down cleanly")
    require(deadline["wall_seconds"] <= deadline["max_seconds"] <= 600, "Training exceeded its time limit")
    require(summary["network_disabled"] and summary["renderer_is_none"], "Training environment contract differs")
    require(config["horizon"] == config["fixed_loss_normalizer"] == 128, "Horizon or loss normalization differs")
    require(config["gamma"] == 0.99 and config["temperature"] == 1.5, "Policy configuration differs")
    stats = summary["worker_stats"]
    require(stats["device"] == "Device(gpu, 0)", "Neural training did not use MLX GPU")
    require(not stats["torch_imported"] and not stats["cpu_neural_training"] and not stats["cpu_neural_inference"],
            "Unexpected neural backend")
    require(stats["duty_target"] == 0.8 and stats["duty_is_soft_pacing"], "GPU pacing differs")
    require(stats["optimizer_updates"] > 0 and stats["gradient_microbatches"] > 0, "No optimizer work happened")
    require(math.isfinite(stats["first_nonzero_gradient_norm"]) and stats["first_nonzero_gradient_norm"] > 0,
            "No finite nonzero gradient was observed")
    parity = stats["sampling_parity"]
    require(parity["before_any_optimizer_update"] and parity["code_count"] == 22 and
            parity["max_logit_error"] <= 1e-4 and parity["joint_log_probability_error"] <= 1e-3,
            "Staged sampling and training likelihood differ")
    for name, expected in provenance["initial_weight_sha256"].items():
        require(checksum(initial_checkpoint / name) == expected, "Initial checkpoint differs")
    for name, expected in provenance["weight_sha256"].items():
        require(checksum(checkpoint / name) == expected, "Learned checkpoint differs")
    require(provenance["initial_weight_sha256"]["tokenizer.safetensors"] ==
            provenance["weight_sha256"]["tokenizer.safetensors"], "Frozen tokenizer changed")
    require(provenance["initial_weight_sha256"]["generator.safetensors"] !=
            provenance["weight_sha256"]["generator.safetensors"] and
            provenance["generator_parameter_delta_l2"] > 0 and provenance["generator_changed_tensor_count"] > 0,
            "Generator parameters did not change")
    counts = Counter()
    baseline, version = 0.0, 0
    last, rewards, behavior, current = None, [], [], None
    path = directory / "game-trace.jsonl"
    stream = path.open() if path.exists() else lzma.open(directory / "game-trace.jsonl.xz", "rt")
    with stream:
        for line in stream:
            row = json.loads(line)
            event = row["event"]
            if event == "initial_reset":
                require(current is None and row["episode_index"] == counts["episodes"], "Episode ordering differs")
                require(row["policy_version"] == version and row["renderer_is_none"], "Episode policy or renderer differs")
                current = row["episode_index"]
                last, rewards, behavior = row["frame_after"], [], []
                check_frame(last)
            elif event in ("action", "invalid_action"):
                require(row["episode_index"] == current and row["step"] == len(rewards), "Decision order differs")
                require(row["policy_version"] == version and row["frame_before"] == last, "On-policy frame chain differs")
                inference = row["inference"]
                require(inference["policy_version"] == version and len(inference["sampled_codes"]) == 22 and
                        all(type(value) is int and 0 <= value < 64 for value in inference["sampled_codes"]),
                        "Sampled hierarchy differs")
                log_probability = inference["behavior_joint_log_probability"]
                require(math.isfinite(log_probability) and log_probability <= 0, "Invalid behavior likelihood")
                if event == "action":
                    after = row["frame_after"]
                    check_frame(after)
                    action = row["action"]
                    require(action["action_id"] in last["available_actions"], "Unavailable command executed")
                    packet = [24 + action["action_id"], 1, 1]
                    if action["action_id"] == 6:
                        packet = [30, 32 + action["data"]["x"], 32 + action["data"]["y"]]
                    require(inference["tokens"][:3] == packet, "Executed action differs from sampled NCM output")
                    completed = max(0, after["levels_completed"] - last["levels_completed"])
                    death = int(after["state"] == "GAME_OVER")
                    expected = {"level_completion": float(completed), "invalid_action": 0.0,
                                "game_over": -float(death), "total": float(completed - death)}
                    counts["actual_game_actions"] += 1
                    counts["level_completion_rewards"] += completed
                    counts["game_over_penalties"] += death
                    last = after
                else:
                    require(row["frame_after"] is None, "Invalid action has a fabricated environment response")
                    expected = {"level_completion": 0.0, "invalid_action": -1.0, "game_over": 0.0, "total": -1.0}
                    counts["invalid_actions"] += 1
                require(row["reward"] == expected, "Reward did not come from the visible game outcome")
                rewards.append(expected["total"])
                behavior.append(log_probability)
                counts["sampled_decisions"] += 1
            elif event == "untrainable_observation":
                require(row["frame_before"] == last and row["prefix_length"] > 1142, "Invalid overflow report")
            elif event == "episode_end":
                require(row["episode_index"] == current and row["sampled_decisions"] == len(rewards) <= 128,
                        "Episode sample count differs")
                update = row["update"]
                require(update["behavior_policy_version"] == version and abs(update["baseline_previous"] - baseline) < 1e-10,
                        "Policy update used a stale version or current-episode baseline")
                returns, running = [0.0] * len(rewards), 0.0
                for index in range(len(rewards) - 1, -1, -1):
                    running = rewards[index] + 0.99 * running
                    returns[index] = running
                weights = [0.99 ** index * (value - baseline) for index, value in enumerate(returns)]
                require(len(update["discounted_returns"]) == len(returns) == len(update["policy_weights"]) and
                        all(abs(a - b) < 1e-10 for a, b in zip(update["discounted_returns"], returns)) and
                        all(abs(a - b) < 1e-10 for a, b in zip(update["policy_weights"], weights)),
                        "Discounted policy weights differ")
                if update["applied"]:
                    require(math.isfinite(update["gradient_norm"]) and update["gradient_norm"] > 0,
                            "Applied update had no finite gradient")
                    expected_loss = -sum(weight * value for weight, value in zip(weights, behavior)) / 128
                    require(abs(update["loss"] - expected_loss) <= 1e-3 * max(1, abs(expected_loss)),
                            "Training loss differs from the sampled on-policy likelihood")
                    version += 1
                require(update["policy_version"] == version, "Policy version increment differs")
                if returns:
                    baseline = 0.9 * baseline + (1 - 0.9) * sum(returns) / len(returns)
                require(abs(update["baseline_next"] - baseline) < 1e-12, "Prior-only baseline update differs")
                require(row["final_levels_completed"] == last["levels_completed"], "Episode completion count differs")
                counts["episodes"] += 1
                current = None
            else:
                raise AssertionError(f"Unknown training trace event: {event}")
    require(current is None, "An episode has no ending")
    require(version == stats["optimizer_updates"] == summary["policy_version"], "Optimizer update totals differ")
    require(counts["sampled_decisions"] == stats["sample_calls"], "Sampled decision totals differ")
    for name, value in summary["counts"].items():
        require(counts[name] == value, f"Training outcome count differs: {name}")
    return {"status": "passed", "wall_seconds": deadline["wall_seconds"], "counts": dict(counts),
            "optimizer_updates": version, "generator_parameter_delta_l2": provenance["generator_parameter_delta_l2"],
            "frozen_tokenizer_unchanged": True, "real_frame_chains_verified": True,
            "visible_rewards_verified": True, "on_policy_losses_verified": True,
            "neural_backend": stats["device"], "process_duty": stats["observed_process_duty"],
            "scope": "LS20 gameplay training evidence; not a held-out benchmark or Kaggle score"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path, default=HERE / "checkpoints/ls20-10min")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory, args.initial_checkpoint)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
