#!/usr/bin/env python3
"""GPU-only NCM sampling and one on-policy policy-gradient update per episode."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

if __package__:
    from .mlx_ncm import Generator, Tokenizer, LEVELS, WIDTH, area_down, linear_up, validate_device
    from .train_gameplay import (BASELINE_DECAY, EpisodeBuffer, HORIZON, SAVE_RESERVE_SECONDS,
                                 TEMPERATURE, digest, discounted_weights, write_json)
else:
    from mlx_ncm import Generator, Tokenizer, LEVELS, WIDTH, area_down, linear_up, validate_device
    from train_gameplay import (BASELINE_DECAY, EpisodeBuffer, HORIZON, SAVE_RESERVE_SECONDS,
                                TEMPERATURE, digest, discounted_weights, write_json)


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def joint_log_probability(logits, codes):
    normalized = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    selected = mx.take_along_axis(normalized, codes[..., None], axis=-1).squeeze(-1)
    return mx.sum(selected, axis=-1)


class GameplayTrainer:
    def __init__(self, args):
        self.args = args
        self.checkpoint = args.output / "checkpoint"
        self.checkpoint.mkdir(parents=True)
        self.metadata = json.loads((args.checkpoint / "metadata.json").read_text())
        if self.metadata["vocab_size"] != 160 or self.metadata["prefix_len"] != 1142:
            raise ValueError("Checkpoint differs from the unchanged LS20 adapter")
        mx.set_default_device(mx.gpu)
        validate_device()
        mx.random.seed(args.seed)
        start = time.monotonic()
        self.tokenizer = Tokenizer(self.metadata["vocab_size"])
        self.tokenizer.load_weights(str(args.checkpoint / "tokenizer.safetensors"))
        self.tokenizer.eval()
        self.tokenizer.freeze()
        self.generator = Generator(self.metadata["vocab_size"], self.metadata["prefix_len"], self.tokenizer)
        self.generator.load_weights(str(args.checkpoint / "generator.safetensors"))
        self.generator.train()
        self.optimizer = optim.Adam(learning_rate=1e-4)
        self.loss_and_grad = nn.value_and_grad(self.generator, self.loss)
        mx.eval(self.tokenizer.parameters(), self.generator.parameters())
        mx.synchronize(mx.gpu)
        self.initial_parameters = dict(tree_flatten(self.generator.parameters()))
        self.initial_hashes = {name: digest(args.checkpoint / name)
                               for name in ("tokenizer.safetensors", "generator.safetensors")}
        shutil.copyfile(args.checkpoint / "tokenizer.safetensors", self.checkpoint / "tokenizer.safetensors")
        self.policy_version = 0
        self.baseline = 0.0
        self.buffer = None
        self.parity = None
        self.stats = {"device": str(mx.default_device()), "mlx_version": importlib.metadata.version("mlx"),
                      "cpu_neural_training": False, "cpu_neural_inference": False,
                      "torch_imported": "torch" in sys.modules, "duty_target": 0.8,
                      "duty_is_soft_pacing": True, "active_seconds": 0.0, "idle_seconds": 0.0,
                      "sample_calls": 0, "gradient_microbatches": 0, "optimizer_updates": 0,
                      "reward_driven_updates": 0, "baseline_only_updates": 0,
                      "zero_gradient_episodes": 0, "discarded_partial_updates": 0,
                      "aborted_episode_decisions": 0,
                      "first_nonzero_gradient_norm": None, "latest_gradient_norm": None,
                      "saved_checkpoints": 0, "inference_control_calls": 0,
                      "initial_weight_sha256": self.initial_hashes}
        self.pace(start)

    def pace(self, start):
        mx.synchronize(mx.gpu)
        active = time.monotonic() - start
        idle_start = time.monotonic()
        delay = min(active * 0.25, max(0.0, self.args.deadline - idle_start - 1))
        time.sleep(delay)
        idle = time.monotonic() - idle_start
        self.stats["active_seconds"] += active
        self.stats["idle_seconds"] += idle
        return {"active_seconds": active, "idle_seconds": idle}

    def has_time(self, reserve=SAVE_RESERVE_SECONDS):
        return time.monotonic() < self.args.deadline - reserve

    def begin(self, request):
        if self.buffer is not None:
            raise ValueError("Previous episode has not been finished")
        if type(request["policy_version"]) is not int or request["policy_version"] != self.policy_version:
            raise ValueError("Stale or malformed policy version")
        self.buffer = EpisodeBuffer(self.policy_version, request["episode_index"])
        return {"type": "begun", "policy_version": self.policy_version}

    def act(self, request):
        if self.buffer is None:
            raise ValueError("Sampling needs an active episode")
        self.buffer.check_version(request["policy_version"])
        prefix = request["prefix"]
        if (not isinstance(prefix, list) or len(prefix) != self.metadata["prefix_len"] or
                any(type(token) is not int or not 0 <= token < self.metadata["vocab_size"] for token in prefix)):
            raise ValueError("Malformed padded prefix")
        if not self.has_time():
            return {"type": "deadline", "policy_version": self.policy_version}
        start = time.monotonic()
        inputs = mx.array([prefix], dtype=mx.int32)
        z = mx.zeros((1, LEVELS[-1], WIDTH))
        chunk = mx.broadcast_to(self.generator.level0_query, (1, 1, WIDTH))
        codes, staged_logits = [], []
        for level, length in enumerate(LEVELS):
            logits = self.generator._logits(inputs, [chunk], [level])[0] / TEMPERATURE
            ids = mx.random.categorical(logits, axis=-1).astype(mx.int32)
            codes.append(ids)
            staged_logits.append(logits)
            book = mx.stop_gradient(self.tokenizer.quantizer.codebooks[level])
            z = z + linear_up(book[ids], LEVELS[-1])
            if level + 1 < len(LEVELS):
                chunk = area_down(z, LEVELS[level + 1])
        sampled_codes = mx.concatenate(codes, axis=1)
        all_logits = mx.concatenate(staged_logits, axis=1)
        log_probability = joint_log_probability(all_logits, sampled_codes)
        tokens = mx.argmax(self.tokenizer.decode(sampled_codes), axis=-1)
        if self.parity is None:
            teacher_logits = mx.concatenate(self.generator.teacher_logits(inputs, sampled_codes), axis=1) / TEMPERATURE
            logit_error = mx.max(mx.abs(teacher_logits - all_logits))
            probability_error = mx.max(mx.abs(joint_log_probability(teacher_logits, sampled_codes) - log_probability))
            mx.eval(tokens, sampled_codes, log_probability, logit_error, probability_error)
            mx.synchronize(mx.gpu)
            self.parity = {"max_logit_error": float(logit_error.item()),
                           "joint_log_probability_error": float(probability_error.item()),
                           "code_count": sum(LEVELS), "temperature": TEMPERATURE,
                           "before_any_optimizer_update": self.stats["optimizer_updates"] == 0}
            if self.parity["max_logit_error"] > 1e-4 or self.parity["joint_log_probability_error"] > 1e-3:
                raise AssertionError(f"Sampling and teacher-forced policy differ: {self.parity}")
        else:
            mx.eval(tokens, sampled_codes, log_probability)
            mx.synchronize(mx.gpu)
        raw_codes = sampled_codes.tolist()[0]
        raw_tokens = tokens.tolist()[0]
        behavior_log_prob = float(log_probability.item())
        if len(raw_codes) != 22 or not math.isfinite(behavior_log_prob):
            raise RuntimeError("Malformed sampled latent action")
        self.buffer.append(prefix, raw_codes, behavior_log_prob, self.policy_version)
        self.stats["sample_calls"] += 1
        timing = self.pace(start)
        return {"type": "prediction", "tokens": raw_tokens, "sampled_codes": raw_codes,
                "behavior_joint_log_probability": behavior_log_prob, "temperature": TEMPERATURE,
                "policy_version": self.policy_version, "sampling_parity": self.parity,
                "peak_mlx_memory_bytes": mx.get_peak_memory(), **timing}

    def loss(self, model, prefixes, codes, weights):
        logits = mx.concatenate(model.teacher_logits(prefixes, mx.stop_gradient(codes)), axis=1) / TEMPERATURE
        return -mx.sum(mx.stop_gradient(weights) * joint_log_probability(logits, codes)) / HORIZON

    def update(self, request):
        if self.buffer is None:
            raise ValueError("Updating needs an active episode")
        self.buffer.validate_rewards(request["rewards"], request["policy_version"])
        rewards = request["rewards"]
        baseline_previous = self.baseline
        returns, weights = discounted_weights(rewards, baseline_previous)
        result = {"type": "updated", "episode_index": self.buffer.episode_index,
                  "behavior_policy_version": self.policy_version, "baseline_previous": baseline_previous,
                  "discounted_returns": returns, "policy_weights": weights,
                  "reward_sum": sum(rewards), "fixed_loss_normalizer": HORIZON,
                  "gradient_norm": None, "loss": None, "applied": False,
                  "reason": "empty_episode" if not rewards else "zero_policy_weights"}
        accumulated = None
        loss_sum = 0.0
        if weights and any(weight != 0.0 for weight in weights):
            for offset in range(0, len(weights), 4):
                if not self.has_time():
                    self.stats["discarded_partial_updates"] += 1
                    accumulated = None
                    result["reason"] = "deadline_discarded_update"
                    break
                start = time.monotonic()
                prefixes = mx.array(self.buffer.prefixes[offset:offset + 4], dtype=mx.int32)
                codes = mx.array(self.buffer.codes[offset:offset + 4], dtype=mx.int32)
                advantages = mx.array(weights[offset:offset + 4], dtype=mx.float32)
                loss, gradients = self.loss_and_grad(self.generator, prefixes, codes, advantages)
                accumulated = gradients if accumulated is None else tree_map(lambda a, b: a + b, accumulated, gradients)
                mx.eval(loss, accumulated)
                mx.synchronize(mx.gpu)
                loss_sum += float(loss.item())
                self.stats["gradient_microbatches"] += 1
                self.pace(start)
            if accumulated is not None and self.has_time():
                if self.parity is None or not self.parity["before_any_optimizer_update"]:
                    raise RuntimeError("Gradient update requires the first-sample policy parity proof")
                start = time.monotonic()
                gradients, norm = optim.clip_grad_norm(accumulated, max_norm=1.0)
                mx.eval(norm, gradients)
                mx.synchronize(mx.gpu)
                norm_value = float(norm.item())
                if not math.isfinite(norm_value) or not math.isfinite(loss_sum):
                    raise RuntimeError("Nonfinite policy gradient")
                self.stats["latest_gradient_norm"] = norm_value
                result.update(gradient_norm=norm_value, loss=loss_sum)
                if norm_value > 0:
                    self.optimizer.update(self.generator, gradients)
                    mx.eval(self.generator.parameters(), self.optimizer.state)
                    mx.synchronize(mx.gpu)
                    self.policy_version += 1
                    self.stats["optimizer_updates"] += 1
                    key = "reward_driven_updates" if any(reward != 0 for reward in rewards) else "baseline_only_updates"
                    self.stats[key] += 1
                    if self.stats["first_nonzero_gradient_norm"] is None:
                        self.stats["first_nonzero_gradient_norm"] = norm_value
                    result.update(applied=True, reason="reward_policy_gradient")
                else:
                    self.stats["zero_gradient_episodes"] += 1
                    result["reason"] = "zero_actor_gradient"
                self.pace(start)
            elif accumulated is not None:
                self.stats["discarded_partial_updates"] += 1
                result["reason"] = "deadline_discarded_update"
        else:
            self.stats["zero_gradient_episodes"] += bool(weights)
        if returns:
            self.baseline = BASELINE_DECAY * baseline_previous + (1 - BASELINE_DECAY) * sum(returns) / len(returns)
        self.buffer = None
        return {**result, "policy_version": self.policy_version, "baseline_next": self.baseline,
                "optimizer_updates": self.stats["optimizer_updates"],
                "reward_driven_updates": self.stats["reward_driven_updates"]}

    def save(self):
        if self.buffer is not None:
            raise ValueError("Save only between complete episodes")
        start = time.monotonic()
        temporary = self.checkpoint / "generator.tmp.safetensors"
        self.generator.save_weights(str(temporary))
        temporary.replace(self.checkpoint / "generator.safetensors")
        write_json(self.checkpoint / "metadata.json", self.metadata)
        fixture_path = Path(__file__).resolve().parent / "fixtures/replay_controls.json"
        fixture = json.loads(fixture_path.read_text())
        predictions = []
        for control in fixture["controls"]:
            inputs = mx.array([control["prefix"]], dtype=mx.int32)
            tokens = mx.argmax(self.tokenizer.decode(self.generator.generate(inputs)), axis=-1)
            mx.eval(tokens)
            mx.synchronize(mx.gpu)
            predictions.append({"index": control["index"], "tokens": tokens.tolist()[0]})
        self.stats["inference_control_calls"] += len(predictions)
        final_parameters = dict(tree_flatten(self.generator.parameters()))
        changes = [mx.sum((value - self.initial_parameters[name]) ** 2) for name, value in final_parameters.items()]
        delta = mx.sqrt(mx.sum(mx.stack(changes)))
        changed = [mx.any(value != self.initial_parameters[name]).astype(mx.int32) for name, value in final_parameters.items()]
        changed_count = mx.sum(mx.stack(changed))
        mx.eval(delta, changed_count)
        mx.synchronize(mx.gpu)
        hashes = {name: digest(self.checkpoint / name) for name in self.initial_hashes}
        if hashes["tokenizer.safetensors"] != self.initial_hashes["tokenizer.safetensors"]:
            raise AssertionError("Frozen tokenizer hash changed")
        provenance = {"description": "Actual-game LS20 on-policy REINFORCE pilot, not a leaderboard submission",
                      "training_kind": "actual_game_policy_gradient", "training_game": "ls20",
                      "training_backend": "MLX GPU", "cpu_neural_training": False,
                      "maximum_training_seconds": 600, "frozen_tokenizer": True,
                      "initial_checkpoint": str(self.args.checkpoint.resolve()),
                      "initial_weight_sha256": self.initial_hashes, "weight_sha256": hashes,
                      "generator_parameter_delta_l2": float(delta.item()),
                      "generator_changed_tensor_count": int(changed_count.item()),
                      "policy_version": self.policy_version, "sampling_temperature": TEMPERATURE,
                      "sampling_parity": self.parity, "baseline": self.baseline,
                      "objective": "REINFORCE sum gamma^t*(G_t-baseline_previous)*joint_22_code_log_probability /128",
                      "reward": "visible completed levels; -1 invalid action; -1 visible GAME_OVER",
                      "stats": self.stats.copy()}
        write_json(self.checkpoint / "provenance.json", provenance)
        write_json(self.checkpoint / "inference_controls.json", {
            "purpose": "Checkpoint reload consistency, not a performance score",
            "fixture_sha256": digest(fixture_path), "weight_sha256": hashes,
            "metadata_sha256": digest(self.checkpoint / "metadata.json"),
            "controls": predictions})
        self.stats["saved_checkpoints"] += 1
        timing = self.pace(start)
        return {"type": "saved", "policy_version": self.policy_version, "weight_sha256": hashes,
                "generator_parameter_delta_l2": provenance["generator_parameter_delta_l2"],
                "generator_changed_tensor_count": provenance["generator_changed_tensor_count"], **timing}

    def close(self):
        elapsed = self.stats["active_seconds"] + self.stats["idle_seconds"]
        return {"type": "closed", **self.stats, "policy_version": self.policy_version,
                "observed_process_duty": self.stats["active_seconds"] / elapsed if elapsed else 0.0,
                "peak_mlx_memory_bytes": mx.get_peak_memory(), "sampling_parity": self.parity,
                "torch_imported": "torch" in sys.modules}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    trainer = GameplayTrainer(args)
    emit({"type": "ready", "policy_version": trainer.policy_version, **trainer.stats})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request["type"]
            if kind == "close":
                emit(trainer.close())
                return
            if kind == "begin":
                response = trainer.begin(request)
            elif kind == "act":
                response = trainer.act(request)
            elif kind == "update":
                response = trainer.update(request)
            elif kind == "save":
                response = trainer.save()
            elif kind == "discard":
                if trainer.buffer is not None:
                    trainer.stats["aborted_episode_decisions"] += len(trainer.buffer.prefixes)
                    trainer.buffer = None
                response = {"type": "discarded", "policy_version": trainer.policy_version}
            else:
                raise ValueError(f"Unknown worker request: {kind}")
            emit(response)
        except Exception as error:
            emit({"type": "error", "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
