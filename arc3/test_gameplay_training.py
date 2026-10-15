import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .train_gameplay import (EpisodeBuffer, create_training_environment, discounted_weights,
                             objective_from_log_probs, parse_args, transition_reward)


class RewardTests(unittest.TestCase):
    def test_only_visible_completion_invalid_and_game_over_change_reward(self):
        before = {"levels_completed": 0, "state": "NOT_FINISHED"}
        self.assertEqual(transition_reward(before, {"levels_completed": 0, "state": "NOT_FINISHED"}),
                         {"level_completion": 0.0, "invalid_action": 0.0, "game_over": 0.0, "total": 0.0})
        self.assertEqual(transition_reward(before, {"levels_completed": 2, "state": "WIN"})["total"], 2.0)
        self.assertEqual(transition_reward(before, {"levels_completed": 0, "state": "GAME_OVER"})["total"], -1.0)
        self.assertEqual(transition_reward(before, invalid=True),
                         {"level_completion": 0.0, "invalid_action": -1.0, "game_over": 0.0, "total": -1.0})
        with self.assertRaises(ValueError):
            transition_reward(before, before, invalid=True)

    def test_discounted_episode_weights_include_absolute_timestep_and_previous_baseline(self):
        returns, weights = discounted_weights([0.0, 0.0, 1.0], 0.25, gamma=0.5)
        self.assertEqual(returns, [0.25, 0.5, 1.0])
        self.assertEqual(weights, [0.0, 0.125, 0.1875])
        self.assertEqual(objective_from_log_probs([-4.0, -2.0, -3.0], weights), 0.8125 / 128)
        self.assertEqual(objective_from_log_probs([-4.0], [-1.0]), -4.0 / 128)

    def test_single_penalty_is_not_centered_away(self):
        returns, weights = discounted_weights([-1.0], 0.0)
        self.assertEqual(returns, [-1.0])
        self.assertEqual(weights, [-1.0])
        self.assertNotEqual(objective_from_log_probs([-20.0], weights), 0.0)
        self.assertEqual(discounted_weights([0.0, 0.0], 0.0), ([0.0, 0.0], [0.0, 0.0]))
        self.assertEqual(discounted_weights([0.0], -0.1)[1], [0.1])

    def test_reject_nonfinite_rewards_or_invalid_lengths(self):
        for rewards in ([math.nan], [math.inf], [True], [0.0] * 129):
            with self.assertRaises(ValueError):
                discounted_weights(rewards, 0.0)
        with self.assertRaises(ValueError):
            objective_from_log_probs([1.0], [])


class OnPolicyTests(unittest.TestCase):
    def test_stale_policy_and_reward_mismatch_are_rejected(self):
        buffer = EpisodeBuffer(7, 3)
        buffer.append([0], [1] * 22, -15.0, 7)
        self.assertEqual(buffer.validate_rewards([-1.0], 7), [-1.0])
        for version in (6, 8, True, 7.0):
            with self.assertRaisesRegex(ValueError, "policy version"):
                buffer.validate_rewards([-1.0], version)
        with self.assertRaisesRegex(ValueError, "sampled decisions"):
            buffer.validate_rewards([], 7)

    def test_fixed_horizon_cannot_silently_expand(self):
        buffer = EpisodeBuffer(0, 0)
        for unused in range(128):
            buffer.append([0], [0] * 22, -1.0, 0)
        with self.assertRaisesRegex(ValueError, "horizon"):
            buffer.append([0], [0] * 22, -1.0, 0)


class SDKAndBudgetTests(unittest.TestCase):
    def test_headless_environment_gets_only_documented_make_arguments(self):
        class FakeArcade:
            def make(self, game_id, **kwargs):
                self.call = (game_id, kwargs)
                return SimpleNamespace(renderer=None, observation_space=object())
        arcade = FakeArcade()
        create_training_environment(arcade, "ls20-abcd", 43, "local-card")
        self.assertEqual(arcade.call, ("ls20-abcd", {"seed": 43, "scorecard_id": "local-card"}))

    def test_renderer_must_be_disabled(self):
        arcade = SimpleNamespace(make=lambda *args, **kwargs:
                                 SimpleNamespace(renderer=object(), observation_space=object()))
        with self.assertRaisesRegex(RuntimeError, "rendering"):
            create_training_environment(arcade, "ls20", 42, "card")

    def test_wall_budget_cannot_exceed_ten_minutes(self):
        arguments = ["--output", "unused", "--mlx-python", "unused-python"]
        self.assertEqual(parse_args(arguments).max_seconds, 600)
        for value in ("601", "nan", "inf", "0"):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args([*arguments, "--max-seconds", value])


class CheckpointControlTests(unittest.TestCase):
    def test_trained_controls_bind_reload_predictions_to_fixture_metadata_and_weights(self):
        from .run_agent import verify_adapter
        fixture_path = Path(__file__).resolve().parent / "fixtures/replay_controls.json"
        fixture = json.loads(fixture_path.read_text())
        predictions = [{"index": control["index"], "tokens": [26, 1, 1] * 4}
                       for control in fixture["controls"]]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            hashes = {}
            for name in ("tokenizer.safetensors", "generator.safetensors"):
                (checkpoint / name).write_bytes(name.encode())
                hashes[name] = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            metadata = checkpoint / "metadata.json"
            metadata.write_text('{"vocab_size":160,"prefix_len":1142}')
            saved = {"fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                     "weight_sha256": hashes, "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                     "controls": predictions}
            control_path = checkpoint / "inference_controls.json"
            control_path.write_text(json.dumps(saved))
            worker = SimpleNamespace(checkpoint=checkpoint, ready={"weight_sha256": hashes},
                                     infer=lambda prefix: {"tokens": [26, 1, 1] * 4})
            result = verify_adapter(worker)
            self.assertEqual(result["prediction_control_source"], "checkpoint_reload_consistency")
            self.assertFalse(result["prediction_controls_are_performance_scores"])
            self.assertEqual(len(result["checks"]), 4)
            (checkpoint / "generator.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(AssertionError, "different weights"):
                verify_adapter(worker)


if __name__ == "__main__":
    unittest.main()
