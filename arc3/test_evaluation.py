import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from .encoding import decode_frame, decode_prediction, encode_frame, encode_prefix
from .verify_score import inspect_run, score_levels, verify_weights


HERE = Path(__file__).resolve().parent


class WeightTests(unittest.TestCase):
    def test_reject_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            weight = checkpoint / "generator.safetensors"
            weight.write_bytes(b"frozen-control")
            expected = hashlib.sha256(weight.read_bytes()).hexdigest()
            (checkpoint / "provenance.json").write_text(json.dumps({
                "weight_sha256": {weight.name: expected}}))
            self.assertEqual(verify_weights(checkpoint), {weight.name: expected})
            weight.write_bytes(b"changed-control")
            with self.assertRaisesRegex(AssertionError, "Frozen checkpoint hash differs"):
                verify_weights(checkpoint)


class EncodingTests(unittest.TestCase):
    def test_frozen_replay_controls(self):
        fixture = json.loads((HERE / "fixtures/replay_controls.json").read_text())
        for control in fixture["controls"]:
            observation = SimpleNamespace(frame=[control["frame"]],
                                          levels_completed=control["levels_completed"],
                                          available_actions=control["available_actions"])
            prefix, info = encode_prefix(observation, control["history"])
            self.assertEqual(prefix, control["prefix"])
            self.assertEqual(info["prefix_sha256"], control["prefix_sha256"])

    def test_frame_round_trip_and_capacity_failure(self):
        frame = np.indices((64, 64)).sum(axis=0).astype(np.uint8) % 16
        tokens, checksum = encode_frame(frame)
        np.testing.assert_array_equal(decode_frame(tokens), frame)
        self.assertEqual(checksum, hashlib.sha256(frame.tobytes()).hexdigest())
        prefix, failure = encode_prefix(SimpleNamespace(frame=[frame], levels_completed=0,
                                                       available_actions=[1, 2, 3, 4]), [])
        self.assertIsNone(prefix)
        self.assertEqual(failure["reason"], "prefix_capacity_overflow")
        self.assertGreater(failure["prefix_length"], failure["capacity"])

    def test_full_frame_and_cold_start(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        prefix, info = encode_prefix(SimpleNamespace(frame=[frame], levels_completed=0,
                                                    available_actions=[1, 2, 3, 4]), [])
        self.assertEqual(len(prefix), 1142)
        self.assertEqual(info["history"], [[0, 0, 0]] * 4)
        begin = prefix.index(3)
        np.testing.assert_array_equal(decode_frame(prefix[begin + 1:info["prefix_length"] - 1]), frame)

    def test_strict_raw_first_packet(self):
        self.assertEqual(decode_prediction([25, 1, 1] * 4, [1]),
                         {"type": "action", "action_id": 1, "data": {}})
        self.assertEqual(decode_prediction([30, 32, 95] * 4, [6]),
                         {"type": "action", "action_id": 6, "data": {"x": 0, "y": 63}})
        cases = [([25, 1, 1] * 3, "malformed_prediction_length_or_type"),
                 ([True, 1, 1] * 4, "malformed_prediction_length_or_type"),
                 ([23, 1, 1] * 4, "malformed_action_token"),
                 ([30, 31, 95] * 4, "malformed_click_coordinates"),
                 ([25, 32, 32] * 4, "coordinates_on_nonclick_action"),
                 ([26, 1, 1] * 4, "unavailable_action")]
        for tokens, reason in cases:
            self.assertEqual(decode_prediction(tokens, [1])["reason"], reason)


def scoring_run():
    game = "zzzz-00000000"
    frame = [[0] * 64 for _ in range(64)]
    checksum = hashlib.sha256(bytes(4096)).hexdigest()
    views = [{"frame": frame, "frame_sha256": checksum, "levels_completed": level,
              "state": "NOT_FINISHED", "available_actions": [1, 2]} for level in range(3)]
    episode = {"game_id": game, "termination": "time_budget", "steps": 2,
               "harness_actions": 2, "sdk_actions": 2, "completed_levels": 2,
               "score": 50.0, "policy_calls": 2, "forced_resets": 0,
               "level_budgets": [10, 20, 30]}
    trace = [{"event": "initial_reset", "game_id": game,
              "action": {"action_id": 0, "data": {}}, "frame_after": views[0],
              "harness_actions": 0, "level_budget_used": 0}]
    for index in range(2):
        trace.append({"event": "action", "game_id": game,
                      "action": {"action_id": index + 1, "data": {}},
                      "frame_before": views[index], "frame_after": views[index + 1],
                      "harness_actions": index + 1, "level_budget_used": 0})
    trace.append({"event": "episode_end", **episode})
    run = {"actions": 2, "level_actions": [1, 1, 0], "level_baseline_actions": [2, 4, 6],
           "level_scores": [115.0, 115.0, 0.0], "levels_completed": 2,
           "score": 50.0, "completed": False}
    card = {"environments": [{"id": game, "runs": [run], "score": 50.0,
                              "actions": 2, "levels_completed": 2}], "score": 50.0,
            "total_environments": 1, "total_actions": 2, "total_levels_completed": 2,
            "total_levels": 3}
    summary = {"episodes": [episode], "policy": "random", "subset_smoke": True,
               "sdk_missing_games": [], "initial_reset_budget_units": 0,
               "initial_reset_sdk_actions": 0, "network_disabled": True, "new_training": False,
               "multiplier": 5, "sdk_scorecard_rhae_percent": 50.0,
               "all_games": {"rhae_percent": 50.0, "games": 1},
               "excluding_training_game": {"rhae_percent": 50.0}, "worker_stats": None}
    return summary, card, trace, {game: [2, 4, 6]}


class ScoreTests(unittest.TestCase):
    def test_scoring_controls(self):
        for control in json.loads((HERE / "fixtures/scoring_control.json").read_text()):
            scores, result = score_levels(control["actions"], control["baselines"], control["completed"])
            self.assertEqual(scores, control["level_scores"])
            self.assertAlmostEqual(result, control["score"])

    def test_real_trace_accounting_and_reject_missing_action(self):
        summary, card, trace, baselines = scoring_run()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "summary.json").write_text(json.dumps(summary))
            (directory / "scorecard.json").write_text(json.dumps(card))
            trace_path = directory / "trace.jsonl"
            trace_path.write_text("\n".join(json.dumps(row) for row in trace))
            result = inspect_run(directory, baselines)
            self.assertEqual(result["rhae_percent"], 50.0)
            self.assertEqual(result["trace"]["actions"], 2)
            trace_path.write_text("\n".join(json.dumps(row) for index, row in enumerate(trace) if index != 1))
            with self.assertRaises(AssertionError):
                inspect_run(directory, baselines)

    def test_reject_wrong_baseline(self):
        summary, card, trace, baselines = scoring_run()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "summary.json").write_text(json.dumps(summary))
            (directory / "scorecard.json").write_text(json.dumps(card))
            (directory / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in trace))
            with self.assertRaises(AssertionError):
                inspect_run(directory, {next(iter(baselines)): [3, 4, 6]})


if __name__ == "__main__":
    unittest.main()
