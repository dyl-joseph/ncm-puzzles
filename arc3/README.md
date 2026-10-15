# ARC-AGI-3 gameplay evaluation

Run the committed LS20 checkpoint as an agent in all 25 public games. This evaluates actual game responses, completed levels and the official human-relative action-efficiency score. The [gameplay report](../docs/arc3-gameplay.md) explains the results and comparison limits.

The model is frozen. It predicts four commands, executes the first raw command, observes the real game response and replans. Invalid commands and boards that exceed its fixed prefix capacity terminate the episode and remain in the score denominator. Human action baselines set budgets and scores; the model never receives them.

## Set up

Use Python 3.12 on an Apple Silicon Mac with Metal support. Keep the game SDK and MLX in separate environments. The SDK runs the games and prepares observations on the CPU. All neural inference uses MLX GPU, with no Torch import or CPU fallback.

```sh
python3.12 -m venv data/arcagi3/sdk-venv
python3.12 -m venv data/arcagi3/mlx-venv
data/arcagi3/sdk-venv/bin/python -m pip install -r arc3/requirements-sdk.txt
data/arcagi3/mlx-venv/bin/python -m pip install -r arc3/requirements-mlx.txt
```

Run these commands from the repository root. The two checkpoint files, metadata and training provenance are already included in [checkpoints/ls20-10min](checkpoints/ls20-10min). No training or human replay dataset is required to evaluate them.

## Download the games

```sh
data/arcagi3/sdk-venv/bin/python -m arc3.download_games
```

This reads the 25 exact revisions in [public-baselines.json](public-baselines.json), downloads their Python sources from ARC Prize, and saves a hash manifest under `data/arcagi3/evaluation_public/`. It obtains an anonymous download key without printing it. A rerun reuses files whose hashes match the manifest. Game sources stay outside Git. Evaluation itself blocks network access and does not publish a scorecard.

To change the cache location, supply `--destination PATH` to the downloader and the same `--environments-dir PATH` to the evaluator. Both scripts also accept `--baselines PATH` for an explicitly chosen baseline file.

## Play and score

```sh
data/arcagi3/sdk-venv/bin/python -m arc3.run_agent \
  --policy ncm --output runs/arc3-ncm \
  --mlx-python "$PWD/data/arcagi3/mlx-venv/bin/python"

data/arcagi3/sdk-venv/bin/python -m arc3.run_agent \
  --policy random --output runs/arc3-random

data/arcagi3/sdk-venv/bin/python -m arc3.run_agent \
  --policy repeat --output runs/arc3-repeat
```

Every run uses a new output directory. Defaults are seed 0, five times the human action allowance per level, and a 12-hour wall limit per game, matching the checked reference benchmark settings. The initial constructor reset costs zero actions. Later resets count toward the current level's budget. The model worker targets 80% process duty by sleeping for one quarter of synchronized inference time. This is soft pacing, not a hardware utilization ceiling.

`random` samples a legal nonreset action uniformly and samples click coordinates uniformly. `repeat` starts with the smallest legal action, clicks the center when needed, and repeats its own previous command. This online repetition agent receives no human trajectory.

For a short smoke check, add `--games ls20 --multiplier 0.05` and use a separate output directory. A subset run sets `subset_smoke` in its summary and cannot establish an all-game score. Use `--checkpoint PATH` for a relocated copy of the bundled frozen checkpoint, with both `.safetensors` files, `metadata.json` and `provenance.json`. The four output controls are specific to these weights.

Each run saves:

- `scorecard.json`, the official SDK scorecard.
- `summary.json`, per-game outcomes, action budgets, model resource statistics, source and weight hashes.
- `trace.jsonl`, every real before/after frame, executed command and explicit failure.
- `adapter_checks.json`, exact matches against four saved replay prefixes and predictions for NCM.
- `worker.stderr.log`, model-worker diagnostics for NCM.

## Verify

```sh
data/arcagi3/sdk-venv/bin/python -m unittest arc3.test_evaluation -v

data/arcagi3/sdk-venv/bin/python -m arc3.verify_score \
  runs/arc3-ncm runs/arc3-random runs/arc3-repeat \
  --output runs/arc3-verification.json
```

The independent verifier recalculates per-level scores, weighting, completion caps and the all-game average. It checks the exact public baselines, game denominator, real frame chains, own-action history, budgets, submitted action counts, frozen checkpoint hashes and GPU-worker statistics. A missing trace action or incorrect human baseline fails verification. Tests include positive scoring controls so a zero result alone cannot validate the score calculation.

The committed results include complete compressed action traces. Verify them without downloading the games or loading MLX:

```sh
python3 -m arc3.verify_score arc3/results/ncm arc3/results/random arc3/results/repeat
```

The saved model is a small MLX adaptation of [HazyResearch NCM](https://github.com/HazyResearch/ncm) at revision `c70274273f489abc73832b2b98584a8483b6b203`. Its configuration and differences from the released Sudoku model are in the checkpoint provenance. The upstream MIT notice is in [LICENSES/ncm-MIT.txt](../LICENSES/ncm-MIT.txt).
