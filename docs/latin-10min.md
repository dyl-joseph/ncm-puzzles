# Ten-minute Latin generator experiment

Run from the project directory.

```sh
caffeinate -i .venv/bin/python run_latin_10min.py
```

This reuses the Latin tokenizer from `runs/latin9-scratch/tokenizer/export`, saved after 213 updates. The wrapper verifies its checksums and copies the complete export into `runs/latin9-10min-resource/tokenizer/export`. The original run is untouched. The generator starts from random weights.

The new generator trains for at most 400 updates with a global batch of 32. Each update accumulates two physical microbatches of 16 to fit the 8 GiB memory budget. That is at most 12,800 fresh puzzle presentations. Its warmup lasts 20 updates. The run targets 80% process GPU duty with brief bursts. Read [Resource limits for Latin training](latin-resources.md) for the memory guard and GPU pacing behavior.

The generator forward uses BF16 autocast by default. Frozen tokenizer preprocessing, logits, loss, saved weights, gradients, and optimizer state remain FP32. Evaluation stays FP32. Select `--generator-precision fp32` for FP32 generator computation, and pass the same choice when resuming that run.

The timer includes setup. At 540 seconds the wrapper asks the trainer to finish its current update, save state, and export. At 590 seconds it kills an unresponsive trainer. It reserves the remaining time for cleanup and export verification. Validation runs separately after training.

Read `runs/latin9-10min-resource/deadline-result.json`. The `ready` field confirms that the saved state and export match this invocation. The record includes the actual update count, elapsed seconds, export path, resource report, and stop reason. A failed or killed invocation does not label an older export ready. A resource guard stop returns a failure exit code even when the latest checkpoint was saved.

```sh
.venv/bin/python resource_runner.py evaluate \
  --local-checkpoint runs/latin9-10min-resource/generator/export \
  --data data/latin9 --split test \
  --count 2000 --clue-count 30 --seed 42 \
  --batch-size 8 --device mlx \
  --output runs/compare-latin-10min-resource-test2000
```

Compare with the released Sudoku pair using the same cases. If `runs/compare-sudoku-test2000/summary.json` already exists and has status `complete`, use that saved baseline. Each evaluation output directory must be new when rerunning.

```sh
.venv/bin/python resource_runner.py evaluate \
  --checkpoint-dir artifacts/sudoku/hard \
  --data data/latin9 --split test \
  --count 2000 --clue-count 30 --seed 42 \
  --batch-size 8 --device mlx \
  --output runs/compare-sudoku-resource-test2000
```

Check that `cases_sha256` matches in the two summaries. Compare `metrics.solve_rate`, which counts valid complete Latin squares that preserve every given cell. The released Sudoku model has a much larger training budget, so this comparison measures the outcome of this short experiment.

To continue an interrupted run, explicitly pass `--resume`. Each invocation has its own maximum ten-minute budget. It preserves the original 400-update learning-rate schedule and stops once those 400 total updates are complete.

```sh
caffeinate -i .venv/bin/python run_latin_10min.py --resume
```

For a bounded verification run, use a separate directory and `--stop-after 3`. `--seconds` can shorten the deadline, but cannot exceed 600 seconds.

The [original ten-minute verification](latin-10min-verification.json) records the earlier run in `runs/latin9-10min` with a physical batch of 32. That verification file and the historical run remain unchanged. The default resource limits and physical microbatch of 16 apply to the new run documented here.
