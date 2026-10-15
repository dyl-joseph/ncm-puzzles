# Continue the Latin generator and evaluate it

From the project directory, run:

```sh
caffeinate -i .venv/bin/python continue_latin.py --evaluate
```

The command continues `runs/latin9-1000` in a new directory, `runs/latin9-1600`. It keeps the frozen 213-update tokenizer and restores the generator weights, AdamW state, random state, and next training sample from update 1,000. It adds up to 600 generator updates, reaching at most 1,600 total.

The global batch stays at 32 puzzles. Each update accumulates gradients from two physical microbatches of 16 to fit the 8 GiB memory budget. The run targets 80% process GPU duty with brief bursts. Both training and evaluation stop on memory pressure or new system swap writes. Read [Resource limits for Latin training](latin-resources.md) for the limits and their scope.

The default generator forward uses BF16 autocast. Frozen tokenizer preprocessing, logits, loss, saved weights, gradients, and optimizer state remain FP32. Evaluation stays FP32. Use `--generator-precision fp32` with a new destination for an FP32 training phase. The choice is recorded in the recipe and must match when resuming.

The continuation uses the saved learning rate of `3e-5` throughout the new phase. It does not restart warmup. The new directory includes `parent-run.json` and `continuation.json` to record the source checkpoint, resource policy, runtime source hashes, and batch sizes. The source run remains unchanged.

Training preparation and the trainer share a ten-minute wall-clock budget. The wrapper requests a graceful stop at nine minutes and kills an unresponsive trainer before ten minutes. After a clean save, `--evaluate` runs all 2,000 validation puzzles through guarded MLX evaluation. User interruption or a resource guard stop skips automatic evaluation. Evaluation takes additional time outside the training budget. Each puzzle has 30 given digits and uses seed 42.

The final output prints the solve count, given-digit agreement, and path to `summary.json`. Read `runs/latin9-1600/continuation-result.json` for the number of new updates, resource report, and evaluation output directory.

If training stops before update 1,600, continue the same new run with:

```sh
caffeinate -i .venv/bin/python continue_latin.py --resume --evaluate
```

Each invocation has its own ten-minute training cap. The update target remains 1,600. A failed trainer does not trigger evaluation of an older export. An evaluation failure keeps the saved training checkpoint available.

For a bounded verification, select a separate destination and fewer updates:

```sh
caffeinate -i .venv/bin/python continue_latin.py \
  --run runs/latin9-resource-continuation-proof \
  --stop-after 2 --evaluate --eval-count 8
```

Run the continuation checks with:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_continue_latin.py'
```

The [original continuation verification](latin-continuation-verification.json) records the earlier 400-to-1,000-update phase. That verification file and the historical run directories remain unchanged. It does not describe the new resource policy or physical microbatch size.
