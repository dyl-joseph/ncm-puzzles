# Run another puzzle task

The shared runner applies resource controls to a task-owned training or evaluation loop. The included second task uses actual upstream NCM classes on 4×4 Sudoku.

## Sub-features

- `puzzle-runtime-proof` compares continuous and split/resumed real NCM training.
- `puzzle-runtime-task` runs a task through the public CLI.
- The proof checks fixed-target resume refusal, saved state bytes, reference fallback, independent Sudoku scoring, and resource limits.

## How to get to it (user POV)

Run `.venv/bin/python puzzle_runner.py --task examples.sudoku4_task --device mps --run runs/sudoku4-demo -- --updates 40 --tokenizer-updates 20 --evaluate`.

Use the [puzzle runtime guide](../../../../docs/puzzle-runtime.md) to add another task. A new task supplies its own data, model, loss, checkpoint state, and scorer. The runner does not infer those from a dataset.

## Driving it with control.py

Complete Launch and a CPU Doctor after the source files are final. Run the exact proof on the explicit reference backend in a new destination:

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" puzzle-runtime-proof -- \
	scripts/verify_puzzle_runtime.py --device cpu \
	--output "$VERIFY_RUN/evidence/artifacts/puzzle-runtime"
cat "$VERIFY_RUN/evidence/artifacts/puzzle-runtime/verification.json"
```

Require exit code zero and `verified: true`. Read `commands.json` and the saved `whole` and `split` result and resource reports. Require two tokenizer and two generator updates, 32 evaluated puzzles, and `ncm_loss.implementation: reference`. Require zero new swapout and pageout bytes, no resource stop, and recorded work duty at most 0.8.

The proof also makes a changed-configuration resume request that must fail with exit code one. The sibling `.split.admission-error-report.json` records the refusal. The previous checkpoint, manifest, and successful reports must remain byte-identical.

Complete an MPS Doctor and wait until the user's accelerator job exits before driving the direct GPU entry point separately:

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" puzzle-runtime-task -- \
	puzzle_runner.py --task examples.sudoku4_task --device mps \
	--run "$VERIFY_RUN/evidence/artifacts/direct-task" -- \
	--updates 2 --tokenizer-updates 2 --batch-size 4 --checkpoint-every 1 --evaluate
```

Inspect its `result.json`, `resource-report.json`, `run-manifest.json`, and `checkpoint.pt` after exit. Cleanup the session through the skill after all commands finish.

## Gotchas

- The example is a small demonstration trained from scratch. Two updates prove execution and resume behavior, not solve quality or a speedup.
- The 16-cell hierarchy must select reference NCM loss. The specialized 89-token target and Metal pooling paths require their own verified contracts.
- Defaults use FP32 and reference attention. Explicit BF16/reduced-math settings change the numerical policy and cannot be introduced during resume.
- The CPU proof establishes exact saved resume. The separate MPS journey establishes GPU execution. Do not count one as proof of the other. Identical-input MPS embedding backward can differ in low bits even with deterministic mode enabled; the strict proof does not relax its comparison for that backend.
- The MLX runtime path requires a native MLX task and `mx.eval` inside each work unit. The included Sudoku task rejects MLX.
- A forced deadline kill leaves only the task's last completed atomic checkpoint safe to resume. Inspect the current exit code and logs.
