# Run another puzzle with the shared runtime

Use `puzzle_runner.py` to apply the shared resource controls to a puzzle task. Prepare the environment and pinned upstream source through the [fresh-checkout instructions](../README.md#set-up-a-fresh-checkout).

## Run the included NCM Sudoku example

Start a new run from the repository root:

```sh
.venv/bin/python puzzle_runner.py --task examples.sudoku4_task --device mps \
	--run runs/sudoku4-demo -- --updates 40 --tokenizer-updates 20 --evaluate
```

The example first trains a small upstream NCM tokenizer, then freezes it and trains the generator. It uses 256 of the 288 ordered 4×4 Sudoku solutions for training and holds out 32. The scorer checks preserved clues, rows, columns, and 2×2 boxes. Any valid completion passes. This small recipe demonstrates the runtime and does not reproduce the published training budget.

For a bounded smoke check, use `--updates 2 --tokenizer-updates 2 --batch-size 4 --checkpoint-every 1 --evaluate` after `--`. A two-update model does not establish puzzle-solving quality.

## Stop and resume at the same target

Limit the first invocation to three applied updates while keeping a fixed target of two tokenizer and two generator updates:

```sh
.venv/bin/python puzzle_runner.py --task examples.sudoku4_task --device mps \
	--run runs/sudoku4-resume -- \
	--updates 2 --tokenizer-updates 2 --batch-size 4 --checkpoint-every 1 --evaluate --stop-after 3
```

Resume the same destination and task settings:

```sh
.venv/bin/python puzzle_runner.py --task examples.sudoku4_task --device mps \
	--run runs/sudoku4-resume --resume -- \
	--updates 2 --tokenizer-updates 2 --batch-size 4 --checkpoint-every 1 --evaluate
```

Keep `--evaluate` and all settings that describe the task unchanged. `--stop-after` limits one invocation and can change between invocations. A changed task, runtime source, backend, dependency version, numerical policy, or resource policy fails admission before the task loads its checkpoint.

Each CLI invocation has a maximum budget of 600 seconds. The supervisor requests a safe stop at 90% of that budget and kills an unresponsive worker before the budget ends. After a forced kill, resume only the task's last completed atomic checkpoint. The example saves both models, AdamW state, tokenizer EMA, RNG, sampler, and completed counters in `checkpoint.pt`.

Inspect `run-manifest.json` for immutable execution identity, `resource-report.json` for measured limits and stop reasons, and `result.json` for the runtime status and task result. Check the return code and current report before accepting a saved result.

For a failure before admission, inspect the sibling file `.RUN_NAME.admission-error-report.json`. A failed startup leaves the task directory empty, so you can retry after fixing the prerequisite. A rejected resume preserves the previous successful reports.

## Add a task module

Use [examples/sudoku4_task.py](../examples/sudoku4_task.py) as the working adapter. Create a module with `build_task(arguments)` that returns an object with these members:

- `name` identifies the task.
- `metadata` is a JSON mapping containing data identity, model configuration, loss, optimizer, scorer, seed, and fixed targets.
- `source_paths` lists source dependencies whose hashes must stay unchanged on resume.
- `run(runtime)` constructs or restores the task and returns a JSON result mapping.

Keep model allocation in `run`. The runner configures the backend and checks memory before importing the task module. Include every source dependency that can affect model math or data ordering in `source_paths`.

Place each complete optimizer update inside `with runtime.work_unit():`. Include uploads, all accumulated forward and backward passes, gradient clipping, `optimizer.step()`, and the completed-counter increment. For evaluation, put each materialized inference batch inside a work unit. Check `runtime.stop_requested` between units. Nested units are rejected.

Apply `runtime.forward_context()` only to the model forward whose precision policy you intend to change. Keep frozen NCM target construction outside that context. Keep FP32 loss calculation, backward, clipping, optimizer state, RNG, and sampler behavior under the task's control.

Save after the work-unit boundary. For Torch payloads, call `runtime.cpu_tree(payload)` before the task's atomic save. The batched mode preserves tensor values and waits for queued MPS copies before returning. The task owns serialization, checkpoint validation, state restoration, and scoring.

Invoke the module through the same runner:

```sh
.venv/bin/python puzzle_runner.py --task your_package.your_task --device mps \
	--run runs/your-puzzle -- YOUR_TASK_ARGUMENTS
```

A new puzzle still needs its representation, objective, and scorer. The shared runner does not infer them from a dataset.

## Select applicable NCM fast paths

Bind the optional selector after constructing the generator and freezing the tokenizer:

```python
from puzzle_ncm import select_ncm_loss

loss = select_ncm_loss(
	generator, tokenizer, reference_loss,
	solution_length=solution_length,
	prefix_length=prefix_length,
	pooling=runtime.numerical.frozen_pooling,
	forward_context=runtime.forward_context,
	objective="mean-ce-all-codes",
)
```

Provide a `reference_loss(generator, tokenizer, solutions, prefixes)` callable for the task's actual hierarchy. Scope its generator forward through `runtime.forward_context()` as the Sudoku example does. Declare `objective="mean-ce-all-codes"` only for unweighted mean cross entropy over every clean target code, with no masking, label smoothing, or auxiliary terms. Otherwise leave the default `objective="task"` to retain the supplied loss.

Record `loss.implementation`, `loss.reason`, and `loss.pooling` in the task result. The original fast target path requires the vendored original NCM classes, the verified 89-token solution and 91-token prefix, the original 322-code hierarchy and generator configuration, and frozen deterministic FP32 MPS tokenizer state. The Metal pooling kernel checks its tensor shapes, strides, and gradient state separately. Unsupported model contracts select the reference callable before encoding, forward passes, or RNG consumption. A failure after an eligible fast path starts propagates without retry.

The 16-token Sudoku example uses reference loss. Its hierarchy differs from the specialized kernel. Do not substitute the Latin reference loss for another hierarchy.

## Keep the numerical policy fixed

Leave precision and attention at their FP32/reference defaults when comparing gradients with an FP32 baseline. Checkpoint batching and duty pacing do not change the model's numerical policy.

Select `--precision bf16 --attention bf16-math` only for a new experiment whose baseline uses the same settings. These options change rounding relative to FP32. They are recorded in the manifest and cannot change during resume. Exact-gradient comparisons apply within the same numerical policy.

Use `--checkpoint-copies blocking` and `--frozen-pooling reference` to select the corresponding reference implementations. Choose all policy flags before creating a run. Run `puzzle_runner.py --help` for the complete shared options, and inspect the task's `build_task` for its own arguments.

## Choose the backend explicitly

Use `--device mps` for Torch on this Mac. Use `--device cpu` for an explicit FP32 reference run. Both use macOS memory telemetry and process duty pacing. The runtime stops on new swapouts or pageouts, high pressure, insufficient available memory, or exceeded process limits. It cannot prevent other applications from using swap or impose a system-wide GPU percentage.

For an MLX-native task, use `--device mlx`, materialize all lazy work with `mx.eval` inside each work unit, and save through the task's native MLX checkpoint code. Torch autocast, reduced-math attention, checkpoint-copy conversion, and NCM pooling options do not apply to MLX. The CLI rejects explicit incompatible options. The included Sudoku task requires Torch.

Use the runtime on macOS with working telemetry. An unsupported platform fails with an actionable telemetry error instead of disabling the critical guards.

## Verify the runner on your Mac

Run the bounded real-model proof with a new destination:

```sh
.venv/bin/python scripts/verify_puzzle_runtime.py --device cpu \
	--output runs/puzzle-runtime-verification
```

The script compares uninterrupted and split/resumed NCM checkpoints byte for byte, validates the independent Sudoku scorer, checks resource reports, and rejects a changed resume configuration. It performs only two tokenizer and two generator updates per completed run. Read `verification.json` for the checks and command logs. This proof verifies execution and resume behavior, not a speedup or solve-quality claim.

Use the explicit CPU backend for this byte-equality proof. A diagnostic on this Mac found different output bytes across identical-input MPS embedding backward calls, including with deterministic mode enabled. An MPS split-resume check differed in one level-embedding weight by `5.96e-8`. The runner retains the original gradient operations. `--device mps` keeps the same strict comparison and can fail on that backend variation. Use the bounded MPS smoke command above to verify GPU execution separately.

Read the [recorded verification](puzzle-runtime-verification.json) for the measured checks and their limits.
