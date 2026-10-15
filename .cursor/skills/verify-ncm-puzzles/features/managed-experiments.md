# Run resource-limited experiments

The guarded runner paces Apple GPU work and checks memory. The ten-minute wrapper starts a fresh generator from a saved tokenizer. Continuation restores a saved generator, optimizer, sampler, and RNG in a new destination.

## Sub-features

- `managed-evaluate` evaluates through the resource guard and saves its report.
- `managed-train` applies the guard to an existing generator recipe and bounded update request.
- `managed-fresh` starts and resumes the ten-minute generator experiment.
- `managed-continue` creates and resumes a separate continuation with source lineage.
- `managed-auto-evaluate` runs held-out validation after a successful continuation save.
- `managed-policy` records precision, attention, pooling, checkpoint copies, memory, pacing method, and stop reasons.

## How to get to it (user POV)

- Run `.venv/bin/python resource_runner.py evaluate` with normal evaluator arguments.
- Run `.venv/bin/python resource_runner.py train generator` with normal generator trainer arguments and a compatible resource recipe.
- Run `.venv/bin/python run_latin_10min.py` and add `--resume` to continue its destination.
- Run `.venv/bin/python continue_latin.py --evaluate` and add `--resume` to continue its destination.
- Choose `--generator-precision`, `--attention-kernel`, `--frozen-pooling`, `--checkpoint-copies`, `--pacing-method`, `--gpu-duty-cycle`, and `--memory-gb` before creating the new run.

## Driving it with control.py

Preconditions:

- Require `resource_runner.py` for guarded entry points, `run_latin_10min.py` for the fresh wrapper, or `continue_latin.py` for continuation, with their local helper modules. These application files were uncommitted local experiments when this skill was created and are not added by the skill. A checkout without them fails this prerequisite.
- Complete Launch and an MLX Doctor for evaluation or an MPS Doctor for training.
- Require normal system memory pressure, at least 6 GiB available RAM, and no other accelerator job. The guard can refuse to start based on observed system conditions.
- `managed-fresh` requires `runs/latin9-scratch/tokenizer/export`, its original 213-update tokenizer with SHA-256 `e7a37a8e46549632314d9031b6bef102521224f94da6b0263936d0391c071b47`, and matching shared `data/latin9`.
- `managed-continue` requires `runs/latin9-1000`, a matching generator state and export below 1,600 updates, a valid `.train.lock`, saved MPS RNG state, and the same shared corpus. Its source trainer must be stopped.
- The wrappers' source and corpus paths are fixed defaults or explicit read-only sources. A newly prepared disposable corpus alone does not satisfy those historical prerequisites.

- **Guarded evaluation entry.** Use a fresh output and explicit cache. Require the independent evaluation check plus `resources-evaluate.json` with return code `0`, no resource stop reason, the recorded policy, and generator precision labeled `not_applicable`.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" managed-evaluate -- \
	resource_runner.py evaluate --gpu-duty-cycle 0.8 --memory-gb 8 --pacing-method idle-credit --checkpoint-copies batched \
	--count 4 --batch-size 2 --device mlx \
	--output "$VERIFY_RUN/evidence/artifacts/guarded-evaluation" \
	--cache-dir "$VERIFY_RUN/work/guarded-source"
"$VERIFY_PYTHON" "$VERIFY_CHECK" "$VERIFY_RUN/evidence/artifacts/guarded-evaluation" \
	--expected-count 4 > "$VERIFY_RUN/evidence/guarded-evaluation-check.json"
```

- **Fresh generator entry.** Run a bounded fresh experiment in this session. Require exit `0`, `deadline-result.json` with `ready: true` and one completed update, matching current generator state and export, and `resources-train.json` without a stop reason. A saved older export does not satisfy this invocation.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" managed-fresh -- \
	run_latin_10min.py --run "$VERIFY_RUN/evidence/artifacts/ten-minute" \
	--stop-after 1 --seconds 120 --pacing-method idle-credit --checkpoint-copies batched
```

- **Fresh wrapper resume.** Resume the same destination and flags. Require total update two, not one, and a new ready result from this invocation.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" managed-fresh-resume -- \
	run_latin_10min.py --run "$VERIFY_RUN/evidence/artifacts/ten-minute" \
	--resume --stop-after 1 --seconds 120 --pacing-method idle-credit --checkpoint-copies batched
```

- **Guarded trainer entry.** Reuse the fresh wrapper's recorded resource recipe and one additional update. Require total update three and a new report. Read the trainer status and export directly; this invocation does not rewrite `deadline-result.json`.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" managed-train -- \
	resource_runner.py train --gpu-duty-cycle 0.8 --memory-gb 8 --pacing-method idle-credit --checkpoint-copies batched \
	--resource-report "$VERIFY_RUN/evidence/artifacts/guarded-trainer-report.json" \
	generator --config "$VERIFY_RUN/evidence/artifacts/ten-minute/resource-recipe.yaml" \
	--data data/latin9 --run "$VERIFY_RUN/evidence/artifacts/ten-minute" \
	--device mps --resume --allow-partial-tokenizer --stop-after 1
```

- **Continuation entry with evaluation.** Record every source file hash, then use the explicit historical source and a new destination. Require exit `0`, `continuation-result.json` with `ready: true`, `new_updates: 1`, the unchanged source hashes, and a complete four-case validation evaluation.

```sh
find runs/latin9-1000 -type f -exec shasum -a 256 {} + > "$VERIFY_RUN/evidence/source-before-continuation.txt"
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" managed-continue-evaluate -- \
	continue_latin.py --source runs/latin9-1000 --run "$VERIFY_RUN/evidence/artifacts/continuation" \
	--stop-after 1 --seconds 120 --evaluate --eval-count 4 --pacing-method idle-credit --checkpoint-copies batched
find runs/latin9-1000 -type f -exec shasum -a 256 {} + > "$VERIFY_RUN/evidence/source-after-continuation.txt"
diff "$VERIFY_RUN/evidence/source-before-continuation.txt" "$VERIFY_RUN/evidence/source-after-continuation.txt"
VERIFY_EVALUATION=$("$VERIFY_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_output"])' \
	"$VERIFY_RUN/evidence/artifacts/continuation/continuation-result.json")
"$VERIFY_PYTHON" "$VERIFY_CHECK" "$VERIFY_EVALUATION" --expected-count 4 \
	> "$VERIFY_RUN/evidence/continuation-evaluation-check.json"
```

- **Continuation resume entry.** Add one update to the same destination. Require its prior completed count plus one, the original `start_update` and target in `continuation.json`, and one new update in the result. No evaluation is requested in this invocation.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" managed-continue-resume -- \
	continue_latin.py --source runs/latin9-1000 --run "$VERIFY_RUN/evidence/artifacts/continuation" \
	--resume --stop-after 1 --seconds 120 --pacing-method idle-credit --checkpoint-copies batched
```

- **Read lineage and export.** Inspect `parent-run.json`, `continuation.json`, `continuation-result.json`, `generator/status.json`, and `generator/export/provenance.json`. Require preserved source update count and batch, constant saved learning rate, matching frozen-tokenizer hashes, current source/runtime hashes, and the same completed update in state and export. Use `torch.load(..., map_location="cpu", weights_only=True)` for read-only state inspection, as in the training recipe. The source hash comparison is part of the proof.

## Gotchas

- If the selected script or one of its imports is absent, report the missing application file as an unmet prerequisite. The mapped recipe remains unverified. Do not substitute a different entry point or create a stand-in script.
- Missing historical tokenizer, source run, MPS RNG state, or shared corpus is a failed prerequisite. Do not manufacture those states, edit checksums, or weaken the wrapper checks.
- `--seconds` includes preparation and training. A short budget may expire before an update and must be reported as a failure. Evaluation after `--evaluate` is outside that budget.
- The ten-minute wrapper caps its generator at 400 total updates. Continuation adds at most 600 and caps the target at 1,600.
- A resource stop is a nonzero failure even when it saved a valid boundary. Check the stop reason and this invocation's status, not only `ready` or file existence.
- BF16 generator computation defaults to `bf16-math` attention and `metal` frozen pooling. FP32 requires reference attention. Evaluation remains FP32 and reports checkpoint copies as `not_applicable`.
- Pacing defaults to `idle-credit`. `full-sleep` is a separate supported policy. Checkpoint copies default to `batched`; `blocking` is the reference choice. Resume requires the same recorded policy, precision, attention, pooling, checkpoint copies, and runtime source hashes.
- The guard paces this process. It cannot impose a system-wide GPU percentage or prevent other apps from writing swap.
- Direct guarded training requires a recipe whose recorded execution policy matches its flags. The standalone tiny training recipe lacks that guard descriptor.
- Automatic validation uses the validation split. It does not prove the separate test-split evaluation entry.
- The seed map contains these historical-path requirements. Only an actual successful run proves that the source state still meets them.
