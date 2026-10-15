---
name: verify-ncm-puzzles
description: "Drive the ncm-puzzles Python CLI to verify Latin-square evaluation, corpus preparation, training and resume, resource-limited experiments, or MLX parity. Use after changing these commands or when collecting proof from the real models."
---

# Verify ncm-puzzles

The primary user interface is the Python CLI in this checkout, named `ncm-puzzles` in `README.md`. There is no web server, account, or listening port. Tests cover smaller pieces of the pipeline. Use the actual commands below to prove user behavior.

Read [the feature index](features/README.md), then the matching recipe before driving a command. Report each entry point you exercised. A frozen-checkpoint evaluation does not prove training or every evaluation split.

## Launch

Run from the repository root. Use Python 3.12 and the pinned packages in `requirements.txt`. If `.venv`, `vendor/ncm`, or `artifacts/sudoku/hard` is missing, complete [Set up a fresh checkout](../../../README.md#set-up-a-fresh-checkout) first, including checkpoint checksums and `provenance.json`. Do not replace an existing environment, checkpoint, or vendor checkout to make a check pass.

These commands prepare a verification session with an empty working directory:

```sh
VERIFY_PYTHON="$PWD/.venv/bin/python"
VERIFY_CONTROL="$PWD/.cursor/skills/verify-ncm-puzzles/scripts/control.py"
VERIFY_CHECK="$PWD/.cursor/skills/verify-ncm-puzzles/scripts/check_evaluation.py"
VERIFY_RUN=$("$VERIFY_PYTHON" "$VERIFY_CONTROL" launch)
"$VERIFY_PYTHON" "$VERIFY_CONTROL" doctor "$VERIFY_RUN" --device mlx > "$VERIFY_RUN/evidence/doctor.json"
cat "$VERIFY_RUN/evidence/doctor.json"
```

Require exit code `0` and `"ready": true`. The doctor checks the interpreter, pinned dependencies, clean upstream revision, checkpoint hashes, provenance, unchanged app source, and availability of the requested backend. It does not fetch data or run inference. Each drive starts its own short-lived process in a separate session with a PTY. There is no server to leave running.

Use MLX Metal on the GPU for evaluations on this Apple Silicon checkout. If Metal is unavailable or another accelerator job is active, report that prerequisite and wait. Do not silently switch evaluations to CPU. The map retains an explicit CPU comparison entry for cases that require it. Training recipes below use MPS. Frozen Sudoku evaluation explicitly rejects `--device mps`.

No initial corpus or seed files are needed for frozen evaluation. Its real CLI downloads a bounded public source prefix into the session's empty `work/source-cache`. Full corpus preparation, training, and the saved-run wrappers have additional preconditions in their feature files. Network failure is a failed journey, not permission to generate replacement puzzles.

The session lives under `runs/verification/ncm-puzzles-<UTC timestamp>-<random suffix>/`. `session.json` records this checkout, Git HEAD, and hashes of the app scripts, configs, and requirements, including uncommitted files. `work/` holds disposable caches. `evidence/` holds command logs and actual app outputs.

Use separate sessions for separate runs. Files are isolated, but GPU and system memory are shared. Run one accelerator workload at a time. The parity command also uses the fixed `.cache/latin` cache. Do not drive it concurrently with another command using that cache. The bounded wrappers read shared, immutable corpus and source-run files. Stop if a source trainer is active. Their locks reject some conflicting operations, but the locks do not isolate GPU use.

For teardown, run the Cleanup command even after a failed drive.

## Doctor

Run this read-only check before driving and whenever output, backend, or provenance looks wrong:

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" doctor "$VERIFY_RUN" --device mlx
```

Use `--device mps` for a training run. The default evaluation backend is `mlx`. A failed doctor means stop driving that session. Read its error, fix the observed prerequisite, then rerun. If app source changed after launch, clean up and launch a new session. A cleaned session cannot be driven again.

## Drive

For the initial proof, run the frozen historical benchmark through its real GPU CLI:

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" evaluation-frozen-mlx -- \
	evaluate_latin.py --count 4 --seed 42 --batch-size 2 --device mlx \
	--output "$VERIFY_RUN/evidence/artifacts/frozen-mlx" \
	--cache-dir "$VERIFY_RUN/work/source-cache"
"$VERIFY_PYTHON" "$VERIFY_CHECK" "$VERIFY_RUN/evidence/artifacts/frozen-mlx" \
	--expected-count 4 > "$VERIFY_RUN/evidence/frozen-mlx-check.json"
cat "$VERIFY_RUN/evidence/frozen-mlx-check.json"
```

Require the progress lines `Completed 4/4` and `Status complete`, command exit code `0`, and checker output `"verified": true` with `backend: mlx-metal`. Four cases cover the 25, 37, 49, and 61 clue bins. The checker reads the saved puzzles and raw token IDs and independently recomputes Latin validity, preserved clues, Sudoku diagnostics, and aggregate metrics. It also checks GPU metadata, the case checksum, and record counts. Zero solved cases can be a correct execution result for a frozen Sudoku model.

The runner accepts only the repo's seven user-facing script paths. It adds the prepared interpreter, `-B`, and `-u`, and runs from the repo root. It requires explicit output destinations under `evidence/artifacts` and explicit disposable caches under `work`. It forwards Ctrl-C and SIGTERM to the exact child it started. Wait for the child to finish saving before cleanup. The training wrappers manage their own trainer processes and deadlines.

Use a unique action label for every invocation, including resumes and expected failures. The label begins with the feature ID, such as `training-tokenizer-resume`. Reusing an action label fails before starting the command. The other [feature recipes](features/README.md) provide their literal commands and required end states.

## Evidence

Evidence remains at `$VERIFY_RUN/evidence/`, inside the Git-ignored `runs/` tree:

- `doctor.json` identifies the prepared environment and frozen checkpoint.
- `actions/<label>/command.json` records the action label, exact argument vector, working directory, and start time.
- `actions/<label>/stdout.txt` and `stderr.txt` retain each stream. Stdout comes from a PTY and can contain carriage returns.
- `actions/<label>/transcript.jsonl` records stream chunks in arrival order with elapsed times. `result.json` records the child PID, exit code, and finish time.
- `artifacts/` contains the actual evaluation summaries, cases, raw predictions, corpora, checkpoints, exports, resource reports, and lineage created by the chosen recipes.
- `frozen-mlx-check.json` records the independent evaluation check. Other recipes use distinct check filenames.
- `manifest.json`, written during cleanup, lists retained files, sizes, and SHA-256 hashes.

Exercise the user command and inspect its saved output from a second read. Do not claim proof from internal setters, mocked model predictions, synthetic replacement corpora, or tests alone. Capture the action and result together. Inspect the file side effects as well as terminal success. For training, require optimizer boundaries, exported hashes, and update counts. For continuation, also compare source hashes before and after the drive.

The source fetch is an actual external boundary. Do not mock it for this proof. A validated existing source cache can support a later offline run, but record that distinction. These CLIs have no general dry-run mode. `--stop-after` performs real optimization and writes checkpoints. The wrappers' `--seconds` deadline covers training setup and work. Automatic evaluation runs afterward and can extend the total invocation.

Keep setup, execution, and model-quality claims separate. A small sample proves the recorded cases and commands. It does not establish solve quality, full training completion, all-case parity, or an unmeasured speedup. Record failed or unavailable entry points with their command and missing prerequisite. Never count a skipped path as verified through another entry point.

## Cleanup

After every attempt, wait for the drive to exit and run:

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" cleanup "$VERIFY_RUN"
test ! -e "$VERIFY_RUN/work"
test -s "$VERIFY_RUN/evidence/manifest.json"
test -s "$VERIFY_RUN/evidence/artifacts/frozen-mlx/summary.json"
test -s "$VERIFY_RUN/evidence/frozen-mlx-check.json"
```

The last two checks apply to the initial proof. For another feature, check its actual artifact and result files. Cleanup removes only this session's `work/` directory. It preserves evidence, metadata, and the lock file. Cleanup is safe to repeat. It refuses a session held by another running command.

Never kill by process name. If a controller was forcibly killed, `active.json` identifies its child PID and command. Read the recorded PID and confirm its command and start time with `ps` before signaling that exact process. Wait for any wrapper's own child teardown. Remove stale `active.json` only after confirming its command and descendants have exited, then rerun cleanup. Do not delete a live session or the user's experiment directories.

The parity script's fixed `.cache/latin` cache is outside this session. Preserve it, including data fetched by parity. The historical source cache created by the initial proof lives under `work` and is removed.

## Helpers

Both scripts are executable and use only Python's standard library for process control and result inspection. Doctor imports the app's existing environment verifier.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" --help
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run --help
"$VERIFY_PYTHON" "$VERIFY_CHECK" --help
```

`control.py launch` creates a session. `doctor SESSION --device mlx` checks it. `run SESSION LABEL -- SCRIPT ARGUMENTS` drives a real CLI and returns its exit code. `cleanup SESSION` removes scratch data and inventories evidence. `check_evaluation.py DIRECTORY --expected-count 4` verifies a completed MLX Metal evaluation without loading model weights or touching its files. It checks the recorded GPU identity. An explicit backend comparison must pass the corresponding `--expected-device`, such as `cpu`.

Use `/maintain-verification-skill` when app changes make this map stale.
