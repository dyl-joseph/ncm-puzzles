# ncm-puzzles verification map

This directory is the maintained source for verifying the Python CLI. Read the matching feature recipe before driving the app. [SKILL.md](../SKILL.md) defines the session, runner, and proof requirements.

## Baseline preconditions

- Run from the checkout root with the prepared Python 3.12 environment and pinned upstream source.
- Define `VERIFY_PYTHON`, `VERIFY_CONTROL`, `VERIFY_CHECK`, and `VERIFY_RUN` through the Launch commands in the skill.
- Require a successful doctor for the backend in use.
- Keep each new output under `$VERIFY_RUN/evidence/artifacts` and each configurable source cache under `$VERIFY_RUN/work`.
- Frozen evaluation starts with no puzzle data. Corpus preparation downloads the real full collection. Training recipes use that prepared corpus.
- Apple-specific recipes require MLX Metal or MPS. The ten-minute and continuation wrappers also require saved source runs and the shared `data/latin9` corpus.
- Require the chosen experiment script and its imports before driving that entry point. See [managed-experiment prerequisites](managed-experiments.md).
- Use one accelerator workload at a time. Do not drive the user's running training job or use its destination as a verification output.

## Driving conventions

- Use `control.py run` for every app invocation. It starts the literal CLI in its own process and PTY.
- Give every action a new label that starts with its feature ID. Capture expected failures under separate labels too.
- Start each recipe from its stated preconditions. A recipe's entry points are separate coverage obligations.
- Keep writes inside the session. Read-only source checkpoints and prepared data may be shared when a recipe explicitly requires them.
- Run the independent evaluation checker for every completed evaluation, with that invocation's expected count.
- Clean up after failures as well as successes. Keep all proof artifacts.

## Proof and failure reporting

- Record the action, stdout, stderr, exit code, and saved end state.
- Inspect file contents and hashes through a second read after the CLI exits.
- Report the feature ID, entry point, backend, and sample or update count.
- Report unmet preconditions as failures with the attempted command or exact missing path. Do not fabricate a prerequisite or weaken the journey.
- A zero solve count can pass execution verification. A nonzero process exit cannot pass because an older export exists.
- Only the journey recorded in [PROOF.md](../PROOF.md) has been executed for this skill's initial delivery. Other recipes are seeded from source and need their own run before anyone claims them verified.

## Features

| Feature | User entry points | Proof |
| --- | --- | --- |
| [Evaluate Latin completions](evaluation.md) | Frozen weights, local exports, historical benchmark, validation and test splits, output refusal, interruption | Raw predictions, independent scores, complete or interrupted summary |
| [Prepare the Latin corpus](corpus.md) | First preparation and validated reuse | Full corpus, pinned source hashes, disjoint splits, unchanged reuse |
| [Train and resume](training.md) | Tokenizer, generator, `all`, full and pilot configs, saved exports | Real optimizer updates, durable resume, frozen tokenizer, export evaluation |
| [Run resource-limited experiments](managed-experiments.md) | Guarded train and evaluate, ten-minute fresh generator, continuation and resumes | Resource reports, current ready exports, source preservation, lineage |
| [Compare MLX and PyTorch](parity.md) | `scripts/verify_mlx.py` | Exact latent and raw IDs plus four solved Sudoku controls |
| [Run another puzzle task](puzzle-runtime.md) | `puzzle_runner.py`, `scripts/verify_puzzle_runtime.py` | Real 4×4 NCM updates, exact saved resume, reference fallback, independent scorer, memory reports |

Each feature file uses the same four H2s. `Sub-features` names the coverage units, `How to get to it (user POV)` lists entry points, `Driving it with control.py` supplies commands and results, and `Gotchas` records prerequisites and traps.
