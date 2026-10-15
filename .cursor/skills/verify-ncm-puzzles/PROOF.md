# Verification proof

The generated skill completed its launch, doctor, real CLI drive, independent result check, and cleanup on 2026-10-15 UTC. Evaluations use MLX on the GPU by default.

## Executed journey

- Feature entry `evaluation-frozen-mlx` ran `evaluate_latin.py --count 4 --seed 42 --batch-size 2 --device mlx` through the generated PTY runner.
- The run started with an empty session and source cache. The actual CLI downloaded the first 4,096 source lines and loaded the pinned published checkpoint.
- The recorded backend is `mlx-metal` on `Apple M5 Pro`, using FP32 with TF32 disabled.
- All four cases completed, one per clue bin of 25, 37, 49, and 61. The independent checker verified raw-token scores, clue counts, case hashes, GPU metadata, and aggregate metrics.
- The frozen Sudoku checkpoint solved 0 of four Latin boards. This proves execution and scoring, not solve quality.
- Feature entry `evaluation-refusal` returned the expected exit code 2 when reusing the output directory. Its summary hash stayed unchanged.
- Doctor passed before and after the drive. Both child processes exited.

## Retained evidence

The reviewable snapshot is [evidence/initial-gpu-proof](evidence/initial-gpu-proof). The original local run remains under `runs/verification/ncm-puzzles-20261015T052940Z-8v1j5kge/evidence`, inside the Git-ignored `runs/` directory.

- [GPU evaluation check](evidence/initial-gpu-proof/frozen-mlx-check.json)
- [Actual summary](evidence/initial-gpu-proof/artifacts/frozen-mlx/summary.json)
- [GPU command and transcript](evidence/initial-gpu-proof/actions/evaluation-frozen-mlx)
- [Expected output-refusal action](evidence/initial-gpu-proof/actions/evaluation-refusal)
- [Doctor](evidence/initial-gpu-proof/doctor.json)
- [Helper failure checks](evidence/initial-gpu-proof/helper-checks.json)
- [Evidence manifest](evidence/initial-gpu-proof/manifest.json)

After cleanup, all 19 retained files matched the manifest's byte counts and SHA-256 hashes. The session's `work/` and `active.json` were absent. Cleanup preserved the proof.

## Limits

This ran directly on the target Apple Silicon Mac with the existing prepared Python 3.12 environment, pinned upstream checkout, and original checkpoint. No container, replacement model, mocked prediction, or generated substitute corpus was used.

Corpus preparation, training and resume, managed experiments, local-export and held-out evaluation, interruption, and full backend parity are mapped recipes. They were not executed as this skill's proof. Their historical source-run requirements remain prerequisites for future drives.

The helper checks rejected output-path escapes, abbreviated destination flags, shared training destinations, missing cache arguments, CPU results presented as GPU proof, and corrupted saved summary metrics.

Use `/maintain-verification-skill` to keep the feature map current.
