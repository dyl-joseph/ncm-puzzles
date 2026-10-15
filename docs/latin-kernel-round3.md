# Exact training runtime optimization

This round shortens the wait for training while retaining the generator's existing computation. It changes pacing and checkpoint transfer scheduling. It does not change forward operations, loss, backward operations, gradient clipping, AdamW, or batch sizes.

The previous controller counted CPU telemetry after GPU synchronization as active work. It then added the full cooling pause. The updated controller counts that verified GPU idle time toward the current unit's required pause.

Let `A` be the time from the start of a guarded unit through successful GPU synchronization. Let `C` be the subsequent CPU telemetry time. Let `D` be the duty target. The controller requests `max(0, A * (1 / D - 1) - C)` seconds of sleep. The unit lasts `A + max(C, A * (1 / D - 1))`, so its guarded work duty stays at or below `D`. Excess idle time never carries into another unit.

Memory checks still run before work, after synchronization, and at completion. A critical telemetry failure stops the run. Optional GPU sampling remains optional. Reports distinguish post-synchronization idle time from sleep. The definition of active time changes, so the speed comparison uses complete command wall time.

Checkpoint copying previously waited after each GPU tensor transfer. The updated path queues contiguous MPS tensor copies together. It retains the source and destination buffers until synchronization finishes. CPU serialization starts only after that synchronization succeeds. Unsupported devices and layouts retain the original blocking copy behavior. Checkpoint frequency, saved fields, and atomic replacement stay unchanged.

The shared training recipe records the pacing method, checkpoint copy method, and helper source hashes. New defaults are `idle-credit` pacing and `batched` checkpoint copies. The previous behavior remains available with `--pacing-method full-sleep --checkpoint-copies blocking` in a new run. Resuming requires the same recorded settings and sources.

The baseline and candidate use the same BF16-math generator and exact Metal frozen pooling from the previous round. They use global batches of 32 puzzles, split into two physical microbatches of 16. The MPS allocation budget and resident memory guard remain 8 GiB. Both sides target 80 percent average duty.

Measurements use five alternating complete training commands per side. Each command continues the original 1,000-update checkpoint for 50 updates. Those updates include checkpoints at 1,025 and 1,050. Preparation, training, cooling, telemetry, save, export, and export verification fall inside the measurement. Evaluation is excluded on both sides. Separate checks compare raw gradients, clipped gradients, updated weights, optimizer tensors, sampler state, and RNG as bytes. A saved resume check compares two one-update processes with one uninterrupted two-update process.

The rejected gradient layout hooks removed duplicate copies but produced no measurable active-time gain. Removing the outer operator adapter was slower. Skipping gradient multiplication by exactly one produced a tiny gain that did not justify a runtime change. Those candidates are not enabled.

Model the Domain kept the current unit's work and idle time separate. Prove It Works required actual checkpoint serialization and resume equality. Explain the Number required the complete command measurement rather than a speed claim based on reclassified active time.

[Verification record](latin-kernel-round3-verification.json) contains the final timings, exact-state checks, resource samples, and evaluation artifact.

Five paired invocations measured 50 updates each. Complete training time fell by 5.4 percent. The earlier ten percent target was not reached.

| Runtime | Median complete training time | Range | Median training loop seconds per update |
| --- | --- | --- | --- |
| Previous pacing and copies | 40.401 seconds | 40.261 to 40.997 seconds | 0.752 |
| Idle credit and batched copies | 38.231 seconds | 38.141 to 38.460 seconds | 0.709 |

The outer process stopwatch also fell by 5.3 percent. The main savings come from about 1.6 seconds of post-synchronization CPU telemetry per 50 updates. The previous controller treated that telemetry as active time and added another cooling pause afterward. A separate checkpoint transfer probe measured 71.3 ms before and 9.0 ms after, over five paired trials. Checkpoint copying contributes only a small part of the complete training gain. Generator matrix computation still dominates.

All five pairs saved byte-identical training state apart from their different immutable execution provenance. Raw and clipped gradients matched byte for byte across two consecutive updates, with dropout enabled. Updated parameters, loss, gradient norm, AdamW state, sampler state, and RNG also matched. Saved resume matched an uninterrupted two-update run. The exported checkpoint completed all 100 requested MLX smoke evaluation puzzles. That evaluation checks compatibility and does not establish a solve-rate improvement. All 109 focused tests passed, including both MPS checkpoint copy tests and the existing MPS pooling tests.

The final comparisons recorded zero new swapout or pageout bytes. Candidate synchronized MPS allocation samples peaked at 4.14 GiB. Peak process RSS stayed below 1 GiB in the complete training comparisons. The allocator and resident-memory guards stayed at 8 GiB. Guarded work duty stayed below the 80 percent target.

Independent whole-invocation GPU sample means ranged from 69.8 to 79.8 percent. Samples between the first and last logged update ranged from 75.3 to 86.3 percent. Brief device-wide samples reached 99 percent. These counter samples include all apps and do not establish a hard hardware percentage cap. The current-unit pacing invariant applies to this process's guarded work. Checkpoint transfers and some input uploads occur outside those units. The verification record retains both sampling windows rather than discarding the higher readings.

Start a new continuation directory to bind the current runtime and settings.

```sh
caffeinate -i .venv/bin/python continue_latin.py --run runs/latin9-fast-exact --evaluate
```

This command continues the saved 1,000-update generator for up to 600 additional updates. Estimated training time is about seven to eight minutes. Evaluation adds roughly one minute outside the training deadline. The estimate comes from the bounded trials. No full 600-update experiment ran during this optimization. Add `--resume` to resume that exact destination after a saved stop.

The original source checkpoint, tokenizer snapshot, trainer, MPS adapter, MLX evaluator, and pinned upstream checkout remain unchanged.
