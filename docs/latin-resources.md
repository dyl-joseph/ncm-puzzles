# Resource limits for Latin training

The next continuation starts from the saved 1,000-update generator and targets 1,600 total updates.
Its optimizer, RNG, sampler, and frozen tokenizer state continue from the source checkpoint.
The global batch stays at 32 puzzles. Each update accumulates two physical microbatches of 16 to fit the memory budget.
The generator forward uses BF16 autocast and the `bf16-math` attention kernel by default.
Its stored parameters remain FP32.
Frozen FP32 area pooling uses the `metal` kernel by default. The [second-round verification](latin-kernel-round2.md) records its exact-gradient checks and 2.9 percent additional reduction. The additional ten percent target was not reached.
The latest runtime batches checkpoint copies and credits mandatory CPU telemetry after synchronization toward the current cooling pause. The [third-round verification](latin-kernel-round3.md) records exact-gradient checks and complete training timings.
The source run stays intact. Use a new destination such as `runs/latin9-fast-exact` to bind the current runtime hashes.

```sh
caffeinate -i .venv/bin/python continue_latin.py --run runs/latin9-fast-exact --evaluate
```

Training has a 600-second wall-clock budget. The wrapper requests a saved stop at 540 seconds and kills an unresponsive child before 600 seconds.
Automatic evaluation runs afterward, outside the training deadline, on the same 2,000 validation puzzles.
User interruption or a memory guard stop skips automatic evaluation.
A memory guard stop returns a failure exit code even when the latest checkpoint was saved successfully.

The default resource policy applies to both training and evaluation.

- The MPS allocator limit is 8 GiB. Process peak resident memory also has an 8 GiB guard.
- The guard requires normal macOS memory pressure and at least 6 GiB of available RAM before it starts.
- Each completed update or evaluation batch synchronizes the GPU before a pause. The target process duty cycle is 80%.
- New system swapouts or pageouts stop the run. A guard detected after an optimizer update lets the trainer count and save that update. A check before new work leaves the previous durable checkpoint intact.
- MLX evaluation uses an 8 GiB allocator guideline and a 256 MiB cache limit. Its synchronized batch guard also checks current allocations and the allocator's peak active memory.

macOS controls swap globally. These limits cannot guarantee zero swap activity or stop another app from causing swap.
The guard stops after observing new writes and refuses to start under memory pressure. It does not disable system swap.
Existing swap usage can remain after an earlier run and does not mean this run wrote more data to swap.

macOS has no supported hard GPU percentage cap. The 80% setting paces this process, with brief bursts allowed.
Other apps can raise global GPU activity. The report distinguishes the measured process duty cycle from optional global GPU activity samples.
The active interval runs through successful GPU synchronization. CPU submission gaps inside that interval make it a conservative estimate of GPU work.
Mandatory CPU memory checks and optional GPU sampling after synchronization count toward the current unit's idle time. Excess idle time never carries into another unit. Checkpoint transfers and some input uploads occur outside guarded units.
The default `idle-credit` pacing targets 80% work duty. `--pacing-method full-sleep` selects the previous conservative pause in a new run.
The optional GPU observations are taken at work boundaries. Their arithmetic mean can favor busy periods and is not a continuous average.

Frozen target construction computes codes and teacher contexts together in one residual pass.
It avoids a second reconstruction pass and skips the unused final reconstruction.
The conditional generator, loss, optimizer, and global training batch remain the same.
Physical microbatches are smaller. Accumulation and dropout can change the numerical training trajectory.
The fast path validates the frozen tokenizer and the original 322-code hierarchy before use.

Only the generator forward uses BF16 autocast. Frozen tokenizer preprocessing, returned logits, cross-entropy loss, parameters, gradients, and AdamW state stay FP32.
The `bf16-math` kernel lets math attention retain BF16 intermediates during that forward pass.
The `reference` kernel uses PyTorch's FP32 math-attention intermediates for BF16 inputs.
The wrapper saves PyTorch's prior math-attention flag and restores it after the generator forward, including when the forward fails.
The frozen tokenizer, cross-entropy loss, and backward run outside that flag context.
Evaluation remains FP32 and does not change the flag.
Evaluation resource reports label both generator training precision and attention kernel `not_applicable`.
BF16 changes numerical rounding and is not bitwise equivalent to FP32 training. A bounded precision comparison does not establish equivalent puzzle-solving accuracy.
Retaining BF16 attention intermediates changes rounding again relative to the BF16 `reference` path.

`--attention-kernel` accepts `bf16-math` or `reference` in `continue_latin.py`, `run_latin_10min.py`, and training through `resource_runner.py`.
If omitted, BF16 generator precision selects `bf16-math` and FP32 selects `reference`.
Selecting `bf16-math` with FP32 generator precision fails before backend setup or continuation preparation.
To keep BF16 generator computation with the prior math-attention behavior, use a new continuation directory and select `reference`.

```sh
caffeinate -i .venv/bin/python continue_latin.py \
  --run runs/latin9-1600-reference --attention-kernel reference --evaluate
```

To use FP32 generator computation, select a new continuation directory and record that choice explicitly.

```sh
caffeinate -i .venv/bin/python continue_latin.py \
  --run runs/latin9-1600-fp32 --generator-precision fp32 --evaluate
```

To resume the default BF16 continuation after a saved stop, add `--resume` to its command.

```sh
caffeinate -i .venv/bin/python continue_latin.py --run runs/latin9-fast-exact --resume --evaluate
```

To resume the FP32 example, retain its destination and precision choice.

```sh
caffeinate -i .venv/bin/python continue_latin.py \
  --run runs/latin9-1600-fp32 --generator-precision fp32 --resume --evaluate
```

Resource policy and runtime source hashes are part of the new run's recipe and continuation lineage.
Changing those values requires a new continuation directory. Older runs keep their original source provenance.
Resuming also requires the same generator precision, attention kernel, frozen pooling kernel, pacing method, and checkpoint copy method.
`--checkpoint-copies batched` queues contiguous MPS checkpoint transfers together and synchronizes before CPU serialization. Other tensors retain the original blocking copy behavior. `--checkpoint-copies blocking` selects the previous transfer path in a new run.
`--frozen-pooling reference` selects the previous frozen pooling implementation in a new run.
The new pooling change matched raw gradients, clipped gradients, weights, AdamW tensors, and RNG byte for byte against the existing BF16-math path. The earlier attention precision change described below used different rounding and remains the baseline for this comparison.
Resume the BF16 reference example with both `--resume` and `--attention-kernel reference`.
The unchanged `train_latin.py` remains the underlying trainer. Direct calls to it do not apply these resource controls.
The guarded training entry point supports only the generator stage. Tokenizer training is outside this bounded-memory path.

The underlying trainer retains legacy `precision: float32` and FP32 execution deviation strings for checkpoint compatibility.
Those strings do not describe the complete wrapped computation.
`recipe.local_choices.execution.generator_precision` and `recipe.local_choices.execution.attention_kernel` record the actual generator forward choices.
The stored parameter and optimizer tensors remain FP32 in either mode.

Each guarded invocation writes `resources-train.json` or `resources-evaluate.json` in its output directory.
The reports contain the policy, active and idle time, post-synchronization telemetry time, sleep time, peak allocations, memory snapshots, new swapout bytes, and stop reason.
The revised active-time definition excludes post-synchronization CPU telemetry. Compare complete command wall time when measuring speed between pacing methods.
MPS allocation maxima are synchronized endpoint samples. The allocator separately enforces the 8 GiB budget between samples.
The guard checks before new work and after synchronized work, then takes a final memory snapshot after checkpoint/export work.
`continuation-result.json` links the training report and any completed evaluation output.

## Earlier attention measurements on this Mac

The [kernel verification record](latin-kernel-verification.json) measures five alternating complete training invocations per kernel.
Each invocation continues the same 1,000-update source for 20 updates, using the same global batch of 32, microbatch of 16, optimizer state, sampler state, initial random state, memory checks, and cooling policy.
Both sides use BF16 generator forward with FP32 stored weights and optimizer state. The attention math policy is the only computation change.

| Attention kernel | Median training time | Range | Median seconds per update in the training loop |
| --- | --- | --- | --- |
| Previous BF16 reference | 21.467 seconds | 21.414 to 22.217 seconds | 0.943 |
| BF16 math | 18.848 seconds | 18.524 to 19.322 seconds | 0.803 |

Complete training is **12.2% shorter**. This includes preparation, updates, memory checks, cooling pauses, checkpoint/export, and export verification.
The outer process stopwatch also improved by 12.0%, from 22.020 to 19.386 seconds.
Evaluation is outside these training measurements on both sides.

PyTorch normally promotes BF16 Q/K/V to FP32 in gradient-bearing composite MPS attention.
The scoped policy retains BF16 attention intermediates, which shortens both forward and backward work.
The policy uses PyTorch's global math-attention setting despite its CUDA namespace. See the [official implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/attention.cpp#L821).
It preserves the mask, dropout probability, architecture, update count, and optimizer.

All 77 focused CPU checks passed. All ten exported checkpoints passed validation.
The final candidate also resumed from update 1,020 to 1,021, with matching AdamW counters and sampler cursor, then completed 100 MLX FP32 evaluation puzzles.
The source checkpoint, frozen tokenizer, original trainer, adapter, and upstream checkout stayed unchanged.
No new system swapouts or pageouts occurred. Candidate MPS endpoint allocations stayed at about 4.14 GiB, under the unchanged 8 GiB allocator budget.
Independently sampled whole-device GPU averages ranged from 65.6% to 78.6% across candidate trials, with brief peaks up to 100%.
Those observations include other applications and do not guarantee future whole-device activity.

With training dropout enabled, the paired kernel check found a 0.0197% relative loss difference, gradient cosine similarity of 0.999888, and 1.51% relative gradient L2 difference.
The frozen codes and contexts matched exactly. The MPS random state matched after paired training and across complete 20-update trials.
BF16 attention changes rounding. These checks establish execution fidelity and short-run numerical agreement, not equivalent long-run solve rates.

At the measured rate, 600 updates are estimated to take roughly 8 to 9 minutes, with the existing nine-minute saved-stop request and ten-minute hard deadline retained.
The subsequent 2,000-puzzle evaluation is estimated to take roughly another minute, outside the training budget.
These full-run times are extrapolations from bounded trials.
The [earlier resource verification](latin-resource-verification.json) records the prior BF16 rollout and its baseline measurements.

Model the Domain shaped the immutable shared resource policy. Explain the Number separated active-compute timing from paced wall time.
Prove It Works required actual checkpoint continuation, resume, sampler replay, and evaluation on this Mac before handing over the command.

Run the CPU behavior checks.

```sh
.venv/bin/python -m unittest tests.test_ncm_fast_targets tests.test_ncm_resource tests.test_ncm_attention tests.test_resource_runner tests.test_continue_latin tests.test_run_latin_10min tests.test_ncm_area_pool tests.test_ncm_checkpoint
```

## Continue after 1,600 updates

By default, each new continuation targets 600 additional updates from its source checkpoint. `--additional-updates` selects a larger fixed target without raising the per-invocation update or time caps. The old 1,600-update ceiling has been removed. The source remains unchanged, and the new run preserves weights, AdamW, sampler, RNG, frozen tokenizer, and the saved constant learning rate.

To continue the completed fast run from 1,600 to 2,200 updates, use a new destination.

```sh
caffeinate -i .venv/bin/python continue_latin.py \
  --source runs/latin9-fast-exact --run runs/latin9-2200 \
  --stop-after 600 --evaluate
```

After a saved stop in that new run, use the same command with `--resume`. The ten-minute training deadline and existing resource guards still apply. [Preparation verification](latin-continuation-2200-verification.json) confirms byte-identical copied training state from the actual 1,600-update checkpoint without starting training.

## Continue the generator for 2,000 additional updates

The 1,600-update source uses the same tokenizer checkpoint that reconstructed 2,000 complete validation boards exactly. Continue the generator while that tokenizer stays frozen. The latest generator evaluation solved zero of 2,000 puzzles. More generator training is an experiment, not an established solve-rate improvement.

```sh
caffeinate -i .venv/bin/python continue_latin.py \
  --source runs/latin9-fast-exact --run runs/latin9-3600 \
  --additional-updates 2000 --evaluate
```

The target is fixed at update 3,600. Each invocation applies at most 600 new updates and keeps the ten-minute training deadline. Repeat the same command with `--resume` after a saved stop until it reaches the target. At the last measured rate, expect four invocations and about 24 minutes of training in total. Evaluation adds time after every invocation with `--evaluate`.

On resume, omitting `--additional-updates` uses the saved target. A conflicting explicit value fails before changing the run. Repeating the resume command never adds another 2,000 updates to the target. Source weights, optimizer, sampler, RNG and tokenizer remain bound by the existing lineage checks.

[Verification](latin-continuation-3600-verification.json) records byte-identical initial state migration, two isolated verification updates across a process restart, and an MLX compatibility evaluation. The requested 2,000-update experiment has not been launched.
