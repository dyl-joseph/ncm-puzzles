# Colab final checkpoint validation

The supplied Colab checkpoint loads on the local MLX Metal GPU and matches all four inference controls saved during training. A new frozen test completed **0 of 183 levels across 25 public development games**, scoring **0.0%**. This is not a Kaggle competition score.

The weights are in [the final checkpoint directory](../arc3/results/colab-final/block-0023/checkpoint). This is the small NCM prototype with 204,608 trainable generator parameters and a frozen tokenizer, not a 25M-parameter model.

## Training evidence

The user supplied the archive for scheduled block 24, indexed `0023`. The four-hour reservation allowed 24 ten-minute attempts, including failed and interrupted attempts. This archive contains one final block; it does not establish how many earlier blocks completed successfully or the total optimizer updates across the run.

| Final-block measurement | Result |
| --- | ---: |
| Training wall time, including initialization and shutdown | 575.600034 seconds |
| Real SDK game actions | 27,082 |
| Sampled policy decisions | 27,083 |
| Episodes | 213 |
| Optimizer updates | 72 |
| Updates from an episode with nonzero reward | 1 |
| Baseline-only updates | 71 |
| Invalid-action penalties | 1 |
| Completed-level rewards | 0 |
| Internal checkpoint-save calls | 11 |

The only nonzero reward was an invalid-action penalty. Of the episodes, 211 reached the 128-decision horizon, one ended on an invalid action, and one reached the training deadline. The recorded frames, rewards, discounted advantages, policy versions, on-policy losses and counter totals passed the trace checks.

The ZIP CRC, file inventory, completion-manifest hashes and weight hashes pass. Stored tensor values are finite, and the tokenizer matches the bundled frozen tokenizer. The original block files are preserved byte for byte. The [archive origin](../arc3/results/colab-final/archive-origin.json) records the supplied ZIP's SHA-256.

The previous block `0022` weights and the run ledger were not supplied. Prior hashes agree between the block manifest, provenance and worker statistics, but the prior weight bytes, per-block parameter delta and full training chain remain unverified. The [training evidence report](../arc3/results/colab-final/training-evidence-verification.json) records this limit. Full training verification requires those preceding weights.

## Frozen gameplay result

The local test used greedy decoding, seed zero, all 25 pinned public games and five times each human action baseline per level. It executed 2,777 SDK actions. Thirteen games exhausted their current level's action allowance; twelve ended in policy failure. Five failures exceeded the fixed prefix capacity, five selected unavailable actions, and two supplied coordinates on nonclick actions.

LS20 completed zero levels within the first level's 110-action allowance. The other 24 games also scored zero. Neural inference used MLX 0.32.4 on Metal with no CPU neural inference or Torch import. Games remained headless, network access was disabled, and the supplied weights did not change.

The [score verification](../arc3/results/colab-final/frozen-evaluation/verification.json) independently checks the game denominator, frame chains, budgets, actions, weight hashes and score. The [validation report](../arc3/results/colab-final/validation.json) summarizes the checks and limitations.

## Verify the archived score

Run from the repository root. This needs no GPU or downloaded game sources:

```sh
python3 -m arc3.verify_score arc3/results/colab-final/frozen-evaluation \
  --checkpoint arc3/results/colab-final/block-0023/checkpoint
```

To run another frozen game evaluation, use the environments from [the setup guide](../arc3/README.md):

```sh
data/arcagi3/sdk-venv/bin/python -m arc3.run_agent \
  --policy ncm --output runs/arc3-colab-final-eval \
  --checkpoint arc3/results/colab-final/block-0023/checkpoint \
  --mlx-python "$PWD/data/arcagi3/mlx-venv/bin/python"
```
