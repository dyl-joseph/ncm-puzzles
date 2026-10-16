# Actual-game training pilot

The NCM generator was fine-tuned on real LS20 gameplay for **575.21 seconds**, within the 600-second limit. It made **644 optimizer updates** from **34,856 real game actions**. Independent GPU tensor comparison confirmed that all 44 generator tensors changed. The tokenizer remained unchanged.

The final frozen policy still completed **0 of 183 levels across 25 public development games**, for a **0.0% score**. This run demonstrates actual gameplay training, but no improvement in the measured game-solving result. It does not establish a Kaggle leaderboard score. The competition target and remaining gaps are documented in [ARC Prize 2026 target](arc3-competition-target.md).

| Measurement | Result |
| --- | ---: |
| Total training wall time, including initialization and save | 575.205739 seconds |
| Actual SDK game actions | 34,856 |
| Sampled policy decisions, including invalid commands | 35,355 |
| Training episodes | 645 |
| Optimizer updates | 644 |
| Completed-level rewards during exploration | 1 |
| Invalid-command penalties | 499 |
| Game-over penalties | 0 |
| Generator parameter change, L2 norm | 3.373645 |
| Peak MLX allocation | 720,447,851 bytes |
| Measured process duty with soft 80% pacing | 76.3% |
| Post-training completed levels, all 25 games | 0 / 183 |
| Post-training completed games | 0 / 25 |
| Post-training game score | 0.0% |
| Post-training score on the other 24 games | 0.0% |

The exploratory policy reached LS20's first level once, after 28 decisions in training episode 133 at seed 175. That was training experience, not a held-out result or reliable final-policy performance. Most nonzero feedback came from invalid-command penalties. Those rewards must not be presented as evidence of learning to solve puzzles.

## Training method

The run initialized the existing generator from the earlier 9m15s imitation checkpoint. It retained the same hard NCM hierarchy of 1, 3, 6 and 12 concept codes and the frozen tokenizer. There were no human-action targets in this fine-tuning run.

Each episode held one policy version fixed and sampled all 22 concept codes at temperature 1.5. The tokenizer decoded four commands; the controller submitted only the first and observed the real game response. It used the same lossless frame encoding, own-action history and strict command validation as evaluation. No invalid output was repaired.

The reward was the visible increase in completed levels, minus one for an invalid command or visible game-over response. Episodes ended on invalid output, game over, a win, the 128-decision horizon or the training deadline. The objective used discounted returns with gamma 0.99 and a baseline estimated from previous episodes only. One Adam step followed each complete rollout, with microbatches of four, a fixed loss denominator of 128 and gradient norm clipping at one. No current-episode centering, entropy reward, replay buffer or repeated off-policy updates were used.

The first sampled hierarchy's staged logits and joint log probability exactly matched the teacher-forced calculation before any update. Neural sampling and gradient computation ran on MLX Metal GPU. The SDK engine and serialization ran on the host CPU. The worker did not import Torch or use CPU neural computation.

Following the [official SDK documentation](https://docs.arcprize.org/), both training and testing used local `OFFLINE` environments. Calls to `arc.make` omitted rendering arguments and checked that the renderer was `None`. An outer process enforced the 600-second limit for the whole training process group. Training reserved time for the final checkpoint and exited normally without a hard timeout.

## Frozen evaluation

The saved checkpoint was evaluated once with greedy decoding on all 25 pinned public games at seed zero. Game revisions, human baselines and action budgets matched the earlier development evaluation. All games remained in the denominator, including failures. The policy submitted 2,182 game actions; 11 games exhausted their current level's action budget and 14 ended in policy failure. Five boards still exceeded the unchanged prefix capacity.

The evaluation reloaded the changed checkpoint and checked hash-bound inference controls. Those controls establish reload consistency, not task accuracy. The real before/after frames, actions, action budgets and score were verified independently. Training used LS20 only; its evaluation remains a training-game result. The other 24 games also scored zero.

## Reproduce and inspect

Run one bounded training pilot using the separate environments described in [the ARC-AGI-3 setup guide](../arc3/README.md):

```sh
data/arcagi3/sdk-venv/bin/python -m arc3.train_gameplay \
  --output runs/arc3-gameplay-training \
  --mlx-python "$PWD/data/arcagi3/mlx-venv/bin/python" \
  --max-seconds 600
```

The committed artifacts include the learned checkpoint, complete compressed game transitions, optimizer logs, deadline result, tensor comparison and independent verification. Verify them without neural computation:

```sh
python3 -m arc3.verify_training arc3/results/gameplay-training
python3 -m arc3.verify_score arc3/results/gameplay-trained-eval \
  --checkpoint arc3/results/gameplay-training/checkpoint
```

The initial generator hash starts with `c1c150ce`; the fine-tuned hash starts with `865332ef`. Full hashes are in the [checkpoint provenance](../arc3/results/gameplay-training/checkpoint/provenance.json). The [training verification](../arc3/results/gameplay-training/verification.json) checks visible rewards, frame chains, on-policy likelihoods and changed weights. The [evaluation verification](../arc3/results/gameplay-trained-eval/verification.json) checks the actual game score.
