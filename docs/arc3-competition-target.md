# ARC Prize 2026 target

The target is the [ARC-AGI-3 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard). At 12:51 UTC on 15 October 2026, its public leaderboard showed Tufa Labs first at 68.91 and Yi-Chia Chen second at 67.49. The public leaderboard uses approximately half the hidden test data. Final standings use the other half.

The pilot's 0.0% on 25 public development games is not a Kaggle score. This project has no Kaggle competition score yet. Improving one training game cannot establish that the leaderboard target is close.

## Environment and submission requirements

The official [Kaggle overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview) requires a notebook with internet disabled and at most nine hours of CPU or GPU runtime. Public external data and pretrained models are allowed. The competition gateway records actions and creates the submission file.

The [official starter documentation](https://docs.arcprize.org/arc-prize-2026) describes the local-to-Kaggle workflow. The checked starter revision is `eeb1535404f321d280a8f9194bbc1d7aca5f05fc` from [ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter). A successful notebook run is a code check; the separate competition rerun produces the leaderboard score.

Local testing uses Python 3.12, `arc-agi==0.9.9`, `arcengine==0.9.3`, and the exact downloaded public game revisions. `Arcade` loads them in `OFFLINE` mode. Calls to `arc.make` omit rendering arguments and assert that the resulting renderer is `None`, following the [official quickstart](https://docs.arcprize.org/) and [environment interface](https://docs.arcprize.org/toolkit/environment_wrapper).

The current neural runtime requires MLX Metal GPU. Kaggle runs Linux with NVIDIA GPUs. MLX has a [CUDA backend](https://ml-explore.github.io/mlx/build/html/install.html), but the current Metal-only entry point is not a tested Kaggle submission. A CUDA entry point needs packaged offline dependencies and numerical checks on the target hardware.

## What the bounded training run establishes

The first gameplay-training run retains the user's 600-second limit, MLX GPU neural computation, and soft 80% process pacing. It fine-tunes the existing hard NCM concept hierarchy on actual LS20 transitions, with the tokenizer frozen. Level completions and explicit failure penalties are logged separately. A subsequent frozen 25-game evaluation measures actual play and reports the other 24 games separately from LS20.

This tests whether the training pipeline receives real game feedback, updates neural parameters, saves them correctly, and changes game-playing behavior. It is a baseline experiment, not a sufficient plan to beat 68.91.

## Work needed for a competitive entry

1. Cover every board and action type. The current fixed prefix cannot represent some public boards, and its imitation training omitted click actions.
2. Train for adaptation across games. Use held-out game splits and actual transitions; changing the seed of LS20 does not make it an unseen game.
3. Evaluate exploration, memory and planning. A four-command history and an immediate action policy do not yet demonstrate the ability to infer unfamiliar game mechanics.
4. Build and validate the offline Kaggle entry point on its target hardware. Keep the local scorer as a development check.
5. Freeze an entry and obtain an actual competition score. Only that score can establish whether it exceeds the public leaderboard; the private final score remains the ultimate outcome.

The bounded pilot completed its training run and frozen evaluation. Work stops with the PR update. No Kaggle score or leaderboard win is claimed from local results.
