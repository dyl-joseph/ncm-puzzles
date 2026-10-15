# Frozen NCM pilot on actual ARC-AGI-3 games

The trained pilot completed **0 of 183 levels across 25 public games**, for **0.0% human-relative action efficiency**. It also completed zero levels in LS20, the game whose human recordings supplied its training data. This experiment gives no evidence that this checkpoint can solve ARC-AGI-3 games.

The earlier 62.8% result measured the next human command on held-out replay inputs. It was not a gameplay score. The claim that it showed promising ARC-AGI-3 capability was too strong.

This report preserves the original imitation checkpoint's evaluation. A subsequent [actual-game training pilot](arc3-gameplay-training.md) made 644 generator updates within ten minutes, then also scored zero on the frozen 25-game test. The [competition target](arc3-competition-target.md) describes the Kaggle leaderboard and submission requirements.

## Actual gameplay

| Policy | Human-relative score | Levels | Complete games | Submitted actions |
| --- | ---: | ---: | ---: | ---: |
| Frozen NCM pilot | 0.00000% | 0 / 183 | 0 / 25 | 1,808 |
| Seeded random legal actions | 0.01132% | 1 / 183 | 0 / 25 | 4,620 |
| Repeat own previous command | 0.00000% | 0 / 183 | 0 / 25 | 4,365 |
| Published GPT-6.1 Sol, Standard Max | 64.26683% | 130 / 183 | 10 / 25 | 10,154 |

Each local policy played each game once with environment seed 0. The published Sol result comes from its [public-game result table and linked scorecards](https://arcprize.org/results/openai-gpt-6-1-sol). Its exact game-version IDs, human baselines and per-level scores are archived in [public-baselines.json](../arc3/public-baselines.json). This comparison uses those same 25 revisions and baselines. It does not use the semi-private leaderboard score.

The NCM and local baselines receive five times the human action allowance per level. The reference benchmark profiles use the same multiplier. Initial constructor reset costs zero actions and budget units. Later forced resets after game over cost actions and retain the level's budget. The allowance refreshes when the level advances. The default wall allowance is 12 hours per game, as in the reference harness; action limits ended these runs much sooner.

All 25 games remain in the denominator. Nine NCM episodes reached their first-level action limit. Sixteen ended in explicit policy failure: five boards exceeded the frozen prefix capacity, six commands selected an unavailable action, three attached coordinates to a non-click action, and two had malformed action tokens. No failed game was replaced or removed. The random agent completed the first SP80 level; its remaining levels contributed zero.

The other 24 games, excluding the training game LS20, give the same NCM score of zero. The corresponding random score is 0.01179%, and repetition scores zero. These are one-seed observations, not a statistical ranking of small differences among weak agents.

## What is comparable

The game-version IDs, scoring baselines, action IDs, per-level action allowance and score calculation match the published Sol public run. The exact game sources are downloaded from the official source endpoint and hashed. The March mirror originally downloaded for training has 15 older game revisions and older baselines for all 25 games; those were replaced in a separate evaluation cache before this result was measured.

The local game engine is `arcengine==0.9.3` with `arc-agi==0.9.9`. The hosted engine implementation and random state used by the published provider run cannot be fully audited from its scorecard. Local seed 0 matches the SDK default used by the checked reference harness. The agent interfaces also differ: this NCM pilot has a frame and four recent commands, while the Standard language-model harness can preserve notes. This is a comparison of public-game outcomes under matched visible rules, not an identical model interface or an officially verified leaderboard submission.

The reference harness revision is `d7f6961bf19738e1e8ffd5d3a982f4e0de138f79` from [ARC-AGI-3 Benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking). Scoring follows the [official methodology](https://docs.arcprize.org/methodology). A completed level receives `min(115, 100 * (human_actions / agent_actions)^2)` points; an incomplete level receives zero. A game weights later levels by their level index and caps the score at its completed-level weight fraction. The final score averages all games equally.

For wider context, averages calculated from the rounded 25-game public Standard tables are about 68.34% for [GPT-6 Astra Max](https://arcprize.org/results/openai-gpt-6-astra), 40.68% for [Claude Opus 5 High](https://arcprize.org/results/anthropic-claude-opus-5), 19.27% for [Gemini 3.8 Flash High](https://arcprize.org/results/google-gemini-3-8-flash), and 5.94% for [Grok 4.7 Medium](https://arcprize.org/results/xai-grok-4-7). Their individual game revisions were not checked here, so these are context rather than controlled comparisons. Provider Adapter results use a separate harness and are not pooled with Standard results.

## The repetition baseline

The earlier replay baseline repeats the **human's immediately previous complete command**, including click coordinates when present. If the human just moved right, it predicts that the human will move right again. It receives the human's history for every test input. It matched 50 of 94 next commands, or 53.2%, versus NCM's 59 of 94, or 62.8%.

Those 94 inputs came from two held-out recordings of LS20, after removing exact input-prefix overlaps with the training sample. Consecutive replay inputs are correlated. All nine net extra correct commands came from one recording. The evidence supports a small same-game imitation gain, not unknown-game reasoning or puzzle completion.

The online repetition policy in the table is different. It starts with the smallest legal action, uses the center pixel for a click, and then repeats its **own** command. It never receives a human trajectory. The random policy samples a legal action uniformly and samples click coordinates uniformly from the 64 by 64 board.

## Frozen model and inference

The checkpoint trained from scratch for 555.28 seconds, approximately 9 minutes 15 seconds, on MLX GPU. No further training took place for gameplay evaluation. It is a small NCM-style adaptation, not the released Sudoku checkpoint or a claim of numerical parity with the published architecture. Its dimensions and deviations are recorded in the [checkpoint provenance](../arc3/checkpoints/ls20-10min/provenance.json).

Each observation uses the final 64 by 64 animation frame with lossless row run-length encoding. The model has a fixed 1,142-token prefix. Four missing historical commands are padded; history clears at resets and level transitions. Those cold starts were excluded from the imitation training windows, so padding is an adapter convention outside that training distribution.

The model generates four commands, submits only the first, observes the actual result, and replans. It receives no human future actions, game source, hidden state or human scoring baselines. Malformed or unavailable first commands terminate the policy. There is no output repair or random fallback.

Neural inference requires MLX Metal GPU. The environment engine and encoding run on the CPU. The worker synchronizes each inference and sleeps for one quarter of its active time to target a soft 80% process duty. This is not a hardware utilization ceiling. The original measured run reported 76.2% process duty, no Torch import, no CPU neural inference, and about 255 MB peak MLX allocation.

## Verification and reproduction

The evaluator saves the official SDK scorecard, every real before/after frame and command, explicit failures, and provenance hashes. Independent checks recompute the score, action totals, frame hashes, own-action history, budgets, all-game denominator and unchanged checkpoint hashes. Four frozen replay controls confirm exact tokenization and saved-model predictions before gameplay. See the [machine-readable results](../arc3/results/gameplay.json) and [run instructions](../arc3/README.md).

The report concerns one small checkpoint with a narrow training task. It does not establish a general limit on NCM. It establishes that this pilot's replay imitation gain did not translate into demonstrated game-solving ability.

## Public claim

> Our 9m15s MLX NCM pilot scored 0.0% in local ARC-AGI-3 gameplay: 0/183 levels across 25 public games, using matching versions and action budgets. It learned some human-action imitation, but this prototype has not demonstrated game-solving ability.
