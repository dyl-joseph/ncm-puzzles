# Train NCM on 9×9 Latin-square completion

This pipeline trains a new Latin-square tokenizer and generator from scratch. It uses the original HazyResearch model classes from the pinned checkout in `vendor/ncm`. The release contains model implementations and evaluation code, but no training loop. This project supplies data preparation, optimization, checkpoint resume, and trained-model evaluation.

The downloaded Sudoku weights remain in `artifacts/sudoku/hard/tokenizer.pt` and `artifacts/sudoku/hard/generator.pt`. New training outputs belong under the chosen run directory.

## Prepare the complete selected collection

Run from the project directory with the prepared Python environment and pinned upstream checkout.

```sh
.venv/bin/python prepare_latin.py --output data/latin9
```

The three source archives contain 2,393,407 completed order-9 Latin squares. They occupy 84,528,806 compressed bytes and expand to 217,800,037 text bytes. Preparation stores compact board arrays rather than millions of text puzzle pairs. Fresh masks and symmetry variants are generated during training.

The prepared directory occupies 203,600,005 bytes. Keeping it alongside the compressed source cache takes 288,128,811 bytes, about 0.29 GB. Preparation does not save the expanded source text. Model checkpoints and training logs use additional disk space.

The collection contains isotopy representatives with nontrivial symmetries. It is a restricted family of Latin squares. Its size does not make it a uniform sample of every possible 9×9 Latin square. [Source collection](https://users.cecs.anu.edu.au/~bdm/data/latin.html).

The split assigns each source identity once, before augmentation or masking.

| Split | Base squares |
| --- | ---: |
| Training | 2,389,307 |
| Validation | 2,000 |
| Test | 2,000 |
| Historical benchmark | 100 |

The historical benchmark preserves the original 100 puzzles and their 25, 37, 49, and 61 clue counts. Its identities never enter training. Independent row, column, and digit permutations stay within the training square's isotopy class. The source supplies one representative per class, so those transformations cannot move a training identity into a held-out class. This is an isotopy split. It does not establish a split by the broader main-class equivalence.

## Train both stages

For a first experiment with a one to two hour budget, use the [pilot recipe](latin-pilot.md). The command below uses the full 10,000-update tokenizer and 50,000-update generator schedule.

```sh
caffeinate -i .venv/bin/python train_latin.py all --data data/latin9 --run runs/latin9-scratch --device mps
```

The tokenizer learns to reconstruct complete Latin boards. After its stage finishes, the generator receives masked puzzle prefixes and learns the tokenizer's solution-code hierarchy. The generator keeps an immutable snapshot of the tokenizer. It never changes the original Sudoku checkpoints.

The training mask contains 30 clues by default. Every target is a valid completion of those clues. A puzzle may have multiple valid completions. Evaluation checks Latin constraints and preserved clues rather than requiring the saved witness.

To resume the saved run, use its stored configuration.

```sh
.venv/bin/python train_latin.py all --data data/latin9 --run runs/latin9-scratch --device mps --resume
```

Use a new run directory for a different recipe. Checkpoints retain model and optimizer state, update counts, data-stream position, random-number states, and the tokenizer dependency. A requested stop saves a complete optimizer boundary. A crash resumes from the last saved boundary.

The trainer saves a resumable checkpoint every 100 updates and at a requested stop. Every 1,000 updates it evaluates 100 fixed validation cases. Stage metrics and validation results go to `metrics.jsonl` and `validation.jsonl` in each stage directory.

`caffeinate -i` keeps the Mac awake while training runs. The same prefix can wrap the resume command.

## Run a bounded pipeline check

These commands perform real optimization with smaller batches. They are execution checks, not the published training budget.

```sh
.venv/bin/python train_latin.py tokenizer --data data/latin9 --run runs/latin9-smoke --device mps --batch-size 2 --stop-after 2
.venv/bin/python train_latin.py tokenizer --data data/latin9 --run runs/latin9-smoke --device mps --resume --stop-after 1
.venv/bin/python train_latin.py generator --data data/latin9 --run runs/latin9-smoke --device mps --batch-size 2 --microbatch-size 2 --allow-partial-tokenizer --stop-after 2
```

`--stop-after` limits new updates in that invocation. It keeps the intended full training schedule intact. A partial tokenizer is permitted only through the explicit bounded generator check. The automatic full run requires complete tokenizer training.

Small-batch checks change tokenizer EMA behavior. The full recipe uses a physical tokenizer batch of 1,024 and does not accumulate tokenizer gradients. Generator microbatches accumulate to its global batch of 1,024.

## Evaluate a saved Latin model

Evaluate the unchanged historical 100-puzzle benchmark with the generator's exported weights.

```sh
.venv/bin/python evaluate_latin.py --local-checkpoint runs/latin9-scratch/generator/export --device mlx --count 100
```

Evaluate newly masked boards from the separate test split.

```sh
.venv/bin/python evaluate_latin.py --local-checkpoint runs/latin9-scratch/generator/export --data data/latin9 --split test --device mlx --count 2000 --clue-count 30
```

The evaluator verifies the export's own hashes, architecture, source revision, training settings, and corpus provenance. The original frozen-Sudoku command retains its published checkpoint checks. Each evaluation writes cases, raw predictions, validity checks, and a summary under a new `runs/` directory.

The generator export contains `tokenizer.pt`, `generator.pt`, and `config.yaml` in the original inference format. Resumable optimizer checkpoints are separate `state.pt` files under the run's `tokenizer` and `generator` directories. Every export records how many updates each stage has completed. Short smoke models have no claimed solve quality.

## Published recipe and project choices

The architecture matches the released Sudoku-hard configuration. The tokenizer has 14.90 million parameters. The generator has 25.46 million. The board uses the original 12-symbol vocabulary, 89 output tokens, a 91-token input prefix, and 12 latent levels totaling 322 codes. Generator training includes the learned level-zero query and all 322 code targets.

The following optimization settings come from the paper's Sudoku column. [NCM Tables 5 and 6](https://arxiv.org/html/2610.14162v1#A3).

| Setting | Tokenizer | Generator |
| --- | ---: | ---: |
| Optimizer updates | 10,000 | 50,000 |
| Global batch | 1,024 | 1,024 |
| Peak learning rate | 0.0005 | 0.0003 |
| Learning-rate floor | 0.00005 | 0.00003 |
| Warmup updates | 500 | 1,000 |
| AdamW betas | 0.9, 0.99 | 0.9, 0.99 |
| Weight decay | 0.01 | 0.01 |
| Gradient clip norm | 1 | 1 |

Seed 42, codebook EMA decay 0.99, and tokenizer nested dropout probability 0.5 also follow the release. The implementation preserves the original quantizer's returned VQ loss and forward-time codebook updates. Its VQ loss averages cumulative-level errors before decoder truncation. This differs from the paper's description of cutoff-only VQ loss. The source implementation is the reference here.

The release does not specify the exact Sudoku code-corruption rates, level-loss weighting, learning-rate decay curve, optimizer epsilon, or decay exclusions. This pipeline records its own choices for those gaps. It uses clean conditioning, ordinary mean code cross-entropy, and cosine decay after warmup. Latin masks, augmentation, source split, and from-scratch initialization are also project choices.

The paper uses BF16 and `torch.compile`. This Mac pipeline uses FP32 eager execution. Its scoped MPS adapter replaces unsupported area pooling and the exact stride-89 learned convolutions with differentiable mathematical equivalents. Original parameter objects and source files stay intact. The existing MLX implementation supports inference. It does not carry training gradients across copied checkpoint tensors.

These differences make this a documented Latin extension of the released method. They do not establish an exact reproduction of the unpublished trainer.

## Verified execution on this Mac

The [verification record](latin-training-verification.json) captures the source hashes, checks, and limits from the Apple M5 Pro with 48 GB of memory.

- All 68 tests passed. They include real model gradients and exact CPU equality between uninterrupted and resumed training.
- The MPS run completed five tokenizer updates with physical batch 1,024 and three generator updates with global batch 1,024. The generator accumulated 32 microbatches of 32 samples per update. Both stages saved and resumed.
- Checkpoint inspection confirmed optimizer steps, data-stream positions, finite parameters, changes to all 12 generator heads, and an unchanged frozen tokenizer snapshot.
- Validation on MPS preserved model weights, codebook state, random-number states, and train/eval modes.
- MLX loaded the trained export. Its 32,200 latent codes and 8,900 raw output tokens matched upstream PyTorch exactly on the historical 100 puzzles.
- The original frozen Sudoku model still solved 1 of the same 100 puzzles. All its output tokens matched the previous reference.

The five-update tokenizer and three-update generator solved zero puzzles on both the historical 100 cases and 100 newly masked test cases. This bounded check verifies execution. It does not measure trained-model quality.

The short resumed segments measured about 1.36 seconds per tokenizer update and 25.65 seconds per generator update. Extrapolating those measurements to the full schedule gives roughly two weeks with the current Mac settings. This is a single short-run estimate. Periodic validation, other workloads, and sustained thermal behavior can add time. The full training job has not started.
