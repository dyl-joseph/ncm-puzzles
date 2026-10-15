# Run a Latin training pilot in one to two hours

Run from the project directory with the prepared environment and `data/latin9` corpus.

```sh
caffeinate -i .venv/bin/python train_latin.py all \
  --config configs/latin9-pilot.yaml \
  --data data/latin9 --run runs/latin9-pilot --device mps
```

The pilot trains both stages from scratch. It uses 500 tokenizer updates and 200 generator updates. The architecture, global batch of 1,024, generator microbatch of 32, and 322-code objective match the full pipeline.

The bounded [Mac checks](latin-pilot-verification.json) measured about 1.55 seconds per tokenizer update and 25.6 seconds per generator update. Those rates imply about 100 minutes. Validation, other workloads, and sustained heat can add time. This is an extrapolation, not a completed pilot measurement.

This shorter schedule changes the training budget. It is an initial learning experiment. It does not establish the quality of a full training run.

## Inspect progress and resume

The trainer prints progress every 10 updates. It saves a checkpoint every 25 updates and evaluates 100 fixed validation puzzles every 100 updates. Each stage writes `metrics.jsonl` and `validation.jsonl`. Its `status.json` records the state when that invocation exits.

For a bounded invocation, add `--stop-after 25`. This limits new updates in each stage. Resume with the same config and run directory.

```sh
caffeinate -i .venv/bin/python train_latin.py all \
  --config configs/latin9-pilot.yaml \
  --data data/latin9 --run runs/latin9-pilot --device mps --resume
```

Use a fresh run directory when changing the recipe. The pilot has its own warmup and cosine decay schedule. It is not an early stop of the full schedule.

## Evaluate the exported pilot

After the generator completes, evaluate 100 puzzles from the separate test split through MLX.

```sh
.venv/bin/python evaluate_latin.py \
  --local-checkpoint runs/latin9-pilot/generator/export \
  --data data/latin9 --split test --count 100 --device mlx
```

New model weights save under `runs/latin9-pilot/tokenizer/export` and `runs/latin9-pilot/generator/export`. The original Sudoku checkpoints stay in `artifacts/sudoku/hard`.

## Compare the training budgets

| Setting | Pilot | Full recipe |
| --- | ---: | ---: |
| Tokenizer updates | 500 | 10,000 |
| Generator updates | 200 | 50,000 |
| Tokenizer warmup updates | 25 | 500 |
| Generator warmup updates | 20 | 1,000 |
| Global batch | 1,024 | 1,024 |
| Approximate runtime on this Mac | 100 minutes | 15 days |

The complete selected corpus remains available. The pilot processes 512,000 training examples in its tokenizer stage and 204,800 in its generator stage. It does not visit every source board. Validation, test, and historical benchmark identities remain outside training.

The [full training guide](latin-training.md) describes the source collection, published settings, platform differences, and full-budget command.
