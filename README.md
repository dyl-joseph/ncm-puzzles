<img src="assets/icon.svg" width="64" height="64" alt="">

# ncm-puzzles

building out puzzle adapters w/ NCM by HazyResearch

Evaluate the published frozen NCM Sudoku checkpoint on 100 order-9 Latin-square completion puzzles.

The [training pipeline](docs/latin-training.md) also trains a new Latin model from scratch with the released NCM architecture.

The [ARC-AGI-3 gameplay experiment](docs/arc3-gameplay.md) evaluates a small frozen MLX pilot in all 25 public games. A subsequent [9m35s gameplay training run](docs/arc3-gameplay-training.md) made 644 optimizer updates using real game feedback. Both frozen checkpoints completed 0 of 183 levels. The [runnable training and evaluation guide](arc3/README.md) includes the checkpoints, matched game revisions, baselines and per-game results. The [competition target](docs/arc3-competition-target.md) distinguishes these development results from the Kaggle leaderboard.

The [Colab notebook](notebooks/arc3_colab_training.ipynb) downloads the public games and continues LS20 training for up to four hours with verified checkpoints in Google Drive. It then evaluates the saved model on all 25 public games.

Run from this directory with the prepared environment and checkpoint:

```sh
.venv/bin/python evaluate_latin.py --count 100 --device mlx
```

The command creates a fresh timestamped directory under `runs/`. It prints progress and an estimated remaining time after each batch. Press Ctrl-C to save an interrupted summary. The first run fetches a small source prefix. Later runs reuse its validated cache.

The runner keeps the published hard Sudoku generator and learned tokenizer frozen. On this Mac, `--device mlx` runs the forward pass on the Apple Silicon Metal GPU. PyTorch loads the original `.pt` weights and creates small static interpolation matrices during startup. The forward pass uses MLX, with float32 weights and `MLX_ENABLE_TF32=0`. It does not train, quantize, repair outputs, or fall back to CPU inference.

To use the original upstream PyTorch operations with eight CPU threads, select `--device cpu`. The original MPS path fails on nondivisible area pooling in the quantizer. The MLX implementation handles that operation with a matrix built from PyTorch's interpolation of a basis.

The MLX implementation follows [HazyResearch NCM](https://github.com/HazyResearch/ncm) at the pinned revision below. The upstream MIT license and copyright notice are in [LICENSES/ncm-MIT.txt](LICENSES/ncm-MIT.txt).

## Inspect the results

Each run saves these files:

- `cases.jsonl` contains the masked puzzles, source line numbers, and known valid completions.
- `predictions.jsonl` contains every raw generated token, decoded output, and independent validity checks. The runner flushes predictions after each batch.
- `summary.json` contains `complete`, `failed`, or `interrupted` status, counts, overall and per-clue-count metrics, timings, and provenance hashes.

A solution passes when all rows and columns contain digits 1 through 9 exactly once and every clue is preserved. Any valid completion passes. Matching the saved completion is unnecessary. Sudoku box validity is a separate diagnostic.

The benchmark samples 100 distinct completed boards from the first 4,096 entries of [Brendan McKay's order-9 collection](https://users.cecs.anu.edu.au/~bdm/data/latin.html). It uses seed 42 and creates 25 puzzles at each clue count of 25, 37, 49, and 61. Every puzzle retains an equal-digit pair inside a 3 by 3 box. These clues make ordinary Sudoku completion impossible while preserving a valid Latin completion.

The source contains isotopy representatives with nontrivial symmetries. The bounded prefix and source collection both bias the sample. Puzzles may have multiple solutions. The checkpoint receives the same board representation used for Sudoku and no new rule instructions. This measures transfer of the frozen model's behavior under a changed scoring rule.

To run a shorter check or save to a chosen new directory:

```sh
.venv/bin/python evaluate_latin.py --count 4 --device mlx --output runs/latin-check
```

Existing output directories are rejected. Use a new directory for each run.

## Set up a fresh checkout

Use Python 3.12. The pinned dependencies are in `requirements.txt`. The MLX dependency installs on macOS with an arm64 Python interpreter. CPU evaluation also works without MLX.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p vendor artifacts/sudoku/hard
git clone https://github.com/HazyResearch/ncm.git vendor/ncm
git -C vendor/ncm checkout --detach c70274273f489abc73832b2b98584a8483b6b203
```

Download the checkpoint and its original configuration at the pinned [checkpoint revision](https://huggingface.co/hazyresearch/ncm-checkpoints/tree/7100573f83178b6e9f45ebffe0e9aa6df384e712/sudoku/hard):

```sh
for file in tokenizer.pt generator.pt config.yaml; do
	curl -fL "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/7100573f83178b6e9f45ebffe0e9aa6df384e712/sudoku/hard/$file" -o "artifacts/sudoku/hard/$file"
done
```

Validate the downloaded files and record their provenance:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import evaluate_latin as benchmark

checkpoint = Path("artifacts/sudoku/hard")
files = {}
for name, expected in benchmark.CHECKPOINT_HASHES.items():
	path = checkpoint / name
	actual = benchmark.sha256_file(path)
	if actual != expected:
		raise SystemExit(f"Checksum mismatch for {path}")
	files[name] = {"bytes": path.stat().st_size, "sha256": actual}
benchmark.write_json(checkpoint / "provenance.json", {
	"repository": "hazyresearch/ncm-checkpoints",
	"revision": benchmark.CHECKPOINT_REVISION,
	"task": "sudoku/hard",
	"files": files,
})
PY
```

The evaluation command verifies the source revision, clean checkout, checkpoint hashes, and dependency versions before fetching puzzle data. It does not install packages, train a model, or load untrusted pickle datasets. Model state dictionaries load with `weights_only=True` and strict parameter matching.

## Compare MLX with upstream PyTorch

Run the reusable parity check on this Mac:

```sh
.venv/bin/python scripts/verify_mlx.py --count 100 --output runs/mlx-parity.json
```

The check compares every greedy latent code and final raw output token on four published Sudoku controls and the same 100 Latin puzzles. It also requires both backends to solve all four Sudoku controls exactly. The control fixtures are embedded with a pinned dataset revision and source hashes. The script never executes dataset pickle files.

The report records the maximum differences in the final reconstruction and decoder logits as diagnostics. Internal floating-point activations are not identical. An earlier eight-case check found an internal block difference above `1e-4`, despite identical generated codes and tokens. Exact parity on these cases does not guarantee exact outputs on every unseen puzzle.

## Check the scorer and dataset builder

Run the scorer, corpus, MPS operator, and training tests in the prepared environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evaluate_latin.py --help
```

The tests cover valid non-Sudoku Latin completions, alternative solutions, changed clues, row and column errors, malformed raw tokens, deterministic masks, retained box conflicts, and cache rejection.

## Train Latin models

Start with the [one to two hour pilot](docs/latin-pilot.md). It trains both stages from scratch with a smaller update budget.

```sh
.venv/bin/python prepare_latin.py --output data/latin9
caffeinate -i .venv/bin/python train_latin.py all --config configs/latin9-pilot.yaml --data data/latin9 --run runs/latin9-pilot --device mps
```

The [full training guide](docs/latin-training.md) retains the complete published update budget. Its current Mac runtime estimate is about 15 days.

Training uses the original PyTorch model classes in `vendor/ncm`. The release does not include a trainer. This project adds that loop around the released architecture and published Sudoku settings. New model outputs stay under the chosen run directory. The original Sudoku weights remain in `artifacts/sudoku/hard`.

## Reuse the optimizations on another puzzle

Use the [shared puzzle runner](docs/puzzle-runtime.md) with a task module. The task owns its data, model, loss, scoring, and saved training state. The runner supplies memory guards, duty pacing, checkpoint copies, numerical settings, and resume checks.

The included 4×4 Sudoku task uses the upstream NCM classes with a smaller model and a 16-cell hierarchy. This command starts a demonstration from scratch and evaluates 32 heldout puzzles:

```sh
.venv/bin/python puzzle_runner.py --task examples.sudoku4_task --device mps \
	--run runs/sudoku4-demo -- --updates 40 --tokenizer-updates 20 --evaluate
```

Defaults retain FP32/reference attention, an 8 GiB allocator limit, an 8 GiB RSS guard, a 6 GiB available-memory reserve, and 80% duty pacing. Exact checkpoint copies and pacing are shared. Specialized NCM target reuse and Metal pooling activate only for their verified model and tensor contracts. Other shapes use the task's reference loss.

The [Latin resource guide](docs/latin-resources.md) documents the existing optimized Latin commands. The [continuation guide](docs/latin-continuation.md) covers additional updates with the saved generator and frozen tokenizer.

Created in [T3 Code](https://t3.codes).
