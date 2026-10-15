# Evaluate Latin completions

Evaluation applies frozen Sudoku weights or a trained Latin export to masked boards. It saves every raw prediction and scores valid Latin completions that preserve the clues.

## Sub-features

- `evaluation-frozen` runs the published Sudoku checkpoint through MLX or CPU.
- `evaluation-splits` evaluates historical, validation, and test boards from a prepared corpus.
- `evaluation-local` loads a verified local generator export.
- `evaluation-refusal` rejects an existing output directory without replacing its results.
- `evaluation-interrupt` saves partial results after Ctrl-C.

## How to get to it (user POV)

- Run `.venv/bin/python evaluate_latin.py` for the default frozen historical benchmark.
- Add `--device cpu` to use upstream PyTorch or `--device mlx` for Apple Metal inference.
- Add `--data` and `--split validation`, `test`, or `benchmark` for prepared cases.
- Add `--local-checkpoint` to evaluate a saved generator export.
- Press Ctrl-C during evaluation to retain completed predictions and an interrupted summary.

## Driving it with control.py

Preconditions:

- Complete Launch and Doctor. MLX is the default below and the preferred evaluation backend. Run the CPU comparison entry only when that backend comparison is explicitly needed.
- For `evaluation-splits`, complete [corpus preparation](corpus.md) in this session.
- For `evaluation-local`, complete the bounded [training recipe](training.md), which writes `artifacts/training/generator/export`.
- Start with fresh output directories. The frozen MLX command is also in `SKILL.md`; run it once per session.

- **Frozen MLX entry.** Run the following commands. Require exit `0`, four saved predictions, a complete summary, and an independent check with `verified: true`.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" evaluation-frozen-mlx -- \
	evaluate_latin.py --count 4 --seed 42 --batch-size 2 --device mlx \
	--output "$VERIFY_RUN/evidence/artifacts/frozen-mlx" \
	--cache-dir "$VERIFY_RUN/work/source-cache"
"$VERIFY_PYTHON" "$VERIFY_CHECK" "$VERIFY_RUN/evidence/artifacts/frozen-mlx" \
	--expected-count 4 > "$VERIFY_RUN/evidence/frozen-mlx-check.json"
```

- **Explicit CPU comparison entry.** Run the following commands. Require the same artifact contract with `device: cpu`. This does not prove MLX parity; use the dedicated parity feature for that claim.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" doctor "$VERIFY_RUN" --device cpu
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" evaluation-frozen-cpu -- \
	evaluate_latin.py --count 4 --seed 42 --batch-size 2 --device cpu \
	--output "$VERIFY_RUN/evidence/artifacts/frozen-cpu" \
	--cache-dir "$VERIFY_RUN/work/source-cache"
"$VERIFY_PYTHON" "$VERIFY_CHECK" "$VERIFY_RUN/evidence/artifacts/frozen-cpu" \
	--expected-count 4 --expected-device cpu > "$VERIFY_RUN/evidence/frozen-cpu-check.json"
```

- **Prepared split entries.** Run each split separately through the loop. Require `summary.split` to equal the requested split. Test and validation cases have 30 clues. Benchmark cases retain their historical masks and span the 25, 37, 49, and 61 clue bins.

```sh
for VERIFY_SPLIT in validation test benchmark
do
	"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" "evaluation-split-$VERIFY_SPLIT" -- \
		evaluate_latin.py --data "$VERIFY_RUN/evidence/artifacts/corpus" --split "$VERIFY_SPLIT" \
		--count 4 --clue-count 30 --seed 42 --device mlx \
		--output "$VERIFY_RUN/evidence/artifacts/split-$VERIFY_SPLIT"
	"$VERIFY_PYTHON" "$VERIFY_CHECK" "$VERIFY_RUN/evidence/artifacts/split-$VERIFY_SPLIT" \
		--expected-count 4 > "$VERIFY_RUN/evidence/split-$VERIFY_SPLIT-check.json"
done
```

- **Local export entry.** Run the saved tiny generator on separate test boards. Require a complete independent check. Its summary must identify the local export, matching corpus manifest, three tokenizer updates, and two generator updates.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" evaluation-local-test -- \
	evaluate_latin.py --local-checkpoint "$VERIFY_RUN/evidence/artifacts/training/generator/export" \
	--data "$VERIFY_RUN/evidence/artifacts/corpus" --split test --count 4 --device mlx \
	--output "$VERIFY_RUN/evidence/artifacts/local-test"
"$VERIFY_PYTHON" "$VERIFY_CHECK" "$VERIFY_RUN/evidence/artifacts/local-test" \
	--expected-count 4 > "$VERIFY_RUN/evidence/local-test-check.json"
```

- **Output refusal.** After the frozen MLX entry, record its summary hash, then repeat with the same output. Require exit `2`, `Output directory already exists` on stderr, and the identical summary hash afterward. A successful refusal is an expected nonzero action, not a successful inference run.

```sh
shasum -a 256 "$VERIFY_RUN/evidence/artifacts/frozen-mlx/summary.json" > "$VERIFY_RUN/evidence/refusal-before.txt"
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" evaluation-refusal -- \
	evaluate_latin.py --count 4 --device mlx \
	--output "$VERIFY_RUN/evidence/artifacts/frozen-mlx" \
	--cache-dir "$VERIFY_RUN/work/source-cache"
shasum -a 256 "$VERIFY_RUN/evidence/artifacts/frozen-mlx/summary.json" > "$VERIFY_RUN/evidence/refusal-after.txt"
diff "$VERIFY_RUN/evidence/refusal-before.txt" "$VERIFY_RUN/evidence/refusal-after.txt"
```

- **Interrupt entry.** Run the following command from an interactive execution session. After the first `Completed` batch, press Ctrl-C once and wait for the child to exit. Require exit `130`, `summary.status: interrupted`, fewer than 4,096 completed cases, and exactly `completed` saved prediction lines. Retain the transcript. Do not run the complete-evaluation checker on an interrupted output.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" evaluation-interrupt -- \
	evaluate_latin.py --count 4096 --batch-size 2 --device mlx \
	--output "$VERIFY_RUN/evidence/artifacts/interrupted" \
	--cache-dir "$VERIFY_RUN/work/source-cache"
```

## Gotchas

- The frozen Sudoku model can solve zero Latin cases. Assert execution and independently scored output rather than a guessed solve count.
- Frozen evaluation rejects MPS because upstream nondivisible area pooling is unsupported there. MLX and CPU are distinct working paths.
- A prepared corpus and a local export must share the manifest hash. A missing or incompatible export fails this entry; frozen evaluation cannot substitute for it.
- `--split` affects prepared-corpus evaluation only. It does not select a split when `--data` is absent.
- A four-case direct historical sample spans all four clue bins. A prepared benchmark subset takes the first saved cases from the fixed 100-case sample. It is not the same set of boards as a direct four-case sample.
- The evaluator creates its output directory before most validation. A failed attempt leaves useful failure evidence and requires a new output path for a retry.
- The first empty-cache run performs a real source download. Reusing a cache covers validation and reuse, not fetching.
- Corpus and trained-export entry points are unverified until their preconditions and commands have actually run.
