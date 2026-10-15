# Compare MLX and upstream PyTorch

The parity command compares greedy latent codes and raw decoded tokens for the frozen checkpoint. It also requires both implementations to solve all four embedded published Sudoku controls exactly.

## Sub-features

- `parity-controls` requires four exact Sudoku solutions on each backend.
- `parity-latin` compares every generated latent and raw output ID on the requested Latin sample.
- `parity-report` saves counts, mismatches, activation diagnostics, and provenance to a new JSON report.

## How to get to it (user POV)

- Run `.venv/bin/python scripts/verify_mlx.py --count 100`.
- Add `--output` to retain the printed report as a file. The output path must not already exist.
- Use `--count` and `--batch-size` for a smaller explicit sample or different batching.

## Driving it with control.py

Preconditions:

- Complete Launch and an MLX Doctor. Both CPU PyTorch and MLX Metal must be available.
- No other accelerator workload or command may use the shared `.cache/latin` cache during this drive.
- The report path is absent. The script fixes the checkpoint to `artifacts/sudoku/hard` and the source cache to `.cache/latin`; it has no override flags for them.

- **Parity entry.** Run a four-Latin-case bounded comparison. Require exit `0` and a saved JSON report with `behavior_passed: true`.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" parity-frozen -- \
	scripts/verify_mlx.py --count 4 --batch-size 2 \
	--output "$VERIFY_RUN/evidence/artifacts/parity.json"
```

- **Read the result.** Reopen the saved report and require all eight cases, zero ID mismatches, and four solved controls per backend. Save the check separately from the actual report.

```sh
"$VERIFY_PYTHON" -B - "$VERIFY_RUN/evidence/artifacts/parity.json" > "$VERIFY_RUN/evidence/parity-check.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["behavior_passed"] is True
assert report["latin_count"] == 4
assert report["code_ids_compared"] == 8 * 322
assert report["raw_ids_compared"] == 8 * 89
assert report["code_mismatches"] == report["raw_mismatches"] == 0
assert report["cpu_control_exact"] == report["mlx_control_exact"] == 4
print(json.dumps({"verified": True, "report": sys.argv[1], "latin_count": 4,
	"controls_per_backend": 4, "code_ids": report["code_ids_compared"],
	"raw_ids": report["raw_ids_compared"]}, indent=2))
PY
```

## Gotchas

- `--count 4` means four Latin boards plus four Sudoku controls. Do not mistake the Latin count for the total comparison count.
- This command verifies only the frozen published checkpoint. It cannot select a local trained export.
- Floating-point activations can differ while greedy IDs match. The report treats exact IDs and solved controls as acceptance criteria.
- A report file can exist after mismatches. Read its `behavior_passed` field and the action's actual exit code.
- The script may fetch the bounded prefix into the shared cache when absent. Cleanup preserves that cache because it is outside the owned session.
- A small sample establishes parity only for its recorded cases, not unseen puzzles or a performance improvement.
