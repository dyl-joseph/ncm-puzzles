# Prepare the Latin corpus

Corpus preparation downloads three pinned order-9 Latin-square archives and writes compact boards, disjoint training and held-out identities, and the historical benchmark. Repeating the command validates and reuses the same corpus.

## Sub-features

- `corpus-prepare` creates the full selected collection from an empty destination and source cache.
- `corpus-reuse` validates an existing corpus without replacing its contents.
- `corpus-integrity` exposes counts, source hashes, and split provenance for downstream training and evaluation.

## How to get to it (user POV)

- Run `.venv/bin/python prepare_latin.py --output data/latin9`.
- Add `--cache-dir` to choose where downloaded archives live.
- Run the same command again to validate and reuse the output.

## Driving it with control.py

Preconditions:

- Complete Launch and Doctor. This feature needs no accelerator workload.
- Have outbound access to the three McKay source archives named in `latin_data.py`.
- Start with absent `artifacts/corpus` and `work/corpus-source` paths in the session. Preparation needs space for the real archives and corpus.

- **Prepare entry.** Run the following command. Require exit `0`, `board_count: 2393407`, and split counts of 2,389,307 train, 2,000 validation, 2,000 test, and 100 benchmark boards.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" corpus-prepare -- \
	prepare_latin.py --output "$VERIFY_RUN/evidence/artifacts/corpus" \
	--cache-dir "$VERIFY_RUN/work/corpus-source"
```

- **Read saved output.** Reopen the actual corpus through its validated reader and inspect its manifest. Require checksummed files, disjoint complete split coverage, and the pinned three-archive description. Keep this second read as evidence.

```sh
"$VERIFY_PYTHON" -B - "$VERIFY_RUN/evidence/artifacts/corpus" > "$VERIFY_RUN/evidence/corpus-check.json" <<'PY'
import json
from pathlib import Path
import sys
from dataclasses import asdict
from latin_data import LatinCorpus, SOURCES

corpus = LatinCorpus.open(Path(sys.argv[1]))
expected = {"train": 2389307, "validation": 2000, "test": 2000, "benchmark": 100}
assert corpus.manifest["board_count"] == corpus.manifest["unique_board_count"] == 2393407
assert corpus.manifest["split_counts"] == expected
assert corpus.manifest["preparation"]["sources"] == [asdict(source) for source in SOURCES]
print(json.dumps({"verified": True, "manifest_sha256": corpus.manifest_sha256,
	"split_counts": expected, "files": corpus.manifest["files"]}, indent=2))
PY
```

- **Reuse entry.** Record corpus hashes, repeat the same preparation command with a new action label, then compare the hashes. Require exit `0` and no changed corpus file.

```sh
shasum -a 256 "$VERIFY_RUN/evidence/artifacts/corpus/"* > "$VERIFY_RUN/evidence/corpus-before-reuse.txt"
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" corpus-reuse -- \
	prepare_latin.py --output "$VERIFY_RUN/evidence/artifacts/corpus" \
	--cache-dir "$VERIFY_RUN/work/corpus-source"
shasum -a 256 "$VERIFY_RUN/evidence/artifacts/corpus/"* > "$VERIFY_RUN/evidence/corpus-after-reuse.txt"
diff "$VERIFY_RUN/evidence/corpus-before-reuse.txt" "$VERIFY_RUN/evidence/corpus-after-reuse.txt"
```

## Gotchas

- The CLI prepares the full pinned collection. It has no count flag. A small fabricated corpus does not verify this user path.
- Downloaded archives total 84,528,806 bytes. The prepared corpus occupies about 204 MB. Keep the corpus as evidence; cleanup removes only its source cache under `work`.
- Existing malformed or incompatible corpora are rejected. Do not delete a user's corpus to repair a verification run.
- Validation, test, and historical benchmark identities are reserved before augmentation. Evaluation must use the prepared split rather than sampling training identities.
- If source retrieval fails, record the URL and failure. A preexisting corpus can support other features, but cannot prove empty-cache preparation.
