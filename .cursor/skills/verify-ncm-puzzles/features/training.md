# Train and resume Latin models

Training learns a tokenizer and a conditional generator, saves resumable optimizer boundaries, and exports weights for normal evaluation. The full and pilot recipes use different update schedules.

## Sub-features

- `training-tokenizer` starts real tokenizer optimization and saves its state and export.
- `training-resume` restores state and performs additional updates with the same recipe.
- `training-generator` freezes the saved tokenizer and trains the generator.
- `training-all` sequences the stages and waits for tokenizer completion before starting the generator.
- `training-recipes` selects the full or pilot schedule.
- `training-export` loads a new generator export through the user-facing evaluator.

## How to get to it (user POV)

- Run `.venv/bin/python train_latin.py tokenizer`, `generator`, or `all` with `--data` and `--run`.
- Choose `configs/latin9.yaml` for the full recipe or `configs/latin9-pilot.yaml` for the pilot.
- Add `--resume` to continue a run under its unchanged recipe.
- Use `--stop-after` for bounded real optimization. A partial-tokenizer generator check also requires `--allow-partial-tokenizer`.
- Evaluate the resulting `generator/export` through `evaluate_latin.py --local-checkpoint`.

## Driving it with control.py

Preconditions:

- Complete Launch, an MPS Doctor, and [corpus preparation](corpus.md) in this session.
- Use an idle Apple accelerator and a fresh `artifacts/training` destination.
- These commands use physical batches of two for a bounded execution check. They do not reproduce the full recipe's optimizer trajectory or training budget.

- **Tokenizer entry.** Run two updates and retain the initial checkpoint before testing resume. Require status `stopped`, a saved `state.pt`, and a tokenizer export.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" doctor "$VERIFY_RUN" --device mps
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" training-tokenizer -- \
	train_latin.py tokenizer --config configs/latin9.yaml \
	--data "$VERIFY_RUN/evidence/artifacts/corpus" --run "$VERIFY_RUN/evidence/artifacts/training" \
	--device mps --batch-size 2 --stop-after 2
cp "$VERIFY_RUN/evidence/artifacts/training/tokenizer/state.pt" \
	"$VERIFY_RUN/evidence/artifacts/tokenizer-before-resume.pt"
```

- **Resume entry.** Request one additional update under the same config and batch. Require a new durable boundary at update three, not a restart at one.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" training-tokenizer-resume -- \
	train_latin.py tokenizer --config configs/latin9.yaml \
	--data "$VERIFY_RUN/evidence/artifacts/corpus" --run "$VERIFY_RUN/evidence/artifacts/training" \
	--device mps --batch-size 2 --resume --stop-after 1
```

- **Generator entry.** Start two generator updates with the explicit partial-tokenizer permission. Require a generator state, frozen tokenizer snapshot, and inference-format export.

```sh
"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" training-generator -- \
	train_latin.py generator --config configs/latin9.yaml \
	--data "$VERIFY_RUN/evidence/artifacts/corpus" --run "$VERIFY_RUN/evidence/artifacts/training" \
	--device mps --batch-size 2 --microbatch-size 2 --allow-partial-tokenizer --stop-after 2
```

- **Read durable state.** Run the following inspection after training exits. Require tokenizer counts 2 then 3, generator count 2, advancing sampler state, changed tokenizer parameters, finite tensors, matching AdamW steps, and unchanged tokenizer bytes across the snapshot and generator export.

```sh
"$VERIFY_PYTHON" -B - "$VERIFY_RUN/evidence/artifacts" > "$VERIFY_RUN/evidence/training-check.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
from evaluate_latin import verify_local_export

artifacts = Path(sys.argv[1])
run = artifacts / "training"
before = torch.load(artifacts / "tokenizer-before-resume.pt", map_location="cpu", weights_only=True)
tokenizer = torch.load(run / "tokenizer/state.pt", map_location="cpu", weights_only=True)
generator = torch.load(run / "generator/state.pt", map_location="cpu", weights_only=True)
assert (before["completed_updates"], tokenizer["completed_updates"], generator["completed_updates"]) == (2, 3, 2)
assert before["sampler"]["cursor"] == 4 and tokenizer["sampler"]["cursor"] == 6
assert any(not torch.equal(value, tokenizer["model"][name]) for name, value in before["model"].items())
for state, updates in ((tokenizer, 3), (generator, 2)):
	assert state["optimizer"]["state"]
	assert all(int(item["step"]) == updates for item in state["optimizer"]["state"].values())
	assert all(torch.isfinite(value).all() for value in state["model"].values())
paths = [run / "tokenizer/export/tokenizer.pt", run / "generator/tokenizer-snapshot.pt",
	run / "generator/export/tokenizer.pt"]
hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
assert len(set(hashes)) == 1
export = verify_local_export(run / "generator/export")["checkpoint"]
assert export["training"]["tokenizer_updates"] == 3 and export["training"]["generator_updates"] == 2
print(json.dumps({"verified": True, "tokenizer_updates": 3, "generator_updates": 2,
	"frozen_tokenizer_sha256": hashes[0], "export": export}, indent=2))
PY
```

- **Export entry.** Run `evaluation-local-test` from [evaluation.md](evaluation.md). Require its independent four-case check to pass. A saved export alone does not prove that the inference command can use it.

- **All-stage full and pilot entries.** Exercise each recipe under a one-update bound and separate destinations. Require tokenizer update one, the printed message `Tokenizer schedule is incomplete`, and no generator state. This proves the stage gate and recipe selection. Completing both full stages remains a separate, long-running journey.

```sh
for VERIFY_RECIPE in latin9 latin9-pilot
do
	"$VERIFY_PYTHON" "$VERIFY_CONTROL" run "$VERIFY_RUN" "training-all-$VERIFY_RECIPE" -- \
		train_latin.py all --config "configs/$VERIFY_RECIPE.yaml" \
		--data "$VERIFY_RUN/evidence/artifacts/corpus" \
		--run "$VERIFY_RUN/evidence/artifacts/all-$VERIFY_RECIPE" \
		--device mps --batch-size 2 --microbatch-size 2 --stop-after 1
	test ! -e "$VERIFY_RUN/evidence/artifacts/all-$VERIFY_RECIPE/generator/state.pt"
done
```

## Gotchas

- `--stop-after` limits new updates in each invocation. It preserves the chosen full or pilot learning-rate schedule.
- Resuming with different config, batch sizes, source, dependency versions, corpus provenance, or tokenizer snapshot fails. Repeat the original flags.
- `all --stop-after 1` stops after its incomplete tokenizer stage. Do not claim generator coverage from this command.
- A partial tokenizer is allowed only for an explicit bounded generator run. Do not remove `--stop-after` from that recipe.
- Direct `train_latin.py` calls have no resource-controller guard. The guarded entry supports only generator training.
- The full and pilot completion budgets are intentionally not run by this small check. Report them as unverified when only the bounded commands ran.
- Prepared data and optimizer checkpoints are prerequisites. If they are missing, record the failure rather than substituting mocked models or editing a state file.
