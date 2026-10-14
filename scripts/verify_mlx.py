#!/usr/bin/env python3
"""Compare frozen MLX inference with upstream PyTorch on Sudoku and Latin cases."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_latin as benchmark


CONTROL_REVISION = "03000aa19966439de99c85a58c8e3d2f895a70ee"
CONTROL_SOURCE = f"https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/{CONTROL_REVISION}/sudoku/hard"
CONTROL_PREFIX_HASHES = {
    "puzzles.pickle": "514eda7f4432e3b3427e6b3f46aa8ac7ce3f7638df3d203b4c8e2ba13104be20",
    "solutions.pickle": "111894620b4d2a2d4426089e982a1a16ca785c7af8d8ff8444c5f010f67f3628",
}
CONTROLS = (
    ("_216_7_85|76____2__|__3__27__|_523_____|_________|8_67_____|2___84_36|6_81__4__|_4__9___7",
     "921647385|764835291|583912764|452369178|179458623|836721549|217584936|698173452|345296817"),
    ("______8__|5__4___9_|__2715_46|6__2__5_7|2____8___|7_4____28|___8___1_|9_6___4_3|__35__7_9",
     "341962875|567483291|892715346|618294537|259378164|734651928|475839612|986127453|123546789"),
    ("3_____4__|42______6|8_62_____|___614__8|_1__5_6__|5__8____1|68_____9_|159__3__4|7_2___18_",
     "395168472|421937856|876245319|937614528|218359647|564872931|683421795|159783264|742596183"),
    ("____745__|84__2____|125_3__9_|____53___|_98____5_|5__8___41|6542_9_7_|_8___6___|2_1______",
     "963174528|847925136|125638497|412753689|398461752|576892341|654219873|789346215|231587964"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Latin cases, plus four fixed published Sudoku controls.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional new JSON report path. The report also prints to stdout.")
    args = parser.parse_args()
    if not 1 <= args.count <= benchmark.SOURCE_LINES or args.batch_size < 1:
        parser.error("--count must be 1 through 4096, and --batch-size must be positive.")
    if args.output and args.output.exists():
        parser.error("Choose a new --output path.")

    checkpoint = ROOT / "artifacts/sudoku/hard"
    provenance = benchmark.verify_setup(checkpoint, "mlx")
    source_path, source = benchmark.source_cache(ROOT / ".cache/latin")
    cases = benchmark.build_cases(source_path, count=args.count, seed=42)
    cpu = benchmark.load_ncm(checkpoint / "config.yaml", checkpoint, "cpu")
    mlx = benchmark.load_ncm(checkpoint / "config.yaml", checkpoint, "mlx")
    import numpy as np
    import torch

    prefixes = [cpu.tokenizer.encode("<BOS>" + puzzle + "<BOS>") for puzzle, _ in CONTROLS]
    prefixes.extend(cpu.tokenizer.encode("<BOS>" + benchmark.board_text(case.puzzle) + "<BOS>") for case in cases)
    expected_controls = [cpu.tokenizer.encode(solution) for _, solution in CONTROLS]
    report = {
        "checkpoint": provenance["checkpoint"],
        "upstream": provenance["upstream"],
        "packages": provenance["packages"],
        "backend": mlx.metadata,
        "verification_sha256": benchmark.sha256_file(Path(__file__)),
        "latin_count": len(cases),
        "source": source,
        "control_source": CONTROL_SOURCE,
        "control_fixture": "First four rows from each published pickle. Embedded literals avoid executing pickle data.",
        "control_source_first8192_sha256": CONTROL_PREFIX_HASHES,
        "control_fixture_sha256": hashlib.sha256(json.dumps(CONTROLS).encode()).hexdigest(),
        "code_ids_compared": 0,
        "raw_ids_compared": 0,
        "code_mismatches": 0,
        "raw_mismatches": 0,
        "cpu_control_exact": 0,
        "mlx_control_exact": 0,
        "final_reconstruction_max_absolute": 0.0,
        "final_decoder_logits_max_absolute": 0.0,
        "activation_diagnostics": "Final arrays only. Exact IDs determine acceptance. Internal activations need not be identical.",
    }
    for offset in range(0, len(prefixes), args.batch_size):
        batch = prefixes[offset:offset + args.batch_size]
        with torch.inference_mode():
            codes = cpu.predictor.net.generate(None, torch.tensor(batch, dtype=torch.long), **benchmark.GENERATION)
            _, levels = cpu.predictor.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            decoder_logits = cpu.predictor.vqvae.decoder(levels[-1])
            raw = decoder_logits.argmax(-1).numpy()
        output = mlx.predictor.infer(batch, trace=True)
        mlx_codes = np.asarray(output["codes"])
        mlx_raw = np.asarray(output["predictions"])
        report["code_ids_compared"] += codes.numel()
        report["raw_ids_compared"] += raw.size
        report["code_mismatches"] += int(np.count_nonzero(codes.numpy() != mlx_codes))
        report["raw_mismatches"] += int(np.count_nonzero(raw != mlx_raw))
        reconstruction_error = float(np.max(np.abs(levels[-1].numpy() - np.asarray(output["reconstruction"]))))
        decoder_error = float(np.max(np.abs(decoder_logits.numpy() - np.asarray(output["decoder_logits"]))))
        report["final_reconstruction_max_absolute"] = max(report["final_reconstruction_max_absolute"], reconstruction_error)
        report["final_decoder_logits_max_absolute"] = max(report["final_decoder_logits_max_absolute"], decoder_error)
        for index in range(min(len(batch), max(0, 4 - offset))):
            expected = expected_controls[offset + index]
            report["cpu_control_exact"] += int(raw[index].tolist() == expected)
            report["mlx_control_exact"] += int(mlx_raw[index].tolist() == expected)
    report["behavior_passed"] = (report["code_mismatches"] == report["raw_mismatches"] == 0
                                 and report["cpu_control_exact"] == report["mlx_control_exact"] == 4)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0 if report["behavior_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
