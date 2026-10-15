"""Build the uploadable Colab notebook from an immutable public source revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(revision):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character Git commit")
    cells = [cell("markdown", """
        # Train NCM on ARC-AGI-3 in Colab

        Choose **Runtime > Change runtime type > GPU**, then **Runtime > Run all**.
        A T4, L4 or A100 with a compatible NVIDIA driver is required. Setup stops if this runtime
        cannot run MLX CUDA. Setup installs its own isolated Python 3.12 using a verified uv
        binary. It does not require the notebook kernel's pip, venv or ensurepip.

        The default is **4 cumulative hours reserved for training processes**, about 25 times the
        earlier 9m35s pilot. Setup, the synthetic GPU check, compression, Drive copies and final
        evaluation take additional time. Colab availability and session duration are not guaranteed.

        This uses the small NCM prototype with 204,608 trainable generator parameters and a frozen
        tokenizer. It continues the bundled LS20 gameplay checkpoint and downloads all 25 official public
        game sources once and then runs headless offline games. Training uses **LS20 only** and
        native level-completion rewards plus invalid-action and game-over penalties. It does not
        download human action labels. MLX handles all neural work on the GPU. SDK simulation,
        observation preparation, file I/O and compression use the CPU. Soft 80% pacing is not an
        instantaneous GPU utilization cap.

        Every block starts from the previous independently verified **weights**. Adam state, RNG,
        the EWMA reward baseline and policy counters reset every block. This is a chain of weight
        warm starts, not an exact optimizer resume. The unchanged model has fixed board-prefix
        capacity and strict command/click limits. More training may still produce zero solves.

        The evaluation cell tests the frozen checkpoint on the 25 public development games. Its score
        is **not a Kaggle competition score**. There is no automatic Kaggle submission.

        Sources: [ARC Prize SDK](https://docs.arcprize.org/),
        [offline toolkit](https://docs.arcprize.org/toolkit/arc_agi),
        [MLX CUDA requirements](https://ml-explore.github.io/mlx/build/html/install.html),
        [Colab runtime limits](https://research.google.com/colaboratory/faq.html).
    """), cell("code", """
        TRAIN_HOURS = 4
        RUN_NAME = "arc3-ls20-4h-v4"
        SAVE_TO_DRIVE = True
        BASE_SEED = 42
        BLOCK_SECONDS = 600

        import os
        import platform
        import re
        import subprocess
        import sys
        from pathlib import Path

        os.environ["MLX_ENABLE_TF32"] = "0"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", RUN_NAME):
            raise ValueError("RUN_NAME must contain 1 to 80 letters, numbers, hyphens or underscores.")
        if not 1 / 120 <= TRAIN_HOURS <= 24 or not 30 <= BLOCK_SECONDS <= 600:
            raise ValueError("Use at most 24 hours total and 30 to 600 seconds per block.")
        libc, version = platform.libc_ver()
        if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 35):
            raise RuntimeError("MLX CUDA requires Linux glibc 2.35 or later.")
        try:
            query = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader,nounits"], text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError("Enable a Colab GPU runtime, then rerun from the first cell.") from error
        gpu = [part.strip() for part in query.strip().splitlines()[0].split(",")]
        driver = tuple(map(int, gpu[2].split(".")))
        if float(gpu[1]) < 7.5 or driver < (550, 54, 14):
            raise RuntimeError("MLX CUDA needs GPU compute capability >= 7.5 and driver >= 550.54.14.")
        print(f"GPU {gpu[0]}. Compute capability {gpu[1]}. Driver {gpu[2]}. Memory {gpu[3]} MiB.")
        print(f"Training reservation {TRAIN_HOURS} hours. Setup, archiving and evaluation are additional.")
    """), cell("markdown", """
        ## Save and resume

        Leave Drive enabled to retain verified blocks across Colab sessions. Approve Colab's native
        Drive mount when prompted. Rerun with the same RUN_NAME, hours, seed and source revision to
        resume. A failed or disconnected block consumes its entire reserved allowance and its
        partial weights are never selected. Rerunning after a block failure tries a fresh block
        from the last verified checkpoint. Blocks without optimizer updates stay failed and charged,
        and the run continues from the last verified weights. Other errors stop the run.
        Local-only storage disappears when Colab resets. Allow several GB for evidence.

        This notebook starts a new v4 run from the bundled checkpoint. Your older runs stay
        in their original folders. Open this notebook in a fresh Colab session so its Python imports
        use the pinned source below.
    """), cell("code", """
        if SAVE_TO_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")
            STORE = Path("/content/drive/MyDrive/arc3-ncm") / RUN_NAME
        else:
            STORE = Path("/content/arc3-local-results") / RUN_NAME
        LOCAL = Path("/content/arc3-work") / RUN_NAME
        LOCAL.mkdir(parents=True, exist_ok=True)
        print(f"Persistent run store {STORE}")
    """), cell("code", f'''
        REVISION = "{revision}"
        REPO_URL = "https://github.com/dyl-joseph/ncm-puzzles.git"
        REPO = Path("/content/ncm-puzzles-colab")
        VENV = Path("/content/arc3-isolated-venv")

        if not REPO.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO)], check=True)
        origin = subprocess.check_output(["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True).strip()
        if origin != REPO_URL:
            raise RuntimeError("The existing Colab source directory belongs to another repository.")
        subprocess.run(["git", "-C", str(REPO), "fetch", "--depth", "1", "origin", REVISION], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", REVISION], check=True)
        actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        if actual != REVISION:
            raise RuntimeError("The source checkout did not match the pinned revision.")
        changes = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True)
        if changes.strip():
            raise RuntimeError("The source checkout has local edits. Preserve them and use a clean checkout.")
        os.chdir(REPO)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from arc3.colab_bootstrap import bootstrap_environment
        from arc3.colab_training import run_process
        UV, PYTHON = bootstrap_environment(REPO, VENV, Path("/content/arc3-bootstrap-tools"))
        print(f"Pinned source {{actual}}. Isolated neural interpreter {{PYTHON}}.")
    '''), cell("markdown", """
        ## Check the SDK and actual model on CUDA

        First check that the isolated interpreter can import the game SDK before any training
        time is reserved.

        This short check samples the real latent hierarchy, verifies teacher likelihoods, applies
        one synthetic gradient to disposable weights and reloads their inference controls. It is
        a backend check, not a game result. The bundled starting checkpoint stays unchanged. A
        CUDA failure stops here before the long job. No neural work runs in the notebook kernel.
    """), cell("code", """
        import json

        run_process([PYTHON, "-c", "from arc_agi import Arcade, OperationMode; print('SDK import passed.')"],
                    timeout=60, log=LOCAL / "sdk-import.log")
        INITIAL_CHECKPOINT = REPO / "arc3/results/gameplay-training/checkpoint"
        run_process([PYTHON, "-m", "arc3.gpu_selfcheck", "--checkpoint", INITIAL_CHECKPOINT,
                     "--require-cuda", "--output", LOCAL / "gpu-selfcheck.json"], timeout=180,
                    log=LOCAL / "gpu-selfcheck.log")
        print(json.dumps(json.loads((LOCAL / "gpu-selfcheck.json").read_text()), indent=2))
    """), cell("markdown", """
        ## Download the public games

        The official source API supplies an anonymous download key in memory. No account token
        or manual dataset upload is needed. Existing files are reused only when their hashes
        match. Training and evaluation disable network access after this download.
    """), cell("code", """
        run_process([PYTHON, "-m", "arc3.download_games"], timeout=300)
        ENVIRONMENTS = REPO / "data/arcagi3/evaluation_public/environment_files"
        GAME_MANIFEST = ENVIRONMENTS.parent / "manifest.json"
        COMMON = ["--store", STORE, "--local", LOCAL, "--revision", REVISION,
                  "--hours", TRAIN_HOURS, "--block-seconds", BLOCK_SECONDS, "--seed", BASE_SEED,
                  "--checkpoint", INITIAL_CHECKPOINT, "--manifest", GAME_MANIFEST,
                  "--environments-dir", ENVIRONMENTS]
    """), cell("markdown", f"""
        ## Train or continue verified weight warm starts

        This cell launches the game controller and GPU trainer. The
        [NCM generator](https://github.com/dyl-joseph/ncm-puzzles/blob/{revision}/arc3/mlx_ncm.py)
        receives real game rewards through the
        [policy-gradient update](https://github.com/dyl-joseph/ncm-puzzles/blob/{revision}/arc3/gameplay_worker.py#L163).
        The tokenizer stays frozen.

        Progress appears after every block. Per-episode logs stay in the local work directory.
        Every clean block is independently verified, its complete trace is compressed and checked
        again, and its immutable completion manifest is written last in the run store. Interrupted
        mutable checkpoints are discarded from selection. Blocks without optimizer updates are
        skipped and never published as trained checkpoints. Other block failures stop the notebook.
    """), cell("code", """
        run_process([PYTHON, "-m", "arc3.colab_training", *COMMON],
                    timeout=int(TRAIN_HOURS * 3600) + 6 * 3600)
    """), cell("markdown", """
        ## Test the final frozen checkpoint

        This runs one strict game test across all 25 public games with the official scoring
        baselines and five times the human action allowance per level. It reports LS20 and the
        other 24 games separately. Invalid commands and unsupported board sizes count as failures.
        A verified finished evaluation is reused on rerun. An interrupted evaluation is preserved
        and can be rerun on the same trained checkpoint without repeating training. Evaluation is outside the
        four-hour training reservation and retains the reference 12-hour maximum per game.
    """), cell("code", """
        run_process([PYTHON, "-m", "arc3.colab_training", *COMMON, "--evaluate"],
                    timeout=25 * 12 * 3600 + 300)
        latest = json.loads((STORE / "latest.json").read_text())
        print(f"Final checkpoint {latest['checkpoint']}")
        print(f"Verified game results {STORE / 'frozen-evaluation'}")
        print("This is a public development score, not a Kaggle leaderboard score.")
    """), cell("markdown", """
        ## Optional checkpoint download

        The run store already holds the weights and complete evidence. Run the next cell if you
        also want a small checkpoint ZIP on your computer. It does not include the full traces.
    """), cell("code", """
        import shutil
        from google.colab import files

        latest = json.loads((STORE / "latest.json").read_text())
        checkpoint = Path(latest["checkpoint"])
        archive = shutil.make_archive(str(LOCAL / "arc3-final-checkpoint"), "zip", root_dir=checkpoint)
        files.download(archive)
    """)]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"name": "arc3_colab_training.ipynb"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}, "source_revision": revision}, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=Path("notebooks/arc3_colab_training.ipynb"))
    args = parser.parse_args()
    artifact = notebook(args.revision)
    for index, item in enumerate(artifact["cells"]):
        if item["cell_type"] == "code":
            compile(item["source"], f"colab-cell-{index}", "exec")
        item["id"] = f"arc3-{index:02d}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
