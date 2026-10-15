#!/usr/bin/env python3
"""Prepare all pinned McKay order-9 Latin-square source archives."""

import argparse
import json
from pathlib import Path

from latin_data import prepare_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/latin9"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/latin-training/source"))
    args = parser.parse_args()
    try:
        corpus = prepare_corpus(args.output, args.cache_dir)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({"directory": str(corpus.directory), "manifest_sha256": corpus.manifest_sha256,
                      "board_count": corpus.manifest["board_count"],
                      "split_counts": corpus.manifest["split_counts"],
                      "prepared_file_bytes": corpus.manifest["prepared_file_bytes"],
                      "compressed_source_bytes": corpus.manifest["compressed_source_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
