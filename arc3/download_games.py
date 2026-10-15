"""Download the exact public game revisions listed in the baseline file."""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
API = "https://three.arcprize.org"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=Path, default=HERE / "public-baselines.json")
    parser.add_argument("--destination", type=Path,
                        default=ROOT / "data/arcagi3/evaluation_public/environment_files")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    baselines_file = args.baselines.resolve()
    destination = args.destination.resolve()
    manifest_path = args.manifest or destination.parent / "manifest.json"
    games = json.loads(baselines_file.read_text())["games"]
    if len(games) != 25 or any(not re.fullmatch(r"[a-z0-9]{4}-[a-f0-9]{8}", game) for game in games):
        raise ValueError("Expected 25 pinned public game revisions")
    cached = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["baselines_sha256"] == digest(baselines_file):
            cached = {row["game_id"]: row for row in manifest["games"]}
    pending = []
    reused = []
    for game_id, provenance in games.items():
        name, version = game_id.split("-", 1)
        directory = destination / name / version
        previous = cached.get(game_id)
        if (previous and (directory / (name + ".py")).exists() and
                (directory / "metadata.json").exists() and
                digest(directory / (name + ".py")) == previous["source_sha256"] and
                digest(directory / "metadata.json") == previous["metadata_sha256"]):
            reused.append(previous)
        else:
            pending.append((game_id, provenance))
    key = None
    if pending:
        with urllib.request.urlopen(API + "/api/games/anonkey", timeout=20) as response:
            key = json.load(response)["api_key"]

    def download(item):
        game_id, provenance = item
        name, version = game_id.split("-", 1)
        directory = destination / name / version
        source_url = API + "/api/games/" + game_id + "/source"
        request = urllib.request.Request(source_url, headers={"X-Api-Key": key, "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            source = response.read()
        compile(source, game_id + ".py", "exec")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (name + ".py")).write_bytes(source)
        metadata = {"game_id": game_id, "title": name.upper(), "default_fps": 30,
                    "class_name": name[0].upper() + name[1:],
                    "baseline_actions": provenance["baseline_actions"], "tags": [],
                    "date_downloaded": datetime.now(timezone.utc).isoformat()}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return {"game_id": game_id, "source_url": source_url,
                "source_sha256": hashlib.sha256(source).hexdigest(), "bytes": len(source),
                "metadata_sha256": digest(directory / "metadata.json"),
                "scoring_metadata_source": provenance["source_url"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        downloaded = list(pool.map(download, pending))
    manifest = {"games": sorted(reused + downloaded, key=lambda row: row["game_id"]),
                "count": len(games), "baselines_sha256": digest(baselines_file),
                "destination": str(destination)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"count": len(games), "downloaded": len(downloaded), "reused": len(reused),
                      "bytes": sum(row["bytes"] for row in manifest["games"]),
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
