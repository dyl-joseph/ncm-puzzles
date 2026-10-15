"""Install an isolated Python 3.12 without the notebook's pip, venv or ensurepip."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import urllib.parse
import urllib.request
import uuid
import zipfile


UV_VERSION = "0.13.1"
WHEELS = {
    ("Linux", "x86_64"): (
        "uv-0.13.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "696dcf6d807d85da978f4db8472da81dccc1254b18b52cbd1367c4d923c421e0"),
    ("Darwin", "arm64"): (
        "uv-0.13.1-py3-none-macosx_11_0_arm64.whl",
        "cfa719fbbd23f26713394dea2692b7f146aa1b5713ce5e776e878701214ea52c"),
}


def _download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def install_uv(tools_dir):
    wheel_name, expected_hash = WHEELS[(platform.system(), platform.machine().lower())]
    metadata = json.loads(_download(f"https://pypi.org/pypi/uv/{UV_VERSION}/json"))
    wheel = next(item for item in metadata["urls"] if item["filename"] == wheel_name)
    url = urllib.parse.urlparse(wheel["url"])
    if url.scheme != "https" or url.hostname != "files.pythonhosted.org":
        raise ValueError("The uv wheel URL is not an official HTTPS PyPI file.")
    if wheel["digests"]["sha256"] != expected_hash:
        raise ValueError("The uv PyPI wheel checksum differs from the pinned checksum.")
    content = _download(wheel["url"])
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError("The downloaded uv wheel failed SHA-256 verification.")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        executable = archive.read(f"uv-{UV_VERSION}.data/scripts/uv")
    tools_dir = Path(tools_dir)
    tools_dir.mkdir(parents=True, exist_ok=True)
    temporary = tools_dir / f"uv-{uuid.uuid4().hex}"
    temporary.write_bytes(executable)
    temporary.chmod(0o755)
    uv = tools_dir / f"uv-{UV_VERSION}"
    temporary.replace(uv)
    return uv


def _run(command):
    result = subprocess.run([str(part) for part in command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {command!r}\n{result.stdout}")


def _environment_health(venv):
    probe = "import json,sys; print(json.dumps([list(sys.version_info[:2]),sys.prefix,sys.base_prefix]))"
    try:
        result = subprocess.run([str(venv / "bin/python"), "-I", "-c", probe],
                                text=True, capture_output=True, timeout=20)
        version, prefix, base_prefix = json.loads(result.stdout)
        healthy = (result.returncode == 0 and version == [3, 12]
                   and Path(prefix).resolve() == venv.resolve() and prefix != base_prefix)
        return healthy, result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as error:
        return False, str(error)


def ensure_python(uv, venv):
    venv = Path(venv)
    healthy, _ = _environment_health(venv)
    if not healthy:
        if venv.exists() or venv.is_symlink():
            preserved = venv.with_name(f"{venv.name}.incomplete-{uuid.uuid4().hex}")
            venv.rename(preserved)
            print(f"Preserved the unusable environment at {preserved}.")
        _run([uv, "venv", "--python", "3.12", "--managed-python", venv])
    healthy, diagnostic = _environment_health(venv)
    if not healthy:
        raise RuntimeError(f"The isolated Python 3.12 environment failed its health check.\n{diagnostic}")
    return venv / "bin/python"


def bootstrap_environment(repo, venv, tools_dir):
    uv = install_uv(tools_dir)
    python = ensure_python(uv, venv)
    _run([uv, "pip", "install", "--python", python,
          "-r", Path(repo) / "arc3/requirements-sdk.txt", "mlx[cuda12]==0.32.4"])
    return uv, python
