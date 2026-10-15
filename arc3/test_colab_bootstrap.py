"""Exercise bootstrap recovery without a GPU or the host's environment installer."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from .colab_bootstrap import ensure_python, install_uv, bootstrap_environment


class PythonProcesses:
    def __init__(self):
        self.created = []

    def __call__(self, command, **_):
        if command[1] == "venv":
            environment = Path(command[-1])
            self.created.append(environment)
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_text("managed Python process boundary")
            (environment / "health.json").write_text(json.dumps([[3, 12], str(environment), "/managed-python"]))
            return subprocess.CompletedProcess(command, 0, "Created Python 3.12\n")
        health = Path(command[0]).parents[1] / "health.json"
        return subprocess.CompletedProcess(command, 0 if health.exists() else 1,
                                           health.read_text() if health.exists() else "", "")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.venv = self.root / "isolated"
        self.uv = self.root / "uv"
        self.processes = PythonProcesses()

    def test_partial_directory_is_preserved_and_recreated(self):
        self.venv.mkdir()
        (self.venv / "partial-user-file").write_text("keep this")
        with patch("arc3.colab_bootstrap.subprocess.run", self.processes):
            result = ensure_python(self.uv, self.venv)
        preserved, = self.root.glob("isolated.incomplete-*")
        self.assertEqual((preserved / "partial-user-file").read_text(), "keep this")
        self.assertEqual(result, self.venv / "bin/python")
        self.assertTrue(result.exists())
        self.assertEqual(self.processes.created, [self.venv])

    def test_healthy_environment_is_reused_without_recreation(self):
        with patch("arc3.colab_bootstrap.subprocess.run", self.processes):
            first = ensure_python(self.uv, self.venv)
            (self.venv / "installed-package").write_text("retained")
            second = ensure_python(self.uv, self.venv)
        self.assertEqual(first, second)
        self.assertEqual(self.processes.created, [self.venv])
        self.assertEqual((self.venv / "installed-package").read_text(), "retained")
        self.assertEqual(list(self.root.glob("isolated.incomplete-*")), [])

    def test_python_313_environment_is_preserved_and_replaced_with_312(self):
        self.venv.mkdir()
        health = self.venv / "health.json"
        health.write_text(json.dumps([[3, 13], str(self.venv), "/host-python"]))
        with patch("arc3.colab_bootstrap.subprocess.run", self.processes):
            ensure_python(self.uv, self.venv)
        preserved, = self.root.glob("isolated.incomplete-*")
        self.assertEqual(json.loads((preserved / "health.json").read_text())[0], [3, 13])
        self.assertEqual(json.loads(health.read_text())[0], [3, 12])

    def test_unisolated_interpreter_is_rejected(self):
        self.venv.mkdir()
        (self.venv / "health.json").write_text(json.dumps([[3, 12], str(self.venv), str(self.venv)]))
        with patch("arc3.colab_bootstrap.subprocess.run", self.processes):
            ensure_python(self.uv, self.venv)
        self.assertEqual(self.processes.created, [self.venv])

    def test_installer_failure_includes_the_actual_diagnostic(self):
        def fail(command, **kwargs):
            if command[1] == "venv":
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                return subprocess.CompletedProcess(command, 2, "Download failed. Synthetic certificate error.\n")
            return self.processes(command, **kwargs)

        with patch("arc3.colab_bootstrap.subprocess.run", fail):
            with self.assertRaisesRegex(RuntimeError, "Synthetic certificate error"):
                ensure_python(self.uv, self.venv)

    def test_package_install_uses_uv_without_pip_in_the_environment(self):
        calls = []

        def processes(command, **kwargs):
            calls.append(command)
            if command[1:3] == ["pip", "install"]:
                return subprocess.CompletedProcess(command, 0, "Installed packages\n")
            return self.processes(command, **kwargs)

        with patch("arc3.colab_bootstrap.install_uv", return_value=self.uv), \
                patch("arc3.colab_bootstrap.subprocess.run", processes):
            uv, python = bootstrap_environment(self.root / "repo", self.venv, self.root / "tools")
        self.assertEqual((uv, python), (self.uv, self.venv / "bin/python"))
        install, = [command for command in calls if command[1:3] == ["pip", "install"]]
        self.assertEqual(install, [str(self.uv), "pip", "install", "--python", str(python),
                                  "-r", str(self.root / "repo/arc3/requirements-sdk.txt"), "mlx[cuda12]==0.32.4"])

    def wheel_download(self, corrupt=False):
        wheel_name = "uv-test.whl"
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("uv-0.13.1.data/scripts/uv", b"verified Rust binary")
            archive.writestr("../../unwanted-file", b"must not extract")
        data = content.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        metadata = {"urls": [{"filename": wheel_name, "url": "https://files.pythonhosted.org/uv-test.whl",
                              "digests": {"sha256": digest}}]}
        downloads = [json.dumps(metadata).encode(), b"corrupt" if corrupt else data]
        return wheel_name, digest, downloads

    def test_verified_wheel_extracts_only_the_uv_binary(self):
        name, digest, downloads = self.wheel_download()
        with patch("arc3.colab_bootstrap.platform.system", return_value="Linux"), \
                patch("arc3.colab_bootstrap.platform.machine", return_value="x86_64"), \
                patch("arc3.colab_bootstrap.WHEELS", {("Linux", "x86_64"): (name, digest)}), \
                patch("arc3.colab_bootstrap._download", side_effect=downloads):
            uv = install_uv(self.root / "tools")
        self.assertEqual(uv.read_bytes(), b"verified Rust binary")
        self.assertTrue(uv.stat().st_mode & 0o100)
        self.assertEqual(list((self.root / "tools").iterdir()), [uv])
        self.assertFalse((self.root / "unwanted-file").exists())

    def test_corrupt_wheel_never_creates_an_executable(self):
        name, digest, downloads = self.wheel_download(corrupt=True)
        with patch("arc3.colab_bootstrap.platform.system", return_value="Linux"), \
                patch("arc3.colab_bootstrap.platform.machine", return_value="x86_64"), \
                patch("arc3.colab_bootstrap.WHEELS", {("Linux", "x86_64"): (name, digest)}), \
                patch("arc3.colab_bootstrap._download", side_effect=downloads):
            with self.assertRaisesRegex(ValueError, "SHA-256 verification"):
                install_uv(self.root / "tools")
        self.assertFalse((self.root / "tools").exists())


if __name__ == "__main__":
    unittest.main()
