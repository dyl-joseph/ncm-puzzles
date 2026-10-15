from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

import run_latin_10min as wrapper


class DeadlineBehaviorTests(unittest.TestCase):
    def test_normal_child_exits_without_waiting_for_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "finished"
            command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('done')", str(marker)]
            started = time.monotonic()
            self.assertEqual(wrapper.run_child(command, started, 5), (0, "completed"))
            self.assertEqual(marker.read_text(), "done")
            self.assertLess(time.monotonic() - started, 3)

    def test_deadline_requests_graceful_save_and_waits_for_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "saved"
            command = [sys.executable, "-c", """
from pathlib import Path
import signal
import sys
import time

def save(signum, frame):
    Path(sys.argv[1]).write_text('saved export at update 7')
    raise SystemExit(0)

signal.signal(signal.SIGTERM, save)
while True:
    time.sleep(0.01)
""", str(marker)]
            started = time.monotonic()
            self.assertEqual(wrapper.run_child(command, started, 2), (0, "deadline"))
            self.assertEqual(marker.read_text(), "saved export at update 7")
            self.assertLess(time.monotonic() - started, 3)

    def test_unresponsive_child_is_killed_and_reaped(self):
        command = [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"]
        started = time.monotonic()
        code, reason = wrapper.run_child(command, started, 2)
        self.assertEqual((code, reason), (-9, "killed"))
        self.assertLess(time.monotonic() - started, 3)


class SetupBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source/export"
        self.source.mkdir(parents=True)
        self.payload = b"immutable tokenizer fixture"
        (self.source / "tokenizer.pt").write_bytes(self.payload)
        (self.source / "provenance.json").write_text('{"original": true}')
        self.provenance = {
            "training": {"tokenizer_updates": 213, "stage_spec": wrapper.asdict(wrapper.stage_spec(OmegaConf.load(wrapper.CONFIG), "tokenizer"))},
            "files": {"tokenizer.pt": {"sha256": hashlib.sha256(self.payload).hexdigest()}},
            "data": {"manifest_sha256": "same-corpus"},
        }
        self.run = self.root / "run"
        self.run.mkdir()
        for name, value in (("SOURCE", self.source), ("TOKENIZER_SHA256", hashlib.sha256(self.payload).hexdigest())):
            active = patch.object(wrapper, name, value)
            active.start()
            self.addCleanup(active.stop)
        active = patch.object(wrapper, "verify_local_export", side_effect=self.verify_export)
        active.start()
        self.addCleanup(active.stop)
        active = patch.object(wrapper.LatinCorpus, "open", return_value=SimpleNamespace(manifest_sha256="same-corpus"))
        active.start()
        self.addCleanup(active.stop)

    def verify_export(self, path, *, stage):
        self.assertEqual(stage, "tokenizer")
        if (path / "tokenizer.pt").read_bytes() != self.payload:
            raise ValueError("Tokenizer checksum mismatch.")
        return {"checkpoint": self.provenance}

    def test_new_run_copies_entire_export_without_changing_source(self):
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.source.iterdir()}
        wrapper.prepare_run(self.run, False)
        copied = self.run / "tokenizer/export"
        self.assertEqual({path.name: path.read_bytes() for path in copied.iterdir()}, {name: record[0] for name, record in before.items()})
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.source.iterdir()}, before)
        self.assertFalse((self.run / "run.json").exists())
        self.assertFalse((self.run / "tokenizer/state.pt").exists())

    def test_setup_rejects_changed_tokenizer_or_schedule_before_copy(self):
        self.provenance["training"]["tokenizer_updates"] = 214
        with self.assertRaisesRegex(ValueError, "213-update"):
            wrapper.prepare_run(self.run, False)
        self.provenance["training"]["tokenizer_updates"] = 213
        self.provenance["training"]["stage_spec"]["warmup_updates"] = 25
        with self.assertRaisesRegex(ValueError, "schedule"):
            wrapper.prepare_run(self.run, False)
        self.assertFalse((self.run / "tokenizer").exists())

    def test_resume_requires_existing_generator_state(self):
        with self.assertRaisesRegex(ValueError, "--resume"):
            wrapper.prepare_run(self.run, True)
        wrapper.prepare_run(self.run, False)
        (self.run / "generator").mkdir()
        (self.run / "generator/state.pt").write_bytes(b"saved state")
        with self.assertRaisesRegex(ValueError, "--resume"):
            wrapper.prepare_run(self.run, False)
        wrapper.prepare_run(self.run, True)
        (self.run / "tokenizer/export/tokenizer.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            wrapper.prepare_run(self.run, True)

    def test_failure_does_not_label_an_older_export_ready(self):
        with patch.object(wrapper, "deadline_lock", return_value=nullcontext()), \
                patch.object(wrapper, "prepare_run"), \
                patch.object(wrapper, "run_child", return_value=(-9, "killed")), \
                patch.object(wrapper, "verify_ready") as verify_ready:
            self.assertEqual(wrapper.main(["--run", str(self.run)]), 1)
            verify_ready.assert_not_called()
        result = json.loads((self.run / "deadline-result.json").read_text())
        self.assertFalse(result["ready"])
        self.assertEqual(result["stop_reason"], "killed")
        self.assertNotIn("export", result)


if __name__ == "__main__":
    unittest.main()
