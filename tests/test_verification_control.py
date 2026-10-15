import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verification_control", ROOT / ".cursor/skills/verify-ncm-puzzles/scripts/control.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class PuzzleRuntimeControlTests(unittest.TestCase):
    def test_task_separator_and_options_reach_the_real_cli_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory).resolve()
            run = session / "evidence/artifacts/sudoku4"
            arguments = ["puzzle_runner.py", "--task", "examples.sudoku4_task",
                         "--device", "mps", "--run", str(run), "--", "--updates", "2"]
            command = control.isolated_arguments(session, arguments)
        self.assertEqual(command, [str(control.PYTHON), "-B", "-u",
                                   str(ROOT / "puzzle_runner.py"), *arguments[1:]])

    def test_task_destination_cannot_escape_the_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "must name a child"):
                control.isolated_arguments(session, ["puzzle_runner.py", "--task", "examples.sudoku4_task",
                                                     "--run", str(session / "outside"), "--", "--updates", "2"])
