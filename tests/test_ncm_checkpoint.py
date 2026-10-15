import copy
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from ncm_checkpoint import batched_cpu_tree, install_batched_cpu_tree
import train_latin as training


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


class CheckpointCPUTests(unittest.TestCase):
    def test_nested_values_are_detached_and_keep_shapes_types_and_signed_zero(self):
        source = torch.tensor([0.0, -0.0, 1.25, -3.5], requires_grad=True)
        source.grad = torch.tensor([2.0, -2.0, 4.0, -4.0])
        marker = object()
        value = {"weights": [source.reshape(2, 2), (torch.tensor(7), None)],
                 "label": "checkpoint", "metadata": (True, 0.25, marker)}
        rng = torch.get_rng_state().clone()

        def unexpected_sync():
            self.fail("A CPU-only checkpoint must not synchronize MPS")

        result = batched_cpu_tree(value, synchronize=unexpected_sync)
        self.assertEqual(list(result), ["weights", "label", "metadata"])
        self.assertIsInstance(result["weights"], list)
        self.assertIsInstance(result["weights"][1], tuple)
        weights = result["weights"][0]
        self.assertEqual(weights.shape, (2, 2))
        self.assertEqual(weights.device.type, "cpu")
        self.assertFalse(weights.requires_grad)
        self.assertIsNone(weights.grad_fn)
        self.assertEqual(tensor_bytes(weights).tolist(),
                         [0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 160, 63, 0, 0, 96, 192])
        self.assertEqual(result["weights"][1][0].item(), 7)
        self.assertIsNone(result["weights"][1][1])
        self.assertEqual(result["label"], "checkpoint")
        self.assertEqual(result["metadata"][:2], (True, 0.25))
        self.assertIs(result["metadata"][2], marker)
        self.assertTrue(source.requires_grad)
        self.assertEqual(source.grad.tolist(), [2.0, -2.0, 4.0, -4.0])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_noncontiguous_and_sparse_cpu_tensors_keep_reference_behavior(self):
        noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        noncontiguous.requires_grad_(True)
        sparse = torch.sparse_coo_tensor([[0, 2], [1, 0]], [3.0, -4.0], (3, 2),
                                        check_invariants=True)
        result = batched_cpu_tree([noncontiguous, sparse])
        self.assertEqual(result[0].stride(), (1, 4))
        self.assertEqual(result[0].tolist(),
                         [[0.0, 4.0, 8.0], [1.0, 5.0, 9.0],
                          [2.0, 6.0, 10.0], [3.0, 7.0, 11.0]])
        self.assertFalse(result[0].requires_grad)
        self.assertEqual(result[1].layout, torch.sparse_coo)
        self.assertEqual(result[1].coalesce().indices().tolist(), [[0, 2], [1, 0]])
        self.assertEqual(result[1].coalesce().values().tolist(), [3.0, -4.0])

    def test_installation_restores_original_after_exception_and_nested_use(self):
        original = lambda value: ("original", value)
        core = SimpleNamespace(cpu_tree=original)
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            with install_batched_cpu_tree(core):
                result = core.cpu_tree({"weight": torch.tensor([9.0], requires_grad=True)})
                self.assertEqual(result["weight"].tolist(), [9.0])
                self.assertFalse(result["weight"].requires_grad)
                outer = core.cpu_tree
                with install_batched_cpu_tree(core):
                    self.assertEqual(core.cpu_tree([torch.tensor(8)])[0].item(), 8)
                self.assertIs(core.cpu_tree, outer)
                raise RuntimeError("save failed")
        self.assertIs(core.cpu_tree, original)
        self.assertEqual(core.cpu_tree(11), ("original", 11))


class DeferredCopies:
    def __init__(self):
        self.pending = []
        self.events = []
        self.completed = []

    def synchronize(self):
        gc.collect()
        self.events.append("synchronize")
        for source_ref, destination_ref, payload in self.pending:
            source = source_ref()
            destination = destination_ref()
            if source is None or destination is None:
                raise AssertionError("A pending copy buffer was freed before synchronization")
            destination.payload = payload
            self.completed.append(destination.read())
        self.pending.clear()


class DeferredTensor:
    def __init__(self, queue, payload, *, device="mps", contiguous=True,
                 layout=torch.strided, fail=False):
        self.queue = queue
        self.payload = payload
        self.device = SimpleNamespace(type=device)
        self.layout = layout
        self.contiguous = contiguous
        self.fail = fail

    def detach(self):
        return copy.copy(self)

    def is_contiguous(self):
        return self.contiguous

    def to(self, *, device, non_blocking):
        if device != "cpu" or non_blocking is not True:
            raise AssertionError("The copy must be queued to CPU")
        self.queue.events.append(("queued", self.payload))
        destination = DeferredTensor(self.queue, None, device="cpu")
        self.queue.pending.append((weakref.ref(self), weakref.ref(destination), self.payload))
        return destination

    def cpu(self):
        self.queue.events.append(("blocking", self.payload))
        if self.fail:
            raise RuntimeError("fallback copy failed")
        return DeferredTensor(self.queue, self.payload, device="cpu")

    def read(self):
        if self.payload is None:
            raise AssertionError("The CPU buffer was read before its copy finished")
        return self.payload


class CheckpointCopyBoundaryTests(unittest.TestCase):
    def torch_api(self, queue):
        return SimpleNamespace(Tensor=DeferredTensor, strided=torch.strided,
                               mps=SimpleNamespace(synchronize=queue.synchronize))

    def test_all_queued_copies_complete_before_results_can_be_read(self):
        queue = DeferredCopies()
        value = {"model": DeferredTensor(queue, (1.25, -3.5)),
                 "state": [DeferredTensor(queue, (7, 9))]}
        with patch("ncm_checkpoint.torch", self.torch_api(queue)):
            result = batched_cpu_tree(value)
        self.assertEqual(result["model"].read(), (1.25, -3.5))
        self.assertEqual(result["state"][0].read(), (7, 9))
        self.assertEqual(queue.events,
                         [("queued", (1.25, -3.5)), ("queued", (7, 9)), "synchronize"])
        self.assertEqual(queue.completed, [(1.25, -3.5), (7, 9)])

    def test_unsupported_device_layout_and_noncontiguous_leaves_use_blocking_copy(self):
        queue = DeferredCopies()
        value = [DeferredTensor(queue, (1,), contiguous=False),
                 DeferredTensor(queue, (2,), layout=torch.sparse_coo),
                 DeferredTensor(queue, (3,), device="cuda"),
                 DeferredTensor(queue, (4,))]
        with patch("ncm_checkpoint.torch", self.torch_api(queue)):
            result = batched_cpu_tree(value)
        self.assertEqual([leaf.read() for leaf in result], [(1,), (2,), (3,), (4,)])
        self.assertEqual(queue.events, [("blocking", (1,)), ("blocking", (2,)),
                                        ("blocking", (3,)), ("queued", (4,)), "synchronize"])
        self.assertEqual(queue.completed, [(4,)])

    def test_traversal_error_keeps_pending_buffers_alive_until_copies_finish(self):
        queue = DeferredCopies()
        value = [DeferredTensor(queue, (5, 6)),
                 DeferredTensor(queue, (7,), contiguous=False, fail=True)]
        with patch("ncm_checkpoint.torch", self.torch_api(queue)):
            with self.assertRaisesRegex(RuntimeError, "fallback copy failed"):
                batched_cpu_tree(value)
        self.assertEqual(queue.events,
                         [("queued", (5, 6)), ("blocking", (7,)), "synchronize"])
        self.assertEqual(queue.completed, [(5, 6)])

    def test_synchronization_error_propagates_and_restores_checkpoint_converter(self):
        queue = DeferredCopies()
        original = lambda value: ("original", value)
        core = SimpleNamespace(cpu_tree=original)

        def failed_synchronize():
            queue.synchronize()
            raise RuntimeError("copy completion failed")

        with patch("ncm_checkpoint.torch", self.torch_api(queue)):
            with self.assertRaisesRegex(RuntimeError, "copy completion failed"):
                with install_batched_cpu_tree(core, synchronize=failed_synchronize):
                    core.cpu_tree(DeferredTensor(queue, (8,)))
        self.assertEqual(queue.completed, [(8,)])
        self.assertIs(core.cpu_tree, original)
        self.assertEqual(core.cpu_tree(9), ("original", 9))


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
class CheckpointMPSTests(unittest.TestCase):
    def checkpoint(self):
        floats = torch.tensor([0.0, -0.0, 1.25, -3.5], device="mps", requires_grad=True)
        floats.grad = torch.tensor([1.0, -2.0, 3.0, -4.0], device="mps")
        base = torch.arange(20, dtype=torch.float32, device="mps").reshape(4, 5)
        noncontiguous = base[:, ::2].requires_grad_(True)
        self.assertFalse(noncontiguous.is_contiguous())
        return {"model": [floats, torch.tensor([0.0, -0.0, 1.25],
                                               dtype=torch.bfloat16, device="mps")],
                "optimizer": {"step": torch.tensor(13),
                              "indices": torch.tensor([0, 2, 81], dtype=torch.int64, device="mps")},
                "views": (noncontiguous, None), "iteration": 13, "label": "latin9"}

    def assert_tree_exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.device.type, "cpu")
            self.assertFalse(actual.requires_grad)
            self.assertTrue(torch.equal(tensor_bytes(actual), tensor_bytes(expected)))
        elif isinstance(expected, dict):
            self.assertEqual(list(actual), list(expected))
            for key in expected:
                self.assert_tree_exact(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for item, reference in zip(actual, expected):
                self.assert_tree_exact(item, reference)
        else:
            self.assertEqual(actual, expected)

    def test_mixed_checkpoint_copies_preserve_bytes_gradients_and_rng(self):
        value = self.checkpoint()
        expected = training.cpu_tree(value)
        gradient = value["model"][0].grad.cpu().clone()
        cpu_rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone()
        result = batched_cpu_tree(value)
        self.assert_tree_exact(result, expected)
        self.assert_tree_exact(training.cpu_tree(value), expected)
        self.assertTrue(value["model"][0].requires_grad)
        self.assertTrue(value["views"][0].requires_grad)
        self.assertTrue(torch.equal(tensor_bytes(value["model"][0].grad), tensor_bytes(gradient)))
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertTrue(torch.equal(torch.mps.get_rng_state(), mps_rng))

    def test_atomic_save_reads_only_completed_exact_cpu_copies(self):
        value = self.checkpoint()
        expected = training.cpu_tree(value)
        original = training.cpu_tree
        cpu_rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            with install_batched_cpu_tree(training):
                training.atomic_torch_save(path, value)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            self.assert_tree_exact(restored, expected)
        self.assertIs(training.cpu_tree, original)
        self.assert_tree_exact(training.cpu_tree(value), expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertTrue(torch.equal(torch.mps.get_rng_state(), mps_rng))


if __name__ == "__main__":
    unittest.main()
