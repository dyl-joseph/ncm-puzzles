from dataclasses import asdict, replace
import json
import plistlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ncm_resource as resource


def snapshot(**changes):
    baseline = resource.MemorySnapshot(48 * resource.GIB, 24 * resource.GIB, resource.GIB,
                                       1, resource.GIB, 100, 200, 16384)
    return replace(baseline, **changes)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ResourceControllerTests(unittest.TestCase):
    def make_controller(self, *, samples=None, gpu=None, synchronize=None, **options):
        self.clock = FakeClock()
        self.sample_memory = Mock(side_effect=samples) if samples else Mock(return_value=snapshot())
        self.sync = synchronize or Mock()
        self.controller = resource.ResourceController(
            sample_memory=self.sample_memory, sample_gpu=gpu or Mock(return_value=None),
            synchronize=self.sync, clock=self.clock.clock, sleep=self.clock.sleep, **options)
        return self.controller

    def test_each_unit_waits_after_synchronizing_without_banking_old_idle(self):
        controller = self.make_controller(synchronize=lambda: setattr(self.clock, "now", self.clock.now + 2))
        controller.check_startup()
        self.clock.now = 1000
        controller.begin_unit()
        self.clock.now += 6
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [2])
        self.clock.now += 1000
        controller.begin_unit()
        controller.begin_unit()
        self.clock.now += 2
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [2, 1])
        report = controller.report()
        self.assertEqual(report["work_units"], 2)
        self.assertEqual(report["active_seconds"], 12)
        self.assertEqual(report["idle_seconds"], 3)
        self.assertEqual(report["work_duty_cycle"], .8)
        self.assertIsNone(report["stop_reason"])
        json.dumps(report, allow_nan=False)

    def test_existing_swap_does_not_block_training_but_new_writes_stop_before_sleep(self):
        controller = self.make_controller(samples=[snapshot(), snapshot(), snapshot(swapouts=101)])
        controller.check_startup()
        controller.begin_unit()
        self.clock.now += 1
        reason = controller.finish_unit()
        self.assertIn("new swap writes", reason)
        self.sync.assert_called_once()
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(controller.report()["new_swapout_bytes"], 16384)
        self.assertEqual(controller.report()["initial_memory"]["swap_used_bytes"], resource.GIB)
        with self.assertRaises(resource.ResourceLimitError):
            controller.begin_unit()

    def test_post_sync_telemetry_counts_toward_required_idle(self):
        controller = self.make_controller(synchronize=lambda: setattr(self.clock, "now", self.clock.now + 2))
        controller.begin_unit()
        self.sample_memory.side_effect = lambda: (setattr(self.clock, "now", self.clock.now + 1) or snapshot())
        self.clock.now += 6
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [1])
        report = controller.report()
        self.assertEqual(report["active_seconds"], 8)
        self.assertEqual(report["idle_seconds"], 2)
        self.assertEqual(report["post_sync_idle_seconds"], 1)
        self.assertEqual(report["sleep_seconds"], 1)
        self.assertEqual(report["work_duty_cycle"], .8)
        self.assertEqual(self.sample_memory.call_count, 2)

    def test_excess_telemetry_idle_does_not_carry_into_next_unit(self):
        controller = self.make_controller()
        controller.begin_unit()
        self.sample_memory.side_effect = lambda: (setattr(self.clock, "now", self.clock.now + 3) or snapshot())
        self.clock.now += 4
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [])
        self.sample_memory.side_effect = None
        controller.begin_unit()
        self.clock.now += 4
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [1])
        self.assertEqual(self.sample_memory.call_count, 4)
        self.assertEqual(controller.report()["active_seconds"], 8)
        self.assertEqual(controller.report()["idle_seconds"], 4)

    def test_optional_sampling_failure_counts_as_idle(self):
        def unavailable():
            self.clock.now += 1
            raise OSError("GPU sample unavailable")
        controller = self.make_controller(gpu=unavailable)
        controller.begin_unit()
        self.clock.now += 8
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [1])
        self.assertEqual(controller.report()["idle_seconds"], 2)

    def test_critical_sampling_failure_records_idle_without_sleeping(self):
        controller = self.make_controller()
        controller.begin_unit()
        def unavailable():
            self.clock.now += 1
            raise OSError("Memory sample unavailable")
        self.sample_memory.side_effect = unavailable
        self.clock.now += 8
        self.assertIn("Memory sample unavailable", controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(controller.report()["active_seconds"], 8)
        self.assertEqual(controller.report()["idle_seconds"], 1)

    def test_failed_sync_earns_no_idle_credit(self):
        def unavailable():
            self.clock.now += 1
            raise RuntimeError("Sync failed")
        controller = self.make_controller(synchronize=unavailable)
        controller.begin_unit()
        self.clock.now += 8
        self.assertIn("Sync failed", controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(controller.report()["active_seconds"], 9)
        self.assertEqual(controller.report()["idle_seconds"], 0)

    def test_interrupted_sleep_records_actual_idle_duration(self):
        controller = self.make_controller()
        def interrupted(seconds):
            self.clock.now += .5
            raise KeyboardInterrupt()
        controller._sleep = interrupted
        controller.begin_unit()
        self.clock.now += 8
        self.assertIn("KeyboardInterrupt", controller.finish_unit())
        self.assertEqual(controller.report()["active_seconds"], 8)
        self.assertEqual(controller.report()["idle_seconds"], .5)
        self.assertEqual(controller.report()["sleep_seconds"], .5)

    def test_reference_pacing_retains_full_sleep_after_telemetry(self):
        controller = self.make_controller(policy=resource.ResourcePolicy(pacing_method="full-sleep"))
        controller.begin_unit()
        self.sample_memory.side_effect = lambda: (setattr(self.clock, "now", self.clock.now + 1) or snapshot())
        self.clock.now += 8
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.clock.sleeps, [2.25])
        self.assertEqual(controller.report()["active_seconds"], 9)
        self.assertEqual(controller.report()["idle_seconds"], 2.25)

    def test_each_memory_limit_requests_stop_after_work_without_raising(self):
        cases = [(snapshot(pressure_level=2), "pressure"),
                 (snapshot(available_bytes=6 * resource.GIB - 1), "reserve"),
                 (snapshot(rss_bytes=8 * resource.GIB + 1), "RSS"),
                 (snapshot(pageouts=201), "pageouts"),
                 (snapshot(swapouts=99), "reset")]
        for changed, expected in cases:
            with self.subTest(expected=expected):
                controller = self.make_controller(samples=[snapshot(), changed])
                controller.begin_unit()
                self.clock.now += 1
                self.assertIn(expected, controller.finish_unit())
                self.assertEqual(controller.report()["work_units"], 1)
                self.assertEqual(self.clock.sleeps, [])

    def test_gpu_and_rss_are_separate_limits_on_unified_memory(self):
        controller = self.make_controller(gpu_memory_bytes=lambda: 7 * resource.GIB,
                                         samples=[snapshot(rss_bytes=7 * resource.GIB)] * 2)
        controller.begin_unit()
        self.clock.now += 1
        self.assertIsNone(controller.finish_unit())
        controller = self.make_controller(gpu_memory_bytes=Mock(side_effect=[0, 8 * resource.GIB + 1]))
        controller.begin_unit()
        self.assertIn("GPU allocations", controller.finish_unit())

    def test_missing_telemetry_blocks_startup_and_stops_applied_update(self):
        controller = self.make_controller(samples=[resource.ResourceLimitError("no telemetry")])
        with self.assertRaisesRegex(resource.ResourceLimitError, "no telemetry"):
            controller.check_startup()
        controller = self.make_controller(samples=[snapshot(), OSError("no telemetry")])
        controller.begin_unit()
        self.assertIn("no telemetry", controller.finish_unit())
        self.assertEqual(controller.report()["work_units"], 1)

    def test_interrupted_sync_returns_stop_request_to_preserve_completed_update(self):
        controller = self.make_controller(synchronize=Mock(side_effect=KeyboardInterrupt))
        controller.begin_unit()
        self.assertIn("KeyboardInterrupt", controller.finish_unit())
        self.assertEqual(controller.report()["work_units"], 1)

    def test_optional_gpu_samples_are_rate_limited_and_not_enforced_as_instantaneous_cap(self):
        gpu = Mock(side_effect=[1.0, .5])
        controller = self.make_controller(gpu=gpu)
        controller.begin_unit()
        self.clock.now += .1
        self.assertIsNone(controller.finish_unit())
        controller.begin_unit()
        self.clock.now += .1
        self.assertIsNone(controller.finish_unit())
        controller.begin_unit()
        self.clock.now += 1
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(gpu.call_count, 2)
        self.assertEqual(controller.report()["global_gpu_activity"],
                         {"samples": 2, "mean": .75, "minimum": .5, "maximum": 1.0})
        controller = self.make_controller(gpu=Mock(side_effect=OSError("unsupported counter")))
        controller.begin_unit()
        self.assertIsNone(controller.finish_unit())

    def test_startup_check_does_not_reset_swap_write_baseline(self):
        controller = self.make_controller(samples=[snapshot(), snapshot(swapouts=101)])
        controller.check_startup()
        with self.assertRaisesRegex(resource.ResourceLimitError, "new swap writes"):
            controller.check_startup()

    def test_close_checks_memory_after_checkpoint_work(self):
        cases = [(snapshot(swapouts=101), "new swap writes"),
                 (snapshot(rss_bytes=8 * resource.GIB + 1), "RSS"),
                 (snapshot(pressure_level=2), "pressure")]
        for changed, expected in cases:
            with self.subTest(expected=expected):
                controller = self.make_controller(samples=[snapshot(), snapshot(), changed])
                controller.begin_unit()
                self.clock.now += 1
                self.assertIsNone(controller.finish_unit())
                report = controller.close()
                self.assertIn(expected, report["stop_reason"])
                self.assertEqual(report["latest_memory"], asdict(changed))
                self.assertEqual(self.sync.call_count, 2)

    def test_close_fails_closed_without_raising_or_replacing_prior_reason(self):
        controller = self.make_controller(samples=[snapshot(), OSError("telemetry disappeared")])
        controller.check_startup()
        self.assertIn("telemetry disappeared", controller.close()["stop_reason"])
        controller = self.make_controller(samples=[snapshot(), snapshot(swapouts=101), OSError("later failure")])
        controller.begin_unit()
        first_reason = controller.finish_unit()
        self.assertIn("new swap writes", first_reason)
        self.assertEqual(controller.close()["stop_reason"], first_reason)

    def test_allocation_report_identifies_endpoint_measurement_scope(self):
        controller = self.make_controller()
        self.assertIn("Maximum synchronized endpoint", controller.report()["gpu_memory_measurement"])

    def test_new_unit_rechecks_memory_after_idle_without_restarting_accumulated_unit(self):
        controller = self.make_controller(samples=[snapshot(), snapshot(), snapshot(pressure_level=2)])
        controller.begin_unit()
        self.clock.now += .5
        controller.begin_unit()
        self.assertEqual(self.sample_memory.call_count, 1)
        self.clock.now += .5
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(self.sample_memory.call_count, 2)
        self.assertEqual(controller.report()["active_seconds"], 1)
        with self.assertRaisesRegex(resource.ResourceLimitError, "pressure"):
            controller.begin_unit()
        self.assertEqual(controller.report()["work_units"], 1)


class PolicyAndTelemetryTests(unittest.TestCase):
    def test_policy_rejects_invalid_caps_and_nonfinite_values(self):
        for value in (0, -.1, .81, 1, float("nan"), float("inf"), True):
            with self.subTest(duty=value), self.assertRaises(ValueError):
                resource.ResourcePolicy(gpu_duty_cycle=value)
        for field in ("memory_limit_bytes", "rss_limit_bytes", "min_available_bytes"):
            for value in (-1, float("nan"), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    resource.ResourcePolicy(**{field: value})
        for value in (0, float("nan"), float("inf")):
            with self.subTest(interval=value), self.assertRaises(ValueError):
                resource.ResourcePolicy(gpu_sample_interval_seconds=value)
        self.assertEqual(asdict(resource.ResourcePolicy())["memory_limit_bytes"], 8 * 1024 ** 3)
        with self.assertRaises(ValueError):
            resource.ResourcePolicy(pacing_method="unlimited")

    def test_mac_memory_snapshot_parses_existing_swap_and_does_not_double_count_purgeable(self):
        vm = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages active: 900.
Pages inactive: 200.
Pages speculative: 300.
Pages purgeable: 400.
Swapouts: 731289.
Pageouts: 233.
"""
        sysctl = "1\n51539607552\ntotal = 2048.00M  used = 918.25M  free = 1129.75M  (encrypted)\n"
        with patch.object(resource.sys, "platform", "darwin"), \
                patch.object(resource, "_command", side_effect=[sysctl, vm]), \
                patch.object(resource.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=42)):
            result = resource.read_memory_snapshot()
        self.assertEqual(result.available_bytes, 600 * 16384)
        self.assertEqual(result.swap_used_bytes, 962854912)
        self.assertEqual((result.swapouts, result.pageouts, result.rss_bytes), (731289, 233, 42))

    def test_missing_critical_counter_and_unsupported_platform_fail_closed(self):
        with patch.object(resource.sys, "platform", "darwin"), patch.object(resource, "_command", return_value="bad"):
            with self.assertRaisesRegex(resource.ResourceLimitError, "required macOS memory telemetry"):
                resource.read_memory_snapshot()
        with patch.object(resource.sys, "platform", "linux"):
            with self.assertRaisesRegex(resource.ResourceLimitError, "requires macOS"):
                resource.read_memory_snapshot()

    def test_gpu_counter_is_optional_and_reads_global_percent(self):
        payload = plistlib.dumps([{"PerformanceStatistics": {"Device Utilization %": 80}}])
        with patch.object(resource.subprocess, "run", return_value=SimpleNamespace(stdout=payload)):
            self.assertEqual(resource.read_gpu_fraction(), .8)
        with patch.object(resource.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(resource.read_gpu_fraction())


class BackendLimitsTests(unittest.TestCase):
    def test_mlx_credits_only_telemetry_after_successful_sync(self):
        clock = FakeClock()
        memory = Mock(return_value=snapshot())
        mx = SimpleNamespace(set_memory_limit=Mock(), set_cache_limit=Mock(),
                             synchronize=lambda: setattr(clock, "now", clock.now + 2),
                             get_active_memory=lambda: 0, get_cache_memory=lambda: 0, get_peak_memory=lambda: 0)
        controller = resource.make_mlx_controller(mx, sample_memory=memory, sample_gpu=lambda: None,
                                                 clock=clock.clock, sleep=clock.sleep)
        controller.begin_unit()
        memory.side_effect = lambda: (setattr(clock, "now", clock.now + 1) or snapshot())
        clock.now += 6
        self.assertIsNone(controller.finish_unit())
        self.assertEqual(clock.sleeps, [1])
        self.assertEqual(controller.report()["active_seconds"], 8)
        self.assertEqual(controller.report()["idle_seconds"], 2)
        self.assertEqual(controller.report()["backend"], "mlx")

    def test_mps_cap_is_set_before_any_model_allocation(self):
        mps = SimpleNamespace(recommended_max_memory=Mock(return_value=40 * resource.GIB),
                              set_per_process_memory_fraction=Mock(), synchronize=Mock(),
                              driver_allocated_memory=Mock(return_value=0))
        controller = resource.make_torch_controller(SimpleNamespace(mps=mps),
                                                     sample_memory=lambda: snapshot(), sample_gpu=lambda: None)
        mps.set_per_process_memory_fraction.assert_called_once_with(.2)
        self.assertEqual(controller.report()["backend"], "mps")

    def test_pressure_blocks_backend_initialization(self):
        torch = Mock()
        with self.assertRaises(resource.ResourceLimitError):
            resource.make_torch_controller(torch, sample_memory=lambda: snapshot(pressure_level=2))
        self.assertEqual(torch.mock_calls, [])

    def test_mlx_guideline_and_cache_cap_are_both_set_and_counted(self):
        mx = SimpleNamespace(set_memory_limit=Mock(), set_cache_limit=Mock(), synchronize=Mock(),
                             get_active_memory=Mock(return_value=7 * resource.GIB),
                             get_cache_memory=Mock(return_value=256 * 1024 ** 2),
                             get_peak_memory=Mock(return_value=7 * resource.GIB))
        controller = resource.make_mlx_controller(mx, sample_memory=lambda: snapshot(), sample_gpu=lambda: None)
        mx.set_memory_limit.assert_called_once_with(8 * resource.GIB)
        mx.set_cache_limit.assert_called_once_with(256 * 1024 ** 2)
        self.assertEqual(controller.report()["peak_gpu_allocated_bytes"], 7 * resource.GIB + 256 * 1024 ** 2)

    def test_mlx_intermediate_peak_over_budget_stops_even_after_buffers_are_freed(self):
        mx = SimpleNamespace(set_memory_limit=Mock(), set_cache_limit=Mock(), synchronize=Mock(),
                             get_active_memory=Mock(return_value=resource.GIB),
                             get_cache_memory=Mock(return_value=0),
                             get_peak_memory=Mock(return_value=resource.GIB))
        controller = resource.make_mlx_controller(mx, sample_memory=lambda: snapshot(), sample_gpu=lambda: None)
        controller.begin_unit()
        mx.get_peak_memory.return_value = 8 * resource.GIB + 1
        self.assertIn("GPU allocations", controller.finish_unit())
        self.assertEqual(controller.report()["peak_gpu_allocated_bytes"], 8 * resource.GIB + 1)
        self.assertIn("MLX allocator active-memory peak", controller.report()["gpu_memory_measurement"])

    def test_mlx_invalid_peak_telemetry_fails_closed(self):
        mx = SimpleNamespace(set_memory_limit=Mock(), set_cache_limit=Mock(), synchronize=Mock(),
                             get_active_memory=Mock(return_value=resource.GIB),
                             get_cache_memory=Mock(return_value=0),
                             get_peak_memory=Mock(return_value=resource.GIB))
        controller = resource.make_mlx_controller(mx, sample_memory=lambda: snapshot(), sample_gpu=lambda: None)
        controller.begin_unit()
        mx.get_peak_memory.return_value = float("nan")
        self.assertIn("MLX allocation telemetry", controller.finish_unit())


if __name__ == "__main__":
    unittest.main()
