import importlib.util
import json
import pathlib
import subprocess
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tests/ascend/production/run_logical_scale_out_smoke.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "logical_scale_out_smoke", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Dist:
    def __init__(self, rank=0, peer_failures=None):
        self.rank = rank
        self.peer_failures = peer_failures or {}

    def get_rank(self, group):
        del group
        return self.rank

    def get_world_size(self, group):
        del group
        return 4

    def all_gather_object(self, output, value, group):
        del group
        output[:] = [None] * 4
        output[self.rank] = value
        for rank, failure in self.peer_failures.items():
            output[rank] = failure


class _Destroyable:
    def __init__(self):
        self.destroyed = False
        self.destroy_calls = 0

    def destroy(self):
        self.destroyed = True
        self.destroy_calls += 1


class _FailingDestroyable:
    def __init__(self):
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1
        raise RuntimeError(f"buffer destroy {self.destroy_calls}")


class LogicalScaleOutSmokeContractTest(unittest.TestCase):
    def test_contract_describes_literal_2x2_route_and_reference_fixture(self):
        result = subprocess.run(
            ["python3", str(RUNNER), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = json.loads(result.stdout)
        self.assertEqual(contract, {
            "barrier_generations": 100,
            "barrier_timeout_seconds": 240,
            "case_names": [
                "symmetric-all-to-all",
                "asymmetric-routing",
                "empty-source-ranks",
                "cached-handle-reuse",
            ],
            "contract_checks": [
                "literal-independent-fixtures",
                "literal-expected-receives",
                "bounded-step-elapsed",
                "collective-rank-error-aggregation",
                "cached-handle-identity",
                "dispatch-output-validation",
                "combine-round-trip-validation",
                "aggregate-teardown",
            ],
            "environment": {
                "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
                "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
            },
            "evidence": "logical-single-host",
            "expected_logical_domain": [2, 2],
            "expected_physical_domain": [1, 4],
            "expected_world_size": 4,
            "rank_mapping": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "route_matrix": {
                "0": {"diagonal": 3, "local": 1, "rail": 2, "self": 0},
                "1": {"diagonal": 2, "local": 0, "rail": 3, "self": 1},
                "2": {"diagonal": 1, "local": 3, "rail": 0, "self": 2},
                "3": {"diagonal": 0, "local": 2, "rail": 1, "self": 3},
            },
            "step_timeout_seconds": 30,
            "system_under_test": [
                "ElasticBuffer.barrier",
                "ElasticBuffer.dispatch",
                "ElasticBuffer.combine",
            ],
        })
        self.assertNotIn("roce", result.stdout.lower())

    def test_environment_and_world_are_exact_not_minimums(self):
        runner = _load_runner()
        runner._validate_environment({
            "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
            "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
        })
        for environment in (
                {},
                {"DEEP_EP_ASCEND_LOGICAL_SIMULATION": "0",
                 "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2"},
                {"DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
                 "DEEP_EP_ASCEND_SCALE_UP_SIZE": "4"}):
            with self.assertRaises(RuntimeError):
                runner._validate_environment(environment)
        for world_size in (1, 2, 8):
            with self.assertRaises(RuntimeError):
                runner._validate_world(world_size)
        runner._validate_world(4)

    def test_case_specs_use_literal_inputs_and_expected_world_source_order(self):
        runner = _load_runner()
        specs = runner._case_specs()
        self.assertEqual(tuple(specs), (
            "symmetric-all-to-all", "asymmetric-routing",
            "empty-source-ranks", "cached-handle-reuse"))

        symmetric = specs["symmetric-all-to-all"]
        self.assertEqual(symmetric.payloads, (
            ((0, 1, 2, 3), (8, 9, 10, 11),
             (16, 17, 18, 19), (24, 25, 26, 27)),
            ((64, 65, 66, 67), (72, 73, 74, 75),
             (80, 81, 82, 83), (88, 89, 90, 91)),
            ((128, 129, 130, 131), (136, 137, 138, 139),
             (144, 145, 146, 147), (152, 153, 154, 155)),
            ((192, 193, 194, 195), (200, 201, 202, 203),
             (208, 209, 210, 211), (216, 217, 218, 219)),
        ))
        self.assertEqual(symmetric.routes, (
            (0, 1, 2, 3), (1, 0, 3, 2),
            (2, 3, 0, 1), (3, 2, 1, 0)))
        self.assertEqual(
            symmetric.expected_receives,
            (
                ((0, 1, 2, 3), (72, 73, 74, 75),
                 (144, 145, 146, 147), (216, 217, 218, 219)),
                ((8, 9, 10, 11), (64, 65, 66, 67),
                 (152, 153, 154, 155), (208, 209, 210, 211)),
                ((16, 17, 18, 19), (88, 89, 90, 91),
                 (128, 129, 130, 131), (200, 201, 202, 203)),
                ((24, 25, 26, 27), (80, 81, 82, 83),
                 (136, 137, 138, 139), (192, 193, 194, 195)),
            ))

        asymmetric = specs["asymmetric-routing"]
        self.assertEqual(asymmetric.payloads, (
            ((30, 31, 32, 33), (34, 35, 36, 37), (38, 39, 40, 41)),
            ((60, 61, 62, 63),),
            ((90, 91, 92, 93), (94, 95, 96, 97),
             (98, 99, 100, 101), (102, 103, 104, 105)),
            ((120, 121, 122, 123), (124, 125, 126, 127)),
        ))
        self.assertEqual(asymmetric.routes, (
            (0, 0, 3), (2,), (1, 1, 1, 3), (0, 2)))
        self.assertEqual(asymmetric.expected_receives, (
            ((30, 31, 32, 33), (34, 35, 36, 37),
             (120, 121, 122, 123)),
            ((90, 91, 92, 93), (94, 95, 96, 97), (98, 99, 100, 101)),
            ((60, 61, 62, 63), (124, 125, 126, 127)),
            ((38, 39, 40, 41), (102, 103, 104, 105)),
        ))

        empty = specs["empty-source-ranks"]
        self.assertEqual(empty.payloads, (
            ((140, 141, 142, 143), (144, 145, 146, 147)), (),
            ((160, 161, 162, 163), (164, 165, 166, 167)), ()))
        self.assertEqual(empty.routes, ((0, 2), (), (1, 3), ()))
        self.assertEqual(empty.expected_receives, (
            ((140, 141, 142, 143),), ((160, 161, 162, 163),),
            ((144, 145, 146, 147),), ((164, 165, 166, 167),)))

        cached = specs["cached-handle-reuse"]
        self.assertEqual(cached.routes, (
            (0, 3), (1, 2), (2, 1), (3, 0)))
        self.assertEqual(cached.payloads, (
            ((10, 11, 12, 13), (14, 15, 16, 17)),
            ((50, 51, 52, 53), (54, 55, 56, 57)),
            ((90, 91, 92, 93), (94, 95, 96, 97)),
            ((130, 131, 132, 133), (134, 135, 136, 137)),
        ))
        self.assertEqual(cached.expected_receives, (
            ((10, 11, 12, 13), (134, 135, 136, 137)),
            ((50, 51, 52, 53), (94, 95, 96, 97)),
            ((54, 55, 56, 57), (90, 91, 92, 93)),
            ((14, 15, 16, 17), (130, 131, 132, 133)),
        ))
        self.assertEqual(cached.cached_payloads, (
            ((18, 19, 20, 21), (22, 23, 24, 25)),
            ((58, 59, 60, 61), (62, 63, 64, 65)),
            ((98, 99, 100, 101), (102, 103, 104, 105)),
            ((138, 139, 140, 141), (142, 143, 144, 145)),
        ))
        self.assertEqual(cached.cached_expected_receives, (
            ((18, 19, 20, 21), (142, 143, 144, 145)),
            ((58, 59, 60, 61), (102, 103, 104, 105)),
            ((62, 63, 64, 65), (98, 99, 100, 101)),
            ((22, 23, 24, 25), (138, 139, 140, 141)),
        ))

    def test_synchronized_step_aggregates_rank_error_details_and_elapsed_bound(self):
        runner = _load_runner()
        self.assertEqual(
            runner._synchronized_step(
                _Dist(rank=2), "group", lambda: 7, "ok",
                timeout_seconds=5),
            7)
        with self.assertRaises(RuntimeError) as local_failure:
            runner._synchronized_step(
                _Dist(rank=2), "group",
                lambda: (_ for _ in ()).throw(ValueError("local")), "step",
                timeout_seconds=5)
        self.assertIn('"rank": 2', str(local_failure.exception))
        self.assertIn('"error_type": "ValueError"',
                      str(local_failure.exception))
        self.assertIn('"message": "local"', str(local_failure.exception))

        peer_record = {
            "elapsed_seconds": 0.25,
            "error_type": "KeyError",
            "label": "step",
            "message": "peer",
            "rank": 1,
        }
        with self.assertRaises(RuntimeError) as peer_failure:
            runner._synchronized_step(
                _Dist(rank=2, peer_failures={1: peer_record}), "group",
                lambda: 7, "step", timeout_seconds=5)
        self.assertIn('"rank": 1', str(peer_failure.exception))
        self.assertIn('"error_type": "KeyError"',
                      str(peer_failure.exception))
        self.assertIn('"message": "peer"', str(peer_failure.exception))

        with mock.patch.object(
                runner.time, "monotonic", side_effect=(10.0, 15.25)):
            with self.assertRaises(RuntimeError) as timeout_failure:
                runner._synchronized_step(
                    _Dist(rank=3), "group", lambda: 7, "slow",
                    timeout_seconds=5)
        self.assertIn('"rank": 3', str(timeout_failure.exception))
        self.assertIn('"error_type": "TimeoutError"',
                      str(timeout_failure.exception))
        self.assertIn("exceeded 5.000s", str(timeout_failure.exception))

    def test_cleanup_exercises_idempotent_destroy(self):
        runner = _load_runner()
        buffer = _Destroyable()
        runner._destroy_twice(buffer)
        self.assertTrue(buffer.destroyed)
        self.assertEqual(buffer.destroy_calls, 2)

    def test_cleanup_aggregates_buffer_and_process_group_failures(self):
        runner = _load_runner()
        buffer = _FailingDestroyable()

        def synchronized_step(operation, label):
            self.assertEqual(label, "buffer teardown")
            return operation()

        def destroy_process_group():
            raise OSError("process group destroy")

        with self.assertRaises(RuntimeError) as failure:
            runner._cleanup_runtime(
                buffer, synchronized_step, destroy_process_group)
        self.assertEqual(buffer.destroy_calls, 2)
        message = str(failure.exception)
        self.assertIn("buffer destroy 1", message)
        self.assertIn("buffer destroy 2", message)
        self.assertIn("process group destroy", message)


if __name__ == "__main__":
    unittest.main()
