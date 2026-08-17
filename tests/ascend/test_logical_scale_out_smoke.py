import importlib.util
import json
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tests/ascend/production/run_logical_scale_out_smoke.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "logical_scale_out_smoke", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Flag:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Torch:
    int32 = "int32"

    @staticmethod
    def tensor(values, dtype, device):
        del dtype, device
        return _Flag(values[0])


class _Dist:
    def __init__(self, peer_failed=False):
        self.peer_failed = peer_failed

    def all_reduce(self, flag, group):
        del group
        flag.value += int(self.peer_failed)


class _Destroyable:
    def __init__(self):
        self.destroyed = False
        self.destroy_calls = 0

    def destroy(self):
        self.destroyed = True
        self.destroy_calls += 1


class LogicalScaleOutSmokeContractTest(unittest.TestCase):
    def test_contract_describes_literal_2x2_route_and_reference_fixture(self):
        result = subprocess.run(
            ["python3", str(RUNNER), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = json.loads(result.stdout)
        self.assertEqual(contract, {
            "barrier_generations": 100,
            "cases": ["barrier", "bf16-all-to-all", "combine-round-trip"],
            "environment": {
                "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
                "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
            },
            "evidence": "logical-single-host",
            "expected_domain": [2, 2],
            "expected_world_size": 4,
            "rank_mapping": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "route_matrix": {
                "0": {"diagonal": 3, "local": 1, "rail": 2, "self": 0},
                "1": {"diagonal": 2, "local": 0, "rail": 3, "self": 1},
                "2": {"diagonal": 1, "local": 3, "rail": 0, "self": 2},
                "3": {"diagonal": 0, "local": 2, "rail": 1, "self": 3},
            },
            "rank0_expected_receive": [
                [0, 1, 2, 3],
                [72, 73, 74, 75],
                [144, 145, 146, 147],
                [216, 217, 218, 219],
            ],
            "expected_local_topk": [[0], [0], [0], [0]],
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

    def test_reference_receive_uses_world_source_order(self):
        runner = _load_runner()
        gathered_payloads = [
            [[0, 1, 2, 3], [8, 9, 10, 11],
             [16, 17, 18, 19], [24, 25, 26, 27]],
            [[64, 65, 66, 67], [72, 73, 74, 75],
             [80, 81, 82, 83], [88, 89, 90, 91]],
            [[128, 129, 130, 131], [136, 137, 138, 139],
             [144, 145, 146, 147], [152, 153, 154, 155]],
            [[192, 193, 194, 195], [200, 201, 202, 203],
             [208, 209, 210, 211], [216, 217, 218, 219]],
        ]
        gathered_routes = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ]
        self.assertEqual(
            runner._expected_receive(
                gathered_payloads, gathered_routes, destination_rank=0),
            [[0, 1, 2, 3], [72, 73, 74, 75],
             [144, 145, 146, 147], [216, 217, 218, 219]])

    def test_synchronized_step_reports_local_and_peer_failures(self):
        runner = _load_runner()
        self.assertEqual(
            runner._synchronized_step(
                _Torch, _Dist(), "group", "device", lambda: 7, "ok"),
            7)
        with self.assertRaisesRegex(ValueError, "local"):
            runner._synchronized_step(
                _Torch, _Dist(), "group", "device",
                lambda: (_ for _ in ()).throw(ValueError("local")), "step")
        with self.assertRaisesRegex(RuntimeError, "failed on a peer rank"):
            runner._synchronized_step(
                _Torch, _Dist(peer_failed=True), "group", "device",
                lambda: 7, "step")

    def test_cleanup_exercises_idempotent_destroy(self):
        runner = _load_runner()
        buffer = _Destroyable()
        runner._destroy_twice(buffer)
        self.assertTrue(buffer.destroyed)
        self.assertEqual(buffer.destroy_calls, 2)


if __name__ == "__main__":
    unittest.main()
