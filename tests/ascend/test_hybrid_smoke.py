import importlib.util
import json
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tests/ascend/production/run_hybrid_smoke.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("hybrid_smoke", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_runner_contract_is_exact():
    result = subprocess.run(
        ["python3", str(RUNNER), "--contract"], cwd=ROOT,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "allow_multiple_reduction": [False, True],
        "case_names": [
            "local-scale-up-scale-out-diagonal",
            "asymmetric-empty-ranks",
            "repeated-generations",
            "cached-dispatch",
            "reverse-combine",
        ],
        "environment": {
            "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
            "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
        },
        "evidence": "logical-single-host",
        "handle_layout": [
            "DispatchHandleDescriptor",
            "HybridRouteRecord[route_record_count]",
        ],
        "rank_mapping": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "route_schedule_from_rank_0": {
            "0": {"ingress": 0, "stages": []},
            "1": {"ingress": 1, "stages": ["scale-up"]},
            "2": {"ingress": 2, "stages": ["scale-out"]},
            "3": {"ingress": 2, "stages": ["scale-out", "scale-up"]},
        },
        "world_size": 4,
    }


def test_independent_reference_covers_all_rank_zero_route_classes():
    model = _load_runner()
    payloads = (((10,), (11,), (12,), (13,)), (), (), ())
    destinations = ((0, 1, 2, 3), (), (), ())
    receives, records = model.reference_dispatch(payloads, destinations, 7)
    assert receives == (((10,),), ((11,),), ((12,),), ((13,),))
    assert records[0][0] == model.RouteRecord(
        0, 0, 0, 0, 0, None, None, 7)
    assert records[1][0] == model.RouteRecord(
        0, 1, 1, 1, 0, None, None, 7)
    assert records[2][0] == model.RouteRecord(
        0, 2, 2, 2, 0, None, None, 7)
    assert records[3][0] == model.RouteRecord(
        0, 3, 2, 3, 0, 0, 0, 7)


def test_independent_reference_orders_asymmetric_and_empty_ranks_by_world_rank():
    model = _load_runner()
    payloads = (((0,), (1,), (2,)), (), ((20,), (21,)), ())
    destinations = ((3, 0, 3), (), (3, 1), ())
    receives, records = model.reference_dispatch(payloads, destinations, 11)
    assert receives == (((1,),), ((21,),), (), ((0,), (2,), (20,)))
    assert tuple((row.origin_world_rank, row.origin_source_row)
                 for row in records[3]) == ((0, 0), (0, 2), (2, 0))
    assert tuple((row.ingress_slot, row.forwarded_slot)
                 for row in records[3]) == ((0, 0), (1, 1), (None, None))


def test_cached_reference_reuses_geometry_across_generations():
    model = _load_runner()
    destinations = ((3, 3), (2,), (), ())
    first_payloads = (((1,), (2,)), ((3,),), (), ())
    _, first = model.reference_dispatch(first_payloads, destinations, 19)
    second_payloads = (((4,), (5,)), ((6,),), (), ())
    receives, second = model.reference_cached_dispatch(
        second_payloads, destinations, first, 20)
    assert receives[3] == ((4,), (5,))
    assert tuple(row.generation for rows in second for row in rows) == (20, 20, 20)
    assert tuple((row.ingress_slot, row.forwarded_slot)
                 for row in second[3]) == ((0, 0), (1, 1))


def test_reverse_combine_matches_both_reduction_modes():
    model = _load_runner()
    payloads = (((1,), (2,)), ((3,),), (), ())
    destinations = ((0, 3), (2,), (), ())
    _, records = model.reference_dispatch(payloads, destinations, 23)
    contributions = tuple(tuple((float(row.origin_source_row + 1), 10.0)
                                for row in rows) for rows in records)
    assert model.reference_combine(
        records, contributions, (2, 1, 0, 0), False) == (
            (1.0, 2.0), (1.0,), (), ())
    assert model.reference_combine(
        records, contributions, (2, 1, 0, 0), True) == (
            (11.0, 12.0), (11.0,), (), ())


def test_reverse_schedule_uses_recorded_slots_not_contributor_order():
    model = _load_runner()
    later_forward = model.RouteRecord(0, 0, 2, 3, 0, 0, 1, 29)
    earlier_forward = model.RouteRecord(0, 1, 2, 3, 0, 1, 0, 29)
    records = ((), (), (), (later_forward, earlier_forward))
    contributions = ((), (), (), ("first", "second"))
    reverse_forward, reverse_return = model.reference_reverse_schedule(
        records, contributions)
    assert reverse_forward == ((2, 0, "second"), (2, 1, "first"))
    assert reverse_return == ((0, 0, "first"), (0, 1, "second"))


def test_production_exposes_device_only_two_stage_hybrid_contract():
    buffer_source = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
    runtime = (ROOT / "csrc/backends/ascend/elastic/runtime.cpp").read_text()
    dispatch = (ROOT / "csrc/backends/ascend/elastic/dispatch.asc").read_text()
    combine = (ROOT / "csrc/backends/ascend/elastic/combine.asc").read_text()
    kernels = (ROOT / "csrc/backends/ascend/elastic/kernels.hpp").read_text()
    compact_runtime = " ".join(runtime.split())

    assert "hybrid mode is unsupported" not in buffer_source
    assert "does not support hybrid mode" not in buffer_source
    assert "route_records" in kernels
    assert "HybridRouteRecord" in kernels
    assert "CoreMode::kHybrid" not in runtime[runtime.index(
        "bool has_deferred_mode"):runtime.index("bool same_topology")]
    for field in (
            "hybrid_route_record_offset",
            "hybrid_dispatch_ingress_control_offset",
            "hybrid_dispatch_forward_control_offset",
            "hybrid_combine_reverse_forward_control_offset",
            "hybrid_combine_return_control_offset"):
        assert f"lhs.{field} == rhs.{field}" in compact_runtime
    assert dispatch.count("service::execute") >= 2
    assert "TransportTeam::kScaleOut" in dispatch
    assert "TransportTeam::kScaleUp" in dispatch
    assert "hybrid_dispatch_ingress" in dispatch
    assert "hybrid_dispatch_forward" in dispatch
    for source in (dispatch, combine):
        assert "has_mode(" not in source
        assert "mode_bit(" not in source
    assert not re.search(r"route_records\s*\[[^]]+\]\s*=", dispatch)
    assert combine.count("service::execute") >= 2
    assert "TransportTeam::kScaleUp" in combine
    assert "TransportTeam::kScaleOut" in combine
    assert "hybrid_combine_reverse_forward" in combine
    assert "hybrid_combine_return" in combine
    producer = combine[
        combine.index("__simt_vf__ inline void combine_producer_vf"):
        combine.index("__simt_vf__ inline void hybrid_combine_return_vf")]
    staging = producer[producer.index("// Record counts are derived") :]
    assert "route_record.forwarded_slot" in staging
    dispatch_forward = dispatch[
        dispatch.index("__simt_vf__ inline void hybrid_dispatch_forward_vf"):
        dispatch.index(
            "__simt_vf__ inline void hybrid_dispatch_prepare_epilogue_vf")]
    assert dispatch_forward.index("transport.system_fence()") < \
        dispatch_forward.index("transport.put(")
    reverse_return = combine[
        combine.index("__simt_vf__ inline void hybrid_combine_return_vf"):
        combine.index(
            "__simt_vf__ inline void hybrid_combine_prepare_epilogue_vf")]
    assert "route_metadata.ingress_slot" in reverse_return
    assert reverse_return.index("transport.system_fence()") < \
        reverse_return.index("transport.put(")
