import ast
import importlib.util
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER = ROOT / "csrc/backends/ascend/elastic_buffer.hpp"
API_CONTRACT = ROOT / "tests/platform/api_contract.py"
EXTENSION_CONTRACT = ROOT / "tests/platform/test_extension_contract.py"
STUB_TEST = ROOT / "tests/ascend/test_stub.py"


PYBIND11_HEADER = r"""
#pragma once
#include <exception>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

struct _object {};
using PyObject = _object;
inline PyObject* PyExc_NotImplementedError = nullptr;
inline std::string python_error;
inline void PyErr_SetString(PyObject*, const char* message) { python_error = message; }

namespace pybind11 {
class object {};
class error_already_set : public std::exception {};

class module_ {
public:
    std::set<std::string> names;
    std::map<std::string, std::set<std::string>> class_methods;

    template <typename Function>
    void def(const char* name, Function&&) {
        names.emplace(name);
    }
};

template <typename... Args>
struct init {};

template <typename Type>
class class_ {
    module_* module;
    std::string name;

public:
    class_(module_& owner, const char* class_name)
        : module(&owner), name(class_name) {
        module->names.emplace(name);
    }

    template <typename... Args>
    class_& def(init<Args...>) {
        static_assert(std::is_constructible_v<Type, Args...>);
        return *this;
    }

    template <typename Function>
    class_& def(const char* method_name, Function&&) {
        module->class_methods[name].emplace(method_name);
        return *this;
    }
};
}  // namespace pybind11
"""


TORCH_HEADER = r"""
#pragma once
#include <array>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>

namespace c10 {
enum class DeviceType { PrivateUse1 };
}
namespace torch {
using ScalarType = int;
inline constexpr ScalarType kBFloat16 = 1;
inline constexpr ScalarType kLong = 2;
inline constexpr ScalarType kFloat = 3;
inline constexpr ScalarType kInt = 4;
inline constexpr ScalarType kByte = 5;

class TensorOptions {
public:
    TensorOptions dtype(ScalarType) const { return {}; }
};

class Device {
public:
    c10::DeviceType type() const { return c10::DeviceType::PrivateUse1; }
    bool operator==(const Device&) const { return true; }
};

class Tensor {
public:
    bool is_contiguous() const { return true; }
    Device device() const { return {}; }
    std::int64_t dim() const { return 2; }
    ScalarType scalar_type() const { return kBFloat16; }
    std::int64_t size(std::int64_t dimension) const {
        return dimension == 0 ? 1 : 1;
    }
    std::array<std::int64_t, 2> sizes() const { return {1, 1}; }
    std::int64_t numel() const { return 1; }
    TensorOptions options() const { return {}; }
    void* data_ptr() const { return nullptr; }
    template <typename T>
    T* data_ptr() const { return nullptr; }
    Tensor narrow(std::int64_t, std::int64_t, std::int64_t) const { return {}; }
    Tensor clone() const { return {}; }
};

inline Tensor empty(std::initializer_list<std::int64_t>, const TensorOptions&) {
    return {};
}

inline Tensor empty_like(const Tensor&) { return {}; }
namespace detail {
template <typename... Args>
std::string torch_check_message(Args&&... args) {
    std::ostringstream stream;
    (stream << ... << args);
    return stream.str();
}
}  // namespace detail
}  // namespace torch

#define TORCH_CHECK(condition, ...)                                           \
    do {                                                                      \
        if (!(condition))                                                     \
            throw std::runtime_error(                                         \
                ::torch::detail::torch_check_message(__VA_ARGS__));           \
    } while (false)
"""


PROBE = r"""
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "csrc/elastic/api.hpp"

using Buffer = deep_ep::ascend::ElasticBuffer;
using Event = deep_ep::ascend::EventHandle;
using Tensor = torch::Tensor;

using DispatchResult = std::tuple<
    Tensor, std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Tensor>, int, int, std::vector<int>, Tensor, Tensor, Tensor,
    Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Event>>;
using Dispatch = DispatchResult (Buffer::*)(
    const Tensor&, const std::optional<Tensor>&, const Tensor&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<int>&, const std::optional<int>&,
    const std::optional<std::vector<int>>&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<Tensor>&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const int&, const int&, const int&, const int&, const int&,
    const std::optional<Event>&,
    const std::optional<Event>&, const bool&, const bool&, const bool&,
    const bool&, const bool&, const bool&, const bool&) const;

using CombineResult = std::tuple<Tensor, std::optional<Tensor>, std::optional<Event>>;
using Combine = CombineResult (Buffer::*)(
    const Tensor&, const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<Tensor>&, const Tensor&, const Tensor&, const Tensor&,
    const std::optional<Tensor>&, const std::optional<Tensor>&, const int&,
    const int&, const int&, const int&, const std::optional<Event>&,
    const std::optional<Event>&, const bool&, const bool&, const bool&) const;

static_assert(std::is_same_v<decltype(&Buffer::dispatch), Dispatch>);
static_assert(std::is_same_v<decltype(&Buffer::combine), Combine>);
static_assert(std::is_same_v<decltype(&Buffer::get_comm_stream),
                             pybind11::object (Buffer::*)() const>);

template <typename Call>
bool raises_transport_error(const char* operation, const char* detail, Call call) {
    python_error.clear();
    try {
        call();
    } catch (const pybind11::error_already_set&) {
        return python_error == std::string("DeepEP Ascend backend: ") + operation +
                                   " " + detail;
    }
    return false;
}

int main() {
    pybind11::module_ module;
    deep_ep::elastic::register_apis(module);
    if (module.names != std::set<std::string>{
            "EventHandle", "ElasticBuffer", "calculate_elastic_buffer_size"})
        return 17;
    if (module.class_methods["EventHandle"] !=
        std::set<std::string>{"current_stream_wait"})
        return 18;
    if (module.class_methods["ElasticBuffer"] != std::set<std::string>{
            "destroy", "get_comm_stream", "get_physical_domain_size",
            "get_logical_domain_size", "barrier", "dispatch", "combine"})
        return 19;

    Buffer::cpu_comm_t cpu_comm;
    Buffer buffer(0, 1, 0, cpu_comm, 4096, 0,
                  false, true, true, 3, 0, 300, 100, true);
    buffer.destroy();
    buffer.destroy();

    try {
        Buffer invalid(0, 1, 7, cpu_comm, 4096, 0,
                       false, true, true, 3, 0, 300, 100, true);
        return 1;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(
                "communicator_handle must be zero in Phase 2A") == std::string::npos)
            return 2;
    }

    try {
        Buffer invalid(0, 1, 0, cpu_comm, 0, 0,
                       false, true, true, 3, 0, 300, 100, true);
        return 3;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("device_buffer_bytes must be positive") ==
            std::string::npos)
            return 4;
    }

    try {
        Buffer invalid(0, 0, 0, cpu_comm, 4096, 0,
                       false, true, true, 3, 0, 300, 100, true);
        return 5;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("world_size must be positive") == std::string::npos)
            return 6;
    }

    try {
        Buffer invalid(1, 1, 0, cpu_comm, 4096, 0,
                       false, true, true, 3, 0, 300, 100, true);
        return 7;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("rank must be in [0, world_size)") ==
            std::string::npos)
            return 8;
    }

    if (!raises_transport_error(
            "current_stream_wait",
            "is unavailable until the Ascend device transport is implemented",
            [] { Event().current_stream_wait(); }))
        return 9;
    if (!raises_transport_error(
            "get_comm_stream",
            "is unavailable until the Ascend device transport is implemented",
            [&] { buffer.get_comm_stream(); }))
        return 10;
    if (!raises_transport_error(
            "get_physical_domain_size",
            "is unavailable until the Ascend device transport is implemented",
            [&] { buffer.get_physical_domain_size(); }))
        return 11;
    if (!raises_transport_error(
            "get_logical_domain_size",
            "is unavailable until the Ascend device transport is implemented",
            [&] { buffer.get_logical_domain_size(); }))
        return 12;
    try {
        Buffer::cpu_comm_t unsupported_cpu_comm{{1, 2}};
        Buffer invalid(0, 1, 0, unsupported_cpu_comm, 4096, 0,
                       false, true, true, 3, 0, 300, 100, true);
        return 20;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(
                "cpu_communicator must be empty in Phase 2A") == std::string::npos)
            return 21;
    }

    try {
        Buffer invalid(0, 1, 0, cpu_comm, 4096, 4096,
                       false, true, true, 3, 0, 300, 100, true);
        return 22;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("cpu_buffer_bytes must be zero in Phase 2A") ==
            std::string::npos)
            return 23;
    }

    if (!raises_transport_error(
            "barrier", "requires unavailable device transport capabilities: remote_signal, "
            "system_memory_ordering, device_barrier",
            [&] { buffer.barrier(true, false, true); }))
        return 13;
    const auto buffer_bytes = Buffer::calculate_buffer_size(
        7, 128, 7168, 8, false, false, true);
    if (buffer_bytes <= 0 ||
        buffer_bytes % deep_ep::ascend::elastic::kPublicElasticBufferAlignment != 0)
        return 14;
    try {
        Buffer::calculate_buffer_size(0, 128, 7168, 8, false, false, true);
        return 29;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("communicator_handle must be nonzero") ==
            std::string::npos)
            return 30;
    }
    if (!raises_transport_error(
            "calculate_elastic_buffer_size", "does not support FP8", [] {
                Buffer::calculate_buffer_size(7, 128, 7168, 8, true, false, true);
            }))
        return 31;
    if (!raises_transport_error(
            "calculate_elastic_buffer_size", "does not support hybrid mode", [] {
                Buffer::calculate_buffer_size(7, 128, 7168, 8, false, true, true);
            }))
        return 32;

    Tensor tensor;
    std::optional<Tensor> optional_tensor;
    std::optional<int> optional_int;
    std::optional<std::vector<int>> optional_ints;
    std::optional<Event> optional_event;
    if (!raises_transport_error(
            "dispatch", "requires unavailable device transport capabilities: "
            "symmetric_window, device_put, device_put_value, remote_signal, "
            "system_memory_ordering, "
            "device_barrier", [&] {
                buffer.dispatch(
                    tensor, optional_tensor, tensor, optional_tensor, optional_tensor,
                    optional_int, optional_int, optional_ints,
                    optional_tensor, optional_tensor, optional_tensor, optional_tensor,
                    optional_tensor, optional_tensor, optional_tensor,
                    1, 1, 1, 1, 0, optional_event, optional_event,
                    false, false, true, true, false, false, false);
            }))
        return 15;
    if (!raises_transport_error(
            "combine", "requires unavailable device transport capabilities: "
            "symmetric_window, direct_peer_pointer, device_put, remote_atomic_add_release, "
            "remote_signal, system_memory_ordering, device_barrier", [&] {
                buffer.combine(
                    tensor, optional_tensor, optional_tensor, optional_tensor,
                    tensor, tensor, tensor, optional_tensor, optional_tensor,
                    1, 1, 1, 0, optional_event, optional_event, false, false, false);
            }))
        return 16;

    Buffer hybrid_buffer(0, 1, 0, cpu_comm, 4096, 0,
                         true, true, true, 3, 0, 300, 100, true);
    if (!raises_transport_error(
            "barrier", "requires unavailable device transport capabilities: remote_signal, "
            "system_memory_ordering, device_barrier, scale_up_team, scale_out_team",
            [&] { hybrid_buffer.barrier(true, false, true); }))
        return 24;
    if (!raises_transport_error(
            "dispatch", "requires unavailable device transport capabilities: "
            "symmetric_window, device_put, device_put_value, remote_signal, "
            "system_memory_ordering, "
            "device_barrier, scale_up_team, scale_out_team", [&] {
                hybrid_buffer.dispatch(
                    tensor, optional_tensor, tensor, optional_tensor, optional_tensor,
                    optional_int, optional_int, optional_ints,
                    optional_tensor, optional_tensor, optional_tensor, optional_tensor,
                    optional_tensor, optional_tensor, optional_tensor,
                    1, 1, 1, 1, 0, optional_event, optional_event,
                    false, false, true, true, false, false, false);
            }))
        return 25;
    if (!raises_transport_error(
            "combine", "requires unavailable device transport capabilities: "
            "symmetric_window, direct_peer_pointer, device_put, remote_atomic_add_release, "
            "remote_signal, async_completion, system_memory_ordering, device_barrier, "
            "scale_up_team, scale_out_team", [&] {
                hybrid_buffer.combine(
                    tensor, optional_tensor, optional_tensor, optional_tensor,
                    tensor, tensor, tensor, optional_tensor, optional_tensor,
                    1, 1, 1, 0, optional_event, optional_event, false, false, false);
            }))
        return 26;

    try {
        deep_ep::ascend::raise_transport_status(
            deep_ep::ascend::transport::TransportStatus::runtime_failure(
                "export_device_context", 73, "driver rejected the context"), 4);
        return 27;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) !=
            "DeepEP Ascend backend: export_device_context failed on rank 4 "
            "with backend error 73: driver rejected the context")
            return 28;
    }
    return 0;
}
"""

PRODUCTION_PROBE = r"""
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "csrc/elastic/api.hpp"

using Buffer = deep_ep::ascend::ElasticBuffer;
using Event = deep_ep::ascend::EventHandle;
using Tensor = torch::Tensor;

using DispatchResult = std::tuple<
    Tensor, std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Tensor>, int, int, std::vector<int>, Tensor, Tensor, Tensor,
    Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Event>>;

static_assert(std::is_same_v<decltype(&Buffer::get_comm_stream),
                             pybind11::object (Buffer::*)() const>);

extern "C" int deep_ep_ascend_launch_barrier(
    deep_ep::ascend::elastic::BarrierArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch(
    deep_ep::ascend::elastic::DispatchArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    deep_ep::ascend::elastic::DispatchArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine(
    deep_ep::ascend::elastic::CombineArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine_epilogue(
    deep_ep::ascend::elastic::CombineArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }

template <typename Call>
bool raises_transport_error(const char* operation, const char* detail, Call call) {
    python_error.clear();
    try {
        call();
    } catch (const pybind11::error_already_set&) {
        return python_error == std::string("DeepEP Ascend backend: ") + operation +
                                   " " + detail;
    }
    return false;
}

int main() {
    pybind11::module_ module;
    deep_ep::elastic::register_apis(module);
    if (module.names != std::set<std::string>{
            "EventHandle", "ElasticBuffer", "calculate_elastic_buffer_size"})
        return 1;
    if (module.class_methods["ElasticBuffer"] != std::set<std::string>{
            "destroy", "get_comm_stream", "get_physical_domain_size",
            "get_logical_domain_size", "barrier", "dispatch", "combine"})
        return 2;

    const auto buffer_bytes = Buffer::calculate_buffer_size(
        7, 128, 7168, 8, false, false, true);
    if (buffer_bytes <= 0 ||
        buffer_bytes % deep_ep::ascend::elastic::kPublicElasticBufferAlignment != 0)
        return 3;
    if (!raises_transport_error(
            "calculate_elastic_buffer_size", "does not support FP8", [] {
                Buffer::calculate_buffer_size(7, 128, 7168, 8, true, false, true);
            }))
        return 4;
    try {
        Buffer::cpu_comm_t cpu_comm;
        Buffer invalid_timeout(0, 2, 7, cpu_comm, 2 * 1024 * 1024, 0,
                               false, true, true, 3, 0, 300, 0, true);
        return 5;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(
                "num_gpu_timeout_secs must be positive") == std::string::npos)
            return 6;
    }
    return 0;
}
"""


class AscendStubSourceTest(unittest.TestCase):
    def test_header_type_checks_and_host_behavior_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            include = pathlib.Path(directory)
            (include / "pybind11").mkdir()
            (include / "torch").mkdir()
            (include / "pybind11/pybind11.h").write_text(PYBIND11_HEADER)
            (include / "torch/python.h").write_text(TORCH_HEADER)
            probe = include / "probe.cpp"
            probe.write_text(textwrap.dedent(PRODUCTION_PROBE))
            binary = include / "probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Werror=return-type",
                 "-DDEEP_EP_PLATFORM_ASCEND=1",
                 f"-I{include}", f"-I{ROOT}",
                 str(probe),
                 str(ROOT / "csrc/backends/ascend/elastic/runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/transport/cann_transport.cpp"),
                 "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_header_has_no_accelerator_dependencies(self):
        source = HEADER.read_text()
        includes = [line.strip().lower() for line in source.splitlines()
                    if line.lstrip().startswith("#include")]
        self.assertIn('"runtime/cann_runtime.hpp"', " ".join(includes))
        for forbidden in ("cuda", "nccl", "nvshmem", "hccl", "torch_npu",
                          "acl/", "hcomm"):
            self.assertFalse(any(forbidden in include for include in includes), forbidden)

    def test_contract_defines_exact_public_allowlists(self):
        spec = importlib.util.spec_from_file_location("api_contract", API_CONTRACT)
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)

        self.assertEqual(contract.ASCEND_MODULE_NAMES, {
            "get_platform", "topk_idx_t", "EventHandle", "ElasticBuffer",
            "calculate_elastic_buffer_size",
        })
        self.assertEqual(contract.CUDA_MODULE_NAMES - contract.ASCEND_MODULE_NAMES, {
            "is_sm90_compiled", "init_jit", "Config", "Buffer",
            "get_low_latency_rdma_size_hint", "create_cpu_handle",
            "get_elastic_buffer_alignment", "get_local_nccl_unique_id",
            "create_nccl_comm", "destroy_nccl_comm",
            "get_physical_domain_size", "get_logical_domain_size",
        })
        self.assertEqual(contract.ASCEND_ELASTIC_BUFFER_METHODS,
                         contract.COMMON_BUFFER_METHODS)
        self.assertEqual(contract.CUDA_ELASTIC_BUFFER_METHODS,
                         contract.COMMON_BUFFER_METHODS |
                         contract.CUDA_ONLY_BUFFER_METHODS)

        tree = ast.parse(EXTENSION_CONTRACT.read_text())
        called_attributes = {
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        called_names = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("assertSetEqual", called_attributes)
        self.assertIn("public_names", called_names)
        self.assertNotIn("hasattr", called_names)

    def test_python_transport_error_helper_requires_exact_type_and_message(self):
        tree = ast.parse(STUB_TEST.read_text())
        helper = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and
                      node.name == "assert_transport_error")
        calls = {
            node.func.attr: [ast.unparse(argument) for argument in node.args]
            for node in ast.walk(helper)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
            node.func.attr.startswith("assert")
        }
        self.assertNotIn("assertRaisesRegex", calls)
        self.assertEqual(calls["assertRaises"], ["NotImplementedError"])
        self.assertEqual(calls["assertIs"],
                         ["type(exception)", "NotImplementedError"])
        self.assertEqual(calls["assertEqual"], ["str(exception)", "message"])


if __name__ == "__main__":
    unittest.main()
