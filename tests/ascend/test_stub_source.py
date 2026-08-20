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
MAPPED_MEMORY_PROBE = ROOT / "tests/ascend/mapped_memory_probe.cpp"


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
class gil_scoped_release {};

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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace c10 {
enum class DeviceType { PrivateUse1 };
using DeviceIndex = int;
using StreamId = std::int64_t;
class Stream {
public:
    static Stream unpack3(StreamId, DeviceIndex, DeviceType) { return {}; }
};
}
namespace torch {
using ScalarType = int;
inline constexpr ScalarType kBFloat16 = 1;
inline constexpr ScalarType kLong = 2;
inline constexpr ScalarType kFloat = 3;
inline constexpr ScalarType kInt = 4;
inline constexpr ScalarType kByte = 5;
inline constexpr ScalarType kFloat8_e4m3fn = 6;

class TensorOptions {
    ScalarType type_ = kBFloat16;
    int device_ = 0;
public:
    TensorOptions dtype(ScalarType type) const { auto result = *this; result.type_ = type; return result; }
    TensorOptions device(int device) const { auto result = *this; result.device_ = device; return result; }
    ScalarType dtype() const { return type_; }
    int device_index() const { return device_; }
};

class Device {
    int index_ = 0;
public:
    Device() = default;
    explicit Device(int index) : index_(index) {}
    c10::DeviceType type() const { return c10::DeviceType::PrivateUse1; }
    int index() const { return index_; }
    bool operator==(const Device& other) const { return index_ == other.index_; }
};

class Tensor {
    std::vector<std::int64_t> sizes_{1, 1};
    ScalarType type_ = kBFloat16;
    Device device_{};
    std::shared_ptr<std::vector<std::uint8_t>> storage_ =
        std::make_shared<std::vector<std::uint8_t>>(2);
public:
    Tensor() = default;
    Tensor(std::initializer_list<std::int64_t> sizes, TensorOptions options)
        : sizes_(sizes), type_(options.dtype()), device_(options.device_index()),
          storage_(std::make_shared<std::vector<std::uint8_t>>(numel() * bytes())) {}
    bool is_contiguous() const { return true; }
    Device device() const { return device_; }
    std::int64_t dim() const { return sizes_.size(); }
    ScalarType scalar_type() const { return type_; }
    std::int64_t size(std::int64_t dimension) const { return sizes_.at(dimension); }
    std::int64_t stride(std::int64_t dimension) const {
        std::int64_t result = 1;
        for (std::int64_t index = dimension + 1;
             index < static_cast<std::int64_t>(sizes_.size()); ++index)
            result *= sizes_[index];
        return result;
    }
    const std::vector<std::int64_t>& sizes() const { return sizes_; }
    std::int64_t numel() const { std::int64_t result = 1; for (auto value : sizes_) result *= value; return result; }
    TensorOptions options() const {
        return TensorOptions().dtype(type_).device(device_.index());
    }
    void* data_ptr() const { return numel() == 0 ? nullptr : storage_->data(); }
    template <typename T>
    T* data_ptr() const { return static_cast<T*>(data_ptr()); }
    Tensor narrow(std::int64_t dimension, std::int64_t, std::int64_t length) const { auto result = *this; result.sizes_[dimension] = length; return result; }
    Tensor clone() const {
        auto result = *this;
        result.storage_ =
            std::make_shared<std::vector<std::uint8_t>>(*storage_);
        return result;
    }
    Tensor& copy_(const Tensor&) { return *this; }
private:
    std::size_t bytes() const { return type_ == kBFloat16 ? 2 : type_ == kLong ? 8 : type_ == kFloat || type_ == kInt ? 4 : 1; }
};

inline Tensor empty(std::initializer_list<std::int64_t> sizes, const TensorOptions& options) {
    return Tensor(sizes, options);
}

inline Tensor empty_strided(std::initializer_list<std::int64_t> sizes,
                            std::initializer_list<std::int64_t>,
                            const TensorOptions& options) {
    return Tensor(sizes, options);
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
                             c10::Stream (Buffer::*)() const>);

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
            "get_logical_domain_size", "barrier", "dispatch", "combine",
            "get_dispatch_handle_generation"})
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
    const auto fp8_buffer_bytes = Buffer::calculate_buffer_size(
        7, 128, 7168, 8, true, false, true);
    if (fp8_buffer_bytes <= 0 ||
        fp8_buffer_bytes % deep_ep::ascend::elastic::kPublicElasticBufferAlignment != 0 ||
        fp8_buffer_bytes == buffer_bytes)
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
            "symmetric_window, device_put, device_put_value, remote_signal, "
            "system_memory_ordering, device_barrier, scale_up_team", [&] {
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
            "symmetric_window, device_put, device_put_value, remote_signal, "
            "system_memory_ordering, device_barrier, scale_up_team", [&] {
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
                             c10::Stream (Buffer::*)() const>);

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

int fake_get_size(void*, std::int64_t, std::uint32_t* size) {
    *size = 3;
    return 0;
}

int main() {
    pybind11::module_ module;
    deep_ep::elastic::register_apis(module);
    if (module.names != std::set<std::string>{
            "EventHandle", "ElasticBuffer", "calculate_elastic_buffer_size"})
        return 1;
    if (module.class_methods["EventHandle"] !=
        std::set<std::string>{"__copy__", "current_stream_wait"})
        return 2;
    if (module.class_methods["ElasticBuffer"] != std::set<std::string>{
            "destroy", "get_comm_stream", "get_physical_domain_size",
            "get_logical_domain_size", "barrier", "dispatch", "combine",
            "is_destroyed", "get_dispatch_handle_generation"})
        return 3;

    deep_ep::ascend::transport::CannHostApi host_api{};
    host_api.get_size = fake_get_size;
    const auto buffer_bytes = Buffer::calculate_buffer_size_for_testing(
        7, 128, 7168, 8, false, false, true, host_api);
    const auto fp8_buffer_bytes = Buffer::calculate_buffer_size_for_testing(
        7, 128, 7168, 8, true, false, true, host_api);
    deep_ep::ascend::elastic::SymmetricWindowInput input{};
    input.world_size = 3;
    input.num_max_tokens_per_rank = 128;
    input.hidden = 7168;
    input.num_topk = 8;
    input.element_bytes = 2;
    input.expanded = true;
    input.allow_multiple_reduction = true;
    deep_ep::ascend::elastic::SymmetricWindowLayout layout{};
    if (!deep_ep::ascend::elastic::build_symmetric_window_layout(
            input, &layout).ok() ||
        buffer_bytes != static_cast<std::int64_t>(layout.total_bytes))
        return 3;
    input.element_bytes = 1;
    input.scale_factor_bytes = ((input.hidden + 31) / 32) * 4;
    if (!deep_ep::ascend::elastic::build_symmetric_window_layout(
            input, &layout).ok() ||
        fp8_buffer_bytes != static_cast<std::int64_t>(layout.total_bytes) ||
        fp8_buffer_bytes == buffer_bytes)
        return 4;
    try {
        Buffer::calculate_buffer_size(
            7, 128, 7168, 8, false, false, true);
        return 7;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(
                "CANN public host headers are unavailable") == std::string::npos)
            return 8;
    }
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
    def test_fp8_dispatch_forwards_strides_and_zeros_scale_padding(self):
        source = (ROOT / "csrc/backends/ascend/elastic/dispatch.asc").read_text()
        for marker in (
                "arguments.scale_factor_token_stride",
                "arguments.scale_factor_pack_stride",
                "arguments.recv_scale_factor_token_stride",
                "arguments.recv_scale_factor_pack_stride"):
            self.assertIn(marker, source)
        self.assertGreaterEqual(source.count("scale_factor_byte_offset("), 4)
        zero_padding = source[
            source.index("if (expanded && zero_padding)"):
            source.index("std::uint64_t compact_slot")]
        self.assertIn("recv_scale_factors", zero_padding)

    def test_mapped_memory_owner_lifecycle_probe(self):
        """Catches a missing lifetime edge or any out-of-order mapped teardown."""
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "mapped_memory_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(MAPPED_MEMORY_PROBE),
                 str(ROOT / "csrc/backends/ascend/runtime/mapped_memory.cpp"),
                 "-o", str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

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
                 "-DDEEP_EP_ASCEND_TESTING=1",
                 "-DDEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR=1",
                 f"-I{include}", f"-I{ROOT}",
                 "-include", str(include / "torch/python.h"),
                 str(probe),
                 str(ROOT / "csrc/backends/ascend/elastic/runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/elastic/async_state.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"),
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

    def test_elastic_buffer_uses_one_operation_coordinator(self):
        source = HEADER.read_text()
        self.assertIn('"elastic/operation_coordinator.hpp"', source)
        self.assertIn("std::shared_ptr<elastic::AsyncBufferState> async_state_", source)
        self.assertIn("async_state_->coordinator().reserve(kind)", source)
        self.assertNotIn("elastic::BufferOperationCoordinator coordinator_", source)
        for legacy_state in (
                "barrier_sequence_", "dispatch_sequence_", "combine_sequence_",
                "destroyed_"):
            self.assertNotIn(legacy_state, source)
        for operation_kind in (
                "kTopologyQuery", "kBarrier", "kDispatch", "kCombine"):
            self.assertIn(
                f"elastic::BufferOperationKind::{operation_kind}", source)
        self.assertIn("async_state_->coordinator().reserve(kind)", source)
        self.assertIn("async_state_->destroy()", source)

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
                         contract.COMMON_BUFFER_METHODS |
                         contract.ASCEND_ONLY_BUFFER_METHODS)
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
