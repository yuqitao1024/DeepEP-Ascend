import pathlib
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER = ROOT / "csrc/backends/ascend/elastic_buffer.hpp"


PYBIND11_HEADER = r"""
#pragma once
#include <exception>
#include <string>

struct _object {};
using PyObject = _object;
inline PyObject* PyExc_NotImplementedError = nullptr;
inline std::string python_error;
inline void PyErr_SetString(PyObject*, const char* message) { python_error = message; }

namespace pybind11 {
class object {};
class error_already_set : public std::exception {};
}  // namespace pybind11
"""


TORCH_HEADER = r"""
#pragma once
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace torch {
class Tensor {};
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
#include <vector>

#include "csrc/backends/ascend/elastic_buffer.hpp"

using Buffer = deep_ep::ascend::ElasticBuffer;
using Event = deep_ep::ascend::EventHandle;
using Tensor = torch::Tensor;

using DispatchResult = std::tuple<
    Tensor, std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Tensor>, std::vector<int>, Tensor, Tensor, Tensor, Tensor,
    std::optional<Tensor>, std::optional<Tensor>, std::optional<Event>>;
using Dispatch = DispatchResult (Buffer::*)(
    const Tensor&, const std::optional<Tensor>&, const Tensor&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<int>&, const std::optional<std::vector<int>>&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<Tensor>&, const int&, const int&, const int&,
    const int&, const int&, const std::optional<Event>&,
    const std::optional<Event>&, const bool&, const bool&, const bool&,
    const bool&, const bool&, const bool&) const;

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
bool raises_phase_error(const char* operation, Call call) {
    python_error.clear();
    try {
        call();
    } catch (const pybind11::error_already_set&) {
        return python_error == std::string("DeepEP Ascend backend: ") + operation +
                                   " is not implemented in phase 1";
    }
    return false;
}

int main() {
    Buffer buffer(0, 1, 0, 4096, false, true, true, true, 3, 0, 300, 100, true);
    buffer.destroy();
    buffer.destroy();

    try {
        Buffer invalid(0, 1, 7, 4096, false, true, true, true, 3, 0, 300, 100, true);
        return 1;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("comm_handle must be zero") == std::string::npos)
            return 2;
    }

    try {
        Buffer invalid(0, 1, 0, 0, false, true, true, true, 3, 0, 300, 100, true);
        return 3;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("num_buffer_bytes must be positive") == std::string::npos)
            return 4;
    }

    try {
        Buffer invalid(0, 0, 0, 4096, false, true, true, true, 3, 0, 300, 100, true);
        return 5;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("num_ranks must be positive") == std::string::npos)
            return 6;
    }

    try {
        Buffer invalid(1, 1, 0, 4096, false, true, true, true, 3, 0, 300, 100, true);
        return 7;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("rank_idx must be in [0, num_ranks)") ==
            std::string::npos)
            return 8;
    }

    if (!raises_phase_error("current_stream_wait", [] { Event().current_stream_wait(); }))
        return 9;
    if (!raises_phase_error("get_comm_stream", [&] { buffer.get_comm_stream(); }))
        return 10;
    if (!raises_phase_error("get_physical_domain_size",
                            [&] { buffer.get_physical_domain_size(); }))
        return 11;
    if (!raises_phase_error("get_logical_domain_size",
                            [&] { buffer.get_logical_domain_size(); }))
        return 12;
    if (!raises_phase_error("barrier", [&] { buffer.barrier(true, false); }))
        return 13;
    if (!raises_phase_error("calculate_elastic_buffer_size", [] {
            Buffer::calculate_buffer_size(0, 128, 7168, 8, false, true, true);
        }))
        return 14;

    Tensor tensor;
    std::optional<Tensor> optional_tensor;
    std::optional<int> optional_int;
    std::optional<std::vector<int>> optional_ints;
    std::optional<Event> optional_event;
    if (!raises_phase_error("dispatch", [&] {
            buffer.dispatch(
                tensor, optional_tensor, tensor, optional_tensor, optional_tensor,
                optional_int, optional_ints, optional_tensor, optional_tensor,
                optional_tensor, optional_tensor, optional_tensor,
                1, 1, 1, 1, 0, optional_event, optional_event,
                false, false, true, true, false, false);
        }))
        return 15;
    if (!raises_phase_error("combine", [&] {
            buffer.combine(
                tensor, optional_tensor, optional_tensor, optional_tensor,
                tensor, tensor, tensor, optional_tensor, optional_tensor,
                1, 1, 1, 0, optional_event, optional_event, false, false, false);
        }))
        return 16;
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
            probe.write_text(textwrap.dedent(PROBE))
            binary = include / "probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", f"-I{include}", f"-I{ROOT}",
                 str(probe), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_header_has_no_accelerator_dependencies(self):
        source = HEADER.read_text()
        includes = [line.strip().lower() for line in source.splitlines()
                    if line.lstrip().startswith("#include")]
        for forbidden in ("cuda", "nccl", "nvshmem", "cann", "hccl", "torch_npu"):
            self.assertFalse(any(forbidden in include for include in includes), forbidden)


if __name__ == "__main__":
    unittest.main()
