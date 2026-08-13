#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_ep::ascend {

[[noreturn]] inline void raise_phase_one_error(const char* operation) {
    const std::string message =
        std::string("DeepEP Ascend backend: ") + operation +
        " is not implemented in phase 1";
    PyErr_SetString(PyExc_NotImplementedError, message.c_str());
    throw pybind11::error_already_set();
}

struct EventHandle {
    void current_stream_wait() const {
        raise_phase_one_error("current_stream_wait");
    }
};

class ElasticBuffer {
    int rank_idx_;
    int num_ranks_;
    int64_t num_buffer_bytes_;
    bool destroyed_ = false;

public:
    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& comm_handle, const int64_t& num_buffer_bytes,
                  const bool&, const bool&, const bool&, const bool&,
                  const int&, const int&, const int&, const int&, const bool&)
        : rank_idx_(rank_idx), num_ranks_(num_ranks),
          num_buffer_bytes_(num_buffer_bytes) {
        TORCH_CHECK(num_ranks > 0, "num_ranks must be positive");
        TORCH_CHECK(rank_idx >= 0 && rank_idx < num_ranks,
                    "rank_idx must be in [0, num_ranks)");
        TORCH_CHECK(comm_handle == 0,
                    "DeepEP Ascend backend: comm_handle must be zero in phase 1");
        TORCH_CHECK(num_buffer_bytes > 0,
                    "DeepEP Ascend backend: num_buffer_bytes must be positive");
    }

    void destroy() { destroyed_ = true; }

    pybind11::object get_comm_stream() const {
        raise_phase_one_error("get_comm_stream");
    }

    std::tuple<int, int> get_physical_domain_size() const {
        raise_phase_one_error("get_physical_domain_size");
    }

    std::tuple<int, int> get_logical_domain_size() const {
        raise_phase_one_error("get_logical_domain_size");
    }

    void barrier(const bool&, const bool&) const {
        raise_phase_one_error("barrier");
    }

    static int64_t calculate_buffer_size(
        const int64_t&, const int&, const int&, int, const bool&,
        const bool&, const bool&) {
        raise_phase_one_error("calculate_elastic_buffer_size");
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor&, const std::optional<torch::Tensor>&,
             const torch::Tensor&, const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&, const std::optional<int>&,
             const std::optional<std::vector<int>>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const int&, const int&, const int&, const int&, const int&,
             const std::optional<EventHandle>&,
             const std::optional<EventHandle>&,
             const bool&, const bool&, const bool&, const bool&, const bool&,
             const bool&) const {
        raise_phase_one_error("dispatch");
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    combine(const torch::Tensor&, const std::optional<torch::Tensor>&,
            const std::optional<torch::Tensor>&,
            const std::optional<torch::Tensor>&, const torch::Tensor&,
            const torch::Tensor&, const torch::Tensor&,
            const std::optional<torch::Tensor>&,
            const std::optional<torch::Tensor>&,
            const int&, const int&, const int&, const int&,
            const std::optional<EventHandle>&,
            const std::optional<EventHandle>&,
            const bool&, const bool&, const bool&) const {
        raise_phase_one_error("combine");
    }
};

}  // namespace deep_ep::ascend
