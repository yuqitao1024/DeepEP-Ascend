#include <cstdlib>

bool binding_probe_gil_released = false;
bool binding_probe_expect_release = false;
int binding_probe_destroy_calls = 0;

#include "csrc/elastic/binding.hpp"

namespace {

struct FakeElasticBuffer {
    using cpu_comm_t = int;

    void destroy() {
        if (binding_probe_gil_released != binding_probe_expect_release)
            std::abort();
        ++binding_probe_destroy_calls;
    }

    void get_comm_stream() {}
    void get_physical_domain_size() {}
    void get_logical_domain_size() {}
    void barrier() {}
    void dispatch() {}
    void combine() {}
    static void calculate_buffer_size() {}
};

}  // namespace

int main() {
    pybind11::module_ cuda_module;
    binding_probe_expect_release = false;
    deep_ep::elastic::binding::register_common_apis<
        FakeElasticBuffer, false>(cuda_module);

    pybind11::module_ ascend_module;
    binding_probe_expect_release = true;
    deep_ep::elastic::binding::register_common_apis<
        FakeElasticBuffer, true>(ascend_module);

    return binding_probe_destroy_calls == 2 ? 0 : 1;
}
