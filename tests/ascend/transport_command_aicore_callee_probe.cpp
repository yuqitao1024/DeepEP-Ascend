#define __aicore__ __attribute__((noinline))
#define DEEP_EP_ASCEND_AICORE_URMA_SERVICE 1

#include "csrc/backends/ascend/transport/transport_commands.hpp"

namespace command = deep_ep::ascend::transport::command;

static_assert(__builtin_has_attribute(
    command::barrier_team_enabled, noinline));
static_assert(__builtin_has_attribute(
    command::barrier_peer_in_team, noinline));

int main() {
    return 0;
}
