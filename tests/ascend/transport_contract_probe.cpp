#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "csrc/backends/ascend/transport/device_transport.hpp"
#include "csrc/backends/ascend/transport/execution_domain_helpers.hpp"
#include "csrc/backends/ascend/transport/stage_profile.hpp"
#include "csrc/backends/ascend/transport/stub_transport.hpp"

using namespace deep_ep::ascend::transport;

TransportConfig valid_config() {
    TransportConfig config;
    config.rank = 0;
    config.world_size = 1;
    config.communicator_handle = 0;
    config.cpu_communicator_empty = true;
    config.device_buffer_bytes = 4096;
    config.cpu_buffer_bytes = 0;
    config.requested_channels = 3;
    return config;
}

static_assert(std::is_trivially_copyable_v<TransportTopology>);
static_assert(std::is_trivially_copyable_v<DeviceTransportContext>);
static_assert(std::is_trivially_copyable_v<DeviceRequest>);
static_assert(alignof(DeviceRequest) == 16);
static_assert(sizeof(DeviceRequest) == 32);
static_assert(std::is_trivially_copyable_v<RemoteAction>);
static_assert(std::is_trivially_copyable_v<TransportStageProfile>);
static_assert(alignof(TransportStageProfile) == 64);
static_assert(kTransportProfileStageCount == 16);
static_assert(kTransportProfileMaxBlocks == 72);
static_assert(sizeof(TransportStageBlockCycles) == 64);
static_assert(alignof(TransportStageBlockCycles) == 64);
static_assert(offsetof(TransportStageProfile, command_bytes) == 64);
static_assert(kTransportStageProfileHeaderCacheLineCount == 2);
static_assert(kDefaultOptions == 0);
static_assert((kAggregateRequests & kDefaultOptions) == 0);
static_assert(kNoCapabilities == 0);
static_assert(capability_bit(TransportCapability::kDevicePut) != 0);
static_assert(capability_bit(TransportCapability::kDeviceGet) !=
              capability_bit(TransportCapability::kDevicePut));

using DevicePut = void (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    DeviceAddress, DeviceAddress, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions, const RemoteAction&);
using DeviceGet = void (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    DeviceAddress, DeviceAddress, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions);
using ReadSignal = SignalValue (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    std::uint32_t);
using WaitSignal = void (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    std::uint32_t, SignalValue, std::uint64_t);

static_assert(std::is_same_v<decltype(&device::put), DevicePut>);
static_assert(std::is_same_v<decltype(&device::get), DeviceGet>);
static_assert(std::is_same_v<decltype(&device::read_signal), ReadSignal>);
static_assert(std::is_same_v<decltype(&device::wait_signal), WaitSignal>);

int main() {
    const auto missing = capability_bit(TransportCapability::kDevicePut) |
                         capability_bit(TransportCapability::kRemoteSignal);
    if (capability_names(missing) != "device_put, remote_signal")
        return 1;

    const auto status = TransportStatus::unsupported(
        "dispatch", "missing device transport capabilities: " +
                    capability_names(missing));
    if (status.ok() ||
        status.code != TransportStatusCode::kUnsupportedCapability ||
        status.operation != "dispatch" || status.backend_code != 0)
        return 2;

    DeviceTransportContext context = make_device_transport_context();
    if (context.abi_version != kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(DeviceTransportContext))
        return 3;

    TransportStageProfile profile{};
    if (profile.abi_version != kTransportStageProfileAbiVersion ||
        profile.struct_size != sizeof(TransportStageProfile) ||
        profile.operation != TransportProfileOperation::kNone)
        return 5;

    constexpr auto full_mask = std::uint64_t{1};
    constexpr auto dispatch_pipeline_mask =
        ((std::uint64_t{1} << 14U) - 1U) & ~full_mask;
    constexpr auto combine_pipeline_mask =
        ((std::uint64_t{1} << 12U) - 1U) & ~full_mask;
    constexpr auto observed_dispatch_mask =
        (std::uint64_t{1} << 1U) | (std::uint64_t{1} << 2U) |
        (std::uint64_t{1} << 3U) | (std::uint64_t{1} << 5U) |
        (std::uint64_t{1} << 6U) | (std::uint64_t{1} << 8U) |
        (std::uint64_t{1} << 9U) | (std::uint64_t{1} << 10U) |
        (std::uint64_t{1} << 12U) | (std::uint64_t{1} << 13U);
    constexpr auto observed_combine_mask =
        (std::uint64_t{1} << 1U) | (std::uint64_t{1} << 2U) |
        (std::uint64_t{1} << 3U) | (std::uint64_t{1} << 4U) |
        (std::uint64_t{1} << 5U) | (std::uint64_t{1} << 6U) |
        (std::uint64_t{1} << 8U) | (std::uint64_t{1} << 11U);
    if (stage_profile_mask_status(
            TransportProfileOperation::kDispatch, std::uint64_t{1} << 63U) !=
            TransportStageProfileMaskStatus::kInvalidMask ||
        stage_profile_mask_status(
            TransportProfileOperation::kDispatch,
            dispatch_pipeline_mask & ~(std::uint64_t{1} << 13U)) !=
            TransportStageProfileMaskStatus::kPartialMask ||
        stage_profile_mask_status(
            TransportProfileOperation::kCombine,
            combine_pipeline_mask & ~(std::uint64_t{1} << 11U)) !=
            TransportStageProfileMaskStatus::kPartialMask ||
        stage_profile_mask_status(
            TransportProfileOperation::kDispatch, full_mask) !=
            TransportStageProfileMaskStatus::kValid ||
        stage_profile_mask_status(
            TransportProfileOperation::kCombine, combine_pipeline_mask) !=
            TransportStageProfileMaskStatus::kValid)
        return 6;
    if (stage_profile_mask_status(
            TransportProfileOperation::kDispatch, observed_dispatch_mask) !=
            TransportStageProfileMaskStatus::kPartialMask ||
        stage_profile_mask_status(
            TransportProfileOperation::kCombine, observed_combine_mask) !=
            TransportStageProfileMaskStatus::kPartialMask ||
        stage_profile_mask_status(
            TransportProfileOperation::kDispatch,
            std::uint64_t{1} << 13U) !=
            TransportStageProfileMaskStatus::kPartialMask ||
        stage_profile_mask_status(
            TransportProfileOperation::kDispatch,
            full_mask | (std::uint64_t{1} << 1U) |
                (std::uint64_t{1} << 13U)) !=
            TransportStageProfileMaskStatus::kPartialMask)
        return 9;

    if (transport_stage_profile_service_cycles_valid(0, 0, 0) ||
        transport_stage_profile_service_cycles_valid(11, 0, 0) ||
        transport_stage_profile_service_cycles_valid(11, 10, 0) ||
        transport_stage_profile_service_cycles_valid(10, 20, 11) ||
        !transport_stage_profile_service_cycles_valid(10, 20, 10))
        return 7;

    const auto first_queue = command::aicore_merge_queue_depth_snapshots(
        TransportQueueDepthSnapshot{}, TransportQueueDepthSnapshot{3, 2});
    const auto aggregated_queue = command::aicore_merge_queue_depth_snapshots(
        first_queue, TransportQueueDepthSnapshot{1, 4});
    if (aggregated_queue.sq_depth != 3 || aggregated_queue.cq_depth != 4)
        return 32;

    TransportStageProfile valid_metrics{};
    valid_metrics.command_count = 6;
    valid_metrics.put_command_count = 1;
    valid_metrics.command_bytes = 24;
    valid_metrics.sq_high_watermark = 7;
    valid_metrics.cq_high_watermark = 7;
    if (transport_stage_profile_command_metrics_status(
            valid_metrics, true) !=
            TransportStageProfileCommandMetricsStatus::kValid)
        return 33;

    const auto expect_metric_status = [](
        TransportStageProfile candidate,
        TransportStageProfileCommandMetricsStatus expected,
        const char* expected_reason) {
        const auto observed = transport_stage_profile_command_metrics_status(
            candidate, true);
        const auto* reason = transport_stage_profile_command_metrics_reason(
            observed);
        return observed == expected && reason != nullptr &&
            std::string(reason) == expected_reason;
    };
    auto invalid_metrics = valid_metrics;
    invalid_metrics.put_command_count = 7;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kPutCommandCountExceedsCommandCount,
            "put_command_count_exceeds_command_count"))
        return 34;
    invalid_metrics = valid_metrics;
    invalid_metrics.sq_depth = 8;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kSqDepthExceedsHighWatermark,
            "sq_depth_exceeds_high_watermark"))
        return 35;
    invalid_metrics = valid_metrics;
    invalid_metrics.cq_depth = 8;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kCqDepthExceedsHighWatermark,
            "cq_depth_exceeds_high_watermark"))
        return 36;
    invalid_metrics = valid_metrics;
    invalid_metrics.sq_depth = 1;
    invalid_metrics.cq_depth = 0;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::kQueueDepthMismatch,
            "queue_depth_mismatch"))
        return 37;
    invalid_metrics = valid_metrics;
    invalid_metrics.cq_high_watermark = 0;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kQueueHighWatermarkMismatch,
            "queue_high_watermark_mismatch"))
        return 38;
    invalid_metrics = TransportStageProfile{};
    invalid_metrics.sq_high_watermark = 1;
    invalid_metrics.cq_high_watermark = 1;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kQueueActivityWithoutCommands,
            "queue_activity_without_commands"))
        return 39;
    invalid_metrics = valid_metrics;
    invalid_metrics.sq_depth = 1;
    invalid_metrics.cq_depth = 1;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kCompletedServiceHasOutstandingRequests,
            "completed_service_has_outstanding_requests"))
        return 40;
    invalid_metrics = valid_metrics;
    invalid_metrics.sq_high_watermark = 0;
    invalid_metrics.cq_high_watermark = 0;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kPayloadWithoutQueueActivity,
            "payload_without_queue_activity"))
        return 41;
    invalid_metrics = valid_metrics;
    invalid_metrics.put_command_count = 1;
    invalid_metrics.command_bytes = 0;
    if (!expect_metric_status(
            invalid_metrics,
            TransportStageProfileCommandMetricsStatus::
                kPutCommandsWithoutPayload,
            "put_commands_without_payload"))
        return 42;
    invalid_metrics = valid_metrics;
    invalid_metrics.sq_depth = 1;
    invalid_metrics.cq_depth = 1;
    if (transport_stage_profile_command_metrics_status(
            invalid_metrics, false) !=
            TransportStageProfileCommandMetricsStatus::kValid)
        return 43;

    std::array<std::uint64_t, kTransportProfileStageCount> stage_spans{};
    stage_spans[0] = 140;
    const auto full_phases = derive_stage_profile_phase_cycles(
        TransportProfileOperation::kDispatch, full_mask, stage_spans.data(),
        100, 200, 20);
    if (full_phases.producer != 140 || full_phases.publication != 0 ||
        full_phases.service_submit != 0 || full_phases.cq_wait != 0 ||
        full_phases.consumer_wait != 0 ||
        full_phases.consumer_compute != 0 || full_phases.epilogue != 0)
        return 8;

    const auto no_action = RemoteAction::none();
    const auto signal_add = RemoteAction::signal_add(128, 7);
    const auto signal_increment = RemoteAction::signal_increment(3);
    if (no_action.kind != RemoteActionKind::kNone ||
        signal_add.kind != RemoteActionKind::kSignalAdd ||
        signal_add.symmetric_offset != 128 || signal_add.value != 7 ||
        signal_increment.kind != RemoteActionKind::kSignalIncrement ||
        signal_increment.signal_index != 3 || signal_increment.value != 1)
        return 4;

    auto created = make_stub_transport(valid_config());
    if (!created.status.ok() || !created.transport)
        return 10;
    if (created.transport->capabilities() != kNoCapabilities)
        return 11;

    const auto required = capability_bit(TransportCapability::kDevicePut) |
                          capability_bit(TransportCapability::kRemoteSignal);
    const auto requirement = created.transport->require_capabilities(
        required, "dispatch");
    if (requirement.code != TransportStatusCode::kUnsupportedCapability ||
        requirement.message.find("device_put, remote_signal") == std::string::npos)
        return 12;

    TransportTopology topology;
    if (created.transport->query_topology(&topology).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 13;
    if (created.transport->register_symmetric_window(nullptr, 4096).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 14;
    std::uintptr_t peer_pointer = 0;
    if (created.transport->get_peer_base_pointer(
            TransportTeam::kWorld, 0, &peer_pointer).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 15;
    if (created.transport->acquire_channels(
            3, CooperationScope::kWorkgroup).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 16;
    DeviceTransportContext exported = make_device_transport_context();
    if (created.transport->export_device_context(&exported).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 17;
    if (created.transport->reset_stage_profile().code !=
            TransportStatusCode::kUnsupportedCapability ||
        created.transport->read_stage_profile(&profile).code !=
            TransportStatusCode::kUnsupportedCapability)
        return 31;
    if (created.transport->host_barrier().code !=
        TransportStatusCode::kUnsupportedCapability)
        return 18;
    if (!created.transport->unregister_symmetric_window().ok() ||
        !created.transport->release_channels().ok())
        return 19;
    if (!created.transport->destroy().ok() || !created.transport->destroy().ok())
        return 20;

    const auto expect_invalid = [](TransportConfig invalid,
                                   const std::string& fragment) {
        auto rejected = make_stub_transport(invalid);
        return !rejected.transport &&
               rejected.status.code == TransportStatusCode::kInvalidArgument &&
               rejected.status.message.find(fragment) != std::string::npos;
    };

    auto invalid = valid_config();
    invalid.communicator_handle = 7;
    if (!expect_invalid(invalid, "communicator_handle must be zero"))
        return 21;
    invalid = valid_config();
    invalid.world_size = 0;
    if (!expect_invalid(invalid, "world_size must be positive"))
        return 22;
    invalid = valid_config();
    invalid.rank = invalid.world_size;
    if (!expect_invalid(invalid, "rank must be in [0, world_size)"))
        return 23;
    invalid = valid_config();
    invalid.cpu_communicator_empty = false;
    if (!expect_invalid(invalid, "cpu_communicator must be empty"))
        return 24;
    invalid = valid_config();
    invalid.device_buffer_bytes = 0;
    if (!expect_invalid(invalid, "device_buffer_bytes must be positive"))
        return 25;
    invalid = valid_config();
    invalid.cpu_buffer_bytes = 4096;
    if (!expect_invalid(invalid, "cpu_buffer_bytes must be zero"))
        return 26;
    invalid = valid_config();
    invalid.requested_channels = -1;
    if (!expect_invalid(invalid, "requested_channels must be non-negative"))
        return 27;

    return 0;
}
