#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_rank_graph.h"
#include "hcomm/hcomm_res.h"

#include "netlayer_ab_abi.hpp"

extern "C" int netlayer_ab_launch(void* control, void* stream);

namespace {

struct Options {
    int device = -1;
    std::uint32_t rank = 0;
    std::uint32_t world_size = 8;
    std::uint32_t layer_index = 0;
    std::uint64_t payload_bytes = 64ULL << 20;
    std::uint64_t chunk_bytes = 4ULL << 20;
    std::uint32_t iterations = 10;
    std::uint32_t warmup = 3;
    std::uint32_t timeout_cycles = 100000000;
    std::string root_info_path;
    bool publish_root_info = false;
    std::string make_root_info_path;
    std::string result_path;
};

struct RootInfo {
    char data[4108];
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", argument.c_str());
                std::exit(2);
            }
            return argv[++index];
        };
        if (argument == "--device") {
            options.device = std::stoi(next());
        } else if (argument == "--rank") {
            options.rank = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (argument == "--world-size") {
            options.world_size = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (argument == "--layer-index") {
            options.layer_index = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (argument == "--payload-bytes") {
            options.payload_bytes = std::stoull(next());
        } else if (argument == "--chunk-bytes") {
            options.chunk_bytes = std::stoull(next());
        } else if (argument == "--iterations") {
            options.iterations = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (argument == "--warmup") {
            options.warmup = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (argument == "--timeout-cycles") {
            options.timeout_cycles = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (argument == "--root-info") {
            options.root_info_path = next();
        } else if (argument == "--publish-root-info") {
            options.publish_root_info = true;
        } else if (argument == "--make-root-info") {
            options.make_root_info_path = next();
        } else if (argument == "--result") {
            options.result_path = next();
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argument.c_str());
            std::exit(2);
        }
    }
    if (options.device < 0)
        options.device = static_cast<int>(options.rank);
    const bool making_root_info = !options.make_root_info_path.empty();
    if (!making_root_info &&
        (options.world_size < 2 || options.rank >= options.world_size)) {
        std::fprintf(stderr, "invalid rank/world_size\n");
        std::exit(2);
    }
    if (!making_root_info &&
        (options.payload_bytes == 0 || options.chunk_bytes == 0 ||
        options.chunk_bytes > options.payload_bytes ||
        options.payload_bytes % options.chunk_bytes != 0 ||
        options.chunk_bytes > 0xffffffffULL)) {
        std::fprintf(stderr, "payload_bytes must be a positive multiple of chunk_bytes <= 4GB\n");
        std::exit(2);
    }
    if (making_root_info)
        options.root_info_path = options.make_root_info_path;
    if (!making_root_info && options.root_info_path.empty()) {
        std::fprintf(stderr, "--root-info is required\n");
        std::exit(2);
    }
    return options;
}

bool read_file(const std::string& path, void* destination, std::size_t bytes) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr)
        return false;
    const std::size_t read = std::fread(destination, 1, bytes, file);
    std::fclose(file);
    return read == bytes;
}

bool write_file(const std::string& path, const std::string& content) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
        return false;
    const std::size_t written = std::fwrite(content.data(), 1, content.size(), file);
    std::fclose(file);
    return written == content.size();
}

std::string format_status(const char* operation, int status) {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "%s failed with status %d", operation, status);
    return buffer;
}

bool check_acl(aclError status, const char* operation, std::string* error) {
    if (status == ACL_SUCCESS)
        return true;
    *error = format_status(operation, static_cast<int>(status));
    return false;
}

bool check_hccl(HcclResult status, const char* operation, std::string* error) {
    if (status == HCCL_SUCCESS)
        return true;
    *error = format_status(operation, static_cast<int>(status));
    return false;
}

bool init_root_info(const std::string& path) {
    HcclRootInfo root{};
    // HcclGetRootInfo requires an accessible device context in some
    // container configurations.  The benchmark's launcher instead exchanges
    // a rank-0 generated root blob; this helper remains for manual use.
    (void)HcclGetRootInfo(&root);
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
        return false;
    const std::size_t written =
        std::fwrite(root.internal, 1, sizeof(root.internal), file);
    std::fclose(file);
    return written == sizeof(root.internal);
}

bool make_root_info(const Options& options) {
    // HcclGetRootInfo accepts a rank-independent bootstrap token and can run
    // without reserving an NPU context.  This avoids contending with another
    // process when the batch scheduler has already allocated all devices.
    HcclRootInfo root{};
    std::string error;
    if (!check_hccl(HcclGetRootInfoScalable(&root),
                    "HcclGetRootInfoScalable", &error)) {
        std::fprintf(stderr, "make_root_info: %s\n", error.c_str());
        return false;
    }
    return write_file(options.make_root_info_path,
                      std::string(root.internal,
                                  root.internal + sizeof(root.internal)));
}

bool protocol_supported(std::int32_t protocol) {
#if defined(NETLAYER_AB_USE_UB_MEM)
    return protocol == COMM_PROTOCOL_UB_MEM;
#else
    return protocol == COMM_PROTOCOL_UB_CTP;
#endif
}

struct Resources {
    aclrtStream stream = nullptr;
    HcclComm comm = nullptr;
    void* payload = nullptr;
    HcclMemHandle payload_handle = nullptr;
    void* remote_addresses = nullptr;
    void* channels = nullptr;
    void* control = nullptr;
    std::vector<ChannelHandle> channel_handles;
    std::vector<std::uint32_t> active_peers;
};

void destroy(Resources& resources, const Options& options) {
    if (resources.comm != nullptr &&
        std::any_of(resources.channel_handles.begin(),
                    resources.channel_handles.end(),
                    [](ChannelHandle handle) { return handle != 0; })) {
        (void)HcclChannelDestroy(resources.comm,
                                 resources.channel_handles.data(),
                                 options.world_size);
    }
    (void)HcclCommDestroy(resources.comm);
    if (resources.stream != nullptr)
        (void)aclrtDestroyStream(resources.stream);
    if (resources.payload != nullptr)
        (void)aclrtFree(resources.payload);
    if (resources.remote_addresses != nullptr)
        (void)aclrtFree(resources.remote_addresses);
    if (resources.channels != nullptr)
        (void)aclrtFree(resources.channels);
    if (resources.control != nullptr)
        (void)aclrtFree(resources.control);
    (void)aclrtResetDevice(options.device);
    (void)aclFinalize();
}

bool remote_payload_for_channel(HcclComm comm,
                                ChannelHandle handle,
                                const char* tag,
                                std::uint64_t* address,
                                std::string* error) {
    std::uint32_t count = 0;
    CommMem* memories = nullptr;
    char** tags = nullptr;
    if (!check_hccl(HcclChannelGetRemoteMems(comm, handle, &count, &memories, &tags),
                    "HcclChannelGetRemoteMems", error))
        return false;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (tags != nullptr && tags[index] != nullptr &&
            std::strcmp(tags[index], tag) == 0) {
            *address = reinterpret_cast<std::uint64_t>(memories[index].addr);
            return true;
        }
    }
    *error = "remote payload memory tag was not found";
    return false;
}

bool init_channels(Resources& resources,
                   const Options& options,
                   std::string* error) {
    uint32_t* layers = nullptr;
    std::uint32_t layer_count = 0;
    if (!check_hccl(HcclRankGraphGetLayers(resources.comm, &layers, &layer_count),
                    "HcclRankGraphGetLayers", error))
        return false;
    std::fprintf(stderr,
                 "rank=%u layers=%u values=%u,%u selected=%u\n",
                 options.rank,
                 layer_count,
                 layer_count > 0 ? layers[0] : 0,
                 layer_count > 1 ? layers[1] : 0,
                 options.layer_index);
    if (layers == nullptr || layer_count == 0 || options.layer_index >= layer_count) {
        *error = "selected network layer is unavailable";
        return false;
    }
    // HcclRankGraphGetLinks is allowed to invalidate library-managed results
    // from a previous query.  Copy the layer value before entering the loop.
    const std::uint32_t selected_layer = layers[options.layer_index];

    resources.channel_handles.assign(options.world_size, 0);
    std::vector<std::uint32_t> supported_peers;
    std::vector<std::uint32_t> unsupported_peers;
    for (std::uint32_t peer = 0; peer < options.world_size; ++peer) {
        if (peer == options.rank)
            continue;
        CommLink* links = nullptr;
        std::uint32_t link_count = 0;
        const HcclResult links_status = HcclRankGraphGetLinks(
            resources.comm, selected_layer, options.rank, peer,
            &links, &link_count);
        if (links_status != HCCL_SUCCESS) {
            std::fprintf(stderr,
                         "rank=%u layer=%u peer=%u HcclRankGraphGetLinks failed with status %d\n",
                         options.rank,
                         options.layer_index,
                         peer,
                         static_cast<int>(links_status));
            unsupported_peers.push_back(peer);
            continue;
        }
        std::fprintf(stderr,
                     "rank=%u layer=%u peer=%u links=%u\n",
                     options.rank,
                     options.layer_index,
                     peer,
                     link_count);
        bool acquired = false;
        for (std::uint32_t index = 0; index < link_count; ++index) {
            if (!protocol_supported(links[index].linkAttr.linkProtocol))
                continue;
            HcclChannelDesc description{};
            if (!check_hccl(HcclChannelDescInit(&description, 1),
                            "HcclChannelDescInit", error))
                return false;
            description.remoteRank = peer;
            description.channelProtocol = links[index].linkAttr.linkProtocol;
            description.localEndpoint = links[index].srcEndpointDesc;
            description.remoteEndpoint = links[index].dstEndpointDesc;
            description.notifyNum = 0;
            description.memHandles = &resources.payload_handle;
            description.memHandleNum = 1;
            if (!check_hccl(HcclChannelAcquire(resources.comm,
                                                COMM_ENGINE_AIV,
                                                &description,
                                                1,
                                                &resources.channel_handles[peer]),
                            "HcclChannelAcquire", error))
                return false;
            acquired = true;
            supported_peers.push_back(peer);
            resources.active_peers.push_back(peer);
            break;
        }
        if (!acquired)
            unsupported_peers.push_back(peer);
    }
    std::fprintf(stderr,
                 "rank=%u layer=%u supported_peers=%u unsupported_peers=%u\n",
                 options.rank,
                 options.layer_index,
                 static_cast<std::uint32_t>(supported_peers.size()),
                 static_cast<std::uint32_t>(unsupported_peers.size()));
    if (!unsupported_peers.empty()) {
        std::string peers;
        for (const auto peer : unsupported_peers) {
            if (!peers.empty())
                peers.push_back(',');
            peers += std::to_string(peer);
        }
        std::fprintf(stderr,
                     "rank=%u layer=%u unsupported_peer_list=%s\n",
                     options.rank,
                     options.layer_index,
                     peers.c_str());
    }
    std::string supported_list;
    for (const auto peer : supported_peers) {
        if (!supported_list.empty())
            supported_list.push_back(',');
        supported_list += std::to_string(peer);
    }
    std::fprintf(stderr,
                 "rank=%u layer=%u supported_peer_list=%s\n",
                 options.rank,
                 options.layer_index,
                 supported_list.c_str());
    if (supported_peers.empty()) {
        *error = "selected network layer has no supported peer";
        return false;
    }

    // Channel acquire is connection-oriented.  Let every rank finish local
    // acquire before querying remote memory-registration tables.
    if (!check_hccl(HcclBarrier(resources.comm, resources.stream),
                    "HcclBarrier after channel acquire", error))
        return false;

    if (!check_acl(aclrtMalloc(&resources.remote_addresses,
                               options.world_size * sizeof(std::uint64_t),
                               ACL_MEM_MALLOC_HUGE_FIRST),
                   "allocate remote addresses", error) ||
        !check_acl(aclrtMalloc(&resources.channels,
                               options.world_size * sizeof(std::uint64_t),
                               ACL_MEM_MALLOC_HUGE_FIRST),
                   "allocate channel table", error))
        return false;

    std::fprintf(stderr,
                 "rank=%u allocated communication tables\n",
                 options.rank);

    std::vector<std::uint64_t> remote_addresses(options.world_size);
    std::vector<std::uint64_t> channel_addresses(options.world_size);
    for (std::uint32_t peer = 0; peer < options.world_size; ++peer) {
        if (peer == options.rank) {
            remote_addresses[peer] = reinterpret_cast<std::uint64_t>(resources.payload);
            channel_addresses[peer] = 0;
            continue;
        }
        if (resources.channel_handles[peer] == 0) {
            remote_addresses[peer] = 0;
            channel_addresses[peer] = 0;
            continue;
        }
        if (!remote_payload_for_channel(resources.comm,
                                        resources.channel_handles[peer],
                                        "NetlayerABPayload",
                                        &remote_addresses[peer],
                                        error))
            return false;
        channel_addresses[peer] = static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(resources.channel_handles[peer]));
    }
    if (!check_acl(aclrtMemcpy(resources.remote_addresses,
                               remote_addresses.size() * sizeof(remote_addresses[0]),
                               remote_addresses.data(),
                               remote_addresses.size() * sizeof(remote_addresses[0]),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "copy remote addresses", error) ||
        !check_acl(aclrtMemcpy(resources.channels,
                               channel_addresses.size() * sizeof(channel_addresses[0]),
                               channel_addresses.data(),
                               channel_addresses.size() * sizeof(channel_addresses[0]),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "copy channel table", error))
        return false;
    return true;
}

bool initialize(Resources& resources, const Options& options, std::string* error) {
    if (!check_acl(aclInit(nullptr), "aclInit", error) ||
        !check_acl(aclrtSetDevice(options.device), "aclrtSetDevice", error) ||
        !check_acl(aclrtCreateStream(&resources.stream), "aclrtCreateStream", error))
        return false;
    HcclRootInfo root{};
    if (options.publish_root_info) {
        if (options.rank != 0) {
            *error = "only rank 0 may publish root info";
            return false;
        }
        if (!check_hccl(HcclGetRootInfo(&root), "HcclGetRootInfo", error))
            return false;
        if (!write_file(options.root_info_path,
                        std::string(root.internal,
                                    root.internal + sizeof(root.internal)))) {
            *error = "failed to publish root info";
            return false;
        }
        const std::string ready_path = options.root_info_path + ".ready";
        if (!write_file(ready_path, "ready\n")) {
            *error = "failed to publish root-info ready marker";
            return false;
        }
    } else {
        const std::string ready_path = options.root_info_path + ".ready";
        for (int attempt = 0; attempt < 3000; ++attempt) {
            FILE* ready = std::fopen(ready_path.c_str(), "rb");
            if (ready != nullptr) {
                std::fclose(ready);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    if (!read_file(options.root_info_path, root.internal, sizeof(root.internal))) {
        *error = "failed to read root info";
        return false;
    }
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = options.rank;
    config.hcclJobID = 1;
    config.hcclSymWinMaxMemSizePerRank = 1;
    if (!check_hccl(HcclCommInitRootInfoConfig(static_cast<std::uint32_t>(options.world_size),
                                                &root,
                                                static_cast<std::uint32_t>(options.rank),
                                                &config,
                                                &resources.comm),
                    "HcclCommInitRootInfoConfig", error))
        return false;

    if (!check_acl(aclrtMalloc(&resources.payload,
                               static_cast<std::size_t>(options.payload_bytes),
                               ACL_MEM_MALLOC_HUGE_FIRST),
                   "allocate payload", error))
        return false;
    std::vector<std::uint64_t> words(options.payload_bytes / sizeof(std::uint64_t));
    for (std::uint64_t index = 0; index < words.size(); ++index)
        words[index] = (static_cast<std::uint64_t>(options.rank + 1) << 32) |
                       (index / (1024 * 1024 / sizeof(std::uint64_t)));
    if (!check_acl(aclrtMemcpy(resources.payload,
                               static_cast<std::size_t>(options.payload_bytes),
                               words.data(),
                               static_cast<std::size_t>(options.payload_bytes),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "initialize payload", error))
        return false;

    CommMem memory{};
    memory.type = COMM_MEM_TYPE_DEVICE;
    memory.addr = resources.payload;
    memory.size = options.payload_bytes;
    if (!check_hccl(HcclCommMemReg(resources.comm,
                                   "NetlayerABPayload",
                                   &memory,
                                   &resources.payload_handle),
                    "HcclCommMemReg", error))
        return false;
    if (!init_channels(resources, options, error))
        return false;
    if (!check_acl(aclrtMalloc(&resources.control,
                               sizeof(netlayer_ab::DeviceControl),
                               ACL_MEM_MALLOC_HUGE_FIRST),
                   "allocate control", error))
        return false;
    if (!check_hccl(HcclBarrier(resources.comm, resources.stream), "HcclBarrier", error))
        return false;
    return true;
}

bool verify(Resources& resources, const Options& options, std::string* error) {
    std::vector<std::uint64_t> words(options.payload_bytes / sizeof(std::uint64_t));
    if (!check_acl(aclrtMemcpy(words.data(),
                               words.size() * sizeof(words[0]),
                               resources.payload,
                               words.size() * sizeof(words[0]),
                               ACL_MEMCPY_DEVICE_TO_HOST),
                   "copy payload for verification", error))
        return false;
    for (std::uint32_t source = 0; source < options.world_size; ++source) {
        if (source == options.rank)
            continue;
        bool found = false;
        // Each remote rank overwrites the same local payload, so the current
        // value can come from any source.  Validate that it is a valid rank
        // marker rather than stale/zero memory.
        for (std::uint64_t index = 0; index < words.size(); index += 4096) {
            const auto tag = words[index] >> 32;
            if (tag >= 1 && tag <= options.world_size && tag != options.rank + 1) {
                found = true;
                break;
            }
        }
        (void)source;
        if (!found) {
            *error = "payload did not contain a remote rank marker";
            return false;
        }
    }
    return true;
}

bool run(Resources& resources,
         const Options& options,
         std::vector<double>& elapsed_ms,
         std::string* error) {
    netlayer_ab::DeviceControl control{};
    control.local_payload = reinterpret_cast<std::uint64_t>(resources.payload);
    control.payload_bytes = options.payload_bytes;
    control.chunk_bytes = options.chunk_bytes;
    control.rank = options.rank;
    control.world_size = options.world_size;
    control.remote_addresses =
        reinterpret_cast<std::uint64_t>(resources.remote_addresses);
    control.channels =
        reinterpret_cast<std::uint64_t>(resources.channels);
    control.timeout_cycles = options.timeout_cycles;
    if (!check_acl(aclrtMemcpy(resources.control,
                               sizeof(control),
                               &control,
                               sizeof(control),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "copy control", error))
        return false;

    for (std::uint32_t iteration = 0;
         iteration < options.warmup + options.iterations;
         ++iteration) {
        if (!check_hccl(HcclBarrier(resources.comm, resources.stream), "HcclBarrier", error))
            return false;
        const auto start = std::chrono::steady_clock::now();
        const int launch_status = netlayer_ab_launch(resources.control, resources.stream);
        if (launch_status != 0) {
            *error = format_status("netlayer_ab_launch", launch_status);
            return false;
        }
        if (!check_acl(aclrtSynchronizeStream(resources.stream),
                       "aclrtSynchronizeStream", error))
            return false;
        const auto end = std::chrono::steady_clock::now();
        if (!check_hccl(HcclBarrier(resources.comm, resources.stream), "HcclBarrier", error))
            return false;

        netlayer_ab::DeviceControl result{};
        if (!check_acl(aclrtMemcpy(&result,
                                   sizeof(result),
                                   resources.control,
                                   sizeof(result),
                                   ACL_MEMCPY_DEVICE_TO_HOST),
                       "copy result", error))
            return false;
        if (result.success != 2) {
            char buffer[256];
            std::snprintf(buffer,
                          sizeof(buffer),
                          "kernel failed: stage=%u peer=%u status=%u",
                          result.failed_stage,
                          result.failed_peer,
                          result.backend_status);
            *error = buffer;
            return false;
        }
        if (iteration >= options.warmup)
            elapsed_ms.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
    }
    if (!verify(resources, options, error))
        return false;
    return true;
}

}  // namespace

int netlayer_ab_main(const std::vector<std::string>& argument_values) {
    std::vector<char*> argv_storage;
    argv_storage.reserve(argument_values.size() + 2);
    const char program[] = "netlayer_ab";
    argv_storage.push_back(const_cast<char*>(program));
    for (const auto& argument : argument_values)
        argv_storage.push_back(const_cast<char*>(argument.c_str()));
    argv_storage.push_back(nullptr);
    char** argv = argv_storage.data();
    int argc = static_cast<int>(argv_storage.size() - 1);
    const Options options = parse_options(argc, argv);
    if (!options.make_root_info_path.empty())
        return make_root_info(options) ? 0 : 1;
    Resources resources{};
    std::string error;
    std::vector<double> elapsed_ms;
    if (!initialize(resources, options, &error) ||
        !run(resources, options, elapsed_ms, &error)) {
        std::fprintf(stderr,
                     "rank=%u ERROR %s\n",
                     options.rank,
                     error.c_str());
        destroy(resources, options);
        return 1;
    }

    double total = 0;
    double minimum = elapsed_ms.empty() ? 0 : elapsed_ms[0];
    double maximum = 0;
    for (const auto value : elapsed_ms) {
        total += value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const double mean = elapsed_ms.empty() ? 0 : total / elapsed_ms.size();
    const double bytes = static_cast<double>(resources.active_peers.size()) *
                         options.payload_bytes;
    const double gbps = elapsed_ms.empty() ?
        0 : bytes / (mean / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    std::printf(
        "rank=%u layer=%u payload=%llu chunk=%llu iterations=%u "
        "mean_ms=%.6f min_ms=%.6f max_ms=%.6f send_gibps=%.6f\n",
        options.rank,
        options.layer_index,
        static_cast<unsigned long long>(options.payload_bytes),
        static_cast<unsigned long long>(options.chunk_bytes),
        options.iterations,
        mean,
        minimum,
        maximum,
        gbps);
    if (!options.result_path.empty()) {
        std::string content = std::to_string(mean) + "\n" +
                              std::to_string(minimum) + "\n" +
                              std::to_string(maximum) + "\n" +
                              std::to_string(gbps) + "\n" +
                              std::to_string(resources.active_peers.size()) + "\n";
        if (!write_file(options.result_path, content)) {
            std::fprintf(stderr, "rank=%u failed to write result\n", options.rank);
            destroy(resources, options);
            return 1;
        }
    }
    destroy(resources, options);
    return 0;
}
