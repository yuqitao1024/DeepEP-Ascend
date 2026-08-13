COMMON_MODULE_NAMES = {
    "get_platform", "topk_idx_t", "EventHandle", "ElasticBuffer",
    "calculate_elastic_buffer_size",
}
COMMON_BUFFER_METHODS = {
    "destroy", "get_comm_stream", "get_physical_domain_size",
    "get_logical_domain_size", "barrier", "dispatch", "combine",
}
CUDA_ONLY_MODULE_NAMES = {
    "is_sm90_compiled", "init_jit", "Config", "Buffer",
    "get_low_latency_rdma_size_hint", "get_local_nccl_unique_id",
    "create_nccl_comm", "destroy_nccl_comm", "get_physical_domain_size",
    "get_logical_domain_size",
}
CUDA_ONLY_BUFFER_METHODS = {
    "engram_write", "engram_fetch", "pp_set_config", "pp_send", "pp_recv",
    "create_agrs_session", "destroy_agrs_session", "agrs_set_config",
    "agrs_get_inplace_tensor", "all_gather",
}

ASCEND_MODULE_NAMES = COMMON_MODULE_NAMES
CUDA_MODULE_NAMES = COMMON_MODULE_NAMES | CUDA_ONLY_MODULE_NAMES
ASCEND_ELASTIC_BUFFER_METHODS = COMMON_BUFFER_METHODS
CUDA_ELASTIC_BUFFER_METHODS = COMMON_BUFFER_METHODS | CUDA_ONLY_BUFFER_METHODS

# These sets mirror only methods registered through the current PyBind surface.
CUDA_BUFFER_METHODS = {
    "is_available", "get_num_rdma_ranks", "get_rdma_rank", "get_root_rdma_rank",
    "get_local_device_id", "get_local_ipc_handle", "get_local_nvshmem_unique_id",
    "get_local_buffer_tensor", "get_comm_stream", "sync", "destroy",
    "get_dispatch_layout", "intranode_dispatch", "intranode_combine",
    "internode_dispatch", "internode_combine", "clean_low_latency_buffer",
    "low_latency_dispatch", "low_latency_combine", "low_latency_update_mask_buffer",
    "low_latency_query_mask_buffer", "low_latency_clean_mask_buffer",
    "get_next_low_latency_combine_buffer",
}
CUDA_CONFIG_METHODS = {
    "get_nvl_buffer_size_hint", "get_rdma_buffer_size_hint",
}
CUDA_EVENT_HANDLE_METHODS = {"current_stream_wait"}
