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
