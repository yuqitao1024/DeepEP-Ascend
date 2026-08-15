cmake_minimum_required(VERSION 3.16)
set(DEEP_EP_PLATFORM "cuda" CACHE STRING "DeepEP platform: cuda or ascend")
set_property(CACHE DEEP_EP_PLATFORM PROPERTY STRINGS cuda ascend)

if(NOT DEEP_EP_PLATFORM STREQUAL "cuda" AND
   NOT DEEP_EP_PLATFORM STREQUAL "ascend")
    message(FATAL_ERROR "DEEP_EP_PLATFORM must be cuda or ascend")
endif()

if(DEEP_EP_PLATFORM STREQUAL "cuda")
    project(deep_ep LANGUAGES CUDA CXX)
else()
    set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
    find_package(ASC REQUIRED)
    project(deep_ep LANGUAGES ASC CXX)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

if(DEEP_EP_PLATFORM STREQUAL "cuda")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CUDA_STANDARD 20)
    set(CUDA_SEPARABLE_COMPILATION ON)
    list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
    list(APPEND CUDA_NVCC_FLAGS "-O3")
    list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-rdynamic")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo")
    # Suppress warnings for the `fmt` library
    list(APPEND CUDA_NVCC_FLAGS "--diag-suppress=128,2417")

    set(USE_SYSTEM_NVTX on)
    set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
    set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

    find_package(CUDAToolkit REQUIRED)

    # NVSHMEM
    find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)
    add_library(nvshmem ALIAS nvshmem::nvshmem)
    add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
    add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

    # NCCL
    # TODO: use `find_package` instead of manual checks
    if (NOT NCCL_ROOT_DIR)
        message(FATAL_ERROR "NCCL_ROOT_DIR is not set.")
    endif()

    include_directories(deep_ep/include third-party/fmt/include)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include/cccl ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NVSHMEM_INCLUDE_DIR} ${NCCL_ROOT_DIR}/include .)
    link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib ${NVSHMEM_LIB_DIR} ${NCCL_ROOT_DIR}/lib)

    # Add kernels
    add_subdirectory(csrc)

    pybind11_add_module(_C csrc/python_api.cpp)
    target_link_libraries(_C PRIVATE nccl ${RUNTIME_CUDA_LIBRARIES} ${LEGACY_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
    target_compile_definitions(_C PRIVATE DEEP_EP_PLATFORM_CUDA=1)
endif()

if(DEEP_EP_PLATFORM STREQUAL "cuda")
    # Enable kernel code indexing with CMake-based IDEs
    cuda_add_library(deep_ep_indexing_cuda STATIC csrc/indexing/main.cu)
endif()

if(DEEP_EP_PLATFORM STREQUAL "ascend")
    set(CMAKE_CXX_STANDARD 17)
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    set(HCOMM_ROOT "$ENV{HCOMM_ROOT}")
    if(NOT ASCEND_HOME_PATH)
        message(FATAL_ERROR "ASCEND_HOME_PATH is required for an Ascend build")
    endif()
    if(NOT HCOMM_ROOT)
        message(FATAL_ERROR "HCOMM_ROOT is required for an Ascend build")
    endif()
    if(NOT TORCH_NPU_ROOT)
        message(FATAL_ERROR "TORCH_NPU_ROOT is required for an Ascend build")
    endif()

    set(HCOMM_ARCH_ROOT "${HCOMM_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(ASCEND_ARCH_ROOT "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")

    pybind11_add_module(_C
        csrc/python_api.cpp
        csrc/backends/ascend/elastic/barrier.asc
        csrc/backends/ascend/elastic/dispatch.asc
        csrc/backends/ascend/elastic/combine.asc
        csrc/backends/ascend/elastic/runtime.cpp
        csrc/backends/ascend/runtime/cann_runtime.cpp
        csrc/backends/ascend/transport/cann_transport.cpp)

    target_include_directories(_C PRIVATE
        ${HCOMM_ARCH_ROOT}/pkg_inc
        ${HCOMM_ARCH_ROOT}/include
        ${HCOMM_ARCH_ROOT}/include/hcomm
        ${HCOMM_ROOT}/include
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_ARCH_ROOT}/asc/include
        ${TORCH_NPU_ROOT}/include
        ${TORCH_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/deep_ep/include
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_directories(_C PRIVATE
        ${HCOMM_ROOT}/lib64
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_ARCH_ROOT}/lib64
        ${TORCH_NPU_ROOT}/lib
        ${TORCH_INSTALL_PREFIX}/lib)
    target_compile_options(_C PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>)
    target_compile_definitions(_C PRIVATE
        DEEP_EP_PLATFORM_ASCEND=1
        $<$<COMPILE_LANGUAGE:ASC>:DEEP_EP_ASCEND_STAGED_URMA=1>
        $<$<COMPILE_LANGUAGE:ASC>:DEEP_EP_ASCEND_AICORE_URMA_SERVICE=1>)
    target_link_libraries(_C PRIVATE
        ${TORCH_LIBRARIES} torch_python torch_npu hcomm ascendcl c_sec)
    set_target_properties(_C PROPERTIES LINKER_LANGUAGE ASC)
endif()
