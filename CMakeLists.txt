# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
set(DEEP_EP_PLATFORM "cuda" CACHE STRING "DeepEP platform: cuda or ascend")
set_property(CACHE DEEP_EP_PLATFORM PROPERTY STRINGS cuda ascend)

if(NOT DEEP_EP_PLATFORM STREQUAL "cuda" AND
   NOT DEEP_EP_PLATFORM STREQUAL "ascend")
    message(FATAL_ERROR "DEEP_EP_PLATFORM must be cuda or ascend")
endif()

if(DEEP_EP_PLATFORM STREQUAL "cuda")
    project(deep_ep LANGUAGES CUDA CXX)
else()
    project(deep_ep LANGUAGES CXX)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

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
endif()

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

if(DEEP_EP_PLATFORM STREQUAL "cuda")
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
else()
    set(CMAKE_CXX_STANDARD 17)
    include_directories(deep_ep/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} .)
    link_directories(${TORCH_INSTALL_PREFIX}/lib)
endif()

# Link CPP and CUDA together
pybind11_add_module(_C csrc/python_api.cpp)

if(DEEP_EP_PLATFORM STREQUAL "cuda")
    target_link_libraries(_C PRIVATE nccl ${RUNTIME_CUDA_LIBRARIES} ${LEGACY_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
    target_compile_definitions(_C PRIVATE DEEP_EP_PLATFORM_CUDA=1)
else()
    target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python)
    target_compile_definitions(_C PRIVATE DEEP_EP_PLATFORM_ASCEND=1)
endif()

if(DEEP_EP_PLATFORM STREQUAL "cuda")
    # Enable kernel code indexing with CMake-based IDEs
    cuda_add_library(deep_ep_indexing_cuda STATIC csrc/indexing/main.cu)
endif()
