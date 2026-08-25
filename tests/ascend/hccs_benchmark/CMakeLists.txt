cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)
project(deep_ep_hccs_benchmark LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
set(HCOMM_ROOT "$ENV{HCOMM_ROOT}")
if(NOT ASCEND_HOME_PATH OR NOT HCOMM_ROOT)
    message(FATAL_ERROR "ASCEND_HOME_PATH and HCOMM_ROOT are required")
endif()

set(ASCEND_ARCH_ROOT "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
set(HCOMM_ARCH_ROOT "${HCOMM_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux")
get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_library(hccs_benchmark SHARED
    hccs_benchmark.asc
    hccs_benchmark_main.cpp
    ${REPOSITORY_ROOT}/csrc/backends/ascend/transport/cann_transport.cpp)

set_target_properties(hccs_benchmark PROPERTIES
    LINKER_LANGUAGE ASC
    PREFIX "")

target_compile_options(hccs_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_compile_definitions(hccs_benchmark PRIVATE
    DEEP_EP_ASCEND_TESTING=0
    $<$<COMPILE_LANGUAGE:ASC>:DEEP_EP_ASCEND_STAGED_URMA=1>
    $<$<COMPILE_LANGUAGE:ASC>:DEEP_EP_ASCEND_AICORE_URMA_SERVICE=1>
    $<$<COMPILE_LANGUAGE:ASC>:DEEP_EP_ASCEND_AICORE_WQE_CALLEE=__aicore__>)
target_include_directories(hccs_benchmark PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${REPOSITORY_ROOT}
    ${HCOMM_ARCH_ROOT}/pkg_inc
    ${HCOMM_ARCH_ROOT}/include
    ${HCOMM_ARCH_ROOT}/include/hcomm
    ${HCOMM_ROOT}/include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_ARCH_ROOT}/asc/include)
target_link_directories(hccs_benchmark PRIVATE
    ${HCOMM_ROOT}/lib64
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_ARCH_ROOT}/lib64)
target_link_libraries(hccs_benchmark PRIVATE hcomm ascendcl c_sec)
