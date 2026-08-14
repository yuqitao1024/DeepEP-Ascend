cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")

find_package(ASC REQUIRED)
project(deep_ep_ascend_simt_urma_probe LANGUAGES ASC CXX)

option(DEEP_EP_BUILD_URMA_RUNTIME "Build the two-NPU URMA runtime probe" OFF)
option(DEEP_EP_ASCEND_STAGED_URMA "Enable the staged Ascend URMA path" OFF)

add_library(deep_ep_ascend_staged_primitive_probe OBJECT
    phase_boundary_probe.asc)

add_library(deep_ep_ascend_command_encoder_probe OBJECT
    command_encoder_probe.asc)

add_library(deep_ep_ascend_aicore_service_probe OBJECT
    aicore_service_probe.asc)

target_compile_options(deep_ep_ascend_staged_primitive_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

target_compile_options(deep_ep_ascend_command_encoder_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

target_compile_options(deep_ep_ascend_aicore_service_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

target_include_directories(deep_ep_ascend_staged_primitive_probe PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../..
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include)

target_include_directories(deep_ep_ascend_command_encoder_probe PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../..
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include)

target_include_directories(deep_ep_ascend_aicore_service_probe PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../..
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include)

if(DEEP_EP_BUILD_URMA_RUNTIME)
    add_library(deep_ep_ascend_urma_runner SHARED
        runtime_probe.asc
        runtime_probe_main.cpp
        ../../../csrc/backends/ascend/transport/cann_transport.cpp)

    set_target_properties(deep_ep_ascend_urma_runner PROPERTIES
        LINKER_LANGUAGE ASC
        PREFIX "")

    target_compile_options(deep_ep_ascend_urma_runner PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

    target_include_directories(deep_ep_ascend_urma_runner PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../..
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include)

    target_link_directories(deep_ep_ascend_urma_runner PRIVATE
        $ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64)
    target_link_libraries(deep_ep_ascend_urma_runner PRIVATE
        hcomm ascendcl c_sec)
endif()
