cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")

find_package(ASC REQUIRED)
project(deep_ep_ascend_core_operator_probe LANGUAGES ASC CXX)

add_library(deep_ep_ascend_core_operator_probe OBJECT
    ../../../csrc/backends/ascend/elastic/barrier.asc
    ../../../csrc/backends/ascend/elastic/dispatch.asc
    ../../../csrc/backends/ascend/elastic/combine.asc
    ../../../csrc/backends/ascend/elastic/runtime.cpp
    core_operator_compile_probe.asc)

target_compile_options(deep_ep_ascend_core_operator_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

target_include_directories(deep_ep_ascend_core_operator_probe PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../..
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include)

add_executable(deep_ep_ascend_core_operator_runner
    core_operator_runner.asc
    $<TARGET_OBJECTS:deep_ep_ascend_core_operator_probe>)

set_target_properties(deep_ep_ascend_core_operator_runner PROPERTIES
    LINKER_LANGUAGE ASC)

target_compile_options(deep_ep_ascend_core_operator_runner PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

target_include_directories(deep_ep_ascend_core_operator_runner PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../..
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include)
