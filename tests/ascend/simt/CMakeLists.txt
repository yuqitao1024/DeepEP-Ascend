cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")

find_package(ASC REQUIRED)

project(deep_ep_ascend_transport_probe LANGUAGES ASC CXX)

add_library(deep_ep_ascend_transport_probe OBJECT
    ../device_transport_simt_probe.asc)

# ASC CMake supplies --asc-aicore-lang. On CANN 9.2, adding --enable-simt
# selects pure-SIMT mode, which does not expose the explicit VF_CALL seam.
target_compile_options(deep_ep_ascend_transport_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

target_include_directories(deep_ep_ascend_transport_probe PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../..
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include)
