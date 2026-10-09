cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)

project(netlayer_ab LANGUAGES ASC CXX)

add_library(netlayer_ab SHARED
    netlayer_ab.asc
    netlayer_ab.cpp)

set_target_properties(netlayer_ab PROPERTIES
    LINKER_LANGUAGE ASC
    PREFIX "")

target_compile_options(netlayer_ab PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

target_include_directories(netlayer_ab PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include)
