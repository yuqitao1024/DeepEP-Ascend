cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)
project(deep_ep_ascend_argument_list_330_repro LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(DEEP_EP_REPRO_LEGACY_ARGS
    "Also compile the pre-struct individual-argument kernel" OFF)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH is required")
endif()

add_executable(argument_list_330_repro
    argument_list_330_repro.asc
    argument_list_330_repro_main.cpp)

set_target_properties(argument_list_330_repro PROPERTIES LINKER_LANGUAGE ASC)

target_compile_options(argument_list_330_repro PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_compile_definitions(argument_list_330_repro PRIVATE
    $<$<BOOL:${DEEP_EP_REPRO_LEGACY_ARGS}>:DEEP_EP_REPRO_LEGACY_ARGS>)

target_include_directories(argument_list_330_repro PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/x86_64-linux/asc/include)

target_link_directories(argument_list_330_repro PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/x86_64-linux/lib64)
target_link_libraries(argument_list_330_repro PRIVATE ascendcl c_sec)
