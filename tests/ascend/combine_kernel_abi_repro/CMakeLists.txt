cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)
project(deep_ep_ascend_combine_kernel_abi_repro LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(DEEP_EP_REPRO_BAD_ORDER "Compile the pre-fix stage/tiling order" OFF)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH is required")
endif()

add_executable(combine_kernel_abi_repro combine_kernel_abi_repro.asc
    combine_kernel_abi_repro_main.cpp)
set_target_properties(combine_kernel_abi_repro PROPERTIES LINKER_LANGUAGE ASC)
target_compile_options(combine_kernel_abi_repro PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_compile_definitions(combine_kernel_abi_repro PRIVATE
    $<$<BOOL:${DEEP_EP_REPRO_BAD_ORDER}>:DEEP_EP_REPRO_BAD_ORDER>)
target_include_directories(combine_kernel_abi_repro PRIVATE
    ${CMAKE_CURRENT_LIST_DIR} ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/x86_64-linux/asc/include)
target_link_directories(combine_kernel_abi_repro PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/x86_64-linux/lib64)
target_link_libraries(combine_kernel_abi_repro PRIVATE ascendcl c_sec)
