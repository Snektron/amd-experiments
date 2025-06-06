cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(gpu-experiments LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_HIP_STANDARD 23)

include(FetchContent)

FetchContent_Declare(
    rocm_smi
    GIT_REPOSITORY https://github.com/ROCm/rocm_smi_lib
    GIT_TAG rocm-6.4.1
    OVERRIDE_FIND_PACKAGE
)

find_package(rocm_smi REQUIRED)

function(add_experiment NAME SOURCES)
    add_executable(${NAME} ${SOURCES})
    target_link_libraries(${NAME} PRIVATE stdc++_libbacktrace rocm_smi64)
endfunction()

add_experiment(arithmetic arithmetic.hip)
add_experiment(atomic_local atomic_local.hip)
add_experiment(atomic_global atomic_global.hip)
add_experiment(memory memory.hip)
add_experiment(shuffle shuffle.hip)
