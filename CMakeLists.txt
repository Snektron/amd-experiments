cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(gpu-experiments LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_HIP_STANDARD 23)

include(FetchContent)

find_package(amd_smi REQUIRED)
find_package(hsa-runtime64 REQUIRED)

function(add_experiment NAME SOURCES)
    add_executable(${NAME} ${SOURCES})
    target_link_libraries(${NAME} PRIVATE stdc++_libbacktrace amd_smi hsa-runtime64)
endfunction()

add_experiment(arithmetic arithmetic.hip)
add_experiment(atomic_local atomic_local.hip)
add_experiment(atomic_global atomic_global.hip)
add_experiment(cache_coalescing cache_coalescing.hip)
add_experiment(memory memory.hip)
add_experiment(mma mma.hip)
add_experiment(shuffle shuffle.hip)
