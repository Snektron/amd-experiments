cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(gpu-experiments LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_HIP_STANDARD 23)

add_executable(memory memory.hip)
target_link_libraries(memory stdc++_libbacktrace)
