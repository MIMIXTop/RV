cmake_minimum_required(VERSION 4.1)
project(LABA_2 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

find_package(TBB REQUIRED)

add_executable(k_means_gpu k_means_gpu.cu)
add_executable(GetPointers getPointers.cpp)
add_executable(k_means_cpu k_means_cpu.cpp)

target_link_libraries(k_means_cpu TBB::tbb)

set_target_properties(k_means_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(k_means_cpu PROPERTIES CXX_STANDARD 23)
set_target_properties(k_means_gpu PROPERTIES CUDA_STANDARD 23)
