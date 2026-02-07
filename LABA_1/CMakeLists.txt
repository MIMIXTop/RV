cmake_minimum_required(VERSION 3.21)

project( LABA_1 )

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TBB REQUIRED)

add_executable( GetPointers getPointers.cpp )
add_executable(k_means kmeans.cpp)

target_link_libraries(k_means PRIVATE TBB::tbb)


