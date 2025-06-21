cmake_minimum_required(VERSION 3.22)
project(collatz_hpx LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)    # adjust to your GPU

# Point CMake at your HPX installation
find_package(HPX REQUIRED COMPONENTS)

set(SRCS
    collatz_hpx.cpp
    kernel.cu
)

set_source_files_properties(kernel.cu PROPERTIES LANGUAGE CUDA)

add_hpx_executable(
    collatz
    SOURCES             ${SRCS}
    COMPONENT_DEPENDENCIES partitioned_vector iostreams # <-- tells HPX to link it
)