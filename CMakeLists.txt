cmake_minimum_required(VERSION 3.19)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70 75)

project(sppark CUDA C ASM)


include_directories(./ depends/blst/src)

set(LIB_BLST_SRC "depends/blst/src/server.c" "depends/blst/build/assembly.S")
set_source_files_properties(depends/blst/src/server.c PROPERTIES LANGUAGE C)
set_source_files_properties(depends/blst/build/assembly.S PROPERTIES LANGUAGE ASM)

add_subdirectory(test)

# to make intellisense happy
file(GLOB_RECURSE SOURCES "ec/*.hpp" "ec/*.cuh" "ff/*.hpp" "ff/*.cuh" "msm/*.hpp" "msm/*.cuh" "util/*.hpp" "util/*.cuh")
list(APPEND SOURCES "util/all_gpus.cu")
add_library(sppark ${SOURCES} ${LIB_BLST_SRC})

set_target_properties(sppark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)