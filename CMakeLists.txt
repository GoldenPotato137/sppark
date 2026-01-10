cmake_minimum_required(VERSION 3.19)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 87 89)

project(sppark CUDA C ASM)


include_directories(./ depends/blst/src)

set(LIB_BLST_SRC "depends/blst/src/server.c" "depends/blst/build/assembly.S")
set_source_files_properties(depends/blst/src/server.c PROPERTIES LANGUAGE C)
set_source_files_properties(depends/blst/build/assembly.S PROPERTIES LANGUAGE ASM)

add_subdirectory(test)

# to make intellisense happy
file(GLOB_RECURSE SOURCES "ec/*.hpp" "ec/*.cuh" "ff/*.hpp" "ff/*.cuh" "msm/*.hpp" "msm/*.cuh" "util/*.hpp" "util/*.cuh")
list(APPEND SOURCES "util/all_gpus.cu")

# Single-definition TU for device constants (avoid per-TU .const duplication).
# This must match the field header included by clients (see SPPARK_FF_ALT_VARIANT).
set(SPPARK_FF_ALT_VARIANT "2048" CACHE STRING "ALT curve variant for device constants: 1024, 2048, 4096")
set_property(CACHE SPPARK_FF_ALT_VARIANT PROPERTY STRINGS 1024 2048 4096)

if(SPPARK_FF_ALT_VARIANT STREQUAL "1024")
    list(APPEND SOURCES "ff/alt1024_constants.cu")
elseif(SPPARK_FF_ALT_VARIANT STREQUAL "2048")
    list(APPEND SOURCES "ff/alt2048_constants.cu")
elseif(SPPARK_FF_ALT_VARIANT STREQUAL "4096")
    list(APPEND SOURCES "ff/alt4096_constants.cu")
else()
    message(FATAL_ERROR "Unsupported SPPARK_FF_ALT_VARIANT='${SPPARK_FF_ALT_VARIANT}'. Expected 1024, 2048, or 4096.")
endif()

add_library(sppark ${SOURCES} ${LIB_BLST_SRC})
target_include_directories(sppark PUBLIC "depends/cutlass/include")

# Propagate the selected variant to dependents so they include matching headers.
target_compile_definitions(sppark PUBLIC SPPARK_FF_ALT_VARIANT=${SPPARK_FF_ALT_VARIANT})

set_target_properties(sppark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)