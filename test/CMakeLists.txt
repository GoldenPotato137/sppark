project(sppark_test CUDA)

file(GLOB_RECURSE SOURCES "*.cu")

add_executable(sppark_test ${SOURCES})
target_link_libraries(sppark_test sppark)
