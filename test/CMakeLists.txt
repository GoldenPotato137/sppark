project(sppark_test CUDA)

file(GLOB_RECURSE SOURCES "*.cu")

add_executable(sppark_test ${SOURCES})
#target_compile_options(sppark_test PRIVATE -Wno-subobject-linkage)
target_compile_definitions(sppark_test PRIVATE _IOSTREAM_)
target_link_libraries(sppark_test sppark)
