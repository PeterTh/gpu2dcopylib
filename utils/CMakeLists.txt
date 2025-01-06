
SET(EXECUTABLES
    info
)

foreach(EXECUTABLE ${EXECUTABLES})
    add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
    add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE}.cpp)
    set_target_properties(${EXECUTABLE} PROPERTIES CXX_STANDARD 20)
    target_link_libraries(${EXECUTABLE} PRIVATE copylib)
    target_compile_options(${EXECUTABLE} PRIVATE -Wall -Wextra -Wno-unknown-cuda-version)
endforeach()
