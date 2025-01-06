Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

SET(TEST_FILES
    backend_tests.cpp
    core_tests.cpp
    support_tests.cpp
    utils_tests.cpp
)

if(AdaptiveCpp_FOUND)
    # ACPP gets some CUDA fat binary linker error with the individual test executables, so let's just not
    add_executable(all_tests ${TEST_FILES})
    add_sycl_to_target(TARGET all_tests SOURCES ${TEST_FILES})
    target_link_libraries(all_tests PRIVATE copylib Catch2::Catch2WithMain)
    target_compile_options(all_tests PRIVATE -Wall -Wextra -Wno-unknown-cuda-version)
    set_target_properties(all_tests PROPERTIES CXX_STANDARD 20)
else()
    # for each test file, build an object library, an executable, and add the library to the list of objects
    # so we can have individual test executables and a single all_tests executable without duplicate compilation
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_library(${TEST_NAME}_obj OBJECT ${TEST_FILE})
        add_sycl_to_target(TARGET ${TEST_NAME}_obj SOURCES ${TEST_FILE})
        set_target_properties(${TEST_NAME}_obj PROPERTIES CXX_STANDARD 20)
        target_link_libraries(${TEST_NAME}_obj PRIVATE copylib Catch2::Catch2)
        target_compile_options(${TEST_NAME}_obj PRIVATE -Wall -Wextra -Wno-unknown-cuda-version)
        add_executable(${TEST_NAME})
        target_link_libraries(${TEST_NAME} PRIVATE ${TEST_NAME}_obj Catch2::Catch2WithMain)
        list(APPEND OBJECTS ${TEST_NAME}_obj)
    endforeach()

    add_executable(all_tests)
    target_link_libraries(all_tests PRIVATE ${OBJECTS} Catch2::Catch2WithMain)
endif()
