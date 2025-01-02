cmake_minimum_required(VERSION 3.23.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(transfer_test CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

find_package(SimSYCL CONFIG QUIET)
if(NOT SimSYCL_FOUND)
  find_package(AdaptiveCpp CONFIG QUIET)
  if(NOT AdaptiveCpp_FOUND)
      find_package(IntelSYCL CONFIG QUIET)
      if(IntelSYCL_FOUND)
        message("Using Intel SYCL ${IntelSYCL_VERSION}  (${IntelSYCL_DIR})")
      else()
        message(FATAL_ERROR "No SYCL implementation (SimSYCL, AdaptiveCpp, or IntelSYCL) found in CMAKE_PREFIX_PATH")
      endif()
  else()
    message("Using AdaptiveCpp ${AdaptiveCpp_VERSION} (${AdaptiveCpp_DIR})")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      message("With CUDA Tooolkit ${CUDAToolkit_VERSION} (${CUDAToolkit_DIR})")
      set(COPYLIB_CUDA ON)
    endif()
  endif()
else()
  message("Using SimSYCL ${SimSYCL_VERSION} (${SimSYCL_DIR})")
endif()

SET(COPYLIB_SRC
    copylib_core.cpp
    copylib_backend.cpp
    copylib_support.cpp
    utils.cpp
)

add_library(copylib ${COPYLIB_SRC})
# Only the backend actually needs SYCL compilation, the others just need the libs/include path, but hard to make work consistently over different SYCL implementations
add_sycl_to_target(TARGET copylib SOURCES ${COPYLIB_SRC})
set_target_properties(copylib PROPERTIES CXX_STANDARD 20)
target_compile_options(copylib PRIVATE -Wall -Wextra -Wno-unknown-cuda-version)
target_include_directories(copylib PUBLIC .)
target_link_libraries(copylib PUBLIC ${SYCLLib})
if(COPYLIB_CUDA)
  target_link_libraries(copylib PUBLIC CUDA::cudart)
  target_compile_definitions(copylib PUBLIC COPYLIB_CUDA)
endif()

add_executable(transfer_bench transfer_bench.cpp)
add_sycl_to_target(TARGET transfer_bench SOURCES transfer_bench.cpp)
set_target_properties(transfer_bench PROPERTIES CXX_STANDARD 20)
target_link_libraries(transfer_bench PRIVATE copylib)

add_subdirectory(tests)
