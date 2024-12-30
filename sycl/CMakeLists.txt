cmake_minimum_required(VERSION 3.23.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(transfer_test CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

find_package(SimSYCL CONFIG)
if(NOT SimSYCL_FOUND)
  find_package(AdaptiveCpp CONFIG)
  if(NOT AdaptiveCpp_FOUND)
      find_package(IntelSYCL REQUIRED)
      message("Using Intel SYCL ${IntelSYCL_VERSION}")
  else()
    message("Using AdaptiveCpp ${AdaptiveCpp_VERSION}")
  endif()
else()
  message("Using SimSYCL ${SimSYCL_VERSION}")
endif()

add_library(copylib copylib_core.cpp copylib_backend.cpp copylib_support.cpp utils.cpp)
add_sycl_to_target(TARGET copylib SOURCES copylib_core.cpp copylib_support.cpp copylib_backend.cpp)
set_target_properties(copylib PROPERTIES CXX_STANDARD 20)
target_include_directories(copylib PUBLIC .)

add_executable(transfer_bench transfer_bench.cpp)
add_sycl_to_target(TARGET transfer_bench SOURCES transfer_bench.cpp)
set_target_properties(transfer_bench PROPERTIES CXX_STANDARD 20)
target_link_libraries(transfer_bench PRIVATE copylib)

add_subdirectory(tests)
