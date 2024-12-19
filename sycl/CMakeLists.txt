cmake_minimum_required(VERSION 3.23.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(transfer_test CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

find_package(AdaptiveCpp CONFIG)
if(NOT AdaptiveCpp_FOUND)
    find_package(IntelSYCL REQUIRED)
endif()

add_executable(transfer_test transfer_test.cpp)
add_sycl_to_target(TARGET transfer_test SOURCES transfer_test.cpp)
set_target_properties(transfer_test PROPERTIES CXX_STANDARD 20)
