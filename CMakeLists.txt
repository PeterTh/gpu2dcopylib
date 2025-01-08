cmake_minimum_required(VERSION 3.23.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(syclcopylib CXX)

option(COPYLIB_USE_MIMALLOC "Use the mimalloc memory allocator" ON)

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
      message("With CUDA Tooolkit ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_DIR})")
      set(COPYLIB_CUDA ON)
    endif()
  endif()
else()
  message("Using SimSYCL ${SimSYCL_VERSION} (${SimSYCL_DIR})")
endif()

if(COPYLIB_USE_MIMALLOC)
  Include(FetchContent)

  FetchContent_Declare(
    Mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG        v1.8.9 # or a later release
  )
  set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_SHARED ON CACHE BOOL "" FORCE)
  set(MI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(CMAKE_MESSAGE_LOG_LEVEL "WARNING")
  FetchContent_MakeAvailable(Mimalloc)
  set(CMAKE_MESSAGE_LOG_LEVEL "STATUS")
  message("Using MiMalloc")
endif()

add_subdirectory(lib)

add_subdirectory(tests)
add_subdirectory(utils)
add_subdirectory(benchmarks)
