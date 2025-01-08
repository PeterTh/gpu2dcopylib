SET(COPYLIB_SRC
    copylib_core.cpp
    copylib_backend.cpp
    copylib_backend_kernels.cpp
    copylib_support.cpp
    utils.cpp
)

add_library(copylib ${COPYLIB_SRC})
# Only the backend actually needs SYCL compilation, the others just need the libs/include path, but hard to make work consistently over different SYCL implementations
add_sycl_to_target(TARGET copylib SOURCES ${COPYLIB_SRC})
set_target_properties(copylib PROPERTIES CXX_STANDARD 20)
target_compile_options(copylib PRIVATE -Wall -Wextra -Wno-unknown-cuda-version)
target_include_directories(copylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../vendor)
if(AdaptiveCpp_FOUND)
  target_compile_definitions(copylib PUBLIC ACPP_ALLOW_INSTANT_SUBMISSION=1) # massive performance improvement for fine-grained kernels
endif()
target_include_directories(copylib PUBLIC .)
target_link_libraries(copylib PUBLIC ${SYCLLib})
if(COPYLIB_CUDA)
  target_link_libraries(copylib PUBLIC CUDA::cudart)
  target_compile_definitions(copylib PUBLIC COPYLIB_CUDA)
endif()
if(COPYLIB_USE_MIMALLOC)
  target_link_libraries(copylib PUBLIC mimalloc)
  target_compile_definitions(copylib PUBLIC COPYLIB_USE_MIMALLOC)
endif()
if(COPYLIB_USE_FMT)
  target_link_libraries(copylib PUBLIC fmt::fmt)
  target_compile_definitions(copylib PUBLIC COPYLIB_USE_FMT)
endif()
