# GPU 2D Copy Lib

This library intends to provide a simple very high performance way to copy strided data with a 2D layout between GPU devices, and between CPU and GPU memory spaces.

It is implemented in C++ and SYCL, and provides a C++ API.

## Prerequisites

- A SYCL implementation
    - The library has been tested with SimSYCL (in CI), DPC++, and AdaptiveCpp
- CMake 3.23.5 or later
- [optional] CUDA Toolkit for CUDA-backend-specific interop features

## Building

This library uses CMake as its build system.  
To build, run the following commands, adjusted for your environment:

```bash
mkdir build
cd build
cmkae .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/sycl
ninja
```

The CMake script will report which SYCL implementation it has found and is using.

### Configuration Options

- `COPYLIB_USE_MIMALLOC`: Use MiMalloc as the memory allocator. Default: `ON`
- `COPYLIB_USE_FMT`: Use the FMT library instead of relying on the C++20 `std::format`. Default: `ON`

Both dependencies are fetched automatically if their use is enabled.

## Usage

To use the library for copy operations, first, an instance of an executor must be created.  
Then, copy operations can be specified, turned into optimized parallel copy sets, and executed.

Here is a complete example which manually specifies all parameters to serve as a reference:

```cpp
#include <copylib/copylib.hpp>

using namespace copylib;

// === 1. Initialization

const int64_t buffer_size = 128 * 1024 * 1024; // 128 MiB for staging buffers
const int64_t queues_per_device = 2; // number of in-order queues per device for asynchronicity
executor exec(buffer_size, 2, queues_per_device); // create an executor
utils::print(exec.get_info()); // [optional] print information about the executution environment

// === 2. Specifying a copy operation

// provide these pointers as appropriate to source and target memory
intptr_t src_ptr = 0x1; // pointer to the source
intptr_t dst_ptr = 0x2; // pointer to the destination

// all the following values are in bytes
const int64_t offset = 0; // offset of the data to copy from the start of the buffer
const int64_t length = 1024; // length of one fragment (row) of data to copy
const int64_t count = 1024; // number of fragments (rows) to copy
const int64_t stride = 1024*1024; // stride between the start of two consecutive fragments (rows)

const data_layout source_layout{src_ptr, offset, length, count, stride}; // source data layout
const data_layout target_layout{dst_ptr, source_layout}; // target data layout, same structure as the source

// copy from device 0 to device 1
const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec); // [optional] check if the copy spec is valid

// === 3. Manifesting the copy operations into an optimized parallel copy set

const copy_type type = copy_type::staged; // perform linearization
const copy_properties props = copy_properties::use_kernel; // use a kernel for linearization, generally faster
const d2d_implementation d2d = d2d_implementation::host_staging_at_source; // use manual host staging
const int64_t chunk_size = 1024*1024; // generate 1 MiB chunks
const strategy strat(type, props, d2d, chunk_size); // create a strategy
const auto copy_set = manifest_strategy(spec, strat, basic_staging_provider{}); // manifest the copy set
COPYLIB_ENSURE(is_valid(copy_set), "Invalid copy set: {}", copy_set); // [optional] validate the copy set

// === 4. Executing the copy set

execute_copy(exec, copy_set);
```

## Benchmarks and Utilities

Some benchmarks and utilities are provided:

- `utils/info`: Print information about the execution environment and its features
- `benchmarks/manifest`: Micro-benchmark measuring strategy manifesting performance
- `benchmarks/intra_device`: Benchmark for intra-device linearization performnce
- `benchmarks/chunk_parallel`: Benchmark for optimized device-to-device copy performance
- `benchmarks/full_set`: Perform a very large run of various benchmarks to characterize platform performance
