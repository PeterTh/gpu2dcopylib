#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call_)                                                                                                                                    \
        {                                                                                                                                                          \
                cudaError_t status = call_;                                                                                                                  \
                if(status != cudaSuccess) {                                                                                                                            \
                        fprintf(stderr, "CUDA Error in call %s on line %d: %s\n", #call_, __LINE__, cudaGetErrorString(status));                                              \
                        abort();                                                                                                                                           \
                }                                                                                                                                                      \
        }

struct Metric {
    double min = std::numeric_limits<double>::max();
    double max = 0;
    double avg = 0;

    void update(double value) {
        min = std::min(min, value);
        max = std::max(max, value);
        avg += value;
    }
};

struct CombinedMetric {
    Metric individual;
    Metric concurrent;
};

struct Device {
    int id;
    cudaStream_t stream;
    int *d_data;
    int *h_data;
    int *d_staging;
    int *h_staging;
    CombinedMetric d2h_bandwidth;
    CombinedMetric d2h_bw_strided;
    CombinedMetric d2h_bw_d_strided;
    CombinedMetric d2h_bw_h_strided;
    CombinedMetric d2h_bw_strided_kernel;
    CombinedMetric h2d_bandwidth;
    CombinedMetric h2d_bw_strided;
    CombinedMetric h2d_bw_d_strided;
    CombinedMetric h2d_bw_h_strided;
};

// kernel to gather strided date into contiguous memory
__global__ void linearize(int *src, int *dst, size_t stride, size_t size) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < size) {
        dst[idx] = src[idx * stride];
    }
}

void host_linearize(int *src, int *dst, size_t stride, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i * stride];
    }
}

// kernel to scatter contiguous data into strided memory
__global__ void delinearize(int *src, int *dst, size_t stride, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx * stride] = src[idx];
    }
}

void host_delinearize(int *src, int *dst, size_t stride, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i * stride] = src[i];
    }
}

int main(int argc, char **argv) {
    int repeats = 10;
    const int warmups = 3;
    if (argc > 1) {
        repeats = atol(argv[1]);
    }
    size_t memStride = 4096;
    if (argc > 2) {
        memStride = atol(argv[2]);
    }
    size_t transferSize = 1024 * 1024 * sizeof(int);
    if (argc > 3) {
        transferSize = atol(argv[3]);
    }
    std::cout << "Repeats: " << repeats << ", Stride: " << memStride << ", Transfer Size: " << transferSize << std::endl;

    using namespace std::chrono_literals;

    // init CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;

    // print device info
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }

    const size_t bytesTotal = memStride * transferSize;
    std::cout << "Allocating " << (bytesTotal) / (1024 * 1024) << " MB of memory per device as main buffer, and " << (transferSize) / (1024 * 1024) << " MB as staging buffer" << std::endl;

    // initialize device structs
    std::vector<Device> devices(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        devices[i].id = i;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&devices[i].stream));
        CUDA_CHECK(cudaMalloc(&devices[i].d_data, bytesTotal));
        CUDA_CHECK(cudaMallocHost(&devices[i].h_data, bytesTotal));
        CUDA_CHECK(cudaMalloc(&devices[i].d_staging, transferSize));
        CUDA_CHECK(cudaMallocHost(&devices[i].h_staging, transferSize));
    }

    const auto measureIndividual = [&](auto operation, auto metric) {
        for (int r = 0; r < repeats+warmups; ++r) {
            for (int i = 0; i < deviceCount; i++) {
                CUDA_CHECK(cudaStreamSynchronize(devices[i].stream));
                const auto start = std::chrono::high_resolution_clock::now();
                operation(i);
                CUDA_CHECK(cudaStreamSynchronize(devices[i].stream));
                const auto end = std::chrono::high_resolution_clock::now();
                const auto elapsed = end - start;
                if(r>=warmups) {
                    double current_bw = (double)transferSize / (1024 * 1024) / (elapsed / 1.0s);
                    metric(i).individual.update(current_bw);
                }
            }
        }
    };
    
    const auto measureConcurrent = [&](auto operation, auto metric) {
        for (int r = 0; r < repeats+warmups; ++r) {
            for (int i = 0; i < deviceCount; i++) {
                CUDA_CHECK(cudaStreamSynchronize(devices[i].stream));
            }
            const auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < deviceCount; i++) {
                operation(i);
            }
            for (int i = 0; i < deviceCount; i++) {
                CUDA_CHECK(cudaStreamSynchronize(devices[i].stream));
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const auto elapsed = end - start;
            if(r>=warmups) {
                double current_bw = (double)transferSize / (1024 * 1024) / (elapsed / 1.0s);
                for (int i = 0; i < deviceCount; i++) {
                    metric(i).concurrent.update(current_bw);
                }
            }
        }
    };

    auto measureBoth = [&](auto operation, auto metric) {
        measureIndividual(operation, metric);
        measureConcurrent(operation, metric);
    };

    // measure device to host bandwidth with contiguous access
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpyAsync(devices[i].h_data, devices[i].d_data, transferSize, cudaMemcpyDeviceToHost, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].d2h_bandwidth;
    });

    // measure device to host bandwidth with strided access on both
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpy2DAsync(devices[i].h_data, memStride, devices[i].d_data, memStride, 
            sizeof(int), transferSize/sizeof(int), cudaMemcpyDeviceToHost, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].d2h_bw_strided;
    });

    // measure device to host bandwidth with strided access on device
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpy2DAsync(devices[i].h_data, sizeof(int), devices[i].d_data, memStride, 
            sizeof(int), transferSize/sizeof(int), cudaMemcpyDeviceToHost, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].d2h_bw_d_strided;
    });

    // measure device to host bandwidth with strided access on host
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpy2DAsync(devices[i].h_data, memStride, devices[i].d_data, sizeof(int), 
            sizeof(int), transferSize/sizeof(int), cudaMemcpyDeviceToHost, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].d2h_bw_h_strided;
    });

    // measure device to host bandwidth with strided access using kernel linearize / delinearize
    measureBoth([&](int i) {
        linearize<<<(transferSize/sizeof(int) + 255) / 256, 256, 0, devices[i].stream>>>(devices[i].d_data, devices[i].d_staging, memStride/sizeof(int), transferSize/sizeof(int));
        CUDA_CHECK(cudaMemcpyAsync(devices[i].h_staging, devices[i].d_staging, transferSize, cudaMemcpyDeviceToHost, devices[i].stream));
        cudaStreamSynchronize(devices[i].stream);
        host_delinearize(devices[i].h_staging, devices[i].h_data, memStride/sizeof(int), transferSize/sizeof(int));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].d2h_bw_strided_kernel;
    });

    // measure host to device bandwidth with contiguous access
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpyAsync(devices[i].d_data, devices[i].h_data, transferSize, cudaMemcpyHostToDevice, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].h2d_bandwidth;
    });

    // measure host to device bandwidth with strided access on both
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpy2DAsync(devices[i].d_data, memStride, devices[i].h_data, memStride, 
            sizeof(int), transferSize/sizeof(int), cudaMemcpyHostToDevice, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].h2d_bw_strided;
    });

    // measure host to device bandwidth with strided access on device
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpy2DAsync(devices[i].d_data, memStride, devices[i].h_data, sizeof(int), 
            sizeof(int), transferSize/sizeof(int), cudaMemcpyHostToDevice, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].h2d_bw_d_strided;
    });

    // measure host to device bandwidth with strided access on host
    measureBoth([&](int i) {
        CUDA_CHECK(cudaMemcpy2DAsync(devices[i].d_data, sizeof(int), devices[i].h_data, memStride, 
            sizeof(int), transferSize/sizeof(int), cudaMemcpyHostToDevice, devices[i].stream));
    }, [&](int i) -> CombinedMetric& {
        return devices[i].h2d_bw_h_strided;
    });

    std::cout << std::fixed << std::setprecision(1);
    const auto printMetric = [&](const char* name, auto metric) {
        for(int i=0; i<deviceCount; i++) {
            std::cout << std::setw(24) << name << " -   Device " << i << " " << ": " << std::setw(9) << metric(i).individual.min << " MB/s (min), " << std::setw(9) << metric(i).individual.max << " MB/s (max), " << std::setw(9) << metric(i).individual.avg / repeats << " MB/s (avg)" << std::endl;
        }
        std::cout << std::setw(24) << name << " - Concurrent " << ": " << std::setw(9) << metric(0).concurrent.min << " MB/s (min), " << std::setw(9) << metric(0).concurrent.max << " MB/s (max), " << std::setw(9) << metric(0).concurrent.avg / repeats << " MB/s (avg)" << std::endl;
    };

    printMetric("D2H Contiguous", [&](int i) -> const CombinedMetric& {
        return devices[i].d2h_bandwidth;
    });
    printMetric("D2H Strided Both", [&](int i) -> const CombinedMetric& {
        return devices[i].d2h_bw_strided;
    });
    printMetric("D2H Strided Device", [&](int i) -> const CombinedMetric& {
        return devices[i].d2h_bw_d_strided;
    });
    printMetric("D2H Strided Host", [&](int i) -> const CombinedMetric& {
        return devices[i].d2h_bw_h_strided;
    });
    printMetric("D2H Strided w/ Kernel", [&](int i) -> const CombinedMetric& {
        return devices[i].d2h_bw_strided_kernel;
    });
    printMetric("H2D Contiguous", [&](int i) -> const CombinedMetric& {
        return devices[i].h2d_bandwidth;
    });
    printMetric("H2D Strided Both", [&](int i) -> const CombinedMetric& {
        return devices[i].h2d_bw_strided;
    });
    printMetric("H2D Strided Device", [&](int i) -> const CombinedMetric& {
        return devices[i].h2d_bw_d_strided;
    });
    printMetric("H2D Strided Host", [&](int i) -> const CombinedMetric& {
        return devices[i].h2d_bw_h_strided;
    });

    // free device structs
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaFree(devices[i].d_data);
        cudaFreeHost(devices[i].h_data);
        cudaStreamDestroy(devices[i].stream);
    }
}
