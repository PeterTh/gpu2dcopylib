
# on GPUC3 (NV)

export CUDA_VISIBLE_DEVICES=2,3
export COPYLIB_ALLOC_CPU_IDS=64,64

# dev
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=~/installs/simsycl/

# ACPP
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/installs/AdaptiveCpp/ -DACPP_TARGETS=cuda:sm_86

# ACPP generic
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/installs/AdaptiveCpp-generic/ -DACPP_TARGETS=generic -DCMAKE_CXX_COMPILER=clang++-18

# DPCPP NV
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx -DCMAKE_CXX_FLAGS="-fsycl -fsyc
l-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80"

. /opt/intel/oneapi/setvars.sh
env ONEAPI_DEVICE_SELECTOR=cuda:*

# on GPUC5 (Intel)

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx 

. /opt/intel/oneapi/setvars.sh
env ONEAPI_DEVICE_SELECTOR=level_zero:*

export COPYLIB_ALLOC_CPU_IDS=2,3
