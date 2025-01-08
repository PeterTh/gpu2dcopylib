
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
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80"

. /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=cuda:*

# on GPUC5 (Intel)

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx 

. /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:*
export COPYLIB_ALLOC_CPU_IDS=2,3

# on Leonardo

module load cmake/3.27.7
module load cuda
module load ninja
# module load gcc/12.2.0 # NOO!! breaks the build in absolutely mysterious ways

. /leonardo_work/L-AUT_Thoman/intel-oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=cuda:*

cmake .. -G Ninja -DCOPYLIB_USE_MIMALLOC=false -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/leonardo_work/L-AUT_Thoman/intel-oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/leonardo_work/L-AUT_Thoman/intel-oneapi/compiler/latest/bin/icpx -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80"

srun -A L-AUT_Thoman --partition boost_usr_prod -n 1 --gres=gpu:2 --pty /usr/bin/bash
