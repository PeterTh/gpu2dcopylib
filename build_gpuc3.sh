cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/installs/AdaptiveCpp/ -DACPP_TARGETS=cuda:sm_86

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=~/installs/simsycl/

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx -DCMAKE_CXX_FLAGS="-fsycl -fsyc
l-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80"

. /opt/intel/oneapi/setvars.sh
