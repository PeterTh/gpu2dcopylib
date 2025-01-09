
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
cmake .. -G Ninja -DCOPYLIB_USE_MIMALLOC=false -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80"

. /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=cuda:*

# on GPUC5 (Intel)

cmake .. -G Ninja -DCOPYLIB_USE_MIMALLOC=false -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx 

. /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:*
export COPYLIB_ALLOC_CPU_IDS=2,3

# ACPP?
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/installs/AdaptiveCpp-gpuc5/ -G Ninja
export ACPP_VISIBILITY_MASK=ze
export COPYLIB_ALLOC_CPU_IDS=2,3

# on Leonardo #####################################################################################################################

module load cmake/3.27.7
module load cuda
module load ninja
# module load gcc/12.2.0 # NOO!! breaks the build in absolutely mysterious ways; do this AFTER CMake has run....

. /leonardo_work/L-AUT_Thoman/intel-oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=cuda:*

cmake .. -G Ninja -DCOPYLIB_USE_MIMALLOC=false -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/leonardo_work/L-AUT_Thoman/intel-oneapi/compiler/latest/ -DCMAKE_CXX_COMPILER=/leonardo_work/L-AUT_Thoman/intel-oneapi/compiler/latest/bin/icpx -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -fPIC -fuse-ld=/leonardo_work/L-AUT_Thoman/intel-oneapi/compiler/latest/bin/compiler/ld.lld -lpthread"

srun -A L-AUT_Thoman --partition boost_usr_prod -n 1 --gres=gpu:2 --pty /usr/bin/bash


# Leonardo ACPP ################

## boost

./bootstrap.sh --with-libraries=fiber,context,test --prefix=/leonardo_work/L-AUT_Thoman/boost-1.87.0
./b2 install

## ACPP

# strange CMAKE issue (finds context but not fiber) without the many additional boost incantations
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_INCLUDEDIR=/leonardo_work/L-AUT_Thoman/pthoman0/boost/boost_1_87_0 -DBOOST_LIBRARYDIR=/leonardo_work/L-AUT_Thoman/pthoman0/boost/boost_1_87_0/stage/lib -DBOOST_ROOT=/leonardo_work/L-AUT_Thoman/boost-1.87.0 -DWITH_CUDA_BACKEND=1 -DWITH_CPU_BACKEND=1 -DWITH_OPENCL_BACKEND=0 -DWITH_SSCP_COMPILER=0 -DCMAKE_INSTALL_PREFIX=/leonardo_work/L-AUT_Thoman/acpp-cuda

# nvcxx
module load nvhpc/24.3
cmake .. -G Ninja -DDEFAULT_TARGETS=cuda:sm_80 -DACPP_COMPILER_FEATURE_PROFILE=none -DCMAKE_BUILD_TYPE=Release -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_INCLUDEDIR=/leonardo_work/L-AUT_Thoman/pthoman0/boost/boost_1_87_0 -DBOOST_LIBRARYDIR=/leonardo_work/L-AUT_Thoman/pthoman0/boost/boost_1_87_0/stage/lib -DBOOST_ROOT=/leonardo_work/L-AUT_Thoman/boost-1.87.0 -DNVCXX_COMPILER=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvc++ -DCMAKE_INSTALL_PREFIX=/leonardo_work/L-AUT_Thoman/acpp-cuda

# spack
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DDEFAULT_TARGETS=cuda:sm_80 -DCMAKE_INSTALL_PREFIX=/leonardo_work/L-AUT_Thoman/acpp-cuda
ninja install

## copylib
cmake .. -G Ninja -DCOPYLIB_USE_MIMALLOC=false -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/leonardo_work/L-AUT_Thoman/acpp-cuda -DCMAKE_CXX_FLAGS="-fuse-ld=lld -lpthread"
