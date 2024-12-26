cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/installs/AdaptiveCpp/ -DACPP_TARGETS=cuda:sm_86

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=~/installs/simsycl/