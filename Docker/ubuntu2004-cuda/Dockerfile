FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install software-properties-common -y && \
    apt-add-repository multiverse && \
    apt-add-repository restricted && \
    apt-get update -y

RUN apt-get install -y \
    git cmake g++-8 gcc-8 pkg-config \
    libcfitsio-bin libcfitsio-dev \
    liblapack-dev libopenblas-dev \
    wcslib-dev wcslib-tools \
    libglib2.0-dev libgfortran5 \
    libcasa-casa4 casacore-dev casacore-data casacore-tools \
    mpich \
    fftw3-dev libfftw3-mpi3 libfftw3-bin \
    nvidia-cuda-dev nvidia-cuda-toolkit \
    libcublas10 libcusolver10 libnvidia-ml-dev

# make symlink to gfortran
RUN ln -s /usr/lib/x86_64-linux-gnu/libgfortran.so.5 /usr/lib/x86_64-linux-gnu/libgfortran.so

## compile sagecal
# Note: First edit
# 1) 'arch=compute_75,code=sm_75' to match your GPU
# 2) -DNUM_GPU to match the number of GPUs to use (1 for one GPU, 2 for two etc.)
# The gcc/g++ compiler versions are to match cuda 10.1
RUN git clone --depth 1 --branch master \
     https://github.com/nlesc-dirac/sagecal.git && cd sagecal && \ 
     mkdir build-ubuntu && cd build-ubuntu && \
     cmake -DCMAKE_INSTALL_PREFIX=/opt/sagecal \
     -DHAVE_CUDA=ON -DCUDA_NVCC_FLAGS="-gencode arch=compute_75,code=sm_75 -O3" \
     -DNUM_GPU=2 -DBLA_VENDOR=OpenBLAS \
     -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_C_COMPILER=gcc-8 \
     -DCMAKE_CXX_FLAGS="-O3 -ffast-math -lmvec -lm" \
     -DCMAKE_C_FLAGS="-O3 -ffast-math -lmvec -lm" .. && \
     make -j4 && \
     make install
RUN ls -alsrt /opt/sagecal && \
     /opt/sagecal/bin/sagecal_gpu
