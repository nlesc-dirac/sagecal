di 30 apr 2024 13:00:36 CEST
# SAGECal Installation

## Cmake Build
#### Ubuntu 22.04 (quick install), also works mostly for 20.04
```
 sudo apt-get install -y git cmake g++ pkg-config libcfitsio-bin libcfitsio-dev libopenblas-base libopenblas-dev wcslib-dev wcslib-tools libglib2.0-dev libcasa-casa4 casacore-dev casacore-data casacore-tools gfortran libopenmpi-dev libfftw3-dev

```
Run cmake (with GPU support) for example like
```
 mkdir build && cd build
 cmake .. -DHAVE_CUDA=ON -DNUM_GPU=1 -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_C_COMPILER=gcc-8 -DCUDA_NVCC_FLAGS='-gencode arch=compute_75,code=sm_75' -DCMAKE_CUDA_ARCHITECTURES=75 -DBLA_VENDOR=OpenBLAS
```
where *-DNUM_GPU=1* is when there is only one GPU. If you have more GPUs, increase this number to 2,3, and so on. This will produce *sagecal_gpu* and *sagecal-mpi_gpu* binary files (after running *make* of course). Architecture of the GPU is specified in the *-DCUDA_NVCC_FLAGS* option, and in newer cmake, using *-DCMAKE_CUDA_ARCHITECTURES*. It is important to select the gcc and g++ compilers to match the CUDA version, above example uses *gcc-8* and *g++-8*.

CPU only version can be build as
```
 cmake .. -DBLA_VENDOR=OpenBLAS
```
which will produce *sagecal* and *sagecal-mpi*.

The option *-DBLA_VENDOR=OpenBLAS* is to select OpenBLAS explicitly, but other BLAS  flavours can also be given. If not specified, whatever BLAS installed will be used.


To only build *libdirac* (shared) library, use *-DLIB_ONLY=1* option (also *-DBLA_VENDOR* to select the BLAS flavour). This library can be used with pkg-config using *lib/pkgconfig/libdirac.pc*. To build *libdirac* with GPU support, use *-DHAVE_CUDA=ON* with *-DLIB_ONLY=1* and give *-fPIC* compiler flag (for both *-DCMAKE_CXX_FLAGS* and *-DCMAKE_C_FLAGS*). With GPU support, only a static library is built because it needs to match the GPU architecture.

### Vectorized math operations
SAGECal can use ***libmvec*** vectorized math operations, both in GPU and CPU versions. In order to enable this, use compiler options *-ffast-math -lmvec -lm* for both gcc and g++. Also *-mavx*, *-mavx2* etc. can be added. Here is an example for CPU version

```
cmake ..  -DCMAKE_CXX_FLAGS='-g -O3 -Wall -ffast-math -lmvec -lm -mavx2' -DCMAKE_C_FLAGS='-g -O3 -Wall -ffast-math -lmvec -lm -mavx2' 
```


### Linking with CSPICE
See [linking with CSPICE](https://github.com/nlesc-dirac/sagecal/blob/cspice/scripts/CSPICE/README.md).
