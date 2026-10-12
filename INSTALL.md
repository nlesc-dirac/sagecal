# SAGECal Installation

## Cmake Build
#### Ubuntu 24.04 (quick install); also works for 22.04 and 20.04

The package names can vary slightly across Ubuntu releases. On recent Ubuntu versions, the Casacore package is typically `libcasa-casa6`.

```
 sudo apt-get install -y git cmake g++ pkg-config \
     libcfitsio-bin libcfitsio-dev libopenblas-base libopenblas-dev \
     wcslib-dev wcslib-tools libglib2.0-dev libcasa-casa6 casacore-dev casacore-data
```

If your Ubuntu release still ships an older CasaCore package name (for example `libcasa-casa4` on older Debian/Ubuntu systems), replace `libcasa-casa6` with the matching package name available on your system.

Run cmake (with GPU support, CUDA version => 12.8) for example like
```
 mkdir build && cd build
 cmake .. -DHAVE_CUDA=ON -DNUM_GPU=1 -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_C_COMPILER=gcc-8 -DCUDA_NVCC_FLAGS='-gencode arch=compute_75,code=sm_75' -DCMAKE_CUDA_ARCHITECTURES=75 -DBLA_VENDOR=OpenBLAS
```
where *-DNUM_GPU=1* is when there is only one GPU. If you have more GPUs, increase this number to 2,3, and so on. This will produce *sagecal_gpu* and *sagecal-mpi_gpu* binary files (after running *make*).

CPU only version can be build as
```
 cmake .. -DBLA_VENDOR=OpenBLAS
```
which will produce *sagecal* and *sagecal-mpi*.

The option *-DBLA_VENDOR=OpenBLAS* is to select OpenBLAS explicitly, but other BLAS flavours can also be given. If not specified, whatever BLAS installed will be used.

To only build *libdirac* (shared) library, use *-DLIB_ONLY=1* option (also *-DBLA_VENDOR* to select the BLAS flavour). This library can be used with pkg-config using *lib/pkgconfig/libdirac.pc*. To build this library, see the CMake options below.

### Vectorized math operations
SAGECal can use ***libmvec*** vectorized math operations, both in GPU and CPU versions. In order to enable this, use compiler options *-ffast-math -lmvec -lm* for both gcc and g++. Also *-mavx*, *-mavx2* and similar options may be useful depending on the target CPU.

```
cmake ..  -DCMAKE_CXX_FLAGS='-g -O3 -Wall -ffast-math -lmvec -lm -mavx2' -DCMAKE_C_FLAGS='-g -O3 -Wall -ffast-math -lmvec -lm -mavx2' 
```

### Linking with CSPICE
For lunar coordinate support, you need to first download and build CSPICE, for more information, see [linking with CSPICE](/scripts/CSPICE/README.md).
