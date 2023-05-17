ma 27 mrt 2023 10:16:16 CEST
# SAGECal Installation

## Cmake Build
#### Ubuntu 20.04 (quick install)
```
 sudo apt-get install -y git cmake g++ pkg-config libcfitsio-bin libcfitsio-dev libopenblas-base libopenblas-dev wcslib-dev wcslib-tools libglib2.0-dev libcasa-casa4 casacore-dev casacore-data casacore-tools gfortran libopenmpi-dev libfftw3-dev

```
Run cmake (with GPU support) for example like
```
 mkdir build && cd build
 cmake .. -DHAVE_CUDA=ON -DCMAKE_CXX_FLAGS='-DMAX_GPU_ID=0' -DCMAKE_CXX_COMPILER=g++-8  -DCMAKE_C_FLAGS='-DMAX_GPU_ID=0' -DCMAKE_C_COMPILER=gcc-8 -DCUDA_NVCC_FLAGS='-gencode arch=compute_75,code=sm_75' -DBLA_VENDOR=OpenBLAS
```
where *MAX_GPU_ID=0* is when there is only one GPU (ordinal 0). If you have more GPUs, increase this number to 1,2, and so on. This will produce *sagecal_gpu* and *sagecal-mpi_gpu* binary files (after running *make* of course). You can also use *-DNUM_GPU* to specify the number of GPUs to use, for example *-DNUM_GPU=4*.

CPU only version can be build as
```
 cmake .. -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_C_COMPILER=gcc-8 -DBLA_VENDOR=OpenBLAS
```
which will produce *sagecal* and *sagecal-mpi*.

The option *-DBLA_VENDOR=OpenBLAS* is to select OpenBLAS explicitly, but other BLAS  flavours can also be given. If not specified, whatever BLAS installed will be used.

If you get **-lgfortran is not found** error, run the following in the build directory
```
 cd dist/lib
 ln -s /usr/lib/x86_64-linux-gnu/libgfortran.so.5 libgfortran.so
```
to make a symbolic link to libgfortran.so.5 or whatever version that is installed.

To only build *libdirac* (shared) library, use *-DLIB_ONLY=1* option (also *-DBLA_VENDOR* to select the BLAS flavour). This library can be used with pkg-config using *lib/pkgconfig/libdirac.pc*. To build *libdirac* with GPU support, use *-DHAVE_CUDA=ON* with *-DLIB_ONLY=1* and give *-fPIC* compiler flag (for both *-DCMAKE_CXX_FLAGS* and *-DCMAKE_C_FLAGS*). With GPU support, only a static library is built.

### Vectorized math operations (New)
SAGECal can use ***libmvec*** vectorized math operations, both in GPU and CPU versions. In order to enable this, use compiler options *-fopenmp -ffast-math -lmvec -lm* for both gcc and g++. Also *-mavx*, *-mavx2* etc. can be added. Here is an example for CPU version

```
cmake ..  -DCMAKE_CXX_FLAGS='-g -O3 -Wall -fopenmp -ffast-math -lmvec -lm -mavx2' -DCMAKE_C_FLAGS='-g -O3 -Wall -fopenmp -ffast-math -lmvec -lm -mavx2' 
```

### Requirements for older installations
checkout the source code and compile it with the instructions below(in source folder):
```
git clone https://github.com/nlesc-dirac/sagecal.git

cd sagecal && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH
make
make install
```
$INSTALL_PATH is where you want to install SageCal.

#### Other systems

- Install equivalent packages for your distribution
    - g++
    - cmake
    - git
    - pkg-config
    - openblas
    - libglib2.0-dev
    - follow the instructions at
[https://github.com/casacore/casacore](https://github.com/casacore/casacore) to install casacore.
    - Additional packages (not essential, but recommended): MPI (openmpi), FFTW



### Building
- Clone the repository
```
    git clone -b master https://git@github.com/nlesc-dirac/sagecal.git

```

- Build SAGECal
```
    mkdir build && cd build
    cmake ..
```

**OPTIONAL:** You can also define a custom casacore path:

```
    cmake .. -DCASACORE_ROOT_DIR=/opt/soft/casacore
```
**OPTIONAL:** You can also define a custom paths to everything:

```
    cmake -DCFITSIO_ROOT_DIR=/cm/shared/package/cfitsio/3380-gcc-4.9.3 -DCASACORE_ROOT_DIR=/cm/shared/package/casacore/v2.3.0-gcc-4.9.3 -DWCSLIB_INCLUDE_DIR=/cm/shared/package/wcslib/5.13-gcc-4.9.3/include -DWCSLIB_LIBRARY=/cm/shared/package/wcslib/5.13-gcc-4.9.3/lib/libwcs.so -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_LINKER=/cm/shared/package/gcc/4.9.3/bin/gcc -DCMAKE_CXX_FLAGS=-L/cm/shared/package/cfitsio/3380-gcc-4.9.3/lib -DCMAKE_C_FLAGS=-L/cm/shared/package/cfitsio/3380-gcc-4.9.3/lib ..
```

    Compile with:
```
    make
```
    Install at your favorite place
```
    make DEST=/path/to/sagecal/dir install
```

- The sagecal executable can be found in **/path/to/sagecal/dir/usr/local/bin**, also **sagecal-mpi**,**buildsky** and **restore** might be installed depending on the availability of MPI and WCSLIB/FFTW.

### MPI support
MPI support is automatically detected, otherwise, it can be forced with:
```
cmake -DENABLE_MPI=ON
