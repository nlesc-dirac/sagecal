di 28 aug 2018  9:37:58 CEST
# SAGECal Installation

## Cmake Build

### Requirements
#### Ubuntu (tested with 16.04)
- Add KERN repository. Instructions can also be found at [http://kernsuite.info/](http://kernsuite.info/)
```
    sudo apt-get install software-properties-common
    sudo add-apt-repository -s ppa:kernsuite/kern-3
    sudo apt-add-repository multiverse
    sudo apt-get update
```

- Install following packages:
```
    sudo apt-get install -y git cmake g++ pkg-config libcfitsio-bin libcfitsio-dev libopenblas-base libopenblas-dev wcslib-dev wcslib-tools libglib2.0-dev libcasa-casa2 casacore-dev casacore-data casacore-tools
```
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
    cmake .. -DENABLE_CUDA=OFF
```

**OPTIONAL:** You can also define a custom casacore path:

```
    cmake .. -DCASACORE_ROOT_DIR=/opt/soft/casacore/ -DENABLE_CUDA=OFF
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



### Via Anaconda (WIP)
```
    conda install -c sagecal=0.6.0
```



## Manual installation
For expert users, and for custom architectures (GPU), the manual install is recommended.
### 1 Prerequisites:
 - CASACORE http://casacore.googlecode.com/
 - glib http://developer.gnome.org/glib
 - BLAS/LAPACK
   Highly recommended is OpenBLAS http://www.openblas.net/
   Also, to avoid any linking issues (and to get best performance), build OpenBLAS from source and link SAGECal with the static library (libopenblas***.a) and NOT libopenblas***.so
 - Compilers gcc/g++ or Intel icc/icpc 
 - If you have NVIDIA GPUs, 
  -- CUDA/CUBLAS/CUSOLVER and nvcc
  -- NVML Nvidia management library
 - If you are using Intel Xeon Phi MICs.
  -- Intel MKL and other libraries
 - Get the source for SAGECal 
```
    git clone -b master https://git@github.com/nlesc-dirac/sagecal.git
```

### 2 The basic way to build is
  1.a) go to ./src/lib/Dirac and ./src/lib/Radio  and run make (which will create libdirac.a and libradio.a)
  1.b) go to ./src/MS and run make (which will create the executable)


### 3 Build settings
In ./src/lib and ./src/MS you MUST edit the Makefiles to suit your system. Some common items to edit are:
 - LAPACK: directory where LAPACK/OpenBLAS is installed
 - GLIBI/GLIBL: include/lib files for glib
 - CASA_LIBDIR/CASA_INCDIR/CASA_LIBS : casacore include/library location and files:
  Note with new CASACORE might need two include paths, e.g.
    -I/opt/casacore/include/ -I/opt/casacore/include/casacore
 - CUDAINC/CUDALIB : where CUDA/CUBLAS/CUSOLVER is installed
 - NVML_INC/NVML_LIB : NVML include/lib path
 - NVCFLAGS : flags to pass to nvcc, especially -arch option to match your GPU  
 - MKLROOT : for Intel MKL

 Example makefiles: 
   Makefile : plain build
   Makefile.gpu: with GPU support
   Note: Edit ./lib/Radio/Radio.h MAX_GPU_ID to match the number of available GPUs, e.g., for 2 GPUs, MAX_GPU_ID=1



## SAGECAL-MPI Manual Installation 
This is for manually installing the distributed version of sagecal (sagecal-mpi), the cmake build will will work for most cases.
## 1 Prerequsites:
 - Same as for SAGECal.
 - MPI (e.g. OpenMPI)

## 2 Build ./src/lib as above (using mpicc -DMPI_BUILD)

## 3 Build ./src/MPI using mpicc++



## BUILDSKY Installation

  - See INSTALL in ./src/buildsky


## RESTORE Installation

  - See INSTALL in ./src/restore
  
  

