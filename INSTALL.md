wo  4 apr 2018 13:36:14 CEST
# SAGECal Installation

## Cmake Build

### Requirements
#### Ubuntu (tested with 16.04)
- Add KERN repository. Intructions can also be found at [http://kernsuite.info/](http://kernsuite.info/)
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


### Building
- Clone the repository
```
    git clone -b sprint_cmake https://git@github.com/nlesc-dirac/sagecal.git

```

- Build SageCal
```
    mkdir build && cd build
    cmake .. -DENABLE_CUDA=OFF
```

**OPTIONAL:** You can also define a custon casacore path:

```
    cmake .. -DCASACORE_ROOT_DIR=/opt/soft/casacore/ -DENABLE_CUDA=OFF
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compile with:
```
    make -j4
```


- The sagecal executable can be found in
    **dist/bin** folder. All the libraries will be stored in **dist/lib** folder. 



### Via Anaconda (WIP)
```
    conda install -c sagecal=0.6.0
```



## Manual installation
### 1 Prerequsites:
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
 - Get the source for SAGECal : git clone git://git.code.sf.net/p/sagecal/code sagecal-code


### 2 The basic way to build is
  1.a) go to ./src/lib  and run make (which will create libsagecal.a)
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
   Note: Edit sagecal.h MAX_GPU_ID to match the number of available GPUs
   Makefile.MIC : with Intel Xeon Phi support




# SAGECAL-MPI Installation

## 1 Prerequsites:
 - Same as above 
 - MPI (e.g. OpenMPI)

## 2 Build ./src/lib as above (using mpicc -DMPI_BUILD)

## 3 Build ./src/MPI using mpicc++



## BUILDSKY Installation

  - See INSTALL in ./src/buildsky


## RESTORE Installation

  - See INSTALL in ./src/restore
  
  

