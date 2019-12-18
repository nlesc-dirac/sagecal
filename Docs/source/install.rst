Installation Instructions
=========================

DAS5_
-----

.. _DAS5: https://www.cs.vu.nl/das5/ASTRON.shtml 

Load the modules below before compiling SageCal.

::

   module load cmake/3.8.2
   module load mpich/ge/gcc/64/3.2
   module load gcc/4.9.3
   module load casacore/2.3.0-gcc-4.9.3
   module load wcslib/5.13-gcc-4.9.3
   module load wcslib/5.16-gcc-4.9.3
   module load cfitsio/3.410-gcc-4.9.3

checkout the source code and compile it with the instructions below(in
source folder):

::

   git clone https://github.com/nlesc-dirac/sagecal.git

   cd sagecal && mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH
   make
   make install

$INSTALL_PATH is where you want to install SageCal.

Cmake Build
-----------

Requirements
~~~~~~~~~~~~

Ubuntu (tested with 16.04)
^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Add KERN repository. Instructions can also be found at
   http://kernsuite.info/

::

       sudo apt-get install software-properties-common
       sudo add-apt-repository -s ppa:kernsuite/kern-3
       sudo apt-add-repository multiverse
       sudo apt-get update

-  Install following packages:

::

       sudo apt-get install -y git cmake g++ pkg-config libcfitsio-bin libcfitsio-dev libopenblas-base libopenblas-dev wcslib-dev wcslib-tools libglib2.0-dev libcasa-casa2 casacore-dev casacore-data casacore-tools

Other systems
^^^^^^^^^^^^^

-  Install equivalent packages for your distribution

   -  g++
   -  cmake
   -  git
   -  pkg-config
   -  openblas
   -  libglib2.0-dev
   -  follow the instructions at https://github.com/casacore/casacore to
      install casacore.
   -  Additional packages (not essential, but recommended): MPI
      (openmpi), FFTW

Building
~~~~~~~~

-  Clone the repository

::

       git clone -b master https://git@github.com/nlesc-dirac/sagecal.git

-  Build SAGECal

::

       mkdir build && cd build
       cmake ..

**OPTIONAL:** You can also define a custom CASACORE path:

::

       cmake .. -DCASACORE_ROOT_DIR=/opt/soft/casacore

**OPTIONAL:** You can also define custom paths to everything:

::

       cmake -DCFITSIO_ROOT_DIR=/cm/shared/package/cfitsio/3380-gcc-4.9.3 -DCASACORE_ROOT_DIR=/cm/shared/package/casacore/v2.3.0-gcc-4.9.3 -DWCSLIB_INCLUDE_DIR=/cm/shared/package/wcslib/5.13-gcc-4.9.3/include -DWCSLIB_LIBRARY=/cm/shared/package/wcslib/5.13-gcc-4.9.3/lib/libwcs.so -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_LINKER=/cm/shared/package/gcc/4.9.3/bin/gcc -DCMAKE_CXX_FLAGS=-L/cm/shared/package/cfitsio/3380-gcc-4.9.3/lib -DCMAKE_C_FLAGS=-L/cm/shared/package/cfitsio/3380-gcc-4.9.3/lib ..

Compile with:

::

       make

Install at your favorite place

::

       make DEST=/path/to/sagecal/dir install

-  The sagecal executable can be found in
   **/path/to/sagecal/dir/usr/local/bin**, also
   **sagecal-mpi**,\ **buildsky** and **restore** might be installed
   depending on the availability of MPI and WCSLIB/FFTW.

MPI support
~~~~~~~~~~~

MPI support is automatically detected, otherwise, it can be forced with:

::

   cmake -DENABLE_MPI=ON

GPU Support
-----------

Loading modules on DAS5
~~~~~~~~~~~~~~~~~~~~~~~

See scripts folder for the modules.

::

   source ./scripts/load_das5_modules_gcc6.sh

Compiling with GPU support
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   mkdir -p build && cd build
   cmake -DCUDA_DEBUG=ON -DDEBUG=ON -DVERBOSE=ON -DHAVE_CUDA=ON ..
   make VERBOSE=1

Installation via Anaconda (WIP)
-------------------------------

::

       conda install -c sagecal=0.6.0

Manual installation
-------------------

For expert users, and for custom architectures (GPU), the manual install
is recommended. 
-  Prerequisites

   -  CASACORE_
   -  glib_
   -  BLAS/LAPACK Highly recommended is OpenBLAS_.
      Also, to avoid any linking issues (and to get best performance), build
      OpenBLAS from source and link SAGECal with the static library
      (libopenblas**.a) and NOT libopenblas**.so 
   -  Compilers gcc/g++ or Intel icc/icpc 
   -  If you have NVIDIA GPUs: CUDA/CUBLAS/CUSOLVER, nvcc and
      NVML (Nvidia management library) 
   -  If you are using Intel Xeon Phi MICs: Intel MKL and other libraries 

.. _CASACORE: http://casacore.googlecode.com/ 
.. _glib:  http://developer.gnome.org/glib 
.. _OpenBLAS: http://www.openblas.net/

- Get the source for SAGECal

::

       git clone -b master https://git@github.com/nlesc-dirac/sagecal.git


2 The basic way to build is
~~~~~~~~~~~~~~~~~~~~~~~~~~~

a) go to ./src/lib/Dirac and ./src/lib/Radio and run make (which will
create libdirac.a and libradio.a) 
b) go to ./src/MS and run make
(which will create the executable)

3 Build settings
~~~~~~~~~~~~~~~~

In ./src/lib/Dirac and ./src/lib/Radio and ./src/MS you MUST edit the
Makefiles to suit your system. 
Some common items to edit are: 

- LAPACK: directory where LAPACK/OpenBLAS is installed 
- GLIBI/GLIBL: include/lib files for glib 
- CASA_LIBDIR/CASA_INCDIR/CASA_LIBS : casacore include/library location and files: 
  Note with new CASACORE might need two include paths, e.g. -I/opt/casacore/include/
  -I/opt/casacore/include/casacore 
- CUDAINC/CUDALIB : where CUDA/CUBLAS/CUSOLVER is installed 
- NVML_INC/NVML_LIB : NVML include/lib path 
- NVCFLAGS : flags to pass to nvcc, especially -arch option to match your GPU 
- MKLROOT : for Intel MKL

Example makefiles: 

- Makefile : plain build 
- Makefile.gpu: with GPU support. 
  Note: Edit ./lib/Radio/Radio.h MAX_GPU_ID to match the number of
  available GPUs, e.g., for 2 GPUs, MAX_GPU_ID=1

SAGECAL-MPI Manual Installation
-------------------------------

This is for manually installing the distributed version of sagecal
(sagecal-mpi), the cmake build will will work for most cases. 

Prerequisites: 

- Same as for SAGECal. 
- MPI (e.g. OpenMPI)

2 Build ./src/lib/Dirac ./src/lib/Radio as above (using mpicc -DMPI_BUILD)
--------------------------------------------------------------------------

3 Build ./src/MPI using mpicc++
-------------------------------

BUILDSKY Installation
---------------------

-  See INSTALL in ./src/buildsky

RESTORE Installation
--------------------

-  See INSTALL in ./src/restore
