FROM scientificlinux/sl:7
MAINTAINER f.diblen@esciencecenter.nl

# add EPEL repository for openblas
RUN yum -y \
    install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

# install dependencies
RUN yum -y install \
    wget git pkgconfig make cmake3 cmake3-gui gcc-gfortran gcc-c++ flex bison \
    openblas openblas-devel glib2-devel lapack lapack-devel cfitsio cfitsio-devel \
    wcslib wcslib-devel ncurses ncurses-devel readline readline-devel \
    python-devel boost boost-devel fftw fftw-devel hdf5 hdf5-devel \
    numpy boost-python mpich mpich-devel fftw fftw-libs fftw-devel

RUN mkdir /build && cd /build

# compile casacore
RUN git clone --progress --verbose https://github.com/casacore/casacore.git casacore_src && \
    cd casacore_src && \
    mkdir build && cd build && \
    cmake3 .. -DUSE_FFTW3=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/casacore \
    -DDATA_DIR=/opt/casacore/data -DUSE_OPENMP=ON \
    -DUSE_HDF5=ON \
    -DBUILD_PYTHON=ON \
    -DUSE_THREADS=ON && \
    make -j4 && \
    make install


## compile sagecal
#RUN cd /build && \
#    mkdir build-sl && cd build-sl && \
#    cmake3 .. -DCMAKE_INSTALL_PREFIX=/opt/sagecal \
#    -DCASACORE_ROOT_DIR=/opt/casacore \
#    -DCASACORE_INCLUDE=/opt/casacore/include/casacore
#    make -j4 && \
#    make install && \
#    ls -alsrt /opt/sagecal && \
#    /opt/sagecal/bin/sagecal && \
#    /opt/sagecal/bin/sagecal-mpi
