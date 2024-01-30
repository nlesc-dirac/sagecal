FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install software-properties-common -y && \
    apt-add-repository multiverse && \
    apt-add-repository restricted && \
    apt-get update -y

RUN apt-get install -y \
    git cmake g++ pkg-config \
    libcfitsio-bin libcfitsio-dev \
    liblapack-dev libopenblas-dev \
    wcslib-dev wcslib-tools \
    libglib2.0-dev \
    libcasa-casa4 casacore-dev casacore-data casacore-tools \
    mpich \
    fftw3-dev libfftw3-mpi3 libfftw3-bin


## compile sagecal
RUN git clone --depth 1 --branch master \
     https://github.com/nlesc-dirac/sagecal.git && cd sagecal && \ 
     mkdir build-ubuntu && cd build-ubuntu && \
     cmake -DCMAKE_INSTALL_PREFIX=/opt/sagecal \
     -DBLA_VENDOR=OpenBLAS \
     -DCMAKE_CXX_FLAGS="-O3 -ffast-math -lmvec -lm" \
     -DCMAKE_C_FLAGS="-O3 -ffast-math -lmvec -lm" .. && \
     make -j4 && \
     make install
RUN ls -alsrt /opt/sagecal && \
     /opt/sagecal/bin/sagecal
