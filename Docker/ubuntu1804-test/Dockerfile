FROM ubuntu:bionic
MAINTAINER f.diblen@esciencecenter.nl

RUN apt-get update -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -s ppa:kernsuite/kern-5 -y && \
    apt-add-repository multiverse && \
    apt-add-repository restricted && \
    apt-get update -y

RUN apt-get install -y \
    git cmake g++ pkg-config \
    libcfitsio-bin libcfitsio-dev \
    libopenblas-base libopenblas-dev \
    wcslib-dev wcslib-tools \
    libglib2.0-dev \
    libcasa-casa2 casacore-dev casacore-data casacore-tools \
    mpich \
    fftw3-dev libfftw3-mpi3 libfftw3-bin


## compile sagecal
# RUN git clone https://github.com/nlesc-dirac/sagecal.git && cd sagecal && \ 
#     mkdir build-ubuntu && cd build-ubuntu && \
#     cmake .. -DCMAKE_INSTALL_PREFIX=/opt/sagecal && \
#     make -j4 && \
#     make install
# RUN ls -alsrt /opt/sagecal && \
#     /opt/sagecal/bin/sagecal
