#!/usr/bin/env bash

echo 'script: ' $0

echo "Building SageCal" && \
echo "Branch --> $BRANCH" && \
echo "Image --> $IMAGE"

cd /travis/workdir && \
    mkdir build && cd build

OPTS=''

case $IMAGE in
    ubuntu)
        OPTS=''
        ;;
    sl7)
        OPTS='-DUSE_FFTW3=ON \
              -DCMAKE_INSTALL_PREFIX=/opt/casacore \
              -DDATA_DIR=/opt/casacore/data -DUSE_OPENMP=ON \
              -DUSE_HDF5=ON \
              -DBUILD_PYTHON=ON \
              -DUSE_THREADS=ON'
        ;;
    arch)
        OPTS=''
        ;;
    *)
        echo 'Unknown image $IMAGE!'
        exit 1
        ;;
esac


echo 'CMake options: ' $OPTS

cmake .. -DCMAKE_INSTALL_PREFIX=/opt/sagecal \
    $OPTS

make -j4 && \
make install && \
ls -alsrt /opt/sagecal && \
/opt/sagecal/bin/sagecal
