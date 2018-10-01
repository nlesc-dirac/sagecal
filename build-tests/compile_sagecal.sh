#!/usr/bin/env bash

echo 'script: ' $0

echo "Building SageCal" && \
echo "Branch --> $BRANCH" && \
echo "Image --> $IMAGE"

BUILD_DIR=$IMAGE'-build'

cd /travis/workdir && \
    mkdir $BUILD_DIR && cd $BUILD_DIR

CMAKE_EXE=''
OPTS=''

case $IMAGE in
    ubuntu)
        CMAKE_EXE=$(which cmake)
        OPTS=''
        ;;
    sl7)
        CMAKE_EXE=$(which cmake3)
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

echo 'CMAKE_EXE: ' $CMAKE_EXE
echo 'CMake options: ' $OPTS
echo 'pwd: ' $PWD
ls -asl

$CMAKE_EXE .. -DCMAKE_INSTALL_PREFIX=/opt/sagecal \
    $OPTS

make -j4 && \
make install && \
ls -alsrt /opt/sagecal && \
/opt/sagecal/bin/sagecal
