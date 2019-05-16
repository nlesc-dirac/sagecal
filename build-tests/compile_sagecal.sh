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
        echo 'Building for Ubuntu'
        CMAKE_EXE=$(which cmake)
        OPTS=''
        ;;
    sl7)
        echo 'Building for Scientific Linux'
        CMAKE_EXE=$(which cmake3)
        OPTS='-DCASACORE_ROOT_DIR=/opt/casacore'
        ;;
    ubuntu1804)
        echo 'Building for Ubuntu 18.04'
        CMAKE_EXE=$(which cmake)
        OPTS=''
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

echo 'ls -asl: '
ls -asl

echo 'ls -asl /travis/workdir: '
ls -asl /travis/workdir

echo 'ls -asl /travis/workdir/$BUILD_DIR: '
ls -asl /travis/workdir/$BUILD_DIR


$CMAKE_EXE /travis/workdir -DCMAKE_INSTALL_PREFIX=/opt/sagecal $OPTS

make -j4 && \
make install && \
ls -alsrt /opt/sagecal && \
/opt/sagecal/bin/sagecal
