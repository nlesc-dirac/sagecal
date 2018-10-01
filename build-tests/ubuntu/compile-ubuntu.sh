#!/bin/bash

echo "Building SageCal" && \
echo "Branch --> $BRANCH" && \
cd /travis/workdir && \
mkdir build-ubuntu && cd build-ubuntu && \
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/sagecal && \
make -j4 && \
make install && \
ls -alsrt /opt/sagecal && \
/opt/sagecal/bin/sagecal

