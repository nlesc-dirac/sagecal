#!/bin/bash

echo "Building SageCal" && \
echo "Branch --> $BRANCH" && \
cd /travis/workdir && \
mkdir build-arch && cd build-arch && \
cmake .. -DENABLE_CUDA=OFF && \
make -j4 && \
ls -alsrt ./dist/bin && \
./dist/bin/sagecal

