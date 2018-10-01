#!/bin/bash

echo "Building SageCal for Scientific Linux" && \
echo "Branch --> $BRANCH" && \
cd /travis/workdir

# compile casacore first

# mkdir -p /opt/soft/casacore/data
# cd /opt/soft/casacore/data
# wget -c ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar
# tar zxfv WSRT_Measures.ztar && rm -f WSRT_Measures.ztar

cd /travis/workdir
git clone --progress --verbose https://github.com/casacore/casacore.git casacore_src && cd casacore_src

mkdir build && cd build
cmake3 -DUSE_FFTW3=ON -DCMAKE_INSTALL_PREFIX=/opt/soft/casacore -DDATA_DIR=/opt/soft/casacore/data -DUSE_OPENMP=ON \
    -DUSE_HDF5=ON -DBUILD_PYTHON=ON -DUSE_THREADS=ON ..
make -j4
make install

# compile sagecal
cd /travis/workdir && \
mkdir build-sl && cd build-sl

cmake3 .. -DCMAKE_INSTALL_PREFIX=/opt/sagecal && \
    -DCASACORE_ROOT_DIR=/opt/soft/casacore -DCASACORE_INCLUDE=/opt/soft/casacore/include/casacore
make -j4 && \
make install \
ls -alsrt /opt/sagecal && \
/opt/sagecal/bin/sagecal
