#!/bin/bash

# add EPEL repository for openblas
yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

# install dependencies
yum -y install wget git pkgconfig make cmake3 cmake3-gui gcc-gfortran gcc-c++ flex bison \
       openblas openblas-devel glib2-devel lapack lapack-devel cfitsio cfitsio-devel \
       wcslib wcslib-devel ncurses ncurses-devel readline readline-devel\
       python-devel boost boost-devel fftw fftw-devel hdf5 hdf5-devel\
       numpy boost-python