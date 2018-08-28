#!/bin/bash

apt-get update -y
apt-get install software-properties-common -y
add-apt-repository -s ppa:kernsuite/kern-3 -y
apt-add-repository multiverse
apt-get update -y
apt-get install -y git cmake g++ pkg-config libcfitsio-bin libcfitsio-dev libopenblas-base libopenblas-dev wcslib-dev wcslib-tools libglib2.0-dev libcasa-casa2 casacore-dev casacore-data casacore-tools

