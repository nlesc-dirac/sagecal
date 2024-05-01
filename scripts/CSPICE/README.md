# Installing CSPICE toolkit
This document describes building the CSPICE toolkit to link with sagecal.

# Download
Get the latest [cspice.tar.Z](https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z)

```
wget https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z
```

Extract the files

```
tar xvf cspice.tar.Z
```
and there will be directory ```cspice```.

# Copy files
Copy all files in this directory to the ```cspice``` directory.

```
cp makeall.sh mkprodct.sh pkgconfig.stub /full_path_to/cspice/
```

Note that ```/full_path_to/cspice/``` is the directory created by extracting cspice.tar.Z.

# Build
In the ```cspice``` directory, run

```
bash ./makeall.sh
```
and you are ready to link with sagecal.


# Linking with sagecal
Use cmake flag ```-DCSPICE_PREFIX=/full_path_to/cspice``` to point to ```cspice``` directory created after exctracting CSPICE.


# Download kernels
The following kernels are required.

```
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_220930.tf
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc
```

Download them and save them in a directory, and set your environment variable ```CSPICE_KERNEL_PATH``` to point to this directory before running sagecal.

zo 28 apr 2024 20:16:15 CEST
