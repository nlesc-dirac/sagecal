# Custom beam models in sagecal
This document describes creating custom element beam models to be used by sagecal.

# Model files
In order to create a model, the following numpy files (**".npy"** extension) are needed.

  * **theta.npy**: coordinates for theta (elevation, in degrees), 1 dimensional array, *n_theta*
  * **phi.npy**: coordinates for phi (azimuth, in degrees), 1 dimensional array, *n_phi*
  * **frequency.npy**: coordinates for frequency (in Hz), 1 dimensional array, *n_frequency*
  * **etheta.npy**: voltage pattern for E-theta field, shape *n_frequency x n_theta x n_phi* (complex)
  * **ephi.npy**: voltage pattern for E-phi field, shape *n_frequency x n_theta x n_phi* (complex)


# Creating model
Run the script like

```
./create_header.py --scale 0.8 --order 10 --output elementcoeff_new.h
```
where *--scale* is the model scaling factor, *--order* is the model order, and *--output* is the output file to create. 

Running with the default values like

```
./create_header.py
```
will create *output.h* as the output file.

To see all options, run

```
./create_header.py --help
```

# Copy and compile
Copy the newly created header (*output.h* for example) to *../../src/lib/Radio/elementcoeff_ALO.h*.

Thereafter, rebuild sagecal (*make clean && make*).

do 28 aug 2025 10:25:03 CEST
