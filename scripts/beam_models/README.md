# Custom beam models in sagecal
This document describes creating custom element beam models to be used by sagecal. The model is common to all receivers. If you need to create unique element beam models for each receiver, go down to [creating per-station element beams](#creating-per-station-element-beams). 

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

# Creating per-station element beams
When each receiver dipole has its own unique element beam pattern, it is possible to provide a unique model file for each receiver. Also, these models will be saved as pure ASCII text files, hence no need to re-compile sagecal whenever the models are updated. The drawback is that these model files need to be read every time sagecal is run, but this is a minor penalty to pay. 

The format of each text file can be given as follows:


```
# comments are ignored
# first line: model_order (int) number of frequencies (int) scale beta (double) 
n_order n_frequencies beta
# frequency values (n_frequencies) in GHz
freq1 freq2 freq3 ....
# n_order x (n_order + 1) / 2 values for 1-st frequency 
real00 imag00
real01 imag01
...
# n_order x (n_order + 1) / 2 values for 2-nd frequency
real10 imag10
real11 imag11
...
...
# and so on...
```

Several things to keep in mind:

* Only stations with a single dipole can use unique beam models (no beamforming mode).
* All models for the full array should have same model hyper-parameters, such as the model order, the scale and the frequencies.
* All stations either should have a custom model or none, it is not possible for some stations to have a unique model while others to have a common model, in that case, copy the model file to all the station indexed directories.

The text files created for each receiver model should be saved under a directory having the receiver number as the name. The file name could be anything, but should end with *.model* suffix. Other files like the numpy files could also be saved under each directory.

wo  3 jun 2026 12:06:58 CEST
