

| **master**  | **dev** |
| ------------- | ------------- |
| [![Build Status](https://travis-ci.org/nlesc-dirac/sagecal.svg?branch=master)](https://travis-ci.org/nlesc-dirac/sagecal)  | [![Build Status](https://travis-ci.org/nlesc-dirac/sagecal.svg?branch=dev)](https://travis-ci.org/nlesc-dirac/sagecal)  |

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1289316.svg)](https://doi.org/10.5281/zenodo.1289316)
[![Documentation](https://codedocs.xyz/nlesc-dirac/sagecal.svg)](https://codedocs.xyz/nlesc-dirac/sagecal/)

# SAGECAL


## Features

- Levenberg-Marquardt, LBFGS, Riemannian Trust Region, Nesterov's accelerated gradient descent algorithms
- GPU acceleration using CUDA
- Fast and accurate interferometric calibration
- Gaussian and Student's t noise models
- Shapelet source models
- CASA MS data format supported
- Distributed calibration using MPI - consensus optimization with data multiplexing
- Tools to build sky models and restore sky models to images
- Adaptive update of ADMM penalty (Barzilai-Borwein a.k.a. Spectral method)
- Bandpass calibration and unprecedented RFI mitigation with stochastic LBFGS
- Stochastic calibration for handling data at highest resolution (with federated averaging and consensus optimization)


Please read INSTALL.md for installation instructions, but 'cmake' should work in most cases. We give a brief guide to use SAGECal here but there is extensive documentation here.

## Contributing
Read the [contributing guide](https://github.com/nlesc-dirac/sagecal/blob/master/CONTRIBUTING.md)


## Code documentation
Code documentation can be found [here](https://codedocs.xyz/nlesc-dirac/sagecal).


## Step by Step Introduction:

### 1) Input Data
Input to sagecal must be in CASA MS format, make sure to create a column in the MS to write output data as well. The data can be in raw or averaged form, also initial calibration using other software can be also applied.


### 2) Sky Model:
#### 2a) Make an image of your MS (using ExCon/casapy). 
Use Duchamp to create a mask for the image. Use buildsky to create a sky model. (see the README file on top level directory). Also create a proper cluster file.
Special options to buildsky: -o 1 (NOTE: not -o 2)

Alternatively, create these files by hand according to the following formats.

#### 2b) Cluster file format:
cluster_id chunk_size source1 source2 ...
e.g.
```

0 1 P0C1 P0C2
2 3 P11C2 P11C1 P13C1

```

Note: putting -ve values for cluster_id will not subtract them from data.
chunk_size: find hybrid solutions during one solve run. Eg. if -t 120 is used 
to select 120 timeslots, cluster 0 will find a solution using the full 120 timeslots while cluster 2 will solve for every 120/3=40 timeslots.

#### 2c) Sky model format:
```
#name h m s d m s I Q U V spectral_index RM extent_X(rad) extent_Y(rad) pos_angle(rad) freq0
```

or

```
#name h m s d m s I Q U V spectral_index1 spectral_index2 spectral_index3 RM extent_X(rad) extent_Y(rad) pos_angle(rad) freq0
```

e.g.:

```
P1C1 0 12 42.996 85 43 21.514 0.030498 0 0 0 -5.713060 0 0 0 0 115039062.0
P5C1 1 18 5.864 85 58 39.755 0.041839 0 0 0 -6.672879 0 0 0 0 115039062.0
#A Gaussian mjor,minor 0.1375,0.0917 deg diameter -> radius(rad), PA 43.4772 deg (-> rad)
#Position Angle: "West from North (counter-clockwise)" (0 deg = North, 90 deg = West). 
#Note: PyBDSM and BBS use "North from East (counter-clockwise)" (0 deg = East, 90 deg = North). 
G0  5 34 31.75 22 00 52.86 100 0 0 0 0.00 0 0.0012  0.0008 -2.329615801 130.0e6
#A Disk radius=0.041 deg
D01 23 23 25.67 58 48 58 80 0 0 0 0 0 0.000715 0.000715 0 130e6
#A Ring radius=0.031 deg
R01 23 23 25.416 58 48 57 70 0 0 0 0 0 0.00052 0.00052 0 130e6
#A shapelet ('S3C61MD.fits.modes' file must be in the current directory)
S3C61MD 2 22 49.796414 86 18 55.913266 0.135 0 0 0 -6.6 0 1 1 0.0 115000000.0
```


Note: Comments starting with a '#' are allowed for both sky model and cluster files.
Note: 3rd order spectral indices are also supported, use -F 1 option in sagecal.
Note: Spectral indices use natural logarithm, ```exp(ln(I0) + p1 * ln(f/f0) + p2 * ln(f/f0)^2 + ..)``` so if you have a model with common logarithms like ```10^(log(J0) + q1*log(f/f0) + q2*log(f/f0)^2 + ..)``` then, conversion is

```
ln(I0)+p1*ln(f/f0)+p2*ln(f/f0)^2+... = ln(10)*(log(J0)+q1*log(f/f0)+q2*log(f/f0))^2)+...)
=ln(10)*(ln(J0)/ln(10)+q1*ln(f/f0)/ln(10)+q2*ln(f/f0)^2/ln(10)^2+...)
```
so
```
I0=J0
p1=q1
p2=q2/ln(10)
p3=q3/ln(10)^2
...
```

### 3) Run sagecal
Optionally: Make sure your machine has (1/2 working NVIDIA GPU cards or Intel Xeon Phi MICs) to use sagecal.
Recommended usage: (with GPUs)

```
sagecal -d my_data.MS -s my_skymodel -c my_clustering -n no.of.threads -t 60 -p my_solutions -e 3 -g 2 -l 10 -m 7 -w 1 -b 1
```

Use your solution interval (-t 60) so that its big enough to get a decent solution and not too big to make the parameters vary too much. (about 20 minutes per solution is reasonable).

Note: It is also possible to calibrate more than one MS together. See section 4 below.
Note: To fully use GPU acceleration use -E 1 option.

Simulations:
With -a 1, only a simulation of the sky model is done.
With -a 1 and -p 'solutions_file', simulation is done with the sky model corrupted with solutions in 'solutions_file'.
With -a 1 and -p 'solutions_file' and -z 'ignore_file', simulation is done with the solutions in the 'solutions_file', but ignoring the cluster ids in the 'ignore_file'.
Eg. If you need to ignore cluster ids '-1', '10', '999', create a text file :

```
-1
10
999
```

and use it as the 'ignore_file'.


### 4) Distributed calibration

Use mpirun to run sagecal-mpi, example:
```
 mpirun  -np 11 -hostfile ./machines --map-by node --cpus-per-proc 8 
 --mca yield_when_idle 1 -mca orte_tmpdir_base /scratch/users/sarod 
 /full/path/to/sagecal-mpi -f 'MS*pattern' -A 30 -P 2 -r 5 
 -s sky.txt -c cluster.txt -n 16 -t 1 -e 3 -g 2 -l 10 -m 7 -x 10 -F 1 -j 5
```

Specific options : 
```-np 11``` : 11 processes : starts 10 slaves + 1 master

```./machines``` : will list the host names of the 11 (or fewer) nodes used ( 1st name is the master ) : normally the node where you invoke mpirun

```-f 'MS*pattern'``` : Search MS names that match this pattern and calibrate all of them together. The total number of MS being calibrated can be higher than the actual number of slaves (multiplexing).

```-A 30``` : 30 ADMM iterations.

```-P 2``` : polynomial in frequency has 2 terms.

```-Q``` : can change the type of polynomial used (```-Q 2``` gives Bernstein polynomials).

```-r 5``` : regularization factor is 5.0.

```-G textfile```: each cluster can have a different regularization factor, instead of using ```-r``` option when the regularization is the same for all clusters.

MPI specific options:

```/scratch/users/sarod``` : this is where MPI stores temp files (default is probably ```/tmp```).

```--mca*```: various options to tune the networking and scheduling.

Note: the number of slaves (-np option) can be lower than the number of MS calibrated. The program will divide the workload among the number of available slaves.


The rest of the options are similar to sagecal.


### 5) Solution format
All SAGECal solutions are stored as text files. Lines starting with '#' are comments.
The first non-comment line includes some general information, i.e.
freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters

The remaining lines contain solutions for each cluster as a single column, the first column is just a counter. 
Let's say there are K effective clusters and N directions. Then there will be K+1 columns, the first column will start from 0 and increase to 8N-1, 
which can be used to count the row number. It will keep repeating this, for each time interval.
The rows 0 to 7 belong to the solutions for the 1st station. The rows 8 to 15 for the 2nd station and so on. 
Each 8 rows of any given column represent the 8 values of a 2x2 Jones matrix. Lets say these are ```S0,S1,S2,S3,S4,S5,S6``` and ```S7```. Then the Jones matrix is ```[S0+j*S1, S4+j*S5; S2+j*S3, S6+j*S7]``` (the ';' denotes the 1st row of the 2x2 matrix).

When a cluster has a chunk size > 1, there will be more than 1 solution per given time interval. 
So for this cluster, there will be more than 1 column in the solution file, the exact number of columns being equal to the chunk size.



### Additional Info
See [LOFAR Cookbook Chapter](https://support.astron.nl/LOFARImagingCookbook/sagecal.html).

Please cite this code using the DOI.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1289316.svg)](https://doi.org/10.5281/zenodo.1289316)
