# Calibration examples
This directory includes a few examples of calibration using SAGECal. Direction independent calibration, direction dependent calibration as well as bandpass calibration is possible. Two shell scripts are included:

  * `dosage.sh`: calibration in a single computer
  * `dosage-mpi.sh`: calibration using a cluster of computers, using MPI

A small test data file `sm.ms.tar` is included, and you need to untar it before running the examples. The input model is given by:
  * `3c196.sky.txt`: sky model 
  * `3c196.sky.txt.cluster`: cluster file that specifies the directions being calibrated

## Distributed calibration
Here is a step-by-step guide to get going with distributed calibration. You only need one computer to test this, but using MPI configuration options, the same steps can be carried out over a cluster.

1 Untar `sm.ms.tar` like
```console
tar xvf sm.ms.tar
```
 then you will have `sm.ms` dataset (it is actually a directory).

2 Copy this to multiple files such as
```console
cp -r sm.ms sm1.ms
cp -r sm.ms sm2.ms
...
```

3 After this, use [this script](https://github.com/nlesc-dirac/sagecal/blob/master/test/Calibration/Change_freq.py) to change the frequency of each data file, like
```console
python2 ./Change_freq.py sm1.ms 110e6
python2 ./Change_freq.py sm2.ms 120e6
python2 ./Change_freq.py sm3.ms 130e6
...
```
 Each data file should have a distinct frequency, and the above method changes the frequency of each dataset to a unique value.

4 After this step, you can run `dosage-mpi.sh` to calibrate all datasets matching `*.ms` in the current directory.
