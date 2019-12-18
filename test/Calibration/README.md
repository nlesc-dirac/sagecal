# Calibration examples
This directory includes a few examples of calibration using SAGECal. Direction independent calibration, direction dependent calibration as well as bandpass calibration is possible. Two shell scripts are included:

  * `dosage.sh`: calibration in a single computer
  * `dosage-mpi.sh`: calibration using a cluster of computers, using MPI

A small test data file `sm.ms.tar` is included, and you need to untar it before running the examples. The input model is given by:
  * `3c196.sky.txt`: sky model 
  * `3c196.sky.txt.cluster`: cluster file that specifies the directions being calibrated

In order to generate data at different frequencies, use `Change_freq.py` on the test data `sm.ms`. To generate mock sky models, use `Generate_sources.py`. Use `buildsky` for extracting sky models from FITS files. For creating a cluster file from a sky model file, use `create_clusters.py` script.
