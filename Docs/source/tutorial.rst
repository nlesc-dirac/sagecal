Tutorial
========

Introduction
^^^^^^^^^^^^

This tutorial will guide you through the most common steps in calibration for imaging purposes. This involves direction dependent calibration using self calibration. Automatically this will cover calibration on a fixed sky model (e.g., for the LOFAR EoR KSP) as this is a single step in the self-calibration process.

It is assumed that you have performed direction independent calibration prior to this, for instance by using prefactor_.

.. _prefactor: https://github.com/lofar-astron/prefactor

Selfcal
^^^^^^^
We will demonstrate selfcal using the SAGECal executable for a GPU - sagecal_gpu - built with cmake, but instructions are, of course, similar for the containerized version of sagecal_gpu.

After you have cloned, built and installed SAGECal - e.g., in a directory called "install" - from the top level directory in the cloned repo, do:

::

   cd test/Calibration

and inspect the cluster file 3c196.sky.txt.cluster. This file sets the number of directions to calibrate on and we want to start with one direction, so this should show the contents of the cluster file:
::

   cat 3c196.sky.txt.cluster
   # cluster file
   # id chunks source list
   # -ve cluster ids are not subtracted from the data
   1 2 P3C196C1 P3C196C2 P3C196C3 P3C196C4
   # 2 1 P2C1
   # 3 2 P3C1 P3C2 P3C3
   # 4 1 P12C1 P12C2

So we start with a single cluster which we do not subtract from the data. And the number of time slots which we use as a command line argument for sagecal_gpu - through "-t" - will be halved, because of the "2" in the second column. This means that we will have smaller solution intervals.

We will use the small MeasurementSet sm.ms provided with the SAGECal repo to calibrate on the 3c196.sky.txt sky model.

::   

   module load openblas cuda91 casacore/2.3.0-gcc-4.9.3 (or a similar instruction, if necessary)
   ../../install/bin/sagecal_gpu -d sm.ms -s 3c196.sky.txt -c 3c196.sky.txt.cluster -n 40 -t 2 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 5 -k 1 -B 1 -E 1  > sm.ms.output

The "-t 2" together with the "2" in the second column of the cluster file means that we have effectively used a solution interval equal to the time sampling interval of the sm.ms observation. Also, we have used 40 CPU threads; optimally, this value coincides with the number of logical cores of your CPU. 
And we have "-k 1" to correct the residuals of our first - and only - cluster. These and other arguments are explained when you run 

::

   ../../install/bin/sagecal_gpu -h

Within a few minutes, the calibration will have completed and we will image the calibrated visibilities using 

:: 

   module load wsclean (or a similar instruction, if necessary)
   wsclean -size 1024 1024 -scale 0.7amin -niter 10000 -mgain 0.8 -auto-threshold 3 sm.ms


