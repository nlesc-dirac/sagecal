Tutorial
========

Introduction
^^^^^^^^^^^^

This tutorial will guide you through the most common steps in calibration for imaging purposes. This involves direction dependent calibration using self calibration. Automatically this will cover calibration on a fixed sky model (e.g., for the LOFAR EoR KSP) as this is a single step in the self-calibration process.

It is assumed that you have performed direction independent calibration prior to this, for instance by using prefactor_.

.. _prefactor: https://github.com/lofar-astron/prefactor

Selfcal
^^^^^^^
We will demonstrate selfcal using the SAGECal executable for a GPU - sagecal_gpu - built with cmake, but instructions are, of course, similar for the containerized version of sagecal_gpu. Building sagecal will also automatically build buildsky, which we need for self-calibration.

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
   ../../install/bin/sagecal_gpu -d sm.ms -s 3c196.sky.txt -c 3c196.sky.txt.cluster -n 40 -t 2 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k 1 -B 1 -E 1  > sm.ms.output

The "-t 2" together with the "2" in the second column of the cluster file means that we have effectively used a solution interval equal to the time sampling interval of the sm.ms observation. Also, we have used 40 CPU threads; optimally, this value coincides with the number of logical cores of your CPU. 
And we have "-k 1" to correct the residuals of our first - and only - cluster. These and other arguments are explained when you run 

::

   ../../install/bin/sagecal_gpu -h

This will also show you other options for "-j". "-j 5" uses a robust Riemannian trust region (RRTR), which is much faster than "-j 2" (OSRLM = Ordered Subsets Accelerated Robust Levenberg Marquardt). The downside from using RRTR is that it will only work properly if the power level of the visibilities that you are calibrating matches the power level of the sky model that you are using. If this is not the case, rounding errors may prevent you from finding accurate solutions. Use this Python 2 script - Scale.py - to scale your visibilities and write the output to the CORRECTED_DATA column:

::

   #!/usr/bin/env python2
   import pyrap.tables as pt
   import string
   def read_corr(msname,scalefac):
       tt=pt.table(msname,readonly=False)
       c=tt.getcol('DATA')
       tt.putcol('CORRECTED_DATA',c*scalefac)
       tt.close()
   if __name__ == '__main__':
       # args MS scalefac
       import sys
       argc=len(sys.argv)
       if argc==3:
           read_corr(sys.argv[1],float(sys.argv[2]))
       exit()

You can run this script like this:

::

   ./Scale.py sm.ms 1e5

Beware that it any subsequent steps you will have to use the "-I" option in SAGECal to use the "CORRECTED_DATA" column as input. For now, we will take the easy way by performing coarse calibration - using our initial sky model - with "-j 2" - which does not require any scaling - and using "-j 5" from the first selfcal loop onwards.
Within a few minutes, SAGECal will have completed coarse calibration and we can image the calibrated visibilities using 

:: 

   module load wsclean (or a similar instruction, if necessary)
   # wsclean -size 2048 2048 -scale 0.3amin -niter 10000 -mgain 0.8 -auto-threshold 3 sm.ms 
   wsclean -size 1024 1024 -scale 0.7amin -niter 10000 -mgain 0.8 -auto-threshold 3 sm.ms

This will produce an image wsclean-image.fits that we can use for the first round of self-calibration. To extract the sky model from wsclean-image.fits we will use Duchamp_. This three-dimensional source finder is most easily installed - after downloading and extracting the source code tar archive - using

::

   ./configure --prefix=/my/favorite/install/dir
   make
   make install


However, you may run into a missing "wcslib/cpgsbox.h" error. This can be solved by reconfiguring:

::

   ./configure --without-pgplot --prefix=/my/favorite/install/dir

Next, we need to supply Duchamp with a configuration file to extract a sky model from the FITS image. You can use this minimal configuration file:

:: 

   ##########################################
   imageFile       wsclean-image.fits
   logFile         logfile.txt
   outFile         results.txt
   spectraFile     spectra.ps
   minPix          5
   snrRecon        10.
   flagKarma 1
   karmaFile duchamp.ann
   flagnegative 0
   flagMaps 0
   flagOutputMask 1
   flagMaskWithObjectNum 1
   flagXOutput 0
   ############################################

which we call my-Duchamp-conf.txt.

Simply run it like this:

::

   Duchamp -p my-Duchamp-conf.txt 

.. _Duchamp: https://www.atnf.csiro.au/people/Matthew.Whiting/Duchamp/

Next, build the sky model using the mask file:

::

   /path/to/buildsky -f wsclean-image.fits -m wsclean-image.MASK.fits -o 1

This will create a sky model file wsclean-image.fits.sky.txt, in LSM format [#]_.

From this, we need to construct a cluster file, which determines the directions for which we seek calibration solutions. src/buildsky/create_clusters.py can be used to construct such a file by setting the number of clusters for a given sky model. It is a Python 3 script that requires the source model to be in LSM format. Thankfully, we have run buildsky in the appropriate manner.

::

   /path/to/create_clusters.py -s wsclean-image.fits.sky.txt -c 10 -o wsclean-image.fits.sky.txt.cluster -i 10

This will produce a cluster file wsclean-image.fits.sky.txt.cluster defining 10 clusters. A maximum of 10 iterations was set, but 5 were sufficient.
Next, we run a selfcal loop:

::
   ../../install/bin/sagecal_gpu -I CORRECTED_DATA -d sm.ms -s wsclean-image.fits.sky.txt -c wsclean-image.fits.sky.txt.cluster -n 40 -t 2 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k 1 -B 1 -E 1  > sm.ms.output
   wsclean -size 1024 1024 -scale 0.7amin -niter 10000 -mgain 0.8 -auto-threshold 3 sm.ms

Note the "-I CORRECTED_DATA". It is essential since our new model wsclean-image.fits.sky.txt and our new cluster file wsclean-image.fits.sky.txt.cluster have the 3C196 cluster subtracted, so the visibilities should also exclude this source. We could have run this calibration loop faster by using "-j 5".

.. rubric:: Footnotes

.. [#] I was not able to find a document describing the format of a sky model file in LSM format.


