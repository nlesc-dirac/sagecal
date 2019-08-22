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

and download an initial, coarse, sky model from SkyView_ to calibrate our small observation sm.ms - provided with sagecal repo - of 3C196 on.  
Enter "3C196" in the "Coordinates or Source" field and select "TGSS ADR1" from the "Radio: MHz:" window. Otherwise, use default values for donwloading the FITS image. For some reason SkyView will not provide a SIN projected image from TGSS, but you can get these by downloading from the `TGSS archive`_ directly. For now, a TAN projected image will suffice, because we only need the central source.

.. _skyview: https://skyview.gsfc.nasa.gov/current/cgi/query.pl
.. _`TGSS archive`: https://vo.astron.nl/tgssadr/q_fits/cutout/form
 
To extract the sky model from SkyView image - let's call it skyview-image.fits - we will use Duchamp_. This three-dimensional source finder is most easily installed - after downloading and extracting the source code tar archive - using

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
   imageFile       skyview-image.fits
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

Now build the sky model using the mask file wsclean-image.MASK.fits we just obtained:

::

   /path/to/buildsky -f skyview-image.fits -m skyview-image.MASK.fits -o 1 -a 25 -b 25 -p 0

This will create a sky model file skyview-image.fits.sky.txt, in `LSM format`_, making use of the clean beam size of the TGSS, which is not provided in the header of this SkyView image.

.. _`LSM format`: https://github.com/nlesc-dirac/sagecal/blob/master/README.md#2c-sky-model-format 

From this, we need to construct a cluster file, which determines the directions for which we seek calibration solutions. src/buildsky/create_clusters.py can be used to construct such a file by setting the number of clusters for a given sky model. It is a Python 3 script that requires the source model to be in LSM format. Thankfully, we have run buildsky in the appropriate manner.

::

   /path/to/create_clusters.py -s skyview-image.fits.sky.txt -c -1 -o skyview-image.fits.sky.txt.cluster -i 10

This will produce a cluster file skyview-image.fits.sky.txt.cluster with just one cluster. We chose to enter "-1" instead of "1" to give the cluster a negative id, such that it will not be subtracted when we run sagecal. A maximum of 10 iterations was set, but 2 were sufficient.


::   

   module load openblas cuda91 casacore/2.3.0-gcc-4.9.3 (or a similar instruction, if necessary)
   ../../install/bin/sagecal_gpu -d sm.ms -s skyview-image.fits.sky.txt -c skyview-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k 1 -B 1 -E 1  > sm.ms.output

The "-t 1" means that we have chosen a solution interval equal to the time sampling interval of the sm.ms observation. Also, we have used 40 CPU threads; optimally, this value coincides with the number of logical cores of your CPU. 
And we have "-k 1" to correct the residuals of cluster number 1. 

.. 
   We can also correct for the second (and third and fourth) direction - towards cluster # 2 - by entering "-k 2" while including the corrections we already derived for cluster # 1, by adding "-I CORRECTED_DATA". The latter accommodates for using the output of the first sagecal run as input. We will have to use this argument for all our subsequent sagecal runs. So issue these commands:

..   ../../install/bin/sagecal_gpu -d sm.ms -s skyview-image.fits.sky.txt -c skyview-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 5 -k 2 -B 1 -E 1 -I CORRECTED_DATA > sm.ms.output
..   ../../install/bin/sagecal_gpu -d sm.ms -s skyview-image.fits.sky.txt -c skyview-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 5 -k 3 -B 1 -E 1 -I CORRECTED_DATA > sm.ms.output
..   ../../install/bin/sagecal_gpu -d sm.ms -s skyview-image.fits.sky.txt -c skyview-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 5 -k 4 -B 1 -E 1 -I CORRECTED_DATA > sm.ms.output
   
These and other arguments are explained when you run 

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

We will use "-j 5" from the first selfcal loop onwards.
Within a few minutes, SAGECal will have completed coarse calibration and we can image the calibrated visibilities using 

:: 

   module load wsclean (or a similar instruction, if necessary)
   # wsclean -size 2048 2048 -scale 0.3amin -niter 10000 -mgain 0.8 -auto-threshold 3 sm.ms 
   wsclean -size 1024 1024 -scale 0.7amin -niter 10000 -mgain 0.8 -auto-threshold 3 sm.ms

This will produce an image wsclean-image.fits, that looks like this:

.. image:: image_after_initial_calibration.png

It is a pretty coarsely calibrated image that has a rms noise of more than 40 mJy/bm. We can use it for the first round of self-calibration. To do so, we will have to extract a new sky model from it. Modify your Duchamp configuration file to work on our image wsclean-image.fits instead of skyview-image.fits and add a line "fileOutputMask  wsclean-image-MASK.fits" to prevent Duchamp from producing a mask file with a space in the file name, which ds9 cannot handle. Also, increase the minimum number of pixels - minPix 50 - and the required signal to noise for a detection - snrRecon 50. This should make sure that no sidelobes of the central source 3C196 are detected and included in the mask file.
Let's call this configuration file Duchamp-conf-for-first-selfcal-loop.txt. Run Duchamp with this configuration file and also buildsky and create_clusters.py with one cluster:

::

   Duchamp -p Duchamp-conf-for-first-selfcal-loop.txt
   buildsky -f wsclean-image.fits -m wsclean-image-MASK.fits -o 1
   create_clusters.py -s wsclean-image.fits.sky.txt -c -1 -o wsclean-image.fits.sky.txt.cluster -i 10

Now we can do another round of calibration and imaging, for which one sagecal command will suffice, because we only have one cluster:

::
   
   ../../install/bin/sagecal_gpu -d sm.ms -s wsclean-image.fits.sky.txt -c wsclean-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 5 -k 1 -B 1 -E 1 -I CORRECTED_DATA > sm.ms.output
   wsclean -size 1024 1024 -name after_one_selfcal_loop -scale 0.7amin -niter 10000 -mgain 0.8 -auto-threshold 3 sm.ms

Note that  we left out the "-I CORRECTED_DATA", because it is redundant in this case. However, adding it will yield the same calibrated visibilities and hence the same image. Also note the "-name after_one_selfcal_loop" to make sure that wsclean does not overwrite our previous image.

