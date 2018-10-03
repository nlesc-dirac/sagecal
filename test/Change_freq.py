#!/usr/bin/env python
from pyrap.tables import *

# You can use this script to create different MS data files by copying a single MS
# into MS with different names and changing their frequencies to unique values

# change freq
def read_corr(msname,freq):
  import os
  import math
  tf=table(msname+'/SPECTRAL_WINDOW',readonly=False)
  ch0=tf.getcol('CHAN_FREQ')
  ch0[0,0]=freq
  reffreq=tf.getcol('REF_FREQUENCY')
  reffreq[0]=ch0[0,0]
  tf.putcol('CHAN_FREQ',ch0)
  tf.putcol('REF_FREQUENCY',reffreq)
  tf.close()


if __name__ == '__main__':
  # args MS
  import sys
  argc=len(sys.argv)
  if argc==3:
   read_corr(sys.argv[1],float(sys.argv[2]))
  else:
   print("thisscript MS frequency")
  exit()
