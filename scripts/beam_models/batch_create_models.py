#! /usr/bin/env python

import math
import numpy as np
import argparse
from create_header import BeamGenerator

import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(211,projection='3d')
ax2=fig.add_subplot(212,projection='3d')

def main(args):
    # glob all directories
    bg=BeamGenerator(n0=args.order, beta=args.scale, outfile='/tmp/xx.out', preamble='', verbose=args.verbose)
    bg.load_model()
    bg.setup_basis()
    if args.show:
       bg.show_basis()
    bg.decompose_write_header()


if __name__=='__main__':
    parser=argparse.ArgumentParser(
      description='Generate element beam coefficients for sagecal',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--scale',type=float,default=0.5,
        help='basis scale factor')
    parser.add_argument('--order',type=int,default=7,
        help='basis model order')
    parser.add_argument('--root_dir',type=str,default='',
        help='root directory under which the subdirectories with .npy files')
    parser.add_argument('--verbose', action='store_true', default=False,
       help='print more information')
    # *.npy files in directory of station name
    # parse dir pattern like 'HBA*' and output named text files in current dir,
    # the names will be like HBA0.txt, HBA1.txt ...
 
    args=parser.parse_args()
    if len(args.root_dir) > 0:
      main(args)
    else:
      parser.print_help()
