#!/usr/bin/env python
#
# Copyright (C) 2011- Sarod Yatawatta <sarod@users.sf.net>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

import re
import math
import optparse


def annotate_lsm_sky(infilename,clusterfilename,outfilename,clid=None,color='yellow',rname=False):
  # LSM format
  # NAME RA(hours, min, sec) DEC(degrees, min, sec) sI sQ sU sV SI RM eX eY eP f0
  # or 3rd order spectra
  # NAME RA(hours, min, sec) DEC(degrees, min, sec) sI sQ sU sV SI0 SI1 SI2 RM eX eY eP f0

  # regexp pattern
  pp=re.compile(r"""
   ^(?P<col1>[A-Za-z0-9_.]+)  # column 1 name: must start with a character
   \s+             # skip white space
   (?P<col2>[-+]?\d+(\.\d+)?)   # RA angle - hours 
   \s+             # skip white space
   (?P<col3>[-+]?\d+(\.\d+)?)   # RA angle - min 
   \s+             # skip white space
   (?P<col4>[-+]?\d+(\.\d+)?)   # RA angle - sec 
   \s+             # skip white space
   (?P<col5>[-+]?\d+(\.\d+)?)   # Dec angle - degrees
   \s+             # skip white space
   (?P<col6>[-+]?\d+(\.\d+)?)   # Dec angle - min
   \s+             # skip white space
   (?P<col7>[-+]?\d+(\.\d+)?)   # Dec angle - sec 
   \s+             # skip white space
   (?P<col8>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes I - Flux 
   \s+             # skip white space
   (?P<col9>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes Q - Flux 
   \s+             # skip white space
   (?P<col10>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes U - Flux 
   \s+             # skip white space
   (?P<col11>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes V - Flux 
   \s+             # skip white space
   (?P<col12>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Spectral index
   \s+             # skip white space
   (?P<col13>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Rotation measure
   \s+             # skip white space
   (?P<col14>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)   # ext source major axis: rad
   \s+             # skip white space
   (?P<col15>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)   # ext source minor axis: rad
   \s+             # skip white space
   (?P<col16>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)   # ext source position angle : rad
   \s+             # skip white space
   (?P<col17>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)?   # reference frequency
   [\S\s]*""",re.VERBOSE)
  pp1=re.compile(r"""
   ^(?P<col1>[A-Za-z0-9_.]+)  # column 1 name: must start with a character
   \s+             # skip white space
   (?P<col2>[-+]?\d+(\.\d+)?)   # RA angle - hours 
   \s+             # skip white space
   (?P<col3>[-+]?\d+(\.\d+)?)   # RA angle - min 
   \s+             # skip white space
   (?P<col4>[-+]?\d+(\.\d+)?)   # RA angle - sec 
   \s+             # skip white space
   (?P<col5>[-+]?\d+(\.\d+)?)   # Dec angle - degrees
   \s+             # skip white space
   (?P<col6>[-+]?\d+(\.\d+)?)   # Dec angle - min
   \s+             # skip white space
   (?P<col7>[-+]?\d+(\.\d+)?)   # Dec angle - sec 
   \s+             # skip white space
   (?P<col8>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes I - Flux 
   \s+             # skip white space
   (?P<col9>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes Q - Flux 
   \s+             # skip white space
   (?P<col10>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes U - Flux 
   \s+             # skip white space
   (?P<col11>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Stokes V - Flux 
   \s+             # skip white space
   (?P<col12a>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Spectral index
   \s+             # skip white space
   (?P<col12b>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Spectral index
   \s+             # skip white space
   (?P<col12c>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Spectral index
   \s+             # skip white space
   (?P<col13>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)  # Rotation measure
   \s+             # skip white space
   (?P<col14>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)   # ext source major axis: rad
   \s+             # skip white space
   (?P<col15>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)   # ext source minor axis: rad
   \s+             # skip white space
   (?P<col16>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)   # ext source position angle : rad
   \s+             # skip white space
   (?P<col17>[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?)?   # reference frequency
   [\S\s]*""",re.VERBOSE)


  infile=open(infilename,'r')
  all=infile.readlines()
  infile.close()

  SR={} # sources
  for eachline in all:
    v=pp1.search(eachline)
    if v!= None:
      # find RA,DEC (rad) and flux, with proper sign
      if (float(v.group('col2'))) >= 0.0:
         mysign=1.0
      else:
         mysign=-1.0
      mra=mysign*(abs(float(v.group('col2')))+float(v.group('col3'))/60.0+float(v.group('col4'))/3600.0)*360.0/24.0
      if (float(v.group('col5'))) >= 0.0:
         mysign=1.0
      else:
         mysign=-1.0

      mdec=mysign*(abs(float(v.group('col5')))+float(v.group('col6'))/60.0+float(v.group('col7'))/3600.0)
      SR[str(v.group('col1'))]=(mra,mdec,float(v.group('col8')))
    else:  
      v=pp.search(eachline)
      if v!= None:
        if (float(v.group('col2'))) >= 0.0:
           mysign=1.0
        else:
           mysign=-1.0
        mra=mysign*(abs(float(v.group('col2')))+float(v.group('col3'))/60.0+float(v.group('col4'))/3600.0)*360.0/24.0
        if (float(v.group('col5'))) >= 0.0:
           mysign=1.0
        else:
           mysign=-1.0

        mdec=mysign*(abs(float(v.group('col5')))+float(v.group('col6'))/60.0+float(v.group('col7'))/3600.0)
        SR[str(v.group('col1'))]=(mra,mdec,float(v.group('col8')))

  print 'Read %d sources'%len(SR)

  CL={} # clusters 
  pp=re.compile(r"""
   ^(?P<col1>[-+]?\d+)  # ID: an integer 
   \s+             # skip white space
   (?P<col2>[-+]?\d+)   # hybrid parameter : integer
   \s+             # skip white space
   (?P<col3>[\S\s]*)   # list of clusters
   """,re.VERBOSE)
  infile=open(clusterfilename,'r')
  all=infile.readlines()
  infile.close()
  for eachline in all:
    v=pp.search(eachline)
    if v!= None:
       # iterate over list of source names (names can also have a '.')
       CL[str(v.group('col1'))]=re.split('[^a-zA-Z0-9_\.]+',re.sub('\n','',str(v.group('col3'))))

  print 'Read %d clusters'%len(CL)


  # region file
  outfile=open(outfilename,'w+')

  # which clusters to annotate
  if CL.has_key(str(clid)):
    annlist=(str(clid),)
  else:
    annlist=CL.keys()

  outfile.write('# Region file format: DS9 version 4.1\n')
  outfile.write('global color=blue dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
  
  for clname in annlist:
    clinfo=CL[clname]
    for slname in clinfo:
      if SR.has_key(slname):
        sinfo=SR[slname]
        if rname:
          sline='fk5;point('+str(sinfo[0])+','+str(sinfo[1])+') # point=x color='+color+' text={'+slname+'}\n'
        else: 
          sline='fk5;point('+str(sinfo[0])+','+str(sinfo[1])+') # point=x color='+color+' text={'+clname+'}\n'
        outfile.write(sline)
  outfile.close()


if __name__ == '__main__':
  import sys
  parser=optparse.OptionParser()
  parser.add_option('-s', '--skymodel', help='Input sky model')
  parser.add_option('-c', '--clusters', help='Input cluster file')
  parser.add_option('-o', '--outfile', help='Output DS9 region file')
  parser.add_option('-n', '--names', help='Output source names (default is cluster id)', dest='rname', action='store_true')
  parser.add_option('-i', '--id', type='int', dest='num', help='Cluster id to annotate (default all clusters)')
  parser.add_option('-C', '--color', help='Colour white|black|red|green|blue|cyan|magenta|yellow', default='yellow')
  (opts,args)=parser.parse_args()

  if opts.skymodel and opts.clusters and opts.outfile:
    annotate_lsm_sky(opts.skymodel,opts.clusters,opts.outfile,opts.num,opts.color,opts.rname)
  else:
   parser.print_help()
  exit()
