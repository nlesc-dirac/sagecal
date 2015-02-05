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


def convert_sky_bbs_lsm(infilename,outfilename):
 # BBS format parser
 # P1C1, POINT, PATCH, 14:16:57.07, +50.57.57.51, 0.406232, 0.0, 0.0, 0.0
 # P1C1, POINT, PATCH, 14:16:57.07, +50.57.57.51, 0.406232, 0.0, 0.0, 0.0,  52708393.061927, [0.040956]
 # P1C1, GAUSSIAN, PATCH, 14:16:57.07, +50.57.57.51, 0.406232, 0.0, 0.0, 0.0, MajorAxis, MinorAxis, Orientation,  52708393.061927, [0.040956]
 # P1C1, GAUSSIAN, 14:16:57.07, +50.57.57.51, 0.406232, 0.0, 0.0, 0.0, MajorAxis, MinorAxis, Orientation,  52708393.061927, [0.040956]
 # Also GSM formats:
 # 1602.3+8016, GAUSSIAN, 16:02:20.64000000, +80.16.00.40800000, 22.9939, , , , , [-0.8319, -0.116], 47.0, 38.3, 125.9
 # or
 # 1714.2+7612, POINT, 17:14:16.97040000, +76.12.43.88400000, 10.2048, , , , , [-0.8936, -0.0674]
 # note spectra [] and Gaussian parameters might be missing
 pp=re.compile(r"""
   ^(?P<col1>[A-Za-z0-9_.]+)  # column 1 name: must start with a character
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col2>\S+)  # source type i.e. 'POINT'
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col3>\S+)  # patch name i.e. 'CENTER'
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col4>\d+)   # RA angle - hr 
   \:           # skip colon
   (?P<col5>\d+)   # RA angle - min 
   \:           # skip colon
   (?P<col6>\d+(\.\d+)?)   # RA angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col7>[-+]?\d+)   # Dec angle - hr 
   \.           # skip dot 
   (?P<col8>\d+)   # Dec angle - min 
   \.           # skip dot
   (?P<col9>\d+(\.\d+)?)   # Dec angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col10>[-+]?\d+(\.\d+)?)   # sI
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col11>[-+]?\d+(\.\d+)?)   # sQ
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col12>[-+]?\d+(\.\d+)?)   # sU
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col13>[-+]?\d+(\.\d+)?)   # sV
   [\S\s]*""",re.VERBOSE)

 pp1=re.compile(r"""
   ^(?P<col1>[A-Za-z0-9_.]+)  # column 1 name: must start with a character
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col2>\S+)  # source type i.e. 'POINT'
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col3>\S+)  # patch name i.e. 'CENTER'
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col4>\d+)   # RA angle - hr 
   \:           # skip colon
   (?P<col5>\d+)   # RA angle - min 
   \:           # skip colon
   (?P<col6>\d+(\.\d+)?)   # RA angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col7>[-+]?\d+)   # Dec angle - hr 
   \.           # skip dot 
   (?P<col8>\d+)   # Dec angle - min 
   \.           # skip dot
   (?P<col9>\d+(\.\d+)?)   # Dec angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col10>[-+]?\d+(\.\d+)?)   # sI
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col11>[-+]?\d+(\.\d+)?)   # sQ
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col12>[-+]?\d+(\.\d+)?)   # sU
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col13>[-+]?\d+(\.\d+)?)   # sV
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col14>\d+(\.\d+)?([eE](-|\+)(\d+))?)   # freq0
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   \[             # skip [
   \s*             # skip white space
   (?P<col15>[-+]?\d+(\.\d+)?)   # [spec_index] 
   \s*             # skip white space
   \]             # skip ]
   [\S\s]*""",re.VERBOSE)

 pp2=re.compile(r"""
   ^(?P<col1>[A-Za-z0-9_.]+)  # column 1 name: must start with a character
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col2>\S+)  # source type i.e. 'POINT', 'GAUSSIAN'
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col3>\S+)  # patch name i.e. 'CENTER'
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col4>\d+)   # RA angle - hr 
   \:           # skip colon
   (?P<col5>\d+)   # RA angle - min 
   \:           # skip colon
   (?P<col6>\d+(\.\d+)?)   # RA angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col7>[-+]?\d+)   # Dec angle - hr 
   \.           # skip dot 
   (?P<col8>\d+)   # Dec angle - min 
   \.           # skip dot
   (?P<col9>\d+(\.\d+)?)   # Dec angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col10>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sI
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col11>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sQ
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col12>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sU
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col13>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sV
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col14>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # bMaj
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col15>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # bMin
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col16>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # bPA
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col17>\d+(\.\d+)?([eE](-|\+)(\d+))?)   # freq0
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   \[             # skip [
   \s*             # skip white space
   (?P<col18>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # [spec_index] 
   \s*             # skip white space
   \]             # skip ]
   [\S\s]+""",re.VERBOSE)

 pp3=re.compile(r"""
   ^(?P<col1>[A-Za-z0-9_.]+)  # column 1 name: must start with a character
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col2>\S+)  # source type i.e. 'POINT', 'GAUSSIAN'
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col4>\d+)   # RA angle - hr 
   \:           # skip colon
   (?P<col5>\d+)   # RA angle - min 
   \:           # skip colon
   (?P<col6>\d+(\.\d+)?)   # RA angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col7>[-+]?\d+)   # Dec angle - hr 
   \.           # skip dot 
   (?P<col8>\d+)   # Dec angle - min 
   \.           # skip dot
   (?P<col9>\d+(\.\d+)?)   # Dec angle - sec
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col10>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sI
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col11>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sQ
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col12>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sU
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col13>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # sV
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col14>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # bMaj
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col15>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # bMin
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col16>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # bPA
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   (?P<col17>\d+(\.\d+)?([eE](-|\+)(\d+))?)   # freq0
   \s*             # skip white space
   \,           # skip comma
   \s*             # skip white space
   \[             # skip [
   \s*             # skip white space
   (?P<col18>[-+]?\d+(\.\d+)?([eE](-|\+)(\d+))?)   # [spec_index] 
   \s*             # skip white space
   \]             # skip ]
   [\S\s]+""",re.VERBOSE)



 infile=open(infilename,'r')
 all=infile.readlines()
 infile.close()
 outfile=open(outfilename,'w')
 outfile.write("## LSM file\n")
 outfile.write("### Name  | RA (hr,min,sec) | DEC (deg,min,sec) | I | Q | U |  V | SI | RM | eX | eY | eP | freq0\n")
 for eachline in all:
   v=pp1.search(eachline)
   if v!= None:
      strline=str(v.group('col1'))+' '+str(v.group('col4'))+' '+str(v.group('col5'))+' '+str(v.group('col6'))
      strline=strline+' '+str(v.group('col7'))+' '+str(v.group('col8'))+' '+str(v.group('col9'))+' '+str(v.group('col10'))
      strline=strline+' '+str(v.group('col11'))+' '+str(v.group('col12'))+' '+str(v.group('col13'))+' '+str(v.group('col15'))
      strline=strline+' 0 0 0 0 '+str(v.group('col14'))+'\n'
      outfile.write(strline)
   else:
     v=pp.search(eachline)
     if v!= None:
      strline=str(v.group('col1'))+' '+str(v.group('col4'))+' '+str(v.group('col5'))+' '+str(v.group('col6'))
      strline=strline+' '+str(v.group('col7'))+' '+str(v.group('col8'))+' '+str(v.group('col9'))+' '+str(v.group('col10'))
      strline=strline+' 0 0 0 0 0 0 0 0\n'
      outfile.write(strline)
     else:
       v=pp2.search(eachline)
       if v!= None:
        stype=v.group('col2')
        sname=str(v.group('col1'))
        bad_source=False
        if stype=='GAUSSIAN':
          sname='G'+sname
        strline=sname+' '+str(v.group('col4'))+' '+str(v.group('col5'))+' '+str(v.group('col6'))
        strline=strline+' '+str(v.group('col7'))+' '+str(v.group('col8'))+' '+str(v.group('col9'))+' '+str(v.group('col10'))
        strline=strline+' '+str(v.group('col11'))+' '+str(v.group('col12'))+' '+str(v.group('col13'))+' '+str(v.group('col18'))
        # need the right conversion factor for Gaussians: Bmaj,Bmin arcsec (rad) diameter, PA (deg) West-> counterclockwise, BDSM its North->counterclockwise
        if stype=='GAUSSIAN':
          bmaj=float(v.group('col14'))*(0.5/3600)*math.pi/180.0
          bmin=float(v.group('col15'))*(0.5/3600)*math.pi/180.0
          bpa=math.pi/2-(math.pi-float(v.group('col16'))/180.0*math.pi)
          # also throw away bad Gaussians with zero bmaj or bmin
          if bmaj<1e-6 or bmin<1e-6:
            bad_source=True
        else:
          bmaj=0.0
          bmin=0.0
          bpa=0.0
        
        strline=strline+' 0 '+str(bmaj)+' '+str(bmin)+' '+str(bpa)
        strline=strline+' '+str(v.group('col17'))+'\n'
        # only write good sources
        if not bad_source:
          outfile.write(strline)
        else:
          print 'Error in source '+strline
       else:
         v=pp3.search(eachline)
         if v!= None:
          stype=v.group('col2')
          sname=str(v.group('col1'))
          bad_source=False
          if stype=='GAUSSIAN':
            sname='G'+sname
          strline=sname+' '+str(v.group('col4'))+' '+str(v.group('col5'))+' '+str(v.group('col6'))
          strline=strline+' '+str(v.group('col7'))+' '+str(v.group('col8'))+' '+str(v.group('col9'))+' '+str(v.group('col10'))
          strline=strline+' '+str(v.group('col11'))+' '+str(v.group('col12'))+' '+str(v.group('col13'))+' '+str(v.group('col18'))
          # need the right conversion factor for Gaussians: Bmaj,Bmin arcsec (rad) diameter, PA (deg) West-> counterclockwise, BDSM its North->counterclockwise
          if stype=='GAUSSIAN':
            bmaj=float(v.group('col14'))*(0.5/3600.0)*math.pi/180.0
            bmin=float(v.group('col15'))*(0.5/3600.0)*math.pi/180.0
            bpa=math.pi/2-(math.pi-float(v.group('col16'))/180.0*math.pi)
            # also throw away bad Gaussians with zero bmaj or bmin
            if bmaj<1e-6 or bmin<1e-6:
              bad_source=True
          else:
            bmaj=0.0
            bmin=0.0
            bpa=0.0
        
          strline=strline+' 0 '+str(bmaj)+' '+str(bmin)+' '+str(bpa)
          strline=strline+' '+str(v.group('col17'))+'\n'
          # only write good sources
          if not bad_source:
            outfile.write(strline)
          else:
            print 'Error in source '+strline
         else:
            splitfields=eachline.split(',');
            if len(splitfields)>10:
             # type
             if 'GAUSSIAN' in splitfields[1]:
              # read last 3 fields for shape
              bpa=splitfields.pop()
              bmin=splitfields.pop()
              bmaj=splitfields.pop()
              # convert to right format
              bmaj=float(bmaj)*(0.5/3600.0)*math.pi/180.0
              bmin=float(bmin)*(0.5/3600.0)*math.pi/180.0
              bpa=math.pi/2-(math.pi-float(bpa)/180.0*math.pi)
              sname='G'+splitfields[0]
             else:
              sname='P'+splitfields[0]
              bmaj=0
              bmin=0
              bpa=0
             # read in 2nd,3rd field for RA,DEC
             raline=splitfields[2].split(':')
             decline=splitfields[3].split('.')
             # finally sI
             sI=splitfields[4]
             strline=sname+' '+str(int(raline[0]))+' '+str(int(raline[1]))+' '+str(float(raline[2]))+' '+str(int(decline[0]))+' '+str(int(decline[1]))+' '+str(float(decline[2]+'.'+decline[3]))+' '+str(float(sI))
             # add missing fields (Q U V SI RM)
             strline=strline+' 0 0 0 0 0 '+str(bmaj)+' '+str(bmin)+' '+str(bpa)+' 150e6\n'
             outfile.write(strline)

def convert_sky_lsm_bbs(infilename,outfilename):
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
   (?P<col8>[-+]?\d+(\.\d+)?)   # Stokes I - Flux
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
   (?P<col8>[-+]?\d+(\.\d+)?)   # Stokes I - Flux
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
  outfile=open(outfilename,'w')
  outfile.write("# (Name, Type, Patch, Ra, Dec, I, Q, U, V, ReferenceFrequency='150e6',  SpectralIndex='[0.0]', Ishapelet) = format\n")
  outfile.write("# The above line defines the field order and is required.\n")
  outfile.write(", , CENTER, put:ra:here, put.dec.here\n") 

  for eachline in all:
    v=pp.search(eachline)
    if v!= None:
     # check for /GAUSSIANS
     firstchar=v.group('col1')[0]
     if firstchar=='G' or firstchar=='g':
       strline=str(v.group('col1'))+', GAUSSIAN, CENTER, '+str(v.group('col2'))+':'+str(v.group('col3'))+':'+str(v.group('col4'))
       strline=strline+', '+str(v.group('col5'))+'.'+str(v.group('col6'))+'.'+str(v.group('col7'))+', '+str(v.group('col8'))+', '+str(v.group('col9'))+', '+str(v.group('col10'))+', '+str(v.group('col11'))+', '
       strline=strline+str(v.group('col17'))+', ['+str(v.group('col12'))+']\n'
     else:  
       strline=str(v.group('col1'))+', POINT, CENTER, '+str(v.group('col2'))+':'+str(v.group('col3'))+':'+str(v.group('col4'))
       strline=strline+', '+str(v.group('col5'))+'.'+str(v.group('col6'))+'.'+str(v.group('col7'))+', '+str(v.group('col8'))+', '+str(v.group('col9'))+', '+str(v.group('col10'))+', '+str(v.group('col11'))+', '
       strline=strline+str(v.group('col17'))+', ['+str(v.group('col12'))+']\n'
     outfile.write(strline)


if __name__ == '__main__':
  import sys
  parser=optparse.OptionParser()
  parser.add_option('-i', '--infile', help='Input sky model')
  parser.add_option('-o', '--outfile', help='Output sky model (overwritten!)')
  parser.add_option('-b', '--bbstolsm',action="store_true", default=False, help='BBS to LSM')
  parser.add_option('-l', '--lsmtobbs',action="store_true", default=False, help='LSM to BBS')
  (opts,args)=parser.parse_args()

  if opts.bbstolsm and opts.infile and opts.outfile:
    convert_sky_bbs_lsm(opts.infile,opts.outfile)
  elif opts.lsmtobbs and opts.infile and opts.outfile:
   convert_sky_lsm_bbs(opts.infile,opts.outfile)
  else:
   parser.print_help()
  exit()

