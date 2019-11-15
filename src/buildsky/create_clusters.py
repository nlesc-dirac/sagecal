#!/usr/bin/env python
#
# Copyright (C) 2016- Sarod Yatawatta <sarod@users.sf.net>
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
import math,numpy
import optparse


# read sky model text file and return dictionary
# the dictionary will contain NAME,RA,DEC,sI only, (things needed to cluster)
def read_lsm_sky(infilename):
  # LSM format
  # NAME RA(hours, min, sec) DEC(degrees, min, sec) sI sQ sU sV SI RM eX eY eP f0
  # or 3rd order spectra
  # NAME RA(hours, min, sec) DEC(degrees, min, sec) sI sQ sU sV SI0 SI1 SI2 RM eX eY eP f0

  # regexp pattern
  pp=re.compile(r"""
   ^(?P<col1>[A-Za-z0-9_.-/+]+)  # column 1 name: must start with a character
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
   ^(?P<col1>[A-Za-z0-9_.-/+]+)  # column 1 name: must start with a character
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
      mra=mysign*(abs(float(v.group('col2')))+float(v.group('col3'))/60.0+float(v.group('col4'))/3600.0)*360.0/24.0*math.pi/180.0
      if (float(v.group('col5'))) >= 0.0:
         mysign=1.0
      else:
         mysign=-1.0
      mdec=mysign*(abs(float(v.group('col5')))+float(v.group('col6'))/60.0+float(v.group('col7'))/3600.0)*math.pi/180.0
      SR[str(v.group('col1'))]=(v.group('col1'),mra,mdec,float(v.group('col8')))
    else:  
      v=pp.search(eachline)
      if v!= None:
        if (float(v.group('col2'))) >= 0.0:
           mysign=1.0
        else:
           mysign=-1.0

        mra=mysign*(abs(float(v.group('col2')))+float(v.group('col3'))/60.0+float(v.group('col4'))/3600.0)*360.0/24.0*math.pi/180.0
        if (float(v.group('col5'))) >= 0.0:
           mysign=1.0
        else:
           mysign=-1.0


        mdec=mysign*(abs(float(v.group('col5')))+float(v.group('col6'))/60.0+float(v.group('col7'))/3600.0)*math.pi/180.0
        SR[str(v.group('col1'))]=(v.group('col1'),mra,mdec,float(v.group('col8')))

  print('Read %d sources'%len(SR))

  return SR


# find closet cluster for this source ra,dec
# C: ra,dec of cluster centroids
# Ccos: cos(C), Csin: sin(C)
def find_closest(ra,dec,C,Ccos,Csin):
   sin_deltaalpha=numpy.sin(C[:,0]-ra)
   cos_deltaalpha=numpy.cos(C[:,0]-ra)

   sin_delta_a=math.sin(dec)
   cos_delta_a=math.cos(dec)
   denominator=sin_delta_a*Csin+cos_delta_a*numpy.multiply(Ccos,cos_deltaalpha)
   numerator=numpy.square(numpy.multiply(Ccos,sin_deltaalpha)) + numpy.square(cos_delta_a*Csin-sin_delta_a*numpy.multiply(Ccos,cos_deltaalpha))
   d=numpy.arctan2(numpy.sqrt(numerator),denominator)
   idx=numpy.argmin(numpy.abs(d))
   return idx


# convert lm coords to ra,dec
def lm_to_radec(ra0,dec0,l,m):
    sind0=math.sin(dec0)
    cosd0=math.cos(dec0)
    dl=l
    dm=m
    d0=dm*dm*sind0*sind0+dl*dl-2*dm*cosd0*sind0
    sind=math.sqrt(abs(sind0*sind0-d0))
    cosd=math.sqrt(abs(cosd0*cosd0+d0))
    if (sind0>0.0):
     sind=abs(sind)
    else:
     sind=-abs(sind)

    dec=math.atan2(sind,cosd)

    if l!=0.0:
     ra=math.atan2(-dl,(cosd0-dm*sind0))+ra0
    else:
     ra=math.atan2((1e-10),(cosd0-dm*sind0))+ra0


    return (ra,dec)

## convert ra,dec (arrays) to lm arrays (SIN projection)
def radec_to_lm_SIN(ra0,dec0,ra,dec):
    sin_alpha=numpy.sin(ra-ra0)
    cos_alpha=numpy.cos(ra-ra0)
    sin_dec=numpy.sin(dec)
    cos_dec=numpy.cos(dec)
    #l=-math.sin(ra-ra0)*math.cos(dec)
    #m=-(math.cos(ra-ra0)*math.cos(dec)*math.sin(dec0)-math.cos(dec0)*math.sin(dec))
    l=-numpy.multiply(sin_alpha,cos_dec)
    m=-math.sin(dec0)*numpy.multiply(cos_alpha,cos_dec)+math.cos(dec0)*sin_dec
    return (l,m)




#### main clustering routine : Q clusters
def cluster_this(skymodel,Q,outfile,max_iterations=5):
   SKY=read_lsm_sky(skymodel)
   K=len(SKY)

   # negative_cluster_ids will keep a record of the user's request for negative cluster ids,
   # The user can indicate this  by entering a negative number of clusters.
   if Q<0:
       negative_cluster_ids=True
       Q=-Q
   else:
       negative_cluster_ids=False

   # check if we have more sources than clusters, otherwise change Q
   if Q>K:
     Q=K

   # create arrays for all source info (ra,dec,sI) for easy access
   X=numpy.zeros([K,3])
   # iterate over sources
   ci=0;
   for val in SKY.values():
      X[ci,0]=val[1]
      X[ci,1]=val[2]
      X[ci,2]=val[3]
      ci=ci+1
   # source names
   sources=list(SKY.keys())
   # centroids of Q clusters
   C=numpy.zeros([Q,2])
   # 1: select the Q brightest sources, initialize cluster centroids as their locations
   sItmp=numpy.copy(X[:,2])
   for ci in range(0,Q):
      # find max 
      sImax=numpy.argmax(sItmp)
      C[ci,0]=X[sImax,0]
      C[ci,1]=X[sImax,1]
      sItmp[sImax]=0.0
   #print(C)
   # calculate weights

   # arrays to store which cluster each source belongs to
   CL=numpy.zeros(K)
   CLold=numpy.copy(CL)

   no_more_cluster_changes=False
   niter=1
   # loop
   while (not no_more_cluster_changes) and niter<max_iterations:
      Ccos=numpy.cos(C[:,1])
      Csin=numpy.sin(C[:,1])


      D={} # empty dict
      # 2:  assign each source to the cluster closest to it 
      for ci in range(0,K):
          mra=X[ci,0]
          mdec=X[ci,1]
          closest=find_closest(mra,mdec,C,Ccos,Csin)
          #print("src %d closest %d"%(ci,closest))
          CL[ci]=closest
          # add this source to dict
          if closest in D:
              D[closest].append(ci)
          else:
              D[closest]=list()
              D[closest].append(ci)
      #print(D)
      
      # check to see also if source assignment changes
      if numpy.sum(CL-CLold)==0:
       no_more_cluster_changes=True

      CLold=numpy.copy(CL)
  
      # 3: update the  cluster centroids
      for (clusid,sourcelist) in list(D.items()):
        # update centroid of cluster id 'clusid'
        # project soure ra,dec coordinates to l,m with center of projection
        # taken as current centroid, then take the weighted average 
        #print clusid,sourcelist
        aRa=numpy.zeros(len(sourcelist))
        aDec=numpy.zeros(len(sourcelist))
        aW=numpy.zeros(len(sourcelist))
        ck=0
        for sourceid in sourcelist:
          aRa[ck]=X[sourceid,0] 
          aDec[ck]=X[sourceid,1] 
          aW[ck]=X[sourceid,2] 
          #print sources[sourceid],aRa[ck],aDec[ck],aW[ck]
          ck=ck+1
        ### check
        ###(l1,m1)=radec_to_lm_SIN(C[clusid,0],C[clusid,1],aRa[0],aDec[0])
        ###(ra1,dec1)=lm_to_radec(C[clusid,0],C[clusid,1],l1,m1)
        ###print "CHECK %f,%f -> %f,%f"%(aRa[0],aDec[0],ra1,dec1)
        (L,M)=radec_to_lm_SIN(C[clusid,0],C[clusid,1],aRa,aDec)
        #print L,M
        sumsI=numpy.sum(aW)
        Lmean=numpy.sum(numpy.multiply(aW,L))/sumsI
        Mmean=numpy.sum(numpy.multiply(aW,M))/sumsI
        (ra1,dec1)=lm_to_radec(C[clusid,0],C[clusid,1],Lmean,Mmean)
        ##sumsI=numpy.sum(aW)
        ##ra1=numpy.sum(numpy.multiply(aW,aRa))/sumsI
        ##dec1=numpy.sum(numpy.multiply(aW,aDec))/sumsI
        #print "Cen %f,%f -> %f,%f"%(C[clusid,0],C[clusid,1],ra1,dec1)
        # update centroid
        C[clusid,0]=ra1
        C[clusid,1]=dec1
      niter=niter+1
   
   if no_more_cluster_changes:
     print("Stopping after "+str(niter)+" iterations because cluster geometry did not change.")
   # write output
   outF=open(outfile,'w+')
   outF.write('# Cluster file\n')
   for (clusid,sourcelist) in list(D.items()):
     if negative_cluster_ids==False:
         outF.write(str(clusid+1)+' 1')
     else:
         outF.write(str(-clusid-1)+' 1')
     for sourceid in sourcelist:
       outF.write(' '+sources[sourceid])
     outF.write('\n')
   outF.close()

if __name__ == '__main__':
  import sys
  parser=optparse.OptionParser()
  parser.add_option('-s', '--skymodel', help='Input sky model')
  parser.add_option('-c', '--clusters', type='int', help='Number of clusters. Absolute value if negative and the cluster ids will be negative.')
  parser.add_option('-o', '--outfile', help='Output cluster file')
  parser.add_option('-i', '--iterations', type='int', help='Number of iterations')
  (opts,args)=parser.parse_args()

  if opts.skymodel and opts.clusters and opts.outfile and opts.iterations:
    cluster_this(opts.skymodel,opts.clusters,opts.outfile,opts.iterations)
  elif opts.skymodel and opts.clusters and opts.outfile:
    cluster_this(opts.skymodel,opts.clusters,opts.outfile)
  else:
   parser.print_help()
  exit()
