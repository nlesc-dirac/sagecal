/*
 *
 Copyright (C) 2016 Sarod Yatawatta <sarod@users.sf.net>  
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 $Id$
 */



#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Radio.h"


/* 
  ra,dec: source direction (rad)
  ra0,dec0: beam center (rad)
  f: frequency (Hz)
  f0: beam forming frequency (Hz)
  
  longitude,latitude : Nx1 array of station positions (rad,rad)
  time_jd: JD (day) time
  Nelem : Nx1 array, no. of elements used in each station
  x,y,z: Nx1 pointer arrays to station positions, each station has Nelem[]x1 arrays

  beamgain: Nx1 array of station beam gain along the source direction
*/ 
int
arraybeam(double ra, double dec, double ra0, double dec0, double f, double f0, int N, double *longitude, double *latitude, double time_jd, int *Nelem, double **x, double **y, double **z, double *beamgain) {

  double gmst;
  jd2gmst(time_jd,&gmst); /* JD (day) to GMST (deg) */
  int ci,cj,K;
  double az,el,az0,el0;
  double theta,phi,theta0,phi0;
  double *px,*py,*pz;
  double r1,r2,r3;
  double sint,cost,sint0,cost0,sinph,cosph,sinph0,cosph0;
  double csum,ssum,tmpc,tmps;
  /* 2*PI/C */
  const double tpc=2.0*M_PI/CONST_C;

  /* iterate over stations */
  for (ci=0; ci<N; ci++) {
   /* find az,el for both source direction and beam center */
   radec2azel_gmst(ra,dec, longitude[ci], latitude[ci], gmst, &az, &el);
   radec2azel_gmst(ra0,dec0, longitude[ci], latitude[ci], gmst, &az0, &el0);
   /* transform : theta = 90-el, phi=-az? 45 only needed for element beam */
   theta=M_PI_2-el;
   phi=-az; /* */
   theta0=M_PI_2-el0;
   phi0=-az0; /* */

   if (el>=0.0) {
   K=Nelem[ci];
   px=x[ci]; 
   py=y[ci]; 
   pz=z[ci]; 

   sincos(theta,&sint,&cost);
   sincos(phi,&sinph,&cosph);
   sincos(theta0,&sint0,&cost0);
   sincos(phi0,&sinph0,&cosph0);

   /*r1=f0*sint0*cosph0-f*sint*cosph;
   r2=f0*sint0*sinph0-f*sint*sinph;
   r3=f0*cost0-f*cost;
   */

   /* try to improve computations */
   double rat1=f0*sint0;
   double rat2=f*sint;
   r1=(rat1*cosph0-rat2*cosph);
   r2=(rat1*sinph0-rat2*sinph);
   r3=(f0*cost0-f*cost);
   
   csum=0.0;
   ssum=0.0;
   for (cj=0; cj<K; cj++) {
     sincos(-tpc*(r1*px[cj]+r2*py[cj]+r3*pz[cj]),&tmps,&tmpc);
     ssum+=tmps;
     csum+=tmpc;
   }
   double invK=1.0/(double)K;
   csum*=invK;
   ssum*=invK;

   /* beam gain is | |, only for +ve elevation */
   beamgain[ci]=sqrt(csum*csum+ssum*ssum);
   } else {
    beamgain[ci]=0.0;
   }
  }
 

  return 0;
}
