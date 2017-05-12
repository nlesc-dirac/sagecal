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
 convert xyz ITRF 2000 coords (m) to
 long,lat, (rad) height (m)
 References: xyz2llh.m MATLAB routine
 Also : Hoffmann-Wellenhof, B., Lichtenegger, H. and J. Collins (1997). GPS.
   Theory and Practice. 4th revised edition. Springer, New York, pp. 389
*/

int
xyz2llh(double *x, double *y, double *z, double *longitude, double *latitude, double *height, int N) {
 /* constants */
 double a=6378137.0; /* semimajor axis */
 double f=1.0/298.257223563; /* flattening */
 double b=(1.0-f)*a; /* semiminor axis */
 double e2=2*f-f*f; /* exxentricity squared */
 double ep2=(a*a-b*b)/(b*b); /* second numerical eccentricity */
 double *p,*theta;
 if ((p=(double*)calloc((size_t)N,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
 }
 if ((theta=(double*)calloc((size_t)N,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
 }

 int ci;

 for (ci=0; ci<N; ci++) {
  p[ci]=sqrt(x[ci]*x[ci]+y[ci]*y[ci]); /* radius of parallel */
  /* longitude */
  /* change sign such that E longitude is positive */
  longitude[ci]=atan2(y[ci],x[ci]);
 }

 for (ci=0; ci<N; ci++) {
  theta[ci]=atan(z[ci]*a/(p[ci]*b));
 }
 /* latitude */
 for (ci=0; ci<N; ci++) {
  double stheta,ctheta;
  sincos(theta[ci],&stheta,&ctheta);
  latitude[ci]=atan((z[ci]+ep2*b*stheta*stheta*stheta)/(p[ci]-e2*a*ctheta*ctheta*ctheta));
 }
 /* radius of curvature */
 for (ci=0; ci<N; ci++) {
   double stheta,ctheta;
   sincos(latitude[ci],&stheta,&ctheta);
   double r=a/sqrt(1.0-e2*stheta*stheta);
   /* height (m) */
   height[ci]=p[ci]/ctheta-r;
 }
/* for (ci=0; ci<N; ci++) {
printf("%lf %lf %lf %lf %lf %lf\n",x[ci],y[ci],z[ci],longitude[ci]*180.0/M_PI,latitude[ci]*180.0/M_PI,height[ci]);
 }
*/
 free(p);
 free(theta);
 return 0;
}



/* convert ra,dec to az,el
   ra,dec: radians
   longitude,latitude: rad,rad 
   time_jd: JD days

   az,el: output  rad,rad

References: Darin C. Koblick MATLAB code, based on
  % Fundamentals of Astrodynamics and Applications 
 % D. Vallado, Second Edition
 % Example 3-5. Finding Local Siderial Time (pg. 192) 
 % Algorithm 28: AzElToRaDec (pg. 259)
*/
int
radec2azel(double ra, double dec, double longitude, double latitude, double time_jd, double *az, double *el) {
  double t_ut1=(time_jd-2451545.0)/36525.0;
  double thetaGMST=67310.54841 + (876600.0*3600.0 + 8640184.812866)*t_ut1 + 0.093104*(t_ut1*t_ut1)-(6.2*10e-6)*(t_ut1*t_ut1*t_ut1);
  thetaGMST = fmod((fmod(thetaGMST,86400.0*(thetaGMST/fabs(thetaGMST)))/240.0),360.0);
  double thetaLST=thetaGMST+longitude*180.0*M_1_PI;

  double LHA=fmod(thetaLST-ra*180.0*M_1_PI,360.0);
  
  double sinlat,coslat,sindec,cosdec,sinLHA,cosLHA;
  sincos(latitude,&sinlat,&coslat);
  sincos(dec,&sindec,&cosdec);
  sincos(LHA*M_PI/180.0,&sinLHA,&cosLHA);

  double tmp=sinlat*sindec+coslat*cosdec*cosLHA;
  *el=asin(tmp);

  double sinel,cosel;
  sincos(*el,&sinel,&cosel);

  *az=fmod(atan2(-sinLHA*cosdec/cosel,(sindec-sinel*sinlat)/(cosel*coslat)),2.0*M_PI);
  if (*az<0) {
   *az+=2.0*M_PI;
  }

printf("%lf %lf %lf %lf %lf %lf %lf\n",ra,dec,longitude,latitude,time_jd,*az,*el);
 return 0;
}



/* convert time to Greenwitch Mean Sideral Angle (deg)
   time_jd : JD days
   thetaGMST : GMST angle (deg)
*/
int
jd2gmst(double time_jd, double *thetaGMST) {
  double t_ut1=(time_jd-2451545.0)/36525.0;
  //double theta=67310.54841 + (876600.0*3600.0 + 8640184.812866)*t_ut1 + 0.093104*(t_ut1*t_ut1)-(6.2*10e-6)*(t_ut1*t_ut1*t_ut1);
  /* use Horners rule */
  double theta=67310.54841 + t_ut1*((876600.0*3600.0 + 8640184.812866) + t_ut1*(0.093104-(6.2*10e-6)*(t_ut1)));
  *thetaGMST = fmod((fmod(theta,86400.0*(theta/fabs(theta)))/240.0),360.0);
  return 0;
}

/* convert ra,dec to az,el
   ra,dec: radians
   longitude,latitude,: rad,rad 
   thetaGMST : GMST angle (deg)

   az,el: output  rad,rad

*/
int
radec2azel_gmst(double ra, double dec, double longitude, double latitude, double thetaGMST, double *az, double *el) {
  double thetaLST=thetaGMST+longitude*180.0*M_1_PI;

  double LHA=fmod(thetaLST-ra*180.0*M_1_PI,360.0);
  
  double sinlat,coslat,sindec,cosdec,sinLHA,cosLHA;
  sincos(latitude,&sinlat,&coslat);
  sincos(dec,&sindec,&cosdec);
  sincos(LHA*M_PI/180.0,&sinLHA,&cosLHA);

  double tmp=sinlat*sindec+coslat*cosdec*cosLHA;
  *el=asin(tmp);

  double sinel,cosel;
  sincos(*el,&sinel,&cosel);

  *az=fmod(atan2(-sinLHA*cosdec/cosel,(sindec-sinel*sinlat)/(cosel*coslat)),2.0*M_PI);
  if (*az<0) {
   *az+=2.0*M_PI;
  }

//printf("%lf %lf %lf %lf %lf %lf %lf\n",ra,dec,longitude,latitude,thetaGMST,*az,*el);
 return 0;
}




/* given the epoch jd_tdb2, 
 calculate rotation matrix params needed to precess from J2000 
   PURPOSE:
      Precesses equatorial rectangular coordinates from one epoch to
      another.  One of the two epochs must be J2000.0.  The coordinates
      are referred to the mean dynamical equator and equinox of the two
      respective epochs.

   REFERENCES:
      Explanatory Supplement To The Astronomical Almanac, pp. 103-104.
      Capitaine, N. et al. (2003), Astronomy And Astrophysics 412,
         pp. 567-586.
      Hilton, J. L. et al. (2006), IAU WG report, Celest. Mech., 94,
         pp. 351-367.

*/
int
get_precession_params(double jd_tdb2, double Tr[9]) {
   double eps0 = 84381.406;
   double jd_tdb1=2451545.0; /* J2000 */
   double  t, psia, omegaa, chia, sa, ca, sb, cb, sc, cc, sd, cd;

/*
   't' is time in TDB centuries between the two epochs.
*/

   t = (jd_tdb2 - jd_tdb1) / 36525.0;



/*
   Numerical coefficients of psi_a, omega_a, and chi_a, along with
   epsilon_0, the obliquity at J2000.0, are 4-angle formulation from
   Capitaine et al. (2003), eqs. (4), (37), & (39).
*/

      psia   = ((((-    0.0000000951  * t
                   +    0.000132851 ) * t
                   -    0.00114045  ) * t
                   -    1.0790069   ) * t
                   + 5038.481507    ) * t;

      omegaa = ((((+    0.0000003337  * t
                   -    0.000000467 ) * t
                   -    0.00772503  ) * t
                   +    0.0512623   ) * t
                   -    0.025754    ) * t + eps0;

      chia   = ((((-    0.0000000560  * t
                   +    0.000170663 ) * t
                   -    0.00121197  ) * t
                   -    2.3814292   ) * t
                   +   10.556403    ) * t;

      eps0 = eps0 * ASEC2RAD;
      psia = psia * ASEC2RAD;
      omegaa = omegaa * ASEC2RAD;
      chia = chia * ASEC2RAD;

      sincos(eps0,&sa,&ca);
      sincos(-psia,&sb,&cb);
      sincos(-omegaa,&sc,&cc);
      sincos(chia,&sd,&cd);
/*
   Compute elements of precession rotation matrix equivalent to
   R3(chi_a) R1(-omega_a) R3(-psi_a) R1(epsilon_0).
*/

      Tr[0] =  cd * cb - sb * sd * cc;
      Tr[3] =  cd * sb * ca + sd * cc * cb * ca - sa * sd * sc;
      Tr[6] =  cd * sb * sa + sd * cc * cb * sa + ca * sd * sc;
      Tr[1] = -sd * cb - sb * cd * cc;
      Tr[4] = -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc;
      Tr[7] = -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc;
      Tr[2] =  sb * sc;
      Tr[5] = -sc * cb * ca - sa * cc;
      Tr[8] = -sc * cb * sa + cc * ca;

  return 0;
}
/* precess  ra0,dec0 at J2000
   to ra,dec at epoch given by transform Tr
 using NOVAS library */
int 
precession(double ra0, double dec0, double Tr[9], double *ra, double *dec) {
    
    double pos1[3],pos2[3];
    pos1[0]=cos(ra0)*sin(dec0);
    pos1[1]=sin(ra0)*sin(dec0);
    pos1[2]=cos(dec0);

/*
   Perform rotation from J2000.0 to epoch.
*/

      pos2[0] = Tr[0] * pos1[0] + Tr[3] * pos1[1] + Tr[6] * pos1[2];
      pos2[1] = Tr[1] * pos1[0] + Tr[4] * pos1[1] + Tr[7] * pos1[2];
      pos2[2] = Tr[2] * pos1[0] + Tr[5] * pos1[1] + Tr[8] * pos1[2];

  *ra=atan2(pos2[1],pos2[0]);
  *dec=atan(sqrt(pos2[0]*pos2[0]+pos2[1]*pos2[1])/pos2[2]);


   return 0;
}
