/*
 *
 Copyright (C) 2024 Sarod Yatawatta <sarod@users.sf.net>
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


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <SpiceUsr.h>
#include "Dirac_radio.h"

//#define DEBUG

void
cspice_load_kernels(void) {
  /* get env CSPICE_KERNEL_PATH */
  char* cspice_path = getenv("CSPICE_KERNEL_PATH");
  if (cspice_path) {
    const char *kname="/pck00011.tpc\0";
    char *fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    kname="/naif0012.tls\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    kname="/moon_de440_220930.tf\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    kname="/moon_pa_de440_200625.bpc\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    /* following kernel only needed for ITRF93 frame */
    /*
    kname="/earth_000101_240713_240419.bpc\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);
    */

  } else {
    fprintf(stderr,"CSPICE kernel path 'CSPICE_KERNEL_PATH' is not found in environment variables\n");
    fprintf(stderr,"Download the kernels\n"
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc\n"
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls\n"
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_220930.tf\n"
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc\n");
    fprintf(stderr,"And rerun the program after setting the directory where these kernels stored as CSPICE_KERNEL_PATH in your environment.\n");
    exit(1);
  }
}

/* x,y,z: selenocentric rectangular coordinates (m)
 * lon,lat: longitude,latitude (rad)
 * alt: altitude (m) */
int
cspice_xyz_to_latlon(double x,double y, double z,double *lon, double *lat, double *alt) {
  SpiceDouble radii[3];
  SpiceInt dim;
  bodvrd_c("MOON","RADII",3,&dim,radii);
#ifdef DEBUG
  printf("MOON RADII %lf %lf %lf\n",radii[0],radii[1],radii[2]);
#endif
  /* flattening coefficient */
  SpiceDouble re=radii[0];
  SpiceDouble rp=radii[2];
  SpiceDouble fl=(re-rp)/re;
  /* rectangular (km) X,Y,Z to planetographic */
  SpiceDouble rect[3]={x*0.001,y*0.001,z*0.001};
  recpgr_c("MOON",rect,re,fl,lon,lat,alt);

  (*alt ) *=1000.0; /* km -> m */

  return 0;
}


int
cspice_element_beam_lunar(double ra, double dec, double f, double f0, int N, double *longitude, double *latitude, double time_jd, elementcoeff *ecoeff, double *elementgain, int wideband, int findex, pthread_mutex_t *mutex) {

  pthread_mutex_lock(mutex);
  /* ephemeris time from time_jd (days) to (s) */
  double ep_t0=unitim_c(time_jd,"JED","ET");
#ifdef DEBUG
  printf("UTC %le (d) ET %le (s)\n",time_jd,ep_t0);
#endif
  /* rectangular coords */
  SpiceDouble srcrect[3],mtrans[3][3],v2000[3];
  /* ra,dec to rectangular */
  radrec_c(1.0,ra*rpd_c(),dec*rpd_c(),v2000);
  /* precess ep_t0 on lunar frame ME: mean Earth/polar axis, PA: principle axis */
  pxform_c("J2000","MOON_ME",ep_t0,mtrans); // use MOON_ME instead of MOON_PA
  /* rotate v2000 onto lunar frame */
  mxv_c(mtrans,v2000,srcrect);
  /* rectangular to lat,long */
  SpiceDouble s_radius,s_lon,s_lat;
  reclat_c(srcrect, &s_radius, &s_lon, &s_lat);
  pthread_mutex_unlock(mutex);

  int ci;
  double az,el;
  double theta;
  /* iterate over stations */
  for (ci=0; ci<N; ci++) {

  /* find difference between station=lon1,lat1, source=lon2,lat2 */
  /* Haversine formulae */
  double d_lon=s_lon-longitude[ci];
  double d_lat=s_lat-latitude[ci];
#ifdef DEBUG
  printf("d_lon %lf %lf %lf\n",s_lon,longitude[ci],d_lon);
  printf("d_lat %lf %lf %lf\n",s_lat,latitude[ci],d_lat);
#endif
  double s_dlat_2=sin(d_lat*0.5);
  double s_dlon_2=sin(d_lon*0.5);
  double a=s_dlat_2*s_dlat_2+cos(latitude[ci])*cos(s_lat)*s_dlon_2*s_dlon_2;
#ifdef DEBUG
  printf("a=%lf\n",a);
#endif
  double a_2=(a>0.0?sqrt(a):1.0);
  /* great circle distance in rad */
  double c=2.0*asin((a_2>1.0?1.0:a_2));

  /* azimuth angle, not valid for poles */
  az=acos( (sin(s_lat)-sin(latitude[ci])*cos(c))/(sin(c)*cos(latitude[ci])) );
  if (sin(d_lon)>0.0) { az= 2.0*M_PI-az; }

  /* limit latitude difference to [-pi/2,pi/2] */
  if (fabs(d_lat)<=M_PI_2) {
  /* elevation = pi/2 - (latitude difference) */
  el=M_PI_2-fabs(d_lat);
  } else {
    el=-1.0; /* invalid */
  }
#ifdef DEBUG
  printf("azimuth %lf elevation %lf\n",az,el);
#endif

   /* transform : theta = 90-el, phi=-az? 45 only needed for element beam */
   theta=M_PI_2-el;

   if (el>=0.0) {
   /* element beam EJones */
   /* evaluate on r=(zenith angle) 0..pi/2, theta=azimuth grid 0..2pi */
   /* real data r<- gamma=pi/2-elevation =theta from above code
    * theta <- beta=azimuth-pi/4  for XX, -pi/2 for YY
      E = [ Etheta(gamma,beta) Ephi(gamma,beta);
            Etheta(gamma,beta+pi/2) Ehpi(gamma,beta+pi/2) ]; */
   if (!wideband) {
     elementval evalX=eval_elementcoeffs(theta,az-M_PI_4,ecoeff);
     elementval evalY=eval_elementcoeffs(theta,az-M_PI_4+M_PI_2,ecoeff);
     elementgain[8*ci]=creal(evalX.theta);
     elementgain[8*ci+1]=cimag(evalX.theta);
     elementgain[8*ci+2]=creal(evalX.phi);
     elementgain[8*ci+3]=cimag(evalX.phi);
     elementgain[8*ci+4]=creal(evalY.theta);
     elementgain[8*ci+5]=cimag(evalY.theta);
     elementgain[8*ci+6]=creal(evalY.phi);
     elementgain[8*ci+7]=cimag(evalY.phi);
   } else {
     elementval evalX=eval_elementcoeffs_wb(theta,az-M_PI_4,ecoeff,findex);
     elementval evalY=eval_elementcoeffs_wb(theta,az-M_PI_4+M_PI_2,ecoeff,findex);
     elementgain[8*ci]=creal(evalX.theta);
     elementgain[8*ci+1]=cimag(evalX.theta);
     elementgain[8*ci+2]=creal(evalX.phi);
     elementgain[8*ci+3]=cimag(evalX.phi);
     elementgain[8*ci+4]=creal(evalY.theta);
     elementgain[8*ci+5]=cimag(evalY.theta);
     elementgain[8*ci+6]=creal(evalY.phi);
     elementgain[8*ci+7]=cimag(evalY.phi);
   }
   } else {
    elementgain[8*ci]=0.0;
    elementgain[8*ci+1]=0.0;
    elementgain[8*ci+2]=0.0;
    elementgain[8*ci+3]=0.0;
    elementgain[8*ci+4]=0.0;
    elementgain[8*ci+5]=0.0;
    elementgain[8*ci+6]=0.0;
    elementgain[8*ci+7]=0.0;
   }
  }
 

  return 0;
}
