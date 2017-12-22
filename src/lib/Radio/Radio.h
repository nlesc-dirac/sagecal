/*
 *
 Copyright (C) 2006-2008 Sarod Yatawatta <sarod@users.sf.net>  
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

#ifndef SAGECAL_H
#define SAGECAL_H
#ifdef __cplusplus
        extern "C" {
#endif

#include <glib.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>

#include <pthread.h>

/* for gcc 4.8 and above */
#ifndef complex
#define complex _Complex
#endif
 
#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#endif /* HAVE_CUDA */

#ifndef MAX_GPU_ID
#define MAX_GPU_ID 3 /* use 0 (1 GPU), 1 (2 GPUs), ... */
#endif
/* default value for threads per block */
#ifndef DEFAULT_TH_PER_BK 
#define DEFAULT_TH_PER_BK 64
#endif
#ifndef DEFAULT_TH_PER_BK_2
#define DEFAULT_TH_PER_BK_2 32
#endif

/* speed of light */
#ifndef CONST_C
#define CONST_C 299792458.0
#endif

#ifndef MIN
#define MIN(x,y) \
  ((x)<=(y)? (x): (y))
#endif

#ifndef MAX
#define MAX(x,y) \
  ((x)>=(y)? (x): (y))
#endif

/* soure types */
#define STYPE_POINT 0
#define STYPE_GAUSSIAN 1
#define STYPE_DISK 2
#define STYPE_RING 3
#define STYPE_SHAPELET 4

/* max source name length, increase it if names get longer */
#define MAX_SNAME 2048

/********* constants - from levmar ******************/
#define CLM_INIT_MU       1E-03
#define CLM_STOP_THRESH   1E-17
#define CLM_DIFF_DELTA    1E-06
#define CLM_EPSILON       1E-12
#define CLM_ONE_THIRD     0.3333333334 /* 1.0/3.0 */
#define CLM_OPTS_SZ       5 /* max(4, 5) */
#define CLM_INFO_SZ       10
#define CLM_DBL_MAX       1E12    /* max double value */

#include <Common.h>
// #include <Removed-from-Radio.h>

/* given the epoch jd_tdb2, 
 calculate rotation matrix params needed to precess from J2000 
   NOVAS (Naval Observatory Vector Astronomy Software)
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
extern int
get_precession_params(double jd_tdb2, double Tr[9]);
/* precess  ra0,dec0 at J2000
   to ra,dec at epoch given by transform Tr
 using NOVAS library */
extern int
precession(double ra0, double dec0, double Tr[9], double *ra, double *dec);

/****************************** stationbeam.c ****************************/
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
extern int
arraybeam(double ra, double dec, double ra0, double dec0, double f, double f0, int N, double *longitude, double *latitude, double time_jd, int *Nelem, double **x, double **y, double **z, double *beamgain);


/****************************** predict_withbeam.c ****************************/
/* precalculate cluster coherencies
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: coherencies size Nbase*4*Mx 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  N: no of stations
  Nbase: total no of baselines (including more than one tile or timeslot)
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  uvmin: baseline length sqrt(u^2+v^2) below which not to include in solution
  uvmax: baseline length higher than this not included in solution

  Station beam specific parameters
  ph_ra0,ph_dec0: beam pointing rad,rad
  ph_freq0: beam reference freq
  longitude,latitude: Nx1 arrays (rad,rad) station locations
  time_utc: JD (day) : tilesz x 1 
  tilesz: how many tiles: == unique time_utc
  Nelem: Nx1 array, size of stations (elements)
  xx,yy,zz: Nx1 arrays of station element locations arrays xx[],yy[],zz[]
  Nt: no of threads

  NOTE: prediction is done for all baselines, even flagged ones
  and flags are set to 2 for baselines lower than uvcut
*/

extern int
precalculate_coherencies_withbeam(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tileze, int *Nelem, double **xx, double **yy, double **zz, int Nt);


extern int
predict_visibilities_multifreq_withbeam(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, int Nt, int add_to_data);

extern int
calculate_residuals_multifreq_withbeam(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,
double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, int Nt, int ccid, double rho, int phase_only);


/* change epoch of soure ra,dec from J2000 to JAPP */
/* also the beam pointing ra_beam,dec_beam */
extern int
precess_source_locations(double jd_tdb, clus_source_t *carr, int M, double *ra_beam, double *dec_beam, int Nt);

/****************************** predict_withbeam_gpu.c ****************************/
/* if dobeam==0, beam calculation is off */
extern int
precalculate_coherencies_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tileze, int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt);

extern int
predict_visibilities_multifreq_withbeam_gpu(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt, int add_to_data);



/****************************** predict_model.cu ****************************/
extern void
cudakernel_array_beam(int N, int T, int K, int F, float *freqs, float *longitude, float *latitude,
 double *time_utc, int *Nelem, float **xx, float **yy, float **zz, float *ra, float *dec, float ph_ra0, float  ph_dec0, float ph_freq0, float *beam);


extern void
cudakernel_coherencies(int B, int N, int T, int K, int F, float *u, float *v, float *w,baseline_t *barr, float *freqs, float *beam, float *ll, float *mm, float *nn, float *sI,
  unsigned char *stype, float *sI0, float *f0, float *spec_idx, float *spec_idx1, float *spec_idx2, int **exs, float deltaf, float deltat, float dec0, float *coh,int dobeam);


extern void
cudakernel_convert_time(int T, double *time_utc);
#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* SAGECAL_H */
