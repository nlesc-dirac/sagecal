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

#include <Solvers.h>

/* structures to store extra source info for extended sources */
typedef struct exinfo_gaussian_ {
  double eX,eY,eP; /* major,minor,PA */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_gaussian;

typedef struct exinfo_disk_ {
  double eX; /* diameter */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_disk;

typedef struct exinfo_ring_ {
  double eX; /* diameter */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_ring;

typedef struct exinfo_shapelet_ {
  int n0; /* model order, no of modes=n0*n0 */
  double beta; /* scale*/
  double *modes; /* array of n0*n0 x 1 values */
  double eX,eY,eP; /* linear transform parameters */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_shapelet;


/* when to project l,m coordinates */
#ifndef PROJ_CUT
#define PROJ_CUT 0.998
#endif


/* struct for a cluster GList item */
typedef struct clust_t_{
 int id; /* cluster id */
 int nchunk; /* no of chunks the data is divided for solving */
 GList *slist; /* list of sources in this cluster (string)*/
} clust_t;

typedef struct clust_n_{
 char *name; /* source name (string)*/
} clust_n;

/* struct to store source info in hash table */
typedef struct sinfo_t_ {
 double ll,mm,ra,dec,sI[4]; /* sI:4x1 for I,Q,U,V, note sI is updated for central freq (ra,dec) for Az,El */
 unsigned char stype; /* source type */
 void *exdata; /* pointer to carry additional data, if needed */
 double sI0[4],f0,spec_idx,spec_idx1,spec_idx2; /* for multi channel data, original sI,Q,U,V, f0 and spectral index */
} sinfo_t;

/* struct for array of the sky model, with clusters */
typedef struct clus_source_t_ {
 int N; /* no of source in this cluster */
 int id; /* cluster id */
 double *ll,*mm,*nn,*sI,*sQ,*sU,*sV; /* arrays Nx1 of source info, note: sI is at reference freq of data */
 /* nn=sqrt(1-ll^2-mm^2)-1 */
 double *ra,*dec; /* arrays Nx1 for Az,El calculation */
 unsigned char *stype; /* source type array Nx1 */
 void **ex; /* array for extra source information Nx1 */

 int nchunk; /* no of chunks the data is divided for solving */
 int *p; /* array nchunkx1 points to parameter array indices */


 double *sI0,*sQ0,*sU0,*sV0,*f0,*spec_idx,*spec_idx1,*spec_idx2; /* for multi channel data, original sI, f0 and spectral index */
} clus_source_t;

/* strutct to store baseline to station mapping */
typedef struct baseline_t_ {
 int sta1,sta2;
 unsigned char flag; /* if this baseline is flagged, set to 1, otherwise 0: 
             special case: 2 if baseline is not used in solution, but will be
              subtracted */
} baseline_t;


/****************************** readsky.c ****************************/
/* read sky/cluster files, 
   carr:  return array size Mx1 of clusters
   M : no of clusters
   freq0: obs frequency Hz
   ra0,dec0 : ra,dec of phase center (radians)
   format: 0: LSM, 1: LSM with 3 order spec index
   each element has source infor for that cluster */
extern int
read_sky_cluster(const char *skymodel, const char *clusterfile, clus_source_t **carr, int *M, double freq0, double ra0, double dec0,int format);

/* read solution file, only a set of solutions and load to p
  sfp: solution file pointer
  p: solutions vector Mt x 1
  carr: for getting correct offset in p
  N : stations 
  M : clusters
*/
extern int
read_solutions(FILE *sfp,double *p,clus_source_t *carr,int N,int M);

/* set ignlist[ci]=1 if 
  cluster id 'cid' is mentioned in ignfile and carr[ci].id==cid
*/ 
extern int
update_ignorelist(const char *ignfile, int *ignlist, int M, clus_source_t *carr);

/* read ADMM regularization factor per cluster from text file, format:
 cluster_id  hybrid_parameter admm_rho
 ...
 ...
 (M values)
 and store it in array arho : size Mtx1, taking into account the hybrid parameter
 also in array arhoslave : size Mx1, without taking hybrid params into account

 admm_rho : can be 0 to ignore consensus, just normal calibration
*/
 
extern int
read_arho_fromfile(const char *admm_rho_file,int Mt,double *arho, int M, double *arhoslave);

/****************************** predict.c ****************************/
/************* extended source contributions ************/
extern complex double
shapelet_contrib(void*dd, double u, double v, double w);

extern complex double
gaussian_contrib(void*dd, double u, double v, double w);

extern complex double
ring_contrib(void*dd, double u, double v, double w);

extern complex double
disk_contrib(void*dd, double u, double v, double w);


/* time smearing TMS eq. 6.80 for EW-array formula 
  note u,v,w: meter/c so multiply by freq. to get wavelength 
  ll,mm: source
  dec0: phase center declination
  tdelta: integration time */
extern double 
time_smear(double ll,double mm,double dec0,double tdelta,double u,double v,double w,double freq0);

/* predict visibilities
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: data to write size Nbase*8*tileze x 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  N: no of stations
  Nbase: no of baselines
  tilesz: tile size
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  Nt: no of threads
*/
extern int
predict_visibilities(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, int Nt); 
  

/* precalculate cluster coherencies
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: coherencies size Nbase*4*Mx 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  N: no of stations
  Nbase: no of baselines (including more than one tile)
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  uvmin: baseline length sqrt(u^2+v^2) below which not to include in solution
  uvmax: baseline length higher than this not included in solution
  Nt: no of threads

  NOTE: prediction is done for all baselines, even flagged ones
  and flags are set to 2 for baselines lower than uvcut
*/
extern int
precalculate_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int Nt); 



/* rearranges coherencies for GPU use later */
/* barr: 2*Nbase x 1
   coh: M*Nbase*4 x 1 complex
   ddcoh: M*Nbase*8 x 1
   ddbase: 2*Nbase x 1 (sta1,sta2) = -1 if flagged
*/
extern int
rearrange_coherencies(int Nbase, baseline_t *barr, complex double *coh, double *ddcoh, short *ddbase, int M, int Nt);
/* ddbase: 3*Nbase x 1 (sta1,sta2,flag) */
extern int
rearrange_coherencies2(int Nbase, baseline_t *barr, complex double *coh, double *ddcoh, short *ddbase, int M, int Nt);

/* rearranges baselines for GPU use later */
/* barr: 2*Nbase x 1
   ddbase: 2*Nbase x 1
*/
extern int
rearrange_baselines(int Nbase, baseline_t *barr, short *ddbase, int Nt);

/* cont how many baselines contribute to each station */
extern int
count_baselines(int Nbase, int N, float *iw, short *ddbase, int Nt);

/* initialize array b (size Nx1) to given value a */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
setweights(int N, double *b, double a, int Nt);

/* update baseline flags, also make data zero if flagged
  this is needed for solving (calculate error) ignore flagged data */
/* Nbase: total actual data points = Nbasextilesz
   flag: flag array Nbasex1
   barr: baseline array Nbasex1
   x: data Nbase*8 x 1 ( 8 value per baseline ) 
   Nt: no of threads 
*/
extern int
preset_flags_and_data(int Nbase, double *flag, baseline_t *barr, double *x, int Nt);

/* generte baselines -> sta1,sta2 pairs for later use */
/* barr: Nbasextilesz
   N : stations
   Nt : threads 
*/
extern int
generate_baselines(int Nbase, int tilesz, int N, baseline_t *barr,int Nt);

/* convert types */
/* both arrays size nx1 
   Nt: no of threads
*/
extern int
double_to_float(float *farr, double *darr,int n, int Nt);
extern int
float_to_double(double *darr, float *farr,int n, int Nt);

/* create a vector with 1's at flagged data points */
/* 
   ddbase: 3*Nbase x 1 (sta1,sta2,flag)
   x: 8*Nbase (set to 0's and 1's)
*/
extern int
create_onezerovec(int Nbase, short *ddbase, float *x, int Nt);

/* 
  find sum1=sum(|x|), and sum2=y^T |x|
  x,y: nx1 arrays
*/
extern int
find_sumproduct(int N, float *x, float *y, float *sum1, float *sum2, int Nt);

/****************************** transforms.c ****************************/
#ifndef ASEC2RAD
#define ASEC2RAD 4.848136811095359935899141e-6
#endif

/* 
 convert xyz ITRF 2000 coords (m) to
 long,lat, (rad) height (m)
 References:
*/
extern int
xyz2llh(double *x, double *y, double *z, double *longitude, double *latitude, double *height, int N);

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
extern int
radec2azel(double ra, double dec, double longitude, double latitude, double time_jd, double *az, double *el);

/* convert time to Greenwitch Mean Sideral Angle (deg)
   time_jd : JD days
   thetaGMST : GMST angle (deg)
*/
extern int
jd2gmst(double time_jd, double *thetaGMST); 


/* convert ra,dec to az,el
   ra,dec: radians
   longitude,latitude: rad,rad 
   thetaGMST : GMST angle (deg)

   az,el: output  rad,rad

*/
extern int
radec2azel_gmst(double ra, double dec, double longitude, double latitude, double thetaGMST, double *az, double *el); 



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
