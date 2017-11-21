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
#define DEFAULT_TH_PER_BK 128
#endif
#ifndef DEFAULT_TH_PER_BK_2
#define DEFAULT_TH_PER_BK_2 64
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

/* simulation options */
#define SIMUL_ONLY 1 /* only predict model */
#define SIMUL_ADD 2 /* add to input */
#define SIMUL_SUB 3 /* subtract from input */

/********* constants - from levmar ******************/
#define CLM_INIT_MU       1E-03
#define CLM_STOP_THRESH   1E-17
#define CLM_DIFF_DELTA    1E-06
#define CLM_EPSILON       1E-12
#define CLM_ONE_THIRD     0.3333333334 /* 1.0/3.0 */
#define CLM_OPTS_SZ       5 /* max(4, 5) */
#define CLM_INFO_SZ       10
#define CLM_DBL_MAX       1E12    /* max double value */

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


/* structure for worker threads for various function calculations */
typedef struct thread_data_base_ {
  int Nb; /* no of baselines this handle */
  int boff; /* baseline offset per thread */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *u,*v,*w; /* pointers to uwv arrays,size Nbx1 */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  double *x; /* output vector Nbx8 array re,im,re,im .... */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  /* following are only used while predict with gain */
  double *p; /* parameter array, size could be 8*N*Mx1 (full) or 8*Nx1 (single)*/
  int N; /* no of stations */
  int clus; /* which cluster to process, 0,1,...,M-1 if -1 all clusters */
  double uvmin; /* baseline length sqrt(u^2+v^2) lower limit, below this is not 
                 included in calibration, but will be subtracted */
  double uvmax;
  /* following used for freq/time smearing calculation */
  double freq0;
  double fdelta;
  double tdelta; /* integration time for time smearing */
  double dec0; /* declination for time smearing */

  /* following used for interpolation */
  double *p0; /* old parameters, same as p */
  int tilesz; /* tile size */
  int Nbase; /* total no of baselines */
  /* following for correction of data */
  double *pinv; /* inverted solution array, if null no correction */
  int ccid; /* which cluster id (not user specified id) for correction, >=0 */

  /* following for ignoring clusters in simulation */
  int *ignlist; /* Mx1 array, if any value 1, that cluster will not be simulated */
  /* flag for adding/subtracting model to data */
  int add_to_data; /* see SIMUL* defs */

  /* following used for multifrequency (channel) data */
  double *freqs;
  int Nchan;

  /* following used for calculating beam */
  double *arrayfactor; /* storage for precomputed beam */
  /* if clus==0, reset memory before adding */

} thread_data_base_t;

/* structure for worker threads for 
   precalculating beam array factor */
typedef struct thread_data_arrayfac_ {
  int Ns; /* total no of sources per thread */
  int soff; /* starting source */
  int Ntime; /* total timeslots */
  double *time_utc; /* Ntimex1 array */
  int N; /* no. of stations */
  double *longitude, *latitude;

  double ra0,dec0,freq0; /* reference pointing and freq */
  int Nf; /* no. of frequencies to calculate */
  double *freqs; /* Nfx1 array */

  int *Nelem; /* Nx1 array of element counts */
  double **xx,**yy,**zz; /* Nx1 arrays to element coords of each station, size Nelem[]x1 */
  
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int cid; /* cluster id to calculate beam */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *beamgain; /* output */
} thread_data_arrayfac_t;


/* structure for worker threads for presetting
   flagged data before solving */
typedef struct thread_data_preflag_ {
  int Nbase; /* total no of baselines */
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *x; /* data */
  double *flag; /* flag array 0 or 1 */
} thread_data_preflag_t;


/* structure for worker threads for arranging coherencies for GPU use */
typedef struct thread_data_coharr_ {
  int M; /* no of clusters */
  int Nbase; /* no of baselines */
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  double *ddcoh; /* coherencies, rearranged for easy copying to GPU, also real,imag instead of complex */
  short *ddbase; /* baseline to station maps, same as barr, assume no of stations < 32k, if flagged set to -1 OR (sta1,sta2,flag) 3 values for each baseline */
} thread_data_coharr_t;

/* structure for worker threads for type conversion */
typedef struct thread_data_typeconv_{
  int starti; /* starting baseline */
  int endi; /* ending baseline */
  double *darr; /* double array */
  float *farr; /* float array */
} thread_data_typeconv_t;

/* structure for worker threads for baseline generation */
typedef struct thread_data_baselinegen_{
  int starti; /* starting tile */
  int endi; /* ending tile */
  baseline_t *barr; /* baseline array */
  int N; /* stations */
  int Nbase; /* baselines */
} thread_data_baselinegen_t;

/* structure for counting baselines for each station (RTR)*/
typedef struct thread_data_count_ {
 int Nb; /* no of baselines this handle */
 int boff; /* baseline offset per thread */

 short *ddbase;

 int *bcount;

  /* mutexs: N x 1, one for each station */
  pthread_mutex_t *mx_array;
} thread_data_count_t;


/* structure for initializing an array */
typedef struct thread_data_setwt_ {
 int Nb; /* no of baselines this handle */
 int boff; /* baseline offset per thread */

 double *b;
 double a;

} thread_data_setwt_t;

/* structure for weight calculation for baselines */
typedef struct thread_data_baselinewt_ {
 int Nb; /* no of baselines this handle */
 int boff; /* baseline offset per thread */

 double *wt; /* 8 values per baseline */
 double *u,*v;
 double freq0;

} thread_data_baselinewt_t;



/* structure for worker threads for jacobian calculation */
typedef struct thread_data_jac_ {
  int Nb; /* no of baselines this handle */
  int n; /* function dimension n=8*Nb  is implied */
  int m; /* no of parameters */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *u,*v,*w; /* pointers to uwv arrays,size Nbx1 */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  double *jac; /* output jacobian Nbx8 rows, re,im,re,im .... */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  /* following are only used while predict with gain */
  double *p; /* parameter array, size could be 8*N*Mx1 (full) or 8*Nx1 (single)*/
  int N; /* no of stations */
  int clus; /* which cluster to process, 0,1,...,M-1 if -1 all clusters */
  int start_col;
  int end_col; /* which column of jacobian we calculate */
} thread_data_jac_t;


/* structure for levmar */
typedef struct me_data_t_ {
  int clus; /* which cluster 0,1,...,M-1 if -1 all clusters */
  double *u,*v,*w; /* uvw coords size Nbase*tilesz x 1 */
  int Nbase; /* no of baselines */
  int tilesz; /* tile size */
  int N; /* no of stations */
  baseline_t *barr; /* baseline->station mapping, size Nbase*tilesz x 1 */
  clus_source_t *carr; /* sky model, with clusters size Mx1 */
  int M; /* no of clusters */
  int Mt; /* apparent no of clusters, due to hybrid solving, Mt>=M */
  double *freq0; /* frequency */
  int Nt; /* no of threads */

  complex double *coh; /* pre calculated cluster coherencies, per cluster 4xNbase values, total size 4*M*Nbase*tilesz x 1 */
  /* following only used by CPU LM */
  int tileoff; /* tile offset for hybrid solution */

  /* following only used by GPU LM version */
  double *ddcoh; /* coherencies, rearranged for easy copying to GPU, also real,imag instead of complex */
  short *ddbase; /* baseline to station maps, same as barr, size 2*Nbase*tilesz x 1, assume no of stations < 32k, if flagged set to -1 */
  /* following used only by LBFGS */
  short *hbb; /*  baseline to station maps, same as ddbase size 2*Nbase*tilesz x 1, assume no of stations < 32k, if flagged set to -1 */
  int *ptoclus; /* param no -> cluster mapping, size 2*M x 1 
      for each cluster : chunk size, start param index */

  /* following used only by mixed precision solver */
  float *ddcohf; /* float version of ddcoh */

  /* following used only by robust T cost/grad functions */
  double robust_nu;

  /* following used only by RTR */
} me_data_t;


/* structure for gpu driver threads for LBFGS */
typedef struct thread_gpu_data_t {
  int ThreadsPerBlock;
  int BlocksPerGrid;
  int card; /* which gpu ? */
   
  int Nbase; /* no of baselines */
  int tilesz; /* tile size */
  baseline_t *barr; /* baseline->station mapping, size Nbase*tilesz x 1 */
  int M; /* no of clusters */
  int N; /* no of stations */
  complex double *coh; /* pre calculated cluster coherencies, per cluster 4xNbase values, total size 4*M*Nbase*tilesz x 1 */
  int m; /* no of parameters */
  int n; /* no of observations */
  double *xo; /* observed data size n x 1 */
  double *p;/* parameter vectors size m x 1 */
  double *g; /* gradient vector (output) size m x 1*/
  int g_start; /* at which point in g do we start calculation */
  int g_end; /* at which point in g do we end calculation */

  short *hbb; /*  baseline to station maps, same as ddbase size 2*Nbase*tilesz x 1, assume no of stations < 32k, if flagged set to -1 */
  int *ptoclus; /* param no -> cluster mapping, size 2*M x 1 
      for each cluster : chunk size, start param index */

  /* only used in robust LBFGS */
  double robust_nu;
} thread_gpu_data;


/* structure for driver threads to evaluate gradient */
typedef struct thread_data_grad_ {
  int Nbase; /* no of baselines */
  int tilesz; /* tile size */
  baseline_t *barr; /* baseline->station mapping, size Nbase*tilesz x 1 */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  int N; /* no of stations */
  complex double *coh; /* pre calculated cluster coherencies, per cluster 4xNbase values, total size 4*M*Nbase*tilesz x 1 */
  int m; /* no of parameters */
  int n; /* no of observations */
  double *x; /* residual data size n x 1 x=observed-func*/
  double *p;/* parameter vectors size m x 1 */
  double *g; /* gradient vector (output) size m x 1*/
  int g_start; /* at which point in g do we start calculation */
  int g_end; /* at which point in g do we end calculation */

  /* only used in robust version */
  double robust_nu;
} thread_data_grad_t;

/* structure for weight product calculation in robust LM  */
typedef struct thread_data_vec_{
  int starti,endi;
  double *ed;
  double *wtd;
} thread_data_vec_t;

/* structure for weight calculation + nu update in robust LM  */
typedef struct thread_data_vecnu_{
  int starti,endi;
  double *ed;
  double *wtd;
  double *q;
  double nu0;
  double sumq;
  double nulow,nuhigh;
} thread_data_vecnu_t;


/* structure for worker threads for setting 1/0 */
typedef struct thread_data_onezero_ {
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  short *ddbase; /* baseline to station maps, (sta1,sta2,flag) */
  float *x; /* data vector */
} thread_data_onezero_t;


/* structure for worker threads for finding sum(|x|) and y^T |x| */
typedef struct thread_data_findsumprod_ {
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  float *x; /* can be -ve*/
  float *y;
  float sum1; /* sum(|x|) */
  float sum2; /* y^T |x| */
} thread_data_findsumprod_t;


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

/****************************** dataio.c ****************************/
/* open binary file for input/output
 datfile: data file descriptor id
 d: array of input/output stream, size (count-(header length))x1
 N: no of stations
 freq0: frequency  Hz
 ra0,dec0: ra,dec of phase center (radians)
*/
extern int
open_data_stream(int file, double **d, int *count, int *N, double *freq0, double *ra0, double *dec0);

/* close the data stream */
extern int
close_data_stream(double *d, int count);


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

/****************************** myblas.c ****************************/
/* BLAS wrappers */
/* machine precision */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double 
dlamch(char CMACH);

/* blas dcopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_dcopy(int N, double *x, int Nx, double *y, int Ny);

/* blas scale */
/* x = a. x */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_dscal(int N, double a, double *x);
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_sscal(int N, float a, float *x);

/* x^T*y */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double
my_ddot(int N, double *x, double *y);

/* ||x||_2 */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double
my_dnrm2(int N, double *x);
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern float
my_fnrm2(int N, float *x);

/* sum||x||_1 */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double
my_dasum(int N, double *x);
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern float
my_fasum(int N, float *x);

/* BLAS y = a.x + y */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_daxpy(int N, double *x, double a, double *y);

/* BLAS y = a.x + y */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_daxpys(int N, double *x, int incx, double a, double *y, int incy);

#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_saxpy(int N, float *x, float a, float *y);

/* max |x|  index (start from 1...)*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_idamax(int N, double *x, int incx);
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_isamax(int N, float *x, int incx);

/* min |x|  index (start from 1...)*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
int
my_idamin(int N, double *x, int incx);

/* BLAS DGEMM C = alpha*op(A)*op(B)+ beta*C */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_dgemm(char transa, char transb, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

/* BLAS DGEMV  y = alpha*op(A)*x+ beta*y : op 'T' or 'N' */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_dgemv(char trans, int M, int N, double alpha, double *A, int lda, double *x, int incx,  double beta, double *y, int incy);

/* following routines used in LAPACK solvers */
/* cholesky factorization: real symmetric */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_dpotrf(char uplo, int N, double *A, int lda);

/* solve Ax=b using cholesky factorization */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_dpotrs(char uplo, int N, int nrhs, double *A, int lda, double *b, int ldb);

/* solve Ax=b using QR factorization */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_dgels(char TRANS, int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *WORK, int LWORK);

/* A=U S VT, so V needs NOT to be transposed */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_dgesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S,
   double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK);
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_zgesvd(char JOBU, char JOBVT, int M, int N, complex double *A, int LDA, double *S,
   complex double *U, int LDU, complex double *VT, int LDVT, complex double *WORK, int LWORK, double *RWORK); 

/* QR factorization QR=A, only TAU is used for Q, R stored in A*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_dgeqrf(int M, int N, double *A, int LDA, double *TAU, double *WORK, int LWORK);

/* calculate Q using elementary reflections */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_dorgqr(int M,int  N,int  K,double *A,int  LDA,double *TAU,double *WORK,int  LWORK);

/* solves a triangular system of equations Ax=b, A triangular */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_dtrtrs(char UPLO, char TRANS, char DIAG,int N,int  NRHS,double *A,int  LDA,double *B,int  LDB);


/* blas ccopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_ccopy(int N, complex double *x, int Nx, complex double *y, int Ny);

/* blas scale */
/* x = a. x */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_cscal(int N, complex double a, complex double *x);


/* BLAS y = a.x + y */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_caxpy(int N, complex double *x, complex double a, complex double *y);


/* BLAS x^H*y */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern complex double
my_cdot(int N, complex double *x, complex double *y);

/* solve Ax=b using QR factorization */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_zgels(char TRANS, int M, int N, int NRHS, complex double *A, int LDA, complex double *B, int LDB, complex double *WORK, int LWORK);
extern int
my_cgels(char TRANS, int M, int N, int NRHS, complex float *A, int LDA, complex float *B, int LDB, complex float *WORK, int LWORK);

/* BLAS ZGEMM C = alpha*op(A)*op(B)+ beta*C */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_zgemm(char transa, char transb, int M, int N, int K, complex double alpha, complex double *A, int lda, complex double *B, int ldb, complex double beta, complex double *C, int ldc);

/* ||x||_2 */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double
my_cnrm2(int N, complex double *x);


/* blas fcopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern void
my_fcopy(int N, float *x, int Nx, float *y, int Ny);


/* LAPACK eigen value expert routine, real symmetric  matrix */
extern int
my_dsyevx(char jobz, char range, char uplo, int N, double *A, int lda,
  double vl, double vu, int il, int iu, double abstol, int M, double  *W,
  double *Z, int ldz, double *WORK, int lwork, int *iwork, int *ifail);

/* BLAS vector outer product
   A= alpha x x^H + A
*/
extern void
my_zher(char uplo, int N, double alpha, complex double *x, int incx, complex double *A, int lda);
/****************************** lbfgs.c ****************************/
/****************************** lbfgs_nocuda.c ****************************/
/* LBFGS routines */
/* func: vector function to minimize, actual cost minimized is ||func-x||^2
   NOTE: gradient function given seperately
   p: parameters m x 1 (used as initial value, output final value)
   x: data  n x 1
   itmax: max iterations
   lbfgs_m: memory size
   gpu_threads: GPU threads per block
   adata: additional data
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
lbfgs_fit(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads, void *adata);
 
/****************************** robust_lbfgs_nocuda.c ****************************/
typedef struct thread_data_logf_t_ {
  double *f;
  double *x;
  double nu;
  int start,end;
  double sum;
} thread_data_logf_t;

/* robust_nu: nu in T distribution */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
lbfgs_fit_robust(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads,
 void *adata);
#ifdef HAVE_CUDA
extern int
lbfgs_fit_robust_cuda(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads, void *adata);
#endif

/****************************** residual.c ****************************/
/* residual calculation, with/without linear interpolation */
/* 
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  p0,p: parameter arrays 8*N*M x1 double values (re,img) for each station/direction
  p0: old value, p new one, interpolate between the two
  x: data to write size Nbase*8*tilesz x 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  input: x is actual data, output: x is the residual
  N: no of stations
  Nbase: no of baselines
  tilesz: tile size
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  coh: coherencies size Nbase*tilesz*4*M x 1
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  Nt: no. of threads
  ccid: which cluster to use as correction
  rho: MMSE robust parameter J+rho I inverted

  phase_only: if >0, and if there is any correction done, use only phase of diagonal elements for correction 
*/
extern int
calculate_residuals(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double freq0,double fdelta,double tdelta,double dec0, int Nt, int ccid, double rho);

/* 
  residuals for multiple channels
  data to write size Nbase*8*tilesz*Nchan x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots, channels
  input: x is actual data, output: x is the residual
  freqs: Nchanx1 of frequency values
  fdelta: total bandwidth, so divide by Nchan to get each channel bandwith
  tdelta: integration time for time smearing
  dec0: declination for time smearing
*/
extern int
calculate_residuals_multifreq(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0, int Nt, int ccid, double rho, int phase_only);

/* 
  calculate visibilities for multiple channels, no solutions are used
  note: output column x is set to 0 if add_to_data ==0, else model is added/subtracted (==1 or 2) to data
*/
extern int
predict_visibilities_multifreq(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,int Nt,int add_to_data);


/* predict with solutions in p , ignore clusters flagged in ignorelist (Mx1) array
 also correct final data with solutions for cluster ccid, if valid
*/
extern int
predict_visibilities_multifreq_withsol(double *u,double *v,double *w,double *p,double *x,int *ignorelist,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,int Nt,int add_to_data, int ccid, double rho,int phase_only);
/****************************** mderiv.cu ****************************/
/* cuda driver for kernel */
/* ThreadsPerBlock: keep <= 128
   BlocksPerGrid: depends on the threads/baselines> Threads*Blocks approx baselines
   N: no of baselines (total, including tilesz >1)
   tilesz: tile size
   M: no of clusters
   Ns: no of stations
   Nparam: no of actual parameters  <=total 
   goff: starting point of gradient calculation 0..Nparams
   x: N*8 x 1 residual
   coh: N*8*M x 1
   p: M*Ns*8 x 1
   bb: 2*N x 1
   ptoclus: 2*M x 1
   grad: Nparamsx1 gradient values
*/
extern void 
cudakernel_lbfgs(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad);
/* x: data vector, not residual */
extern void 
cudakernel_lbfgs_r(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad);
extern void 
cudakernel_lbfgs_r_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad, double robust_nu);


/* cost function calculation, each GPU works with Nbase baselines out of Nbasetotal baselines 
 */
extern double
cudakernel_lbfgs_cost(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus);
extern double
cudakernel_lbfgs_cost_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus, double robust_nu);


/* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps */
extern void 
cudakernel_diagdiv(int ThreadsPerBlock, int BlocksPerGrid, int M, double eps, double *Dpd, double *Sd);

/* cuda driver for calculating
  A<= A+mu I, adding mu to diagonal entries of A
  A: size MxM
  ThreadsPerBlock, BlocksPerGrid calculated to meet M
*/
extern void
cudakernel_diagmu(int ThreadsPerBlock, int BlocksPerGrid, int M, double *A, double mu);

/* cuda driver for calculating f() */
/* p: params (Mx1): for all chunks, x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations  */
extern void
cudakernel_func(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations);

/* cuda driver for calculating jacf() */
/* p: params (Mx1): for all chunks, jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations);


/****************************** mderiv_fl.cu ****************************/
/* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps */
extern void 
cudakernel_diagdiv_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float eps, float *Dpd, float *Sd);
/* cuda driver for calculating
  A<= A+mu I, adding mu to diagonal entries of A
  A: size MxM
  ThreadsPerBlock, BlocksPerGrid calculated to meet M
*/
extern void
cudakernel_diagmu_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float *A, float mu);
/* cuda driver for calculating f() */
/* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_func_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, short *bbh, int Nbase, int Mclus, int Nstations);
/* cuda driver for calculating jacf() */
/* p: params (Mx1), jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, short *bbh, int Nbase, int Mclus, int Nstations);
/****************************** robust.cu ****************************/
/* cuda driver for calculating wt \odot f() */
/* p: params (Mx1): for all chunks, x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations  */
extern void
cudakernel_func_wt(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations);

/* cuda driver for calculating wt \odot jacf() */
/* p: params (Mx1): for all chunks, jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf_wt(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations);


/* set initial weights to 1 by a cuda kernel */
extern void
cudakernel_setweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wtd, double alpha);

/* hadamard product by a cuda kernel x<= x*wt */
extern void
cudakernel_hadamard(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x);

/* update weights by a cuda kernel */
extern void
cudakernel_updateweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x, double *q, double robust_nu);

/* make sqrt() weights */
extern void
cudakernel_sqrtweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt);

/* evaluate expression for finding optimum nu for 
  a range of nu values */
extern void
cudakernel_evaluatenu(int ThreadsPerBlock, int BlocksPerGrid, int Nd, double qsum, double *q, double deltanu,double nulow);

/* ThreadsPerBlock: keep <= 128
   BlocksPerGrid: depends on the threads/baselines> Threads*Blocks approx baselines
   N: no of baselines (total, including tilesz >1)
   tilesz: tile size
   M: no of clusters
   Ns: no of stations
   Nparam: no of actual parameters  <=total 
   goff: starting point of gradient calculation 0..Nparams
   x: N*8 x 1 residual
   coh: N*8*M x 1
   p: M*Ns*8 x 1
   bb: 2*N x 1
   ptoclus: 2*M x 1
   grad: Nparamsx1 gradient values
*/
extern void 
cudakernel_lbfgs_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double robust_nu, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad);

/****************************** robust_fl.cu ****************************/
/* cuda driver for calculating wt \odot f() */
/* p: params (Mx1): for all chunks, x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations  */
extern void
cudakernel_func_wt_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations);

/* cuda driver for calculating wt \odot jacf() */
/* p: params (Mx1): for all chunks, jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf_wt_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations);


/* set initial weights to 1 by a cuda kernel */
extern void
cudakernel_setweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wtd, float alpha);

/* hadamard product by a cuda kernel x<= x*wt */
extern void
cudakernel_hadamard_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x);

/* update weights by a cuda kernel */
extern void
cudakernel_updateweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x, float *q, float robust_nu);

/* make sqrt() weights */
extern void
cudakernel_sqrtweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt);

/* evaluate expression for finding optimum nu for 
  a range of nu values */
extern void
cudakernel_evaluatenu_fl(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow);

/* evaluate expression for finding optimum nu for 
  a range of nu values , 8 variate T distrubution
  using AECM */
extern void
cudakernel_evaluatenu_fl_eight(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow, float nu0);


/****************************** barrier.c ****************************/
typedef struct t_barrier_ {
  int tcount; /* current no. of threads inside barrier */
  int nthreads; /* the no. of threads the barrier works
                with. This is a constant */
  pthread_mutex_t enter_mutex;
  pthread_mutex_t exit_mutex;
  pthread_cond_t lastthread_cond;
  pthread_cond_t exit_cond;
} th_barrier;


/* initialize barrier */
/* N - no. of accomodated threads */
extern void
init_th_barrier(th_barrier *barrier, int N);

/* destroy barrier */
extern void
destroy_th_barrier(th_barrier *barrier);

/* the main operation of the barrier */
extern void
sync_barrier(th_barrier *barrier);



/****************************** clmfit.c ****************************/
#ifdef HAVE_CUDA
/* LM with GPU */
extern int
clevmar_der_single(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  int card,   /* GPU to use */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  void *adata); /* pointer to possibly additional data  */

/* function to set up a GPU, should be called only once */
extern void
attach_gpu_to_thread(int card, cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle);
extern void
attach_gpu_to_thread1(int card, cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle, double **WORK, int64_t work_size);
extern void
attach_gpu_to_thread2(int card,  cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle, float **WORK, int64_t work_size, int usecula);


/* function to detach a GPU from a thread */
extern void
detach_gpu_from_thread(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
extern void
detach_gpu_from_thread1(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle, double *WORK);
extern void
detach_gpu_from_thread2(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle, float *WORK, int usecula);
/* function to set memory to zero */
extern void
reset_gpu_memory(double *WORK, int64_t work_size);


/* same as above, but f() and jac() calculations are done 
  entirely in the GPU */
extern int
clevmar_der_single_cuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  double *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  void *adata); /* pointer to possibly additional data  */

/** keep interface almost the same as in levmar **/
extern int
mlm_der_single_cuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[6],   /* I: minim. options [\mu, \m, \p0, \p1, \p2, \delta].
                        delta: 1 or 2
                       */
  double info[10], 
                      /* O: information regarding the minimization. Set to NULL if don't care
                      */
  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  double *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */

  void *adata);       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */

#endif /* HAVE_CUDA */
/****************************** robustlm.c ****************************/
/* robust, iteratively weighted non linear least squares using LM 
  entirely in the GPU */
#ifdef HAVE_CUDA
extern int
rlevmar_der_single_cuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  double *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  void *adata);

/* robust, iteratively weighted non linear least squares using LM 
  entirely in the GPU, using float data */
int
rlevmar_der_single_cuda_fl(
  float *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  float *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  float *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  void *adata); 

/* robust, iteratively weighted non linear least squares using LM 
  entirely in the GPU, using float data, OS acceleration */
extern int
osrlevmar_der_single_cuda_fl(
  float *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  float *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  float *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  int randomize, /* if >0 randomize */
  void *adata);
#endif /* HAVE_CUDA */

#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
rlevmar_der_single_nocuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int Nt, /* no of threads */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  void *adata);

/* robust LM, OS acceleration */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
osrlevmar_der_single_nocuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int Nt, /* no of threads */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  int randomize, /* if >0 randomize */
  void *adata);

/****************************** updatenu.c ****************************/
/* update nu (degrees of freedom)

   nu0: current value of nu (need for AECM update)
   sumlogw = 1/N sum(log(w_i)-w_i)
   use Nd values in [nulow,nuhigh] to find nu
   p: 1 or 8 depending on scalar or 2x2 matrix formulation
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double
update_nu(double sumlogw, int Nd, int Nt, double nulow, double nuhigh, int p, double nu0);

/* update w and nu together 
   nu0: current value of nu
   w: Nx1 weight vector
   ed: Nx1 error vector
   Nt: no of threads

   return new nu, w is also updated, search range [nulow,nuhigh]
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double
update_w_and_nu(double nu0, double *w, double *ed, int N, int Nt,  double nulow, double nuhigh);

/* 
  taper data by weighting based on uv distance (for short baselines)
  for example: use weights as the inverse density function
  1/( 1+f(u,v) ) 
 as u,v->inf, f(u,v) -> 0 so long baselines are not affected 
 x: Nbase*8 x 1 (input,output) data
 u,v : Nbase x 1
 note: u = u/c, v=v/c here, so need freq to convert to wavelengths */
extern void
whiten_data(int Nbase, double *x, double *u, double *v, double freq0, int Nt);
/****************************** clmfit_nocuda.c ****************************/
/* LM with LAPACK */
/** keep interface almost the same as in levmar **/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
clevmar_der_single_nocuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  void *adata); /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                * Set to NULL if not needed
                */

extern int
mlm_der_single(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[6],   /* I: minim. options [\mu, \m, \p0, \p1, \p2, \delta].
                        delta: 1 or 2
                       */
  double info[10], 
                      /* O: information regarding the minimization. Set to NULL if don't care
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  void *adata);     /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */

#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
oslevmar_der_single_nocuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int randomize, /* if >0 randomize */
  void *adata);
/****************************** oslmfit.c ****************************/
#ifdef HAVE_CUDA
/* OS-LM, but f() and jac() calculations are done 
  entirely in the GPU */
extern int
oslevmar_der_single_cuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  double *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  int randomize, /* if >0 randomize */
  void *adata); /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */

#endif /* !HAVE_CUDA */

/****************************** clmfit_fl.c ****************************/
#ifdef HAVE_CUDA
extern int
clevmar_der_single_cuda_fl(
  float *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  float *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  float *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  void *adata); /* pointer to possibly additional data  */

extern int
oslevmar_der_single_cuda_fl(
  float *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  float *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10],
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  float *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  int randomize, /* if >0 randomize */
  void *adata); /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */

#endif /* !HAVE_CUDA */
/****************************** rtr_solve.c ****************************/
/* structure for worker threads for function calculation */
typedef struct thread_data_rtr_ {
  int Nb; /* no of baselines this handle */
  int boff; /* baseline offset per thread */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  double *y; /* data vector Nbx8 array re,im,re,im .... */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  /* following are only used while predict with gain */
  complex double *x; /* parameter array, */
 /* general format of element in manifold x
   x: size 4N x 1 vector
   x[0:2N-1] : first column, x[2N:4N-1] : second column
   x=[J_1(1,1)  J_1(1,2);
      J_1(2,1)  J_1(2,2);
      ...       ....
      J_N(1,1)  J_N(1,2);
      J_N(2,1)  J_N(2,2)];
  */
  int N; /* no of stations */
  int clus; /* which cluster to process, 0,1,...,M-1 if -1 all clusters */
  
  /* output of cost function */
  double fcost;
  /* gradient */
  complex double *grad;
  /* Hessian */
  complex double *hess;
  /* Eta (used in Hessian) */
  complex double *eta;

  /* normalization factors for grad,hess calculation */
  /* size Nx1 */
  int *bcount;
  double *iw; /* 1/bcount */

  /* for robust solver */
  double *wtd; /* weights for baseline */
  double nu0;

  /* mutexs: N x 1, one for each station */
  pthread_mutex_t *mx_array;
} thread_data_rtr_t;

/* structure for common data */
typedef struct global_data_rtr_ {
  me_data_t *medata; /* passed from caller */

  /* normalization factors for grad,hess calculation */
  /* size Nx1 */
  double *iw; /* 1/bcount */

  /* for robust solver */
  double *wtd; /* weights for baseline */
  double nulow,nuhigh;

  /* for ADMM cost */
  complex double *Y; /* size 2Nx2 */
  complex double *BZ; /* size 2Nx2 */
  double admm_rho;

  /* thread stuff  Nt x 1 threads */
  pthread_t *th_array;
  /* mutexs: N x 1, one for each station */
  pthread_mutex_t *mx_array;
  pthread_attr_t attr;
} global_data_rtr_t;


/* RTR (ICASSP 2013) */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
rtr_solve_nocuda(
  double *x,         /* initial values and updated solution at output (size 8*N double) */
  double *y,         /* data vector (size 8*M double) */
  int N,              /* no. of stations */
  int M,              /* no. of constraints */
  int itmax_sd,          /* maximum number of iterations RSD */
  int itmax_rtr,          /* maximum number of iterations RTR */
  double Delta_bar, double Delta0, /* Trust region radius and initial value */
  double *info, /* initial and final residuals */
  me_data_t *adata); /* pointer to additional data
                */

/****************************** rtr_solve_robust.c ****************************/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
rtr_solve_nocuda_robust(
  double *x0,         /* initial values and updated solution at output (size 8*N double) */
  double *y,         /* data vector (size 8*M double) */
  int N,              /* no. of stations */
  int M,              /* no. of constraints */
  int itmax_rsd,          /* maximum number of iterations RSD */
  int itmax_rtr,          /* maximum number of iterations RTR */
  double Delta_bar, double Delta0, /* Trust region radius and initial value */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  double *info, /* initial and final residuals */
  me_data_t *adata);

/* Nesterov's SD */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
nsd_solve_nocuda_robust(
  double *x,         /* initial values and updated solution at output (size 8*N double) */
  double *y,         /* data vector (size 8*M double) */
  int N,              /* no. of stations */
  int M,              /* no. of constraints */
  int itmax,          /* maximum number of iterations */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  double *info, /* initial and final residuals */
  me_data_t *adata); /* pointer to additional data
                */

/****************************** rtr_solve_robust_admm.c ****************************/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
rtr_solve_nocuda_robust_admm(
  double *x0,         /* initial values and updated solution at output (size 8*N double) */
  double *Y,         /* Lagrange multiplier (size 8*N double) */
  double *BZ,         /* consensus B Z (size 8*N double) */
  double *y,         /* data vector (size 8*M double) */
  int N,              /* no. of stations */
  int M,              /* no. of constraints */
  int itmax_rsd,          /* maximum number of iterations RSD */
  int itmax_rtr,          /* maximum number of iterations RTR */
  double Delta_bar, double Delta0, /* Trust region radius and initial value */
  double admm_rho, /* ADMM regularization value */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  double *info, /* initial and final residuals */
  me_data_t *adata);
#ifdef HAVE_CUDA
/****************************** manifold_fl.cu ****************************/
extern float 
cudakernel_fns_f(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh);
extern void
cudakernel_fns_fgradflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh);
extern void
cudakernel_fns_fhessflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh);
extern void
cudakernel_fns_fscale(int N, cuFloatComplex *eta, float *iw);
extern float 
cudakernel_fns_f_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh,  float *wtd);
extern void
cudakernel_fns_fgradflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
extern void
cudakernel_fns_fgradflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd);
extern void
cudakernel_fns_fgradflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
extern void
cudakernel_fns_fhessflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd);
extern void
cudakernel_fns_fhessflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
extern void
cudakernel_fns_fhessflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
extern void
cudakernel_fns_fupdate_weights(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float nu0);
extern void
cudakernel_fns_fupdate_weights_q(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float *qd, float nu0);
/****************************** rtr_solve_cuda.c ****************************/
extern int
rtr_solve_cuda_fl(
  float *x,         /* initial values and updated solution at output (size 8*N float) */
  float *y,         /* data vector (size 8*M float) */
  int N,              /* no of stations */
  int M,              /* no of constraints */
  int itmax_sd,          /* maximum number of iterations RSD */
  int itmax_rtr,          /* maximum number of iterations RTR */
  float Delta_bar, float Delta0, /* Trust region radius and initial value */
  double *info, /* initial and final residuals */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  me_data_t *adata); /* pointer to possibly additional data  */


extern void
cudakernel_fns_R(int N, cuFloatComplex *x, cuFloatComplex *r, cuFloatComplex *rnew, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
extern float
cudakernel_fns_g(int N,cuFloatComplex *x,cuFloatComplex *eta, cuFloatComplex *gamma,cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
extern void
cudakernel_fns_proj(int N, cuFloatComplex *x, cuFloatComplex *z, cuFloatComplex *rnew, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
/****************************** rtr_solve_robust_cuda.c ****************************/
extern int
rtr_solve_cuda_robust_fl(
  float *x0,         /* initial values and updated solution at output (size 8*N float) */
  float *y,         /* data vector (size 8*M float) */
  int N,              /* no of stations */
  int M,              /* no of constraints */
  int itmax_sd,          /* maximum number of iterations RSD */
  int itmax_rtr,          /* maximum number of iterations RTR */
  float Delta_bar, float Delta0, /* Trust region radius and initial value */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  double *info, /* initial and final residuals */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  me_data_t *adata);


/* Nesterov's steepest descent */
extern int
nsd_solve_cuda_robust_fl(
  float *x0,         /* initial values and updated solution at output (size 8*N float) */
  float *y,         /* data vector (size 8*M float) */
  int N,              /* no of stations */
  int M,              /* no of constraints */
  int itmax,          /* maximum number of iterations */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  double *info, /* initial and final residuals */
  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  me_data_t *adata);

/****************************** rtr_solve_robust_cuda_admm.c ****************************/
/* ADMM solver */
extern int
rtr_solve_cuda_robust_admm_fl(
  float *x0,         /* initial values and updated solution at output (size 8*N float) */
  float *Y, /* Lagrange multiplier size 8N */
  float *Z, /* consensus term B Z  size 8N */
  float *y,         /* data vector (size 8*M float) */
  int N,              /* no of stations */
  int M,              /* no of constraints */
  int itmax_rtr,          /* maximum number of iterations */
  float Delta_bar, float Delta0, /* Trust region radius and initial value */
  float admm_rho, /* ADMM regularization */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  double *info, /* initial and final residuals */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  me_data_t *adata);


/* Nesterov's SD */
extern int
nsd_solve_cuda_robust_admm_fl(
  float *x0,         /* initial values and updated solution at output (size 8*N float) */
  float *Y, /* Lagrange multiplier size 8N */
  float *Z, /* consensus term B Z  size 8N */
  float *y,         /* data vector (size 8*M float) */
  int N,              /* no of stations */
  int M,              /* no of constraints */
  int itmax,          /* maximum number of iterations */
  float admm_rho, /* ADMM regularization */
  double robust_nulow, double robust_nuhigh, /* robust nu range */
  double *info, /* initial and final residuals */
  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  me_data_t *adata);
#endif /* HAVE_CUDA */
/****************************** lmfit.c ****************************/
/****************************** lmfit_nocuda.c ****************************/
/* struct for calling parallel LM jobs */
typedef struct thread_clm_data_t {
  double *p; /* parameters */
  double *x; /* data */
  int M;
  int N;
  int itermax;
  double *opts;
  double *info;
  int card;
  int linsolv;
  me_data_t *lmdata;
} thread_clm_data;


/* generate a random permutation of given integers */
/* note: free returned value after use */
/* n: no of entries, 
   weighter_iter: if 1, take weight into account
                  if 0, only generate a random permutation
   w: weights (size nx1): sort them in descending order and 
      give permutation accordingly
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int*
random_permutation(int n, int weighted_iter, double *w);

/********* solver modes *********/
#define SM_LM_LBFGS 1
#define SM_OSLM_LBFGS 0
#define SM_OSLM_OSRLM_RLBFGS 3
#define SM_RLM_RLBFGS 2
#define SM_RTR_OSLM_LBFGS 4
#define SM_RTR_OSRLM_RLBFGS 5
#define SM_NSD_RLBFGS 6
/* fit visibilities
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: data to write size Nbase*8*tileze x 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  N: no of stations
  Nbase: no of baselines
  tilesz: tile size
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  coh: coherencies size Nbase*tilesz*4*M x 1
  M: no of clusters
  Mt: actual no of cluster/parameters (for hybrid solutions) Mt>=M
  freq0: frequency
  fdelta: bandwidth for freq smearing
  pp: parameter array 8*N*M x1 double values (re,img) for each station/direction
  uvmin: baseline length sqrt(u^2+v^2) below which not to include in solution
  Nt: no. of threads
  max_emiter: EM iterations
  max_iter: iterations within a single EM 
  max_lbfgs: LBFGS iterations (if>0 outside minimization will be LBFGS)
  lbfgs_m: memory size for LBFGS
  gpu_threads: GPU threads per block (LBFGS)
  linsolv: (GPU/CPU versions) 0: Cholesky, 1: QR, 2: SVD
  solver_mode:  0: OS-LM, 1: LM , 2: OS-Robust LM, 3: Robust LM, 4: OS-LM + RTR, 5: OS-LM, RTR, OS-Robust LM
  nulow,nuhigh: robust nu search range
  randomize: if >0, randomize cluster selection in SAGE and OS subset selection

  mean_nu: output mean value of nu
  res_0,res_1: initial and final residuals (output)
  return val=0 if final residual< initial residual
  return val=-1 if final residual>initial residual
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
sagefit_visibilities(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt,int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv, int solver_mode, double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1); 

/* same as above, but uses 2 GPUS in the LM stage */
extern int
sagefit_visibilities_dual(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt,int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv, double nulow, double nuhigh, int randomize,  double *mean_nu, double *res_0, double *res_1); 



#ifdef USE_MIC
/* wrapper function with bitwise copyable carr[] for MIC */
/* nchunks: Mx1 array of chunk sizes for each cluster */
/* pindex: Mt x 1 array of index of solutions for each cluster  in pp */
__attribute__ ((target(MIC)))
extern int
sagefit_visibilities_mic(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *mean_nu, double *res_0, double *res_1);

__attribute__ ((target(MIC)))
extern int
bfgsfit_visibilities_mic(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode,double nu_mean, double *res_0, double *res_1);
#endif


/* BFGS only fit for multi channel data, interface same as sagefit_visibilities_xxx 
  NO EM iterations are taken  */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
bfgsfit_visibilities(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double mean_nu, double *res_0, double *res_1); 


extern int
bfgsfit_visibilities_gpu(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode,  double mean_nu, double *res_0, double *res_1); 



/* struct to keep histoty of last used GPU */
typedef struct taskhist_{
  int prev; /* last used GPU (by any thread) */
  pthread_mutex_t prev_mutex; /* mutex to lock changing prev value */
  unsigned int rseed; /* random seed used in rand_r() */
} taskhist;

/* structs for thread pool (reusable), using a barrier */
/* slave thread data struct */
typedef struct slave_tdata_ {
  struct pipeline_ *pline; /* forward declaration */
  int tid; /* 0,1 for 2 GPUs */
} slave_tdata;

/* pipeline struct */
typedef struct pipeline_ {
  void *data; /* all data needed by two threads */
  int terminate; /* 1: terminate, default 0*/
  pthread_t slave0;
  pthread_t slave1;
  slave_tdata *sd0; /* note recursive types */
  slave_tdata *sd1;
  th_barrier gate1;
  th_barrier gate2;
  pthread_attr_t attr;
  taskhist *thst;
} th_pipeline;

/* pipeline state values */
#ifndef PT_DO_NOTHING
#define PT_DO_NOTHING 0
#endif
#ifndef PT_DO_AGPU
#define PT_DO_AGPU 1 /* allocate GPU memory, attach GPU */
#endif
#ifndef PT_DO_DGPU
#define PT_DO_DGPU 2 /* free GPU memory, detach GPU */
#endif
#ifndef PT_DO_WORK_LM /* plain LM */
#define PT_DO_WORK_LM 3
#endif
#ifndef PT_DO_WORK_OSLM /* OS accel LM */
#define PT_DO_WORK_OSLM 4
#endif
#ifndef PT_DO_WORK_RLM /* robust LM */
#define PT_DO_WORK_RLM 5
#endif
#ifndef PT_DO_WORK_OSRLM /* robust LM, OS accel */
#define PT_DO_WORK_OSRLM 6
#endif
#ifndef PT_DO_WORK_RTR /* RTR */
#define PT_DO_WORK_RTR 7
#endif
#ifndef PT_DO_WORK_RRTR /* Robust RTR */
#define PT_DO_WORK_RRTR 8
#endif
#ifndef PT_DO_WORK_NSD /* Nesterov's SD */
#define PT_DO_WORK_NSD 9
#endif
#ifndef PT_DO_MEMRESET 
#define PT_DO_MEMRESET 99
#endif
/* for BFGS pipeline */
#ifndef PT_DO_CDERIV
#define PT_DO_CDERIV 20
#endif
#ifndef PT_DO_CCOST
#define PT_DO_CCOST 21
#endif



#ifdef HAVE_CUDA
/* data struct shared by all threads */
typedef struct gb_data_ {
  int status[2]; /* 0: do nothing, 
              1: allocate GPU  memory, attach GPU
              2: free GPU memory, detach GPU 
              3,4..: do work on GPU 
              99: reset GPU memory (memest all memory) */
  double *p[2]; /* pointer to parameters being solved by each thread */
  double *x[2]; /* pointer to data being fit by each thread */
  int M[2];
  int N[2];
  int itermax[2];
  double *opts[2];
  double *info[2];
  int linsolv;
  me_data_t *lmdata[2]; /* two for each thread */

  /* GPU related info */
  cublasHandle_t cbhandle[2]; /* CUBLAS handles */
  cusolverDnHandle_t solver_handle[2]; /* solver handle */
  double *gWORK[2]; /* GPU buffers */
  int64_t data_size; /* size of buffer (bytes) */

  double nulow,nuhigh; /* used only in robust version */
  int randomize; /* >0 for randomization */
} gbdata;

/* same as above, but using floats */
typedef struct gb_data_fl_ {
  int status[2]; /* 0: do nothing, 
              1: allocate GPU  memory, attach GPU
              3: free GPU memory, detach GPU 
              3,4..: do work on GPU 
              99: reset GPU memory (memest all memory) */
  float *p[2]; /* pointer to parameters being solved by each thread */
  float *x[2]; /* pointer to data being fit by each thread */
  int M[2];
  int N[2];
  int itermax[2];
  double *opts[2];
  double *info[2];
  int linsolv;
  me_data_t *lmdata[2]; /* two for each thread */

  /* GPU related info */
  cublasHandle_t cbhandle[2]; /* CUBLAS handles */
  cusolverDnHandle_t solver_handle[2]; /* solver handle */
  float *gWORK[2]; /* GPU buffers */
  int64_t data_size; /* size of buffer (bytes) */

  double nulow,nuhigh; /* used only in robust version */
  int randomize; /* >0 for randomization */
} gbdatafl;

/* for ADMM solver */
typedef struct gb_data_admm_fl_ {
  int status[2]; /* 0: do nothing, 
              1: allocate GPU  memory, attach GPU
              3: free GPU memory, detach GPU 
              3,4..: do work on GPU 
              99: reset GPU memory (memest all memory) */
  float *p[2]; /* pointer to parameters being solved by each thread */
  float *Y[2]; /* pointer to Lagrange multiplier */
  float *Z[2]; /* pointer to consensus term */
  float admm_rho[2];
  float *x[2]; /* pointer to data being fit by each thread */
  int M[2];
  int N[2];
  int itermax[2];
  double *opts[2];
  double *info[2];
  int linsolv;
  me_data_t *lmdata[2]; /* two for each thread */

  /* GPU related info */
  cublasHandle_t cbhandle[2]; /* CUBLAS handles */
  cusolverDnHandle_t solver_handle[2]; /* solver handle */
  float *gWORK[2]; /* GPU buffers */
  int64_t data_size; /* size of buffer (bytes) */

  double nulow,nuhigh; /* used only in robust version */
  int randomize; /* >0 for randomization */
} gbdatafl_admm;


#endif /* !HAVE_CUDA */

/* with 2 GPUs */
extern int
sagefit_visibilities_dual_pt(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt,int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv, int solver_mode, double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1); 

/* with 1 GPU and 1 CPU thread */
extern int
sagefit_visibilities_dual_pt_one_gpu(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1);

/* with mixed precision */
extern int
sagefit_visibilities_dual_pt_flt(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1);

/****************************** diagnostics.c ****************************/
#ifdef HAVE_CUDA
/*  Calculate St.Laurent-Cook Jacobian leverage
x: input: residual, output: levarage
  flags: 2 for flags based on uvcut, 1 for normal flags
  coh: coherencies are calculated for all baselines, regardless of flag
  diagmode: 1: replaces residual with Jacobian Leverage, 2: calculates (prints) fraction of leverage/noise
 */
extern int
calculate_diagnostics(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, complex double *coh, int M,int Mt,int diagmode,int Nt);
#endif


/****************************** diag_fl.cu ****************************/
#ifdef HAVE_CUDA
/* cuda driver for calculating Jacobian for leverage */
/* p: params (Mx1), jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
/* flags are always ignored */
extern void
cudakernel_jacf_fl2(float *p, float *jac, int M, int N, float *coh, short *bbh, int Nbase, int Mclus, int Nstations);

/* invert sqrt(singular values)  Sd[]=1/sqrt(Sd[])  for Sd[]> eps */
extern void
cudakernel_sqrtdiv_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float eps, float *Sd);

/* U <= U D, 
   U : MxM
   D : Mx1, diagonal matrix
*/
extern void
cudakernel_diagmult_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float * U, float *D); 

/* diag(J^T J)
   d[i] = J[i,:] * J[i,:]
   J: NxM (in row major order, so J[i,:] is actually J[:,i]
   d: Nx1
*/
extern void
cudakernel_jnorm_fl(int ThreadsPerBlock, int BlocksPerGrid, float *J, int N, int M, float *d);
#endif


/****************************** manifold_average.c ****************************/
/* calculate manifold average of 2Nx2 solution blocks,
   then project each solution to this average 
   Y: 2Nx2 x M x Nf values (average calculated for each 2Nx2 x Nf blocks)
   N: no of stations
   M: no of directions
   Nf: no of frequencies
   Niter: everaging iterations
   Nt: threads
*/
extern int
calculate_manifold_average(int N,int M,int Nf,double *Y,int Niter,int Nt);


/* find U to  minimize 
  ||J-J1 U|| solving Procrustes problem 
  J,J1 : 8N x 1 vectors, in standard format
  will be reshaped to 2Nx2 format and J1 will be modified as J1 U
*/
extern int
project_procrustes(int N,double *J,double *J1);

/* same as above, but J,J1 are in right 2Nx2 matrix format */
/* J1 is modified */
extern int
project_procrustes_block(int N,complex double *J,complex double *J1);


/* Extract only the phase of diagonal entries from solutions 
   p: 8Nx1 solutions, orders as [(real,imag)vec(J1),(real,imag)vec(J2),...]
   pout: 8Nx1 phases (exp(j*phase)) of solutions, after joint diagonalization of p
   N: no. of 2x2 Jones matrices in p, having common unitary ambiguity
   niter: no of iterations for Jacobi rotation */
extern int
extract_phases(double *p, double *pout, int N, int niter);
/****************************** consensus_poly.c ****************************/
/* build matrix with polynomial terms
  B : Npoly x Nf, each row is one basis function
  Npoly : total basis functions
  Nf: frequencies
  freqs: Nfx1 array freqs
  freq0: reference freq
  type : 0 for [1 ((f-fo)/fo) ((f-fo)/fo)^2 ...] basis functions
     1 : same as type 0, normalize each row such that norm is 1
     2 : Bernstein poly \sum N_C_r x^r (1-x)^r where x in [0,1] : use min,max values of freq to normalize
*/
extern int
setup_polynomials(double *B, int Npoly, int Nf, double *freqs, double freq0, int type);

/* build matrix with polynomial terms
  B : Npoly x Nf, each row is one basis function
  Bi: Npoly x Npoly pseudo inverse of sum( B(:,col) x B(:,col)' )
  Npoly : total basis functions
  Nf: frequencies
  fratio: Nfx1 array of weighing factors depending on the flagged data of each freq
  Sum taken is a weighted sum, using weights in fratio
*/
extern int
find_prod_inverse(double *B, double *Bi, int Npoly, int Nf, double *fratio);

/* build matrix with polynomial terms
  B : Npoly x Nf, each row is one basis function
  Bi: Npoly x Npoly pseudo inverse of sum( B(:,col) x B(:,col)' ) : M times
  Npoly : total basis functions
  Nf: frequencies
  M: clusters
  rho: NfxM array of regularization factors (for each freq, M values)
  Sum taken is a weighted sum, using weights in rho, rho is assumed to change for each freq,cluster pair 

  Nt: no. of threads
*/
extern int
find_prod_inverse_full(double *B, double *Bi, int Npoly, int Nf, int M, double *rho, int Nt);

/* update Z
   Z: 8NxNpoly x M double array (real and complex need to be updated separate)
   N : stations
   M : clusters
   Npoly: no of basis functions
   z : right hand side 8NMxNpoly (note the different ordering from Z)
   Bi : NpolyxNpoly matrix, Bi^T=Bi assumed
*/
extern int
update_global_z(double *Z,int N,int M,int Npoly,double *z,double *Bi);

/* update Z
   Z: 8N Npoly x M double array (real and complex need to be updated separate)
   N : stations
   M : clusters
   Npoly: no of basis functions
   z : right hand side 8NM Npoly x 1 (note the different ordering from Z)
   Bi : M values of NpolyxNpoly matrices, Bi^T=Bi assumed

   Nt: no. of threads
*/
extern int 
update_global_z_multi(double *Z,int N,int M,int Npoly,double *z,double *Bi, int Nt);


/* generate a random integer in the range 0,1,...,maxval */
extern int
random_int(int maxval);


extern int
update_rho_bb(double *rho, double *rhoupper, int N, int M, int Mt, clus_source_t *carr, double *Yhat, double *Yhat_k0, double *J, double *J_k0, int Nt);

/****************************** admm_solve.c ****************************/
/* ADMM cost function  = normal_cost + ||Y^H(J-BZ)|| + rho/2 ||J-BZ||^2 */
/* extra params
   Y : Lagrange multiplier
   BZ : consensus term
   Y,BZ : size same as pp : 8*N*Mt x1 double values (re,img) for each station/direction 
   admm_rho : regularization factor array size Mx1
*/ 
extern int
sagefit_visibilities_admm(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *admm_rho, double *mean_nu, double *res_0, double *res_1);

/* ADMM cost function  = normal_cost + ||Y^H(J-BZ)|| + rho/2 ||J-BZ||^2 */
/* extra params
   Y : Lagrange multiplier
   BZ : consensus term
   Y,BZ : size same as pp : 8*N*Mt x1 double values (re,img) for each station/direction 
   admm_rho : regularization factor  array size Mx1
*/ 
#ifdef HAVE_CUDA
extern int
sagefit_visibilities_admm_dual_pt_flt(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *admm_rho, double *mean_nu, double *res_0, double *res_1);
#endif


/****************************** load_balance.c ****************************/
/* select a GPU from 0,1..,max_gpu
   in such a way to allow load balancing */
/* also keep a global variableto ensure same GPU is 
   not assigned to one process */
#ifdef HAVE_CUDA
extern void
init_task_hist(taskhist *th);
extern void
destroy_task_hist(taskhist *th);

extern int
select_work_gpu(int max_gpu, taskhist *th);
#endif
/****************************** OpenBLAS ************************************/
/* prototype declaration */
extern void
openblas_set_num_threads(int num_threads);

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
#ifdef HAVE_CUDA
/* if dobeam==0, beam calculation is off */
extern int
precalculate_coherencies_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tileze, int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt);


extern int
predict_visibilities_multifreq_withbeam_gpu(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt, int add_to_data);

extern int
calculate_residuals_multifreq_withbeam_gpu(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt, int ccid, double rho, int phase_only);



#endif /*!HAVE_CUDA */


/****************************** predict_model.cu ****************************/
#ifdef HAVE_CUDA

#ifndef ARRAY_USE_SHMEM /* use shared memory for calculation station beam */
#define ARRAY_USE_SHMEM 1
#endif
#ifndef ARRAY_MAX_ELEM /* if using shared memory, max possible elements for a station */
#define ARRAY_MAX_ELEM 512
#endif
/* default GPU heap size (in MB) needed to calculate some shapelet models,
    if model has n0>20 or so, try increasing this and recompiling
   the default GPU values is ~ 8MB */
#ifndef GPU_HEAP_SIZE
#define GPU_HEAP_SIZE 20 
#endif


extern void
cudakernel_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, int *Nelem, float **xx, float **yy, float **zz, float *ra, float *dec, float ph_ra0, float  ph_dec0, float ph_freq0, float *beam);


extern void
cudakernel_coherencies(int B, int N, int T, int K, int F, double *u, double *v, double *w,baseline_t *barr, double *freqs, float *beam, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
  unsigned char *stype, double *sI0, double *sQ0, double *sU0, double *sV0, double *f0, double *spec_idx, double *spec_idx1, double *spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh, int dobeam);

extern void
cudakernel_residuals(int B, int N, int T, int K, int F, double *u, double *v, double *w, double *p, int nchunk, baseline_t *barr, double *freqs, float *beam, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
  unsigned char *stype, double *sI0, double *sQ0, double *sU0, double *sV0, double *f0, double *spec_idx, double *spec_idx1, double *spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh,int dobeam);

/* B: total baselines
   N: stations
   Nb: baselines worked by this kernel 
   boff: baseline offset 0..B-1
   F: frequencies
   nchunk: how many solutions
   x: residual to be corrected: size Nb*8*F
   p: solutions (inverted) size N*8*nchunk
   barr: baseline array to get station indices: size Nb
*/
extern void
cudakernel_correct_residuals(int B, int N, int Nb, int boff, int F, int nchunk, double *x, double *p, baseline_t *barr);

extern void
cudakernel_convert_time(int T, double *time_utc);
#endif /* !HAVE_CUDA */


/****************************** mdl.c ****************************/
/*
  change polynomial order from Kstart to Kfinish
  evaluate Z for each poly order, then find MDL
   N: stations
   M: clusters
   F: frequencies
   J: weightxrhoxJ solutions (note: not true J, but J scaled by each slaves' rho), 8NMxF blocks
   rho: regularization, no weighting applied, Mx1 
   freqs: frequencies, Fx1
   freq0: reference freq
   weight: weight for each freq, based on flagged data, Fx1
   polytype: type of polynomial
  Kstart, Kfinish: range of order of polynomials to calculate the MDL
   Nt: no. of threads
*/
extern int
minimum_description_length(int N, int M, int F, double *J, double *rho, double *freqs, double freq0, double *weight, int polytype, int Kstart, int Kfinish, int Nt);


#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* SAGECAL_H */
