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

#ifdef HAVE_CUDA
// new version
#include <cula_lapack_device.h>
#include <cula_blas_device.h>
//#include <cuda_runtime.h> /* comment this out to link with g++ */
#include <cublas_v2.h>
#endif /* HAVE_CUDA */


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

/* macros for version info */
#ifndef CLMV
#define CLMV "LM"
#endif
#ifndef CLBFGSV
#define CLBFGSV "LBFGS"
#endif


/* soure types */
#define STYPE_POINT 0
#define STYPE_GAUSSIAN 1
#define STYPE_DISK 2
#define STYPE_RING 3
#define STYPE_SHAPELET 4

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
 double ll,mm,sI; /* note sI is updated for central freq */
 unsigned char stype; /* source type */
 void *exdata; /* pointer to carry additional data, if needed */
 double sI0,f0,spec_idx,spec_idx1,spec_idx2; /* for multi channel data, original sI, f0 and spectral index */
} sinfo_t;

/* struct for array of the sky model, with clusters */
typedef struct clus_source_t_ {
 int N; /* no of source in this cluster */
 int id; /* cluster id */
 double *ll,*mm,*nn,*sI; /* arrays Nx1 of source info */
 /* nn=sqrt(1-ll^2-mm^2)-1 */
 unsigned char *stype; /* source type array Nx1 */
 void **ex; /* array for extra source information Nx1 */

 int nchunk; /* no of chunks the data is divided for solving */
 int *p; /* array nchunkx1 points to parameter array indices */


 double *sI0,*f0,*spec_idx,*spec_idx1,*spec_idx2; /* for multi channel data, original sI, f0 and spectral index */
} clus_source_t;

/* strutct to store baseline to station mapping */
typedef struct baseline_t_ {
 int sta1,sta2;
 unsigned char flag; /* if this baseline is flagged, set to 1, otherwise 0: 
             special case: 2 if baseline is not used in solution, but will be
              subtracted */
} baseline_t;


/* structure for worker threads for function calculation */
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
  /* following used for freq smearing calculation */
  double freq0;
  double fdelta;

  /* following used for interpolation */
  double *p0; /* old parameters, same as p */
  int tilesz; /* tile size */
  int Nbase; /* total no of baselines */
  /* following for correction of data */
  double *pinv; /* inverted solution array, if null no correction */
  int ccid; /* which cluster id (not user specified id) for correction, >=0 */


  /* following used for multifrequency (channel) data */
  double *freqs;
  int Nchan;
} thread_data_base_t;


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
  char *ddbase; /* baseline to station maps, same as barr, assume no of stations < 127, if flagged set to -1 */
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
  char *ddbase; /* baseline to station maps, same as barr, size 2*Nbase*tilesz x 1, assume no of stations < 127, if flagged set to -1 */
  /* following used only by LBFGS */
  char *hbb; /*  baseline to station maps, same as ddbase size 2*Nbase*tilesz x 1, assume no of stations < 127, if flagged set to -1 */
  int *ptoclus; /* param no -> cluster mapping, size 2*M x 1 
      for each cluster : chunk size, start param index */

  /* following used only by mixed precision solver */
  float *ddcohf; /* float version of ddcoh */

  /* following used only by robust T cost/grad functions */
  double robust_nu;
} me_data_t;


/* structure for gpu driver threads for LBFGS */
typedef struct thread_gpu_data_t {
  int ThreadsPerBlock;
  int BlocksPerGrid;
  int card; /* which gpu ? 0 or 1 */
   
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

  char *hbb; /*  baseline to station maps, same as ddbase size 2*Nbase*tilesz x 1, assume no of stations < 127, if flagged set to -1 */
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
  Nt: no of threads
*/
extern int
predict_visibilities(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, int M, double freq0, double fdelta, int Nt); 
  

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
  uvmin: baseline length sqrt(u^2+v^2) below which not to include in solution
  Nt: no of threads

  NOTE: prediction is done for all baselines, even flagged ones
*/
extern int
precalculate_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double uvmin, int Nt); 



/* rearranges coherencies for GPU use later */
/* barr: 2*Nbase x 1
   coh: M*Nbase*4 x 1 complex
   ddcoh: M*Nbase*8 x 1
   ddbase: 2*Nbase x 1
*/
extern int
rearrange_coherencies(int Nbase, baseline_t *barr, complex double *coh, double *ddcoh, char *ddbase, int M, int Nt);

/* rearranges baselines for GPU use later */
/* barr: 2*Nbase x 1
   ddbase: 2*Nbase x 1
*/
extern int
rearrange_baselines(int Nbase, baseline_t *barr, char *ddbase, int Nt);

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
/****************************** myblas.c ****************************/
/* BLAS wrappers */
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

/* sum||x||_1 */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern double
my_dasum(int N, double *x);

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

/* max |x|  index (start from 1...)*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
extern int
my_idamax(int N, double *x, int incx);

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
   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads, void *adata);
#ifdef HAVE_CUDA
extern int
lbfgs_fit_robust_cuda(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads, void *adata);
#endif

/****************************** residual.c ****************************/
/* residual calculation, with linear interpolation */
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
  Nt: no. of threads
  ccid: which cluster to use as correction
  rho: MMSE robust parameter J+rho I inverted
*/
extern int
calculate_residuals(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double freq0,double fdelta,int Nt, int ccid, double rho);

/* 
  residuals for multiple channels
  data to write size Nbase*8*tilesz*Nchan x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots, channels
  input: x is actual data, output: x is the residual
  freqs: Nchanx1 of frequency values
  fdelta: total bandwidth, so divide by Nchan to get each channel bandwith
*/
extern int
calculate_residuals_multifreq(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,int Nt, int ccid, double rho);

/* 
  calculate visibilities for multiple channels, no solutions are used
  note: output column x is set to 0
*/
extern int
predict_visibilities_multifreq(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,int Nt);


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
cudakernel_lbfgs(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, char *bb, int *ptoclus, double *grad);


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
cudakernel_func(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, char *bbh, int Nbase, int Mclus, int Nstations);

/* cuda driver for calculating jacf() */
/* p: params (Mx1): for all chunks, jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, char *bbh, int Nbase, int Mclus, int Nstations);


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
cudakernel_func_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, char *bbh, int Nbase, int Mclus, int Nstations);
/* cuda driver for calculating jacf() */
/* p: params (Mx1), jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, char *bbh, int Nbase, int Mclus, int Nstations);
/****************************** robust.cu ****************************/
/* cuda driver for calculating wt \odot f() */
/* p: params (Mx1): for all chunks, x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations  */
extern void
cudakernel_func_wt(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, char *bbh, double *wt, int Nbase, int Mclus, int Nstations);

/* cuda driver for calculating wt \odot jacf() */
/* p: params (Mx1): for all chunks, jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf_wt(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, char *bbh, double *wt, int Nbase, int Mclus, int Nstations);


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
cudakernel_lbfgs_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double robust_nu, double *x, double *coh, double *p, char *bb, int *ptoclus, double *grad);

/****************************** robust_fl.cu ****************************/
/* cuda driver for calculating wt \odot f() */
/* p: params (Mx1): for all chunks, x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations  */
extern void
cudakernel_func_wt_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, char *bbh, float *wt, int Nbase, int Mclus, int Nstations);

/* cuda driver for calculating wt \odot jacf() */
/* p: params (Mx1): for all chunks, jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
extern void
cudakernel_jacf_wt_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, char *bbh, float *wt, int Nbase, int Mclus, int Nstations);


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

  int card,   /* device 0, 1 */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  void *adata); /* pointer to possibly additional data  */

/* function to set up a GPU, should be called only once */
extern void
attach_gpu_to_thread(int card, cublasHandle_t *cbhandle);
extern void
attach_gpu_to_thread1(int card, cublasHandle_t *cbhandle, double **WORK, int64_t work_size);
extern void
attach_gpu_to_thread2(int card,  cublasHandle_t *cbhandle,float **WORK, int64_t work_size);


/* function to detach a GPU from a thread */
extern void
detach_gpu_from_thread(cublasHandle_t cbhandle);
extern void
detach_gpu_from_thread1(int card, cublasHandle_t cbhandle, double *WORK);
extern void
detach_gpu_from_thread2(int card,cublasHandle_t cbhandle,float *WORK);
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
  float *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  int randomize, /* if >0 randomize */
  void *adata); /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */

#endif /* !HAVE_CUDA */
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
  solver_mode:  0: with OS, 1: No OS, 2: OS-Robust LM, 3: NO-OS Robust LM
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

/* structs for thread pool (reusable), using a barrier */
/* slave thread data struct */
typedef struct slave_tdata_ {
  struct pipeline_ *pline; /* forward declaration */
  int tid; /* 0,1 used as the GPU card */
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
#ifndef PT_DO_MEMRESET 
#define PT_DO_MEMRESET 99
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
  float *gWORK[2]; /* GPU buffers */
  int64_t data_size; /* size of buffer (bytes) */

  double nulow,nuhigh; /* used only in robust version */
  int randomize; /* >0 for randomization */
} gbdatafl;

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

#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* SAGECAL_H */
