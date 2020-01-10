/*
 *
 Copyright (C) 2018 Sarod Yatawatta <sarod@users.sf.net>  
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

#ifndef Common_H
#define Common_H
#ifdef __cplusplus
        extern "C" {
#endif

/* simulation options */
#define SIMUL_ONLY 1 /* only predict model */
#define SIMUL_ADD 2 /* add to input */
#define SIMUL_SUB 3 /* subtract from input */

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
  int Nb; /* no of baselines this thread handles */
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

  /* following used for interpolation,stochastic calibration */
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

  /* following for calibration of multi channel data */
  int Nchan;

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


  /* only used in multifreq data */
  int Nchan;

  /* only used in batch mode operation */
  int noff; /* offset of the batch data, in baselines */
  int nlen; /* size of the batch, in baselines */
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


typedef struct t_barrier_ {
  int tcount; /* current no. of threads inside barrier */
  int nthreads; /* the no. of threads the barrier works
                with. This is a constant */
  pthread_mutex_t enter_mutex;
  pthread_mutex_t exit_mutex;
  pthread_cond_t lastthread_cond;
  pthread_cond_t exit_cond;
} th_barrier;


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


/****************************** predict.c ****************************/
/* rearranges coherencies for GPU use later */
/* barr: 2*Nbase x 1
 * coh: M*Nbase*4 x 1 complex
 * ddcoh: M*Nbase*8 x 1
 * ddbase: 2*Nbase x 1 (sta1,sta2) = -1 if flagged
 * */
extern int
rearrange_coherencies(int Nbase, baseline_t *barr, complex double *coh, double *ddcoh, short *ddbase, int M, int Nt);

/* rearranges baselines for GPU use later */
/* barr: 2*Nbase x 1
 * ddbase: 2*Nbase x 1
 * */
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

/* BLAS y = a.x + y with different strides in x and y given by cx and cy */
extern void
my_daxpy_inc(int N, double *x, int cx, double a, double *y, int cy);

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

/* following routines used in LAPACK dirac */
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

/****************************** manifold_average.c ****************************/
/* Extract only the phase of diagonal entries from solutions 
   p: 8Nx1 solutions, orders as [(real,imag)vec(J1),(real,imag)vec(J2),...]
   pout: 8Nx1 phases (exp(j*phase)) of solutions, after joint diagonalization of p
   N: no. of 2x2 Jones matrices in p, having common unitary ambiguity
   niter: no of iterations for Jacobi rotation */
extern int
extract_phases(double *p, double *pout, int N, int niter);

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

/****************************** barrier.c ****************************/

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


#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* Common_H */
