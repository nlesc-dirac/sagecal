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

/****************************** OpenBLAS ************************************/
/* prototype declaration */
extern void
openblas_set_num_threads(int num_threads);

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
