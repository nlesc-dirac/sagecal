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


#include "Dirac.h"
#include <string.h> /* for memcpy */

/* machine precision */
double
dlamch(char CMACH) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern double dlamch_(char *CMACH);
  return(dlamch_(&CMACH));
}


/* blas dcopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
void
my_dcopy(int N, double *x, int Nx, double *y, int Ny) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dcopy_(int *N, double *x, int *incx, double *y, int *incy);
  /* use memcpy if Nx=Ny=1 */
  if (Nx==1&&Ny==1) {
   memcpy((void*)y,(void*)x,sizeof(double)*(size_t)N);
  } else {
   dcopy_(&N,x,&Nx,y,&Ny);
  }
}
/* blas scale */
/* x = a. x */
void
my_dscal(int N, double a, double *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dscal_(int *N, double *alpha, double *x, int *incx);
  int i=1;
  dscal_(&N,&a,x,&i);
}
void
my_sscal(int N, float a, float *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void sscal_(int *N, float *alpha, float *x, int *incx);
  int i=1;
  sscal_(&N,&a,x,&i);
}

/* x^T*y */
double
my_ddot(int N, double *x, double *y) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern double  ddot_(int *N, double *x, int *incx, double *y, int *incy);
  int i=1;
  return(ddot_(&N,x,&i,y,&i));
}

/* ||x||_2 */
double
my_dnrm2(int N, double *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern double  dnrm2_(int *N, double *x, int *incx);
  int i=1;
  return(dnrm2_(&N,x,&i));
}
float
my_fnrm2(int N, float *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern float snrm2_(int *N, float *x, int *incx);
  int i=1;
  return(snrm2_(&N,x,&i));
}



/* sum||x||_1 */
double
my_dasum(int N, double *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern double  dasum_(int *N, double *x, int *incx);
  int i=1;
  return(dasum_(&N,x,&i));
}
float
my_fasum(int N, float *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern float sasum_(int *N, float *x, int *incx);
  int i=1;
  return(sasum_(&N,x,&i));
}

/* BLAS y = a.x + y */
void
my_daxpy(int N, double *x, double a, double *y) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern void daxpy_(int *N, double *alpha, double *x, int *incx, double *y, int *incy);
    int i=1; /* strides */
    daxpy_(&N,&a,x,&i,y,&i);
}

/* BLAS y = a.x + y with different strides in x and y given by cx and cy */
void
my_daxpy_inc(int N, double *x, int cx, double a, double *y, int cy) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern void daxpy_(int *N, double *alpha, double *x, int *incx, double *y, int *incy);
    daxpy_(&N,&a,x,&cx,y,&cy);
}

/* BLAS y = a.x + y */
void
my_daxpys(int N, double *x, int incx, double a, double *y, int incy) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern void daxpy_(int *N, double *alpha, double *x, int *incx, double *y, int *incy);
    daxpy_(&N,&a,x,&incx,y,&incy);
}

void
my_saxpy(int N, float *x, float a, float *y) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern void saxpy_(int *N, float *alpha, float *x, int *incx, float *y, int *incy);
    int i=1; /* strides */
    saxpy_(&N,&a,x,&i,y,&i);
}



/* max |x|  index (start from 1...)*/
int
my_idamax(int N, double *x, int incx) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern int idamax_(int *N, double *x, int *incx);
    return idamax_(&N,x,&incx);
}

int
my_isamax(int N, float *x, int incx) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern int isamax_(int *N, float *x, int *incx);
    return isamax_(&N,x,&incx);
}

/* min |x|  index (start from 1...)*/
int
my_idamin(int N, double *x, int incx) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern int idamin_(int *N, double *x, int *incx);
    return idamin_(&N,x,&incx);
}

/* BLAS DGEMM C = alpha*op(A)*op(B)+ beta*C */
void
my_dgemm(char transa, char transb, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, double *ALPHA, double *A, int *LDA, double *B, int * LDB, double *BETA, double *C, int *LDC);
  dgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

/* BLAS DGEMV  y = alpha*op(A)*x+ beta*y : op 'T' or 'N' */
void
my_dgemv(char trans, int M, int N, double alpha, double *A, int lda, double *x, int incx,  double beta, double *y, int incy) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);
  dgemv_(&trans, &M, &N, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}


/* following routines used in LAPACK solvers */
/* cholesky factorization: real symmetric */
int
my_dpotrf(char uplo, int N, double *A, int lda) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dpotrf_(char *uplo, int *N, double *A, int *lda, int *info);
  int info;
  dpotrf_(&uplo,&N,A,&lda,&info);
  return info;
}

/* solve Ax=b using cholesky factorization */
int 
my_dpotrs(char uplo, int N, int nrhs, double *A, int lda, double *b, int ldb){
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
   extern void dpotrs_(char  *uplo, int *N, int *nrhs, double *A, int *lda, double *b, int *ldb, int *info);
   int info;
   dpotrs_(&uplo,&N,&nrhs,A,&lda,b,&ldb,&info);
   return info;
}

/* solve Ax=b using QR factorization */
int
my_dgels(char TRANS, int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *WORK, int LWORK) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dgels_(char *TRANS, int *M, int *N, int *NRHS, double *A, int *LDA, double *B, int *LDB, double *WORK, int *LWORK, int *INFO);
  int info;
  dgels_(&TRANS,&M,&N,&NRHS,A,&LDA,B,&LDB,WORK,&LWORK,&info);
  return info;
}


/* A=U S VT, so V needs NOT to be transposed */
int
my_dgesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S,
   double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
   extern void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, 
    int *LDA, double *S, double *U, int *LDU, double *VT, int *LDVT,
    double *WORK, int *LWORK, int *info);
   int info;
   dgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,&info);
   return info;
}

/* QR factorization QR=A, only TAU is used for Q, R stored in A*/
int
my_dgeqrf(int M, int N, double *A, int LDA, double *TAU, double *WORK, int LWORK) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
 extern void dgeqrf_(int *M, int *N, double *A,  int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);
  int info;
  dgeqrf_(&M,&N,A,&LDA,TAU,WORK,&LWORK,&info);
  return info;
}

/* calculate Q using elementary reflections */
int
my_dorgqr(int M,int  N,int  K,double *A,int  LDA,double *TAU,double *WORK,int  LWORK) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dorgqr_(int *M, int *N, int *K, double *A, int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);
  int info;
  dorgqr_(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &info);

  return info;
}

/* solves a triangular system of equations Ax=b, A triangular */
int
my_dtrtrs(char UPLO, char TRANS, char DIAG,int N,int  NRHS,double *A,int  LDA,double *B,int  LDB) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void dtrtrs_(char *UPLO,char *TRANS,char  *DIAG,int *N,int *NRHS,double *A,int *LDA,double *B,int *LDB,int *INFO);
  int info;
  dtrtrs_(&UPLO,&TRANS,&DIAG,&N,&NRHS,A,&LDA,B,&LDB,&info);

  return info;
}


/* blas ccopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
void
my_ccopy(int N, complex double *x, int Nx, complex double *y, int Ny) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void zcopy_(int *N, complex double *x, int *incx, complex double *y, int *incy);
  /* use memcpy if Nx=Ny=1 */
  if (Nx==1&&Ny==1) {
   memcpy((void*)y,(void*)x,sizeof(complex double)*(size_t)N);
  } else {
   zcopy_(&N,x,&Nx,y,&Ny);
  }
}

/* blas scale */
/* x = a. x */
void
my_cscal(int N, complex double a, complex double *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void zscal_(int *N, complex double *alpha, complex double *x, int *incx);
  int i=1;
  zscal_(&N,&a,x,&i);
}

/* BLAS y = a.x + y */
void
my_caxpy(int N, complex double *x, complex double a, complex double *y) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
    extern void zaxpy_(int *N, complex double *alpha, complex double *x, int *incx, complex double *y, int *incy);
    int i=1; /* strides */
    zaxpy_(&N,&a,x,&i,y,&i);
}


/* BLAS x^H*y */
complex double
my_cdot(int N, complex double *x, complex double *y) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern complex double  zdotc_(int *N, complex double *x, int *incx, complex double *y, int *incy);
  int i=1;
  return(zdotc_(&N,x,&i,y,&i));
}

/* A=U S VT, so V needs NOT to be transposed */
int
my_zgesvd(char JOBU, char JOBVT, int M, int N, complex double *A, int LDA, double *S,
   complex double *U, int LDU, complex double *VT, int LDVT, complex double *WORK, int LWORK, double *RWORK) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
   extern void zgesvd_(char *JOBU, char *JOBVT, int *M, int *N, complex double *A, 
    int *LDA, double *S, complex double *U, int *LDU, complex double *VT, int *LDVT,
    complex double *WORK, int *LWORK, double *RWORK, int *info);
   int info;
   zgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,RWORK,&info);
   return info;
}

/* solve Ax=b using QR factorization */
int
my_zgels(char TRANS, int M, int N, int NRHS, complex double *A, int LDA, complex double *B, int LDB, complex double *WORK, int LWORK) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void zgels_(char *TRANS, int *M, int *N, int *NRHS, complex double *A, int *LDA, complex double *B, int *LDB, complex double *WORK, int *LWORK, int *INFO);
  int info;
  zgels_(&TRANS,&M,&N,&NRHS,A,&LDA,B,&LDB,WORK,&LWORK,&info);
  return info;
}


/* solve Ax=b using QR factorization */
int
my_cgels(char TRANS, int M, int N, int NRHS, complex float *A, int LDA, complex float *B, int LDB, complex float *WORK, int LWORK) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void cgels_(char *TRANS, int *M, int *N, int *NRHS, complex float *A, int *LDA, complex float *B, int *LDB, complex float *WORK, int *LWORK, int *INFO);
  int info;
  cgels_(&TRANS,&M,&N,&NRHS,A,&LDA,B,&LDB,WORK,&LWORK,&info);
  return info;
}




/* BLAS ZGEMM C = alpha*op(A)*op(B)+ beta*C */
void
my_zgemm(char transa, char transb, int M, int N, int K, complex double alpha, complex double *A, int lda, complex double *B, int ldb, complex double beta, complex double *C, int ldc) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void zgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, complex double *ALPHA, complex double *A, int *LDA, complex double *B, int * LDB, complex double *BETA, complex double *C, int *LDC);
  zgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

/* ||x||_2 */
double
my_cnrm2(int N, complex double *x) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern double  dznrm2_(int *N, complex double *x, int *incx);
  int i=1;
  return(dznrm2_(&N,x,&i));
}

/* blas fcopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
void
my_fcopy(int N, float *x, int Nx, float *y, int Ny) {
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
  extern void scopy_(int *N, float *x, int *incx, float *y, int *incy);
  /* use memcpy if Nx=Ny=1 */
  if (Nx==1&&Ny==1) {
   memcpy((void*)y,(void*)x,sizeof(float)*(size_t)N);
  } else {
   scopy_(&N,x,&Nx,y,&Ny);
  }
}


/* LAPACK eigen value expert routine, real symmetric  matrix */
int 
my_dsyevx(char jobz, char range, char uplo, int N, double *A, int lda,
  double vl, double vu, int il, int iu, double abstol, int M, double  *W,
  double *Z, int ldz, double *WORK, int lwork, int *iwork, int *ifail) {

  extern void dsyevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A, int *LDA,
   double  *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M, double *W, double *Z, 
   int *LDZ, double *WORK, int *LWORK, int *IWORK, int *IFAIL, int *INFO);
  int info;
  dsyevx_(&jobz,&range,&uplo,&N,A,&lda,&vl,&vu,&il,&iu,&abstol,&M,W,Z,&ldz,WORK,&lwork,iwork,ifail,&info);
  return info;
} 



/* BLAS vector outer product
   A= alpha x x^H + A
*/
void
my_zher(char uplo, int N, double alpha, complex double *x, int incx, complex double *A, int lda) {

  extern void zher_(char *UPLO, int *N, double *ALPHA, complex double *X, int *INCX, complex double *A, int *LDA);
  
  zher_(&uplo,&N,&alpha,x,&incx,A,&lda);
}
