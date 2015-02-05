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


#include "buildsky.h"
#include <string.h> /* for memcpy */

/* blas dcopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
void
my_dcopy(int N, double *x, int Nx, double *y, int Ny) {
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
  extern void dscal_(int *N, double *alpha, double *x, int *incx);
  int i=1;
  dscal_(&N,&a,x,&i);
}

/* x^T*y */
double
my_ddot(int N, double *x, double *y) {
  extern double  ddot_(int *N, double *x, int *incx, double *y, int *incy);
  int i=1;
  return(ddot_(&N,x,&i,y,&i));
}

/* ||x||_2 */
double
my_dnrm2(int N, double *x) {
  extern double  dnrm2_(int *N, double *x, int *incx);
  int i=1;
  return(dnrm2_(&N,x,&i));
}

/* sum||x||_1 */
double
my_dasum(int N, double *x) {
  extern double  dasum_(int *N, double *x, int *incx);
  int i=1;
  return(dasum_(&N,x,&i));
}

/* BLAS y = a.x + y */
void
my_daxpy(int N, double *x, double a, double *y) {
    extern void daxpy_(int *N, double *alpha, double *x, int *incx, double *y, int *incy);
    int i=1; /* strides */
    daxpy_(&N,&a,x,&i,y,&i);
}

/* BLAS y = a.x + y */
void
my_daxpys(int N, double *x, int incx, double a, double *y, int incy) {
    extern void daxpy_(int *N, double *alpha, double *x, int *incx, double *y, int *incy);
    daxpy_(&N,&a,x,&incx,y,&incy);
}



/* max |x|  index (start from 1...)*/
int
my_idamax(int N, double *x, int incx) {
    extern int idamax_(int *N, double *x, int *incx);
    return idamax_(&N,x,&incx);
}

/* BLAS DGEMM C = alpha*op(A)*op(B)+ beta*C */
void
my_dgemm(char transa, char transb, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
  extern void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, double *ALPHA, double *A, int *LDA, double *B, int * LDB, double *BETA, double *C, int *LDC);
  dgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

/* BLAS DGEMV  y = alpha*op(A)*x+ beta*y : op 'T' or 'N' */
void
my_dgemv(char trans, int M, int N, double alpha, double *A, int lda, double *x, int incx,  double beta, double *y, int incy) {
  extern void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);
  dgemv_(&trans, &M, &N, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}


/* following routines used in LAPACK solvers */
/* cholesky factorization: real symmetric */
int
my_dpotrf(char uplo, int N, double *A, int lda) {
  extern void dpotrf_(char *uplo, int *N, double *A, int *lda, int *info);
  int info;
  dpotrf_(&uplo,&N,A,&lda,&info);
  return info;
}

/* solve Ax=b using cholesky factorization */
int 
my_dpotrs(char uplo, int N, int nrhs, double *A, int lda, double *b, int ldb){
   extern void dpotrs_(char  *uplo, int *N, int *nrhs, double *A, int *lda, double *b, int *ldb, int *info);
   int info;
   dpotrs_(&uplo,&N,&nrhs,A,&lda,b,&ldb,&info);
   return info;
}

/* solve Ax=b using QR factorization */
int
my_dgels(char TRANS, int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *WORK, int LWORK) {
  extern void dgels_(char *TRANS, int *M, int *N, int *NRHS, double *A, int *LDA, double *B, int *LDB, double *WORK, int *LWORK, int *INFO);
  int info;
  dgels_(&TRANS,&M,&N,&NRHS,A,&LDA,B,&LDB,WORK,&LWORK,&info);
  return info;
}


/* A=U S VT, so V needs NOT to be transposed */
int
my_dgesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S,
   double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK) {
   extern void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, 
    int *LDA, double *S, double *U, int *LDU, double *VT, int *LDVT,
    double *WORK, int *LWORK, int *info);
   int info;
   dgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,&info);
   return info;
}
