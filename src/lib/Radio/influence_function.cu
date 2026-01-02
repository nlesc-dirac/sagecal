/*
 *
 Copyright (C) 2025 Sarod Yatawatta <sarod@users.sf.net>  
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

#include "cuda.h"
#include <cuComplex.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "Dirac_radio.h"

/* enable this for checking for kernel failure */
//#define CUDA_DBG

/* matrix multiplications */
/* C=A*B */
static __device__ void
amb(const cuFloatComplex*__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],b[0]),cuCmulf(a[1],b[2]));
 c[1]=cuCaddf(cuCmulf(a[0],b[1]),cuCmulf(a[1],b[3]));
 c[2]=cuCaddf(cuCmulf(a[2],b[0]),cuCmulf(a[3],b[2]));
 c[3]=cuCaddf(cuCmulf(a[2],b[1]),cuCmulf(a[3],b[3]));
} 
/* C=A*B^H */
static __device__ void
ambt(const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],cuConjf(b[0])),cuCmulf(a[1],cuConjf(b[1])));
 c[1]=cuCaddf(cuCmulf(a[0],cuConjf(b[2])),cuCmulf(a[1],cuConjf(b[3])));
 c[2]=cuCaddf(cuCmulf(a[2],cuConjf(b[0])),cuCmulf(a[3],cuConjf(b[1])));
 c[3]=cuCaddf(cuCmulf(a[2],cuConjf(b[2])),cuCmulf(a[3],cuConjf(b[3])));
}

/* C=A x B, A: 4x4 matrix (row major), B: 4x1 vector, C: 4x1 vector */
static __device__ void
mat_vec(const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
    c[0]=cuCaddf(cuCaddf(cuCmulf(a[0],b[0]),cuCmulf(a[1],b[1])),cuCaddf(cuCmulf(a[2],b[2]),cuCmulf(a[3],b[3])));
    c[1]=cuCaddf(cuCaddf(cuCmulf(a[4],b[0]),cuCmulf(a[5],b[1])),cuCaddf(cuCmulf(a[6],b[2]),cuCmulf(a[7],b[3])));
    c[2]=cuCaddf(cuCaddf(cuCmulf(a[8],b[0]),cuCmulf(a[9],b[1])),cuCaddf(cuCmulf(a[10],b[2]),cuCmulf(a[11],b[3])));
    c[3]=cuCaddf(cuCaddf(cuCmulf(a[12],b[0]),cuCmulf(a[13],b[1])),cuCaddf(cuCmulf(a[14],b[2]),cuCmulf(a[15],b[3])));
}

/* Kronecker product C = kron(A,B)
   A,B :2x2 complex, A=[a0, a1; a2, a3], B=[b0, b1; b2, b3]
   C: 4x4 complex,
   C=[a_11 B, a_12 B;  = [a0 B, a1 B;
      a_21 B, a_22 B]     a2 B, a3 B]
   all matrices in row major order */
static __device__ void
kron_ab(const cuFloatComplex*__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
  c[0]=cuCmulf(a[0],b[0]);
  c[1]=cuCmulf(a[0],b[2]);
  c[2]=cuCmulf(a[2],b[0]);
  c[3]=cuCmulf(a[2],b[2]);
  c[4]=cuCmulf(a[0],b[1]);
  c[5]=cuCmulf(a[0],b[3]);
  c[6]=cuCmulf(a[2],b[1]);
  c[7]=cuCmulf(a[2],b[3]);
  c[8]=cuCmulf(a[1],b[0]);
  c[9]=cuCmulf(a[1],b[2]);
  c[10]=cuCmulf(a[3],b[0]);
  c[11]=cuCmulf(a[3],b[2]);
  c[12]=cuCmulf(a[1],b[1]);
  c[13]=cuCmulf(a[1],b[3]);
  c[14]=cuCmulf(a[3],b[1]);
  c[15]=cuCmulf(a[3],b[3]);
}

__global__ void
kernel_hessian(int B, int N, int T, int F, const double *__restrict__ p, int nchunk, baseline_t *barr,
    const float *__restrict__ coh, const float *__restrict__ res, float *hess) {

  /* only work with the first freq, so F==1 taken */
  /* x: baseline */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;
  /* y: station, column block of Hessian */
  unsigned int m=threadIdx.y+blockDim.y*blockIdx.y;

  /* hessian: 4N x 4N complex float, column major order,
    each column 4N complex float = 8N float, so, value at (row, col)
   is hess[col*4*N*2+row*4*2]+1j*hess[col*4*N*2+row*4*2+1],
   see code below for exact offset calculation */

  if (n<B) {
    int sta1=barr[n].sta1;
    int sta2=barr[n].sta2;
    if (m==sta1 or m==sta2) {
      /* this thread will work on column block m, 
         sta1,sta2 will update row,col of hess 
         (sta1,sta2), (sta2,sta1), (sta1,sta1), (sta2,sta2)
         and depending on m, select which values to update */
    cuFloatComplex C[4],R[4],H[16];
    cuFloatComplex I2[4];
    I2[0].x=1.0f;
    I2[0].y=0.0f;
    I2[1].x=0.0f;
    I2[1].y=0.0f;
    I2[2].x=0.0f;
    I2[2].y=0.0f;
    I2[3].x=1.0f;
    I2[3].y=0.0f;

    C[0].x=__ldg(&coh[8*n]);
    C[0].y=__ldg(&coh[8*n+1]);
    C[1].x=__ldg(&coh[8*n+2]);
    C[1].y=__ldg(&coh[8*n+3]);
    C[2].x=__ldg(&coh[8*n+4]);
    C[2].y=__ldg(&coh[8*n+5]);
    C[3].x=__ldg(&coh[8*n+6]);
    C[3].y=__ldg(&coh[8*n+7]);
    R[0].x=__ldg(&res[8*n]);
    R[0].y=__ldg(&res[8*n+1]);
    R[1].x=__ldg(&res[8*n+2]);
    R[1].y=__ldg(&res[8*n+3]);
    R[2].x=__ldg(&res[8*n+4]);
    R[2].y=__ldg(&res[8*n+5]);
    R[3].x=__ldg(&res[8*n+6]);
    R[3].y=__ldg(&res[8*n+7]);

    /* find out which chunk to select from p : 0,1..nchunk-1 */
    int chunk=(n)/((B+nchunk-1)/nchunk);

    if (m==sta1) {
      /* update (sta2,sta1)=(q,p) and (sta1,sta1)=(p,p) */
      /* need kron(-C^T, res^H ) -> q,p and 
         kron(res1^T, I_2), res1=C J_q^H J_q C^H =(C J_q^H) (C J_q^H)^H -> p,p */

    /* create G2 Jones matrices from q */
    cuFloatComplex G2[4];
    G2[0].x=(float)__ldg(&p[chunk*8*N+sta2*8]);
    G2[0].y=(float)__ldg(&p[chunk*8*N+sta2*8+1]);
    G2[1].x=(float)__ldg(&p[chunk*8*N+sta2*8+2]);
    G2[1].y=(float)__ldg(&p[chunk*8*N+sta2*8+3]);
    G2[2].x=(float)__ldg(&p[chunk*8*N+sta2*8+4]);
    G2[2].y=(float)__ldg(&p[chunk*8*N+sta2*8+5]);
    G2[3].x=(float)__ldg(&p[chunk*8*N+sta2*8+6]);
    G2[3].y=(float)__ldg(&p[chunk*8*N+sta2*8+7]);

    /* terms in the product */
    cuFloatComplex A[4],B[4];
    /* A=-C^T */
    A[0].x=-C[0].x;
    A[0].y=-C[0].y;
    A[1].x=-C[2].x;
    A[1].y=-C[2].y;
    A[2].x=-C[1].x;
    A[2].y=-C[1].y;
    A[3].x=-C[3].x;
    A[3].y=-C[3].y;
    /* B=res^H */
    B[0].x=R[0].x;
    B[0].y=-R[0].y;
    B[1].x=R[2].x;
    B[1].y=-R[2].y;
    B[2].x=R[1].x;
    B[2].y=-R[1].y;
    B[3].x=R[3].x;
    B[3].y=-R[3].y;
 
      /* -C^T \kron R^H */
    kron_ab(A,B,H);
    for (int off=0; off<4; off++) {
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2],H[0+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2+1],H[0+off].y);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2+2],H[4+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2+3],H[4+off].y);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2+4],H[8+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2+5],H[8+off].y);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2+6],H[12+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta2*4*2+7],H[12+off].y);
    }

     ambt(C,G2,A); /* A = C J_q^H */
     B[0].x=A[0].x;
     B[0].y=A[0].y;
     B[1].x=A[1].x;
     B[1].y=A[1].y;
     B[2].x=A[2].x;
     B[2].y=A[2].y;
     B[3].x=A[3].x;
     B[3].y=A[3].y;

     cuFloatComplex D[4];
     ambt(A,B,D); /* D = A A^H = (C J_q^H) (C J_q^H)^H */
     cuFloatComplex E[4];
     /* E= D^T */
     E[0]=D[0]; E[1]=D[2]; E[2]=D[1]; E[3]=D[3];
     /* D^T \kron I_2, D= C J_q^H J_q C^H */
     kron_ab(E,I2,H);
     for (int off=0; off<4; off++) {
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2],H[0+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2+1],H[0+off].y);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2+2],H[4+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2+3],H[4+off].y);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2+4],H[8+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2+5],H[8+off].y);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2+6],H[12+off].x);
      atomicAdd(&hess[sta1*4*4*N*2+off*4*N*2+sta1*4*2+7],H[12+off].y);
     }
    } else { /* m==sta2 */
      /* update (sta1,sta2)=(p,q) and (sta2,sta2)=(q,q) */
      /* need kron(-conj(C), res) and
         kron(res1^T, I_2), res1=C^H J_p^H J_p C =(J_p C)^H (J_p C) */
    /* create G1 Jones matrices from p */
    cuFloatComplex G1[4];
    G1[0].x=(float)__ldg(&p[chunk*8*N+sta1*8]);
    G1[0].y=(float)__ldg(&p[chunk*8*N+sta1*8+1]);
    G1[1].x=(float)__ldg(&p[chunk*8*N+sta1*8+2]);
    G1[1].y=(float)__ldg(&p[chunk*8*N+sta1*8+3]);
    G1[2].x=(float)__ldg(&p[chunk*8*N+sta1*8+4]);
    G1[2].y=(float)__ldg(&p[chunk*8*N+sta1*8+5]);
    G1[3].x=(float)__ldg(&p[chunk*8*N+sta1*8+6]);
    G1[3].y=(float)__ldg(&p[chunk*8*N+sta1*8+7]);

    cuFloatComplex A[4],B[4];
    /* A=-conj(C) */
    A[0].x=-C[0].x;
    A[0].y=C[0].y;
    A[1].x=-C[1].x;
    A[1].y=C[1].y;
    A[2].x=-C[2].x;
    A[2].y=C[2].y;
    A[3].x=-C[3].x;
    A[3].y=C[3].y;
 
      /* -C^* \kron R */
    kron_ab(A,R,H);
    for (int off=0; off<4; off++) {
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2],H[0+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2+1],H[0+off].y);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2+2],H[4+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2+3],H[4+off].y);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2+4],H[8+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2+5],H[8+off].y);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2+6],H[12+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta1*4*2+7],H[12+off].y);
    }

     amb(G1,C,B); /* B = J_p C */
     /* A = B^H */
     A[0].x=B[0].x;
     A[0].y=-B[0].y;
     A[1].x=B[2].x;
     A[1].y=-B[2].y;
     A[2].x=B[1].x;
     A[2].y=-B[1].y;
     A[3].x=B[3].x;
     A[3].y=-B[3].y;
 
     cuFloatComplex D[4];
     amb(A,B,D); /* D = B^H B = (J_p C)^H (J_p C) */
     cuFloatComplex E[4];
     /* E= D^T */
     E[0]=D[0]; E[1]=D[2]; E[2]=D[1]; E[3]=D[3];
     /* D^T \kron I_2, D= (J_p C)^H J_p C */
     kron_ab(E,I2,H);
     for (int off=0; off<4; off++) {
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2],H[0+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2+1],H[0+off].y);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2+2],H[4+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2+3],H[4+off].y);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2+4],H[8+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2+5],H[8+off].y);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2+6],H[12+off].x);
      atomicAdd(&hess[sta2*4*4*N*2+off*4*N*2+sta2*4*2+7],H[12+off].y);
     }
    }
    }
  }
}
 

__global__ void
kernel_d_solutions(int B, int N, int T, int F, const double *__restrict__ p, int nchunk, baseline_t *barr,
    const float *__restrict__ coh, float *AdV) {

  /* only work with the first freq, so F==1 taken */
  /* x: baseline = N(N-1)/2 x T */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;
  /* y: station, row block of AdV */
  unsigned int m=threadIdx.y+blockDim.y*blockIdx.y;
  /* AdV : 2*4NxBt blocks, Bt=B/T=N(N-1)/2 */
  /* AdV : 4N x Bt complex float column major order,
    each column 4N complex float = 8N float, so, value at (row, col) 
   AdV[col*4*N*2+row*4*2]+1j*AdV[col*4*N*2+row*4*2+1] 
   ...
   AdV[col*4*N*2+row*4*2+6]+1j*AdV[col*4*N*2+row*4*2+7] 

   station p maps to two row blocks in each column of size 4N
   p*2:p*2+1 and 2*N+p*2:2*N+p*2+1
  */
  cuFloatComplex A[4],G2[4],C[4],H[16],I2[4];
  I2[0].x=1.0f;
  I2[0].y=0.0f;
  I2[1].x=0.0f;
  I2[1].y=0.0f;
  I2[2].x=0.0f;
  I2[2].y=0.0f;
  I2[3].x=1.0f;
  I2[3].y=0.0f;

  int Bt=((N*(N-1)/2));

  /* left hand side (J_q C^H)^T, right hand side I_2
     kron(lhs,I_2) stored at row block p */
  if (n<B) {
    int sta1=barr[n].sta1; //station p
    int sta2=barr[n].sta2; //station q
    // fill column block bl
    int bl=n % Bt; // baseline index
    if (m<N && (m==sta2)) {

      /* C^\star */
      C[0].x=__ldg(&coh[8*n]);
      C[0].y=-__ldg(&coh[8*n+1]);
      C[1].x=__ldg(&coh[8*n+2]);
      C[1].y=-__ldg(&coh[8*n+3]);
      C[2].x=__ldg(&coh[8*n+4]);
      C[2].y=-__ldg(&coh[8*n+5]);
      C[3].x=__ldg(&coh[8*n+6]);
      C[3].y=-__ldg(&coh[8*n+7]);

    /* find out which chunk to select from p : 0,1..nchunk-1 */
    int chunk=(n)/((B+nchunk-1)/nchunk);

    /* G2 Jones matrices from q, J_q^T*/
    G2[0].x=(float)__ldg(&p[chunk*8*N+sta2*8]);
    G2[0].y=(float)__ldg(&p[chunk*8*N+sta2*8+1]);
    G2[2].x=(float)__ldg(&p[chunk*8*N+sta2*8+2]);
    G2[2].y=(float)__ldg(&p[chunk*8*N+sta2*8+3]);
    G2[1].x=(float)__ldg(&p[chunk*8*N+sta2*8+4]);
    G2[1].y=(float)__ldg(&p[chunk*8*N+sta2*8+5]);
    G2[3].x=(float)__ldg(&p[chunk*8*N+sta2*8+6]);
    G2[3].y=(float)__ldg(&p[chunk*8*N+sta2*8+7]);

    /* (J_q C^H)^T = C^\star J_q^T */
    amb(C,G2,A); /* A = C^\star J_q^T */
    /* (C^star J_q^T) \kron I_2 */
      kron_ab(A,I2,H);

      /* row block p(=sta1), column bl */
      /* find product H [1+j; 1+j; 1+j; 1+j] */
      /* row major H */
      for (int row=0; row<2; row++) {
        float product_r=0.0f;
        float product_i=0.0f;
        for (int col=0; col<4; col++) {
         product_r+=H[4*row+col].x-H[4*row+col].y;
         product_i+=H[4*row+col].x+H[4*row+col].y;
        }
        atomicAdd(&AdV[bl*4*N*2+sta1*2*2+2*row],product_r);
        atomicAdd(&AdV[bl*4*N*2+sta1*2*2+2*row+1],product_i);
      }
      for (int row=0; row<2; row++) {
        float product_r=0.0f;
        float product_i=0.0f;
        for (int col=0; col<4; col++) {
         product_r+=H[4*(row+2)+col].x-H[4*(row+2)+col].y;
         product_i+=H[4*(row+2)+col].x+H[4*(row+2)+col].y;
        }
        atomicAdd(&AdV[bl*4*N*2+N*2*2+sta1*2*2+2*row],product_r);
        atomicAdd(&AdV[bl*4*N*2+N*2*2+sta1*2*2+2*row+1],product_i);
      }
    }
  }
}


__global__ void
kernel_d_residuals(int B, int N, int T, int F, const double *__restrict__ p, int nchunk, baseline_t *barr,
    const float *__restrict__ coh, const float *__restrict__ dJ, float *dR) {

  /* only work with the first freq, so F==1 taken */
  /* x: baseline = N(N-1)/2 x T */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;
  /* dJ : 2*4N x Bt(columns Bt=B/T=N(N-1)/2), only use 
   the column corresponding to the baseline (i.e. diagonal terms) */
  /* dR : 2*4Bt x 1, Bt=B/T=N(N-1)/2 */
  /* in dJ
   station p maps to two row blocks in each column of size 4N
   p*2:p*2+1 and 2*N+p*2:2*N+p*2+1
  */
  cuFloatComplex A[4],J[4],G2[4],C[4],H[16],I2[4];
  I2[0].x=1.0f;
  I2[0].y=0.0f;
  I2[1].x=0.0f;
  I2[1].y=0.0f;
  I2[2].x=0.0f;
  I2[2].y=0.0f;
  I2[3].x=1.0f;
  I2[3].y=0.0f;

  int Bt=((N*(N-1)/2));

  /* left hand side -(C J_q^H)^T, right hand side I_2
     left hand side =  J_q^star (-C^T)
     kron(lhs,I_2) x (row block p of dJ) */
  if (n<B) {
    int sta1=barr[n].sta1; //station p
    int sta2=barr[n].sta2; //station q
    // fill column block bl of dR
    int bl=n % Bt; // baseline index
    if (sta1>=0 && sta2>=0) {
      /* -C^T */
      C[0].x=__ldg(&coh[8*n]);
      C[0].y=__ldg(&coh[8*n+1]);
      C[2].x=__ldg(&coh[8*n+2]);
      C[2].y=__ldg(&coh[8*n+3]);
      C[1].x=__ldg(&coh[8*n+4]);
      C[1].y=__ldg(&coh[8*n+5]);
      C[3].x=__ldg(&coh[8*n+6]);
      C[3].y=__ldg(&coh[8*n+7]);

    /* find out which chunk to select from p : 0,1..nchunk-1 */
    int chunk=(n)/((B+nchunk-1)/nchunk);

    /* G2 Jones matrices from q, J_q^star*/
    G2[0].x=(float)__ldg(&p[chunk*8*N+sta2*8]);
    G2[0].y=-(float)__ldg(&p[chunk*8*N+sta2*8+1]);
    G2[1].x=(float)__ldg(&p[chunk*8*N+sta2*8+2]);
    G2[1].y=-(float)__ldg(&p[chunk*8*N+sta2*8+3]);
    G2[2].x=(float)__ldg(&p[chunk*8*N+sta2*8+4]);
    G2[2].y=-(float)__ldg(&p[chunk*8*N+sta2*8+5]);
    G2[3].x=(float)__ldg(&p[chunk*8*N+sta2*8+6]);
    G2[3].y=-(float)__ldg(&p[chunk*8*N+sta2*8+7]);

    /* -(C J_q^H)^T = J_q^\star (-C)^T */
    amb(G2,C,A); /* A = -J_q^\star C^T */
    /* -(J_q^star C^T) \kron I_2 */
    kron_ab(A,I2,H);

    /* dJ row block p(=sta1), column : bl (diagonal term) */
    /* row major H */
    /* find product H dJ[p*2:p*2+1 and 2*N+p*2:2*N+p*2+1] */
    J[0].x=dJ[bl*N*8+sta1*2*2];
    J[0].y=dJ[bl*N*8+sta1*2*2+1];
    J[1].x=dJ[bl*N*8+sta1*2*2+2];
    J[1].y=dJ[bl*N*8+sta1*2*2+3];
    J[2].x=dJ[bl*N*8+N*2*2+sta1*2*2];
    J[2].y=dJ[bl*N*8+N*2*2+sta1*2*2+1];
    J[3].x=dJ[bl*N*8+N*2*2+sta1*2*2+2];
    J[3].y=dJ[bl*N*8+N*2*2+sta1*2*2+3];
    mat_vec(H,J,A);
    /* fill to dR[bl*8:bl*8+7] */
    atomicAdd(&dR[bl*8],A[0].x);
    atomicAdd(&dR[bl*8+1],A[0].y);
    atomicAdd(&dR[bl*8+2],A[1].x);
    atomicAdd(&dR[bl*8+3],A[1].y);
    atomicAdd(&dR[bl*8+4],A[2].x);
    atomicAdd(&dR[bl*8+5],A[2].y);
    atomicAdd(&dR[bl*8+6],A[3].x);
    atomicAdd(&dR[bl*8+7],A[3].y);
    }
  }
}

__global__ void
kernel_sum_col(int M, int N, float *A) {
  unsigned int m=threadIdx.x+blockDim.x*blockIdx.x;
  if (m<M) {
    float block_sum=0.0f;
    for (int n=0; n<N; n++) {
      block_sum+=A[M*n+m];
    }
    A[m]=block_sum;
  }
}

/* only use extern if calling code is C */
extern "C"
{

#ifdef CUDA_DBG
static void
checkCudaError(cudaError_t err, const char *file, int line)
{
    if(!err)
        return;
    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
}
#endif


void
cudakernel_hessian(int B, int N, int T, int F, baseline_t *barr, double *p, int nchunk, float *coh, float *res, float *hess) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  /* spawn threads to handle baselines, stations */
  /* thread x : baseline, thread y: station */
  dim3 threadsPerBlock(16,8); 
  dim3 numBlocks((B+threadsPerBlock.x-1)/threadsPerBlock.x,
         (N+threadsPerBlock.y-1)/threadsPerBlock.y);
  kernel_hessian<<<numBlocks,threadsPerBlock>>>(B, N, T, F, p, nchunk, 
      barr, coh, res, hess);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
}

void
cudakernel_d_solutions(int B, int N, int T, int F, baseline_t *barr, double *p, int nchunk, float *coh, float *AdV) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  /* AdV: 2*4Nx(B/T) values, baselines=B/T here */
  /* spawn threads to handle baselines, stations */
  /* thread x : baseline, thread y: station */
  dim3 threadsPerBlock(16,8); 
  dim3 numBlocks((B+threadsPerBlock.x-1)/threadsPerBlock.x,
         (N+threadsPerBlock.y-1)/threadsPerBlock.y);
  cudaMemset(AdV,0,2*4*N*(B/T)*sizeof(float));
  kernel_d_solutions<<<numBlocks,threadsPerBlock>>>(B, N, T, F, p, nchunk, 
      barr, coh, AdV);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

}


void 
cudakernel_sum_col(int M, int N, float *A) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif
 
  /* A: M x N matrix, add all columns to first column */
  /* spawn M threads to handle each row */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;
  kernel_sum_col<<<BlocksPerGrid,ThreadsPerBlock>>>(M,N,A);

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
}

void
cudakernel_d_residuals(int B, int N, int T, int F, baseline_t *barr, double *p, int nchunk, float *coh, float *dJ, float *dR) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif
  
  int Bt=B/T;

  /* dJ: 2*4Nx(B/T) values, baselines=B/T here (averaged over T) */
  /* dR: (B/T)*4*2 values, select diagonal block of full matrix dJ , hence
     full dR of size (4*Nbase*2)xNbase reduces to just on column */

  cudaMemset(dR,0,2*4*Bt*sizeof(float));
  /* spawn threads to handle baselines */
  /* thread x : baseline(all times) */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int BlocksPerGrid=(B+ThreadsPerBlock-1)/ThreadsPerBlock;
  kernel_d_residuals<<<BlocksPerGrid,ThreadsPerBlock>>>(B, N, T, F, p, nchunk, 
      barr, coh, dJ, dR);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

}

} /* extern "C" */
