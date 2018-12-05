/*
 *
 Copyright (C) 2014 Sarod Yatawatta <sarod@users.sf.net>  
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
#include "GPUtune.h"

/* enable this for checking for kernel failure */
//#define CUDA_DBG

/* matrix multiplications */
/* C=A*B */
__device__ void
amb(const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],b[0]),cuCmulf(a[1],b[2]));
 c[1]=cuCaddf(cuCmulf(a[0],b[1]),cuCmulf(a[1],b[3]));
 c[2]=cuCaddf(cuCmulf(a[2],b[0]),cuCmulf(a[3],b[2]));
 c[3]=cuCaddf(cuCmulf(a[2],b[1]),cuCmulf(a[3],b[3]));
}
/* C=A*B^H */
__device__ void
ambt(const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],cuConjf(b[0])),cuCmulf(a[1],cuConjf(b[1])));
 c[1]=cuCaddf(cuCmulf(a[0],cuConjf(b[2])),cuCmulf(a[1],cuConjf(b[3])));
 c[2]=cuCaddf(cuCmulf(a[2],cuConjf(b[0])),cuCmulf(a[3],cuConjf(b[1])));
 c[3]=cuCaddf(cuCmulf(a[2],cuConjf(b[2])),cuCmulf(a[3],cuConjf(b[3])));
}

/* C=A^H * B */
__device__ void
atmb(const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(cuConjf(a[0]),b[0]),cuCmulf(cuConjf(a[2]),b[2]));
 c[1]=cuCaddf(cuCmulf(cuConjf(a[0]),b[1]),cuCmulf(cuConjf(a[2]),b[3]));
 c[2]=cuCaddf(cuCmulf(cuConjf(a[1]),b[0]),cuCmulf(cuConjf(a[3]),b[2]));
 c[3]=cuCaddf(cuCmulf(cuConjf(a[1]),b[1]),cuCmulf(cuConjf(a[3]),b[3]));
}


__global__ void 
kernel_fns_fhess(int N, int Nbase, const cuFloatComplex *__restrict__ x, const cuFloatComplex *__restrict__ eta, cuFloatComplex *__restrict__ hess0, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh) {

  /* eta0: each block will store result in its own block */
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  /* 4x2xblockDim.x cuFloatComplex values and 2xblockDim.x int values */
  extern __shared__ cuFloatComplex hs[];
  int *stm= (int*)&hs[8*blockDim.x];
  stm[2*tid]=-1;
  stm[2*tid+1]=-1;
  /* x,eta: 2Nx2 matrix */ 
  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    if (sta1>=0 && sta2>=0) {   
     stm[2*tid]=sta1;
     stm[2*tid+1]=sta2;

     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     cuFloatComplex E1[4];
     cuFloatComplex E2[4];
     /* J1 */
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     /* conjugate this to get J2^H */
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];
     E1[0]=eta[2*sta1];
     E1[1]=eta[2*sta1+2*N];
     E1[2]=eta[2*sta1+1];
     E1[3]=eta[2*sta1+2*N+1];
     E2[0]=eta[2*sta2];
     E2[1]=eta[2*sta2+2*N];
     E2[2]=eta[2*sta2+1];
     E2[3]=eta[2*sta2+2*N+1];


     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* G1*C*G2' */
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     ambt(T1,G2,T2);

     /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
     /* V->U */
     cuFloatComplex res[4],res1[4];
     res[0]=cuCsubf(make_cuFloatComplex(y[8*n],y[8*n+1]),T2[0]);
     res[1]=cuCsubf(make_cuFloatComplex(y[8*n+2],y[8*n+3]),T2[1]);
     res[2]=cuCsubf(make_cuFloatComplex(y[8*n+4],y[8*n+5]),T2[2]);
     res[3]=cuCsubf(make_cuFloatComplex(y[8*n+6],y[8*n+7]),T2[3]);

   /*
      res1=x(2*p-1:2*p,:)*C*eta(2*q-1:2*q,:)'+eta(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)';
   */
   /* G1*C*E2' */
     amb(G1,C,T1);
     ambt(T1,E2,T2);
     res1[0]=T2[0];
     res1[1]=T2[1];
     res1[2]=T2[2];
     res1[3]=T2[3];
     /* E1*C*G2' */
     amb(E1,C,T1);
     ambt(T1,G2,T2);
     res1[0]=cuCaddf(res1[0],T2[0]);
     res1[1]=cuCaddf(res1[1],T2[1]);
     res1[2]=cuCaddf(res1[2],T2[2]);
     res1[3]=cuCaddf(res1[3],T2[3]);


  /* 
      hess(2*p-1:2*p,:)=hess(2*p-1:2*p,:)+(res*eta(2*q-1:2*q,:)-res1*x(2*q-1:2*q,:))*C';
   */

    /* (res*E2-res1*G2)*C' */
    amb(res,E2,T1);
    amb(res1,G2,T2);
    T1[0]=cuCsubf(T1[0],T2[0]);
    T1[1]=cuCsubf(T1[1],T2[1]);
    T1[2]=cuCsubf(T1[2],T2[2]);
    T1[3]=cuCsubf(T1[3],T2[3]);
    ambt(T1,C,T2);

     hs[8*tid]=T2[0];
     hs[8*tid+1]=T2[1];
     hs[8*tid+2]=T2[2];
     hs[8*tid+3]=T2[3];


   /* 
      hess(2*q-1:2*q,:)=hess(2*q-1:2*q,:)+(res'*eta(2*p-1:2*p,:)-res1'*x(2*p-1:2*p,:))*C;
   */

     /* (res'*E1-res1'*G1)*C */
     atmb(res,E1,T1);
     atmb(res1,G1,T2);
     T1[0]=cuCsubf(T1[0],T2[0]);
     T1[1]=cuCsubf(T1[1],T2[1]);
     T1[2]=cuCsubf(T1[2],T2[2]);
     T1[3]=cuCsubf(T1[3],T2[3]);
     amb(T1,C,T2);


     hs[8*tid+4]=T2[0];
     hs[8*tid+5]=T2[1];
     hs[8*tid+6]=T2[2];
     hs[8*tid+7]=T2[3];

    }
  } 

  __syncthreads();
  /* copy back to global memory */
  if (tid==0) {
   for(int ci=0; ci<blockDim.x; ci++) {
    int sta1=stm[2*ci]; 
    int sta2=stm[2*ci+1];
    if (sta1>=0 && sta2>=0) {
      hess0[2*sta1+bid*4*N]=cuCaddf(hess0[2*sta1+bid*4*N],hs[8*ci]);
      hess0[2*sta1+2*N+bid*4*N]=cuCaddf(hess0[2*sta1+2*N+bid*4*N],hs[8*ci+1]);
      hess0[2*sta1+1+bid*4*N]=cuCaddf(hess0[2*sta1+1+bid*4*N],hs[8*ci+2]);
      hess0[2*sta1+2*N+1+bid*4*N]=cuCaddf(hess0[2*sta1+2*N+1+bid*4*N],hs[8*ci+3]);
      hess0[2*sta2+bid*4*N]=cuCaddf(hess0[2*sta2+bid*4*N],hs[8*ci+4]);
      hess0[2*sta2+2*N+bid*4*N]=cuCaddf(hess0[2*sta2+2*N+bid*4*N],hs[8*ci+5]);
      hess0[2*sta2+1+bid*4*N]=cuCaddf(hess0[2*sta2+1+bid*4*N],hs[8*ci+6]);
      hess0[2*sta2+2*N+1+bid*4*N]=cuCaddf(hess0[2*sta2+2*N+1+bid*4*N],hs[8*ci+7]);
    }
   }
  }
  __syncthreads();
}

__global__ void 
kernel_fns_fhess_robust1(int N, int Nbase, const cuFloatComplex *__restrict__ x, const cuFloatComplex *__restrict__ eta, cuFloatComplex *__restrict__ hess0, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, const float *__restrict__ wtd) {

  /* eta0: each block will store result in its own block */
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  /* 4x2xblockDim.x cuFloatComplex values and 2xblockDim.x int values */
  extern __shared__ cuFloatComplex hs[];
  int *stm= (int*)&hs[8*blockDim.x];
  stm[2*tid]=-1;
  stm[2*tid+1]=-1;
  /* x,eta: 2Nx2 matrix */ 
  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    if (sta1>=0 && sta2>=0) {   
     stm[2*tid]=sta1;
     stm[2*tid+1]=sta2;

     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     cuFloatComplex E1[4];
     cuFloatComplex E2[4];
     /* J1 */
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     /* conjugate this to get J2^H */
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];
     E1[0]=eta[2*sta1];
     E1[1]=eta[2*sta1+2*N];
     E1[2]=eta[2*sta1+1];
     E1[3]=eta[2*sta1+2*N+1];
     E2[0]=eta[2*sta2];
     E2[1]=eta[2*sta2+2*N];
     E2[2]=eta[2*sta2+1];
     E2[3]=eta[2*sta2+2*N+1];


     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* G1*C*G2' */
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     ambt(T1,G2,T2);

     /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
     /* V->U */
     cuFloatComplex res[4],res1[4];
     res[0]=cuCsubf(make_cuFloatComplex(y[8*n],y[8*n+1]),T2[0]);
     res[1]=cuCsubf(make_cuFloatComplex(y[8*n+2],y[8*n+3]),T2[1]);
     res[2]=cuCsubf(make_cuFloatComplex(y[8*n+4],y[8*n+5]),T2[2]);
     res[3]=cuCsubf(make_cuFloatComplex(y[8*n+6],y[8*n+7]),T2[3]);

   /*
      res1=x(2*p-1:2*p,:)*C*eta(2*q-1:2*q,:)'+eta(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)';
   */
   /* G1*C*E2' */
     amb(G1,C,T1);
     ambt(T1,E2,T2);
     res1[0]=T2[0];
     res1[1]=T2[1];
     res1[2]=T2[2];
     res1[3]=T2[3];
     /* E1*C*G2' */
     amb(E1,C,T1);
     ambt(T1,G2,T2);
     res1[0]=cuCaddf(res1[0],T2[0]);
     res1[1]=cuCaddf(res1[1],T2[1]);
     res1[2]=cuCaddf(res1[2],T2[2]);
     res1[3]=cuCaddf(res1[3],T2[3]);


  /* 
      hess(2*p-1:2*p,:)=hess(2*p-1:2*p,:)+(res*eta(2*q-1:2*q,:)-res1*x(2*q-1:2*q,:))*C';
   */

    /* (res*E2-res1*G2)*C' */
    amb(res,E2,T1);
    amb(res1,G2,T2);
    T1[0]=cuCsubf(T1[0],T2[0]);
    T1[1]=cuCsubf(T1[1],T2[1]);
    T1[2]=cuCsubf(T1[2],T2[2]);
    T1[3]=cuCsubf(T1[3],T2[3]);
    ambt(T1,C,T2);

     float wtdn=wtd[n];
     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;


     hs[8*tid]=T2[0];
     hs[8*tid+1]=T2[1];
     hs[8*tid+2]=T2[2];
     hs[8*tid+3]=T2[3];


   /* 
      hess(2*q-1:2*q,:)=hess(2*q-1:2*q,:)+(res'*eta(2*p-1:2*p,:)-res1'*x(2*p-1:2*p,:))*C;
   */

     /* (res'*E1-res1'*G1)*C */
     atmb(res,E1,T1);
     atmb(res1,G1,T2);
     T1[0]=cuCsubf(T1[0],T2[0]);
     T1[1]=cuCsubf(T1[1],T2[1]);
     T1[2]=cuCsubf(T1[2],T2[2]);
     T1[3]=cuCsubf(T1[3],T2[3]);
     amb(T1,C,T2);

     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;


     hs[8*tid+4]=T2[0];
     hs[8*tid+5]=T2[1];
     hs[8*tid+6]=T2[2];
     hs[8*tid+7]=T2[3];

    }
  } 

  __syncthreads();
  /* copy back to global memory */
  if (tid==0) {
   for(int ci=0; ci<blockDim.x; ci++) {
    int sta1=stm[2*ci]; 
    int sta2=stm[2*ci+1];
    if (sta1>=0 && sta2>=0) {
      hess0[2*sta1+bid*4*N]=cuCaddf(hess0[2*sta1+bid*4*N],hs[8*ci]);
      hess0[2*sta1+2*N+bid*4*N]=cuCaddf(hess0[2*sta1+2*N+bid*4*N],hs[8*ci+1]);
      hess0[2*sta1+1+bid*4*N]=cuCaddf(hess0[2*sta1+1+bid*4*N],hs[8*ci+2]);
      hess0[2*sta1+2*N+1+bid*4*N]=cuCaddf(hess0[2*sta1+2*N+1+bid*4*N],hs[8*ci+3]);
      hess0[2*sta2+bid*4*N]=cuCaddf(hess0[2*sta2+bid*4*N],hs[8*ci+4]);
      hess0[2*sta2+2*N+bid*4*N]=cuCaddf(hess0[2*sta2+2*N+bid*4*N],hs[8*ci+5]);
      hess0[2*sta2+1+bid*4*N]=cuCaddf(hess0[2*sta2+1+bid*4*N],hs[8*ci+6]);
      hess0[2*sta2+2*N+1+bid*4*N]=cuCaddf(hess0[2*sta2+2*N+1+bid*4*N],hs[8*ci+7]);
    }
   }
  }
  __syncthreads();
}


__global__ void 
kernel_fns_fhess_robust(int N, int Nbase, const cuFloatComplex *__restrict__ x, const cuFloatComplex *__restrict__ eta, cuFloatComplex *__restrict__ hess0, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, const float *__restrict__ wtd) {

  /* hess0: each block will store result in its own block */
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  /* baselines */
  int nbase=N*(N-1)/2;
  /* blocks per timeslot */
  int Bt=(nbase+blockDim.x-1)/blockDim.x;

  /* which timeslot */
  int ntime=bid/Bt;
  /* which offset */
  int noff=bid%Bt;
  /* local index within one timeslot, 0...N(N-1)/2-1 */
  unsigned int m = noff*blockDim.x+threadIdx.x;
  /* global thread index : less than the total baselines */
  unsigned int n = ntime*nbase+m;

  /* 4x2xblockDim.x cuFloatComplex values and 2xblockDim.x int values */
  extern __shared__ cuFloatComplex hs[];
  int *stm= (int*)&hs[8*blockDim.x];
  stm[2*tid]=-1;
  stm[2*tid+1]=-1;
  /* x,eta: 2Nx2 matrix */ 
  if(m<nbase && n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    if (sta1>=0 && sta2>=0) {   
     stm[2*tid]=sta1;
     stm[2*tid+1]=sta2;

     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     cuFloatComplex E1[4];
     cuFloatComplex E2[4];
     /* J1 */
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     /* conjugate this to get J2^H */
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];
     E1[0]=eta[2*sta1];
     E1[1]=eta[2*sta1+2*N];
     E1[2]=eta[2*sta1+1];
     E1[3]=eta[2*sta1+2*N+1];
     E2[0]=eta[2*sta2];
     E2[1]=eta[2*sta2+2*N];
     E2[2]=eta[2*sta2+1];
     E2[3]=eta[2*sta2+2*N+1];


     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* G1*C*G2' */
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     ambt(T1,G2,T2);

     /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
     /* V->U */
     cuFloatComplex res[4],res1[4];
     res[0]=cuCsubf(make_cuFloatComplex(y[8*n],y[8*n+1]),T2[0]);
     res[1]=cuCsubf(make_cuFloatComplex(y[8*n+2],y[8*n+3]),T2[1]);
     res[2]=cuCsubf(make_cuFloatComplex(y[8*n+4],y[8*n+5]),T2[2]);
     res[3]=cuCsubf(make_cuFloatComplex(y[8*n+6],y[8*n+7]),T2[3]);

   /*
      res1=x(2*p-1:2*p,:)*C*eta(2*q-1:2*q,:)'+eta(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)';
   */
   /* G1*C*E2' */
     amb(G1,C,T1);
     ambt(T1,E2,T2);
     res1[0]=T2[0];
     res1[1]=T2[1];
     res1[2]=T2[2];
     res1[3]=T2[3];
     /* E1*C*G2' */
     amb(E1,C,T1);
     ambt(T1,G2,T2);
     res1[0]=cuCaddf(res1[0],T2[0]);
     res1[1]=cuCaddf(res1[1],T2[1]);
     res1[2]=cuCaddf(res1[2],T2[2]);
     res1[3]=cuCaddf(res1[3],T2[3]);


  /* 
      hess(2*p-1:2*p,:)=hess(2*p-1:2*p,:)+(res*eta(2*q-1:2*q,:)-res1*x(2*q-1:2*q,:))*C';
   */

    /* (res*E2-res1*G2)*C' */
    amb(res,E2,T1);
    amb(res1,G2,T2);
    T1[0]=cuCsubf(T1[0],T2[0]);
    T1[1]=cuCsubf(T1[1],T2[1]);
    T1[2]=cuCsubf(T1[2],T2[2]);
    T1[3]=cuCsubf(T1[3],T2[3]);
    ambt(T1,C,T2);

     float wtdn=wtd[n];
     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;


     hs[8*tid]=T2[0];
     hs[8*tid+1]=T2[1];
     hs[8*tid+2]=T2[2];
     hs[8*tid+3]=T2[3];


   /* 
      hess(2*q-1:2*q,:)=hess(2*q-1:2*q,:)+(res'*eta(2*p-1:2*p,:)-res1'*x(2*p-1:2*p,:))*C;
   */

     /* (res'*E1-res1'*G1)*C */
     atmb(res,E1,T1);
     atmb(res1,G1,T2);
     T1[0]=cuCsubf(T1[0],T2[0]);
     T1[1]=cuCsubf(T1[1],T2[1]);
     T1[2]=cuCsubf(T1[2],T2[2]);
     T1[3]=cuCsubf(T1[3],T2[3]);
     amb(T1,C,T2);

     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;


     hs[8*tid+4]=T2[0];
     hs[8*tid+5]=T2[1];
     hs[8*tid+6]=T2[2];
     hs[8*tid+7]=T2[3];

    }
  } 
  __syncthreads();

  /* copy back to global memory */
  if (tid<N) {
   for(int ci=0; ci<blockDim.x; ci++) {
    int sta1=stm[2*ci];
    int sta2=stm[2*ci+1];
    if (sta1==tid) { /* note, tid >=0 always */
      hess0[2*tid+bid*4*N]=cuCaddf(hess0[2*tid+bid*4*N],hs[8*ci]);
      hess0[2*tid+2*N+bid*4*N]=cuCaddf(hess0[2*tid+2*N+bid*4*N],hs[8*ci+1]);
      hess0[2*tid+1+bid*4*N]=cuCaddf(hess0[2*tid+1+bid*4*N],hs[8*ci+2]);
      hess0[2*tid+2*N+1+bid*4*N]=cuCaddf(hess0[2*tid+2*N+1+bid*4*N],hs[8*ci+3]);
    }
    if (sta2==tid) { /* note, tid >=0 always */
      hess0[2*tid+bid*4*N]=cuCaddf(hess0[2*tid+bid*4*N],hs[8*ci+4]);
      hess0[2*tid+2*N+bid*4*N]=cuCaddf(hess0[2*tid+2*N+bid*4*N],hs[8*ci+5]);
      hess0[2*tid+1+bid*4*N]=cuCaddf(hess0[2*tid+1+bid*4*N],hs[8*ci+6]);
      hess0[2*tid+2*N+1+bid*4*N]=cuCaddf(hess0[2*tid+2*N+1+bid*4*N],hs[8*ci+7]);
    }
   }
  }
  __syncthreads();
}


__global__ void 
kernel_fns_fgrad(int N, int Nbase, const cuFloatComplex *__restrict__ x, cuFloatComplex *__restrict__ eta0, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh) {

  /* eta0: each block will store result in its own block */
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  /* 4x2xblockDim.x cuFloatComplex values and 2xblockDim.x int values */
  extern __shared__ cuFloatComplex eta[];
  int *stm= (int*)&eta[8*blockDim.x];
  stm[2*tid]=-1;
  stm[2*tid+1]=-1;
  /* x,eta: 2Nx2 matrix */ 
  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    if (sta1>=0 && sta2>=0) {   
     stm[2*tid]=sta1;
     stm[2*tid+1]=sta2;

     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     /* J1 */
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     /* conjugate this to get J2^H */
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* G1*C*G2' */
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     ambt(T1,G2,T2);

     /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
     /* V->U */
     cuFloatComplex res[4];
     res[0]=cuCsubf(make_cuFloatComplex(y[8*n],y[8*n+1]),T2[0]);
     res[1]=cuCsubf(make_cuFloatComplex(y[8*n+2],y[8*n+3]),T2[1]);
     res[2]=cuCsubf(make_cuFloatComplex(y[8*n+4],y[8*n+5]),T2[2]);
     res[3]=cuCsubf(make_cuFloatComplex(y[8*n+6],y[8*n+7]),T2[3]);

     /* 
      grad(2*p-1:2*p,:)=grad(2*p-1:2*p,:)+res*x(2*q-1:2*q,:)*C';
      grad(2*q-1:2*q,:)=grad(2*q-1:2*q,:)+res'*x(2*p-1:2*p,:)*C;
     */
     /* res*G2*C' */
     amb(res,G2,T1);
     ambt(T1,C,T2);

     eta[8*tid]=T2[0];
     eta[8*tid+1]=T2[1];
     eta[8*tid+2]=T2[2];
     eta[8*tid+3]=T2[3];


     /* res'*G1*C */
     atmb(res,G1,T1);
     amb(T1,C,T2);

     eta[8*tid+4]=T2[0];
     eta[8*tid+5]=T2[1];
     eta[8*tid+6]=T2[2];
     eta[8*tid+7]=T2[3];

    }  
  } 

  __syncthreads();
  /* copy back to global memory */
  if (tid==0) {
   for(int ci=0; ci<blockDim.x; ci++) {
    int sta1=stm[2*ci]; 
    int sta2=stm[2*ci+1];
    if (sta1>=0 && sta2>=0) {
      eta0[2*sta1+bid*4*N]=cuCaddf(eta0[2*sta1+bid*4*N],eta[8*ci]);
      eta0[2*sta1+2*N+bid*4*N]=cuCaddf(eta0[2*sta1+2*N+bid*4*N],eta[8*ci+1]);
      eta0[2*sta1+1+bid*4*N]=cuCaddf(eta0[2*sta1+1+bid*4*N],eta[8*ci+2]);
      eta0[2*sta1+2*N+1+bid*4*N]=cuCaddf(eta0[2*sta1+2*N+1+bid*4*N],eta[8*ci+3]);
      eta0[2*sta2+bid*4*N]=cuCaddf(eta0[2*sta2+bid*4*N],eta[8*ci+4]);
      eta0[2*sta2+2*N+bid*4*N]=cuCaddf(eta0[2*sta2+2*N+bid*4*N],eta[8*ci+5]);
      eta0[2*sta2+1+bid*4*N]=cuCaddf(eta0[2*sta2+1+bid*4*N],eta[8*ci+6]);
      eta0[2*sta2+2*N+1+bid*4*N]=cuCaddf(eta0[2*sta2+2*N+1+bid*4*N],eta[8*ci+7]);

    }
   }
  }
  __syncthreads();
}

__global__ void 
kernel_fns_fgrad_robust(int N, int Nbase, const cuFloatComplex *__restrict__ x, cuFloatComplex *__restrict__ eta0, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, const float *__restrict__ wtd) {

  /* eta0: each block will store result in its own block */
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  /* baselines */
  int nbase=N*(N-1)/2;
  /* blocks per timeslot */
  int Bt=(nbase+blockDim.x-1)/blockDim.x;
  
  /* which timeslot */
  int ntime=bid/Bt;
  /* which offset */
  int noff=bid%Bt;
  /* local index within one timeslot, 0...N(N-1)/2-1 */
  unsigned int m = noff*blockDim.x+threadIdx.x;
  /* global thread index : less than the total baselines */
  unsigned int n = ntime*nbase+m;
  /* 4x2xblockDim.x cuFloatComplex values and 2xblockDim.x int values */
  extern __shared__ cuFloatComplex eta[];
  int *stm= (int*)&eta[8*blockDim.x];
  stm[2*tid]=-1;
  stm[2*tid+1]=-1;
  /* x,eta: 2Nx2 matrix */ 
  if(m<nbase && n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    if (sta1>=0 && sta2>=0) {   
     stm[2*tid]=sta1;
     stm[2*tid+1]=sta2;

     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     /* J1 */
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     /* conjugate this to get J2^H */
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* G1*C*G2' */
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     ambt(T1,G2,T2);

     /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
     /* V->U */
     cuFloatComplex res[4];
     res[0]=cuCsubf(make_cuFloatComplex(y[8*n],y[8*n+1]),T2[0]);
     res[1]=cuCsubf(make_cuFloatComplex(y[8*n+2],y[8*n+3]),T2[1]);
     res[2]=cuCsubf(make_cuFloatComplex(y[8*n+4],y[8*n+5]),T2[2]);
     res[3]=cuCsubf(make_cuFloatComplex(y[8*n+6],y[8*n+7]),T2[3]);

     /* 
      grad(2*p-1:2*p,:)=grad(2*p-1:2*p,:)+res*x(2*q-1:2*q,:)*C';
      grad(2*q-1:2*q,:)=grad(2*q-1:2*q,:)+res'*x(2*p-1:2*p,:)*C;
     */
     /* res*G2*C' */
     amb(res,G2,T1);
     ambt(T1,C,T2);
    
     float wtdn=wtd[n];
     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;

     eta[8*tid]=T2[0];
     eta[8*tid+1]=T2[1];
     eta[8*tid+2]=T2[2];
     eta[8*tid+3]=T2[3];


     /* res'*G1*C */
     atmb(res,G1,T1);
     amb(T1,C,T2);

     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;


     eta[8*tid+4]=T2[0];
     eta[8*tid+5]=T2[1];
     eta[8*tid+6]=T2[2];
     eta[8*tid+7]=T2[3];

    }  
  } 

  __syncthreads();
  /* copy back to global memory */
  if (tid<N) {
   for(int ci=0; ci<blockDim.x; ci++) {
    int sta1=stm[2*ci]; 
    int sta2=stm[2*ci+1];
    if (sta1==tid) { /* note, tid >=0 always */
      eta0[2*tid+bid*4*N]=cuCaddf(eta0[2*tid+bid*4*N],eta[8*ci]);
      eta0[2*tid+2*N+bid*4*N]=cuCaddf(eta0[2*tid+2*N+bid*4*N],eta[8*ci+1]);
      eta0[2*tid+1+bid*4*N]=cuCaddf(eta0[2*tid+1+bid*4*N],eta[8*ci+2]);
      eta0[2*tid+2*N+1+bid*4*N]=cuCaddf(eta0[2*tid+2*N+1+bid*4*N],eta[8*ci+3]); 
    }
    if (sta2==tid) { /* note, tid >=0 always */
      eta0[2*tid+bid*4*N]=cuCaddf(eta0[2*tid+bid*4*N],eta[8*ci+4]);
      eta0[2*tid+2*N+bid*4*N]=cuCaddf(eta0[2*tid+2*N+bid*4*N],eta[8*ci+5]);
      eta0[2*tid+1+bid*4*N]=cuCaddf(eta0[2*tid+1+bid*4*N],eta[8*ci+6]);
      eta0[2*tid+2*N+1+bid*4*N]=cuCaddf(eta0[2*tid+2*N+1+bid*4*N],eta[8*ci+7]);
    }
   }
  }
  __syncthreads();
}

__global__ void
kernel_fns_sumblocks_pertime(int N, int Nblocks, int offset, cuFloatComplex *__restrict__ eta0) {
 /* offset: values in 0...4N
    each block will sum Nblocks in eta0 and store it in first value */
  extern __shared__ cuFloatComplex etas[];
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  int gtid=tid+offset;
  /* this block will work on blocks bid*Nblocks,bid*Nblocks+1,...,(bid+1)Nblocks-1 */
  /* each thread will work with Nblocks values */
  /* load global data */
  if (gtid < 4*N) {
   for (int ci=0; ci<Nblocks; ci++) {
    etas[tid*Nblocks+ci]=eta0[gtid+(bid*Nblocks+ci)*4*N];
   }
  }
  __syncthreads();
  if (gtid < 4*N) {
   for (int ci=1; ci<Nblocks; ci++) {
     etas[tid*Nblocks]=cuCaddf(etas[tid*Nblocks+ci],etas[tid*Nblocks]);
   }
  }
  __syncthreads();
  if (gtid < 4*N) {
   eta0[gtid+bid*Nblocks*4*N]=etas[tid*Nblocks];
  }
  __syncthreads();
}


__global__ void
kernel_fns_sumblocks_alltime(int N, int Nblocks, int Ntime, int offset, cuFloatComplex *__restrict__ eta0, cuFloatComplex *__restrict__ eta) {
  /* offset: value in 0...Ntime-1 */
  /* each block will sum values out of 4N, each thread will read values from different blocks separated by Nblocks */
  /* also move blocks separed by Nblocks close together */
  extern __shared__ cuFloatComplex etas[];
  int bid=blockIdx.x; /* 0..4N-1 */
  int tid=threadIdx.x;
  int gbid=4*N*Nblocks*(tid+offset)+bid;
  etas[tid]=make_cuFloatComplex(0.0f,0.0f);
  if (tid+offset<Ntime) {
   etas[tid]=eta0[gbid];
  }
  __syncthreads();

  /* also copy back the value to eta0, removing Nblocks space
    this can be done now because the value is already copied to shared mem. */
  if (tid+offset<Ntime) {
   eta0[4*N*(tid+offset)+bid]=etas[tid];
  }
  __syncthreads();

  /* summation over elements assuming length is a power of two */
  for(int s=blockDim.x/2; s>0; s=s/2) {
    if(tid < s) { etas[tid] = cuCaddf(etas[tid],etas[tid + s]); }
   __syncthreads();
  }


  /* add to proper location in eta */
  if(tid==0 && bid<4*N) {
   eta[bid]=cuCaddf(etas[tid],eta[bid]);
  }
  __syncthreads();
}


__global__ void
kernel_fns_sumelements_alltime(int Ntime,int offset, const cuFloatComplex *__restrict__ eta0, cuFloatComplex *__restrict__ C) {
  /* C: 2x2, eta0: 2x2Ntime, sum eta0 and store it in C (C initialized to 0) */
  /* 4 blocks, blockDim.x threads */
  extern __shared__ cuFloatComplex etas[];
  int bid=blockIdx.x; /* 0..3 add to C[bid] */
  int tid=threadIdx.x; /* 0...Ntime-1 */
  int gtid=4*(tid+offset)+bid;
  etas[tid]=make_cuFloatComplex(0.0f,0.0f);
  if (tid+offset<Ntime) {
   etas[tid]=eta0[gtid];
  }
  __syncthreads();

  /* summation over elements assuming length is a power of two */
  for(int s=blockDim.x/2; s>0; s=s/2) {
    if(tid < s) { etas[tid] = cuCaddf(etas[tid],etas[tid + s]); }
   __syncthreads();
  }

  /* add to proper location in C */
  if(tid==0 && bid<4) {
   C[bid]=cuCaddf(etas[tid],C[bid]);
  }
  __syncthreads();

}


__global__ void
kernel_fns_rhs_alltime(cuFloatComplex *__restrict__ C) {
 /* C: 2 x 2 Nblocks , each block (4) threads will work on 2x2 matrix */
  extern __shared__ cuFloatComplex etas[];
  int bid=blockIdx.x; /* 0..ntime-1 */
  int tid=threadIdx.x; /* 0..3 */

  /* load data to shared mem, X^H Z */
  if (tid<4) {
    etas[tid]=C[bid*4+tid];
  }
  __syncthreads();

  /* now find X^H-Z^H X */
  cuFloatComplex a,b;
  if (tid==0) {
   a=etas[0]; b=etas[0];  
  } else if (tid==1) {
   a=etas[2]; b=etas[1];  
  } else if (tid==2) {
   a=etas[1]; b=etas[2];  
  } else {
   a=etas[3]; b=etas[3];  
  }
  etas[tid]=cuCsubf(a,cuConjf(b));
  __syncthreads();

  /* write back to C */
  if (tid<4) {
    C[bid*4+tid]=etas[tid];
  }
  __syncthreads();

}

__global__ void 
kernel_fns_fgrad_robust1(int N, int Nbase, const cuFloatComplex *__restrict__ x, cuFloatComplex *__restrict__ eta0, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, const float *__restrict__ wtd) {

  /* eta0: each block will store result in its own block */
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  /* 4x2xblockDim.x cuFloatComplex values and 2xblockDim.x int values */
  extern __shared__ cuFloatComplex eta[];
  int *stm= (int*)&eta[8*blockDim.x];
  stm[2*tid]=-1;
  stm[2*tid+1]=-1;
  /* x,eta: 2Nx2 matrix */ 
  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    if (sta1>=0 && sta2>=0) {   
     stm[2*tid]=sta1;
     stm[2*tid+1]=sta2;

     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     /* J1 */
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     /* conjugate this to get J2^H */
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* G1*C*G2' */
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     ambt(T1,G2,T2);

     /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
     /* V->U */
     cuFloatComplex res[4];
     res[0]=cuCsubf(make_cuFloatComplex(y[8*n],y[8*n+1]),T2[0]);
     res[1]=cuCsubf(make_cuFloatComplex(y[8*n+2],y[8*n+3]),T2[1]);
     res[2]=cuCsubf(make_cuFloatComplex(y[8*n+4],y[8*n+5]),T2[2]);
     res[3]=cuCsubf(make_cuFloatComplex(y[8*n+6],y[8*n+7]),T2[3]);

     /* 
      grad(2*p-1:2*p,:)=grad(2*p-1:2*p,:)+res*x(2*q-1:2*q,:)*C';
      grad(2*q-1:2*q,:)=grad(2*q-1:2*q,:)+res'*x(2*p-1:2*p,:)*C;
     */
     /* res*G2*C' */
     amb(res,G2,T1);
     ambt(T1,C,T2);
    
     float wtdn=wtd[n];
     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;

     eta[8*tid]=T2[0];
     eta[8*tid+1]=T2[1];
     eta[8*tid+2]=T2[2];
     eta[8*tid+3]=T2[3];


     /* res'*G1*C */
     atmb(res,G1,T1);
     amb(T1,C,T2);

     /* mult with weights */
     T2[0].x=wtdn*T2[0].x;
     T2[0].y=wtdn*T2[0].y;
     T2[1].x=wtdn*T2[1].x;
     T2[1].y=wtdn*T2[1].y;
     T2[2].x=wtdn*T2[2].x;
     T2[2].y=wtdn*T2[2].y;
     T2[3].x=wtdn*T2[3].x;
     T2[3].y=wtdn*T2[3].y;


     eta[8*tid+4]=T2[0];
     eta[8*tid+5]=T2[1];
     eta[8*tid+6]=T2[2];
     eta[8*tid+7]=T2[3];

    }  
  } 

  __syncthreads();
  /* copy back to global memory */
  if (tid==0) {
   for(int ci=0; ci<blockDim.x; ci++) {
    int sta1=stm[2*ci]; 
    int sta2=stm[2*ci+1];
    if (sta1>=0 && sta2>=0) {
      eta0[2*sta1+bid*4*N]=cuCaddf(eta0[2*sta1+bid*4*N],eta[8*ci]);
      eta0[2*sta1+2*N+bid*4*N]=cuCaddf(eta0[2*sta1+2*N+bid*4*N],eta[8*ci+1]);
      eta0[2*sta1+1+bid*4*N]=cuCaddf(eta0[2*sta1+1+bid*4*N],eta[8*ci+2]);
      eta0[2*sta1+2*N+1+bid*4*N]=cuCaddf(eta0[2*sta1+2*N+1+bid*4*N],eta[8*ci+3]);
      eta0[2*sta2+bid*4*N]=cuCaddf(eta0[2*sta2+bid*4*N],eta[8*ci+4]);
      eta0[2*sta2+2*N+bid*4*N]=cuCaddf(eta0[2*sta2+2*N+bid*4*N],eta[8*ci+5]);
      eta0[2*sta2+1+bid*4*N]=cuCaddf(eta0[2*sta2+1+bid*4*N],eta[8*ci+6]);
      eta0[2*sta2+2*N+1+bid*4*N]=cuCaddf(eta0[2*sta2+2*N+1+bid*4*N],eta[8*ci+7]);

    }
   }
  }
  __syncthreads();
}

__global__ void 
kernel_fns_fgradsum(int N, int B, int blockDim_2, const cuFloatComplex *__restrict__ etaloc, cuFloatComplex *__restrict__ eta) {
  int bid=blockIdx.x;
  int tid=threadIdx.x;
  /* B x cuFloatComplex values */
  extern __shared__ cuFloatComplex etas[];
  etas[tid]=make_cuFloatComplex(0.0f,0.0f);
  if (tid<B) {
   etas[tid]=etaloc[tid*4*N+bid]; 
  }  
  __syncthreads();
  // Build summation tree over elements, handling case where B is not a power of two.
  int nTotalThreads = blockDim_2; // Total number of threads, rounded up to the next power of two
  while(nTotalThreads > 1) {
   int halfPoint = (nTotalThreads >> 1); // divide by two
    if (tid < halfPoint) {
     int thread2 = tid + halfPoint;
     if (thread2 < blockDim.x) { // Skipping the fictitious threads blockDim.x ... blockDim_2-1
      etas[tid] = cuCaddf(etas[tid],etas[thread2]);
     }
    }
    __syncthreads();
    nTotalThreads = halfPoint; // Reducing the binary tree size by two
  }

  /* add back the sum to proper location in eta */
  if(tid==0) {
   eta[bid]=cuCaddf(eta[bid],etas[0]);
  }
}

__global__ void 
kernel_fns_f(int N, int Nbase, const cuFloatComplex *__restrict__ x, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, float *__restrict__ ed) {

  // Each block saves error into shared memory
  extern __shared__ float ek[];
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int tid = threadIdx.x;

  /* this thread works on 
    coh[8*M*n:8*M*n+8*M-1]
    bb[2*n:2*n+1] (sta1,sta2)
    organization of p (N stations and M clusters)
             sta 0          sta 1           sta 2        ....  sta N-1 
  clus 0   0...7            8...15          16...23      ...   8N-8     8N-1
  clus 1   8N..8N+7         8N+8..8N+15     8N+16..8N+23 ....  8N+8N-8...8N+8N-1
  ......
  clus M-1 (M-1)N..(M-1)N+7 (M-1)N+8..(M-1)N+15....  ...(M-1)N+8N-8 (M-1)N+8N-1

    organization of coherencies (coh)
        [0, 8*M-1] : baseline 0
        [8*M, 8*M+8*M-1]: baseline 1
        [n*8*M, n*8*M+8*M-1]: baseline n
        ......
        [n*8*M+cm*8, n*8*M+cm*8+7]  cluster cm, baseline n

    x: 2Nx2 matrix
  */ 
  ek[tid]=0.0f;  
  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];

    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    float sumn=0.0f;
    float temp1,temp2,tt,yy,c=0.0f;
    if (sta1>=0 && sta2>=0) {
     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     /* T=T*G2' */
     ambt(T1,G2,T2);

     /* error using Kahan summation */
     /* V->U */
     temp1=y[8*n]-T2[0].x; 
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+1]-T2[0].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+2]-T2[1].x;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+3]-T2[1].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+4]-T2[2].x;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+5]-T2[2].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+6]-T2[3].x;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+7]-T2[3].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     ek[tid]=sumn;  
    } 
  }

  __syncthreads();
  // Build summation tree over elements, assuming blockDim.x is power of 2.
  for(int s=blockDim.x/2; s>0; s=s/2) {
    if(tid < s) ek[tid] += ek[tid + s];
   __syncthreads();
  }

  /* copy back the sum to proper location in ed */
  if(tid==0) {
   ed[blockIdx.x]=ek[0];
  }
}

__global__ void 
kernel_fns_f_robust(int N, int Nbase, const cuFloatComplex *__restrict__ x, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, const float *__restrict__ wtd, float *__restrict__ ed) {

  // Each block saves error into shared memory
  extern __shared__ float ek[];
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int tid = threadIdx.x;

  /* this thread works on 
    coh[8*M*n:8*M*n+8*M-1]
    bb[2*n:2*n+1] (sta1,sta2)
    organization of p (N stations and M clusters)
             sta 0          sta 1           sta 2        ....  sta N-1 
  clus 0   0...7            8...15          16...23      ...   8N-8     8N-1
  clus 1   8N..8N+7         8N+8..8N+15     8N+16..8N+23 ....  8N+8N-8...8N+8N-1
  ......
  clus M-1 (M-1)N..(M-1)N+7 (M-1)N+8..(M-1)N+15....  ...(M-1)N+8N-8 (M-1)N+8N-1

    organization of coherencies (coh)
        [0, 8*M-1] : baseline 0
        [8*M, 8*M+8*M-1]: baseline 1
        [n*8*M, n*8*M+8*M-1]: baseline n
        ......
        [n*8*M+cm*8, n*8*M+cm*8+7]  cluster cm, baseline n

    x: 2Nx2 matrix
  */ 
  ek[tid]=0.0f;  
  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];

    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    float sumn=0.0f;
    float temp1,temp2,tt,yy,c=0.0f;
    if (sta1>=0 && sta2>=0) {
     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     /* T=T*G2' */
     ambt(T1,G2,T2);

     /* error using Kahan summation */
     /* V->U */
     temp1=y[8*n]-T2[0].x; 
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+1]-T2[0].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+2]-T2[1].x;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+3]-T2[1].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+4]-T2[2].x;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+5]-T2[2].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+6]-T2[3].x;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=y[8*n+7]-T2[3].y;
     temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     ek[tid]=wtd[n]*sumn;  
    } 
  }

  __syncthreads();
  // Build summation tree over elements, assuming blockDim.x is power of 2.
  for(int s=blockDim.x/2; s>0; s=s/2) {
    if(tid < s) { ek[tid] += ek[tid + s]; }
   __syncthreads();
  }

  /* copy back the sum to proper location in ed */
  if(tid==0) {
   ed[blockIdx.x]=ek[0];
  }
}


/* update weights */
__global__ void 
kernel_fns_fupdate_weights(int N, int Nbase, const cuFloatComplex *__restrict__ x, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, float *__restrict__ wtd, float nu0) {

  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;

  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    wtd[n]=1.0f; /* catch flagged baselines */
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    float sumn=0.0f;
    float temp1,temp2,tt;
    if (sta1>=0 && sta2>=0) {
     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     /* T=T*G2' */
     ambt(T1,G2,T2);

     /* use p=2, find MAX value of residual error out of XX,XY,YX,YY
        instead of the sum */
     /* V->U */
     temp1=y[8*n]-T2[0].x; 
     temp2=y[8*n+1]-T2[0].y;
     sumn=temp1*temp1+temp2*temp2;
     temp1=y[8*n+2]-T2[1].x;
     temp2=y[8*n+3]-T2[1].y;
     tt=temp1*temp1+temp2*temp2;
     if (sumn<tt) { sumn=tt; }
     
     temp1=y[8*n+4]-T2[2].x;
     temp2=y[8*n+5]-T2[2].y;
     tt=temp1*temp1+temp2*temp2;
     if (sumn<tt) { sumn=tt; }

     temp1=y[8*n+6]-T2[3].x;
     temp2=y[8*n+7]-T2[3].y;
     tt=temp1*temp1+temp2*temp2;
     if (sumn<tt) { sumn=tt; }
     //wtd[n]=(nu0+8.0f)/(nu0+sumn); /* 8 variate T distribution */ 
     wtd[n]=(nu0+2.0f)/(nu0+sumn); /* 2 variate T distribution */ 
    } 
  }

}

/* update weights and log(weight) */
__global__ void 
kernel_fns_fupdate_weights_q(int N, int Nbase, const cuFloatComplex *__restrict__ x, const float *__restrict__ y, const float *__restrict__ coh, const short *__restrict__ bbh, float *__restrict__ wtd, float *__restrict__ qd, float nu0) {

  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;

  if(n<Nbase) {
    int sta1=(int)bbh[2*n];
    int sta2=(int)bbh[2*n+1];
    wtd[n]=1.0f; /* catch flagged baselines */
    qd[n]=1.0f; /* catch flagged baselines */
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    float sumn=0.0f;
    float temp1,temp2,tt;
    if (sta1>=0 && sta2>=0) {
     cuFloatComplex G1[4];
     cuFloatComplex G2[4];
     G1[0]=x[sta1*2];
     G1[1]=x[sta1*2+N*2];
     G1[2]=x[sta1*2+1];
     G1[3]=x[sta1*2+N*2+1];
     G2[0]=x[sta2*2];
     G2[1]=x[sta2*2+N*2];
     G2[2]=x[sta2*2+1];
     G2[3]=x[sta2*2+N*2+1];

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     cuFloatComplex T1[4];
     cuFloatComplex T2[4];
     /* T=G1*C */
     amb(G1,C,T1);
     /* T=T*G2' */
     ambt(T1,G2,T2);

     /* use p=2, find MAX value of residual error out of XX,XY,YX,YY
        instead of the sum */
     /* V->U */
     temp1=y[8*n]-T2[0].x; 
     temp2=y[8*n+1]-T2[0].y;
     sumn=temp1*temp1+temp2*temp2;
     temp1=y[8*n+2]-T2[1].x;
     temp2=y[8*n+3]-T2[1].y;
     tt=temp1*temp1+temp2*temp2;
     if (sumn<tt) { sumn=tt; }
     
     temp1=y[8*n+4]-T2[2].x;
     temp2=y[8*n+5]-T2[2].y;
     tt=temp1*temp1+temp2*temp2;
     if (sumn<tt) { sumn=tt; }

     temp1=y[8*n+6]-T2[3].x;
     temp2=y[8*n+7]-T2[3].y;
     tt=temp1*temp1+temp2*temp2;
     if (sumn<tt) { sumn=tt; }
     //wtd[n]=(nu0+8.0f)/(nu0+sumn); /* 8 variate T distribution */ 
     wtd[n]=(nu0+2.0f)/(nu0+sumn); /* 2 variate T distribution */ 
     qd[n]=wtd[n]-logf(wtd[n]);  
    } 
  }

}



/* sum up all N elements of vector input 
 and save (per block) in output (size > number of blocks) */
__global__ void 
plus_reduce_multi(const float *__restrict__ input, int N, int blockDim_2, float *__restrict__ output) {
 // Each block loads its elements into shared memory
 extern __shared__ float x[];
 int tid = threadIdx.x;
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 x[tid] = (i<N) ? input[i] : 0.0f; // last block may pad with 0’s
 __syncthreads();
 // Build summation tree over elements, handling case where B is not a power of two.
  int nTotalThreads = blockDim_2; // Total number of threads, rounded up to the next power of two
  while(nTotalThreads > 1) {
   int halfPoint = (nTotalThreads >> 1); // divide by two
    if (tid < halfPoint) {
     int thread2 = tid + halfPoint;
     if (thread2 < blockDim.x) { // Skipping the fictitious threads blockDim.x ... blockDim_2-1
      x[tid] = x[tid]+x[thread2];
     }
    }
    __syncthreads();
    nTotalThreads = halfPoint; // Reducing the binary tree size by two
 }

 /* add back to total */
 if( tid == 0 ) {
  output[blockIdx.x]=x[tid];
 }
}


/* sum up all N elements of vector input 
 NOTE: only 1 block should be used */
__global__ void 
plus_reduce(const float *__restrict__ input, int N, int blockDim_2, float *total) {
 // Each block loads its elements into shared memory
 extern __shared__ float x[];
 int tid = threadIdx.x;
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 x[tid] = (i<N) ? input[i] : 0.0f; // last block may pad with 0’s
 __syncthreads();
 // Build summation tree over elements, handling case where B is not a power of two.
  int nTotalThreads = blockDim_2; // Total number of threads, rounded up to the next power of two
  while(nTotalThreads > 1) {
   int halfPoint = (nTotalThreads >> 1); // divide by two
    if (tid < halfPoint) {
     int thread2 = tid + halfPoint;
     if (thread2 < blockDim.x) { // Skipping the fictitious threads blockDim.x ... blockDim_2-1
      x[tid] = x[tid]+x[thread2];
     }
    }
    __syncthreads();
    nTotalThreads = halfPoint; // Reducing the binary tree size by two
 }

 /* add back to total */
 if( tid == 0 ) {
  *total=*total+x[tid];
 }
}


__global__ void 
kernel_fns_fscale(int N,  cuFloatComplex *__restrict__ eta, const float *__restrict__ iw) {
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  if (n<N) {
   float w=iw[n];
   eta[2*n].x=eta[2*n].x*w;
   eta[2*n].y=eta[2*n].y*w;
   eta[2*n+1].x=eta[2*n+1].x*w;
   eta[2*n+1].y=eta[2*n+1].y*w;
   eta[2*n+2*N].x=eta[2*n+2*N].x*w;
   eta[2*n+2*N].y=eta[2*n+2*N].y*w;
   eta[2*n+2*N+1].x=eta[2*n+2*N+1].x*w;
   eta[2*n+2*N+1].y=eta[2*n+2*N+1].y*w;
  }
  __syncthreads();
}

/* only use extern if calling code is C */
extern "C"
{

static void
checkCublasError(cublasStatus_t cbstatus, const char *file, int line)
{
#ifdef CUDA_DBG
   if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUBLAS failure\n",file,line);
    exit(EXIT_FAILURE);
   }
#endif
}

/* need power of 2 for tree reduction to work */
static int 
NearestPowerOf2 (int n){
  if (!n) return n;  //(0 == 2^0)
 
  int x = 1;
  while(x < n) {
      x <<= 1;
  }
  return x;
}

/* 
 cost function:
  N: no of stations
  M: no of constraints (baselines)
  x: solution 2Nx2 complex float
  y: data 8M float (8 for each baseline)
  coh: coherency
  bbh: baseline->station mapping

  return ed: error vector, BlocksPerGridx1
*/
/* need BlocksPerGrid+1+L float storage */
float 
cudakernel_fns_f(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  float *ed,*eo;
  cudaMalloc((void**)&ed, sizeof(float)*BlocksPerGrid);
  cudaMemset(ed, 0, sizeof(float)*BlocksPerGrid);
  kernel_fns_f<<< BlocksPerGrid, ThreadsPerBlock, sizeof(float)*ThreadsPerBlock >>>(N, M, x, y, coh, bbh,ed);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
  float total;
  float *totald;
  cudaMalloc((void**)&totald, sizeof(float));
  cudaMemset(totald, 0, sizeof(float));
  int T=DEFAULT_TH_PER_BK; /* max possible threads, use a smaller no to have large no. of blocks, but not too large to exceed no. of. SMs in the card*/
  /* we use 1 block, so need to launch BlocksPerGrid number of threads */
  if (BlocksPerGrid<T) {
    /* one kernel launch is enough */
    plus_reduce<<< 1, BlocksPerGrid, sizeof(float)*BlocksPerGrid>>>(ed, BlocksPerGrid, NearestPowerOf2(BlocksPerGrid), totald);
  } else {
    /* multiple kernel launches */
    int L=(BlocksPerGrid+T-1)/T;
    cudaMalloc((void**)&eo, sizeof(float)*L);
    plus_reduce_multi<<< L, T, sizeof(float)*T>>>(ed, BlocksPerGrid, NearestPowerOf2(T), eo);
    plus_reduce<<< 1, L, sizeof(float)*L>>>(eo, L, NearestPowerOf2(L), totald);
    cudaFree(eo);
  }
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
  cudaMemcpy(&total,totald,sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(ed);
  cudaFree(totald);
  return total;
}

/* 
 robust cost function:
  N: no of stations
  M: no of constraints (baselines)
  x: solution 2Nx2 complex float
  y: data 8M float (8 for each baseline)
  coh: coherency
  bbh: baseline->station mapping
  wtd: weight Mx1

  return ed: error vector, BlocksPerGridx1
*/
/* need BlocksPerGrid+4+L float storage <= (2 BlocksPerGrid + 4) */
float 
cudakernel_fns_f_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  float *ed,*eo;
  cudaMalloc((void**)&ed, sizeof(float)*BlocksPerGrid);
  cudaMemset(ed, 0, sizeof(float)*BlocksPerGrid);
  kernel_fns_f_robust<<< BlocksPerGrid, ThreadsPerBlock, sizeof(float)*ThreadsPerBlock >>>(N, M, x, y, coh, bbh, wtd, ed);

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
  float total;
  float *totald;
  cudaMalloc((void**)&totald, sizeof(float));
  cudaMemset(totald, 0, sizeof(float));
  int T=DEFAULT_TH_PER_BK; /* max possible threads, use a smaller no to have large no. of blocks, but not too large to exceed no. of. SMs in the card*/
  /* we use 1 block, so need to launch BlocksPerGrid number of threads */
  if (BlocksPerGrid<T) {
    /* one kernel launch is enough */
    plus_reduce<<< 1, BlocksPerGrid, sizeof(float)*BlocksPerGrid>>>(ed, BlocksPerGrid, NearestPowerOf2(BlocksPerGrid), totald);
  } else {
    /* multiple kernel launches */
    int L=(BlocksPerGrid+T-1)/T;
    cudaMalloc((void**)&eo, sizeof(float)*L);
    plus_reduce_multi<<< L, T, sizeof(float)*T>>>(ed, BlocksPerGrid, NearestPowerOf2(T), eo);
    plus_reduce<<< 1, L, sizeof(float)*L>>>(eo, L, NearestPowerOf2(L), totald);
    cudaFree(eo);
  }
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
  cudaMemcpy(&total,totald,sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(ed);
  cudaFree(totald);

  return total;
}

/* gradient, output eta: reset to 0 initially */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fgradflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
  /* eachekernel require 2xThreadsPerBlocx2 x 2 complex float for storing eta values
    2*ThreadsPerBloc x1 int array for station numbers
     and  */
  kernel_fns_fgrad<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, etaloc, y, coh, bbh);

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  int T=256; /* max possible threads */
  /* now create 4N blocks, threads in each block  will read BlocksPerGrid values from etalocal and find average, so no of threads>= BlocksPerGrid */
  /* each block need BlocksPerGrid float complex values */
  if (T>BlocksPerGrid) {
   int B=((BlocksPerGrid+1)/2)*2; /* even no of threads */
   kernel_fns_fgradsum<<< 4*N, B, sizeof(cuFloatComplex)*B>>>(N, BlocksPerGrid, NearestPowerOf2(B),  etaloc, eta);
#ifdef CUDA_DBG
   error = cudaGetLastError();
   if(error != cudaSuccess) {
     // print the CUDA error message and exit
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
   }
#endif
  } else {
   /* iterate over T values */
   int L=(BlocksPerGrid+T-1)/T;
   int ct=0;
   int myT;
   for (int ci=0; ci<L; ci++) {
    if (ct+T<BlocksPerGrid) {
      myT=T;
    } else {
      myT=BlocksPerGrid-ct;
    }
    kernel_fns_fgradsum<<< 4*N, myT, sizeof(cuFloatComplex)*myT >>>(N, myT, NearestPowerOf2(myT),  &etaloc[ct*4*N], eta);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    if(error != cudaSuccess) {
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
     exit(-1);
    }
#endif
    ct=ct+T;
   }
  }

  cudaFree(etaloc);
}

/* Robust gradient, output eta: reset to 0 initially */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fgradflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
  /* eachekernel require 2xThreadsPerBlocx2 x 2 complex float for storing eta values
    2*ThreadsPerBloc x1 int array for station numbers
     and  */
  kernel_fns_fgrad_robust1<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, etaloc, y, coh, bbh, wtd);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  int T=256; /* max possible threads */
  /* now create 4N blocks, threads in each block  will read BlocksPerGrid values from etalocal and find average, so no of threads>= BlocksPerGrid */
  /* each block need BlocksPerGrid float complex values */
  if (T>BlocksPerGrid) {
   int B=((BlocksPerGrid+1)/2)*2; /* even no of threads */
   kernel_fns_fgradsum<<< 4*N, B, sizeof(cuFloatComplex)*B>>>(N, BlocksPerGrid, NearestPowerOf2(B),  etaloc, eta);
#ifdef CUDA_DBG
   error = cudaGetLastError();
   if(error != cudaSuccess) {
     // print the CUDA error message and exit
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
   }
#endif
  } else {
   /* iterate over T values */
   int L=(BlocksPerGrid+T-1)/T;
   int ct=0;
   int myT;
   for (int ci=0; ci<L; ci++) {
    if (ct+T<BlocksPerGrid) {
      myT=T;
    } else {
      myT=BlocksPerGrid-ct;
    }
    kernel_fns_fgradsum<<< 4*N, myT, sizeof(cuFloatComplex)*myT >>>(N, myT, NearestPowerOf2(myT),  &etaloc[ct*4*N], eta);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    if(error != cudaSuccess) {
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
     exit(-1);
    }
#endif
    ct=ct+T;
   }
  }
  cudaFree(etaloc);
}


/* Robust gradient, output eta: reset to 0 initially */
/* Ai: inverse of A matrix for projection */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fgradflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
  /* each block requires 2xThreadsPerBloc x2 x 2 complex float for storing eta values
   and 2*ThreadsPerBloc x1 int array for station numbers
   each block requires Nx2x2 complex float to store calculated value

   also ThreadsPerBlock>= N
   */
  kernel_fns_fgrad_robust<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, etaloc, y, coh, bbh, wtd);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
  /* now etaloc has [Z_1,Z_2,....,Z_K] where K: total blocks,
     need to add it to
    [Z_1, Z_2,....Z_t] where t: total timeslots.
    so each blocks_per_timeslot blocks will be added to just one block
    
    project [P_1,P_2,...,P_t]=[Z_1,Z_2,..,Z_t]-J[U_1,U_2,...,U_t]
    where U_i is the projection matrix obtained by solving Sylvester equation,
    for that we need J^H [Z_1,Z_2,...,Z_t]
  */
  /* baselines */
  int nbase=N*(N-1)/2;
  /* blocks per timeslot */
  int Bt=(nbase+ThreadsPerBlock-1)/ThreadsPerBlock;
  /* timeslots */
  int ntime=(M+nbase-1)/nbase;
  /* threads to use (need to use small value to enable enough shared memory) */
  int T=DEFAULT_TH_PER_BK_2;
  /* sum Bt values each to the first value */
  for (int ci=0; ci<(4*N+T-1)/T; ci++) {
    /* create blocks equal to timeslots, each will need Bt*T complex float storage, ci*T is the offset of 0...4N-1 values */
    /* each thread will sum Bt values */
    kernel_fns_sumblocks_pertime<<< ntime, T, sizeof(cuFloatComplex)*Bt*T >>>(N, Bt, ci*T, etaloc);
#ifdef DEBUG
    printf("sum blocks %d, threads %d thread offset %d, numblocks/time %d\n",ntime,T,ci*T,Bt);
#endif
  }

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now create 4N blocks, each block will sum elements in 0...4N-1 of ntime values, separated by Bt blocks */
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumblocks_alltime<<< 4*N, T, sizeof(cuFloatComplex)*T >>>(N, Bt, ntime, ci*T, etaloc, eta);
#ifdef DEBUG
    printf("sum all blocks %d, timeslots %d, threads %d block offset %d, spacing %d\n",4*N,ntime,T,ci*T,Bt);
#endif
  }

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now etaloc : 4N x ntime (== 2N x 2ntime) blocks correspond to eta for each timeslot */
  /* find the product x^H etaloc == x^H Z, 
  reuse tail end of etaloc to store result, since BlocksPerGrid >> ntime */
  cuFloatComplex *C;
  C=&etaloc[8*N*ntime]; /* size 2 x 2ntime */
  //cudaMemset(C, 0, sizeof(cuFloatComplex)*4*ntime); Not needed because a2=0
  cublasStatus_t cbstatus;
  cuFloatComplex a1,a2;
  a1.x=1.0f; a1.y=0.0f;
  a2.x=0.0f; a2.y=0.0f;
  cbstatus=cublasCgemm(cbhandle,CUBLAS_OP_C,CUBLAS_OP_N,2,2*ntime,2*N,&a1,x,2*N,etaloc,2*N,&a2,C,2);
  checkCublasError(cbstatus,__FILE__,__LINE__);

  /* setup RHS matrices x^H Z - Z^H x */
  /* 2x2 matrix: 4 threads per block, ntime blocks */
  T=4;
  kernel_fns_rhs_alltime<<< ntime, T, sizeof(cuFloatComplex)*T >>>(C);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif


  /* now consider C as 4xntime matrix and multiply it with Ai */
  /* reuse etaloc first block, size needed 4 x ntime << 4N x ntime */
  cbstatus=cublasCgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_N,4,ntime,4,&a1,Ai,4,C,4,&a2,etaloc,4);
  checkCublasError(cbstatus,__FILE__,__LINE__);

  /* now average 2x 2ntime matrix etaloc to one 2x2 matrix, stoared at C */
  cudaMemset(C, 0, sizeof(cuFloatComplex)*4);
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumelements_alltime<<< 4, T, sizeof(cuFloatComplex)*T >>>(ntime,ci*T, etaloc, C);
  }
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now find eta = -1 x C + eta => C = A B + C */
  a1.x=-1.0f; a1.y=0.0f;
  a2.x=1.0f; a2.y=0.0f;
  cbstatus=cublasCgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_N,2*N,2,2,&a1,x,2*N,C,2,&a2,eta,2*N);
  checkCublasError(cbstatus,__FILE__,__LINE__);

  cudaFree(etaloc);
}


/* Robust gradient (Euclidean), output eta: reset to 0 initially */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fgradflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
  /* each block requires 2xThreadsPerBloc x2 x 2 complex float for storing eta values
   and 2*ThreadsPerBloc x1 int array for station numbers
   each block requires Nx2x2 complex float to store calculated value

   also ThreadsPerBlock>= N
   */
  kernel_fns_fgrad_robust<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, etaloc, y, coh, bbh, wtd);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
  /* now etaloc has [Z_1,Z_2,....,Z_K] where K: total blocks,
     need to add it to
    [Z_1, Z_2,....Z_t] where t: total timeslots.
    so each blocks_per_timeslot blocks will be added to just one block
    
  */
  /* baselines */
  int nbase=N*(N-1)/2;
  /* blocks per timeslot */
  int Bt=(nbase+ThreadsPerBlock-1)/ThreadsPerBlock;
  /* timeslots */
  int ntime=(M+nbase-1)/nbase;
  /* threads to use (need to use small value to enable enough shared memory) */
  int T=DEFAULT_TH_PER_BK_2;
  /* sum Bt values each to the first value */
  for (int ci=0; ci<(4*N+T-1)/T; ci++) {
    /* create blocks equal to timeslots, each will need Bt*T complex float storage, ci*T is the offset of 0...4N-1 values */
    /* each thread will sum Bt values */
    kernel_fns_sumblocks_pertime<<< ntime, T, sizeof(cuFloatComplex)*Bt*T >>>(N, Bt, ci*T, etaloc);
#ifdef DEBUG
    printf("sum blocks %d, threads %d thread offset %d, numblocks/time %d\n",ntime,T,ci*T,Bt);
#endif
  }

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now create 4N blocks, each block will sum elements in 0...4N-1 of ntime values, separated by Bt blocks */
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumblocks_alltime<<< 4*N, T, sizeof(cuFloatComplex)*T >>>(N, Bt, ntime, ci*T, etaloc, eta);
#ifdef DEBUG
    printf("sum all blocks %d, timeslots %d, threads %d block offset %d, spacing %d\n",4*N,ntime,T,ci*T,Bt);
#endif
  }

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now etaloc : 4N x ntime (== 2N x 2ntime) blocks correspond to eta for each timeslot */
  /* now average 2x 2ntime matrix etaloc to one 2x2 matrix, stoared at eta */
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumelements_alltime<<< 4, T, sizeof(cuFloatComplex)*T >>>(ntime,ci*T, etaloc, eta);
  }
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  cudaFree(etaloc);
}



/* Hessian 
  output fhess: reset to 0 initially */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fhessflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
  /* eachekernel require 2xThreadsPerBlocx2 x 2 complex float for storing eta values
    2*ThreadsPerBloc x1 int array for station numbers
     and  */
  kernel_fns_fhess<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, eta, etaloc, y, coh, bbh);

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  int T=256;
  /* now create 4N blocks, threads in each block  will read BlocksPerGrid values from etalocal and find average, so no of threads>= BlocksPerGrid */
  /* each block need BlocksPerGrid float complex values */
  if (T>BlocksPerGrid) {
   int B=((BlocksPerGrid+1)/2)*2; /* even no of threads */
   kernel_fns_fgradsum<<< 4*N, B, sizeof(cuFloatComplex)*B>>>(N, BlocksPerGrid, NearestPowerOf2(B),  etaloc, fhess);
#ifdef CUDA_DBG
   error = cudaGetLastError();
   if(error != cudaSuccess) {
     // print the CUDA error message and exit
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
   }
#endif
  } else {
   /* iterate over T values */
   int L=(BlocksPerGrid+T-1)/T;
   int ct=0;
   int myT;
   for (int ci=0; ci<L; ci++) {
    if (ct+T<BlocksPerGrid) {
      myT=T;
    } else {
      myT=BlocksPerGrid-ct;
    }
    kernel_fns_fgradsum<<< 4*N, myT, sizeof(cuFloatComplex)*myT >>>(N, myT, NearestPowerOf2(myT),  &etaloc[ct*4*N], fhess);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    if(error != cudaSuccess) {
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
     exit(-1);
    }
#endif
    ct=ct+T;
   }
  }

  cudaFree(etaloc);
}


/* Robust Hessian 
  output fhess: reset to 0 initially */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fhessflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
  /* eachekernel require 2xThreadsPerBlocx2 x 2 complex float for storing eta values
    2*ThreadsPerBloc x1 int array for station numbers
     and  */
  kernel_fns_fhess_robust1<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, eta, etaloc, y, coh, bbh, wtd);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  int T=256;
  /* now create 4N blocks, threads in each block  will read BlocksPerGrid values from etalocal and find average, so no of threads>= BlocksPerGrid */
  /* each block need BlocksPerGrid float complex values */
  if (T>BlocksPerGrid) {
   int B=((BlocksPerGrid+1)/2)*2; /* even no of threads */
   kernel_fns_fgradsum<<< 4*N, B, sizeof(cuFloatComplex)*B>>>(N, BlocksPerGrid, NearestPowerOf2(B),  etaloc, fhess);
#ifdef CUDA_DBG
   error = cudaGetLastError();
   if(error != cudaSuccess) {
     // print the CUDA error message and exit
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
   }
#endif
  } else {
   /* iterate over T values */
   int L=(BlocksPerGrid+T-1)/T;
   int ct=0;
   int myT;
   for (int ci=0; ci<L; ci++) {
    if (ct+T<BlocksPerGrid) {
      myT=T;
    } else {
      myT=BlocksPerGrid-ct;
    }
    kernel_fns_fgradsum<<< 4*N, myT, sizeof(cuFloatComplex)*myT >>>(N, myT, NearestPowerOf2(myT),  &etaloc[ct*4*N], fhess);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    if(error != cudaSuccess) {
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
     exit(-1);
    }
#endif
    ct=ct+T;
   }
  }

  cudaFree(etaloc);
}


/* Robust Hessian 
  output fhess: reset to 0 initially */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fhessflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
  /* eachekernel require 2xThreadsPerBlocx2 x 2 complex float for storing eta values
    2*ThreadsPerBloc x1 int array for station numbers
   */
  kernel_fns_fhess_robust<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, eta, etaloc, y, coh, bbh, wtd);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* baselines */
  int nbase=N*(N-1)/2;
  /* blocks per timeslot */
  int Bt=(nbase+ThreadsPerBlock-1)/ThreadsPerBlock;
  /* timeslots */
  int ntime=(M+nbase-1)/nbase;
  /* threads to use (need to use small value to enable enough shared memory) */
  int T=DEFAULT_TH_PER_BK_2;
  /* sum Bt values each to the first value */
  for (int ci=0; ci<(4*N+T-1)/T; ci++) {
    /* create blocks equal to timeslots, each will need Bt*T complex float storage, ci*T is the offset of 0...4N-1 values */
    /* each thread will sum Bt values */
    kernel_fns_sumblocks_pertime<<< ntime, T, sizeof(cuFloatComplex)*Bt*T >>>(N, Bt, ci*T, etaloc);
#ifdef DEBUG
    printf("sum blocks %d, threads %d thread offset %d, numblocks/time %d\n",ntime,T,ci*T,Bt);
#endif
  }
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now create 4N blocks, each block will sum elements in 0...4N-1 of ntime values, separated by Bt blocks */
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumblocks_alltime<<< 4*N, T, sizeof(cuFloatComplex)*T >>>(N, Bt, ntime, ci*T, etaloc, fhess);
#ifdef DEBUG
    printf("sum all blocks %d, timeslots %d, threads %d block offset %d, spacing %d\n",4*N,ntime,T,ci*T,Bt);
#endif
  }

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now etaloc : 4N x ntime (== 2N x 2ntime) blocks correspond to eta for each timeslot */
  /* find the product x^H etaloc == x^H Z, 
  reuse tail end of etaloc to store result, since BlocksPerGrid >> ntime */
  cuFloatComplex *C;
  C=&etaloc[8*N*ntime]; /* size 2 x 2ntime */
  //cudaMemset(C, 0, sizeof(cuFloatComplex)*4*ntime); Not needed because a2=0
  cublasStatus_t cbstatus;
  cuFloatComplex a1,a2;
  a1.x=1.0f; a1.y=0.0f;
  a2.x=0.0f; a2.y=0.0f;
  cbstatus=cublasCgemm(cbhandle,CUBLAS_OP_C,CUBLAS_OP_N,2,2*ntime,2*N,&a1,x,2*N,etaloc,2*N,&a2,C,2);
  checkCublasError(cbstatus,__FILE__,__LINE__);

  /* setup RHS matrices x^H Z - Z^H x */
  /* 2x2 matrix: 4 threads per block, ntime blocks */
  T=4;
  kernel_fns_rhs_alltime<<< ntime, T, sizeof(cuFloatComplex)*T >>>(C);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif


 /* now consider C as 4xntime matrix and multiply it with Ai */
  /* reuse etaloc first block, size needed 4 x ntime << 4N x ntime */
  cbstatus=cublasCgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_N,4,ntime,4,&a1,Ai,4,C,4,&a2,etaloc,4);
  checkCublasError(cbstatus,__FILE__,__LINE__);

  /* now average 2x 2ntime matrix etaloc to one 2x2 matrix, stoared at C */
  cudaMemset(C, 0, sizeof(cuFloatComplex)*4);
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumelements_alltime<<< 4, T, sizeof(cuFloatComplex)*T >>>(ntime,ci*T, etaloc, C);
  }
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now find fhess = -1 x C + fhess => C = A B + C */
  a1.x=-1.0f; a1.y=0.0f;
  a2.x=1.0f; a2.y=0.0f;
  cbstatus=cublasCgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_N,2*N,2,2,&a1,x,2*N,C,2,&a2,fhess,2*N);
  checkCublasError(cbstatus,__FILE__,__LINE__);

  cudaFree(etaloc);
}


/* Robust Hessian (Euclidean)
  output fhess: reset to 0 initially */
/* need 8N*BlocksPerGrid float storage */
void
cudakernel_fns_fhessflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cuFloatComplex *etaloc;
  /* each block stores result in 2Nx2 block, so need BlocksPerGrid x 2Nx 2 storage */
  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
  /* eachekernel require 2xThreadsPerBlocx2 x 2 complex float for storing eta values
    2*ThreadsPerBloc x1 int array for station numbers
   */
  kernel_fns_fhess_robust<<< BlocksPerGrid, ThreadsPerBlock, sizeof(cuFloatComplex)*8*ThreadsPerBlock + sizeof(int)*2*ThreadsPerBlock >>>(N, M, x, eta, etaloc, y, coh, bbh, wtd);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* baselines */
  int nbase=N*(N-1)/2;
  /* blocks per timeslot */
  int Bt=(nbase+ThreadsPerBlock-1)/ThreadsPerBlock;
  /* timeslots */
  int ntime=(M+nbase-1)/nbase;
  /* threads to use (need to use small value to enable enough shared memory) */
  int T=DEFAULT_TH_PER_BK_2;
  /* sum Bt values each to the first value */
  for (int ci=0; ci<(4*N+T-1)/T; ci++) {
    /* create blocks equal to timeslots, each will need Bt*T complex float storage, ci*T is the offset of 0...4N-1 values */
    /* each thread will sum Bt values */
    kernel_fns_sumblocks_pertime<<< ntime, T, sizeof(cuFloatComplex)*Bt*T >>>(N, Bt, ci*T, etaloc);
#ifdef DEBUG
    printf("sum blocks %d, threads %d thread offset %d, numblocks/time %d\n",ntime,T,ci*T,Bt);
#endif
  }

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now create 4N blocks, each block will sum elements in 0...4N-1 of ntime values, separated by Bt blocks */
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumblocks_alltime<<< 4*N, T, sizeof(cuFloatComplex)*T >>>(N, Bt, ntime, ci*T, etaloc, fhess);
#ifdef DEBUG
    printf("sum all blocks %d, timeslots %d, threads %d block offset %d, spacing %d\n",4*N,ntime,T,ci*T,Bt);
#endif
  }

#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* now etaloc : 4N x ntime (== 2N x 2ntime) blocks correspond to eta for each timeslot */
  T=DEFAULT_TH_PER_BK_2; /* even number */
  for (int ci=0; ci<(ntime+T-1)/T; ci++) {
    kernel_fns_sumelements_alltime<<< 4, T, sizeof(cuFloatComplex)*T >>>(ntime,ci*T, etaloc, fhess);
  }
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  cudaFree(etaloc);
}



/* scale eta with weights wt 
  N stations
  eta: 4Nx2 complex float
  iw: N x 1 weights, per station
*/
void
cudakernel_fns_fscale(int N, cuFloatComplex *eta, float *iw) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
 /* since N is small ~60, use small no. of threads per block */
  int T=32;
  int B=(N+T-1)/T;
  kernel_fns_fscale<<< T, B>>>(N, eta, iw);
  cudaDeviceSynchronize();
#ifdef CUDA_DBG
 error = cudaGetLastError();
 if(error != cudaSuccess) {
  fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
  exit(-1);
 }
#endif
}


/* 
  update weight vector (nu+1)/(nu+error^2):
  N: no of stations
  M: no of constraints (baselines)
  x: solution 2Nx2 complex float
  y: data 8M float (8 for each baseline)
  coh: coherency
  bbh: baseline->station mapping
  wtd: weight Mx1

*/
void
cudakernel_fns_fupdate_weights(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float nu0) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_fns_fupdate_weights<<< BlocksPerGrid, ThreadsPerBlock >>>(N, M, x, y, coh, bbh, wtd, nu0);
  cudaDeviceSynchronize();
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
}

/* 
  update weight vector (nu+1)/(nu+error^2) and log(weight) :
  N: no of stations
  M: no of constraints (baselines)
  x: solution 2Nx2 complex float
  y: data 8M float (8 for each baseline)
  coh: coherency
  bbh: baseline->station mapping
  wtd: weight Mx1
  qd: weight Mx1
*/
void
cudakernel_fns_fupdate_weights_q(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float *qd, float nu0) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_fns_fupdate_weights_q<<< BlocksPerGrid, ThreadsPerBlock >>>(N, M, x, y, coh, bbh, wtd, qd, nu0);
  cudaDeviceSynchronize();
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
}
}
