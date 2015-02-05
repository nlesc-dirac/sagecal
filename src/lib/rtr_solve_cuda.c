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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "sagecal.h"
#include <cuda_runtime.h>

//#define DEBUG
/* helper functions for diagnostics */
static void
checkStatus(culaStatus status, char *file, int line)
{
    char buf[80];
    if(!status)
        return;
    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    fprintf(stderr,"GPU (CULA): %s %s %d\n", buf,file,line);
    culaShutdown();
    exit(EXIT_FAILURE);
}


static void
checkCudaError(cudaError_t err, char *file, int line)
{
    if(!err)
        return;
    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    culaShutdown();
    exit(EXIT_FAILURE);
}


static void
checkCublasError(cublasStatus_t cbstatus, char *file, int line)
{
   if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUBLAS failure\n",file,line);
    exit(EXIT_FAILURE);  
   }
}


/* Retraction 
   rnew: new value */
/* rnew = x + r */
void
cudakernel_fns_R(int N, cuFloatComplex *x, cuFloatComplex *r, cuFloatComplex *rnew, cublasHandle_t cbhandle) {
  cublasStatus_t cbstatus;
  cbstatus=cublasCcopy(cbhandle,4*N,x,1,rnew,1);
  cuFloatComplex alpha;
  alpha.x=1.0f; alpha.y=0.0f;
  cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, r, 1, rnew, 1);
}


/* inner product (metric) */
float
cudakernel_fns_g(int N,cuFloatComplex *x,cuFloatComplex *eta, cuFloatComplex *gamma,cublasHandle_t cbhandle) {
 /* 2 x real( trace(eta'*gamma) )
  = 2 x real( eta(:,1)'*gamma(:,1) + eta(:,2)'*gamma(:,2) )
  no need to calculate off diagonal terms
  )*/
 cublasStatus_t cbstatus;
 cuFloatComplex r1,r2;
 //complex double v1=my_cdot(2*N,eta,gamma);
 cbstatus=cublasCdotc(cbhandle,2*N,eta,1,gamma,1,&r1);
 //complex double v2=my_cdot(2*N,&eta[2*N],&gamma[2*N]);
 cbstatus=cublasCdotc(cbhandle,2*N,&eta[2*N],1,&gamma[2*N],1,&r2);

 return 2.0f*(r1.x+r2.x);
}


/* Projection 
   rnew: new value */
void
cudakernel_fns_proj(int N, cuFloatComplex *x, cuFloatComplex *z, cuFloatComplex *rnew, cublasHandle_t cbhandle) {
  /* projection  = Z-X Om, where
   Om X^H X+X^H X Om = X^H  Z - Z^H X
   is solved to find Om */

  cublasStatus_t cbstatus;
  /* find X^H X */
  cuFloatComplex xx00,xx01,xx10,xx11,*bd;
  //xx00=my_cdot(2*N,x,x);
  cbstatus=cublasCdotc(cbhandle,2*N,x,1,x,1,&xx00);
  //xx01=my_cdot(2*N,x,&x[2*N]);
  cbstatus=cublasCdotc(cbhandle,2*N,x,1,&x[2*N],1,&xx01);
  xx10=cuConjf(xx01);
  //xx11=my_cdot(2*N,&x[2*N],&x[2*N]);
  cbstatus=cublasCdotc(cbhandle,2*N,&x[2*N],1,&x[2*N],1,&xx11);

  /* find X^H Z (and using this just calculte Z^H X directly) */
  cuFloatComplex xz00,xz01,xz10,xz11;
  //xz00=my_cdot(2*N,x,z);
  cbstatus=cublasCdotc(cbhandle,2*N,x,1,z,1,&xz00);
  //xz01=my_cdot(2*N,x,&z[2*N]);
  cbstatus=cublasCdotc(cbhandle,2*N,x,1,&z[2*N],1,&xz01);
  //xz10=my_cdot(2*N,&x[2*N],z);
  cbstatus=cublasCdotc(cbhandle,2*N,&x[2*N],1,z,1,&xz10);
  //xz11=my_cdot(2*N,&x[2*N],&z[2*N]);
  cbstatus=cublasCdotc(cbhandle,2*N,&x[2*N],1,&z[2*N],1,&xz11);

  /* find X^H Z - Z^H X */
  cuFloatComplex rr00,rr01,rr10,rr11;
  //rr00=xz00-conj(xz00);
  rr00=cuCsubf(xz00,cuConjf(xz00));
  //rr01=xz01-conj(xz10);
  rr01=cuCsubf(xz01,cuConjf(xz10));
  //rr10=-conj(rr01);
  rr10.x=-rr01.x; rr10.y=rr01.y;
  //rr11=xz11-conj(xz11);
  rr11=cuCsubf(xz11,cuConjf(xz11));

  /* find I_2 kron (X^H X) + (X^H X)^T kron I_2 */
  /* A = [2*xx00  xx01       xx10         0
          xx10    xx11+xx00  0            xx10
          xx01    0          xx11+xx00    xx01
          0       xx01       xx10         2*xx11 ]
  */
  cuFloatComplex A[16],*Ad;
  A[0]=cuCmulf(make_cuFloatComplex(2.0f,0.0f),xx00);
  A[5]=A[10]=cuCaddf(xx00,xx11);
  A[15]=cuCmulf(make_cuFloatComplex(2.0f,0.0f),xx11);
  A[1]=A[8]=A[11]=A[13]=xx10;
  A[2]=A[4]=A[7]=A[14]=xx01;
  A[3]=A[6]=A[9]=A[12]=make_cuFloatComplex(0.0f,0.0f);
  cuFloatComplex b[4];
  b[0]=rr00;
  b[1]=rr10;
  b[2]=rr01;
  b[3]=rr11;

#ifdef DEBUG
  printf("BEFOREA=[\n");
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[0].x,A[0].y,A[4].x,A[4].y,A[8].x,A[8].y,A[12].x,A[12].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[1].x,A[1].y,A[5].x,A[5].y,A[9].x,A[9].y,A[13].x,A[13].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[2].x,A[2].y,A[6].x,A[6].y,A[10].x,A[10].y,A[14].x,A[14].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[3].x,A[3].y,A[7].x,A[7].y,A[11].x,A[11].y,A[15].x,A[15].y);
  printf("];\n");
  printf("BEFOREb=[\n");
  printf("%f+j*(%f)\n",b[0].x,b[0].y);
  printf("%f+j*(%f)\n",b[1].x,b[1].y);
  printf("%f+j*(%f)\n",b[2].x,b[2].y);
  printf("%f+j*(%f)\n",b[3].x,b[3].y);
  printf("];\n");
#endif


  /* solve A u = b to find u , using double precision */
  cudaMalloc((void **)&Ad, 16*sizeof(cuFloatComplex));
  cudaMemcpy(Ad,A,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
  /* copy b to device */
  cudaMalloc((void **)&bd, 4*sizeof(cuFloatComplex));
  cudaMemcpy(bd,b,4*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);

  culaStatus status;
  status=culaDeviceCgels('N',4,4,1,(culaDeviceFloatComplex *)Ad,4,(culaDeviceFloatComplex *)bd,4);
  checkStatus(status,__FILE__,__LINE__);

#ifdef DEBUG
  cudaMemcpy(b,bd,4*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
  printf("Afterb=[\n");
  printf("%f+j*(%f)\n",b[0].x,b[0].y);
  printf("%f+j*(%f)\n",b[1].x,b[1].y);
  printf("%f+j*(%f)\n",b[2].x,b[2].y);
  printf("%f+j*(%f)\n",b[3].x,b[3].y);
  printf("];\n");
#endif

  /* form Z - X * Om, where Om is given by solution b 
    but no need to rearrange b because it is already in col major order */
  //my_ccopy(4*N,z,1,rnew,1);
  cbstatus=cublasCcopy(cbhandle,4*N,z,1,rnew,1);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  //my_zgemm('N','N',2*N,2,2,-1.0+0.0*_Complex_I,z,2*N,b,2,1.0+0.0*_Complex_I,rnew,2*N);
  cuFloatComplex a1,a2;
  a1.x=-1.0f; a1.y=0.0f;
  a2.x=1.0f; a2.y=0.0f;
#ifdef DEBUG
/* read back eta for checking */
 cuFloatComplex *etalocal;
 cudaHostAlloc((void **)&etalocal, sizeof(cuFloatComplex)*4*N,cudaHostAllocDefault);
 cudaMemcpy(etalocal, rnew, 4*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
printf("Rnewbefore=[\n");
 int ci;
 for (ci=0; ci<2*N; ci++) {
  printf("%f+j*(%f) %f+j*(%f);\n",etalocal[ci].x,etalocal[ci].y,etalocal[ci+2*N].x,etalocal[ci+2*N].y);
 }
printf("]\n");
#endif

  cbstatus=cublasCgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_N,2*N,2,2,&a1,x,2*N,bd,2,&a2,rnew,2*N);

#ifdef DEBUG
  checkCublasError(cbstatus,__FILE__,__LINE__);
 cudaMemcpy(etalocal, rnew, 4*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
printf("Rnewafter=[\n");
 for (ci=0; ci<2*N; ci++) {
  printf("%f+j*(%f) %f+j*(%f);\n",etalocal[ci].x,etalocal[ci].y,etalocal[ci+2*N].x,etalocal[ci+2*N].y);
 }
printf("]\n");
 cudaFree(etalocal);
#endif
  checkCublasError(cbstatus,__FILE__,__LINE__);
  cudaFree(Ad); 
  cudaFree(bd); 
}

/* gradient, also projected to tangent space */
/* need 8N*BlocksPerGrid+ 8N*2 float storage */
static void
cudakernel_fns_fgrad(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, char *bbh, float *iw, int negate, cublasHandle_t cbhandle,float *gWORK) {
 cuFloatComplex *tempeta,*tempb;
 cublasStatus_t cbstatus;
 cuFloatComplex alpha;
 unsigned long int moff=0;
 tempeta=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N; 
 tempb=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 float *gWORK1=&gWORK[moff];
 /* max size of M for one kernel call, to determine optimal blocks */
 int T=128*ThreadsPerBlock;
 if (M<T) {
  cudakernel_fns_fgradflat(ThreadsPerBlock, BlocksPerGrid, N, M, x, tempeta, y, coh, bbh, gWORK1);
 } else {
   /* reset memory to zero */
   cudaMemset(tempeta, 0, sizeof(cuFloatComplex)*4*N);
   /* loop through M baselines */
   int L=(M+T-1)/T;
   int ct=0;
   int myT,ci;
   for (ci=0; ci<L; ci++) {
    if (ct+T<M) {
      myT=T;
    } else {
      myT=M-ct;
    }
    int B=(myT+ThreadsPerBlock-1)/ThreadsPerBlock;
    cudakernel_fns_fgradflat(ThreadsPerBlock, B, N, myT, x, tempb, &y[ct*8], &coh[ct*8], &bbh[ct*2], gWORK1);
    alpha.x=1.0f;alpha.y=0.0f;
    /* tempeta <= tempeta + tempb */
    cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, tempb, 1, tempeta, 1);
    ct=ct+T;
   }
 }
 cudakernel_fns_fscale(N, tempeta, iw);
 /* find -ve gradient */
 if (negate) {
  alpha.x=-1.0f;alpha.y=0.0f;
  cbstatus=cublasCscal(cbhandle,4*N,&alpha,tempeta,1);
 } 
 cudakernel_fns_proj(N, x, tempeta, eta, cbhandle);
}

/* Hessian, also projected to tangent space */
/* need 8N*BlocksPerGrid+ 8N*2 float storage */
static void
cudakernel_fns_fhess(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, char *bbh, float *iw, cublasHandle_t cbhandle, float *gWORK) {
 cuFloatComplex *tempeta,*tempb;
 unsigned long int moff=0;
 tempeta=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 tempb=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 float *gWORK1=&gWORK[moff];

 cuFloatComplex alpha;
 cublasStatus_t cbstatus;
 /* max size of M for one kernel call, to determine optimal blocks */
 int T=128*ThreadsPerBlock;
 if (M<T) {
  cudakernel_fns_fhessflat(ThreadsPerBlock, BlocksPerGrid, N, M, x, eta, tempeta, y, coh, bbh, gWORK1);
 } else {
   /* reset memory to zero */
   cudaMemset(tempeta, 0, sizeof(cuFloatComplex)*4*N);
   /* loop through M baselines */
   int L=(M+T-1)/T;
   int ct=0;
   int myT,ci;
   for (ci=0; ci<L; ci++) {
    if (ct+T<M) {
      myT=T;
    } else {
      myT=M-ct;
    }
    int B=(myT+ThreadsPerBlock-1)/ThreadsPerBlock;
    cudakernel_fns_fhessflat(ThreadsPerBlock, B, N, myT, x, eta, tempb, &y[ct*8], &coh[ct*8], &bbh[ct*2], gWORK1);
    alpha.x=1.0f;alpha.y=0.0f;
    /* tempeta <= tempeta + tempb */
    cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, tempb, 1, tempeta, 1);
    ct=ct+T;
   }
 }

 cudakernel_fns_fscale(N, tempeta, iw);
 cudakernel_fns_proj(N, x, tempeta, fhess, cbhandle);
}

/* Armijo step calculation,
  output teta: Armijo gradient 
  return value: 0 : cost reduced, 1: no cost reduction, so do not run again 
  mincost: minimum value of cost found, if possible
*/
/* need 8N*BlocksPerGrid+ 8N*2 float storage */
static int
armijostep(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *teta, float *y, float *coh, char *bbh, float *iw, float *mincost, cublasHandle_t cbhandle, float *gWORK) {
 float alphabar=10.0f;
 float beta=0.2f;
 float sigma=0.5f;
 cublasStatus_t cbstatus;
 /* temp storage, re-using global storage */ 
 cuFloatComplex *eta, *x_prop;
 unsigned long int moff=0;
 eta=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 x_prop=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 float *gWORK1=&gWORK[moff];

 //double fx=fns_f(x,y,gdata);
 float fx=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,x,y,coh,bbh,gWORK1);
 //fns_fgrad(x,eta,y,gdata,0);
 cudakernel_fns_fgrad(ThreadsPerBlock,BlocksPerGrid,N,M,x,eta,y,coh,bbh,iw,0,cbhandle, gWORK1);
#ifdef DEBUG
 float eta_nrm;
 cublasScnrm2(cbhandle,4*N,eta,1,&eta_nrm);
printf("||eta||=%f\n",eta_nrm);
 /* read back eta for checking */
 cuFloatComplex *etalocal;
 cudaHostAlloc((void **)&etalocal, sizeof(cuFloatComplex)*4*N,cudaHostAllocDefault);
 cudaMemcpy(etalocal, eta, 4*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
printf("Eta=[\n");
 int ci;
 for (ci=0; ci<2*N; ci++) {
  printf("%f %f %f %f\n",etalocal[ci].x,etalocal[ci].y,etalocal[ci+2*N].x,etalocal[ci+2*N].y);
 }
printf("]\n");
 cudaFree(etalocal);
#endif
 float beta0=beta;
 float minfx=fx; float minbeta=beta0;
 float lhs,rhs,metric;
 int m,nocostred=0;
 cuFloatComplex alpha;
 *mincost=fx;

 float metric0=cudakernel_fns_g(N,x,eta,eta,cbhandle);
 for (m=0; m<50; m++) {
   /* abeta=(beta0)*alphabar*eta; */
   //my_ccopy(4*dp->N,eta,1,teta,1);
   cbstatus=cublasCcopy(cbhandle,4*N,eta,1,teta,1);
   //my_cscal(4*dp->N,beta0*alphabar+0.0*_Complex_I,teta);
   alpha.x=beta0*alphabar;alpha.y=0.0f;
   cbstatus=cublasCscal(cbhandle,4*N,&alpha,teta,1);
   /* Rx=R(x,teta); */
   //fns_R(dp->N,x,teta,x_prop);
   cudakernel_fns_R(N,x,teta,x_prop,cbhandle);
   //lhs=fns_f(x_prop,y,gdata);
   lhs=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,x_prop,y,coh,bbh,gWORK1);
   if (lhs<minfx) {
     minfx=lhs;
     *mincost=minfx;
     minbeta=beta0;
   }
   //rhs=fx+sigma*fns_g(dp->N,x,eta,teta);
   //metric=cudakernel_fns_g(N,x,eta,teta,cbhandle);
   metric=beta0*alphabar*metric0;
   rhs=fx+sigma*metric;
#ifdef DEBUG
printf("m=%d lhs=%e rhs=%e rat=%e norm=%e\n",m,lhs,rhs,lhs/rhs,metric);
#endif
   if ((!isnan(lhs) && lhs<=rhs)) {
    minbeta=beta0;
    break;
   }
   beta0=beta0*beta;
 }

 /* if no further cost improvement is seen */
 if (lhs>fx) {
     nocostred=1;
 }

 //my_ccopy(4*dp->N,eta,1,teta,1);
 cbstatus=cublasCcopy(cbhandle,4*N,eta,1,teta,1);
 alpha.x=minbeta*alphabar; alpha.y=0.0f;
 //my_cscal(4*dp->N,minbeta*alphabar+0.0*_Complex_I,teta);
 cbstatus=cublasCscal(cbhandle,4*N,&alpha,teta,1);

 return nocostred;
}


/* truncated conjugate gradient method 
  x, grad, eta, r, z, delta, Hxd  : size 2N x 2  complex 
  so, vector size is 4N complex double

  output: eta
  return value: stop_tCG code   

  y: vec(V) visibilities
*/
/* need 8N*(BlocksPerGrid+2)+ 8N*6 float storage */
static int
tcg_solve_cuda(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *grad, cuFloatComplex *eta, cuFloatComplex *fhess, float Delta, float theta, float kappa, int max_inner, int min_inner, float *y, float *coh, char *bbh, float *iw, cublasHandle_t cbhandle, float *gWORK) { 
  cuFloatComplex *r,*z,*delta,*Hxd, *rnew;
  float  e_Pe, r_r, norm_r, z_r, d_Pd, d_Hd, alpha, e_Pe_new,
     e_Pd, Deltasq, tau, zold_rold, beta, norm_r0;
  int cj, stop_tCG;
  unsigned long int moff=0;
  r=(cuFloatComplex*)&gWORK[moff];
  moff+=8*N;
  z=(cuFloatComplex*)&gWORK[moff];
  moff+=8*N;
  delta=(cuFloatComplex*)&gWORK[moff];
  moff+=8*N;
  Hxd=(cuFloatComplex*)&gWORK[moff];
  moff+=8*N;
  rnew=(cuFloatComplex*)&gWORK[moff];
  moff+=8*N;
  float *gWORK1=&gWORK[moff];

  cublasStatus_t cbstatus;
  cuFloatComplex a0;

  /*
  initial values
  */
  cbstatus=cublasCcopy(cbhandle,4*N,grad,1,r,1);
  e_Pe=0.0f;


  r_r=cudakernel_fns_g(N,x,r,r,cbhandle);
  norm_r=sqrtf(r_r);
  norm_r0=norm_r;

  cbstatus=cublasCcopy(cbhandle,4*N,r,1,z,1);

  z_r=cudakernel_fns_g(N,x,z,r,cbhandle);
  d_Pd=z_r;

  /*
   initial search direction
  */
  cudaMemset(delta, 0, sizeof(cuFloatComplex)*4*N); 
  a0.x=-1.0f; a0.y=0.0f;
  cbstatus=cublasCaxpy(cbhandle,4*N, &a0, z, 1, delta, 1);
  e_Pd=cudakernel_fns_g(N,x,eta,delta,cbhandle);

  stop_tCG=5;

  /* % begin inner/tCG loop
    for j = 1:max_inner,
  */
  for(cj=1; cj<=max_inner; cj++) {
    cudakernel_fns_fhess(ThreadsPerBlock,BlocksPerGrid,N,M,x,delta,Hxd,y,coh,bbh,iw, cbhandle, gWORK1);
    d_Hd=cudakernel_fns_g(N,x,delta,Hxd,cbhandle);

    alpha=z_r/d_Hd;
    e_Pe_new = e_Pe + 2.0f*alpha*e_Pd + alpha*alpha*d_Pd;


    Deltasq=Delta*Delta;
    if (d_Hd <= 0.0f || e_Pe_new >= Deltasq) {
      tau = (-e_Pd + sqrtf(e_Pd*e_Pd + d_Pd*(Deltasq-e_Pe)))/d_Pd;
      a0.x=tau;
      cbstatus=cublasCaxpy(cbhandle,4*N, &a0, delta, 1, eta, 1);
      /* Heta = Heta + tau *Hdelta */
      cbstatus=cublasCaxpy(cbhandle,4*N, &a0, Hxd, 1, fhess, 1);
      stop_tCG=(d_Hd<=0.0f?1:2);
      break;
    }

    e_Pe=e_Pe_new;
    a0.x=alpha;
    cbstatus=cublasCaxpy(cbhandle,4*N, &a0, delta, 1, eta, 1);
    /* Heta = Heta + alpha*Hdelta */
    cbstatus=cublasCaxpy(cbhandle,4*N, &a0, Hxd, 1, fhess, 1);
    
    cbstatus=cublasCaxpy(cbhandle,4*N, &a0, Hxd, 1, r, 1);
    cudakernel_fns_proj(N, x, r, rnew, cbhandle);
    cbstatus=cublasCcopy(cbhandle,4*N,rnew,1,r,1);
    r_r=cudakernel_fns_g(N,x,r,r,cbhandle);
    norm_r=sqrtf(r_r);

    /*
      check kappa/theta stopping criterion
    */
    if (cj >= min_inner) {
      float norm_r0pow=powf(norm_r0,theta);
      if (norm_r <= norm_r0*MIN(norm_r0pow,kappa)) {
       stop_tCG=(kappa<norm_r0pow?3:4);
       break;
      }
    }

    cbstatus=cublasCcopy(cbhandle,4*N,r,1,z,1);
    zold_rold=z_r;

    z_r=cudakernel_fns_g(N,x,z,r,cbhandle);

    beta=z_r/zold_rold;
    a0.x=beta; 
    cbstatus=cublasCscal(cbhandle,4*N,&a0,delta,1);
    a0.x=-1.0f; 
    cbstatus=cublasCaxpy(cbhandle,4*N, &a0, z, 1, delta, 1);


    e_Pd = beta*(e_Pd + alpha*d_Pd);
    d_Pd = z_r + beta*beta*d_Pd;
  }

  return stop_tCG;
}


/* follow clmfit_fl.c */
int
rtr_solve_cuda_fl(
  float *x0,         /* initial values and updated solution at output (size 8*N float) */
  float *y,         /* data vector (size 8*M float) */
  int N,              /* no of stations */
  int M,              /* no of constraints */
  int itmax_sd,          /* maximum number of iterations RSD */
  int itmax_rtr,          /* maximum number of iterations RTR */
  float Delta_bar, float Delta0, /* Trust region radius and initial value */
  double *info, /* initial and final residuals */

  cublasHandle_t cbhandle, /* device handle */
  float *gWORK, /* GPU allocated memory */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
  me_data_t *adata)
{

  /* general note: all device variables end with a 'd' */
  cudaError_t err;
  cublasStatus_t cbstatus;

  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  float *cohd;
  /* baseline-station map on device/host */
  char *bbd;

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=128;
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;


  /* reshape x to make J: 2Nx2 complex double 
  */
  complex float *x;
  if ((x=(complex float*)malloc((size_t)4*N*sizeof(complex float)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  /* map x: [(re,im)J_1(0,0) (re,im)J_1(0,1) (re,im)J_1(1,0) (re,im)J_1(1,1)...]
   to
  J: [J_1(0,0) J_1(1,0) J_2(0,0) J_2(1,0) ..... J_1(0,1) J_1(1,1) J_2(0,1) J_2(1,1)....]
 */
  float *Jd=(float*)x;
  /* re J(0,0) */
  my_fcopy(N, &x0[0], 8, &Jd[0], 4);
  /* im J(0,0) */
  my_fcopy(N, &x0[1], 8, &Jd[1], 4);
  /* re J(1,0) */
  my_fcopy(N, &x0[4], 8, &Jd[2], 4);
  /* im J(1,0) */
  my_fcopy(N, &x0[5], 8, &Jd[3], 4);
  /* re J(0,1) */
  my_fcopy(N, &x0[2], 8, &Jd[4*N], 4);
  /* im J(0,1) */
  my_fcopy(N, &x0[3], 8, &Jd[4*N+1], 4);
  /* re J(1,1) */
  my_fcopy(N, &x0[6], 8, &Jd[4*N+2], 4);
  /* im J(1,1) */
  my_fcopy(N, &x0[7], 8, &Jd[4*N+3], 4);


  int ci;

/***************************************************/
 cuFloatComplex *xd,*fgradxd,*etad,*Hetad,*x_propd;
 float *yd;
 /* for counting how many baselines contribute to each station
   grad/hess calculation */
 float *iwd,*iw;
 if ((iw=(float*)malloc((size_t)N*sizeof(float)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }


 unsigned long int moff=0;
 fgradxd=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N; /* 4N complex means 8N float */
 etad=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 Hetad=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 x_propd=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;
 xd=(cuFloatComplex*)&gWORK[moff];
 moff+=8*N;

 yd=&gWORK[moff];
 moff+=8*M;
 cohd=&gWORK[moff];
 moff+=Nbase*8;
 bbd=(char*)&gWORK[moff];
 unsigned long int charstor=(Nbase*2*sizeof(char))/sizeof(float);
 if (!charstor || charstor%4) {
  moff+=(charstor/4+1)*4; /* NOTE +4 multiple to align memory */
 } else {
  moff+=charstor;
 }
 iwd=&gWORK[moff];
 if (!(N%4)) {
  moff+=N;
 } else {
  moff+=(N/4+1)*4;
 }
 /* need 8N*(BlocksPerGrid+8) for tcg_solve+grad/hess storage,
   so total storage needed is 
   8N*(BlocksPerGrid+8) + 8N*5 + 8*M + 8*Nbase + 2*Nbase + N
 */
 /* remaining memory */
 float *gWORK1=&gWORK[moff];

 /* yd <=y : V */
 err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* need to give right offset for coherencies */
 /* offset: cluster offset+time offset */
 /* C */
 err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* correct offset for baselines */
 err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(char), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* xd <=x : solution */
 err=cudaMemcpy(xd, x, 8*N*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);

 float fx,fx0,norm_grad,Delta,fx_prop,rhonum,rhoden,rho;

 /* count how many baselines contribute to each station, store (inverse) in iwd */
 count_baselines(Nbase,N,iw,&(dp->ddbase[2*(dp->Nbase)*(tileoff)]),dp->Nt);
 err=cudaMemcpy(iwd, iw, N*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 free(iw);

 fx=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,gWORK1);
 fx0=fx;
#ifdef DEBUG
printf("Initial Cost=%g\n",fx0);
#endif
/***************************************************/
 int rsdstat=0;
 /* RSD solution */
 for (ci=0; ci<itmax_sd; ci++) {
  /* Armijo step */
  /* teta=armijostep(V,C,N,x); */
  //armijostep(N,x,eta,y,&gdata);
  rsdstat=armijostep(ThreadsPerBlock, BlocksPerGrid, N, M, xd, etad, yd, cohd, bbd,iwd,&fx,cbhandle,gWORK1);
  /* x=R(x,teta); */
  cudakernel_fns_R(N,xd,etad,x_propd,cbhandle);
  //my_ccopy(4*N,x_propd,1,xd,1);
  if (!rsdstat) {
   /* cost reduced, update solution */
   cbstatus=cublasCcopy(cbhandle,4*N,x_propd,1,xd,1);
  } else {
   /* no cost reduction, break loop */
   break; 
  }
 }


 Delta_bar=MIN(fx,0.01f);
 Delta0=Delta_bar*0.125f;
//printf("fx=%g Delta_bar=%g Delta0=%g\n",fx,Delta_bar,Delta0);

#ifdef DEBUG
printf("NEW RSD cost=%g\n",fx);
#endif
/***************************************************/
   int min_inner,max_inner,min_outer,max_outer;
   float epsilon,kappa,theta,rho_prime;

   min_inner=1; max_inner=itmax_rtr;//8*N;
   min_outer=3;//itmax_rtr; //3; 
   max_outer=itmax_rtr;
   epsilon=(float)CLM_EPSILON;
   kappa=0.1f;
   theta=1.0f;
   /* default values 0.25, 0.75, 0.25, 2.0 */
   float eta1=0.0001f; float eta2=0.99f; float alpha1=0.25f; float alpha2=3.5f;
   rho_prime=eta1; /* should be <= 0.25, tune for parallel solve  */
   float rho_regularization; /* use large damping */
   rho_regularization=fx*1e-6f;
   /* damping: too small => locally converge, globally diverge
           |\
        |\ | \___
    -|\ | \|
       \
      
    
    right damping:  locally and globally converge
    -|\      
       \|\  
          \|\
             \____ 

    */
   float rho_reg;
   int model_decreased=0;

  /* RTR solution */
  int k=0;
  int stop_outer=(itmax_rtr>0?0:1);
  int stop_inner=0;
  if (!stop_outer) {
   cudakernel_fns_fgrad(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,1,cbhandle, gWORK1);
   norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle));
  }
  Delta=Delta0;
  /* initial residual */
  info[0]=fx0;

  /*
   % ** Start of TR loop **
  */
   while(!stop_outer) {
    /*  
     % update counter
    */
     k++;
    /* eta = 0*fgradx; */
    cudaMemset(etad, 0, sizeof(cuFloatComplex)*4*N);


    /* solve TR subproblem, also returns Hessian */
    stop_inner=tcg_solve_cuda(ThreadsPerBlock,BlocksPerGrid, N, M, xd, fgradxd, etad, Hetad, Delta, theta, kappa, max_inner, min_inner,yd,cohd,bbd,iwd,cbhandle, gWORK1);
    /*
        Heta = fns.fhess(x,eta);
    */
    /*
      compute the retraction of the proposal
    */
   cudakernel_fns_R(N,xd,etad,x_propd,cbhandle);

    /*
      compute cost of the proposal
    */
    fx_prop=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,x_propd,yd,cohd,bbd,gWORK1);

    /*
      check the performance of the quadratic model
    */
    rhonum=fx-fx_prop;
    rhoden=-cudakernel_fns_g(N,xd,fgradxd,etad,cbhandle)-0.5f*cudakernel_fns_g(N,xd,Hetad,etad,cbhandle);
    /* regularization of rho ratio */
    /* 
    rho_reg = max(1, abs(fx)) * eps * options.rho_regularization;
    rhonum = rhonum + rho_reg;
    rhoden = rhoden + rho_reg;
    */
    rho_reg=MAX(1.0f,fx)*rho_regularization; /* no epsilon */
    rhonum+=rho_reg;
    rhoden+=rho_reg;

     /*
        rho =   rhonum  / rhoden;
     */
     rho=rhonum/rhoden;

    /* model_decreased = (rhoden >= 0); */
   /* OLD CODE if (fabsf(rhonum/fx) <sqrtf_epsilon) {
     rho=1.0f;
    } */
    model_decreased=(rhoden>=0.0f?1:0);

#ifdef DEBUG
    printf("stop_inner=%d rho_reg=%g rho =%g/%g= %g rho'= %g\n",stop_inner,rho_reg,rhonum,rhoden,rho,rho_prime);
#endif
    /*
      choose new TR radius based on performance
    */
    if ( !model_decreased || rho<eta1 ) {
      Delta=alpha1*Delta;
    } else if (rho>eta2 && (stop_inner==2 || stop_inner==1)) {
      Delta=MIN(alpha2*Delta,Delta_bar);
    }

    /*
      choose new iterate based on performance
    */
    if (model_decreased && rho>rho_prime) {
     cbstatus=cublasCcopy(cbhandle,4*N,x_propd,1,xd,1);
     fx=fx_prop;
     cudakernel_fns_fgrad(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,1,cbhandle, gWORK1);
     norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle));
    }

    /*
     Testing for Stop Criteria
    */
    if (norm_grad<epsilon && k>min_outer) {
      stop_outer=1;
    }

    /*
     stop after max_outer iterations
     */
    if (k>=max_outer) {
      stop_outer=1;
    }

#ifdef DEBUG
printf("Iter %d cost=%g\n",k,fx);
#endif

   }
   /* final residual */
   info[1]=fx;
#ifdef DEBUG
printf("NEW RTR cost=%g\n",fx);
#endif

/***************************************************/
 cudaDeviceSynchronize();
  if(fx0>fx) {
  //printf("Cost final %g  initial %g\n",fx,fx0);
  /* copy back current solution */
  err=cudaMemcpy(x,xd,8*N*sizeof(float),cudaMemcpyDeviceToHost);
  checkCudaError(err,__FILE__,__LINE__);


  /* copy back solution to x0 : format checked*/
  /* re J(0,0) */
  my_fcopy(N, &Jd[0], 4, &x0[0], 8);
  /* im J(0,0) */
  my_fcopy(N, &Jd[1], 4, &x0[1], 8);
  /* re J(1,0) */
  my_fcopy(N, &Jd[2], 4, &x0[4], 8);
  /* im J(1,0) */
  my_fcopy(N, &Jd[3], 4, &x0[5], 8);
  /* re J(0,1) */
  my_fcopy(N, &Jd[4*N], 4, &x0[2], 8);
  /* im J(0,1) */
  my_fcopy(N, &Jd[4*N+1], 4, &x0[3], 8);
  /* re J(1,1) */
  my_fcopy(N, &Jd[4*N+2], 4, &x0[6], 8);
  /* im J(1,1) */
  my_fcopy(N, &Jd[4*N+3], 4, &x0[7], 8);

  }
  free(x);

  return 0;
}
