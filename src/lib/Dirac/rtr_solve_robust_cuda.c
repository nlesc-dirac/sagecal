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

#include "Dirac.h"
#include <cuda_runtime.h>

//#define DEBUG
/* helper functions for diagnostics */
static void
checkCudaError(cudaError_t err, char *file, int line)
{
#ifdef CUDA_DEBUG
    if(!err)
        return;
    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
#endif
}


static void
checkCublasError(cublasStatus_t cbstatus, char *file, int line)
{
#ifdef CUDA_DEBUG
   if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUBLAS failure\n",file,line);
    exit(EXIT_FAILURE);  
   }
#endif
}



/* gradient, also projected to tangent space */
/* for many time samples, gradient for each time sample is projected
   to tangent space before it is averaged 
 so calculate grad using N(N-1)/2 constraints each (total M)
*/
/* need 8N*M/ThreadsPerBlock+ 8N  complex float storage
  so actual float is 2 x this value */
static void
cudakernel_fns_fgrad_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *iw, float *wtd, int negate, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
 /* baselines per timeslot = N(N-1)/2 ~2400, timeslots = M/baselines ~120
    blocks per timeslot = baselines/ThreadsPerBlock ~2400/120=20
    so total blocks ~20x120=2400

    each block needs 8*N global storage 
 */
 cuFloatComplex *tempeta;
 cublasStatus_t cbstatus=CUBLAS_STATUS_SUCCESS;
 cuFloatComplex alpha;
 cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);

 /*************************/ 
 /* find A=I_2 kron (X^H X) + (X^H X)^T kron I_2
    and find inv(A) by solving A x B = I_4
    use temp storage
 */
 /* find X^H X */
 cuFloatComplex xx00,xx01,xx10,xx11;
 cbstatus=cublasCdotc(cbhandle,2*N,x,1,x,1,&xx00);
 cbstatus=cublasCdotc(cbhandle,2*N,x,1,&x[2*N],1,&xx01);
 xx10=cuConjf(xx01);
 cbstatus=cublasCdotc(cbhandle,2*N,&x[2*N],1,&x[2*N],1,&xx11);

 cuFloatComplex A[16],*Ad,B[16],*Bd;
 A[0]=cuCmulf(make_cuFloatComplex(2.0f,0.0f),xx00);
 A[5]=A[10]=cuCaddf(xx00,xx11);
 A[15]=cuCmulf(make_cuFloatComplex(2.0f,0.0f),xx11);
 A[1]=A[8]=A[11]=A[13]=xx10;
 A[2]=A[4]=A[7]=A[14]=xx01;
 A[3]=A[6]=A[9]=A[12]=make_cuFloatComplex(0.0f,0.0f);

 B[0]=B[5]=B[10]=B[15]=make_cuFloatComplex(1.0f,0.0f);
 B[1]=B[2]=B[3]=B[4]=B[6]=B[7]=B[8]=B[9]=B[11]=B[12]=B[13]=B[14]=make_cuFloatComplex(0.0f,0.0f);
 
#ifdef DEBUG
  printf("A=[\n");
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[0].x,A[0].y,A[4].x,A[4].y,A[8].x,A[8].y,A[12].x,A[12].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[1].x,A[1].y,A[5].x,A[5].y,A[9].x,A[9].y,A[13].x,A[13].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[2].x,A[2].y,A[6].x,A[6].y,A[10].x,A[10].y,A[14].x,A[14].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[3].x,A[3].y,A[7].x,A[7].y,A[11].x,A[11].y,A[15].x,A[15].y);
  printf("];\n");
#endif


 cudaMalloc((void **)&Ad, 16*sizeof(cuFloatComplex));
 cudaMalloc((void **)&Bd, 16*sizeof(cuFloatComplex));

 cudaMemcpy(Ad,A,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
 cudaMemcpy(Bd,B,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);

  int work_size=0;
  int *devInfo;
  cudaError_t err;
  err=cudaMalloc((void**)&devInfo, sizeof(int)); /* FIXME: get too many errors here */
  checkCudaError(err,__FILE__,__LINE__);
  cuFloatComplex *work,*taud;
  cusolverDnCgeqrf_bufferSize(solver_handle, 4, 4, (cuFloatComplex *)Ad, 4, &work_size);
  err=cudaMalloc((void**)&work, work_size*sizeof(cuFloatComplex));
  err=cudaMalloc((void**)&taud, 4*sizeof(cuFloatComplex));
  checkCudaError(err,__FILE__,__LINE__);
  cusolverDnCgeqrf(solver_handle, 4, 4, Ad, 4, taud, work, work_size, devInfo);
  cusolverDnCunmqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, 4, 4, 4, Ad, 4, taud, Bd, 4, work, work_size, devInfo);
  cuFloatComplex cone; cone.x=1.0f; cone.y=0.0f;
  cbstatus=cublasCtrsm(cbhandle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,4,4,&cone,Ad,4,Bd,4);


  cudaFree(work);
  cudaFree(taud);
  cudaFree(devInfo);


 cudaFree(Ad);

#ifdef DEBUG
 /* copy back the result */
 cudaMemcpy(B,Bd,16*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
  printf("B=[\n");
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[0].x,B[0].y,B[4].x,B[4].y,B[8].x,B[8].y,B[12].x,B[12].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[1].x,B[1].y,B[5].x,B[5].y,B[9].x,B[9].y,B[13].x,B[13].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[2].x,B[2].y,B[6].x,B[6].y,B[10].x,B[10].y,B[14].x,B[14].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[3].x,B[3].y,B[7].x,B[7].y,B[11].x,B[11].y,B[15].x,B[15].y);
  printf("];\n");
#endif


 /*************************/ 
 /* baselines */
 int nbase=N*(N-1)/2;
 /* timeslots */
 int ntime=(M+nbase-1)/nbase;
 /* blocks per timeslot */
 /* total blocks is Bt x ntime */
 int Bt=(nbase+ThreadsPerBlock-1)/ThreadsPerBlock;


#ifdef DEBUG
printf("N=%d Baselines=%d timeslots=%d total=%d,Threads=%d Blocks=%d\n",N,nbase,ntime,M,ThreadsPerBlock,Bt*ntime);
#endif
 
 /* max size of M for one kernel call, to determine optimal blocks */
 cudakernel_fns_fgradflat_robust(ThreadsPerBlock, Bt*ntime, N, M, x, tempeta, y, coh, bbh, wtd, Bd, cbhandle,solver_handle);
 /* weight for missing (flagged) baselines */
 cudakernel_fns_fscale(N, tempeta, iw);
 /* find -ve gradient */
 if (negate) {
  alpha.x=-1.0f;alpha.y=0.0f;
  cbstatus=cublasCscal(cbhandle,4*N,&alpha,tempeta,1);
 } 
 cudaMemcpy(eta,tempeta,4*N*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice);
 checkCublasError(cbstatus,__FILE__,__LINE__);
 cudaFree(Bd);
 cudaFree(tempeta);
}

/* Hessian, also projected to tangent space */
/* for many time samples, gradient for each time sample is projected
   to tangent space before it is averaged 
 so calculate grad using N(N-1)/2 constraints each (total M)
*/
/* need 8N*M/ThreadsPerBlock+ 8N float storage */
static void
cudakernel_fns_fhess_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *iw, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
 cuFloatComplex *tempeta;
 cublasStatus_t cbstatus=CUBLAS_STATUS_SUCCESS;
 cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);

 /*************************/
 /* find A=I_2 kron (X^H X) + (X^H X)^T kron I_2
    and find inv(A) by solving A x B = I_4
    use temp storage
 */
 /* find X^H X */
 cuFloatComplex xx00,xx01,xx10,xx11;
 cbstatus=cublasCdotc(cbhandle,2*N,x,1,x,1,&xx00);
 cbstatus=cublasCdotc(cbhandle,2*N,x,1,&x[2*N],1,&xx01);
 xx10=cuConjf(xx01);
 cbstatus=cublasCdotc(cbhandle,2*N,&x[2*N],1,&x[2*N],1,&xx11);

 cuFloatComplex A[16],*Ad,B[16],*Bd;
 A[0]=cuCmulf(make_cuFloatComplex(2.0f,0.0f),xx00);
 A[5]=A[10]=cuCaddf(xx00,xx11);
 A[15]=cuCmulf(make_cuFloatComplex(2.0f,0.0f),xx11);
 A[1]=A[8]=A[11]=A[13]=xx10;
 A[2]=A[4]=A[7]=A[14]=xx01;
 A[3]=A[6]=A[9]=A[12]=make_cuFloatComplex(0.0f,0.0f);

 B[0]=B[5]=B[10]=B[15]=make_cuFloatComplex(1.0f,0.0f);
 B[1]=B[2]=B[3]=B[4]=B[6]=B[7]=B[8]=B[9]=B[11]=B[12]=B[13]=B[14]=make_cuFloatComplex(0.0f,0.0f);

#ifdef DEBUG
  printf("A=[\n");
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[0].x,A[0].y,A[4].x,A[4].y,A[8].x,A[8].y,A[12].x,A[12].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[1].x,A[1].y,A[5].x,A[5].y,A[9].x,A[9].y,A[13].x,A[13].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[2].x,A[2].y,A[6].x,A[6].y,A[10].x,A[10].y,A[14].x,A[14].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",A[3].x,A[3].y,A[7].x,A[7].y,A[11].x,A[11].y,A[15].x,A[15].y);
  printf("];\n");
#endif


 cudaMalloc((void **)&Ad, 16*sizeof(cuFloatComplex));
 cudaMalloc((void **)&Bd, 16*sizeof(cuFloatComplex));

 cudaMemcpy(Ad,A,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
 cudaMemcpy(Bd,B,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
 //culaStatus status;
 //status=culaDeviceCgels('N',4,4,4,(culaDeviceFloatComplex *)Ad,4,(culaDeviceFloatComplex *)Bd,4);
 //checkStatus(status,__FILE__,__LINE__);
  int work_size=0;
  int *devInfo;
  cudaError_t err;
  err=cudaMalloc((void**)&devInfo, sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  cuFloatComplex *work,*taud;
  cusolverDnCgeqrf_bufferSize(solver_handle, 4, 4, (cuFloatComplex *)Ad, 4, &work_size);
  err=cudaMalloc((void**)&work, work_size*sizeof(cuFloatComplex));
  err=cudaMalloc((void**)&taud, 4*sizeof(cuFloatComplex));
  checkCudaError(err,__FILE__,__LINE__);
  cusolverDnCgeqrf(solver_handle, 4, 4, Ad, 4, taud, work, work_size, devInfo);
  cusolverDnCunmqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, 4, 4, 4, Ad, 4, taud, Bd, 4, work, work_size, devInfo);
  cuFloatComplex cone; cone.x=1.0f; cone.y=0.0f;
  cbstatus=cublasCtrsm(cbhandle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,4,4,&cone,Ad,4,Bd,4);


  cudaFree(work);
  cudaFree(taud);
  cudaFree(devInfo);
 cudaFree(Ad);

#ifdef DEBUG
 /* copy back the result */
 cudaMemcpy(B,Bd,16*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
  printf("B=[\n");
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[0].x,B[0].y,B[4].x,B[4].y,B[8].x,B[8].y,B[12].x,B[12].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[1].x,B[1].y,B[5].x,B[5].y,B[9].x,B[9].y,B[13].x,B[13].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[2].x,B[2].y,B[6].x,B[6].y,B[10].x,B[10].y,B[14].x,B[14].y);
  printf("%f+j*(%f) %f+j*(%f) %f+j*(%f) %f+j*(%f)\n",B[3].x,B[3].y,B[7].x,B[7].y,B[11].x,B[11].y,B[15].x,B[15].y);
  printf("];\n");
#endif
  /*************************/

 /* baselines */
 int nbase=N*(N-1)/2;
 /* timeslots */
 int ntime=(M+nbase-1)/nbase;
 /* blocks per timeslot */
 /* total blocks is Bt x ntime */
 int Bt=(nbase+ThreadsPerBlock-1)/ThreadsPerBlock;


#ifdef DEBUG
printf("N=%d Baselines=%d timeslots=%d total=%d,Threads=%d Blocks=%d\n",N,nbase,ntime,M,ThreadsPerBlock,Bt*ntime);
#endif


 cudakernel_fns_fhessflat_robust(ThreadsPerBlock, Bt*ntime, N, M, x, eta, tempeta, y, coh, bbh, wtd, Bd, cbhandle, solver_handle);

 cudakernel_fns_fscale(N, tempeta, iw);
 cudaMemcpy(fhess,tempeta,4*N*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice);
 checkCublasError(cbstatus,__FILE__,__LINE__);
 cudaFree(Bd);
 cudaFree(tempeta);
}


/* Fine tune initial trust region radius, also update initial value for x
   A. Sartenaer, 1995
   returns : trust region estimate,
   also modifies x
   eta,Heta: used as storage
 */
/* need 8N*2 + MAX(2 Blocks + 4, 8N (1 + ceil(M/Threads))) float storage */
static float
itrr(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta,  cuFloatComplex *Heta, float *y, float *coh, short *bbh, float *iw, float *wtd, cublasHandle_t cbhandle,cusolverDnHandle_t solver_handle) {
 cuFloatComplex alpha;
 cublasStatus_t cbstatus=CUBLAS_STATUS_SUCCESS;
 /* temp storage, re-using global storage */ 
 cuFloatComplex *s, *x_prop;
 cudaMalloc((void**)&s, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&x_prop, sizeof(cuFloatComplex)*4*N);

 float f0,fk,mk,rho,rho1,Delta0;
 /* initialize trust region radii */
 float delta_0=1.0f;
 float delta_m=0.0f; 

 float sigma=0.0f;
 float delta=0.0f;

 // initial cost
 f0=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,y,coh,bbh,wtd);
 // gradient at x0;
 cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,eta,y,coh,bbh,iw,wtd,1,cbhandle, solver_handle);
 // normalize
 float eta_nrm;
 cublasScnrm2(cbhandle,4*N,eta,1,&eta_nrm);
 alpha.x=1.0f/eta_nrm;alpha.y=0.0f;
 cbstatus=cublasCscal(cbhandle,4*N,&alpha,eta,1);

 cbstatus=cublasCcopy(cbhandle,4*N,eta,1,s,1);
 alpha.x=delta_0;alpha.y=0.0f;
 cbstatus=cublasCscal(cbhandle,4*N,&alpha,s,1);
 /* Hessian at s */
 cudakernel_fns_fhess_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,s,Heta,y,coh,bbh,iw,wtd,cbhandle,solver_handle);

 /* constants used */
 float gamma_1=0.0625f; float gamma_2=5.0f; float gamma_3=0.5f; float gamma_4=2.0f;
 float mu_0=0.5f; float mu_1=0.5f; float mu_2=0.35f;
 float teta=0.25f;


 int MK=4;
 int m;
 for (m=0; m<MK; m++) {
   /* x_prop=x0-s */
   cbstatus=cublasCcopy(cbhandle,4*N,x,1,x_prop,1);
   alpha.x=-1.0f;alpha.y=0.0f;
   cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, s, 1, x_prop, 1);

   /* model = f0 - g(x_prop,g0,s) - 0.5 g(x_prop,Hess,s) */
   mk=f0-cudakernel_fns_g(N,x_prop,eta,s,cbhandle,solver_handle)-0.5f*cudakernel_fns_g(N,x_prop,Heta,s,cbhandle,solver_handle);
   fk=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x_prop,y,coh,bbh,wtd);

   if (f0==mk) {
    rho=1e9f;
   } else {
    rho=(f0-fk)/(f0-mk);
   }
   rho1=fabsf(rho-1.0f);
   
   /* update max radius */
   if (rho1<mu_0) {
     delta_m=MAX(delta_m,delta_0);
   }
   if ((f0-fk)>delta) {
     delta=f0-fk;
     sigma=delta_0;
   }
   /* radius update */
   float beta_1,beta_2,beta_i=0.0f;
   beta_1=0.0f;
   beta_2=0.0f;
   
   if (m<MK) {
     float g0_s=cudakernel_fns_g(N,x,eta,s,cbhandle,solver_handle);
     float b1=(teta*(f0-g0_s)+(1.0f-teta)*mk-fk);
     beta_1=(b1==0.0f?1e9f:-teta*g0_s/b1); 
     
     float b2=(-teta*(f0-g0_s)+(1.0f+teta)*mk-fk);
     beta_2=(b2==0.0f?1e9f:teta*g0_s/b2); 
    
     float minbeta=MIN(beta_1,beta_2);
     float maxbeta=MAX(beta_1,beta_2);
     if (rho1>mu_1) {
       if (minbeta>1.0f) {
        beta_i=gamma_3;
       } else if ((maxbeta<gamma_1) || (minbeta<gamma_1 && maxbeta>=1.0f)) {
        beta_i=gamma_1;
       } else if ((beta_1>=gamma_1 && beta_1<1.0f) && (beta_2<gamma_1 || beta_2>=1.0f)) {
        beta_i=beta_1;
       } else if ((beta_2>=gamma_1 && beta_2<1.0f) && (beta_1<gamma_1 || beta_1>=1.0f)) {
        beta_i=beta_2;
      } else {
        beta_i=maxbeta;
      }
     } else if (rho1<=mu_2) {
       if (maxbeta<1.0f) {
         beta_i=gamma_4;
       } else if (maxbeta>gamma_2) {
         beta_i=gamma_2;
       } else if ((beta_1>=1.0f && beta_1<=gamma_2) && beta_2<1.0f) {
         beta_i=beta_1;
       } else if ((beta_2>=1.0f && beta_2<=gamma_2) && beta_1<1.0f) {
         beta_i=beta_2;
       } else {
         beta_i=maxbeta;
       }
     } else {
       if (maxbeta<gamma_3) {
         beta_i=gamma_3;
       } else if (maxbeta>gamma_4) {
         beta_i=gamma_4;
       } else {
         beta_i=maxbeta;
       }
     }
     /* update radius */
     delta_0=delta_0/beta_i;
   }
#ifdef DEBUG
printf("m=%d delta_0=%e delta_max=%e beta=%e rho=%e\n",m,delta_0,delta_m,beta_i,rho);
#endif

   cbstatus=cublasCcopy(cbhandle,4*N,eta,1,s,1);
   alpha.x=delta_0;alpha.y=0.0f;
   cbstatus=cublasCscal(cbhandle,4*N,&alpha,s,1);
 }

 // update initial value
 if (delta>0.0f) {
  alpha.x=-sigma; alpha.y=0.0f;
  cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, eta, 1, x, 1);
 }

 if (delta_m>0.0f) {
  Delta0=delta_m;
 } else {
  Delta0=delta_0;
 }

 checkCublasError(cbstatus,__FILE__,__LINE__);
 cudaFree(s);
 cudaFree(x_prop);
 return Delta0;
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
tcg_solve_cuda(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *grad, cuFloatComplex *eta, cuFloatComplex *fhess, float Delta, float theta, float kappa, int max_inner, int min_inner, float *y, float *coh, short *bbh, float *iw, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) { 
  cuFloatComplex *r,*z,*delta,*Hxd, *rnew;
  float  e_Pe, r_r, norm_r, z_r, d_Pd, d_Hd, alpha, e_Pe_new,
     e_Pd, Deltasq, tau, zold_rold, beta, norm_r0;
  int cj, stop_tCG;

  cudaMalloc((void**)&r, sizeof(cuFloatComplex)*4*N);
  cudaMalloc((void**)&z, sizeof(cuFloatComplex)*4*N);
  cudaMalloc((void**)&delta, sizeof(cuFloatComplex)*4*N);
  cudaMalloc((void**)&Hxd, sizeof(cuFloatComplex)*4*N);
  cudaMalloc((void**)&rnew, sizeof(cuFloatComplex)*4*N);

  cublasStatus_t cbstatus=CUBLAS_STATUS_SUCCESS;
  cuFloatComplex a0;

  /*
  initial values
  */
  cbstatus=cublasCcopy(cbhandle,4*N,grad,1,r,1);
  e_Pe=0.0f;


  r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
  norm_r=sqrtf(r_r);
  norm_r0=norm_r;

  cbstatus=cublasCcopy(cbhandle,4*N,r,1,z,1);

  z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);
  d_Pd=z_r;

  /*
   initial search direction
  */
  cudaMemset(delta, 0, sizeof(cuFloatComplex)*4*N); 
  a0.x=-1.0f; a0.y=0.0f;
  cbstatus=cublasCaxpy(cbhandle,4*N, &a0, z, 1, delta, 1);
  e_Pd=cudakernel_fns_g(N,x,eta,delta,cbhandle,solver_handle);

  stop_tCG=5;

  /* % begin inner/tCG loop
    for j = 1:max_inner,
  */
  for(cj=1; cj<=max_inner; cj++) {
    cudakernel_fns_fhess_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,delta,Hxd,y,coh,bbh,iw,wtd,cbhandle,solver_handle);
    d_Hd=cudakernel_fns_g(N,x,delta,Hxd,cbhandle,solver_handle);

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
    cudakernel_fns_proj(N, x, r, rnew, cbhandle, solver_handle);
    cbstatus=cublasCcopy(cbhandle,4*N,rnew,1,r,1);
    r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
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

    z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);

    beta=z_r/zold_rold;
    a0.x=beta; 
    cbstatus=cublasCscal(cbhandle,4*N,&a0,delta,1);
    a0.x=-1.0f; 
    cbstatus=cublasCaxpy(cbhandle,4*N, &a0, z, 1, delta, 1);


    e_Pd = beta*(e_Pd + alpha*d_Pd);
    d_Pd = z_r + beta*beta*d_Pd;
  }

  checkCublasError(cbstatus,__FILE__,__LINE__);
  cudaFree(r);
  cudaFree(z);
  cudaFree(delta);
  cudaFree(Hxd);
  cudaFree(rnew);

  return stop_tCG;
}



/* storage:
  8N * 5 + N + 8M * 2 + 2M + M (base storage)
  MAX( 2 * Blocks + 4, 8N(6 + ceil(M/Threads)))  for functions
  Blocks = ceil(M/Threads)
*/
int
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
  me_data_t *adata)
{

  /* general note: all device variables end with a 'd' */
  cudaError_t err;
  cublasStatus_t cbstatus=CUBLAS_STATUS_SUCCESS;

  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  float *cohd;
  /* baseline-station map on device/host */
  short *bbd;

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
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
 float *wtd,*qd; /* for robust weight and log(weight) */
 float robust_nu=(float)dp->robust_nu;
 float q_sum,robust_nu1;
 float deltanu;
 int Nd=100; /* no of points where nu is sampled, note Nd<N */
 if (Nd>M) { Nd=M; }
 deltanu=(float)(robust_nuhigh-robust_nulow)/(float)Nd;

 /* for counting how many baselines contribute to each station
   grad/hess calculation */
 float *iwd,*iw;
 if ((iw=(float*)malloc((size_t)N*sizeof(float)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }

 cudaMalloc((void**)&fgradxd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&etad, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&Hetad, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&x_propd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&xd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&yd, sizeof(float)*8*M);
 cudaMalloc((void**)&cohd, sizeof(float)*8*Nbase);
 cudaMalloc((void**)&bbd, sizeof(short)*2*Nbase);
 cudaMalloc((void**)&iwd, sizeof(float)*N);
 cudaMalloc((void**)&wtd, sizeof(float)*M);
 cudaMalloc((void**)&qd, sizeof(float)*M);


 /* need 8N*(BlocksPerGrid+8) for tcg_solve+grad/hess storage,
   so total storage needed is 
   8N*(BlocksPerGrid+8) + 8N*5 + 8*M + 8*Nbase + 2*Nbase + N + M + M
 */

 /* yd <=y : V */
 err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* need to give right offset for coherencies */
 /* offset: cluster offset+time offset */
 /* C */
 err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* correct offset for baselines */
 err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
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

 /* set initial weights to 1 by a cuda kernel */
 cudakernel_setweights_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, wtd, 1.0f);
 fx=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd);
 fx0=fx;
#ifdef DEBUG
printf("Initial Cost=%g\n",fx0);
#endif

 float Delta_new=itrr(ThreadsPerBlock, BlocksPerGrid, N, M, xd, etad, Hetad, yd, cohd, bbd, iwd, wtd, cbhandle,solver_handle);

#ifdef DEBUG
 printf("TR radius given=%f est=%f\n",Delta0,Delta_new);
#endif
 
 //old values
 //Delta_bar=MIN(fx,0.01f);
 //Delta0=Delta_bar*0.125f;
 Delta0=MIN(Delta_new,0.01f); /* need to be more restrictive for EM */
 Delta_bar=Delta0*8.0f;

 cudakernel_fns_fupdate_weights(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,robust_nu);
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
   cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
   norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
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
    stop_inner=tcg_solve_cuda(ThreadsPerBlock,BlocksPerGrid, N, M, xd, fgradxd, etad, Hetad, Delta, theta, kappa, max_inner, min_inner,yd,cohd,bbd,iwd,wtd,cbhandle,solver_handle);
    /*
        Heta = fns.fhess(x,eta);
    */
    /*
      compute the retraction of the proposal
    */
   cudakernel_fns_R(N,xd,etad,x_propd,cbhandle,solver_handle);

    /*
      compute cost of the proposal
    */
    fx_prop=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x_propd,yd,cohd,bbd,wtd);

    /*
      check the performance of the quadratic model
    */
    rhonum=fx-fx_prop;
    rhoden=-cudakernel_fns_g(N,xd,fgradxd,etad,cbhandle,solver_handle)-0.5f*cudakernel_fns_g(N,xd,Hetad,etad,cbhandle,solver_handle);
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
     cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
     norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
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
   /* w <= (p+nu)/(1+error^2), q<=w-log(w) */
   /* p = 2, use MAX() residual of XX,XY,YX,YY, not the sum */
   cudakernel_fns_fupdate_weights_q(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,qd,robust_nu);
   /* sumq<=sum(w-log(w))/N */
   cbstatus=cublasSasum(cbhandle, M, qd, 1, &q_sum);
   q_sum/=(float)M;
#ifdef DEBUG
   printf("deltanu=%f sum(w-log(w))=%f\n",deltanu,q_sum);
#endif
  /* for nu range 2~numax evaluate, p-variate T
     psi((nu0+p)/2)-ln((nu0+p)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 
     note: AECM not ECME
     and find min(| |) */
   int ThreadsPerBlock2=ThreadsPerBlock/4;
   cudakernel_evaluatenu_fl_eight(ThreadsPerBlock2, (Nd+ThreadsPerBlock-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow,robust_nu);
   /* find min(abs()) value */
   cbstatus=cublasIsamin(cbhandle, Nd, qd, 1, &ci); /* 1 based index */
   robust_nu1=(float)robust_nulow+(float)(ci-1)*deltanu;
#ifdef DEBUG
   printf("nu updated %d from %f [%lf,%lf] to %f\n",ci,robust_nu,robust_nulow,robust_nuhigh,robust_nu1);
#endif
   /* seems pedantic, but make sure new value for robust_nu fits within bounds */
   if (robust_nu1<robust_nulow) {
    dp->robust_nu=robust_nulow;
   } else if (robust_nu1>robust_nuhigh) {
    dp->robust_nu=robust_nuhigh;
   } else {
    dp->robust_nu=(double)robust_nu1;
   }
  
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

 checkCublasError(cbstatus,__FILE__,__LINE__);
 cudaFree(fgradxd);
 cudaFree(etad);
 cudaFree(Hetad);
 cudaFree(x_propd);
 cudaFree(xd);
 cudaFree(yd);
 cudaFree(cohd);
 cudaFree(bbd);
 cudaFree(iwd);
 cudaFree(wtd);
 cudaFree(qd);


  return 0;
}



/* storage:
  8N * 6 + N + 8M * 2 + 2M + M (base storage)
  MAX( 2 * Blocks + 4, 8N(1 + ceil(M/Threads)))  for functions
  Blocks = ceil(M/Threads)
*/
int
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
  me_data_t *adata)
{

  /* general note: all device variables end with a 'd' */
  cudaError_t err;
  cublasStatus_t cbstatus=CUBLAS_STATUS_SUCCESS;

  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  float *cohd;
  /* baseline-station map on device/host */
  short *bbd;

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
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
 cuFloatComplex *xd,*fgradxd,*etad,*zd,*x_propd,*z_propd;
 float *yd;
 float *wtd,*qd; /* for robust weight and log(weight) */
 float robust_nu=(float)dp->robust_nu;
 float q_sum,robust_nu1;
 float deltanu;
 int Nd=100; /* no of points where nu is sampled, note Nd<N */
 if (Nd>M) { Nd=M; }
 deltanu=(float)(robust_nuhigh-robust_nulow)/(float)Nd;

 /* for counting how many baselines contribute to each station
   grad/hess calculation */
 float *iwd,*iw;
 if ((iw=(float*)malloc((size_t)N*sizeof(float)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }

 cudaMalloc((void**)&fgradxd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&etad, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&zd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&x_propd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&xd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&z_propd, sizeof(cuFloatComplex)*4*N);
 cudaMalloc((void**)&yd, sizeof(float)*8*M);
 cudaMalloc((void**)&cohd, sizeof(float)*8*Nbase);
 cudaMalloc((void**)&bbd, sizeof(short)*2*Nbase);
 cudaMalloc((void**)&iwd, sizeof(float)*N);
 cudaMalloc((void**)&wtd, sizeof(float)*M);
 cudaMalloc((void**)&qd, sizeof(float)*M);


 /* yd <=y : V */
 err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* need to give right offset for coherencies */
 /* offset: cluster offset+time offset */
 /* C */
 err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* correct offset for baselines */
 err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 /* xd <=x : solution */
 err=cudaMemcpy(xd, x, 8*N*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);

 float fx,fx0;

 /* count how many baselines contribute to each station, store (inverse) in iwd */
 count_baselines(Nbase,N,iw,&(dp->ddbase[2*(dp->Nbase)*(tileoff)]),dp->Nt);
 err=cudaMemcpy(iwd, iw, N*sizeof(float), cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 free(iw);

 /* set initial weights to 1 by a cuda kernel */
 cudakernel_setweights_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, wtd, 1.0f);
 fx=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd);
 fx0=fx;
#ifdef DEBUG
printf("Initial Cost=%g\n",fx0);
#endif
/***************************************************/
  // gradient at x0;
  cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
  // Hessian 
  cudakernel_fns_fhess_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,xd,zd,yd,cohd,bbd,iwd,wtd,cbhandle,solver_handle);
  // initial step = 1/||Hess||
  float hess_nrm;
  cublasScnrm2(cbhandle,4*N,zd,1,&hess_nrm);
  float t=1.0f/hess_nrm;
  /* if initial step too small */
  if (t<1e-6f) {
   t=1e-6f;
  }
  
  /* z <= x */
  cbstatus=cublasCcopy(cbhandle,4*N,xd,1,zd,1);
  float theta=1.0f;
  float ALPHA = 1.01f; // step-size growth factor
  float BETA = 0.5f; // step-size shrinkage factor
  int k;
  cuFloatComplex alpha;

  for (k=0; k<itmax; k++) {
    /* x_prop <= x */
    cbstatus=cublasCcopy(cbhandle,4*N,xd,1,x_propd,1);
    /* z_prop <= z */
    cbstatus=cublasCcopy(cbhandle,4*N,zd,1,z_propd,1);
    
    /* x <= z - t * grad */
    cbstatus=cublasCcopy(cbhandle,4*N,zd,1,xd,1);
    alpha.x=-t;alpha.y=0.0f;
    cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, fgradxd, 1, xd, 1);

    /* if ||x-z|| == t||grad|| is below threshold, stop iteration */
    float grad_nrm,x_nrm;
    cublasScnrm2(cbhandle,4*N,fgradxd,1,&grad_nrm);
    cublasScnrm2(cbhandle,4*N,xd,1,&x_nrm);
    /* norm(y-x)/max(1,norm(x)); */
    if (grad_nrm*t/MAX(1.0f,x_nrm) < 1e-6f) {
      break;
    }


    /* theta = 2/(1 + sqrt(1+4/(theta^2))); */
    theta=2.0f/(1.0f + sqrtf(1.0f+4.0f/(theta*theta)));

    /* z = x + (1-theta)*(x-x_prop); 
       z = (2-theta)*x  - (1-theta) * x_prop */
    cbstatus=cublasCcopy(cbhandle,4*N,xd,1,zd,1);
    alpha.x=(2.0f-theta);alpha.y=0.0f;
    cbstatus=cublasCscal(cbhandle,4*N,&alpha,zd,1);
    alpha.x=-(1.0f-theta);alpha.y=0.0f;
    cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, x_propd, 1, zd, 1);

    /* eta = grad_old;
     grad  <= grad_f( z ) */
    cbstatus=cublasCcopy(cbhandle,4*N,fgradxd,1,etad,1);
    cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,zd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);

    /* z_prop <= z_prop - z */
    alpha.x=-1.0f;alpha.y=0.0f;
    cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, zd, 1, z_propd, 1);
    /* eta <= eta - new_grad */
    cbstatus=cublasCaxpy(cbhandle,4*N, &alpha, fgradxd, 1, etad, 1);
   
    /* ||z-z_prop|| */
    float ydiffnrm;
    cublasScnrm2(cbhandle,4*N,z_propd,1,&ydiffnrm);
    /* (z_zold)'*(grad-grad_old) */
    float dot_ydiff_gdiff;
    cbstatus=cublasSdot(cbhandle, 8*N, (float*)z_propd, 1, (float*)etad, 1, &dot_ydiff_gdiff);
#ifdef DEBUG
   printf("num=%e den=%e\n",ydiffnrm,dot_ydiff_gdiff);
#endif
    /* the above can be NAN, if so break loop */
    if (isnan(dot_ydiff_gdiff) || isinf(dot_ydiff_gdiff)) {
     break;
    }


    /* backtracking
     t_hat = 0.5*(norm(y-y_old)^2)/abs((y - y_old)'*(g_old - g));
     t = min( ALPHA*t, max( BETA*t, t_hat ));
    */
    float t_hat=0.5f*(ydiffnrm*ydiffnrm)/fabsf(dot_ydiff_gdiff);
    t=MIN(ALPHA*t,MAX(BETA*t,t_hat));
#ifdef DEBUG
printf("k=%d theta=%e step=%e\n",k,theta,t);
#endif
  }

  /* final residual */
  fx=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd);
  info[1]=fx;
#ifdef DEBUG
printf("NEW NSD cost=%g\n",fx);
#endif

/***************************************************/
 cudaDeviceSynchronize();
   /* w <= (p+nu)/(1+error^2), q<=w-log(w) */
   /* p = 2, use MAX() residual of XX,XY,YX,YY, not the sum */
   cudakernel_fns_fupdate_weights_q(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,qd,robust_nu);
   /* sumq<=sum(w-log(w))/N */
   cbstatus=cublasSasum(cbhandle, M, qd, 1, &q_sum);
   q_sum/=(float)M;
#ifdef DEBUG
   printf("deltanu=%f sum(w-log(w))=%f\n",deltanu,q_sum);
#endif
  /* for nu range 2~numax evaluate, p-variate T
     psi((nu0+p)/2)-ln((nu0+p)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 
     note: AECM not ECME
     and find min(| |) */
   int ThreadsPerBlock2=ThreadsPerBlock/4;
   cudakernel_evaluatenu_fl_eight(ThreadsPerBlock2, (Nd+ThreadsPerBlock-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow,robust_nu);
   /* find min(abs()) value */
   cbstatus=cublasIsamin(cbhandle, Nd, qd, 1, &ci); /* 1 based index */
   robust_nu1=(float)robust_nulow+(float)(ci-1)*deltanu;
#ifdef DEBUG
   printf("nu updated %d from %f [%lf,%lf] to %f\n",ci,robust_nu,robust_nulow,robust_nuhigh,robust_nu1);
#endif
   /* seems pedantic, but make sure new value for robust_nu fits within bounds */
   if (robust_nu1<robust_nulow) {
    dp->robust_nu=robust_nulow;
   } else if (robust_nu1>robust_nuhigh) {
    dp->robust_nu=robust_nuhigh;
   } else {
    dp->robust_nu=(double)robust_nu1;
   }
  
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
 checkCublasError(cbstatus,__FILE__,__LINE__); 
 cudaFree(fgradxd);
 cudaFree(etad);
 cudaFree(zd);
 cudaFree(x_propd);
 cudaFree(xd);
 cudaFree(z_propd);
 cudaFree(yd);
 cudaFree(cohd);
 cudaFree(bbd);
 cudaFree(iwd);
 cudaFree(wtd);
 cudaFree(qd);

  return 0;
}
