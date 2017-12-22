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

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
/* helper functions for diagnostics */
static void
checkCudaError(cudaError_t err, char *file, int line)
{
#ifdef CUDA_DEBUG
    if(!err)
        return;
    printf("GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
#endif
}


static void
checkCublasError(cublasStatus_t cbstatus, char *file, int line)
{
#ifdef CUDA_DEBUG
   if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    printf("%s: %d: CUBLAS failure\n",file,line);
    exit(EXIT_FAILURE);  
   }
#endif
}



/* robust, iteratively weighted non linear least squares using LM 
  entirely in the GPU */
int
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
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  cudaError_t err;
  cublasStatus_t cbstatus;

  int nu=2,nu2;
  double p_L2, Dp_L2=DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0, pDp_eL2, init_p_eL2;
  double tmp,mu=0.0;
  double tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  double *hxd;
  double *wtd,*qd;

  int nw,wt_itmax=3;
  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  double wt_sum,lambda,robust_nu=dp->robust_nu;
  double q_sum,robust_nu1;
  double deltanu;
  int Nd=100; /* no of points where nu is sampled, note Nd<N */
  if (Nd>N) { Nd=N; }
  /* only search for nu in [2,30] because 30 is almost Gaussian */
  deltanu=(robust_nuhigh-robust_nulow)/(double)Nd;
  

  double *ed;
  double *xd;

  double *jacd;

  double *jacTjacd,*jacTjacd0;

  double *Dpd,*bd;
  double *pd,*pnewd;
  double *jacTed;

  /* used in QR solver */
  double *taud;

  /* used in SVD solver */
  double *Ud;
  double *VTd;
  double *Sd;

  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  double *cohd;
  /* baseline-station map on device/host */
  short *bbd;

  int solve_axb=linsolv;
  double alpha;

  /* setup default settings */
  if(opts){
    tau=opts[0];
    eps1=opts[1];
    eps2=opts[2];
    eps2_sq=opts[2]*opts[2];
    eps3=opts[3];
  } else {
    tau=CLM_INIT_MU;
    eps1=CLM_STOP_THRESH;
    eps2=CLM_STOP_THRESH;
    eps2_sq=CLM_STOP_THRESH*CLM_STOP_THRESH;
    eps3=CLM_STOP_THRESH;
  }

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int ThreadsPerBlock1=DEFAULT_TH_PER_BK; /* DEFAULT_TH_PER_BK/8 for accessing each element of a baseline */
  int ThreadsPerBlock2=Nd/2; /* for evaluating nu */
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;


  unsigned long int moff;
  if (!gWORK) {
  err=cudaMalloc((void**)&xd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&jacd, M*N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&jacTjacd, M*M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&jacTed, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&jacTjacd0, M*M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&Dpd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&bd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&pd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&pnewd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  /* needed for calculating f()  and jac() */
  err=cudaMalloc((void**) &bbd, Nbase*2*sizeof(short));
  checkCudaError(err,__FILE__,__LINE__);
  /* we need coherencies for only this cluster */
  err=cudaMalloc((void**) &cohd, Nbase*8*sizeof(double)); 
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&hxd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&wtd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&qd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&ed, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  /* memory allocation: different solvers */
  if (solve_axb==1) {
    err=cudaMalloc((void**)&taud, M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  } else if (solve_axb==2) {
    err=cudaMalloc((void**)&Ud, M*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&VTd, M*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&Sd, M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }
  } else {
    moff=0;
    xd=&gWORK[moff];
    moff+=N;
    jacd=&gWORK[moff];
    moff+=M*N;
    jacTjacd=&gWORK[moff];
    moff+=M*M;
    jacTed=&gWORK[moff];
    moff+=M;
    jacTjacd0=&gWORK[moff];
    moff+=M*M;
    Dpd=&gWORK[moff];
    moff+=M;
    bd=&gWORK[moff];
    moff+=M;
    pd=&gWORK[moff];
    moff+=M;
    pnewd=&gWORK[moff];
    moff+=M;
    cohd=&gWORK[moff];
    moff+=Nbase*8;
    hxd=&gWORK[moff];
    moff+=N;
    ed=&gWORK[moff];
    moff+=N;
    wtd=&gWORK[moff];
    moff+=N;
    qd=&gWORK[moff];
    moff+=N;
    if (solve_axb==1) {
     taud=&gWORK[moff];
     moff+=M;
    } else if (solve_axb==2) {
     Ud=&gWORK[moff];
     moff+=M*M;
     VTd=&gWORK[moff];
     moff+=M*M;
     Sd=&gWORK[moff];
     moff+=M;
    }
    bbd=(short*)&gWORK[moff];
    moff+=(Nbase*2*sizeof(short))/sizeof(double);
  }

  /* extra storage for cusolver */
  int work_size=0;
  int *devInfo;
  int devInfo_h=0;
  err=cudaMalloc((void**)&devInfo, sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  double *work;
  double *rwork;
  if (solve_axb==0) {
    cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, M, jacTjacd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  } else if (solve_axb==1) {
    cusolverDnDgeqrf_bufferSize(solver_handle, M, M, jacTjacd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  } else {
    cusolverDnDgesvd_bufferSize(solver_handle, M, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }

  err=cudaMemcpy(pd, p, M*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* need to give right offset for coherencies */
  /* offset: cluster offset+time offset */
  err=cudaMemcpy(cohd, &(dp->ddcoh[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* correct offset for baselines */
  err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  cudaDeviceSynchronize();
  /* xd <=x */
  err=cudaMemcpy(xd, x, N*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* set initial weights to 1 by a cuda kernel */
  cudakernel_setweights(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, 1.0);
  /* weight calculation  loop */
  for (nw=0; nw<wt_itmax; nw++) {

  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  /* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
  cudakernel_func_wt(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

  /* e=x */
  cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
  /* e = e \odot wt */
  cudakernel_hadamard(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
  /* e=x-hx */
  alpha=-1.0;
  cbstatus=cublasDaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);

  /* norm ||e|| */
  cbstatus=cublasDnrm2(cbhandle, N, ed, 1, &p_eL2);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finite(p_eL2)) stop=7;


  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("em %d iter=%d err=%lf\n",nw,k,p_eL2);
#endif
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }

    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
    */
    /* p: params (Mx1), jacd: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
    /* FIXME thread/block sizes 16x16=256, so 16 is chosen */
     cudakernel_jacf_wt(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     //status=culaDeviceDgemm('N','T',M,M,N,1.0,jacd,M,jacd,M,0.0,jacTjacd,M);
     //checkStatus(status,__FILE__,__LINE__);
     double cone=1.0; double czero=0.0;
     cbstatus=cublasDgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_T,M,M,N,&cone,jacd,M,jacd,M,&czero,jacTjacd,M);

     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     cbstatus=cublasDcopy(cbhandle, M*M, jacTjacd, 1, jacTjacd0, 1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     //status=culaDeviceDgemv('N',M,N,1.0,jacd,M,ed,1,0.0,jacTed,1);
     //checkStatus(status,__FILE__,__LINE__);
     cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_N,M,N,&cone,jacd,M,ed,1,&czero,jacTed,1);


     /* Compute ||J^T e||_inf and ||p||^2 */
     /* find infinity norm of J^T e, 1 based indexing*/
     cbstatus=cublasIdamax(cbhandle, M, jacTed, 1, &ci);
     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     /* L2 norm of current parameter values */
     /* norm ||Dp|| */
     cbstatus=cublasDnrm2(cbhandle, M, pd, 1, &p_L2);
     p_L2=p_L2*p_L2;
     if(jacTe_inf<0.0) {jacTe_inf=-jacTe_inf;}
#ifdef DEBUG
     printf("Inf norm=%lf\n",jacTe_inf);
#endif
     
    /* check for convergence */
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0; /* no increment for p in this case */
      stop=1;
      break;
    }

    /* compute initial (k=0) damping factor */
    if (k==0) {
      /* find max diagonal element (stride is M+1) */
      /* should be MAX not MAX(ABS) */
      cbstatus=cublasIdamax(cbhandle, M, jacTjacd, M+1, &ci); /* 1 based index */
      ci=(ci-1)*(M+1); /* right value of the diagonal */

      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
      checkCudaError(err,__FILE__,__LINE__);
      mu=tau*tmp;
    }

    
    /* determine increment using adaptive damping */
    while(1){
      /* augment normal equations */
      /* increment A => A+ mu*I, increment diagonal entries */
      /* copy jacTjacd<=jacTjacd0 */
      cbstatus=cublasDcopy(cbhandle, M*M, jacTjacd0, 1, jacTjacd, 1);
      cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);

#ifdef DEBUG
      printf("mu=%lf\n",mu);
#endif
/*************************************************************************/
      issolved=0;
      /* solve augmented equations A x = b */
      /* A==jacTjacd, b==Dpd, after solving, x==Dpd */
      /* b=jacTed : intially right hand side, at exit the solution */
      if (solve_axb==0) {
        /* Cholesky solver **********************/
        /* lower triangle of Ad is destroyed */
        //status=culaDeviceDpotrf('U',M,jacTjacd,M);
        cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, M, jacTjacd, M, work, work_size, devInfo);
        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (!devInfo_h) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix\n");
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
#ifdef DEBUG
         checkCublasError(cbstatus,__FILE__,__LINE__);
#endif
         //status=culaDeviceDpotrs('U',M,1,jacTjacd,M,Dpd,M);
         cusolverDnDpotrs(solver_handle, CUBLAS_FILL_MODE_UPPER,M,1,jacTjacd,M,Dpd,M,devInfo);
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix\n");
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        //status=culaDeviceDgeqrf(M,M,jacTjacd,M,taud);
        cusolverDnDgeqrf(solver_handle, M, M, jacTjacd, M, taud, work, work_size, devInfo);
        cudaDeviceSynchronize();
        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (!devInfo_h) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix\n");
#endif
        }

        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
         //status=culaDeviceDgeqrs(M,M,1,jacTjacd,M,taud,Dpd,M);
         cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, 1, M, jacTjacd, M, taud, Dpd, M, work, work_size, devInfo);
         cudaDeviceSynchronize();
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix\n");
#endif
         } else {
          cone=1.0;
          cbstatus=cublasDtrsm(cbhandle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,M,1,&cone,jacTjacd,M,Dpd,M);
         }
        }
      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
        //status=culaDeviceDgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M);
        //checkStatus(status,__FILE__,__LINE__);
        cusolverDnDgesvd(solver_handle,'A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,work,work_size,rwork,devInfo);
        cudaDeviceSynchronize();
        /* copy Dpd<=jacTed */
        cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
        /* b<=U^T * b */
        //status=culaDeviceDgemv('T',M,M,1.0,Ud,M,Dpd,1,0.0,Dpd,1);
        cone=1.0; czero=0.0;
        cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,Ud,M,Dpd,1,&czero,Dpd,1);
        /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);

        /* b<=VT^T * b */
        //status=culaDeviceDgemv('T',M,M,1.0,VTd,M,Dpd,1,0.0,Dpd,1);
        //checkStatus(status,__FILE__,__LINE__);
        cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,VTd,M,Dpd,1,&czero,Dpd,1);

        issolved=1;
      }
/*************************************************************************/

      /* compute p's new estimate and ||Dp||^2 */
      if (issolved) {
          /* compute p's new estimate and ||Dp||^2 */
          /* pnew=p+Dp */
          /* pnew=p */
          cbstatus=cublasDcopy(cbhandle, M, pd, 1, pnewd, 1);
          /* pnew=pnew+Dp */
          alpha=1.0;
          cbstatus=cublasDaxpy(cbhandle, M, &alpha, Dpd, 1, pnewd, 1);

          /* norm ||Dp|| */
          cbstatus=cublasDnrm2(cbhandle, M, Dpd, 1, &Dp_L2);
          Dp_L2=Dp_L2*Dp_L2;

#ifdef DEBUG
printf("norm ||dp|| =%lf, norm ||p||=%lf\n",Dp_L2,p_L2);
#endif
          if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
           stop=2;
           break;
          }

         if(Dp_L2>=(p_L2+eps2)/(CLM_EPSILON*CLM_EPSILON)){ /* almost singular */
          stop=4;
          break;
         }

        /* new function value */
        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        /* copy to device */
        /* hxd<=hx */
        cudakernel_func_wt(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

        /* e=x */
        cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
        /* e = e \odot wt */
        cudakernel_hadamard(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
        /* e=x-hx */
        alpha=-1.0;
        cbstatus=cublasDaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
        /* note: e is updated */

        /* norm ||e|| */
        cbstatus=cublasDnrm2(cbhandle, N, ed, 1, &pDp_eL2);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finite(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
                                  */
          stop=7;
          break;
        }

        /* dL=Dp'*(mu*Dp+jacTe) */
        /* bd=jacTe+mu*Dp */
        cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, bd, 1);
        cbstatus=cublasDaxpy(cbhandle, M, &mu, Dpd, 1, bd, 1);
        cbstatus=cublasDdot(cbhandle, M, Dpd, 1, bd, 1, &dL);

        dF=p_eL2-pDp_eL2;

#ifdef DEBUG
        printf("dF=%lf, dL=%lf\n",dF,dL);
#endif
        if(dL>0.0 && dF>0.0){ /* reduction in error, increment is accepted */
          tmp=(2.0*dF/dL-1.0);
          tmp=1.0-tmp*tmp*tmp;
          mu=mu*((tmp>=CLM_ONE_THIRD)? tmp : CLM_ONE_THIRD);
          nu=2;

          /* update p's estimate */
          cbstatus=cublasDcopy(cbhandle, M, pnewd, 1, pd, 1);

          /* update ||e||_2 */
          p_eL2=pDp_eL2;
          break;
        }

      }
      /* if this point is reached, either the linear system could not be solved or
       * the error did not reduce; in any case, the increment must be rejected
       */

      mu*=(double)nu;
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). */
        stop=5;
        break;
      }

      nu=nu2;

    } /* inner loop */

  }
  /**** end iteration loop ***********/
  if(k>=itmax) stop=3;

  if (nw>0 && nw<wt_itmax-1) { 
   /* update error ed with no weights, only if not at first or last iteration */
   /* this is needed to update the weights */
   cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
   /* e=x */
   cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
   /* e=x-hx */
   alpha=-1.0;
   cbstatus=cublasDaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
  }

  
  if (nw<wt_itmax-1) { 
   cbstatus=cublasDasum(cbhandle, N, wtd, 1, &lambda); /* 1 based index */
   /* if not at last iteration update weights */
   /* Estimate robust_nu here DIGAMMA */
   /* w <= (nu+1)/(nu+delta^2) */
   /* q <= w-log(w), so all elements are +ve */
   cudakernel_updateweights(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed, qd, robust_nu);
   /* sumq<=sum(w-log(w))/N */ 
   cbstatus=cublasDasum(cbhandle, N, qd, 1, &q_sum); 
   q_sum/=(double)N;
#ifdef DEBUG
   printf("deltanu=%lf sum(w-log(w))=%lf\n",deltanu,q_sum);
#endif
   /* w <= sqrt(w) */
   cudakernel_sqrtweights(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd);
   /* for nu range 2~numax evaluate
     psi((nu+1)/2)-ln((nu+1)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 
     and find min(| |) */
   cudakernel_evaluatenu(ThreadsPerBlock2, (Nd+ThreadsPerBlock2-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,robust_nulow);
   /* find min(abs()) value */
   cbstatus=cublasIdamin(cbhandle, Nd, qd, 1, &ci); /* 1 based index */
   robust_nu1=robust_nulow+(double)(ci-1)*deltanu;
#ifdef DEBUG
   printf("nu updated %d from %lf [%lf,%lf] to %lf\n",ci,robust_nu,robust_nulow,robust_nuhigh,robust_nu1);
#endif
   robust_nu=robust_nu1;

   /* scale weights so sum =N */
   wt_sum =lambda/(double)N;
   cbstatus=cublasDscal(cbhandle, N, &wt_sum, wtd, 1); /* 1 based index */
   stop=0; /* restart LM */
  }

  } /* end of weight calc iterations */

  dp->robust_nu=robust_nu;

  /* copy back current solution */
  err=cudaMemcpyAsync(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost,0);
  checkCudaError(err,__FILE__,__LINE__);

  checkCublasError(cbstatus,__FILE__,__LINE__);

  /* synchronize async operations */
  cudaDeviceSynchronize();

  if (!gWORK) {
  cudaFree(xd);
  cudaFree(jacd);
  cudaFree(jacTjacd);
  cudaFree(jacTjacd0);
  cudaFree(jacTed);
  cudaFree(Dpd);
  cudaFree(bd);
  cudaFree(pd);
  cudaFree(pnewd);
  cudaFree(hxd);
  cudaFree(wtd);
  cudaFree(qd);
  cudaFree(ed);
  if (solve_axb==1) {
   cudaFree(taud);
  } else if (solve_axb==2) {
   cudaFree(Ud);
   cudaFree(VTd);
   cudaFree(Sd);
  }
  cudaFree(cohd);
  cudaFree(bbd);
  }

  cudaFree(devInfo);
  cudaFree(work);
  if (solve_axb==2) {
    cudaFree(rwork);
  }


#ifdef DEBUG
  printf("stop=%d\n",stop);
#endif
  if(info){
    info[0]=init_p_eL2;
    info[1]=p_eL2;
    info[2]=jacTe_inf;
    info[3]=Dp_L2;
    info[4]=mu;
    info[5]=(double)k;
    info[6]=(double)stop;
    info[7]=(double)0;
    info[8]=(double)0;
    info[9]=(double)0;
  }
  return 0;
}


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
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  cudaError_t err;
  cublasStatus_t cbstatus;

  int nu=2,nu2;
  float p_L2, Dp_L2=(float)DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0f, pDp_eL2, init_p_eL2;
  float tmp,mu=0.0f;
  float tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  float *hxd;
  float *wtd,*qd;

  int nw,wt_itmax=3;
  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  float wt_sum,lambda,robust_nu=(float)dp->robust_nu;
  float q_sum,robust_nu1;
  float deltanu;
  int Nd=100; /* no of points where nu is sampled, note Nd<N */
  if (Nd>N) { Nd=N; }
  /* only search for nu in [2,30] because 30 is almost Gaussian */
  deltanu=(float)(robust_nuhigh-robust_nulow)/(float)Nd;
  

  float *ed;
  float *xd;

  float *jacd;

  float *jacTjacd,*jacTjacd0;

  float *Dpd,*bd;
  float *pd,*pnewd;
  float *jacTed;

  /* used in QR solver */
  float *taud=0;

  /* used in SVD solver */
  float *Ud=0;
  float *VTd=0;
  float *Sd=0;

  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  float *cohd;
  /* baseline-station map on device/host */
  short *bbd;

  int solve_axb=linsolv;
  float alpha;

  /* setup default settings */
  if(opts){
    tau=(float)opts[0];
    eps1=(float)opts[1];
    eps2=(float)opts[2];
    eps2_sq=(float)opts[2]*opts[2];
    eps3=(float)opts[3];
  } else {
    tau=(float)CLM_INIT_MU;
    eps1=(float)CLM_STOP_THRESH;
    eps2=(float)CLM_STOP_THRESH;
    eps2_sq=(float)CLM_STOP_THRESH*CLM_STOP_THRESH;
    eps3=(float)CLM_STOP_THRESH;
  }

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* FIXME: might need a large value for large no of baselines */
  int ThreadsPerBlock1=DEFAULT_TH_PER_BK; /* for accessing each element of a baseline */
  int ThreadsPerBlock2=Nd/2; /* for evaluating nu */
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;


  unsigned long int moff;
    moff=0;
    xd=&gWORK[moff];
    moff+=N;
    jacd=&gWORK[moff];
    moff+=M*N;
    jacTjacd=&gWORK[moff];
    moff+=M*M;
    jacTed=&gWORK[moff];
    moff+=M;
    jacTjacd0=&gWORK[moff];
    moff+=M*M;
    Dpd=&gWORK[moff];
    moff+=M;
    bd=&gWORK[moff];
    moff+=M;
    pd=&gWORK[moff];
    moff+=M;
    pnewd=&gWORK[moff];
    moff+=M;
    cohd=&gWORK[moff];
    moff+=Nbase*8;
    hxd=&gWORK[moff];
    moff+=N;
    ed=&gWORK[moff];
    moff+=N;
    wtd=&gWORK[moff];
    moff+=N;
    qd=&gWORK[moff];
    moff+=N;
    if (solve_axb==1) {
     taud=&gWORK[moff];
     moff+=M;
    } else if (solve_axb==2) {
     Ud=&gWORK[moff];
     moff+=M*M;
     VTd=&gWORK[moff];
     moff+=M*M;
     Sd=&gWORK[moff];
     moff+=M;
    }
    bbd=(short*)&gWORK[moff];
    moff+=(Nbase*2*sizeof(short))/sizeof(float);

  /* extra storage for cusolver */
  int work_size=0;
  int *devInfo;
  int devInfo_h=0;
  err=cudaMalloc((void**)&devInfo, sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  float *work;
  float *rwork;
  if (solve_axb==0) {
    cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, M, jacTjacd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
  } else if (solve_axb==1) {
    cusolverDnSgeqrf_bufferSize(solver_handle, M, M, jacTjacd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
  } else {
    cusolverDnSgesvd_bufferSize(solver_handle, M, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&rwork, 5*M*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
  }


  err=cudaMemcpyAsync(pd, p, M*sizeof(float), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  /* need to give right offset for coherencies */
  /* offset: cluster offset+time offset */
  err=cudaMemcpyAsync(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  /* correct offset for baselines */
  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  cudaDeviceSynchronize();
  /* xd <=x */
  err=cudaMemcpyAsync(xd, x, N*sizeof(float), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);

  /* set initial weights to 1 by a cuda kernel */
  cudakernel_setweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, 1.0f);
  /* weight calculation  loop */
  for (nw=0; nw<wt_itmax; nw++) {

  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  /* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
  cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

  /* e=x */
  cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
  /* e = e \odot wt */
  cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
  /* e=x-hx */
  alpha=-1.0f;
  cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);

  /* norm ||e|| */
  cbstatus=cublasSnrm2(cbhandle, N, ed, 1, &p_eL2);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finitef(p_eL2)) stop=7;


  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("em %d iter=%d err=%f\n",nw,k,p_eL2);
#endif
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }

    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
    */
    /* p: params (Mx1), jacd: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
    /* FIXME thread/block sizes 16x16=256, so 16 is chosen */
     cudakernel_jacf_wt_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     //status=culaDeviceSgemm('N','T',M,M,N,1.0f,jacd,M,jacd,M,0.0f,jacTjacd,M);
     //checkStatus(status,__FILE__,__LINE__);
     float cone=1.0f; float czero=0.0f;
     cbstatus=cublasSgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_T,M,M,N,&cone,jacd,M,jacd,M,&czero,jacTjacd,M);

     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     cbstatus=cublasScopy(cbhandle, M*M, jacTjacd, 1, jacTjacd0, 1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     //status=culaDeviceSgemv('N',M,N,1.0f,jacd,M,ed,1,0.0f,jacTed,1);
     //checkStatus(status,__FILE__,__LINE__);
     cbstatus=cublasSgemv(cbhandle,CUBLAS_OP_N,M,N,&cone,jacd,M,ed,1,&czero,jacTed,1);


     /* Compute ||J^T e||_inf and ||p||^2 */
     /* find infinity norm of J^T e, 1 based indexing*/
     cbstatus=cublasIsamax(cbhandle, M, jacTed, 1, &ci);
     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(float),cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     /* L2 norm of current parameter values */
     /* norm ||Dp|| */
     cbstatus=cublasSnrm2(cbhandle, M, pd, 1, &p_L2);
     p_L2=p_L2*p_L2;
     if(jacTe_inf<0.0f) {jacTe_inf=-jacTe_inf;}
#ifdef DEBUG
     printf("Inf norm=%f\n",jacTe_inf);
#endif
     
    /* check for convergence */
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0f; /* no increment for p in this case */
      stop=1;
      break;
    }

    /* compute initial (k=0) damping factor */
    if (k==0) {
      /* find max diagonal element (stride is M+1) */
      /* should be MAX not MAX(ABS) */
      cbstatus=cublasIsamax(cbhandle, M, jacTjacd, M+1, &ci); /* 1 based index */
      ci=(ci-1)*(M+1); /* right value of the diagonal */

      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(float),cudaMemcpyDeviceToHost);
      checkCudaError(err,__FILE__,__LINE__);
      mu=tau*tmp;
    }

    
    /* determine increment using adaptive damping */
    while(1){
      /* augment normal equations */
      /* increment A => A+ mu*I, increment diagonal entries */
      /* copy jacTjacd<=jacTjacd0 */
      cbstatus=cublasScopy(cbhandle, M*M, jacTjacd0, 1, jacTjacd, 1);
      cudakernel_diagmu_fl(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);

#ifdef DEBUG
      printf("mu=%lf\n",mu);
#endif
/*************************************************************************/
      issolved=0;
      /* solve augmented equations A x = b */
      /* A==jacTjacd, b==Dpd, after solving, x==Dpd */
      /* b=jacTed : intially right hand side, at exit the solution */
      if (solve_axb==0) {
        /* Cholesky solver **********************/
        /* lower triangle of Ad is destroyed */
        //status=culaDeviceSpotrf('U',M,jacTjacd,M);
        cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, M, jacTjacd, M, work, work_size, devInfo);
        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (!devInfo_h) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix\n");
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasScopy(cbhandle, M, jacTed, 1, Dpd, 1);
#ifdef DEBUG
         checkCublasError(cbstatus,__FILE__,__LINE__);
#endif
         //status=culaDeviceSpotrs('U',M,1,jacTjacd,M,Dpd,M);
         cusolverDnSpotrs(solver_handle, CUBLAS_FILL_MODE_UPPER,M,1,jacTjacd,M,Dpd,M,devInfo);
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix\n");
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        //status=culaDeviceSgeqrf(M,M,jacTjacd,M,taud);
        cusolverDnSgeqrf(solver_handle, M, M, jacTjacd, M, taud, work, work_size, devInfo);
        cudaDeviceSynchronize();
        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (!devInfo_h) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix\n");
#endif
        }

        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasScopy(cbhandle, M, jacTed, 1, Dpd, 1);
         //status=culaDeviceSgeqrs(M,M,1,jacTjacd,M,taud,Dpd,M);
         cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, 1, M, jacTjacd, M, taud, Dpd, M, work, work_size, devInfo);
         cudaDeviceSynchronize();
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix\n");
#endif
         } else {
          cone=1.0f;
          cbstatus=cublasStrsm(cbhandle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,M,1,&cone,jacTjacd,M,Dpd,M);
         }
        }
      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
        //status=culaDeviceSgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M);
        //checkStatus(status,__FILE__,__LINE__);
        cusolverDnSgesvd(solver_handle,'A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,work,work_size,rwork,devInfo);
        cudaDeviceSynchronize();
        /* copy Dpd<=jacTed */
        cbstatus=cublasScopy(cbhandle, M, jacTed, 1, Dpd, 1);
        /* b<=U^T * b */
        //status=culaDeviceSgemv('T',M,M,1.0f,Ud,M,Dpd,1,0.0f,Dpd,1);
        //checkStatus(status,__FILE__,__LINE__);
        cone=1.0f; czero=0.0f;
        cbstatus=cublasSgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,Ud,M,Dpd,1,&czero,Dpd,1);
        /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        cudakernel_diagdiv_fl(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);

        /* b<=VT^T * b */
        //status=culaDeviceSgemv('T',M,M,1.0f,VTd,M,Dpd,1,0.0f,Dpd,1);
        //checkStatus(status,__FILE__,__LINE__);
        cbstatus=cublasSgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,VTd,M,Dpd,1,&czero,Dpd,1);

        issolved=1;
      }
/*************************************************************************/

      /* compute p's new estimate and ||Dp||^2 */
      if (issolved) {
          /* compute p's new estimate and ||Dp||^2 */
          /* pnew=p+Dp */
          /* pnew=p */
          cbstatus=cublasScopy(cbhandle, M, pd, 1, pnewd, 1);
          /* pnew=pnew+Dp */
          alpha=1.0f;
          cbstatus=cublasSaxpy(cbhandle, M, &alpha, Dpd, 1, pnewd, 1);

          /* norm ||Dp|| */
          cbstatus=cublasSnrm2(cbhandle, M, Dpd, 1, &Dp_L2);
          Dp_L2=Dp_L2*Dp_L2;

#ifdef DEBUG
printf("norm ||dp|| =%f, norm ||p||=%f\n",Dp_L2,p_L2);
#endif
          if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
           stop=2;
           break;
          }

         if(Dp_L2>=(p_L2+eps2)/(float)(CLM_EPSILON*CLM_EPSILON)){ /* almost singular */
          stop=4;
          break;
         }

        /* new function value */
        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        /* copy to device */
        /* hxd<=hx */
        cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

        /* e=x */
        cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
        /* e = e \odot wt */
        cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
        /* e=x-hx */
        alpha=-1.0f;
        cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
        /* note: e is updated */

        /* norm ||e|| */
        cbstatus=cublasSnrm2(cbhandle, N, ed, 1, &pDp_eL2);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finite(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
                                  */
          stop=7;
          break;
        }

        /* dL=Dp'*(mu*Dp+jacTe) */
        /* bd=jacTe+mu*Dp */
        cbstatus=cublasScopy(cbhandle, M, jacTed, 1, bd, 1);
        cbstatus=cublasSaxpy(cbhandle, M, &mu, Dpd, 1, bd, 1);
        cbstatus=cublasSdot(cbhandle, M, Dpd, 1, bd, 1, &dL);

        dF=p_eL2-pDp_eL2;

#ifdef DEBUG
        printf("dF=%f, dL=%f\n",dF,dL);
#endif
        if(dL>0.0f && dF>0.0f){ /* reduction in error, increment is accepted */
          tmp=(2.0f*dF/dL-1.0f);
          tmp=1.0f-tmp*tmp*tmp;
          mu=mu*((tmp>=(float)CLM_ONE_THIRD)? tmp : (float)CLM_ONE_THIRD);
          nu=2;

          /* update p's estimate */
          cbstatus=cublasScopy(cbhandle, M, pnewd, 1, pd, 1);

          /* update ||e||_2 */
          p_eL2=pDp_eL2;
          break;
        }

      }
      /* if this point is reached, either the linear system could not be solved or
       * the error did not reduce; in any case, the increment must be rejected
       */

      mu*=(float)nu;
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). */
        stop=5;
        break;
      }

      nu=nu2;

    } /* inner loop */

  }
  /**** end iteration loop ***********/
  if(k>=itmax) stop=3;

  if (nw>0 && nw<wt_itmax-1) { 
   /* update error ed with no weights, only if not at first or last iteration */
   /* this is needed to update the weights */
   cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
   /* e=x */
   cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
   /* e=x-hx */
   alpha=-1.0f;
   cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
  }

  
  if (nw<wt_itmax-1) { 
   /* find sum of w */
   cbstatus=cublasSasum(cbhandle, N, wtd, 1, &lambda); /* 1 based index */

   /* if not at last iteration update weights */
   /* Estimate robust_nu here DIGAMMA */
   /* w <= (nu+1)/(nu+delta^2) */
   /* q <= w-log(w), so all elements are +ve */
   cudakernel_updateweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed, qd, robust_nu);
   /* sumq<=sum(w-log(w))/N */ 
   cbstatus=cublasSasum(cbhandle, N, qd, 1, &q_sum); 
   q_sum/=(float)N;
#ifdef DEBUG
   printf("deltanu=%f sum(w-log(w))=%f\n",deltanu,q_sum);
#endif
   /* w <= sqrt(w) */
   cudakernel_sqrtweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd);
   /* for nu range 2~numax evaluate
     psi((nu+1)/2)-ln((nu+1)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 
     and find min(| |) */
   cudakernel_evaluatenu_fl(ThreadsPerBlock2, (Nd+ThreadsPerBlock2-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow);
   /* find min(abs()) value */
   cbstatus=cublasIsamin(cbhandle, Nd, qd, 1, &ci); /* 1 based index */
   robust_nu1=(float)robust_nulow+(float)(ci-1)*deltanu;
#ifdef DEBUG
   printf("nu updated %d from %f [%lf,%lf] to %f\n",ci,robust_nu,robust_nulow,robust_nuhigh,robust_nu1);
#endif
   robust_nu=robust_nu1;

   /* scale weights with wt_sum/N */
   wt_sum =lambda/(float)N;
   cbstatus=cublasSscal(cbhandle, N, &wt_sum, wtd, 1); /* 1 based index */
   stop=0; /* restart LM */
  }

  } /* end of weight calc iterations */

  dp->robust_nu=(double)robust_nu;

  /* copy back current solution */
  err=cudaMemcpyAsync(p,pd,M*sizeof(float),cudaMemcpyDeviceToHost,0);
  checkCudaError(err,__FILE__,__LINE__);

  checkCublasError(cbstatus,__FILE__,__LINE__);
  /* synchronize async operations */
  cudaDeviceSynchronize();

  cudaFree(devInfo);
  cudaFree(work);
  if (solve_axb==2) {
    cudaFree(rwork);
  }

#ifdef DEBUG
  printf("stop=%d\n",stop);
#endif
  if(info){
    info[0]=(double)init_p_eL2;
    info[1]=(double)p_eL2;
    info[2]=(double)jacTe_inf;
    info[3]=(double)Dp_L2;
    info[4]=(double)mu;
    info[5]=(double)k;
    info[6]=(double)stop;
    info[7]=(double)0;
    info[8]=(double)0;
    info[9]=(double)0;
  }
  return 0;
}


/* robust, iteratively weighted non linear least squares using LM 
  entirely in the GPU, using float data, OS acceleration */
int
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
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  cudaError_t err;
  cublasStatus_t cbstatus;

  int nu=2,nu2;
  float p_L2, Dp_L2=(float)DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0f, pDp_eL2, init_p_eL2;
  float tmp,mu=0.0f;
  float tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  float *hxd;
  float *wtd,*qd;

  int nw,wt_itmax=3;
  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  float wt_sum,lambda,robust_nu=(float)dp->robust_nu;
  float q_sum,robust_nu1;
  float deltanu;
  int Nd=100; /* no of points where nu is sampled, note Nd<N */
  if (Nd>N) { Nd=N; }
  /* only search for nu in [2,30] because 30 is almost Gaussian */
  deltanu=(float)(robust_nuhigh-robust_nulow)/(float)Nd;
  

  float *ed;
  float *xd;

  float *jacd;

  float *jacTjacd,*jacTjacd0;

  float *Dpd,*bd;
  float *pd,*pnewd;
  float *jacTed;

  /* used in QR solver */
  float *taud=0;

  /* used in SVD solver */
  float *Ud=0;
  float *VTd=0;
  float *Sd=0;

  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  float *cohd;
  /* baseline-station map on device/host */
  short *bbd;

  int solve_axb=linsolv;
  float alpha;

  /* setup default settings */
  if(opts){
    tau=(float)opts[0];
    eps1=(float)opts[1];
    eps2=(float)opts[2];
    eps2_sq=(float)opts[2]*opts[2];
    eps3=(float)opts[3];
  } else {
    tau=(float)CLM_INIT_MU;
    eps1=(float)CLM_STOP_THRESH;
    eps2=(float)CLM_STOP_THRESH;
    eps2_sq=(float)CLM_STOP_THRESH*CLM_STOP_THRESH;
    eps3=(float)CLM_STOP_THRESH;
  }

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* FIXME: might need a large value for large no of baselines */
  int ThreadsPerBlock1=DEFAULT_TH_PER_BK; /* for accessing each element of a baseline */
  int ThreadsPerBlock2=Nd/2; /* for evaluating nu */
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;


  unsigned long int moff;
    moff=0;
    xd=&gWORK[moff];
    moff+=N;
    jacd=&gWORK[moff];
    moff+=M*N;
    jacTjacd=&gWORK[moff];
    moff+=M*M;
    jacTed=&gWORK[moff];
    moff+=M;
    jacTjacd0=&gWORK[moff];
    moff+=M*M;
    Dpd=&gWORK[moff];
    moff+=M;
    bd=&gWORK[moff];
    moff+=M;
    pd=&gWORK[moff];
    moff+=M;
    pnewd=&gWORK[moff];
    moff+=M;
    cohd=&gWORK[moff];
    moff+=Nbase*8;
    hxd=&gWORK[moff];
    moff+=N;
    ed=&gWORK[moff];
    moff+=N;
    wtd=&gWORK[moff];
    moff+=N;
    qd=&gWORK[moff];
    moff+=N;
    if (solve_axb==1) {
     taud=&gWORK[moff];
     moff+=M;
    } else if (solve_axb==2) {
     Ud=&gWORK[moff];
     moff+=M*M;
     VTd=&gWORK[moff];
     moff+=M*M;
     Sd=&gWORK[moff];
     moff+=M;
    }
    bbd=(short*)&gWORK[moff];
    moff+=(Nbase*2*sizeof(short))/sizeof(float);

  /* extra storage for cusolver */
  int work_size=0;
  int *devInfo;
  int devInfo_h=0;
  err=cudaMalloc((void**)&devInfo, sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  float *work;
  float *rwork;
  if (solve_axb==0) {
    cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, M, jacTjacd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
  } else if (solve_axb==1) {
    cusolverDnSgeqrf_bufferSize(solver_handle, M, M, jacTjacd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
  } else {
    cusolverDnSgesvd_bufferSize(solver_handle, M, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&rwork, 5*M*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);
  }


  err=cudaMemcpyAsync(pd, p, M*sizeof(float), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  /* need to give right offset for coherencies */
  /* offset: cluster offset+time offset */
  err=cudaMemcpyAsync(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  /* correct offset for baselines */
  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  cudaDeviceSynchronize();
  /* xd <=x */
  err=cudaMemcpyAsync(xd, x, N*sizeof(float), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);

  /* set initial weights to 1 by a cuda kernel */
  cudakernel_setweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, 1.0f);

  /* setup OS subsets and stating offsets */
  /* ed : N, cohd : Nbase*8, bbd : Nbase*2 full size */
  /* if ntiles<Nsubsets, make Nsubsets=ntiles */
  int Nsubsets=10;
  if (ntiles<Nsubsets) { Nsubsets=ntiles; }
  /* FIXME: is 0.1 enough */
  int max_os_iter=(int)ceil(0.1*(double)Nsubsets);
  int Npersubset=(N+Nsubsets-1)/Nsubsets;
  int Nbasepersubset=(Nbase+Nsubsets-1)/Nsubsets;
  int *Nos,*Nbaseos,*edI,*NbI,*subI=0;
  if ((Nos=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((Nbaseos=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((edI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((NbI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  int l,ositer;
  k=l=0;
  for (ci=0; ci<Nsubsets; ci++) {
    edI[ci]=k;
    NbI[ci]=l;
    if (k+Npersubset<N) {
      Nos[ci]=Npersubset;
      Nbaseos[ci]=Nbasepersubset;
    } else {
      Nos[ci]=N-k;
      Nbaseos[ci]=Nbase-l;
    }
    k=k+Npersubset;
    l=l+Nbasepersubset;
  }

/*******************************************************************/
  /* weight calculation  loop */
  for (nw=0; nw<wt_itmax; nw++) {

  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  /* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
  cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

  /* e=x */
  cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
  /* e = e \odot wt */
  cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
  /* e=x-hx */
  alpha=-1.0f;
  cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);

  /* norm ||e|| */
  cbstatus=cublasSnrm2(cbhandle, N, ed, 1, &p_eL2);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finitef(p_eL2)) stop=7;



  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("em %d iter=%d err=%f\n",nw,k,p_eL2);
#endif
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }

    if (randomize) {
     /* random permutation of subsets */
     subI=random_permutation(Nsubsets,0,0);
    }
/**************** OS loop ***************************/
    for (ositer=0; ositer<max_os_iter; ositer++) {
     /* select subset to compute Jacobian */
     if (randomize) {
      l=subI[ositer];
     } else {
      l=(nw+k+ositer)%Nsubsets;
     }


    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
    */
    /* p: params (Mx1), jacd: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
    /* FIXME thread/block sizes 16x16=256, so 16 is chosen */
     cudakernel_jacf_wt_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, Nos[l], &cohd[8*NbI[l]], &bbd[2*NbI[l]], &wtd[edI[l]], Nbaseos[l], dp->M, dp->N);

     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     //status=culaDeviceSgemm('N','T',M,M,Nos[l],1.0f,jacd,M,jacd,M,0.0f,jacTjacd,M);
     //checkStatus(status,__FILE__,__LINE__);
     float cone=1.0f; float czero=0.0f;
     cbstatus=cublasSgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_T,M,M,Nos[l],&cone,jacd,M,jacd,M,&czero,jacTjacd,M);

     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     cbstatus=cublasScopy(cbhandle, M*M, jacTjacd, 1, jacTjacd0, 1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     //status=culaDeviceSgemv('N',M,Nos[l],1.0f,jacd,M,&ed[edI[l]],1,0.0f,jacTed,1);
     //checkStatus(status,__FILE__,__LINE__);
     cbstatus=cublasSgemv(cbhandle,CUBLAS_OP_N,M,Nos[l],&cone,jacd,M,&ed[edI[l]],1,&czero,jacTed,1);


     /* Compute ||J^T e||_inf and ||p||^2 */
     /* find infinity norm of J^T e, 1 based indexing*/
     cbstatus=cublasIsamax(cbhandle, M, jacTed, 1, &ci);
     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(float),cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     /* L2 norm of current parameter values */
     /* norm ||Dp|| */
     cbstatus=cublasSnrm2(cbhandle, M, pd, 1, &p_L2);
     p_L2=p_L2*p_L2;
     if(jacTe_inf<0.0f) {jacTe_inf=-jacTe_inf;}
#ifdef DEBUG
     printf("Inf norm=%f\n",jacTe_inf);
#endif
     
    /* check for convergence */
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0f; /* no increment for p in this case */
      stop=1;
      break;
    }

    /* compute initial (k=0) damping factor */
    if (k==0) {
      /* find max diagonal element (stride is M+1) */
      /* should be MAX not MAX(ABS) */
      cbstatus=cublasIsamax(cbhandle, M, jacTjacd, M+1, &ci); /* 1 based index */
      ci=(ci-1)*(M+1); /* right value of the diagonal */

      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(float),cudaMemcpyDeviceToHost);
      checkCudaError(err,__FILE__,__LINE__);
      mu=tau*tmp;
    }

    
    /* determine increment using adaptive damping */
    while(1){
      /* augment normal equations */
      /* increment A => A+ mu*I, increment diagonal entries */
      /* copy jacTjacd<=jacTjacd0 */
      cbstatus=cublasScopy(cbhandle, M*M, jacTjacd0, 1, jacTjacd, 1);
      cudakernel_diagmu_fl(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);

#ifdef DEBUG
      printf("mu=%lf\n",mu);
#endif
/*************************************************************************/
      issolved=0;
      /* solve augmented equations A x = b */
      /* A==jacTjacd, b==Dpd, after solving, x==Dpd */
      /* b=jacTed : intially right hand side, at exit the solution */
      if (solve_axb==0) {
        /* Cholesky solver **********************/
        /* lower triangle of Ad is destroyed */
        //status=culaDeviceSpotrf('U',M,jacTjacd,M);
        cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, M, jacTjacd, M, work, work_size, devInfo);
        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (!devInfo_h) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix\n");
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasScopy(cbhandle, M, jacTed, 1, Dpd, 1);
#ifdef DEBUG
         checkCublasError(cbstatus,__FILE__,__LINE__);
#endif
         //status=culaDeviceSpotrs('U',M,1,jacTjacd,M,Dpd,M);
         cusolverDnSpotrs(solver_handle, CUBLAS_FILL_MODE_UPPER,M,1,jacTjacd,M,Dpd,M,devInfo);
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix\n");
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        //status=culaDeviceSgeqrf(M,M,jacTjacd,M,taud);
        cusolverDnSgeqrf(solver_handle, M, M, jacTjacd, M, taud, work, work_size, devInfo);
        cudaDeviceSynchronize();
        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (!devInfo_h) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix\n");
#endif
        }

        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasScopy(cbhandle, M, jacTed, 1, Dpd, 1);
         //status=culaDeviceSgeqrs(M,M,1,jacTjacd,M,taud,Dpd,M);
         cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, 1, M, jacTjacd, M, taud, Dpd, M, work, work_size, devInfo);
         cudaDeviceSynchronize();
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix\n");
#endif
         } else {
          cone=1.0f;
          cbstatus=cublasStrsm(cbhandle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,M,1,&cone,jacTjacd,M,Dpd,M);
         }
        }
      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
        //status=culaDeviceSgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M);
        //checkStatus(status,__FILE__,__LINE__);
        cusolverDnSgesvd(solver_handle,'A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,work,work_size,rwork,devInfo);
        cudaDeviceSynchronize();
        /* copy Dpd<=jacTed */
        cbstatus=cublasScopy(cbhandle, M, jacTed, 1, Dpd, 1);
        /* b<=U^T * b */
        //status=culaDeviceSgemv('T',M,M,1.0f,Ud,M,Dpd,1,0.0f,Dpd,1);
        //checkStatus(status,__FILE__,__LINE__);
        cone=1.0f; czero=0.0f;
        cbstatus=cublasSgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,Ud,M,Dpd,1,&czero,Dpd,1);

        /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        cudakernel_diagdiv_fl(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);

        /* b<=VT^T * b */
        //status=culaDeviceSgemv('T',M,M,1.0f,VTd,M,Dpd,1,0.0f,Dpd,1);
        //checkStatus(status,__FILE__,__LINE__);
        cbstatus=cublasSgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,VTd,M,Dpd,1,&czero,Dpd,1);

        issolved=1;
      }
/*************************************************************************/

      /* compute p's new estimate and ||Dp||^2 */
      if (issolved) {
          /* compute p's new estimate and ||Dp||^2 */
          /* pnew=p+Dp */
          /* pnew=p */
          cbstatus=cublasScopy(cbhandle, M, pd, 1, pnewd, 1);
          /* pnew=pnew+Dp */
          alpha=1.0f;
          cbstatus=cublasSaxpy(cbhandle, M, &alpha, Dpd, 1, pnewd, 1);

          /* norm ||Dp|| */
          cbstatus=cublasSnrm2(cbhandle, M, Dpd, 1, &Dp_L2);
          Dp_L2=Dp_L2*Dp_L2;

#ifdef DEBUG
printf("norm ||dp|| =%f, norm ||p||=%f\n",Dp_L2,p_L2);
#endif
          if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
           stop=2;
           break;
          }

         if(Dp_L2>=(p_L2+eps2)/(float)(CLM_EPSILON*CLM_EPSILON)){ /* almost singular */
          stop=4;
          break;
         }

        /* new function value */
        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        /* copy to device */
        /* hxd<=hx */
        cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);

        /* e=x */
        cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
        /* e = e \odot wt */
        cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
        /* e=x-hx */
        alpha=-1.0f;
        cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
        /* note: e is updated */

        /* norm ||e|| */
        cbstatus=cublasSnrm2(cbhandle, N, ed, 1, &pDp_eL2);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finite(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
                                  */
          stop=7;
          break;
        }

        /* dL=Dp'*(mu*Dp+jacTe) */
        /* bd=jacTe+mu*Dp */
        cbstatus=cublasScopy(cbhandle, M, jacTed, 1, bd, 1);
        cbstatus=cublasSaxpy(cbhandle, M, &mu, Dpd, 1, bd, 1);
        cbstatus=cublasSdot(cbhandle, M, Dpd, 1, bd, 1, &dL);

        dF=p_eL2-pDp_eL2;

#ifdef DEBUG
        printf("dF=%f, dL=%f\n",dF,dL);
#endif
        if(dL>0.0f && dF>0.0f){ /* reduction in error, increment is accepted */
          tmp=(2.0f*dF/dL-1.0f);
          tmp=1.0f-tmp*tmp*tmp;
          mu=mu*((tmp>=(float)CLM_ONE_THIRD)? tmp : (float)CLM_ONE_THIRD);
          nu=2;

          /* update p's estimate */
          cbstatus=cublasScopy(cbhandle, M, pnewd, 1, pd, 1);

          /* update ||e||_2 */
          p_eL2=pDp_eL2;
          break;
        }

      }
      /* if this point is reached, either the linear system could not be solved or
       * the error did not reduce; in any case, the increment must be rejected
       */

      mu*=(float)nu;
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). */
        stop=5;
        break;
      }

      nu=nu2;

    } /* inner loop */

  }
  if (randomize) {
   free(subI);
  }
/**************** end OS loop ***************************/

  }
  /**** end iteration loop ***********/
  if(k>=itmax) stop=3;

  if (nw>0 && nw<wt_itmax-1) { 
   /* update error ed with no weights, only if not at first or last iteration */
   /* this is needed to update the weights */
   cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
   /* e=x */
   cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
   /* e=x-hx */
   alpha=-1.0f;
   cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
  }

  
  if (nw<wt_itmax-1) { 
   /* find sum of w */
   cbstatus=cublasSasum(cbhandle, N, wtd, 1, &lambda); /* 1 based index */
   /* if not at last iteration update weights */
   /* Estimate robust_nu here DIGAMMA */
   /* w <= (nu+1)/(nu+delta^2) */
   /* q <= w-log(w), so all elements are +ve */
   cudakernel_updateweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed, qd, robust_nu);
   /* sumq<=sum(w-log(w))/N */ 
   cbstatus=cublasSasum(cbhandle, N, qd, 1, &q_sum); 
   q_sum/=(float)N;
#ifdef DEBUG
   printf("deltanu=%f sum(w-log(w))=%f\n",deltanu,q_sum);
#endif
   /* w <= sqrt(w) */
   cudakernel_sqrtweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd);
   /* for nu range 2~numax evaluate
     psi((nu+1)/2)-ln((nu+1)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 
     and find min(| |) */
   cudakernel_evaluatenu_fl(ThreadsPerBlock2, (Nd+ThreadsPerBlock2-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow);
   /* find min(abs()) value */
   cbstatus=cublasIsamin(cbhandle, Nd, qd, 1, &ci); /* 1 based index */
   robust_nu1=(float)robust_nulow+(float)(ci-1)*deltanu;
#ifdef DEBUG
   printf("nu updated %d from %f [%lf,%lf] to %f\n",ci,robust_nu,robust_nulow,robust_nuhigh,robust_nu1);
#endif
   robust_nu=robust_nu1;

   /* scale weights by lambda/N */
   wt_sum =lambda/(float)N;
   cbstatus=cublasSscal(cbhandle, N, &wt_sum, wtd, 1); /* 1 based index */
   stop=0; /* restart LM */
  }

  } /* end of weight calc iterations */

  dp->robust_nu=(double)robust_nu;

  free(Nos);
  free(Nbaseos);
  free(edI);
  free(NbI);

  /* copy back current solution */
  err=cudaMemcpyAsync(p,pd,M*sizeof(float),cudaMemcpyDeviceToHost,0);
  checkCudaError(err,__FILE__,__LINE__);
  checkCublasError(cbstatus,__FILE__,__LINE__);

  /* synchronize async operations */
  cudaDeviceSynchronize();

  cudaFree(devInfo);
  cudaFree(work);
  if (solve_axb==2) {
    cudaFree(rwork);
  }

#ifdef DEBUG
  printf("stop=%d\n",stop);
#endif
  if(info){
    info[0]=(double)init_p_eL2;
    info[1]=(double)p_eL2;
    info[2]=(double)jacTe_inf;
    info[3]=(double)Dp_L2;
    info[4]=(double)mu;
    info[5]=(double)k;
    info[6]=(double)stop;
    info[7]=(double)0;
    info[8]=(double)0;
    info[9]=(double)0;
  }
  return 0;
}
#endif /* HAVE_CUDA */

#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
odot_threadfn(void *data) {
 thread_data_vec_t *t=(thread_data_vec_t*)data;
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   t->ed[ci]*=t->wtd[ci];
 }
 return NULL;
}


/* Hadamard product */
/* ed <= ed*wtd , size Nx1
  Nt threads */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
my_odot(double *ed,double *wtd,int N,int Nt) {
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_vec_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(N+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   printf("%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_vec_t*)malloc((size_t)Nt*sizeof(thread_data_vec_t)))==0) {
#ifndef USE_MIC
    printf("%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<N; nth++) {
    if (ci+Nthb0<N) {
     Nthb=Nthb0;
    } else {
     Nthb=N-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].ed=ed;
    threaddata[nth].wtd=wtd;
    pthread_create(&th_array[nth],&attr,odot_threadfn,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

 pthread_attr_destroy(&attr);

 free(th_array);
 free(threaddata);

 return 0;

}

/* robust LM */
int
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
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  int nu=2,nu2;
  double p_L2, Dp_L2=DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0, pDp_eL2, init_p_eL2;
  double tmp,mu=0.0;
  double tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  double *hxd,*hxm=0;
  double *ed,*wtd;
  double *jac;

  double *jacTjacd,*jacTjacd0;

  double *pnew,*Dpd,*bd;
  double *aones;
  double *jacTed;

  /* used in QR solver */
  double *WORK;
  int lwork=0;
  double w[1];

  int status;

  /* used in SVD solver */
  double *Ud;
  double *VTd;
  double *Sd;

  /* for Jacobian evaluation */
  int jac_given;
  double delta,tempp,ddiff;
  if (!jacf) {
   jac_given=0;
   /* need more memory for jacobian calculation */
   if ((hxm=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
   }
   delta=CLM_DIFF_DELTA;
  } else {
   jac_given=1;
  }

  int solve_axb=linsolv;

  /* setup default settings */
  if(opts){
    tau=opts[0];
    eps1=opts[1];
    eps2=opts[2];
    eps2_sq=opts[2]*opts[2];
    eps3=opts[3];
  } else {
    tau=CLM_INIT_MU;
    eps1=CLM_STOP_THRESH;
    eps2=CLM_STOP_THRESH;
    eps2_sq=CLM_STOP_THRESH*CLM_STOP_THRESH;
    eps3=CLM_STOP_THRESH;
  }

  if ((hxd=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((ed=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jac=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd0=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTed=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Dpd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((bd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((pnew=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((aones=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((wtd=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  WORK=Ud=Sd=VTd=0;
  int nw,wt_itmax=3;
  me_data_t *lmdata=(me_data_t*)adata;
  double wt_sum,lambda,robust_nu=lmdata->robust_nu;
  double robust_nu1;

  setweights(M,aones,1.0,lmdata->Nt);
  /*W set initial weights to 1 */
  setweights(N,wtd,1.0,lmdata->Nt);
  /* memory allocation: different solvers */
  if (solve_axb==0) {

  } else if (solve_axb==1) {
    /* workspace query */
    status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  } else {
    if ((Ud=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((VTd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((Sd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    
    status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  }


  /* EM iteration loop */
  /************************************************************/
  for (nw=0; nw<wt_itmax; nw++) {
  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  (*func)(p, hxd, M, N, adata);

  /* e=x */
  my_dcopy(N, x, 1, ed, 1);
  /* e=x-hx */
  my_daxpy(N, hxd, -1.0, ed);

  /*W e<= wt\odot e */
  my_odot(ed,wtd,N,Nt);

  /* norm ||e|| */
  p_eL2=my_dnrm2(N, ed);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finite(p_eL2)) stop=7;


  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("iter=%d err=%lf\n",k,p_eL2);
#endif
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }

    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
    */
    if (jac_given) {
     (*jacf)(p, jac, M, N, adata);
    } else {
      /* estimate jacobian using central differences */
       for (ci=0;ci<M; ci++) {
        /* Jacobian in row major order, so jac[ci],jac[ci+M],jac[ci+2*M]...N values are modified */
        /* modify ci-th parameter */
        tempp=p[ci];
        ddiff=fabs(p[ci]*(1e-4));
        if (ddiff<delta) {
          ddiff=delta;
        }
        p[ci]+=ddiff;
        /* f(p+delta) */
        (*func)(p, hxd, M, N, adata);
        p[ci]=tempp-ddiff;
        /* f(p-delta) */
        (*func)(p, hxm, M, N, adata);
        p[ci]=tempp;
        ddiff=0.5/ddiff;
        /* hxd=hxd-hxm */
        my_daxpy(N, hxm, -1.0, hxd);
        /* hxd=hxd/delta */
        my_dscal(N, ddiff, hxd);

        my_dcopy(N, hxd, 1, &jac[ci], M);
       }
    }

     /*W J<= wt\odot J, each row mult by wt[] */
     /* jac[0..M-1] <- wtd[0],  jac[M...2M-1] <- wtd[1] ... */
     for (ci=0; ci<N; ci++) {
      my_dscal(M, wtd[ci], &jac[ci*M]);
     }

     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     my_dgemm('N','T',M,M,N,1.0,jac,M,jac,M,0.0,jacTjacd,M);
     
     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     my_dcopy(M*M,jacTjacd,1,jacTjacd0,1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     my_dgemv('N',M,N,1.0,jac,M,ed,1,0.0,jacTed,1);


     /* Compute ||J^T e||_inf and ||p||^2 */
     /* find infinity norm of J^T e, 1 based indexing*/
     ci=my_idamax(M,jacTed,1);
     memcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double));
     /* L2 norm of current parameter values */
     /* norm ||Dp|| */
     p_L2=my_dnrm2(M,p);
     p_L2=p_L2*p_L2;
     if(jacTe_inf<0.0) {jacTe_inf=-jacTe_inf;}
#ifdef DEBUG
     printf("Inf norm=%lf\n",jacTe_inf);
#endif
     
    /* check for convergence */
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0; /* no increment for p in this case */
      stop=1;
      break;
    }

    /* compute initial (k=0) damping factor */
    if (k==0) {
      /* find max diagonal element (stride is M+1) */
      /* should be MAX not MAX(ABS) */
      ci=my_idamax(M,jacTjacd,M+1);
      ci=(ci-1)*(M+1); /* right value of the diagonal */

      memcpy(&tmp,&(jacTjacd[ci]),sizeof(double));
      mu=tau*tmp;
    }

    
    /* determine increment using adaptive damping */
    while(1){
      /* augment normal equations */
      /* increment A => A+ mu*I, increment diagonal entries */
      /* copy jacTjacd<=jacTjacd0 */
      memcpy(jacTjacd,jacTjacd0,M*M*sizeof(double));
      my_daxpys(M,aones,1,mu,jacTjacd,M+1);

#ifdef DEBUG
      printf("mu=%lf\n",mu);
#endif
/*************************************************************************/
      /* solve augmented equations A x = b */
      /* A==jacTjacd, b==Dpd, after solving, x==Dpd */
      /* b=jacTed : intially right hand side, at exit the solution */
      if (solve_axb==0) {
        /* Cholesky solver **********************/
        /* lower triangle of Ad is destroyed */
        status=my_dpotrf('U',M,jacTjacd,M);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         memcpy(Dpd,jacTed,M*sizeof(double));
         status=my_dpotrs('U',M,1,jacTjacd,M,Dpd,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix info=%d\n",status);
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        /* copy Dpd<=jacTed */
        memcpy(Dpd,jacTed,M*sizeof(double));
        status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,WORK,lwork);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }

      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
        status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,WORK,lwork);
        /* copy Dpd<=jacTed */
        memcpy(bd,jacTed,M*sizeof(double));
        /* b<=U^T * b */
        my_dgemv('T',M,M,1.0,Ud,M,bd,1,0.0,Dpd,1);
        /* robust correction */
        /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        for (ci=0; ci<M; ci++) {
         if (Sd[ci]>eps1) {
          Dpd[ci]=Dpd[ci]/Sd[ci];
         } else {
          Dpd[ci]=0.0;
         }
        }

        /* b<=VT^T * b */
        memcpy(bd,Dpd,M*sizeof(double));
        my_dgemv('T',M,M,1.0,VTd,M,bd,1,0.0,Dpd,1);

        issolved=1;
      }
/*************************************************************************/

      /* compute p's new estimate and ||Dp||^2 */
      if (issolved) {
          /* compute p's new estimate and ||Dp||^2 */
          /* pnew=p+Dp */
          /* pnew=p */
          memcpy(pnew,p,M*sizeof(double));
          /* pnew=pnew+Dp */
          my_daxpy(M,Dpd,1.0,pnew);

          /* norm ||Dp|| */
          Dp_L2=my_dnrm2(M,Dpd);
          Dp_L2=Dp_L2*Dp_L2;

#ifdef DEBUG
printf("norm ||dp|| =%lf, norm ||p||=%lf\n",Dp_L2,p_L2);
#endif
          if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
           stop=2;
           break;
          }

         if(Dp_L2>=(p_L2+eps2)/(CLM_EPSILON*CLM_EPSILON)){ /* almost singular */
          stop=4;
          break;
         }

        /* new function value */
        (*func)(pnew, hxd, M, N, adata); /* evaluate function at p + Dp */

        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        /* copy to device */
        /* hxd<=hx */

        /* e=x */
        memcpy(ed,x,N*sizeof(double));
        /* e=x-hx */
        my_daxpy(N,hxd,-1.0,ed);
        /* note: e is updated */

        /*W e<= wt\odot e */
        my_odot(ed,wtd,N,Nt);

        /* norm ||e|| */
        pDp_eL2=my_dnrm2(N,ed);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finite(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
                                  */
          stop=7;
          break;
        }

        /* dL=Dp'*(mu*Dp+jacTe) */
        /* bd=jacTe+mu*Dp */
        memcpy(bd,jacTed,M*sizeof(double));
        my_daxpy(M,Dpd,mu,bd);
        dL=my_ddot(M,Dpd,bd);

        dF=p_eL2-pDp_eL2;

#ifdef DEBUG
        printf("dF=%lf, dL=%lf\n",dF,dL);
#endif
        if(dL>0.0 && dF>0.0){ /* reduction in error, increment is accepted */
          tmp=(2.0*dF/dL-1.0);
          tmp=1.0-tmp*tmp*tmp;
          mu=mu*((tmp>=CLM_ONE_THIRD)? tmp : CLM_ONE_THIRD);
          nu=2;

          /* update p's estimate */
          memcpy(p,pnew,M*sizeof(double));

          /* update ||e||_2 */
          p_eL2=pDp_eL2;
          break;
        }

      }
      /* if this point is reached, either the linear system could not be solved or
       * the error did not reduce; in any case, the increment must be rejected
       */

      mu*=(double)nu;
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). */
        stop=5;
        break;
      }

      nu=nu2;

    } /* inner loop */

  }
  /**** end iteration loop ***********/


  if(k>=itmax) stop=3;

  /*W if not at first or last iteration, recalculate error */
  if (nw>0 && nw<wt_itmax-1) {
    (*func)(p, hxd, M, N, adata);
    /* e=x */
    my_dcopy(N, x, 1, ed, 1);
    /* e=x-hx */
    my_daxpy(N, hxd, -1.0, ed);
  }

  /*W if not at the last iteration, update weights */
  if (nw<wt_itmax-1) {
   lambda=my_dasum(N,wtd);
   /* update w<= (nu+1)/(nu+delta^2)
      then nu<= new nu
      then w<= sqrt(w) for LS solving
   */
   robust_nu1=update_w_and_nu(robust_nu, wtd, ed, N, Nt, robust_nulow, robust_nuhigh);
#ifdef DEBUG
   printf("nu updated from %lf in [%lf,%lf] to %lf\n",robust_nu,robust_nulow, robust_nuhigh,robust_nu1);
#endif
   robust_nu=robust_nu1;

   /* normalize weights */
   wt_sum=lambda/(double)N;
   my_dscal(N,wt_sum,wtd);
   stop=0; /* restart LM */
  }


  } /* end EM iteration loop */

  lmdata->robust_nu=robust_nu;

  free(jac);
  free(jacTjacd);
  free(jacTjacd0);
  free(jacTed);
  free(Dpd);
  free(bd);
  free(hxd);
  if (!jac_given) { free(hxm); }
  free(ed);
  free(wtd);
  free(aones);
  free(pnew);
 
  if (solve_axb==0) {
  } else if (solve_axb==1) {
   free(WORK);
  } else {
   free(Ud);
   free(VTd);
   free(Sd);
   free(WORK);
  }
#ifdef DEBUG
  printf("stop=%d\n",stop);
#endif
  if(info){
    info[0]=init_p_eL2;
    info[1]=p_eL2;
    info[2]=jacTe_inf;
    info[3]=Dp_L2;
    info[4]=mu;
    info[5]=(double)k;
    info[6]=(double)stop;
    info[7]=(double)0;
    info[8]=(double)0;
    info[9]=(double)0;
  }
  return 0;
}



/* robust LM, OS acceleration */
int
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
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  int nu=2,nu2;
  double p_L2, Dp_L2=DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0, pDp_eL2, init_p_eL2;
  double tmp,mu=0.0;
  double tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  double *hxd;
  double *ed,*wtd;
  double *jac;

  double *jacTjacd,*jacTjacd0;

  double *pnew,*Dpd,*bd;
  double *aones;
  double *jacTed;

  /* used in QR solver */
  double *WORK;
  int lwork=0;
  double w[1];

  int status;

  /* used in SVD solver */
  double *Ud;
  double *VTd;
  double *Sd;


  int solve_axb=linsolv;

  /* setup default settings */
  if(opts){
    tau=opts[0];
    eps1=opts[1];
    eps2=opts[2];
    eps2_sq=opts[2]*opts[2];
    eps3=opts[3];
  } else {
    tau=CLM_INIT_MU;
    eps1=CLM_STOP_THRESH;
    eps2=CLM_STOP_THRESH;
    eps2_sq=CLM_STOP_THRESH*CLM_STOP_THRESH;
    eps3=CLM_STOP_THRESH;
  }

  if ((hxd=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((ed=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jac=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd0=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTed=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Dpd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((bd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((pnew=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((aones=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((wtd=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  WORK=Ud=Sd=VTd=0;
  me_data_t *lmdata0=(me_data_t*)adata;
  int nw,wt_itmax=3;
  double wt_sum,lambda,robust_nu=lmdata0->robust_nu;
  double robust_nu1;


  setweights(M,aones,1.0,lmdata0->Nt);
  /*W set initial weights to 1 */
  setweights(N,wtd,1.0,lmdata0->Nt);

  /* memory allocation: different solvers */
  if (solve_axb==0) {

  } else if (solve_axb==1) {
    /* workspace query */
    status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  } else {
    if ((Ud=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((VTd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((Sd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    
    status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  }


  /* setup OS subsets and stating offsets */
  /* ME data for Jacobian calculation (need a new one) */
  me_data_t lmdata;
  lmdata.clus=lmdata0->clus;
  lmdata.u=lmdata.v=lmdata.w=0;  /* not needed */
  lmdata.Nbase=lmdata0->Nbase;
  lmdata.tilesz=lmdata0->tilesz;
  lmdata.N=lmdata0->N;
  lmdata.carr=lmdata0->carr;
  lmdata.M=lmdata0->M;
  lmdata.Mt=lmdata0->Mt;
  lmdata.freq0=lmdata0->freq0;
  lmdata.Nt=lmdata0->Nt;
  lmdata.barr=lmdata0->barr;
  lmdata.coh=lmdata0->coh;
  lmdata.tileoff=lmdata0->tileoff;


  int Nsubsets=10;
  if (lmdata0->tilesz<Nsubsets) { Nsubsets=lmdata0->tilesz; }
  /* FIXME: is 0.1 enough ? */
  int max_os_iter=(int)ceil(0.1*(double)Nsubsets);
  int Npersubset=(N+Nsubsets-1)/Nsubsets;
  int Ntpersubset=(lmdata0->tilesz+Nsubsets-1)/Nsubsets;
  int *Nos,*edI,*subI=0,*tileI,*tileoff;
  if ((Nos=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((edI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((tileI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((tileoff=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  int l,ositer;;
  k=l=0;
  for (ci=0; ci<Nsubsets; ci++) {
    edI[ci]=k;
    tileoff[ci]=lmdata0->tileoff+l;
    if (l+Ntpersubset<lmdata0->tilesz) {
      Nos[ci]=Npersubset;
      tileI[ci]=Ntpersubset;
    } else {
      Nos[ci]=N-k;
      tileI[ci]=lmdata0->tilesz-l;
    }
    k=k+Npersubset;
    l=l+Ntpersubset;
  }

  /* EM iteration loop */
  /************************************************************/
  for (nw=0; nw<wt_itmax; nw++) {
  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  (*func)(p, hxd, M, N, adata);

  /* e=x */
  my_dcopy(N, x, 1, ed, 1);
  /* e=x-hx */
  my_daxpy(N, hxd, -1.0, ed);

  /*W e<= wt\odot e */
  my_odot(ed,wtd,N,Nt);

  /* norm ||e|| */
  p_eL2=my_dnrm2(N, ed);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finite(p_eL2)) stop=7;


  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("iter=%d err=%lf\n",k,p_eL2);
#endif
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }

    if (randomize) {
     /* random permutation of subsets */
     subI=random_permutation(Nsubsets,0,0);
    }
/**************** OS loop ***************************/
    for (ositer=0; ositer<max_os_iter; ositer++) {
     /* select subset to compute Jacobian */
     if (randomize) { 
      l=subI[ositer];
     } else {
      /* a deterministic value in 0...Nsubsets-1 */
      l=(nw+k+ositer)%Nsubsets;
     } 
     /* note: adata has to advance */
     lmdata.tileoff=tileoff[l];
     lmdata.tilesz=tileI[l];
    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
    */
     (*jacf)(p, jac, M, Nos[l], (void*)&lmdata);
     /*W J<= wt\odot J, each row mult by wt[] */
     /* jac[0..M-1] <- wtd[0],  jac[M...2M-1] <- wtd[1] ... */
     for (ci=0; ci<Nos[l]; ci++) {
      my_dscal(M, wtd[ci+edI[l]], &jac[ci*M]);
     }

     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     my_dgemm('N','T',M,M,Nos[l],1.0,jac,M,jac,M,0.0,jacTjacd,M);
     
     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     my_dcopy(M*M,jacTjacd,1,jacTjacd0,1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     my_dgemv('N',M,Nos[l],1.0,jac,M,&ed[edI[l]],1,0.0,jacTed,1);


     /* Compute ||J^T e||_inf and ||p||^2 */
     /* find infinity norm of J^T e, 1 based indexing*/
     ci=my_idamax(M,jacTed,1);
     memcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double));
     /* L2 norm of current parameter values */
     /* norm ||Dp|| */
     p_L2=my_dnrm2(M,p);
     p_L2=p_L2*p_L2;
     if(jacTe_inf<0.0) {jacTe_inf=-jacTe_inf;}
#ifdef DEBUG
     printf("Inf norm=%lf\n",jacTe_inf);
#endif
     
    /* check for convergence */
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0; /* no increment for p in this case */
      stop=1;
      break;
    }

    /* compute initial (k=0) damping factor */
    if (k==0) {
      /* find max diagonal element (stride is M+1) */
      /* should be MAX not MAX(ABS) */
      ci=my_idamax(M,jacTjacd,M+1);
      ci=(ci-1)*(M+1); /* right value of the diagonal */

      memcpy(&tmp,&(jacTjacd[ci]),sizeof(double));
      mu=tau*tmp;
    }

    
    /* determine increment using adaptive damping */
    while(1){
      /* augment normal equations */
      /* increment A => A+ mu*I, increment diagonal entries */
      /* copy jacTjacd<=jacTjacd0 */
      memcpy(jacTjacd,jacTjacd0,M*M*sizeof(double));
      my_daxpys(M,aones,1,mu,jacTjacd,M+1);

#ifdef DEBUG
      printf("mu=%lf\n",mu);
#endif
/*************************************************************************/
      /* solve augmented equations A x = b */
      /* A==jacTjacd, b==Dpd, after solving, x==Dpd */
      /* b=jacTed : intially right hand side, at exit the solution */
      if (solve_axb==0) {
        /* Cholesky solver **********************/
        /* lower triangle of Ad is destroyed */
        status=my_dpotrf('U',M,jacTjacd,M);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         memcpy(Dpd,jacTed,M*sizeof(double));
         status=my_dpotrs('U',M,1,jacTjacd,M,Dpd,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix info=%d\n",status);
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        /* copy Dpd<=jacTed */
        memcpy(Dpd,jacTed,M*sizeof(double));
        status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,WORK,lwork);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }

      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
        status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,WORK,lwork);
        /* copy Dpd<=jacTed */
        memcpy(bd,jacTed,M*sizeof(double));
        /* b<=U^T * b */
        my_dgemv('T',M,M,1.0,Ud,M,bd,1,0.0,Dpd,1);
        /* robust correction */
        /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        for (ci=0; ci<M; ci++) {
         if (Sd[ci]>eps1) {
          Dpd[ci]=Dpd[ci]/Sd[ci];
         } else {
          Dpd[ci]=0.0;
         }
        }

        /* b<=VT^T * b */
        memcpy(bd,Dpd,M*sizeof(double));
        my_dgemv('T',M,M,1.0,VTd,M,bd,1,0.0,Dpd,1);

        issolved=1;
      }
/*************************************************************************/

      /* compute p's new estimate and ||Dp||^2 */
      if (issolved) {
          /* compute p's new estimate and ||Dp||^2 */
          /* pnew=p+Dp */
          /* pnew=p */
          memcpy(pnew,p,M*sizeof(double));
          /* pnew=pnew+Dp */
          my_daxpy(M,Dpd,1.0,pnew);

          /* norm ||Dp|| */
          Dp_L2=my_dnrm2(M,Dpd);
          Dp_L2=Dp_L2*Dp_L2;

#ifdef DEBUG
printf("norm ||dp|| =%lf, norm ||p||=%lf\n",Dp_L2,p_L2);
#endif
          if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
           stop=2;
           break;
          }

         if(Dp_L2>=(p_L2+eps2)/(CLM_EPSILON*CLM_EPSILON)){ /* almost singular */
          stop=4;
          break;
         }

        /* new function value */
        (*func)(pnew, hxd, M, N, adata); /* evaluate function at p + Dp */

        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        /* copy to device */
        /* hxd<=hx */

        /* e=x */
        memcpy(ed,x,N*sizeof(double));
        /* e=x-hx */
        my_daxpy(N,hxd,-1.0,ed);
        /* note: e is updated */

        /*W e<= wt\odot e */
        my_odot(ed,wtd,N,Nt);

        /* norm ||e|| */
        pDp_eL2=my_dnrm2(N,ed);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finite(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
                                  */
          stop=7;
          break;
        }

        /* dL=Dp'*(mu*Dp+jacTe) */
        /* bd=jacTe+mu*Dp */
        memcpy(bd,jacTed,M*sizeof(double));
        my_daxpy(M,Dpd,mu,bd);
        dL=my_ddot(M,Dpd,bd);

        dF=p_eL2-pDp_eL2;

#ifdef DEBUG
        printf("dF=%lf, dL=%lf\n",dF,dL);
#endif
        if(dL>0.0 && dF>0.0){ /* reduction in error, increment is accepted */
          tmp=(2.0*dF/dL-1.0);
          tmp=1.0-tmp*tmp*tmp;
          mu=mu*((tmp>=CLM_ONE_THIRD)? tmp : CLM_ONE_THIRD);
          nu=2;

          /* update p's estimate */
          memcpy(p,pnew,M*sizeof(double));

          /* update ||e||_2 */
          p_eL2=pDp_eL2;
          break;
        }

      }
      /* if this point is reached, either the linear system could not be solved or
       * the error did not reduce; in any case, the increment must be rejected
       */

      mu*=(double)nu;
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). */
        stop=5;
        break;
      }

      nu=nu2;

    } /* inner loop */

  }
  if (randomize) {
   free(subI);
  }
/**************** end OS loop ***************************/
  }
  /**** end iteration loop ***********/


  if(k>=itmax) stop=3;

  /*W if not at first or last iteration, recalculate error */
  if (nw>0 && nw<wt_itmax-1) {
    (*func)(p, hxd, M, N, adata);
    /* e=x */
    my_dcopy(N, x, 1, ed, 1);
    /* e=x-hx */
    my_daxpy(N, hxd, -1.0, ed);
  }

  /*W if not at the last iteration, update weights */
  if (nw<wt_itmax-1) {
   lambda=my_dasum(N,wtd);
   /* update w<= (nu+1)/(nu+delta^2)
      then nu<= new nu
      then w<= sqrt(w) for LS solving
   */
   robust_nu1=update_w_and_nu(robust_nu, wtd, ed, N, Nt, robust_nulow, robust_nuhigh);
#ifdef DEBUG
   printf("nu updated from %lf in [%lf,%lf] to %lf\n",robust_nu,robust_nulow, robust_nuhigh,robust_nu1);
#endif
   robust_nu=robust_nu1;

   /* normalize weights */
   wt_sum=lambda/(double)N;
   my_dscal(N,wt_sum,wtd);
   stop=0; /* restart LM */
  }


  } /* end EM iteration loop */
  free(Nos);
  free(edI);
  free(tileI);
  free(tileoff);

  lmdata0->robust_nu=robust_nu;

  free(jac);
  free(jacTjacd);
  free(jacTjacd0);
  free(jacTed);
  free(Dpd);
  free(bd);
  free(hxd);
  free(ed);
  free(wtd);
  free(aones);
  free(pnew);
 
  if (solve_axb==0) {
  } else if (solve_axb==1) {
   free(WORK);
  } else {
   free(Ud);
   free(VTd);
   free(Sd);
   free(WORK);
  }
#ifdef DEBUG
  printf("stop=%d\n",stop);
#endif
  if(info){
    info[0]=init_p_eL2;
    info[1]=p_eL2;
    info[2]=jacTe_inf;
    info[3]=Dp_L2;
    info[4]=mu;
    info[5]=(double)k;
    info[6]=(double)stop;
    info[7]=(double)0;
    info[8]=(double)0;
    info[9]=(double)0;
  }
  return 0;
}
