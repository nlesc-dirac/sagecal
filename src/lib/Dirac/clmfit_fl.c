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
checkCudaError(cudaError_t err, const char *file, int line)
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




/* OS-LM, but f() and jac() calculations are done 
  entirely in the GPU */
int
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
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  float *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
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

  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  float *cohd;
  /* baseline-station map on device/host */
  short *bbd;

  int solve_axb=linsolv;

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

  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  /* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
  cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, Nbase, dp->M, dp->N);

  /* e=x */
  cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
  /* e=x-hx */
  float alpha=-1.0f;
  cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);

  /* norm ||e|| */
  cbstatus=cublasSnrm2(cbhandle, N, ed, 1, &p_eL2);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finitef(p_eL2)) stop=7;

  /* setup OS subsets and stating offsets */
  /* ed : N, cohd : Nbase*8, bbd : Nbase*2 full size */
  /* if ntiles<Nsubsets, make Nsubsets=ntiles */
  int Nsubsets=10; 
  if (ntiles<Nsubsets) { Nsubsets=ntiles; }
  /* FIXME: is 0.1 enough ? */
  int max_os_iter=(int)ceil(0.1*(double)Nsubsets);
  int Npersubset=(N+Nsubsets-1)/Nsubsets;
  int Nbasepersubset=(Nbase+Nsubsets-1)/Nsubsets;
  int *Nos,*Nbaseos,*edI,*NbI,*subI=0;
  if ((Nos=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((Nbaseos=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((edI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((NbI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  int l,ositer;;
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

#ifdef DEBUG
  for (ci=0; ci<Nsubsets; ci++) {
   printf("ci=%d, Nos=%d, edI=%d, Nbseos=%d, NbI=%d\n",ci,Nos[ci],edI[ci],Nbaseos[ci],NbI[ci]);
  }
#endif

  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("iter=%d err=%f\n",k,p_eL2);
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
      l=(k+ositer)%Nsubsets;
     }
     /* NOTE: no. of subsets >= no. of OS iterations, so select
        a random set of subsets */
     /* N, Nbase changes with subset, cohd,bbd,ed gets offsets */
     /* ed : N, cohd : Nbase*8, bbd : Nbase*2 full size */
    /* p: params (Mx1), jacd: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
    /* FIXME thread/block sizes 16x16=256, so 16 is chosen */
     cudakernel_jacf_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, Nos[l], &cohd[8*NbI[l]], &bbd[2*NbI[l]], Nbaseos[l], dp->M, dp->N);

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
      printf("mu=%f\n",mu);
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
         fprintf(stderr,"Singular matrix\n");
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
           fprintf(stderr,"Singular matrix\n");
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
         fprintf(stderr,"Singular matrix\n");
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
           fprintf(stderr,"Singular matrix\n");
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
        cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);

        /* e=x */
        cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
        /* e=x-hx */
        alpha=-1.0f;
        cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
        /* note: e is updated */

        /* norm ||e|| */
        cbstatus=cublasSnrm2(cbhandle, N, ed, 1, &pDp_eL2);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finitef(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
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
  free(Nos);
  free(Nbaseos);
  free(edI);
  free(NbI);

  if(k>=itmax) stop=3;

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


int
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
                      */

  cublasHandle_t cbhandle, /* device handle */
  cusolverDnHandle_t solver_handle, /* solver handle */
  float *gWORK, /* GPU allocated memory */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int tileoff, /* tile offset when solving for many chunks */
  int ntiles, /* total tile (data) size being solved for */
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

  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  float *cohd;
  /* baseline-station map on device/host */
  short *bbd;

  int solve_axb=linsolv;

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

  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  /* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
  cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, Nbase, dp->M, dp->N);

  /* e=x */
  cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
  /* e=x-hx */
  float alpha=-1.0f;
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
    printf("iter=%d err=%f\n",k,p_eL2);
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
     cudakernel_jacf_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, Nbase, dp->M, dp->N);

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
      printf("mu=%f\n",mu);
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
         fprintf(stderr,"Singular matrix\n");
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
           fprintf(stderr,"Singular matrix\n");
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
         fprintf(stderr,"Singular matrix\n");
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
           fprintf(stderr,"Singular matrix\n");
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
        cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);

        /* e=x */
        cbstatus=cublasScopy(cbhandle, N, xd, 1, ed, 1);
        /* e=x-hx */
        alpha=-1.0f;
        cbstatus=cublasSaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
        /* note: e is updated */

        /* norm ||e|| */
        cbstatus=cublasSnrm2(cbhandle, N, ed, 1, &pDp_eL2);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finitef(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
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
