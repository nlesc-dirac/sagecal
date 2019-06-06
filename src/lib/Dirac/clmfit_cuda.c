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
#include <unistd.h>

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



/** keep interface almost the same as in levmar **/
int
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

  int card,   /* device 0, 1 */
  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  cudaError_t err;
  cublasStatus_t cbstatus;
  cublasHandle_t cbhandle;
  cusolverDnHandle_t solver_handle;

  int nu=2,nu2;
  double p_L2, Dp_L2=DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0, pDp_eL2, init_p_eL2;
  double tmp,mu=0.0;
  double tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  double *hx;
  double *hxd;
  
  double *ed;
  double *xd;

  double *jac,*jacd;

  double *jacTjacd,*jacTjacd0;

  double *pnew,*Dpd,*bd;
  double *pd,*pnewd;
  double *jacTed;

  /* used in QR solver */
  double *taud;

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

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;

  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);
  cusolverDnCreate(&solver_handle);

  cbstatus=cublasCreate(&cbhandle);
  if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUBLAS create fail\n",__FILE__,__LINE__);
    exit(1);
  }

  err=cudaMalloc((void**)&xd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  /* xd <=x */
  err=cudaMemcpy(xd, x, N*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  if ((hx=(double*)calloc((size_t)N,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jac=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((pnew=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }


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
  err=cudaMemcpy(pd, p, M*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  /* memory allocation: different solvers */
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
    err=cudaMalloc((void**)&taud, M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    cusolverDnDgeqrf_bufferSize(solver_handle, M, M, jacTjacd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  } else {
    err=cudaMalloc((void**)&Ud, M*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&VTd, M*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&Sd, M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    cusolverDnDgesvd_bufferSize(solver_handle, M, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }


  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  (*func)(p, hx, M, N, adata);
  /* copy to device */
  err=cudaMalloc((void**)&hxd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  /* hxd<=hx */
  err=cudaMemcpy(hxd, hx, N*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaMalloc((void**)&ed, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  /* e=x */
  cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
  /* e=x-hx */
  double alpha=-1.0;
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
    (*jacf)(p, jac, M, N, adata);

    err=cudaMemcpy(jacd, jac, M*N*sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);

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
         fprintf(stderr,"Singular matrix\n");
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
#ifdef DEBUG
         checkCublasError(cbstatus);
#endif
         //status=culaDeviceDpotrs('U',M,1,jacTjacd,M,Dpd,M);
         cusolverDnDpotrs(solver_handle, CUBLAS_FILL_MODE_UPPER,M,1,jacTjacd,M,Dpd,M,devInfo);
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
        //status=culaDeviceDgeqrf(M,M,jacTjacd,M,taud);
        cusolverDnDgeqrf(solver_handle, M, M, jacTjacd, M, taud, work, work_size, devInfo);
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
         cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
         //status=culaDeviceDgeqrs(M,M,1,jacTjacd,M,taud,Dpd,M);
         cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, 1, M, jacTjacd, M, taud, Dpd, M, work, work_size, devInfo);
         cudaDeviceSynchronize();
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           fprintf(stderr,"Singular matrix\n");
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
        //checkStatus(status,__FILE__,__LINE__);
        cone=1.0; czero=0.0;
        cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,Ud,M,Dpd,1,&czero,Dpd,1);
        /* get back the values  jTjdiag<=Sd, pnew<=Dpd*/
//        err=cudaMemcpy(pnew,Dpd,M*sizeof(double),cudaMemcpyDeviceToHost);
//        checkCudaError(err);
//        err=cudaMemcpy(jTjdiag,Sd,M*sizeof(double),cudaMemcpyDeviceToHost);
//        checkCudaError(err);

        /* robust correction */
//        for (ci=0; ci<M; ci++) {
//         pnew[ci]=pnew[ci]/jTjdiag[ci];
//        }
        /* copy back  bd<=xs*/
//        err=cudaMemcpy(Dpd,pnew,M*sizeof(double),cudaMemcpyHostToDevice);
//        checkCudaError(err);
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

          /* copy back the solution to host */
          err=cudaMemcpy(pnew,pnewd,M*sizeof(double),cudaMemcpyDeviceToHost);
          checkCudaError(err,__FILE__,__LINE__);

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
        (*func)(pnew, hx, M, N, adata); /* evaluate function at p + Dp */

        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        /* copy to device */
        /* hxd<=hx */
        err=cudaMemcpy(hxd, hx, N*sizeof(double), cudaMemcpyHostToDevice);
        checkCudaError(err,__FILE__,__LINE__);

        /* e=x */
        cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
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

    /* copy back solution, need for jacobian calculation */
    err=cudaMemcpy(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost);
    checkCudaError(err,__FILE__,__LINE__);
  }
  /**** end iteration loop ***********/


  if(k>=itmax) stop=3;

  /* copy back current solution */
  err=cudaMemcpy(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost);
  checkCudaError(err,__FILE__,__LINE__);



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
  cudaFree(ed);
  cudaFree(devInfo);
  cudaFree(work);
  if (solve_axb==0) {
  } else if (solve_axb==1) {
   cudaFree(taud);
  } else {
   cudaFree(Ud);
   cudaFree(VTd);
   cudaFree(Sd);
   cudaFree(rwork);
  }
  cublasDestroy(cbhandle);
  cusolverDnDestroy(solver_handle);
  free(hx);
  free(jac);
  free(pnew);

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


/* same as above, but f() and jac() calculations are done 
  entirely in the GPU */
int
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

  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  double *cohd;
  /* baseline-station map on device/host */
  short *bbd;

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

  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;


  unsigned long int moff; /* make sure offsets are multiples of 4 */
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


  err=cudaMemcpyAsync(pd, p, M*sizeof(double), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  /* need to give right offset for coherencies */
  /* offset: cluster offset+time offset */
  err=cudaMemcpyAsync(cohd, &(dp->ddcoh[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(double), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  /* correct offset for baselines */
  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  cudaDeviceSynchronize();
  /* xd <=x */
  err=cudaMemcpyAsync(xd, x, N*sizeof(double), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);

  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  /* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
  cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, Nbase, dp->M, dp->N);

  /* e=x */
  cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
  /* e=x-hx */
  double alpha=-1.0;
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
    /* p: params (Mx1), jacd: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
    /* FIXME thread/block sizes 16x16=256, so 16 is chosen */
     cudakernel_jacf(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, Nbase, dp->M, dp->N);

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
         fprintf(stderr,"Singular matrix\n");
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
#ifdef DEBUG
         checkCublasError(cbstatus);
#endif
         //status=culaDeviceDpotrs('U',M,1,jacTjacd,M,Dpd,M);
         cusolverDnDpotrs(solver_handle, CUBLAS_FILL_MODE_UPPER,M,1,jacTjacd,M,Dpd,M,devInfo);
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
        //status=culaDeviceDgeqrf(M,M,jacTjacd,M,taud);
        cusolverDnDgeqrf(solver_handle, M, M, jacTjacd, M, taud, work, work_size, devInfo);
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
         cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
         //status=culaDeviceDgeqrs(M,M,1,jacTjacd,M,taud,Dpd,M);
         cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, 1, M, jacTjacd, M, taud, Dpd, M, work, work_size, devInfo);
         cudaDeviceSynchronize();
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           fprintf(stderr,"Singular matrix\n");
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
        //checkStatus(status,__FILE__,__LINE__);
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
        cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);

        /* e=x */
        cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
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

  /* copy back current solution */
  err=cudaMemcpyAsync(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost,0);
  checkCudaError(err,__FILE__,__LINE__);

  /* check once CUBLAS error */
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


/* function to set up a GPU, should be called only once */
void
attach_gpu_to_thread(int card,  cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle) {

  cudaError_t err;
  cublasStatus_t cbstatus;
  cusolverStatus_t status;
  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);
  status=cusolverDnCreate(solver_handle);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUSOLV create fail %d\n",__FILE__,__LINE__,status);
    exit(1);
  }

  cbstatus=cublasCreate(cbhandle);
  if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUBLAS create fail\n",__FILE__,__LINE__);
    exit(1);
  }

}
void
attach_gpu_to_thread1(int card,  cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle, double **WORK, int64_t work_size) {

  cudaError_t err;
  cublasStatus_t cbstatus;
  cusolverStatus_t status;
  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);
  status=cusolverDnCreate(solver_handle);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUSOLV create fail %d\n",__FILE__,__LINE__,status);
    sleep(10);
    status=cusolverDnCreate(solver_handle);
    if (status != CUSOLVER_STATUS_SUCCESS) {
     fprintf(stderr,"%s: %d: CUSOLV create fail %d\n",__FILE__,__LINE__,status);
     fprintf(stderr,"common problems: not initialized %d, alloc fail %d, no compute %d\n",CUSOLVER_STATUS_NOT_INITIALIZED, CUSOLVER_STATUS_ALLOC_FAILED, CUSOLVER_STATUS_ARCH_MISMATCH);
     exit(1);
    }
  }

  cbstatus=cublasCreate(cbhandle);
  if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    /* retry once more before exiting */
    fprintf(stderr,"%s: %d: CUBLAS create failure, retrying\n",__FILE__,__LINE__);
    sleep(10);
    cbstatus=cublasCreate(cbhandle);
    if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
     fprintf(stderr,"%s: %d: CUBLAS create fail\n",__FILE__,__LINE__);
     exit(1);
    }
  }

  err=cudaMalloc((void**)WORK, (size_t)work_size);
  checkCudaError(err,__FILE__,__LINE__);

}
void
attach_gpu_to_thread2(int card,  cublasHandle_t *cbhandle,  cusolverDnHandle_t *solver_handle, float **WORK, int64_t work_size, int usecula) {

  cudaError_t err;
  cublasStatus_t cbstatus;
  cusolverStatus_t status;
  err=cudaSetDevice(card); /* we need this */
  checkCudaError(err,__FILE__,__LINE__);
  if (usecula) {
   status=cusolverDnCreate(solver_handle);
   if (status != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUSOLV create fail card %d, %d\n",__FILE__,__LINE__,card,status);
    sleep(10);
    status=cusolverDnCreate(solver_handle);
    if (status != CUSOLVER_STATUS_SUCCESS) {
     fprintf(stderr,"%s: %d: CUSOLV create fail card %d, %d\n",__FILE__,__LINE__,card,status);
     exit(1);
    }
   }
  }

  cbstatus=cublasCreate(cbhandle);
  if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    /* retry once more before exiting */
    fprintf(stderr,"%s: %d: CUBLAS create failure, retrying\n",__FILE__,__LINE__);
    sleep(10);
    cbstatus=cublasCreate(cbhandle);
    if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
     fprintf(stderr,"%s: %d: CUBLAS create fail\n",__FILE__,__LINE__);
     exit(1);
    }
  }

  err=cudaMalloc((void**)WORK, (size_t)work_size);
  checkCudaError(err,__FILE__,__LINE__);

}
void
detach_gpu_from_thread(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
  cublasDestroy(cbhandle);
  cusolverDnDestroy(solver_handle);
}
void
detach_gpu_from_thread1(cublasHandle_t cbhandle,cusolverDnHandle_t solver_handle,double *WORK) {

  cublasDestroy(cbhandle);
  cusolverDnDestroy(solver_handle);
  cudaFree(WORK);
}
void
detach_gpu_from_thread2(cublasHandle_t cbhandle,cusolverDnHandle_t solver_handle,float *WORK, int usecula) {

  cublasDestroy(cbhandle);
  if (usecula) {
   cusolverDnDestroy(solver_handle);
  }
  cudaFree(WORK);
}
void
reset_gpu_memory(double *WORK, int64_t work_size) {

  cudaError_t err;

  err=cudaMemset((void*)WORK, 0, (size_t)work_size);
  checkCudaError(err,__FILE__,__LINE__);
}


/** keep interface almost the same as in levmar **/
int
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
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{
  cudaError_t err;
  cublasStatus_t cbstatus;

  /* NOTE F()=func()-data */
  double *xd,*Jkd,*Fxkd,*Fykd,*Jkdkd,*JkTed,*JkTed0,*JkTJkd,*JkTJkd0,*dkd,*dhatkd, *ykd, *skd, *pd;

  double lambda;
  double mu,m,p0,p1,p2; 
  int delta;
  double Fxknrm,Fyknrm,Fykdhatknrm,Fxksknrm,FJkdknrm;
  int niter=0;
  int p_update=1;
  double Fxknrm2,Fxksknrm2;

  double Ak,Pk,rk;

  /* use cudaHostAlloc  and cudaFreeHost */
  /* used in QR solver */
  double *taud=0;
  /* used in SVD solver */
  double *Ud=0;
  double *VTd=0;
  double *Sd=0;

  int issolved;
  int solve_axb=linsolv;

  /* ME data */
  me_data_t *dp=(me_data_t*)adata;
  int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
  /* coherency on device */
  double *cohd;
  /* baseline-station map on device/host */
  short *bbd;


  /* calculate no of cuda threads and blocks */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int BlocksPerGrid=(M+ThreadsPerBlock-1)/ThreadsPerBlock;


  if (opts) {
    mu=opts[0];
    m=opts[1];
    p0=opts[2];
    p1=opts[3];
    p2=opts[4];
    delta=(int)opts[5];  
  } else {
    mu=1e-3;//1e-5;
    m=1e-2;//1e-3;
    p0=0.0001;
    p1=0.25;
    p2=0.75;
    delta=1;  /* 1 or 2 */
  }

  double epsilon=CLM_EPSILON;

  unsigned long int moff;
  if (!gWORK) {
  err=cudaMalloc((void**)&xd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&Jkd, M*N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&Fxkd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&Fykd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&Jkdkd, N*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&JkTed, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&JkTed0, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&JkTJkd, M*M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&JkTJkd0, M*M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&dkd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&dhatkd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&ykd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&skd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&pd, M*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  /* needed for calculating f()  and jac() */
  err=cudaMalloc((void**) &bbd, Nbase*2*sizeof(short));
  checkCudaError(err,__FILE__,__LINE__);
  /* we need coherencies for only this cluster */
  err=cudaMalloc((void**) &cohd, Nbase*8*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  /* memory allocation: different solvers */
  if (solve_axb==1) {
    /* QR solver ********************************/
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
  } else { /* use pre allocated memory */
    moff=0;
    xd=&gWORK[moff];
    moff+=N;
    Jkd=&gWORK[moff];
    moff+=M*N;
    Fxkd=&gWORK[moff];
    moff+=N;
    Fykd=&gWORK[moff];
    moff+=N;
    Jkdkd=&gWORK[moff];
    moff+=N;
    JkTed=&gWORK[moff];
    moff+=M;
    JkTed0=&gWORK[moff];
    moff+=M;
    JkTJkd=&gWORK[moff];
    moff+=M*M;
    JkTJkd0=&gWORK[moff];
    moff+=M*M;
    dkd=&gWORK[moff];
    moff+=M;
    dhatkd=&gWORK[moff];
    moff+=M;
    ykd=&gWORK[moff];
    moff+=M;
    skd=&gWORK[moff];
    moff+=M;
    pd=&gWORK[moff];
    moff+=M;
    cohd=&gWORK[moff];
    moff+=Nbase*8;
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
    cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, M, JkTJkd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  } else if (solve_axb==1) {
    cusolverDnDgeqrf_bufferSize(solver_handle, M, M, JkTJkd, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  } else {
    cusolverDnDgesvd_bufferSize(solver_handle, M, M, &work_size);
    err=cudaMalloc((void**)&work, work_size*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }


  err=cudaMemcpyAsync(pd, p, M*sizeof(double), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  /* need to give right offset for coherencies */
  /* offset: cluster offset+time offset */
  err=cudaMemcpyAsync(cohd, &(dp->ddcoh[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(double), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
 /* correct offset for baselines */
  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
  checkCudaError(err,__FILE__,__LINE__);
  cudaDeviceSynchronize();
  /* xd <=x */
  err=cudaMemcpyAsync(xd, x, N*sizeof(double), cudaMemcpyHostToDevice, 0);
  checkCudaError(err,__FILE__,__LINE__);

  /* F(x_k) = func()-data */
  /* func() */
  //(*func)(p, Fxk, M, N, adata);
  /* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
   cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,Fxkd,M,N, cohd, bbd, Nbase, dp->M, dp->N);

  /* func() - data */
  double alpha=-1.0;
  cbstatus=cublasDaxpy(cbhandle, N, &alpha, xd, 1, Fxkd, 1);
  //my_daxpy(N, x, -1.0, Fxk);

  /* find ||Fxk|| */
  //Fxknrm=my_dnrm2(N,Fxk);
  cbstatus=cublasDnrm2(cbhandle, N, Fxkd, 1, &Fxknrm);

  double init_Fxknrm=Fxknrm;
#ifdef DEBUG
  printf("init norm=%lf\n",Fxknrm);
#endif


  double cone=1.0; double czero=0.0;
  while (niter<itmax) {
     if (delta>1) {
      lambda=mu*Fxknrm*Fxknrm;
     } else {
      lambda=mu*Fxknrm;
     }
     Fxknrm2=Fxknrm*Fxknrm;

     if ( p_update==1 ) {
      /* J_k */
      /* p: params (Mx1), jacd: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations*/
      /* FIXME thread/block sizes 16x16=256, so 16 is chosen */
      cudakernel_jacf(ThreadsPerBlock, ThreadsPerBlock/4, pd, Jkd, M, N, cohd, bbd, Nbase, dp->M, dp->N);

      /* Compute J_k^T J_k and -J_k^T F(x_k) */
      //my_dgemm('N','T',M,M,N,1.0,Jk,M,Jk,M,0.0,JkTJk0,M);
      //my_dgemv('N',M,N,-1.0,Jk,M,Fxk,1,0.0,JkTe0,1);

      //status=culaDeviceDgemm('N','T',M,M,N,1.0,Jkd,M,Jkd,M,0.0,JkTJkd0,M);
      //checkStatus(status,__FILE__,__LINE__);
      double cone=1.0; double czero=0.0;
      cbstatus=cublasDgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_T,M,M,N,&cone,Jkd,M,Jkd,M,&czero,JkTJkd0,M);
      //status=culaDeviceDgemv('N',M,N,-1.0,Jkd,M,Fxkd,1,0.0,JkTed0,1);
      //checkStatus(status,__FILE__,__LINE__);
      cone=-1.0;
      cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_N,M,N,&cone,Jkd,M,Fxkd,1,&czero,JkTed0,1);
     }
     /* if || J_k^T F(x_k) || < epsilon, stop */
     //Fyknrm=my_dnrm2(M,JkTe0);
     cbstatus=cublasDnrm2(cbhandle, M, JkTed0, 1, &Fyknrm);

     if (Fyknrm<epsilon) { 
#ifdef DEBUG
      printf("stopping 1 at iter %d\n",niter);
#endif
      break; 
     }

     //memcpy(JkTe,JkTe0,M*sizeof(double));
     //memcpy(JkTJk,JkTJk0,M*M*sizeof(double));
     cbstatus=cublasDcopy(cbhandle, M, JkTed0, 1, JkTed, 1);
     cbstatus=cublasDcopy(cbhandle, M*M, JkTJkd0, 1, JkTJkd, 1);

     /* add lambdaxI to J^T J */
     //my_daxpys(M,aones,1,lambda,JkTJk,M+1);
     cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, JkTJkd, lambda);
  
/********************************************************************/
     if (solve_axb==0) {
       /* Cholesky solver **********************/
       //status=culaDeviceDpotrf('U',M,JkTJkd,M);
       //checkStatus(status,__FILE__,__LINE__);
       cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, M, JkTJkd, M, work, work_size, devInfo);
       cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

       if (!devInfo_h) {
         issolved=1;
       } else {
         issolved=0;
#ifdef DEBUG
         fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
       }
       if (issolved) {
         /* copy dk<=JkTe */
         cbstatus=cublasDcopy(cbhandle, M, JkTed, 1, dkd, 1);
         //status=culaDeviceDpotrs('U',M,1,JkTJkd,M,dkd,M);
         //checkStatus(status,__FILE__,__LINE__);
         cusolverDnDpotrs(solver_handle, CUBLAS_FILL_MODE_UPPER,M,1,JkTJkd,M,dkd,M,devInfo);
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
         }
        }
     } else if (solve_axb==1) {
       /* QR solver ********************************/
       /* QR factorization: JkTJk and TAU now have that */
       //status=my_dgeqrf(M,M,JkTJk,M,TAU,w,lwork);
       /* copy JkTJk as R (only upper triangle is used) */
       //memcpy(R,JkTJk,M*M*sizeof(double));
       /* form Q in JkTJk */
       //my_dorgqr(M,M,M,JkTJk,M,TAU,WORK,lwork);
       /* dk <= Q^T jacTed */
       //my_dgemv('T',M,M,1.0,JkTJk,M,JkTe,1,0.0,dk,1);
       /* solve R x = b */
       //status=my_dtrtrs('U','N','N',M,1,R,M,dk,M);
       //status=culaDeviceDgeqrf(M,M,JkTJkd,M,taud);
       //checkStatus(status,__FILE__,__LINE__);
       cusolverDnDgeqrf(solver_handle, M, M, JkTJkd, M, taud, work, work_size, devInfo);
       cudaDeviceSynchronize();
       cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
       if (!devInfo_h) {
        issolved=1;
       } else {
        issolved=0;
#ifdef DEBUG
        fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
       }
       if (issolved) {
         cbstatus=cublasDcopy(cbhandle, M, JkTed, 1, dkd, 1);
         //status=culaDeviceDgeqrs(M,M,1,JkTJkd,M,taud,dkd,M);
         cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, 1, M, JkTJkd, M, taud, dkd, M, work, work_size, devInfo);
         cudaDeviceSynchronize();
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           fprintf(stderr,"Singular matrix\n");
#endif
         } else {
          cone=1.0;
          cbstatus=cublasDtrsm(cbhandle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,M,1,&cone,JkTJkd,M,dkd,M);
         }
       }
     } else {
       /* SVD solver *********************************/
       /* U S VT = A */
       //status=my_dgesvd('A','A',M,M,JkTJk,M,Sd,Ud,M,VTd,M,WORK,lwork);
       /* dk <= U^T jacTed */
       //my_dgemv('T',M,M,1.0,Ud,M,JkTe,1,0.0,dk,1);
       /* robust correction */
       /* divide by singular values  dk[]/Sd[]  for Sd[]> epsilon */
       /*for (ci=0; ci<M; ci++) {
         if (Sd[ci]>epsilon) {
          dk[ci]=dk[ci]/Sd[ci];
         } else {
          dk[ci]=0.0;
         }
       } */

       /* dk <= VT^T dk */
       //memcpy(yk,dk,M*sizeof(double));
       //my_dgemv('T',M,M,1.0,VTd,M,yk,1,0.0,dk,1);
        /* U S VT = A */
        //status=culaDeviceDgesvd('A','A',M,M,JkTJkd,M,Sd,Ud,M,VTd,M);
        //checkStatus(status,__FILE__,__LINE__);
        cusolverDnDgesvd(solver_handle,'A','A',M,M,JkTJkd,M,Sd,Ud,M,VTd,M,work,work_size,rwork,devInfo);
        cudaDeviceSynchronize();
        /* b<=U^T * b */
        //status=culaDeviceDgemv('T',M,M,1.0,Ud,M,JkTed,1,0.0,dkd,1);
        //checkStatus(status,__FILE__,__LINE__);
        cone=1.0; czero=0.0;
        cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,Ud,M,JkTed,1,&czero,dkd,1);
 
   /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, epsilon, dkd, Sd);

        /* b<=VT^T * b */
        cbstatus=cublasDcopy(cbhandle, M, dkd, 1, ykd, 1);
        //status=culaDeviceDgemv('T',M,M,1.0,VTd,M,ykd,1,0.0,dkd,1);
        //checkStatus(status,__FILE__,__LINE__);
        cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,VTd,M,ykd,1,&czero,dkd,1);
 
        issolved=1;
     }
/********************************************************************/

     /* y_k<= x_k+ d_k */
     //my_dcopy(M,p,1,yk,1);
     //my_daxpy(M,dk,1.0,yk);

     cbstatus=cublasDcopy(cbhandle, M, pd, 1, ykd, 1);
     alpha=1.0;
     cbstatus=cublasDaxpy(cbhandle, M, &alpha, dkd, 1, ykd, 1);

     /* compute F(y_k) */
     /* func() */
     //(*func)(yk, Fyk, M, N, adata);
     cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, ykd,Fykd,M,N, cohd, bbd, Nbase, dp->M, dp->N);

     /* func() - data */
     //my_daxpy(N, x, -1.0, Fyk);
     /* copy to device */

     /* func() - data */
     alpha=-1.0;
     cbstatus=cublasDaxpy(cbhandle, N, &alpha, xd, 1, Fykd, 1);


     /* Compute -J_k^T F(y_k) */
     //my_dgemv('N',M,N,-1.0,Jk,M,Fyk,1,0.0,JkTe,1);
     //status=culaDeviceDgemv('N',M,N,1.0,Jkd,M,Fykd,1,0.0,JkTed,1);
     //checkStatus(status,__FILE__,__LINE__);
     cone=1.0; czero=0.0;
     cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_N,M,N,&cone,Jkd,M,Fykd,1,&czero,JkTed,1);
 
  
/********************************************************************/
     if (solve_axb==0) {
       /* Cholesky solver **********************/
       /* copy dk<=JkTe */
       //  memcpy(dhatk,JkTe,M*sizeof(double));
       //  status=my_dpotrs('U',M,1,JkTJk,M,dhatk,M);
         cbstatus=cublasDcopy(cbhandle, M, JkTed, 1, dhatkd, 1);
         //status=culaDeviceDpotrs('U',M,1,JkTJkd,M,dhatkd,M);
         cusolverDnDpotrs(solver_handle, CUBLAS_FILL_MODE_UPPER,M,1,JkTJkd,M,dhatkd,M,devInfo);
         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
         if (devInfo_h) {
           issolved=0;
#ifdef DEBUG
           fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
         }
     } else if (solve_axb==1) {
       /* QR solver ********************************/
       /* dhatk <= Q^T jacTed */
       //my_dgemv('T',M,M,1.0,JkTJk,M,JkTe,1,0.0,dhatk,1);
       /* solve R x = b */
       //status=my_dtrtrs('U','N','N',M,1,R,M,dhatk,M);
        cbstatus=cublasDcopy(cbhandle, M, JkTed, 1, dhatkd, 1);
       //status=culaDeviceDgeqrs(M,M,1,JkTJkd,M,taud,dhatkd,M);
        cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, 1, M, JkTJkd, M, taud, dhatkd, M, work, work_size, devInfo);
        cudaDeviceSynchronize();
        cbstatus=cublasDtrsm(cbhandle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,M,1,&cone,JkTJkd,M,dhatkd,M);

        issolved=1;
     } else {
       /* SVD solver *********************************/
       /* dhatk <= U^T jacTed */
       //my_dgemv('T',M,M,1.0,Ud,M,JkTe,1,0.0,dhatk,1);
       /* robust correction */
       /* divide by singular values  dk[]/Sd[]  for Sd[]> epsilon */
       /*for (ci=0; ci<M; ci++) {
         if (Sd[ci]>epsilon) {
          dhatk[ci]=dhatk[ci]/Sd[ci];
         } else {
          dhatk[ci]=0.0;
         }
       }*/
       /* dk <= VT^T dk */
       //memcpy(yk,dhatk,M*sizeof(double));
       //my_dgemv('T',M,M,1.0,VTd,M,yk,1,0.0,dhatk,1);
       //status=culaDeviceDgemv('T',M,M,1.0,Ud,M,JkTed,1,0.0,dhatkd,1);
       //checkStatus(status,__FILE__,__LINE__);
       cone=1.0; czero=0.0;
       cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,Ud,M,JkTed,1,&czero,dhatkd,1);
 
   /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
       cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, epsilon, dhatkd, Sd);

       /* b<=VT^T * b */
       cbstatus=cublasDcopy(cbhandle, M, dhatkd, 1, ykd, 1);
       //status=culaDeviceDgemv('T',M,M,1.0,VTd,M,ykd,1,0.0,dhatkd,1);
       //checkStatus(status,__FILE__,__LINE__);
       cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,M,&cone,VTd,M,ykd,1,&czero,dhatkd,1);

       issolved=1;

     }
/********************************************************************/



  /* s_k<= d_k+ dhat_k */
  //my_dcopy(M,dk,1,sk,1);
  //my_daxpy(M,dhatk,1.0,sk);
  cbstatus=cublasDcopy(cbhandle, M, dkd, 1, skd, 1);
  alpha=1.0;
  cbstatus=cublasDaxpy(cbhandle, M, &alpha, dhatkd, 1, skd, 1);


  /* find norms */
  /* || F(y_k) || */
//  Fyknrm=my_dnrm2(N,Fyk);
  cbstatus=cublasDnrm2(cbhandle, N, Fykd, 1, &Fyknrm);
  Fyknrm=Fyknrm*Fyknrm;

  /* || F(y_k) + J_k dhat_k || */
  //my_dgemv('T',M,N,1.0,Jk,M,dhatk,1,0.0,Jkdk,1);
  //status=culaDeviceDgemv('T',M,N,1.0,Jkd,M,dhatkd,1,0.0,Jkdkd,1);
  cone=1.0; czero=0.0;
  cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,N,&cone,Jkd,M,dhatkd,1,&czero,Jkdkd,1);
 

  /* Fyk <= Fyk+ J_k dhat_k */
//  my_daxpy(N,Jkdk,1.0,Fyk);
//  Fykdhatknrm=my_dnrm2(N,Fyk);
  cbstatus=cublasDaxpy(cbhandle, N, &alpha, Jkdkd, 1, Fykd, 1);
  cbstatus=cublasDnrm2(cbhandle, N, Fykd, 1, &Fykdhatknrm);
  Fykdhatknrm=Fykdhatknrm*Fykdhatknrm;

  /* ||F(x_k+d_k+dhat_k)|| == ||F(x_k+s_k)|| */
  /* y_k<= x_k+ s_k */
  //my_dcopy(M,p,1,yk,1);
  //my_daxpy(M,sk,1.0,yk);
  cbstatus=cublasDcopy(cbhandle, M, pd, 1, ykd, 1);
  cbstatus=cublasDaxpy(cbhandle, M, &alpha, skd, 1, ykd, 1);

  //(*func)(yk, Fyk, M, N, adata);
  cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, ykd,Fykd,M,N, cohd, bbd, Nbase, dp->M, dp->N);

  /* func() - data */
  //my_daxpy(N, x, -1.0, Fyk);
  alpha=-1.0;
  cbstatus=cublasDaxpy(cbhandle, N, &alpha, xd, 1, Fykd, 1);

  //Fxksknrm=my_dnrm2(N,Fyk);
  cbstatus=cublasDnrm2(cbhandle, N, Fykd, 1, &Fxksknrm);

  Fxksknrm2=Fxksknrm*Fxksknrm;

  /* || Fxk + J_k d_k || */
  /* J d_k : since J is row major, transpose */
//  my_dgemv('T',M,N,1.0,Jk,M,dk,1,0.0,Jkdk,1);
  //status=culaDeviceDgemv('T',M,N,1.0,Jkd,M,dkd,1,0.0,Jkdkd,1);
  cone=1.0; czero=0.0;
  cbstatus=cublasDgemv(cbhandle,CUBLAS_OP_T,M,N,&cone,Jkd,M,dkd,1,&czero,Jkdkd,1);
 

  /* Fxk <= Fxk+ J_k d_k or, J_k d_k <= Fxk+ J_k d_k */
  //my_daxpy(N,Fxk,1.0,Jkdk);
  //FJkdknrm=my_dnrm2(N,Jkdk);
  alpha=1.0;
  cbstatus=cublasDaxpy(cbhandle, N, &alpha, Fxkd, 1, Jkdkd, 1);
  cbstatus=cublasDnrm2(cbhandle, N, Jkdkd, 1, &FJkdknrm);


  FJkdknrm=FJkdknrm*FJkdknrm;

  /* find ratio */
  Ak=Fxknrm2-Fxksknrm2;
  Pk=Fxknrm2-FJkdknrm+Fyknrm-Fykdhatknrm;
  /* if Pk<epsilon or rk<epsilon, also stop */
  if (fabs(Pk)<epsilon) {
#ifdef DEBUG
   printf("stopping 2 at iter %d\n",niter);
#endif
   break; 
  }
  rk=Ak/Pk;


  if (rk>=p0) {
    p_update=1;
    /* update p<= p+sk */
    //my_daxpy(M,sk,1.0,p);
    alpha=1.0;
    cbstatus=cublasDaxpy(cbhandle, M, &alpha, skd, 1, pd, 1);
    /* also update auxiliary info */
    /* Fxk <= Fyk */
    //my_dcopy(N,Fyk,1,Fxk,1);
    cbstatus=cublasDcopy(cbhandle, N, Fykd, 1, Fxkd, 1);

    Fxknrm=Fxksknrm;
    /* new Jk needed */
  } else { /* else no p update */
    p_update=0;
    /* use previous Jk, Fxk, JkTJk, JkTe */
  }
  if (rk<p1) {
   mu=4.0*mu; 
  } else if (rk<p2) {
   /* no update */
  } else {
   if (m>0.25*mu) {
    mu=m;
   } else {
    mu=0.25*mu;
   }
  }

#ifdef DEBUG
  printf("Ak=%lf Pk=%lf rk=%lf mu=%lf ||Fxk||=%lf\n",Ak,Pk,rk,mu,Fxknrm);
#endif
   niter++;
  }

  /* copy back solution */
  //err=cudaMemcpy(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost);
  err=cudaMemcpyAsync(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost,0);
  checkCudaError(err,__FILE__,__LINE__);

  /* check once CUBLAS error */
  checkCublasError(cbstatus,__FILE__,__LINE__);


  if (!gWORK) {
  if (solve_axb==1) {
   cudaFree(taud);
  } else if (solve_axb==2) {
    cudaFree(Ud);
    cudaFree(VTd);
    cudaFree(Sd);
  }
  cudaFree(xd);
  cudaFree(Jkd);
  cudaFree(Fxkd);
  cudaFree(Fykd);
  cudaFree(Jkdkd);
  cudaFree(JkTed);
  cudaFree(JkTed0);
  cudaFree(JkTJkd);
  cudaFree(JkTJkd0);
  cudaFree(dkd);
  cudaFree(dhatkd);
  cudaFree(ykd);
  cudaFree(skd);
  cudaFree(pd);

  cudaFree(bbd);
  cudaFree(cohd);
  }

  cudaFree(devInfo);
  cudaFree(work);
  if (solve_axb==2) {
   cudaFree(rwork);
  }

  if(info){
    info[0]=init_Fxknrm;
    info[1]=Fxknrm;
  }
  /* synchronize async operations */
  cudaDeviceSynchronize();
  return 0;
}
