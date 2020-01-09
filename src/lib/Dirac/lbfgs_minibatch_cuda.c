/*
 *
 Copyright (C) 2019 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>


//#define DEBUG
//#define CUDA_DEBUG
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

/************************ pipeline **************************/
/* data struct shared by all threads */
typedef struct gb_data_b_ {
  int status[2]; /* 0: do nothing, 
              1: allocate GPU  memory, attach GPU
              2: free GPU memory, detach GPU 
              3,4..: do work on GPU 
              99: reset GPU memory (memest all memory) */
  thread_gpu_data *lmdata[2]; /* two for each thread */

  /* GPU related info */
  cublasHandle_t cbhandle[2]; /* CUBLAS handles */
  cusolverDnHandle_t solver_handle[2]; /* solver handles */
  double *gWORK[2]; /* GPU buffers */
  int64_t data_size[2]; /* size of buffer (bytes), size gradient vector has different lengths, will be different for each thread */
  /* different pointers to GPU data */
  double *cxo[2]; /* data vector */
  double *ccoh[2]; /* coherency vector */
  double  *cpp[2]; /* parameter vector */
  double *cgrad[2]; /* gradient vector */
  short *cbb[2]; /* baseline map */
  int *cptoclus[2]; /* param to cluster map */

  /* for cost calculation */
  int Nbase[2];
  int boff[2];
  double fcost[2];

  /* for robust LBFGS */
  int do_robust;
} gbdata_b;

/* slave thread 2GPU function */
static void *
pipeline_slave_code_b(void *data)
{
 cudaError_t err;

 slave_tdata *td=(slave_tdata*)data;
 gbdata_b *dp=(gbdata_b*)(td->pline->data);
 int tid=td->tid;
 int Nbase=(dp->lmdata[tid]->Nbase)*(dp->lmdata[tid]->tilesz);
 int M=dp->lmdata[tid]->M;
 int N=dp->lmdata[tid]->N;
 int Nparam=(dp->lmdata[tid]->g_end-dp->lmdata[tid]->g_start+1);
 int m=dp->lmdata[tid]->m;

 while(1) {
  sync_barrier(&(td->pline->gate1)); /* stop at gate 1*/
  if(td->pline->terminate) break; /* if flag is set, break loop */
  sync_barrier(&(td->pline->gate2)); /* stop at gate 2 */
  /* do work */
  if (dp->status[tid]==PT_DO_CDERIV) {
    /* copy the current solution to device */
    err=cudaMemcpy(dp->cpp[tid], dp->lmdata[tid]->p, m*sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);
    if (!dp->do_robust) {
     cudakernel_lbfgs_r(dp->lmdata[tid]->ThreadsPerBlock, dp->lmdata[tid]->BlocksPerGrid, Nbase, dp->lmdata[tid]->tilesz, M, N, Nparam, dp->lmdata[tid]->g_start, dp->cxo[tid], dp->ccoh[tid], dp->cpp[tid], dp->cbb[tid], dp->cptoclus[tid], dp->cgrad[tid]);
    } else {
     cudakernel_lbfgs_r_robust(dp->lmdata[tid]->ThreadsPerBlock, dp->lmdata[tid]->BlocksPerGrid, Nbase, dp->lmdata[tid]->tilesz, M, N, Nparam, dp->lmdata[tid]->g_start, dp->cxo[tid], dp->ccoh[tid], dp->cpp[tid], dp->cbb[tid], dp->cptoclus[tid], dp->cgrad[tid],dp->lmdata[tid]->robust_nu);
    }
    /* read back the result */
    err=cudaMemcpy(&(dp->lmdata[tid]->g[dp->lmdata[tid]->g_start]), dp->cgrad[tid], Nparam*sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaError(err,__FILE__,__LINE__);
  } else if (dp->status[tid]==PT_DO_CCOST) {
   /* divide total baselines by 2 */
   int BlocksPerGrid=(dp->Nbase[tid]+dp->lmdata[tid]->ThreadsPerBlock-1)/dp->lmdata[tid]->ThreadsPerBlock;
   int  boff=dp->boff[tid];
   /* copy the current solution to device */
   err=cudaMemcpy(dp->cpp[tid], dp->lmdata[tid]->p, m*sizeof(double), cudaMemcpyHostToDevice);
   checkCudaError(err,__FILE__,__LINE__);
   if (!dp->do_robust) {
    dp->fcost[tid]=cudakernel_lbfgs_cost(dp->lmdata[tid]->ThreadsPerBlock, BlocksPerGrid, dp->Nbase[tid], boff, M, N, Nbase, &dp->cxo[tid][8*boff], &dp->ccoh[tid][boff*8*M], dp->cpp[tid], &dp->cbb[tid][boff*2], dp->cptoclus[tid]); 
   } else {
    dp->fcost[tid]=cudakernel_lbfgs_cost_robust(dp->lmdata[tid]->ThreadsPerBlock, BlocksPerGrid, dp->Nbase[tid], boff, M, N, Nbase, &dp->cxo[tid][8*boff], &dp->ccoh[tid][boff*8*M], dp->cpp[tid], &dp->cbb[tid][boff*2], dp->cptoclus[tid], dp->lmdata[tid]->robust_nu);
   }
  } else if (dp->status[tid]==PT_DO_AGPU) {
    attach_gpu_to_thread1(select_work_gpu(MAX_GPU_ID,td->pline->thst),&dp->cbhandle[tid],&dp->solver_handle[tid],&dp->gWORK[tid],dp->data_size[tid]);
    err=cudaMalloc((void**)&(dp->cxo[tid]),dp->lmdata[tid]->n*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&(dp->ccoh[tid]),Nbase*8*M*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&(dp->cpp[tid]),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&(dp->cgrad[tid]),Nparam*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&(dp->cptoclus[tid]),M*2*sizeof(int));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&(dp->cbb[tid]),Nbase*2*sizeof(short));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemcpy(dp->cxo[tid], dp->lmdata[tid]->xo, dp->lmdata[tid]->n*sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemcpy(dp->ccoh[tid], dp->lmdata[tid]->coh, Nbase*8*M*sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemcpy(dp->cptoclus[tid], dp->lmdata[tid]->ptoclus, M*2*sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemcpy(dp->cbb[tid], dp->lmdata[tid]->hbb, Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);

  } else if (dp->status[tid]==PT_DO_DGPU) {
    cudaFree(dp->cxo[tid]);
    cudaFree(dp->ccoh[tid]);
    cudaFree(dp->cptoclus[tid]);
    cudaFree(dp->cbb[tid]);
    cudaFree(dp->cpp[tid]);
    cudaFree(dp->cgrad[tid]);

    detach_gpu_from_thread1(dp->cbhandle[tid],dp->solver_handle[tid],dp->gWORK[tid]);
  }

 }
 return NULL;
}

/* initialize the pipeline
  and start the slaves rolling */
static void
init_pipeline_b(th_pipeline *pline,
     void *data)
{
 slave_tdata *t0,*t1;
 pthread_attr_init(&(pline->attr));
 pthread_attr_setdetachstate(&(pline->attr),PTHREAD_CREATE_JOINABLE);

 init_th_barrier(&(pline->gate1),3); /* 3 threads, including master */
 init_th_barrier(&(pline->gate2),3); /* 3 threads, including master */
 pline->terminate=0;
 pline->data=data; /* data should have pointers to t1 and t2 */

 if ((t0=(slave_tdata*)malloc(sizeof(slave_tdata)))==0) {
    fprintf(stderr,"no free memory\n");
    exit(1);
 }
 if ((t1=(slave_tdata*)malloc(sizeof(slave_tdata)))==0) {
    fprintf(stderr,"no free memory\n");
    exit(1);
 }
 if ((pline->thst=(taskhist*)malloc(sizeof(taskhist)))==0) {
    fprintf(stderr,"no free memory\n");
    exit(1);
 }

 init_task_hist(pline->thst);
 t0->pline=t1->pline=pline;
 t0->tid=0;
 t1->tid=1; /* link back t1, t2 to data so they could be freed */
 pline->sd0=t0;
 pline->sd1=t1;
 pthread_create(&(pline->slave0),&(pline->attr),pipeline_slave_code_b,(void*)t0);
 pthread_create(&(pline->slave1),&(pline->attr),pipeline_slave_code_b,(void*)t1);
}


/* destroy the pipeline */
/* need to kill the slaves first */
static void
destroy_pipeline_b(th_pipeline *pline)
{
 pline->terminate=1;
 sync_barrier(&(pline->gate1));
 pthread_join(pline->slave0,NULL);
 pthread_join(pline->slave1,NULL);
 destroy_th_barrier(&(pline->gate1));
 destroy_th_barrier(&(pline->gate2));
 pthread_attr_destroy(&(pline->attr));
 destroy_task_hist(pline->thst);
 free(pline->thst);
 free(pline->sd0);
 free(pline->sd1);
 pline->data=NULL;
}
/************************ end pipeline **************************/

/* use algorithm 9.1 to compute pk=Hk gk */
/* pk,gk: size m x 1
   s, y: size mM x 1 
   rho: size M x 1 
   ii: true location of the k th values in s,y */
static void
mult_hessian(int m, double *pk, double *gk, double *s, double *y, double *rho, int M, int ii) {
 int ci;
 double *alphai;
 int *idx; /* store sorted locations of s, y here */
 double gamma,beta;

 if ((alphai=(double*)calloc((size_t)M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((idx=(int*)calloc((size_t)M,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if (M>0) {
  /* find the location of k-1 th value */
  if (ii>0) {
   ii=ii-1;
  } else {
   ii=M-1;
  }
 /* s,y will have 0,1,...,ii,ii+1,...M-1 */
 /* map this to  ii+1,ii+2,...,M-1,0,1,..,ii */
  for (ci=0; ci<M-ii-1; ci++){
   idx[ci]=(ii+ci+1);
  }
  for(ci=M-ii-1; ci<M; ci++) {
   idx[ci]=(ci-M+ii+1);
  }
 }

#ifdef DEBUG
 printf("prod M=%d, current ii=%d\n",M,ii);
 for(ci=0; ci<M; ci++) {
  printf("%d->%d ",ci,idx[ci]);
 }
 printf("\n");
#endif
 /* q = grad(f)k : pk<=gk */
 my_dcopy(m,gk,1,pk,1);
 /* this should be done in the right order */
 for (ci=0; ci<M; ci++) {
  /* alphai=rhoi si^T*q */
  alphai[M-ci-1]=rho[idx[M-ci-1]]*my_ddot(m,&s[m*idx[M-ci-1]],pk);
  /* q=q-alphai yi */
  my_daxpy(m,&y[m*idx[M-ci-1]],-alphai[M-ci-1],pk);
 }
 /* r=Hk(0) q : initial hessian */
 /* gamma=s(k-1)^T*y(k-1)/y(k-1)^T*y(k-1)*/
 gamma=1.0;
 if (M>0) {
  gamma=my_ddot(m,&s[m*idx[M-1]],&y[m*idx[M-1]]);
  gamma/=my_ddot(m,&y[m*idx[M-1]],&y[m*idx[M-1]]);
  /* Hk(0)=gamma I, so scale q by gamma */
  /* r= Hk(0) q */
  my_dscal(m,gamma,pk);
 } 

 for (ci=0; ci<M; ci++) {
  /* beta=rhoi yi^T * r */
  beta=rho[idx[ci]]*my_ddot(m,&y[m*idx[ci]],pk);
  /* r = r + (alphai-beta)*si */
  my_daxpy(m,&s[m*idx[ci]],alphai[ci]-beta,pk);
 }

 free(alphai);
 free(idx);
}

/* pk,gk,s,y are on the device, rho on the host */
static void
cuda_mult_hessian(int m, double *pk, double *gk, double *s, double *y, double *rho, cublasHandle_t *cbhandle, int M, int ii) {
 int ci;
 double *alphai;
 int *idx; /* store sorted locations of s, y here */
 double gamma,beta;

 cudaError_t err;
 cublasStatus_t cbstatus;

 if ((alphai=(double*)calloc((size_t)M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((idx=(int*)calloc((size_t)M,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if (M>0) {
  /* find the location of k-1 th value */
  if (ii>0) {
   ii=ii-1;
  } else {
   ii=M-1;
  }
 /* s,y will have 0,1,...,ii,ii+1,...M-1 */
 /* map this to  ii+1,ii+2,...,M-1,0,1,..,ii */
  for (ci=0; ci<M-ii-1; ci++){
   idx[ci]=(ii+ci+1);
  }
  for(ci=M-ii-1; ci<M; ci++) {
   idx[ci]=(ci-M+ii+1);
  }
 }

#ifdef DEBUG
 printf("prod M=%d, current ii=%d\n",M,ii);
 for(ci=0; ci<M; ci++) {
  printf("%d->%d ",ci,idx[ci]);
 }
 printf("\n");
#endif
 /* q = grad(f)k : pk<=gk */
 ///my_dcopy(m,gk,1,pk,1);
 err=cudaMemcpy(pk, gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
 checkCudaError(err,__FILE__,__LINE__);

 /* this should be done in the right order */
 for (ci=0; ci<M; ci++) {
  /* alphai=rhoi si^T*q */
  ///alphai[M-ci-1]=rho[idx[M-ci-1]]*my_ddot(m,&s[m*idx[M-ci-1]],pk);
  cbstatus=cublasDdot(*cbhandle,m,&s[m*idx[M-ci-1]],1,pk,1,&alphai[M-ci-1]);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  alphai[M-ci-1]*=rho[idx[M-ci-1]];

  /* q=q-alphai yi */
  ///my_daxpy(m,&y[m*idx[M-ci-1]],-alphai[M-ci-1],pk);
  double tmpi=-alphai[M-ci-1];
  cbstatus=cublasDaxpy(*cbhandle,m,&tmpi,&y[m*idx[M-ci-1]],1,pk,1);
  checkCublasError(cbstatus,__FILE__,__LINE__);
 }
 /* r=Hk(0) q : initial hessian */
 /* gamma=s(k-1)^T*y(k-1)/y(k-1)^T*y(k-1)*/
 gamma=1.0;
 if (M>0) {
  ///gamma=my_ddot(m,&s[m*idx[M-1]],&y[m*idx[M-1]]);
  cbstatus=cublasDdot(*cbhandle,m,&s[m*idx[M-1]],1,&y[m*idx[M-1]],1,&gamma);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  ///gamma/=my_ddot(m,&y[m*idx[M-1]],&y[m*idx[M-1]]);
  double gamma1;
  //cbstatus=cublasDdot(*cbhandle,m,&y[m*idx[M-1]],1,&y[m*idx[M-1]],1,&gamma1);
  cbstatus=cublasDnrm2(*cbhandle,m,&y[m*idx[M-1]],1,&gamma1);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  gamma/=(gamma1*gamma1);
  /* Hk(0)=gamma I, so scale q by gamma */
  /* r= Hk(0) q */
  ///my_dscal(m,gamma,pk);
  cbstatus=cublasDscal(*cbhandle,m,&gamma,pk,1);
  checkCublasError(cbstatus,__FILE__,__LINE__);
 } 

 for (ci=0; ci<M; ci++) {
  /* beta=rhoi yi^T * r */
  ///beta=rho[idx[ci]]*my_ddot(m,&y[m*idx[ci]],pk);
  cbstatus=cublasDdot(*cbhandle,m,&y[m*idx[ci]],1,pk,1,&beta);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  beta*=rho[idx[ci]];
  /* r = r + (alphai-beta)*si */
  ///my_daxpy(m,&s[m*idx[ci]],alphai[ci]-beta,pk);
  double tmpi=alphai[ci]-beta;
  cbstatus=cublasDaxpy(*cbhandle,m,&tmpi,&s[m*idx[ci]],1,pk,1);
  checkCublasError(cbstatus,__FILE__,__LINE__);
 }

 free(alphai);
 free(idx);
}


/* cubic interpolation in interval [a,b] (a>b is possible)
   to find step that minimizes cost function */
/*
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   a/b:  interval for interpolation
   x: size n x 1 (storage)
   xp: size m x 1 (storage)
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
static double 
cubic_interp(
   double *xk, double *pk, double a, double b, double *x, double *xp,  double *xo, int m, int n, double step, void *adata,  th_pipeline *tp, gbdata_b *tpg) {

  double f0,f1,f0d,f1d; /* function values and derivatives at a,b */
  double p01,p02,z0,fz0;
  double aa,cc;

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,a,xp); /* xp<=xp+(a)*pk */
  sync_barrier(&(tp->gate1));
  tpg->lmdata[0]->p=tpg->lmdata[1]->p=xp;
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  f0=tpg->fcost[0]+tpg->fcost[1];
  sync_barrier(&(tp->gate2));

  /* grad(phi_0): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(a+step)*pk */
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  p01=tpg->fcost[0]+tpg->fcost[1];
  sync_barrier(&(tp->gate2));


  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(a-step)*pk */
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  p02=tpg->fcost[0]+tpg->fcost[1];
  sync_barrier(&(tp->gate2));


  f0d=(p01-p02)/(2.0*step);

  my_daxpy(m,pk,-a+step+b,xp); /* xp<=xp+(b)*pk */
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  f1=tpg->fcost[0]+tpg->fcost[1];
  sync_barrier(&(tp->gate2));


  /* grad(phi_1): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(b+step)*pk */
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  p01=tpg->fcost[0]+tpg->fcost[1];
  sync_barrier(&(tp->gate2));


  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(b-step)*pk */
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  p02=tpg->fcost[0]+tpg->fcost[1];
  sync_barrier(&(tp->gate2));


  f1d=(p01-p02)/(2.0*step);


  //printf("Interp a,f(a),f'(a): (%lf,%lf,%lf) (%lf,%lf,%lf)\n",a,f0,f0d,b,f1,f1d);
  /* cubic poly in [0,1] is f0+f0d z+eta z^2+xi z^3 
    where eta=3(f1-f0)-2f0d-f1d, xi=f0d+f1d-2(f1-f0) 
    derivative f0d+2 eta z+3 xi z^2 => cc+bb z+aa z^2 */
   aa=3.0*(f0-f1)/(b-a)+(f1d-f0d);
   p01=aa*aa-f0d*f1d;
  /* root exist? */
  if (p01>0.0) {
   /* root */
   cc=sqrt(p01);
   z0=b-(f1d+cc-aa)*(b-a)/(f1d-f0d+2.0*cc);
   /* FIXME: check if this is within boundary */
   aa=MAX(a,b);
   cc=MIN(a,b);
   //printf("Root=%lf, in [%lf,%lf]\n",z0,cc,aa);
   if (z0>aa || z0<cc) {
    fz0=f0+f1;
   } else {
    /* evaluate function for this root */
    my_daxpy(m,pk,-b+step+a+z0*(b-a),xp); /* xp<=xp+(a+z0(b-a))*pk */
    sync_barrier(&(tp->gate1));
    tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
    sync_barrier(&(tp->gate2));
    sync_barrier(&(tp->gate1));
    tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
    fz0=tpg->fcost[0]+tpg->fcost[1];
    sync_barrier(&(tp->gate2));

   }
   //printf("Val=%lf, [%lf,%lf]\n",fz0,f0,f1);

   /* now choose between f0,f1,fz0,fz1 */
   if (f0<f1 && f0<fz0) {
     return a;
   }
   if (f1<fz0) {
     return b;
   }
   /* else */
   return (z0);
  } else { 

   /* find the value from a or b that minimizes func */
   if (f0<f1) {
    return a;
   } else {
    return b;
   }
  }

  /* fallback value */
  return (a+b)*0.5;
}


/*************** Fletcher line search **********************************/
/* zoom function for line search */
/* 
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   a/b: bracket interval [a,b] (a>b) is possible
   x: size n x 1 (storage)
   xp: size m x 1 (storage)
   phi_0: phi(0)
   gphi_0: grad(phi(0))
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
static double 
linesearch_zoom(
   double *xk, double *pk, double a, double b, double *x, double *xp,  double phi_0, double gphi_0, double sigma, double rho, double t1, double t2, double t3, double *xo, int m, int n, double step, void *adata,  th_pipeline *tp, gbdata_b *tpg) {

  double alphaj,phi_j,phi_aj;
  double gphi_j,p01,p02,aj,bj;
  double alphak=1.0;
  int ci,found_step=0;

  aj=a;
  bj=b;
  ci=0;
  while(ci<10) {
    /* choose alphaj from [a+t2(b-a),b-t3(b-a)] */
    p01=aj+t2*(bj-aj);
    p02=bj-t3*(bj-aj);
    alphaj=cubic_interp(xk,pk,p01,p02,x,xp,xo,m,n,step,adata,tp,tpg);
    //printf("cubic intep [%lf,%lf]->%lf\n",p01,p02,alphaj);

    /* evaluate phi(alphaj) */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,alphaj,xp); /* xp<=xp+(alphaj)*pk */
    sync_barrier(&(tp->gate1));
    tpg->lmdata[0]->p=tpg->lmdata[1]->p=xp;
    tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
    sync_barrier(&(tp->gate2));
    sync_barrier(&(tp->gate1));
    tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
    phi_j=tpg->fcost[0]+tpg->fcost[1];
    sync_barrier(&(tp->gate2));


    /* evaluate phi(aj) */
    my_daxpy(m,pk,-alphaj+aj,xp); /* xp<=xp+(aj)*pk */
    sync_barrier(&(tp->gate1));
    tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
    sync_barrier(&(tp->gate2));
    sync_barrier(&(tp->gate1));
    tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
    phi_aj=tpg->fcost[0]+tpg->fcost[1];
    sync_barrier(&(tp->gate2));


    if ((phi_j>phi_0+rho*alphaj*gphi_0) || phi_j>=phi_aj) {
      bj=alphaj; /* aj unchanged */
    } else {
     /* evaluate grad(alphaj) */
     my_daxpy(m,pk,-aj+alphaj+step,xp); /* xp<=xp+(alphaj+step)*pk */
     sync_barrier(&(tp->gate1));
     tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
     sync_barrier(&(tp->gate2));
     sync_barrier(&(tp->gate1));
     tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
     p01=tpg->fcost[0]+tpg->fcost[1];
     sync_barrier(&(tp->gate2));


     my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphaj-step)*pk */
     sync_barrier(&(tp->gate1));
     tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
     sync_barrier(&(tp->gate2));
     sync_barrier(&(tp->gate1));
     tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
     p02=tpg->fcost[0]+tpg->fcost[1];
     sync_barrier(&(tp->gate2));


     gphi_j=(p01-p02)/(2.0*step);

     /* termination due to roundoff/other errors pp. 38, Fletcher */
     if ((aj-alphaj)*gphi_j<=step) {
      alphak=alphaj;
      found_step=1;
      break;
     }
    
     if (fabs(gphi_j)<=-sigma*gphi_0) {
      alphak=alphaj;
      found_step=1;
      break;
     }
     
     if (gphi_j*(bj-aj)>=0) {
       bj=aj;
     } /* else bj unchanged */
     aj=alphaj;
   }
   ci++;
  }

  if (!found_step) {
   /* use bound to find possible step */
   alphak=alphaj;
  }
   
#ifdef DEBUG
  printf("Found %lf Interval [%lf,%lf]\n",alphak,a,b);
#endif
  return alphak;
}
 
 

/* line search */
/* 
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   alpha1: initial value for step
   sigma,rho,t1,t2,t3: line search parameters (from Fletcher) 
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
static double 
linesearch(
   double *xk, double *pk, double alpha1, double sigma, double rho, double t1, double t2, double t3, double *xo, int m, int n, double step, void *adata, th_pipeline *tp, gbdata_b *tpg) {
 /* phi(alpha)=f(xk+alpha pk)
  for vector function func 
   f(xk) =||func(xk)||^2 */
  
  double *x,*xp;
  double alphai,alphai1;
  double phi_0,phi_alphai,phi_alphai1;
  double p01,p02;
  double gphi_0,gphi_i;
  double alphak;

  double mu;
  double tol; /* lower limit for minimization, need to be just about min value of cost function */

  int ci;

  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xp=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  alphak=1.0;
  /* evaluate phi_0 and grad(phi_0) */
  sync_barrier(&(tp->gate1));
  tpg->lmdata[0]->p=tpg->lmdata[1]->p=xk;
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  phi_0=tpg->fcost[0]+tpg->fcost[1];
  sync_barrier(&(tp->gate2));
//printf("GPU cost=%lf\n",phi_0);
  /* select tolarance 1/100 of current function value */
  tol=MIN(0.01*phi_0,1e-6); 

  /* grad(phi_0): evaluate at -step and +step */
  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(0.0+step)*pk */

  sync_barrier(&(tp->gate1));
  tpg->lmdata[0]->p=tpg->lmdata[1]->p=xp;
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  p01=tpg->fcost[0]+tpg->fcost[1];
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp->gate2));

  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(0.0-step)*pk */
  sync_barrier(&(tp->gate1));
  tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
  sync_barrier(&(tp->gate2));
  sync_barrier(&(tp->gate1));
  p02=tpg->fcost[0]+tpg->fcost[1];
  tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp->gate2));


  gphi_0=(p01-p02)/(2.0*step);


  /* estimate for mu */
  /* mu = (tol-phi_0)/(rho gphi_0) */
  mu=(tol-phi_0)/(rho*gphi_0);
#ifdef DEBUG
  printf("cost=%lf grad=%lf mu=%lf, alpha1=%lf\n",phi_0,gphi_0,mu,alpha1);
#endif
  /* catch if not finite (deltaphi=0 or nan) */
  if (!isnormal(mu)) {
    free(x);
    free(xp);
#ifdef DEBUG
    printf("line interval too small\n");
#endif
    return mu;
  }


  ci=1;
  alphai=alpha1; /* initial value for alpha(i) : check if 0<alphai<=mu */
  alphai1=0.0; /* FIXME: tune for GPU (defalut is 0.0) */
  phi_alphai1=phi_0;
  while(ci<10) {
   /* evalualte phi(alpha(i))=f(xk+alphai pk) */
   my_dcopy(m,xk,1,xp,1); /* xp<=xk */
   my_daxpy(m,pk,alphai,xp); /* xp<=xp+alphai*pk */
   sync_barrier(&(tp->gate1));
   tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
   sync_barrier(&(tp->gate2));
   sync_barrier(&(tp->gate1));
   phi_alphai=tpg->fcost[0]+tpg->fcost[1];
   tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
   sync_barrier(&(tp->gate2));


   if (phi_alphai<tol) {
     alphak=alphai;
#ifdef DEBUG
     printf("Linesearch : Condition 0 met\n");
#endif
     break;
   }

   if ((phi_alphai>phi_0+alphai*gphi_0) || (ci>1 && phi_alphai>=phi_alphai1)) {
      /* ai=alphai1, bi=alphai bracket */
      alphak=linesearch_zoom(xk,pk,alphai1,alphai,x,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,xo,m,n,step,adata,tp,tpg);
#ifdef DEBUG
      printf("Linesearch : Condition 1 met\n");
#endif
      break;
   } 

   /* evaluate grad(phi(alpha(i))) */
   my_daxpy(m,pk,step,xp); /* xp<=xp+(alphai+step)*pk */
   sync_barrier(&(tp->gate1));
   tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
   sync_barrier(&(tp->gate2));
   sync_barrier(&(tp->gate1));
   p01=tpg->fcost[0]+tpg->fcost[1];
   tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
   sync_barrier(&(tp->gate2));

   my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphai-step)*pk */
   sync_barrier(&(tp->gate1));
   tpg->status[0]=tpg->status[1]=PT_DO_CCOST;
   sync_barrier(&(tp->gate2));
   sync_barrier(&(tp->gate1));
   p02=tpg->fcost[0]+tpg->fcost[1];
   tpg->status[0]=tpg->status[1]=PT_DO_NOTHING;
   sync_barrier(&(tp->gate2));


   gphi_i=(p01-p02)/(2.0*step);

   if (fabs(gphi_i)<=-sigma*gphi_0) {
     alphak=alphai;
#ifdef DEBUG
     printf("Linesearch : Condition 2 met\n");
#endif
     break;
   }

   if (gphi_i>=0) {
     /* ai=alphai, bi=alphai1 bracket */
     alphak=linesearch_zoom(xk,pk,alphai,alphai1,x,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,xo,m,n,step,adata,tp,tpg);
#ifdef DEBUG
     printf("Linesearch : Condition 3 met\n");
#endif
     break;
   }

   /* else preserve old values */
   if (mu<=(2.0*alphai-alphai1)) {
     /* next step */
     alphai1=alphai;
     alphai=mu;
   } else {
     /* choose by interpolation in [2*alphai-alphai1,min(mu,alphai+t1*(alphai-alphai1)] */
     p01=2.0*alphai-alphai1;
     p02=MIN(mu,alphai+t1*(alphai-alphai1));
     alphai=cubic_interp(xk,pk,p01,p02,x,xp,xo,m,n,step,adata,tp,tpg);
     //printf("cubic interp [%lf,%lf]->%lf\n",p01,p02,alphai);
   }
   phi_alphai1=phi_alphai;

   ci++;
  }



  free(x);
  free(xp);
#ifdef DEBUG
  printf("Step size=%lf\n",alphak);
#endif
  return alphak;
}
/*************** END Fletcher line search **********************************/


/* note M here  is LBFGS memory size */
static int
lbfgs_fit_common(
   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, int do_robust, void *adata) {

  double *gk; /* gradients at both k+1 and k iter */
  double *xk1,*xk; /* parameters at k+1 and k iter */
  double *pk; /* step direction H_k * grad(f) */

  double step; /* FIXME tune for GPU, use larger if far away from convergence */
  double *y, *s; /* storage for delta(grad) and delta(p) */
  double *rho; /* storage for 1/yk^T*sk */
  int ci,ck,cm;
  double alphak=1.0;
  

  me_data_t *dp=(me_data_t*)adata;
  short *hbb;
  int *ptoclus;
  int Nbase1=dp->Nbase*dp->tilesz;

  thread_gpu_data threaddata[2]; /* 2 for 2 threads/cards */

  if ((gk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xk1=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  if ((pk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }


  /* storage size mM x 1*/
  if ((s=(double*)calloc((size_t)m*M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((y=(double*)calloc((size_t)m*M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((rho=(double*)calloc((size_t)M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

/*********** following are not part of LBFGS, but done here only for GPU use */
  /* auxilliary arrays for GPU */
  if ((hbb=(short*)calloc((size_t)(Nbase1*2),sizeof(short)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* baseline->station mapping */
  rearrange_baselines(Nbase1, dp->barr, hbb, dp->Nt);

  /* parameter->cluster mapping */ 
  /* for each cluster: chunk size, start param index */
  if ((ptoclus=(int*)calloc((size_t)(2*dp->M),sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  for(ci=0; ci<dp->M; ci++) {
   ptoclus[2*ci]=dp->carr[ci].nchunk;
   ptoclus[2*ci+1]=dp->carr[ci].p[0]; /* so end at p[0]+nchunk*8*N-1 */
  }
  dp->hbb=hbb;
  dp->ptoclus=ptoclus;
/*****************************************************************************/
  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = gpu_threads;

  /* partition parameters, per each parameter, one thread */
  /* also account for the no of GPUs using */
  /* parameters per thread (GPU) */
  int Nparm=(m+2-1)/2;
  /* find number of blocks */
  int BlocksPerGrid =(Nparm+ThreadsPerBlock-1)/ThreadsPerBlock;
  ci=0;
  int nth;
  for (nth=0; nth<2; nth++) {
   threaddata[nth].ThreadsPerBlock=ThreadsPerBlock;
   threaddata[nth].BlocksPerGrid=BlocksPerGrid;
   threaddata[nth].card=nth;
   threaddata[nth].Nbase=dp->Nbase;
   threaddata[nth].tilesz=dp->tilesz;
   threaddata[nth].barr=dp->barr;
   threaddata[nth].M=dp->M;
   threaddata[nth].N=dp->N;
   threaddata[nth].coh=dp->coh;
   threaddata[nth].xo=x;
   threaddata[nth].p=p;
   threaddata[nth].g=gk;
   threaddata[nth].m=m;
   threaddata[nth].n=n;
   threaddata[nth].hbb=dp->hbb;
   threaddata[nth].ptoclus=dp->ptoclus;
   threaddata[nth].g_start=ci;
   threaddata[nth].g_end=ci+Nparm-1;
   if (threaddata[nth].g_end>=m) {
     threaddata[nth].g_end=m-1;
   }
   /* for robust mode */
   if (do_robust) { 
    threaddata[nth].robust_nu=dp->robust_nu;
   }
   ci=ci+Nparm;
  }
  
  /* pipeline data */
  th_pipeline tp;
  gbdata_b tpg; 

  tpg.do_robust=do_robust;
  /* divide no of baselines */
  int Nthb0=(Nbase1+2-1)/2;
  tpg.Nbase[0]=Nthb0;
  tpg.Nbase[1]=Nbase1-Nthb0;
  tpg.boff[0]=0;
  tpg.boff[1]=Nthb0;
  
  tpg.lmdata[0]=&threaddata[0];
  tpg.lmdata[1]=&threaddata[1];
  /* calculate total size of memory need to be allocated in GPU, in bytes +2 added to align memory */
  /* note: we do not allocate memory here, use pinned memory for transfer */
  //tpg.data_size[0]=(n+(dp->Nbase*dp->tilesz)*8*dp->M+m+(tpg.lmdata[0]->g_end-tpg.lmdata[0]->g_start+1)+2)*sizeof(double)+(2*dp->M*sizeof(int))+(2*dp->Nbase*dp->tilesz*sizeof(char));
  //tpg.data_size[1]=(n+(dp->Nbase*dp->tilesz)*8*dp->M+m+(tpg.lmdata[1]->g_end-tpg.lmdata[1]->g_start+1)+2)*sizeof(double)+(2*dp->M*sizeof(int))+(2*dp->Nbase*dp->tilesz*sizeof(char));
  tpg.data_size[0]=tpg.data_size[1]=sizeof(float);

  tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  init_pipeline_b(&tp,&tpg);
  sync_barrier(&(tp.gate1));
  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
  sync_barrier(&(tp.gate2));
  sync_barrier(&(tp.gate1));
  tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2));

/*****************************************************************************/
  /* initial value for params xk=p */
  my_dcopy(m,p,1,xk,1);
  sync_barrier(&(tp.gate1));
  threaddata[0].p=threaddata[1].p=xk;
  tpg.status[0]=tpg.status[1]=PT_DO_CDERIV;
  sync_barrier(&(tp.gate2));
  sync_barrier(&(tp.gate1));
  tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2));
  
  double gradnrm=my_dnrm2(m,gk);
  /* if gradient is too small, no need to solve, so stop */
  if (gradnrm<CLM_STOP_THRESH) {
   ck=itmax;
   step=0.0;
  } else {
   ck=0;
   /* step in [1e-6,1e-9] */
   step=MAX(1e-9,MIN(1e-3/gradnrm,1e-6));
  }
#ifdef DEBUG
  printf("||grad||=%g step=%g\n",gradnrm,step);
#endif

  cm=0;
  ci=0;
 
  while (ck<itmax && isnormal(gradnrm) && gradnrm>CLM_STOP_THRESH) {
   /* mult with hessian  pk=-H_k*gk */
   if (ck<M) {
    mult_hessian(m,pk,gk,s,y,rho,ck,ci);
   } else {
    mult_hessian(m,pk,gk,s,y,rho,M,ci);
   }
   my_dscal(m,-1.0,pk);

   /* linesearch to find step length */
   /* parameters alpha1=10.0,sigma=0.1, rho=0.01, t1=9, t2=0.1, t3=0.5 */
   /* FIXME: update paramters for GPU gradient */
   alphak=linesearch(xk,pk,10.0,0.1,0.01,9,0.1,0.5,x,m,n,step,adata,&tp, &tpg);
   /* check if step size is too small, or nan, then stop */
   if (!isnormal(alphak) || fabs(alphak)<CLM_EPSILON) {
    break;
   }

   /* update parameters xk1=xk+alpha_k *pk */
   my_dcopy(m,xk,1,xk1,1);
   my_daxpy(m,pk,alphak,xk1);
  
   /* calculate sk=xk1-xk and yk=gk1-gk */
   /* sk=xk1 */ 
   my_dcopy(m,xk1,1,&s[cm],1); 
   /* sk=sk-xk */
   my_daxpy(m,xk,-1.0,&s[cm]);
   /* yk=-gk */ 
   my_dcopy(m,gk,1,&y[cm],1); 
   my_dscal(m,-1.0,&y[cm]);

   /* update gradient */
  sync_barrier(&(tp.gate1));
  tpg.lmdata[0]->p=tpg.lmdata[1]->p=xk1;
  tpg.status[0]=tpg.status[1]=PT_DO_CDERIV;
  sync_barrier(&(tp.gate2));
  sync_barrier(&(tp.gate1));
  tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2));

   /* yk=yk+gk1 */
   my_daxpy(m,gk,1.0,&y[cm]);

   /* calculate 1/yk^T*sk */
   rho[ci]=1.0/my_ddot(m,&y[cm],&s[cm]);

   /* update xk=xk1 */
   my_dcopy(m,xk1,1,xk,1); 
  
   //printf("iter %d store %d\n",ck,cm);
   ck++;
   /* increment storage appropriately */
   if (cm<(M-1)*m) {
    /* offset of m */
    cm=cm+m;
    ci++;
   } else {
    cm=ci=0;
   }
  }


 /* copy back solution to p */
 my_dcopy(m,xk,1,p,1);

 /* for (ci=0; ci<m; ci++) {
   printf("grad %d=%lf\n",ci,gk[ci]);
  } */

  free(gk);
  free(xk1);
  free(xk);
  free(pk);
  free(s);
  free(y);
  free(rho);
  free(hbb);
  free(ptoclus);
  dp->hbb=NULL;
  dp->ptoclus=NULL;

  /******** free threads ***************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/

  destroy_pipeline_b(&tp);

  return 0;
}



/*****************************************************************************/
/* initialize persistant memory (allocated on the GPU) */
/* also attach to a GPU first */
/* user routines for setting up and clearing persistent data structure
   for using stochastic LBFGS */
int
lbfgs_persist_init(persistent_data_t *pt, int Nminibatch, int m, int n, int lbfgs_m, int Nt) {


    cudaError_t err;
    err=cudaMalloc((void**)&(pt->s),m*lbfgs_m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(pt->s,0,m*lbfgs_m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

    err=cudaMalloc((void**)&(pt->y),m*lbfgs_m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(pt->y,0,m*lbfgs_m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

    /* Note that rho is on the host */
    if ((pt->rho=(double*)calloc((size_t)lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    pt->m=m;
    pt->lbfgs_m=lbfgs_m;

  /* storage for calculating on-line variance of gradient */
    err=cudaMalloc((void**)&(pt->running_avg),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(pt->running_avg,0,m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);


    err=cudaMalloc((void**)&(pt->running_avg_sq),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(pt->running_avg_sq,0,m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);



  pt->nfilled=0; /* always 0 when we start */
  pt->vacant=0; /* cycle in 0..M-1 */
  pt->niter=0; /* cumulative iteration count */
  pt->Nt=Nt; /* no. of threads need to be passed */
  pt->cbhandle=0;
  pt->solver_handle=0;

  return 0;
}


int
lbfgs_persist_clear(persistent_data_t *pt) {
  /* free persistent memory */
    cudaError_t err;
    err=cudaFree(pt->s);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(pt->y);
    checkCudaError(err,__FILE__,__LINE__);

    free(pt->rho);

    err=cudaFree(pt->running_avg);
    checkCudaError(err,__FILE__,__LINE__);

    err=cudaFree(pt->running_avg_sq);
    checkCudaError(err,__FILE__,__LINE__);

  return 0;
}

int
lbfgs_persist_reset(persistent_data_t *pt) {

    cudaError_t err;
    err=cudaMemset(pt->s,0,pt->m*pt->lbfgs_m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(pt->y,0,pt->m*pt->lbfgs_m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

  memset(pt->rho,0,sizeof(double)*(size_t)pt->lbfgs_m);

    err=cudaMemset(pt->running_avg,0,pt->m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(pt->running_avg_sq,0,pt->m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);


  pt->nfilled=0; /* always 0 when we start */
  pt->vacant=0; /* cycle in 0..m-1 */
  pt->niter=0; /* cumulative iteration count */

  return 0;
}

/*************** backtracking line search **********************************/
/* func: cost function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   gk: gradient vector size m x 1
   m: size or parameter vector
   alpha0: initial alpha
   adata:  additional data passed to the function
   xk,pk,gk are on the device
*/
static double
cuda_linesearch_backtrack(
   double (*func)(double *p, int m, void *adata),
   double *xk, double *pk, double *gk, int m, cublasHandle_t *cbhandle, double alpha0, void *adata) {

    cudaError_t err;
    cublasStatus_t cbstatus;

  /* Armijo condition  f(x+alpha p) <= f(x) + c alpha p^T grad(f(x)) */
  const double c=1e-4;
  double alphak=alpha0;
  double *xk1,fnew,fold,product;
    err=cudaMalloc((void**)&(xk1),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(xk1,0,m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

  /* update parameters xk1=xk+alpha_k *pk */
  ///my_dcopy(m,xk,1,xk1,1);
    err=cudaMemcpy(xk1, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCudaError(err,__FILE__,__LINE__);

  ///my_daxpy(m,pk,alphak,xk1);
    cbstatus=cublasDaxpy(*cbhandle,m,&alphak,pk,1,xk1,1);
    checkCublasError(cbstatus,__FILE__,__LINE__);

  fnew=func(xk1,m,adata);
  fold=func(xk,m,adata); /* add threshold to make iterations stop at some point FIXME: is this correct/needed? */
///  product=c*my_ddot(m,pk,gk);
  cbstatus=cublasDdot(*cbhandle,m,gk,1,pk,1,&product);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  product *=c;

  int ci=0;
  while (ci<15 && fnew>fold+alphak*product) { /* FIXME: using higher iterations here gives worse results */
     alphak *=0.5;
///     my_dcopy(m,xk,1,xk1,1);
    err=cudaMemcpy(xk1, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCudaError(err,__FILE__,__LINE__);


///     my_daxpy(m,pk,alphak,xk1);
    cbstatus=cublasDaxpy(*cbhandle,m,&alphak,pk,1,xk1,1);
    checkCublasError(cbstatus,__FILE__,__LINE__);

     fnew=func(xk1,m,adata);
     ci++;
  }

     err=cudaFree(xk1);
     checkCudaError(err,__FILE__,__LINE__);

  return alphak;
}

/*****************************************************************************/

/* LBFGS routine,
 * user has to give cost_func() and grad_func()
 * both p and g should be device pointers, use cudaPointerAttributes() to check
 * indata (persistent_data_t *) should be initialized beforehand
 */
int
lbfgs_fit_cuda(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   /* adata: user supplied data,
   indata: persistant data that need to be kept between batches */
   /* p:mx1 vector, M: memory size */
   double *p, int m, int itmax, int M, void *adata, persistent_data_t *indata) { /* indata=NULL for full batch */

  double *gk; /* gradients at both k+1 and k iter */
  double *xk1,*xk; /* parameters at k+1 and k iter */
  double *pk; /* step direction H_k * grad(f) */

  double *g_min_rold, *g_min_rnew; /* temp storage for updating running averages */

  double *y, *s; /* storage for delta(grad) and delta(p) */
  double *rho; /* storage for 1/yk^T*sk */
  int ci,ck,cm;
  double alphak=1.0;
  double alphabar=1.0; 
  double alpha;


    cudaError_t err;
    cublasStatus_t cbstatus;

    err=cudaMalloc((void**)&(gk),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(gk,0,m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

    err=cudaMalloc((void**)&(xk1),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(xk1,0,m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

    err=cudaMalloc((void**)&(xk),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(xk,0,m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

    err=cudaMalloc((void**)&(pk),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemset(pk,0,m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);



  /* use y,s pairs from the previous run */
  /* storage size mM x 1*/
  s=indata->s;
  y=indata->y;
  rho=indata->rho;
  if (!s || !y || !rho) {
     fprintf(stderr,"%s: %d: storage must be pre allocated befor calling this function.\n",__FILE__,__LINE__);
     exit(1);
  }

  /* find if p is on host or device */
  struct cudaPointerAttributes attributes;
  err=cudaPointerGetAttributes(&attributes,(void*)p);
  checkCudaError(err,__FILE__,__LINE__);
  int p_on_device=(attributes.devicePointer!=NULL?1:0);
  /* initial value for params xk=p */
 /// my_dcopy(m,p,1,xk,1);
  if (p_on_device) {
    err=cudaMemcpy(xk, p, m*sizeof(double), cudaMemcpyDeviceToDevice);
  } else {
    err=cudaMemcpy(xk, p, m*sizeof(double), cudaMemcpyHostToDevice);
  }
    checkCudaError(err,__FILE__,__LINE__);

  /*  gradient gk=grad(f)_k */
  grad_func(xk,gk,m,adata);
  ///double gradnrm=my_dnrm2(m,gk);
  double gradnrm;
  cbstatus=cublasDnrm2(*(indata->cbhandle),m,gk,1,&gradnrm);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  /* if gradient is too small, no need to solve, so stop */
  if (gradnrm<CLM_STOP_THRESH) {
   ck=itmax;
  } else {
   ck=0;
  }
#ifdef DEBUG
  printf("||grad||=%g\n",gradnrm);
#endif

  ci=indata->vacant; /* cycle in 0..(M-1) */
  cm=m*ci; /* cycle in 0..(M-1)m (in strides of m)*/

    while (ck<itmax && isnormal(gradnrm) && gradnrm>CLM_STOP_THRESH) {
#ifdef DEBUG
   printf("iter %d gradnrm %g\n",ck,gradnrm);
#endif
   /* increment global iteration count */
   indata->niter++;
   /* detect if we are at first iteration of a new batch */
   int batch_changed=(indata->niter>1 && ck==0);
   /* if the batch has changed, update running averages */
   if (batch_changed) {
     /* temp vectors : grad-running_avg(old) , grad - running_avg(new) */
     /* running_avg_new = running_avg_old + (grad-running_avg(old))/niter */
    err=cudaMalloc((void**)&(g_min_rold),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);

    err=cudaMalloc((void**)&(g_min_rnew),m*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);


///     my_dcopy(m,gk,1,g_min_rold,1); /* g_min_rold <- grad */
    err=cudaMemcpy(g_min_rold, gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCudaError(err,__FILE__,__LINE__);


///     my_daxpy(m,indata->running_avg,-1.0,g_min_rold); /* g_min_rold <- g_min_rold - running_avg(old) */
     alpha=-1.0;
     cbstatus=cublasDaxpy(*(indata->cbhandle),m,&alpha,indata->running_avg,1,g_min_rold,1);
     checkCublasError(cbstatus,__FILE__,__LINE__);

///     my_daxpy(m,g_min_rold,1.0/(double)indata->niter,indata->running_avg); /* running_avg <- running_avg + 1/niter . g_min_rold */
     alpha=1.0/(double)indata->niter;
     cbstatus=cublasDaxpy(*(indata->cbhandle),m,&alpha,g_min_rold,1,indata->running_avg,1);
     checkCublasError(cbstatus,__FILE__,__LINE__);

///     my_dcopy(m,gk,1,g_min_rnew,1);
     err=cudaMemcpy(g_min_rnew, gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
     checkCudaError(err,__FILE__,__LINE__);

///     my_daxpy(m,indata->running_avg,-1.0,g_min_rnew); /* g_min_rnew <- g_min_rnew - running_avg(new) */
     alpha=-1.0;
     cbstatus=cublasDaxpy(*(indata->cbhandle),m,&alpha,indata->running_avg,1,g_min_rnew,1);
     checkCublasError(cbstatus,__FILE__,__LINE__);


     /* this loop should be parallelized/vectorized */
     /*for (it=0; it<m; it++) {
       indata->running_avg_sq[it] += g_min_rold[it]*g_min_rnew[it];
     }*/
     int ThreadsPerBlock = 256;
     cudakernel_hadamard_sum(ThreadsPerBlock,(m+ThreadsPerBlock-1)/ThreadsPerBlock,m,indata->running_avg_sq,g_min_rold,g_min_rnew);

     /* estimate online variance
       Note: for badly initialized cases, might need to increase initial value of alphabar
       because of gradnrm is too large, alphabar becomes too small */
///     alphabar=10.0/(1.0+my_dasum(m,indata->running_avg_sq)/((double)(indata->niter-1)*gradnrm));
     cbstatus=cublasDasum(*(indata->cbhandle),m,indata->running_avg_sq,1,&alpha);
     checkCublasError(cbstatus,__FILE__,__LINE__);
     alphabar=10.0/(1.0+alpha/((double)(indata->niter-1)*gradnrm));
#ifdef DEBUG
     printf("iter=%d running_avg %lf gradnrm %lf alpha=%lf\n",indata->niter,alpha,gradnrm,alphabar);
#endif
     err=cudaFree(g_min_rold);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(g_min_rnew);
     checkCudaError(err,__FILE__,__LINE__);
   }

   /* mult with hessian  pk=-H_k*gk */
   if (indata->nfilled<M) {
    cuda_mult_hessian(m,pk,gk,s,y,rho,indata->cbhandle,indata->nfilled,ci);
   } else {
    cuda_mult_hessian(m,pk,gk,s,y,rho,indata->cbhandle,M,ci);
   }
///   my_dscal(m,-1.0,pk);
   alpha=-1.0;
   cbstatus=cublasDscal(*(indata->cbhandle),m,&alpha,pk,1);
   checkCublasError(cbstatus,__FILE__,__LINE__);


   /* linesearch to find step length */
   /* Armijo line search */
   alphak=cuda_linesearch_backtrack(cost_func,xk,pk,gk,m,indata->cbhandle,alphabar,adata);
   /* check if step size is too small, or nan, then stop */
   if (!isnormal(alphak) || fabs(alphak)<CLM_EPSILON) {
    break;
   }
   /* update parameters xk1=xk+alpha_k *pk */
///   my_dcopy(m,xk,1,xk1,1);
    err=cudaMemcpy(xk1, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCudaError(err,__FILE__,__LINE__);

///   my_daxpy(m,pk,alphak,xk1);
    cbstatus=cublasDaxpy(*(indata->cbhandle),m,&alphak,pk,1,xk1,1);
    checkCublasError(cbstatus,__FILE__,__LINE__);

   if (!batch_changed) {
   /* calculate sk=xk1-xk and yk=gk1-gk */
   /* sk=xk1 */
///   my_dcopy(m,xk1,1,&s[cm],1);
    err=cudaMemcpy(&s[cm], xk1, m*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCudaError(err,__FILE__,__LINE__);

   /* sk=sk-xk */
///   my_daxpy(m,xk,-1.0,&s[cm]);
    alpha=-1.0;
    cbstatus=cublasDaxpy(*(indata->cbhandle),m,&alpha,xk,1,&s[cm],1);
    checkCublasError(cbstatus,__FILE__,__LINE__);
 
   /* yk=-gk */
///   my_dcopy(m,gk,1,&y[cm],1);
    err=cudaMemcpy(&y[cm], gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCudaError(err,__FILE__,__LINE__);

///    my_dscal(m,-1.0,&y[cm]);
    cbstatus=cublasDscal(*(indata->cbhandle),m,&alpha,&y[cm],1);
    checkCublasError(cbstatus,__FILE__,__LINE__);
   }

   grad_func(xk1,gk,m,adata);
///   gradnrm=my_dnrm2(m,gk);
  cbstatus=cublasDnrm2(*(indata->cbhandle),m,gk,1,&gradnrm);
  checkCublasError(cbstatus,__FILE__,__LINE__);

   /* do a sanity check here */
   if (!isnormal(gradnrm) || gradnrm<CLM_STOP_THRESH) {
     break;
   }

   if (!batch_changed) {
   /* yk=yk+gk1 */
///   my_daxpy(m,gk,1.0,&y[cm]);
    alpha=1.0;
    cbstatus=cublasDaxpy(*(indata->cbhandle),m,&alpha,gk,1,&y[cm],1);
    checkCublasError(cbstatus,__FILE__,__LINE__);
 

   /* yk = yk + lm0* sk, to create a trust region */
   double lm0=1e-6;
   if (gradnrm>1e3*lm0) {
///    my_daxpy(m,&s[cm],lm0,&y[cm]);
    cbstatus=cublasDaxpy(*(indata->cbhandle),m,&lm0,&s[cm],1,&y[cm],1);
    checkCublasError(cbstatus,__FILE__,__LINE__);
   }

   /* calculate 1/yk^T*sk */
///   rho[ci]=1.0/my_ddot(m,&y[cm],&s[cm]);
     cbstatus=cublasDdot(*(indata->cbhandle),m,&y[cm],1,&s[cm],1,&rho[ci]);
     checkCublasError(cbstatus,__FILE__,__LINE__);
     rho[ci]=1.0/rho[ci];
   }

   /* update xk=xk1 */
///   my_dcopy(m,xk1,1,xk,1);
    err=cudaMemcpy(xk, xk1, m*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCudaError(err,__FILE__,__LINE__);

   //printf("iter %d store %d\n",ck,cm);
   ck++;

     if (!batch_changed) {
   indata->nfilled=(indata->nfilled<M?indata->nfilled+1:M);
   /* increment storage appropriately */
   if (cm<(M-1)*m) {
    /* offset of m */
    cm=cm+m;
    ci++;
    indata->vacant++;
   } else {
    cm=ci=0;
    indata->vacant=0;
   }
   }

#ifdef DEBUG
  printf("iter %d alpha=%g ||grad||=%g\n",ck,alphak,gradnrm);
#endif
  }


 /* copy back solution to p */
/// my_dcopy(m,xk,1,p,1);

  if (p_on_device) {
    err=cudaMemcpy(p, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
  } else {
    err=cudaMemcpy(p, xk, m*sizeof(double), cudaMemcpyDeviceToHost);
  }
    checkCudaError(err,__FILE__,__LINE__);

#ifdef DEBUG
//  for (ci=0; ci<m; ci++) {
//   printf("grad %d=%lf\n",ci,gk[ci]);
//  }
#endif

    err=cudaFree(gk);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(xk1);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(xk);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(pk);
    checkCudaError(err,__FILE__,__LINE__);

  return 0;
}
