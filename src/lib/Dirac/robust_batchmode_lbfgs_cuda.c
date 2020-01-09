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
#include <pthread.h>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

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


typedef struct me_data_batchmode_cuda_t_ {
 /* note: all arrays are on the device */
 double *x; /* full data vector nx1 */
 double *coh; /* coherency vector */
 int n;
 int Nbase;
 int tilesz;
 int N;
 int M;
 int Mt;
 int Nchan;
 double robust_nu;
 int nminibatch; /* which minibatch? 0...(totalminibatches-1) */
 int totalminibatch; /* total number of minibatches */
 short *hbb;
 int *ptoclus;

 /* for consensus optimization */
 double *y; /* lagrange multiplier, size equal to p */
 double *z; /* Bz polynomial constraint, size equal to p */
 double *rho; /* regularization, Mt values */
 /* for CUBLAS */
 cublasHandle_t *cbhandle;

} me_data_batchmode_cuda_t;

/* cost function */
static double
costfunc_multifreq(double *p, int m, void *adata) {
 me_data_batchmode_cuda_t *lmdata=(me_data_batchmode_cuda_t *)adata;

 int Nbase=lmdata->tilesz*lmdata->Nbase;
 /* calculate the absolute baseline offset (in the full batch data) 
   using which minibatch and total number of minibatches */
 int boff=lmdata->nminibatch*(Nbase);
 /* the total number of baselines over the full batch */
 int Nbasetotal=lmdata->totalminibatch*(Nbase);
 double fcost=cudakernel_lbfgs_multifreq_cost_robust(Nbase,lmdata->Nchan,lmdata->M,lmdata->N,Nbasetotal,boff,lmdata->x,lmdata->coh,p,m,lmdata->hbb,lmdata->ptoclus,lmdata->robust_nu);

 if (!lmdata->rho && !lmdata->y && !lmdata->z) {
   return fcost;
 } 

 /* else add regularization cost */
 if (m!=lmdata->Mt*lmdata->N*8) {
  fprintf(stderr,"paramter vector dimentions %d does not match %d\n",m,lmdata->Mt*lmdata->N*8);
 }
 /* extra cost  y^T (x-z) + rho/2 (x-z)^T(x-z) */
 double *xp;
 cudaError_t err;
 cublasStatus_t cbstatus;
 err=cudaMalloc((void**)&(xp),m*sizeof(double));
 checkCudaError(err,__FILE__,__LINE__);

 /* find xp=p-z */
 err=cudaMemcpy(xp,p,m*sizeof(double),cudaMemcpyDeviceToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 double tmpi=-1.0;
 cbstatus=cublasDaxpy(*(lmdata->cbhandle),m,&tmpi,lmdata->z,1,xp,1);
 checkCublasError(cbstatus,__FILE__,__LINE__);

 double f1=0.0;
 int ci;
 for (ci=0; ci<lmdata->Mt; ci++) {
  cbstatus=cublasDdot(*(lmdata->cbhandle),8*lmdata->N,&lmdata->y[8*lmdata->N*ci],1,&xp[8*lmdata->N*ci],1,&tmpi);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  f1+=tmpi;

  //cbstatus=cublasDdot(*(lmdata->cbhandle),8*lmdata->N,&xp[8*lmdata->N*ci],1,&xp[8*lmdata->N*ci],1,&tmpi);
  cbstatus=cublasDnrm2(*(lmdata->cbhandle),8*lmdata->N,&xp[8*lmdata->N*ci],1,&tmpi);
  checkCublasError(cbstatus,__FILE__,__LINE__);
  f1+=0.5*lmdata->rho[ci]*tmpi*tmpi; /* need ||x||^2 */
 }

 err=cudaFree(xp);
 checkCudaError(err,__FILE__,__LINE__);

 return fcost+f1;
}

/* gradient function */
static void
gradfunc_multifreq(double *p, double *g, int m, void *adata) {
 me_data_batchmode_cuda_t *lmdata=(me_data_batchmode_cuda_t *)adata;

 int Nbase=lmdata->tilesz*lmdata->Nbase;
 /* calculate the absolute baseline offset (in the full batch data) 
   using which minibatch and total number of minibatches */
 int boff=lmdata->nminibatch*(Nbase);
 /* the total number of baselines over the full batch */
 int Nbasetotal=lmdata->totalminibatch*(Nbase);
 cudakernel_lbfgs_multifreq_r_robust(Nbase,lmdata->tilesz,lmdata->Nchan,lmdata->M,lmdata->N,Nbasetotal,boff,lmdata->x,lmdata->coh,p,m,lmdata->hbb,lmdata->ptoclus,g,lmdata->robust_nu);

 if (lmdata->rho && lmdata->y && lmdata->z) {
 if (m!=lmdata->Mt*lmdata->N*8) {
  fprintf(stderr,"paramter vector dimentions %d does not match %d\n",m,lmdata->Mt*lmdata->N*8);
 }
 /* add regularization grad */
 double *xp;
 cudaError_t err;
 cublasStatus_t cbstatus;
 err=cudaMalloc((void**)&(xp),m*sizeof(double));
 checkCudaError(err,__FILE__,__LINE__);

 /* find xp=p-z */
 err=cudaMemcpy(xp,p,m*sizeof(double),cudaMemcpyDeviceToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 double tmpi=-1.0;
 cbstatus=cublasDaxpy(*(lmdata->cbhandle),m,&tmpi,lmdata->z,1,xp,1);
 checkCublasError(cbstatus,__FILE__,__LINE__);
 /* now multiply by rho */
 int ci;
 for (ci=0; ci<lmdata->Mt; ci++) {
  //  my_dscal(8*dp->N,rho[ci],&xp[8*dp->N*ci]);
  cbstatus=cublasDscal(*(lmdata->cbhandle),8*lmdata->N,&lmdata->rho[ci],&xp[8*lmdata->N*ci],1);
  checkCublasError(cbstatus,__FILE__,__LINE__);
 }

   /* now add y + rho(p-z) to g (-ve for -ve g)*/
 //  my_daxpy(m,y,-1.0,g);
 //  my_daxpy(m,xp,-1.0,g);
   cbstatus=cublasDaxpy(*(lmdata->cbhandle),m,&tmpi,lmdata->y,1,g,1);
   checkCublasError(cbstatus,__FILE__,__LINE__);
   cbstatus=cublasDaxpy(*(lmdata->cbhandle),m,&tmpi,xp,1,g,1);
   checkCublasError(cbstatus,__FILE__,__LINE__);

 err=cudaFree(xp);
 checkCudaError(err,__FILE__,__LINE__);
 }

}



int
bfgsfit_minibatch_visibilities(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz, short *hbb, int *ptoclus, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {

  me_data_batchmode_cuda_t lmdata;
  cudaError_t err;
  double *pdevice;

  /*  no. of true parameters */
  int m=N*Mt*8;
  /* no of data */
  int n=Nbase*tilesz*Nf*8;

  err=cudaMalloc((void**)&(pdevice),m*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  lmdata.n=n;
  /* copy all necerrary data to the GPU, and only pass pointers to
     the data as a struct to the cost and grad functions :lbfgs_cuda.c 140*/
  err=cudaMalloc((void**)&(lmdata.x),n*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.x,x,n*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* setup the necessary ME data */
  lmdata.Nbase=Nbase;
  lmdata.tilesz=tilesz;
  lmdata.N=N;
  lmdata.M=M;
  lmdata.Mt=Mt;
  lmdata.nminibatch=nminibatch;
  lmdata.totalminibatch=totalminibatch;
  err=cudaMalloc((void**)&(lmdata.coh),8*Nbase*tilesz*M*Nf*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.coh,(double*)coh,8*Nbase*tilesz*M*Nf*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  lmdata.Nchan=Nf; /* multichannel data */
  /* fixed robust nu */
  lmdata.robust_nu=robust_nu;

  /* GPU replacement for barr */
  err=cudaMalloc((void**)&(lmdata.hbb),2*Nbase*tilesz*sizeof(short));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.hbb,hbb,2*Nbase*tilesz*sizeof(short),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* GPU replacement for carr */
  err=cudaMalloc((void**)&(lmdata.ptoclus),2*M*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.ptoclus,ptoclus,2*M*sizeof(int),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* rho,y, and z are Null */
  lmdata.rho=0;
  lmdata.y=0;
  lmdata.z=0;

  /* call lbfgs_fit_cuda() with proper cost/grad functions */
  *res_0=costfunc_multifreq(pdevice,m,&lmdata);
  lbfgs_fit_cuda(costfunc_multifreq,gradfunc_multifreq,pdevice,m,max_lbfgs,lbfgs_m,&lmdata,indata);
  *res_1=costfunc_multifreq(pdevice,m,&lmdata);

  err=cudaMemcpy(p,pdevice,m*sizeof(double),cudaMemcpyDeviceToHost);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaFree(pdevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.x);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.coh);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.hbb);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.ptoclus);
  checkCudaError(err,__FILE__,__LINE__);

  double invn=(double)1.0/n;
  *res_0 *=invn;
  *res_1 *=invn;

  return 0;
}




/* minibatch mode with consensus */
/* baseline_t *barr replaced by short *hbb :size 2*Nbase*tilesz x 1
   clus_source_t *carr replaced by int *ptoclus : size 2*M x 1 */
int
bfgsfit_minibatch_consensus(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz, short *hbb, int *ptoclus, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, double *y, double *z, double *rho, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {

  me_data_batchmode_cuda_t lmdata;
  cudaError_t err;
  double *pdevice;

  /*  no. of true parameters */
  int m=N*Mt*8;
  /* no of data */
  int n=Nbase*tilesz*Nf*8;

  err=cudaMalloc((void**)&(pdevice),m*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  lmdata.n=n;
  /* copy all necerrary data to the GPU, and only pass pointers to
     the data as a struct to the cost and grad functions :lbfgs_cuda.c 140*/
  err=cudaMalloc((void**)&(lmdata.x),n*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.x,x,n*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* setup the necessary ME data */
  lmdata.Nbase=Nbase;
  lmdata.tilesz=tilesz;
  lmdata.N=N;
  lmdata.M=M;
  lmdata.Mt=Mt;
  lmdata.nminibatch=nminibatch;
  lmdata.totalminibatch=totalminibatch;
  lmdata.cbhandle=indata->cbhandle;

  err=cudaMalloc((void**)&(lmdata.coh),8*Nbase*tilesz*M*Nf*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.coh,(double*)coh,8*Nbase*tilesz*M*Nf*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  lmdata.Nchan=Nf; /* multichannel data */
  /* fixed robust nu */
  lmdata.robust_nu=robust_nu;

  /* GPU replacement for barr */
  err=cudaMalloc((void**)&(lmdata.hbb),2*Nbase*tilesz*sizeof(short));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.hbb,hbb,2*Nbase*tilesz*sizeof(short),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* GPU replacement for carr */
  err=cudaMalloc((void**)&(lmdata.ptoclus),2*M*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.ptoclus,ptoclus,2*M*sizeof(int),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  lmdata.rho=rho; /* note : this is on host */

  err=cudaMalloc((void**)&(lmdata.y),m*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.y,y,m*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaMalloc((void**)&(lmdata.z),m*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(lmdata.z,z,m*sizeof(double),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

/****************************************/
/* check gradient */
/* int iii=2;
 double *tmpgrad,*tmphost;
 err=cudaMalloc((void**)&(tmpgrad),m*sizeof(double));
 err=cudaHostAlloc((void**)&(tmphost),m*sizeof(double),cudaHostAllocDefault);
 gradfunc_multifreq(pdevice,tmpgrad,m,&lmdata);
 err=cudaMemcpy(tmphost,tmpgrad,m*sizeof(double),cudaMemcpyDeviceToHost);
 checkCudaError(err,__FILE__,__LINE__);
 double p0=p[iii]; double eps=1e-6;
 p[iii]=p0+eps;
 err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 double f00=costfunc_multifreq(pdevice,m,&lmdata);
 p[iii]=p0-eps;
 err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
 checkCudaError(err,__FILE__,__LINE__);
 double f11=costfunc_multifreq(pdevice,m,&lmdata);
 printf("Numerical grad =%lf,%lf=%lf analytical=%lf\n",f00,f11,(f00-f11)/(2.0*eps),tmphost[iii]);
 cudaFree(tmpgrad);
 cudaFreeHost(tmphost);
*/
/****************************************/


  /* call lbfgs_fit_cuda() with proper cost/grad functions */
  *res_0=costfunc_multifreq(pdevice,m,&lmdata);
  lbfgs_fit_cuda(costfunc_multifreq,gradfunc_multifreq,pdevice,m,max_lbfgs,lbfgs_m,&lmdata,indata);
  *res_1=costfunc_multifreq(pdevice,m,&lmdata);

  err=cudaMemcpy(p,pdevice,m*sizeof(double),cudaMemcpyDeviceToHost);
  checkCudaError(err,__FILE__,__LINE__);


  err=cudaFree(pdevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.x);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.coh);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.hbb);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.ptoclus);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.y);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(lmdata.z);
  checkCudaError(err,__FILE__,__LINE__);

  double invn=(double)1.0/n;
  *res_0 *=invn;
  *res_1 *=invn;

  return 0;
}
