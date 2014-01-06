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


#include "sagecal.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>

//#define DEBUG
/* worker thread for a gpu */
static void *
cuda_calc_deriv(void *adata) {
  thread_gpu_data *dp=(thread_gpu_data*)adata;

  int devid=dp->card;
  cudaError_t ret;

  /* this thread will calculate derivative for
     parameters g_start to g_end
     Nparam: g_end-gstart+1  */
  /* GPU needs to store:
    cxo: residual vector size Nbase*8 : equal to n
   ccoh: coherency Nbase*8*M
   cpp: parameter vector size  m (full parameter set)
   cbb: baseline -> station vector : size 2*Nbase
   output: cgrad : gradient size Nparam
  */
   
  /* pointers to gpu device arrays */
  double *cxo, *ccoh, *cpp, *cgrad;
  char *cbb;
  int *cptoclus;


  int Nbase=(dp->Nbase)*(dp->tilesz);
  int M=(dp->M);
  int N=(dp->N);
  int Nparam=(dp->g_end-dp->g_start+1);
#ifdef DEBUG
  size_t cmem_total,cmem_free;
#endif

  /* number of device in current "context" */
  if((ret=cudaSetDevice(devid))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: cannot init card %d\n",__FILE__,__LINE__,devid);
   exit(1);
  }

  /* we need to wait for this device to finish work */
/*  if((ret=cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: cannot init card %d\n",__FILE__,__LINE__,devid);
   exit(1);
  } */


  /* allocate arrays on device (via CUDA) */
  if((ret=cudaMalloc((void**) &cxo, dp->n*sizeof(double)))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if((ret=cudaMalloc((void**) &ccoh, Nbase*8*M*sizeof(double)))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if((ret=cudaMalloc((void**) &cpp, (dp->m)*sizeof(double)))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if((ret=cudaMalloc((void**) &cbb, Nbase*2*sizeof(char)))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if((ret=cudaMalloc((void**) &cgrad, Nparam*sizeof(double)))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if((ret=cudaMalloc((void**) &cptoclus, 2*M*sizeof(int)))!=cudaSuccess) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
  }


  /* copy from host array to device array */
  cudaMemcpy(cxo, dp->xo, (dp->n)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(ccoh, dp->coh, Nbase*8*M*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cbb, dp->hbb, Nbase*2*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(cptoclus, dp->ptoclus, M*2*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cpp, dp->p, (dp->m)*sizeof(double), cudaMemcpyHostToDevice);


#ifdef DEBUG
  cudaMemGetInfo(&cmem_free,&cmem_total);
  printf("GPU %d memory %lf %% free\n",devid,(double)cmem_free/(double)cmem_total*100.0);
#endif


/*    stc=(ci%(8*N))/8;
    stoff=(ci%(8*N))%8;
    stm=ci/(8*N); */
    /* invoke kernel on device */
  cudakernel_lbfgs(dp->ThreadsPerBlock, dp->BlocksPerGrid, Nbase, dp->tilesz, M, N, Nparam, dp->g_start, cxo, ccoh, cpp, cbb, cptoclus, cgrad);
  /* read back the result */
  ret=cudaMemcpy(&(dp->g[dp->g_start]), cgrad, Nparam*sizeof(double), cudaMemcpyDeviceToHost);
  if(ret!=cudaSuccess) {
     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(ret),__FILE__,__LINE__);
     exit(1);
  }


/*  int ci;
  for (ci=0; ci<Nparam; ci++) {
    printf("%d = %lf\n",ci,dp->g[dp->g_start+ci]);
  } */

  cudaFree((void*)cxo);
  cudaFree((void*)ccoh);
  cudaFree((void*)cpp);
  cudaFree((void*)cbb);
  cudaFree((void*)cptoclus);
  cudaFree((void*)cgrad);

  return NULL;
}

/* calculate gradient */
/* func: vector function
   p: parameter values size m x 1 (at which grad is calculated)
   g: gradient size m x 1 
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   gpu_threads: GPU threads per block
   adata:  additional data passed to the function
*/
static int
func_grad(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *g, double *xo, int m, int n, double step, int gpu_threads, void *adata) {
  /* gradient for each parameter is
     (||func(p+step*e_i)-x||^2-||func(p-step*e_i)-x||^2)/2*step
    i=0,...,m-1 for all parameters
    e_i: unit vector, 1 only at i-th location
  */

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_gpu_data *threaddata;
  /* no of threads equal to no of GUPs, so ==1 means one GPU */
  /* FIXME: for COMA=1 */
#ifdef ONE_GPU
  int Nt=1;
#endif /* ONE_GPU */
#ifndef ONE_GPU
  int Nt=2;
#endif /* !ONE_GPU */

  int ci,nth,Nparm;
  double *x; /* residual */
  me_data_t *dp=(me_data_t*)adata;
  

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_gpu_data*)malloc((size_t)Nt*sizeof(thread_gpu_data)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* evaluate func once, store in x, and create threads */
  /* and calculate the residual x=xo-func */
  func(p,x,m,n,adata);
  /* calculate x<=x-xo */
  my_daxpy(n,xo,-1.0,x);


  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = gpu_threads;

  /* partition parameters, per each parameter, one thread */
  /* also account for the no of GPUs using */
  /* parameters per thread (GPU) */
  Nparm=(m+Nt-1)/Nt;
  /* find number of blocks */
  int BlocksPerGrid = (Nparm+ThreadsPerBlock-1)/ThreadsPerBlock;

#ifdef DEBUG
  printf("total parms =%d, per thread=%d\n",m,Nparm);
#endif

  /* iterate over threads */
  ci=0;
  for (nth=0;  nth<Nt; nth++) {
   threaddata[nth].ThreadsPerBlock=ThreadsPerBlock;
   threaddata[nth].BlocksPerGrid=BlocksPerGrid;
   threaddata[nth].card=nth;
   threaddata[nth].Nbase=dp->Nbase;
   threaddata[nth].tilesz=dp->tilesz;
   threaddata[nth].barr=dp->barr;
   threaddata[nth].M=dp->M;
   threaddata[nth].N=dp->N;
   threaddata[nth].coh=dp->coh;
   threaddata[nth].m=m;
   threaddata[nth].n=n;
   threaddata[nth].xo=x;
   threaddata[nth].p=p;
   threaddata[nth].g=g;
   threaddata[nth].hbb=dp->hbb;
   threaddata[nth].ptoclus=dp->ptoclus;
   threaddata[nth].g_start=ci;
   threaddata[nth].g_end=ci+Nparm-1;
   if (threaddata[nth].g_end>=m) {
    threaddata[nth].g_end=m-1;
   }
   ci=ci+Nparm;
#ifdef DEBUG
   printf("thread %d parms (%d-%d)\n",nth,threaddata[nth].g_start,threaddata[nth].g_end);
#endif
   pthread_create(&th_array[nth],&attr,cuda_calc_deriv,(void*)(&threaddata[nth]));
  }

  /* now wait for threads to finish */
  for(nth=0; nth<Nt; nth++) {
   pthread_join(th_array[nth],NULL);
  }

  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);
  free(x);

  return 0;

}

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
 if ((idx=(int*)calloc((size_t)M,sizeof(double)))==0) {
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

/* cubic interpolation in interval [a,b] (a>b is possible)
   to find step that minimizes cost function */
/* func: vector function
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
cubic_interp1(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double a, double b, double *x, double *xp,  double *xo, int m, int n, double step, void *adata) {

  double f0,f1,f0d,f1d; /* function values and derivatives at a,b */
  double p01,p02,z0,z1,fz0,fz1;
  double aa,bb,cc;

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,a,xp); /* xp<=xp+(a)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  f0=my_dnrm2(n,x);
  f0*=f0;
  /* grad(phi_0): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(a+step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p01=my_dnrm2(n,x);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(a-step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p02=my_dnrm2(n,x);
  f0d=(p01*p01-p02*p02)/(2.0*step);

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,b,xp); /* xp<=xp+(b)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  f1=my_dnrm2(n,x);
  f1*=f1;
  /* grad(phi_0): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(b+step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p01=my_dnrm2(n,x);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(b-step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p02=my_dnrm2(n,x);
  f1d=(p01*p01-p02*p02)/(2.0*step);

  //printf("Interp a,f(a),f'(a): (%lf,%lf,%lf) (%lf,%lf,%lf)\n",a,f0,f0d,b,f1,f1d);
  /* cubic poly in [0,1] is f0+f0d z+eta z^2+xi z^3 
    where eta=3(f1-f0)-2f0d-f1d, xi=f0d+f1d-2(f1-f0) 
    derivative f0d+2 eta z+3 xi z^2 => cc+bb z+aa z^2 */
   aa=3.0*(f0d+f1d-2.0*(f1-f0));
   bb=2*(3.0*(f1-f0)-2.0*f0d-f1d);
   cc=f0d;
  
  /* root exist? */
  p01=bb*bb-4.0*aa*cc;
  if (p01>0.0) {
   /* roots */
   p01=sqrt(p01);
   z0=(-bb+p01)/(2.0*aa);
   z1=(-bb-p01)/(2.0*aa);
   /* check if any root is within [0,1] */
   if (z0>=0.0 && z0<=1.0) {
    /* evaluate function for this root */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,a+z0*(b-a),xp); /* xp<=xp+(z0)*pk */
    func(xp,x,m,n,adata);
    my_daxpy(n,xo,-1.0,x);
    fz0=my_dnrm2(n,x);
    fz0*=fz0;
   } else {
    fz0=1e9;
   }
   if (z1>=0.0 && z1<=1.0) {
    /* evaluate function for this root */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,a+z1*(b-a),xp); /* xp<=xp+(z1)*pk */
    func(xp,x,m,n,adata);
    my_daxpy(n,xo,-1.0,x);
    fz1=my_dnrm2(n,x);
    fz1*=fz1;
   } else {
    fz1=1e9;
   }

   /* now choose between f0,f1,fz0,fz1 */
   if (f0<f1 && f0<fz0 && f0<fz1) {
     return a;
   }
   if (f1<fz0 && f1<fz1) {
     return b;
   }
   if (fz0<fz1) {
     return (a+z0*(b-a));
   }
   /* else */
   return (a+z1*(b-a));
  } else { 
   /* find the value from a or b that minimizes func */
   if (f0<f1) {
    return a;
   } else {
    return b;
   }
  }
  return 0;
}
static double 
cubic_interp(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double a, double b, double *x, double *xp,  double *xo, int m, int n, double step, void *adata) {

  double f0,f1,f0d,f1d; /* function values and derivatives at a,b */
  double p01,p02,z0,fz0;
  double aa,cc;

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,a,xp); /* xp<=xp+(a)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  f0=my_dnrm2(n,x);
  f0*=f0;
  /* grad(phi_0): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(a+step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p01=my_dnrm2(n,x);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(a-step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p02=my_dnrm2(n,x);
  f0d=(p01*p01-p02*p02)/(2.0*step);

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,b,xp); /* xp<=xp+(b)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  f1=my_dnrm2(n,x);
  f1*=f1;
  /* grad(phi_1): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(b+step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p01=my_dnrm2(n,x);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(b-step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p02=my_dnrm2(n,x);
  f1d=(p01*p01-p02*p02)/(2.0*step);


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
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,a+z0*(b-a),xp); /* xp<=xp+(z0)*pk */
    func(xp,x,m,n,adata);
    my_daxpy(n,xo,-1.0,x);
    fz0=my_dnrm2(n,x);
    fz0*=fz0;
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

  return 0;
}





/*************** Fletcher line search **********************************/
/* zoom function for line search */
/* func: vector function
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
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double a, double b, double *x, double *xp,  double phi_0, double gphi_0, double sigma, double rho, double t1, double t2, double t3, double *xo, int m, int n, double step, void *adata) {

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
    alphaj=cubic_interp(func,xk,pk,p01,p02,x,xp,xo,m,n,step,adata);
    //printf("cubic intep [%lf,%lf]->%lf\n",p01,p02,alphaj);

    /* evaluate phi(alphaj) */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,alphaj,xp); /* xp<=xp+(alphaj)*pk */
    func(xp,x,m,n,adata);
    /* calculate x<=x-xo */
    my_daxpy(n,xo,-1.0,x);
    phi_j=my_dnrm2(n,x);
    phi_j*=phi_j;

    /* evaluate phi(aj) */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,aj,xp); /* xp<=xp+(alphaj)*pk */
    func(xp,x,m,n,adata);
    /* calculate x<=x-xo */
    my_daxpy(n,xo,-1.0,x);
    phi_aj=my_dnrm2(n,x);
    phi_aj*=phi_aj;


    if ((phi_j>phi_0+rho*alphaj*gphi_0) || phi_j>=phi_aj) {
      bj=alphaj; /* aj unchanged */
    } else {
     /* evaluate grad(alphaj) */
     my_dcopy(m,xk,1,xp,1); /* xp<=xk */
     my_daxpy(m,pk,alphaj+step,xp); /* xp<=xp+(alphaj+step)*pk */
     func(xp,x,m,n,adata);
     /* calculate x<=x-xo */
     my_daxpy(n,xo,-1.0,x);
     p01=my_dnrm2(n,x);
     my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphaj-step)*pk */
     func(xp,x,m,n,adata);
     /* calculate x<=x-xo */
     my_daxpy(n,xo,-1.0,x);
     p02=my_dnrm2(n,x);
     gphi_j=(p01*p01-p02*p02)/(2.0*step);

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
/* func: vector function
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
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double alpha1, double sigma, double rho, double t1, double t2, double t3, double *xo, int m, int n, double step, void *adata) {
 
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
  double tol=step; /* lower limit for minimization */

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
  func(xk,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  phi_0=my_dnrm2(n,x);
  phi_0*=phi_0;
  /* grad(phi_0): evaluate at -step and +step */
  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(0.0+step)*pk */
  func(xp,x,m,n,adata);
  /* calculate x<=x-xo */
  my_daxpy(n,xo,-1.0,x);
  p01=my_dnrm2(n,x);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(0.0-step)*pk */
  func(xp,x,m,n,adata);
  /* calculate x<=x-xo */
  my_daxpy(n,xo,-1.0,x);
  p02=my_dnrm2(n,x);
  gphi_0=(p01*p01-p02*p02)/(2.0*step);

  /* estimate for mu */
  /* mu = (tol-phi_0)/(rho gphi_0) */
  mu=(tol-phi_0)/(rho*gphi_0);
#ifdef DEBUG
  printf("mu=%lf, alpha1=%lf\n",mu,alpha1);
#endif

  ci=1;
  alphai=alpha1; /* initial value for alpha(i) : check if 0<alphai<=mu */
  alphai1=0.0; /* FIXME: tune for GPU (defalut is 0.0) */
  phi_alphai1=phi_0;
  while(ci<10) {
   /* evalualte phi(alpha(i))=f(xk+alphai pk) */
   my_dcopy(m,xk,1,xp,1); /* xp<=xk */
   my_daxpy(m,pk,alphai,xp); /* xp<=xp+alphai*pk */
   func(xp,x,m,n,adata);
   /* calculate x<=x-xo */
   my_daxpy(n,xo,-1.0,x);
   phi_alphai=my_dnrm2(n,x);
   phi_alphai*=phi_alphai;

   if (phi_alphai<tol) {
     alphak=alphai;
#ifdef DEBUG
     printf("Linesearch : Condition 0 met\n");
#endif
     break;
   }

   if ((phi_alphai>phi_0+alphai*gphi_0) || (ci>1 && phi_alphai>=phi_alphai1)) {
      /* ai=alphai1, bi=alphai bracket */
      alphak=linesearch_zoom(func,xk,pk,alphai1,alphai,x,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,xo,m,n,step,adata);
#ifdef DEBUG
      printf("Linesearch : Condition 1 met\n");
#endif
      break;
   } 

   /* evaluate grad(phi(alpha(i))) */
   my_dcopy(m,xk,1,xp,1); /* NOT NEEDED here?? xp<=xk */
   my_daxpy(m,pk,alphai+step,xp); /* xp<=xp+(alphai+step)*pk */
   func(xp,x,m,n,adata);
   /* calculate x<=x-xo */
   my_daxpy(n,xo,-1.0,x);
   p01=my_dnrm2(n,x);
   my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphai-step)*pk */
   func(xp,x,m,n,adata);
   /* calculate x<=x-xo */
   my_daxpy(n,xo,-1.0,x);
   p02=my_dnrm2(n,x);
   gphi_i=(p01*p01-p02*p02)/(2.0*step);

   if (fabs(gphi_i)<=-sigma*gphi_0) {
     alphak=alphai;
#ifdef DEBUG
     printf("Linesearch : Condition 2 met\n");
#endif
     break;
   }

   if (gphi_i>=0) {
     /* ai=alphai, bi=alphai1 bracket */
     alphak=linesearch_zoom(func,xk,pk,alphai,alphai1,x,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,xo,m,n,step,adata);
#ifdef DEBUG
     printf("Linesearch : Condition 3 met\n");
#endif
     break;
   }

   /* else preserve old values */
   if (mu<=(2*alphai-alphai1)) {
     /* next step */
     alphai1=alphai;
     alphai=mu;
   } else {
     /* choose by interpolation in [2*alphai-alphai1,min(mu,alphai+t1*(alphai-alphai1)] */
     p01=2*alphai-alphai1;
     p02=MIN(mu,alphai+t1*(alphai-alphai1));
     alphai=cubic_interp(func,xk,pk,p01,p02,x,xp,xo,m,n,step,adata);
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

/*************** Nocedal/Wright line search ********************************/
/* zoom function for line search */
/* func: vector function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   alphal/alphah: low,high value for alpha 
   x: size n x 1 (storage)
   xp: size m x 1 (storage)
   phi_0: phi(0)
   gphi_0: grad(phi(0))
   c1,c2: limit parameters for strong Wolfe conditions
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
static double 
linesearch_zoom_nw(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double alphal, double alphah, double *x, double *xp,  double phi_0, double gphi_0, double c1, double c2, double *xo, int m, int n, double step, void *adata) {

  double alphaj,phi_j,phi_low;
  double gphi_j,p01,p02;
  double alphak=1.0;
  int ci,found_step=0;

  /* sort out if indeed alphal<alphah */
  if (alphal>alphah) {
    p01=alphah;
    alphah=alphal;
    alphal=p01;
  }

  /* evaluate phi_low =phi(alphal) */
  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,alphal,xp); /* xp<=xp+(alphal)*pk */
  func(xp,x,m,n,adata);
  phi_low=my_dnrm2(n,x);
  phi_low*=phi_low;

  ci=0;
  while(ci<10) {
   /* trial step in [alphal,alphah] */
   //printf("Iter %d [%lf,%lf]\n",ci,alphal,alphah);
   alphaj=cubic_interp(func,xk,pk,alphal,alphah,x,xp,xo,m,n,step,adata);
   /* evaluate phi(alphaj) */
   my_dcopy(m,xk,1,xp,1); /* xp<=xk */
   my_daxpy(m,pk,alphaj,xp); /* xp<=xp+(alphaj)*pk */
   func(xp,x,m,n,adata);
   phi_j=my_dnrm2(n,x);
   phi_j*=phi_j;

   if ((phi_j>phi_0+c1*alphaj*gphi_0) || phi_j>=phi_low) {
     alphah=alphaj;
   } else {
    /* evaluate grad(alphaj) */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,alphaj+step,xp); /* xp<=xp+(alphaj+step)*pk */
    func(xp,x,m,n,adata);
    my_daxpy(n,xo,-1.0,x);
    p01=my_dnrm2(n,x);
    my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphaj-step)*pk */
    func(xp,x,m,n,adata);
    my_daxpy(n,xo,-1.0,x);
    p02=my_dnrm2(n,x);
    gphi_j=(p01*p01-p02*p02)/(2.0*step);
    
    if (fabs(gphi_j)<=-c2*gphi_0) {
      alphak=alphaj;
      found_step=1;
      break;
    }
    if (gphi_j*(alphah-alphal)>=0) {
       alphah=alphal;
    }
    alphal=alphaj;
   }
   ci++;
  }

  if (!found_step) {
   /* use bound to find possible step */
   alphak=cubic_interp(func,xk,pk,alphal,alphah,x,xp,xo,m,n,step,adata);
  }
   

  //printf("Found %lf Interval [%lf,%lf]\n",alphak,alphal,alphah);
  return alphak;
}
 
 

/* line search */
/* func: vector function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   alpha1/alphamax: initial value for step /max value
   c1,c2: limit parameters for strong Wolfe conditions
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
static double 
linesearch_nw(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double alpha1, double alphamax, double c1, double c2, double *xo, int m, int n, double step, void *adata) {
 
 /* phi(alpha)=f(xk+alpha pk)
  for vector function func 
   f(xk) =||func(xk)||^2 */
  
  double *x,*xp;
  double alphai1,alphai;
  double phi_0,phi_alphai,phi_alphai1;
  double p01,p02;
  double gphi_0,gphi_i;
  double alphak;

  int ci;

  alphai1=0.0; /* initial value for alpha(i-1) */
  alphai=alpha1; /* initial value for alpha(i) */
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
  func(xk,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  phi_0=my_dnrm2(n,x);
  phi_0*=phi_0;
  /* grad(phi_0): evaluate alpha0-step and alpha0+step */
  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,alphai1+step,xp); /* xp<=xp+(alphai+step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p01=my_dnrm2(n,x);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphai-step)*pk */
  func(xp,x,m,n,adata);
  my_daxpy(n,xo,-1.0,x);
  p02=my_dnrm2(n,x);
  gphi_0=(p01*p01-p02*p02)/(2.0*step);

  ci=1;
  phi_alphai1=phi_0;
  while(ci<10) {
   /* x=phi(alpha(i))=f(xk+alphai pk) */
   my_dcopy(m,xk,1,xp,1); /* xp<=xk */
   my_daxpy(m,pk,alphai,xp); /* xp<=xp+alphai*pk */
   func(xp,x,m,n,adata);
   /* calculate x<=x-xo */
   my_daxpy(n,xo,-1.0,x);
   phi_alphai=my_dnrm2(n,x);
   phi_alphai*=phi_alphai;
   if ((phi_alphai>phi_0+c1*alphai*gphi_0) || (ci>1 && phi_alphai>=phi_alphai1)) {
      //alphak=(alphai1+alphai)*0.5;
      alphak=linesearch_zoom_nw(func,xk,pk,alphai1,alphai,x,xp,phi_0,gphi_0,1e-4,0.9,xo,m,n,step,adata);
      printf("Linesearch : Condition 1 met\n");
      break;
   } 
   /* evaluate grad(phi(alpha(i))) */
   my_dcopy(m,xk,1,xp,1); /* NOT NEEDED here?? xp<=xk */
   my_daxpy(m,pk,step,xp); /* xp<=xp+(alphai+step)*pk */
   func(xp,x,m,n,adata);
   my_daxpy(n,xo,-1.0,x);
   p01=my_dnrm2(n,x);
   my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphai-step)*pk */
   func(xp,x,m,n,adata);
   my_daxpy(n,xo,-1.0,x);
   p02=my_dnrm2(n,x);
   gphi_i=(p01*p01-p02*p02)/(2.0*step);

   if (fabs(gphi_i)<=-c2*gphi_0) {
     alphak=alphai;
     printf("Linesearch : Condition 2 met\n");
     break;
   }

   if (gphi_i>=0) {
     //alphak=(alphai+alphai1)*0.5;
     alphak=linesearch_zoom_nw(func,xk,pk,alphai,alphai1,x,xp,phi_0,gphi_0,1e-4,0.9,xo,m,n,step,adata);
     printf("Linesearch : Condition 3 met\n");
     break;
   }

   /* else preserve old values */
   alphai1=alphai;
   alphai=cubic_interp(func,xk,pk,alphai,alphamax,x,xp,xo,m,n,step,adata);
   phi_alphai1=phi_alphai;

   ci++;
  }

  free(x);
  free(xp);
  printf("Step size=%lf\n",alphak);
  return alphak;
}
/*************** END Nocedal/Wright line search ****************************/
/* note M here  is LBFGS memory size */
int
lbfgs_fit(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, void *adata) {

  double *gk; /* gradients at both k+1 and k iter */
  double *xk1,*xk; /* parameters at k+1 and k iter */
  double *pk; /* step direction H_k * grad(f) */

  double step=1e-6; /* FIXME: tune for GPU */
  double *y, *s; /* storage for delta(grad) and delta(p) */
  double *rho; /* storage for 1/yk^T*sk */
  int ci,ck,cm;
  double alphak=1.0;
  

  me_data_t *dp=(me_data_t*)adata;
  char *hbb;
  int *ptoclus;
  int Nbase1=dp->Nbase*dp->tilesz;

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
  if ((hbb=(char*)calloc((size_t)(Nbase1*2),sizeof(char)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  rearrange_baselines(Nbase1, dp->barr, hbb, dp->Nt);
  /* baseline->station mapping */
//  for (ci=0; ci<Nbase1; ci++) {
//    if (!dp->barr[ci].flag) {
//     hbb[2*ci]=dp->barr[ci].sta1;
//     hbb[2*ci+1]=dp->barr[ci].sta2;
//    } else { /* flagged baselines have -1 for stations */
//     hbb[2*ci]=-1;
//     hbb[2*ci+1]=-1;
//    }
//  }

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


  /* initial value for params xk=p */
  my_dcopy(m,p,1,xk,1);
  /*  gradient gk=grad(f)_k */
  func_grad(func,xk,gk,x,m,n,step,gpu_threads,adata);
  ck=0;
  cm=0;
  ci=0;
 
  while (ck<itmax) {
   /* mult with hessian  pk=-H_k*gk */
   if (ck<M) {
    mult_hessian(m,pk,gk,s,y,rho,ck,ci);
   } else {
    mult_hessian(m,pk,gk,s,y,rho,M,ci);
   }
   my_dscal(m,-1.0,pk);

   /* linesearch to find step length */
   /* parameters alpha1=10.0,sigma=0.1, rho=0.01, t1=9, t2=0.1, t3=0.5 */
   /* FIXME: update paramters for GUP gradient */
   alphak=linesearch(func,xk,pk,10.0,0.1,0.01,9,0.1,0.5,x,m,n,step,adata);
   /* parameters c1=1e-4 c2=0.9, alpha1=1.0, alphamax=10.0, step (for alpha)=1e-4*/
   //alphak=linesearch_nw(func,xk,pk,1.0,10.0,1e-4,0.9,x,m,n,1e-4,adata);
   //alphak=1.0;
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
   func_grad(func,xk1,gk,x,m,n,step,gpu_threads,adata);
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
  return 0;
}
