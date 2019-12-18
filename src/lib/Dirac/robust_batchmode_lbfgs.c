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


#include "Dirac.h"
#include <pthread.h>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

//#define DEBUG
/**** repeated code here ********************/
/* Jones matrix multiplication 
   C=A*B
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
amb(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=a[0]*b[0]+a[1]*b[2];
 c[1]=a[0]*b[1]+a[1]*b[3];
 c[2]=a[2]*b[0]+a[3]*b[2];
 c[3]=a[2]*b[1]+a[3]*b[3];
}

/* Jones matrix multiplication 
   C=A*B^H
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
ambt(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=a[0]*conj(b[0])+a[1]*conj(b[1]);
 c[1]=a[0]*conj(b[2])+a[1]*conj(b[3]);
 c[2]=a[2]*conj(b[0])+a[3]*conj(b[1]);
 c[3]=a[2]*conj(b[2])+a[3]*conj(b[3]);
}

/**** end repeated code ********************/

/*************************************** ROBUST ***************************/
/***************************************************************/
/* Huber loss function */
/*  r = residual = y_i - f_i 
  elementwise, loss = r^2 if |r|<= nu ; else loss = 2 nu |r| - nu^2 (when |r|>k)
*/
static void *
func_huber_th(void *data) {
  thread_data_logf_t *t=(thread_data_logf_t*)data;
  t->sum=0.0;
  int ci;
  double r,loss;
  for (ci=t->start; ci<=t->end; ci++) {
    r=fabs(t->x[ci]-t->f[ci]); 
    if (r<=t->nu) { /* not r it is |r| here */
      loss=r*r;
    } else {
      loss=2.0*t->nu*r-t->nu*t->nu; 
    }
    t->sum+=loss;
  }
  return NULL;
}

/* worker thread to calculate
   sum ( log(1+ (y_i-f_i)^2/nu) )
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
func_robust_th(void *data) {
  thread_data_logf_t *t=(thread_data_logf_t*)data;
  double inv_nu=1.0/t->nu;
  t->sum=0.0;
  int ci;
  double err;
  for (ci=t->start; ci<=t->end; ci++) {
    err=t->x[ci]-t->f[ci]; 
    err=err*err*inv_nu;
    t->sum+=log(1.0+err);
  } 
  return NULL;
}
/* recalculate log(1+ (y_i-f_i)^2/nu) 
   from function() that calculates f_i 
   y (data)
   f=model vector
   x=log(1+(y_i-f_i)^2/nu) output
   all size n x 1
   Nt: no of threads

   return sum(log(..))
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double
func_robust(
   double *f, double *y, int n, double robust_nu, int Nt) {

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_logf_t *threaddata;
  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_logf_t*)malloc((size_t)Nt*sizeof(thread_data_logf_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  int ci,nth,Nparm;
  Nparm=(n+Nt-1)/Nt;

  ci=0;
  for (nth=0; nth<Nt; nth++) {
    threaddata[nth].f=f; /* model */
    threaddata[nth].x=y; /* data */
    threaddata[nth].nu=robust_nu;
    threaddata[nth].start=ci;
    threaddata[nth].sum=0.0;
    threaddata[nth].end=ci+Nparm-1;
    if (threaddata[nth].end >=n) {
     threaddata[nth].end=n-1;
    }
    ci=ci+Nparm; 
    pthread_create(&th_array[nth],&attr,func_robust_th,(void*)(&threaddata[nth]));
    //pthread_create(&th_array[nth],&attr,func_huber_th,(void*)(&threaddata[nth]));

  }
  /* now wait for threads to finish */
  double mysum=0.0;
  for(nth=0; nth<Nt; nth++) {
   pthread_join(th_array[nth],NULL);
   mysum+=threaddata[nth].sum;
  }
  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);

  return mysum;
}
/***************************************************************/

static double
sign(double x) {
 return (x<0.0?-1.0:1.0);
}

/* Huber loss gradient */
/*  r = residual = y_i - f_i 
  elementwise, loss = r^2 if |r|<= nu ; else loss = 2 nu |r| - nu^2 (when |r|>k)
  Gradient, elementwise
  = 2 * r (dr/dtheta) if  |r|<= nu; else = 2*nu*sign(r) (dr/dtheta)
  sign(x) = -1 if x<0 1 if x>0, 0 not a problem here becasue sign(.) not used when |r|=0
*/
static void *
cpu_calc_deriv_huber(void *adata) {
 thread_data_grad_t *t=(thread_data_grad_t*)adata;

 int ci,nb;
 int stc,stoff,stm,sta1,sta2;
 int N=t->N; /* stations */
 int M=t->M; /* clusters */


 double xr[8]; /* residuals */
 complex double G1[4],G2[4],C[4],T1[4],T2[4];
 double pp[8];
 double dsum;
 int cli,tpchunk,pstart,nchunk,tilesperchunk,stci,ttile,tptile,poff;
 double nu=t->robust_nu;

 /* iterate over each paramter */
 for (ci=t->g_start; ci<=t->g_end; ++ci) {
    t->g[ci]=0.0;
    /* find station and parameter corresponding to this value of ci */
    /* this parameter should correspond to the right baseline (x tilesz)
        to contribute to the derivative */
    cli=0;
    while((cli<M) && (ci<t->carr[cli].p[0] || ci>t->carr[cli].p[0]+8*N*t->carr[cli].nchunk-1)) {
     cli++;
    }
   /* now either cli>=M: cluster not found 
       or cli<M and cli is the right cluster */
   if (cli==M && ci>=t->carr[cli-1].p[0] && ci<=t->carr[cli-1].p[0]+8*N*t->carr[cli-1].nchunk-1) {
    cli--;
   }

   if (cli<M) {
    /* right parameter offset */
    stci=ci-t->carr[cli].p[0];
 
    stc=(stci%(8*N))/8; /* 0..N-1 */
    /* make sure this baseline contribute to this parameter */
    tpchunk=stci/(8*N);
    nchunk=t->carr[cli].nchunk;
    pstart=t->carr[cli].p[0];
    tilesperchunk=(t->tilesz+nchunk-1)/nchunk;


    /* iterate over all baselines and accumulate sum */
    for (nb=t->noff; nb<t->nlen+t->noff; ++nb) {
     /* which tile is this ? */
     ttile=nb/t->Nbase;
     /* which chunk this tile belongs to */
     tptile=ttile/tilesperchunk;
     /* now tptile has to match tpchunk, otherwise ignore calculation */
     if (tptile==tpchunk) {

     sta1=t->barr[nb].sta1;
     sta2=t->barr[nb].sta2;
    
     if (((stc==sta1)||(stc==sta2))&& !t->barr[nb].flag) {
      /* this baseline has a contribution */
      /* which paramter of this station */
      stoff=(stci%(8*N))%8; /* 0..7 */
      /* which cluster */
      stm=cli; /* 0..M-1 */

      /* exact expression for derivative  
      1) for Gaussian \sum( y_i - f_i(\theta))^2
         2 real( vec^H(residual_this_baseline) 
            * vec(-J_{pm}C_{pqm} J_{qm}^H)
        where m: chosen cluster
        J_{pm},J_{qm} Jones matrices for baseline p-q
        depending on the parameter, J ==> E 
        E: zero matrix, except 1 at location of m
      \sum( 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 

      2) for robust \sum( log(1+ (y_i-f_i(\theta))^2/\nu) )
       all calculations are like for the Gaussian case, except
       when taking summation
      \sum( 1/(\nu+(y_i-f_i)^2) 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 
 
      so additonal multiplication by 1/(\nu+(y_i-f_i)^2)

      3) for Huber loss, r_i=y_i-f_i
         2*r_i (-\partial f_i / \partial \theta) if |r_i| <= nu
         2*nu*sign(r_i)  (-\partial f_i / \partial \theta) else
     */
     /* read in residual vector, (real,imag) separately */
     xr[0]=t->x[nb*8];
     xr[1]=t->x[nb*8+1];
     xr[2]=t->x[nb*8+2];
     xr[3]=t->x[nb*8+3];
     xr[4]=t->x[nb*8+4];
     xr[5]=t->x[nb*8+5];
     xr[6]=t->x[nb*8+6];
     xr[7]=t->x[nb*8+7];

     /* read in coherency */
     C[0]=t->coh[4*M*nb+4*stm];
     C[1]=t->coh[4*M*nb+4*stm+1];
     C[2]=t->coh[4*M*nb+4*stm+2];
     C[3]=t->coh[4*M*nb+4*stm+3];

     memset(pp,0,sizeof(double)*8); 
     if (stc==sta1) {
       /* this station parameter gradient */
       pp[stoff]=1.0;
       memset(G1,0,sizeof(complex double)*4); 
       G1[0]=pp[0]+_Complex_I*pp[1];
       G1[1]=pp[2]+_Complex_I*pp[3];
       G1[2]=pp[4]+_Complex_I*pp[5];
       G1[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta2*8;
       G2[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G2[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G2[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+5]);
       G2[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);
     } else if (stc==sta2) {
       memset(G2,0,sizeof(complex double)*4); 
       pp[stoff]=1.0;
       G2[0]=pp[0]+_Complex_I*pp[1];
       G2[1]=pp[2]+_Complex_I*pp[3];
       G2[2]=pp[4]+_Complex_I*pp[5];
       G2[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta1*8;
       G1[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G1[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G1[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+5]);
       G1[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);
     }

     /* T1=G1*C */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* calculate product (\partial f/\partial r) x (\partial r/ \partial \theta) */
     dsum=(fabs(xr[0]) <= nu? xr[0]*creal(T2[0]): nu*sign(xr[0])*creal(T2[0]));
     dsum+=(fabs(xr[1]) <= nu? xr[1]*cimag(T2[0]): nu*sign(xr[1])*cimag(T2[0]));
     dsum+=(fabs(xr[2]) <= nu? xr[2]*creal(T2[1]): nu*sign(xr[2])*creal(T2[1]));
     dsum+=(fabs(xr[3]) <= nu? xr[3]*cimag(T2[1]): nu*sign(xr[3])*cimag(T2[1]));
     dsum+=(fabs(xr[4]) <= nu? xr[4]*creal(T2[2]): nu*sign(xr[4])*creal(T2[2]));
     dsum+=(fabs(xr[5]) <= nu? xr[5]*cimag(T2[2]): nu*sign(xr[5])*cimag(T2[2]));
     dsum+=(fabs(xr[6]) <= nu? xr[6]*creal(T2[3]): nu*sign(xr[6])*creal(T2[3]));
     dsum+=(fabs(xr[7]) <= nu? xr[7]*cimag(T2[3]): nu*sign(xr[7])*cimag(T2[3]));
     /* accumulate sum NOTE
     its important to get the sign right,
     depending on res=data-model or res=model-data  */
     t->g[ci]+=-2.0*(dsum);
     }
     }
    }
   }
 }


 return NULL;
}


/* worker thread for a cpu */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
cpu_calc_deriv_robust(void *adata) {
 thread_data_grad_t *t=(thread_data_grad_t*)adata;

 int ci,nb;
 int stc,stoff,stm,sta1,sta2;
 int N=t->N; /* stations */
 int M=t->M; /* clusters */


 double xr[8]; /* residuals */
 complex double G1[4],G2[4],C[4],T1[4],T2[4];
 double pp[8];
 double dsum;
 int cli,tpchunk,pstart,nchunk,tilesperchunk,stci,ttile,tptile,poff;
 double nu=t->robust_nu;

 /* iterate over each paramter */
 for (ci=t->g_start; ci<=t->g_end; ++ci) {
    t->g[ci]=0.0;
    /* find station and parameter corresponding to this value of ci */
    /* this parameter should correspond to the right baseline (x tilesz)
        to contribute to the derivative */
    cli=0;
    while((cli<M) && (ci<t->carr[cli].p[0] || ci>t->carr[cli].p[0]+8*N*t->carr[cli].nchunk-1)) {
     cli++;
    }
   /* now either cli>=M: cluster not found 
       or cli<M and cli is the right cluster */
   if (cli==M && ci>=t->carr[cli-1].p[0] && ci<=t->carr[cli-1].p[0]+8*N*t->carr[cli-1].nchunk-1) {
    cli--;
   }

   if (cli<M) {
    /* right parameter offset */
    stci=ci-t->carr[cli].p[0];
 
    stc=(stci%(8*N))/8; /* 0..N-1 */
    /* make sure this baseline contribute to this parameter */
    tpchunk=stci/(8*N);
    nchunk=t->carr[cli].nchunk;
    pstart=t->carr[cli].p[0];
    tilesperchunk=(t->tilesz+nchunk-1)/nchunk;


    /* iterate over all baselines and accumulate sum */
    for (nb=t->noff; nb<t->nlen+t->noff; ++nb) {
     /* which tile is this ? */
     ttile=nb/t->Nbase;
     /* which chunk this tile belongs to */
     tptile=ttile/tilesperchunk;
     /* now tptile has to match tpchunk, otherwise ignore calculation */
     if (tptile==tpchunk) {

     sta1=t->barr[nb].sta1;
     sta2=t->barr[nb].sta2;
    
     if (((stc==sta1)||(stc==sta2))&& !t->barr[nb].flag) {
      /* this baseline has a contribution */
      /* which paramter of this station */
      stoff=(stci%(8*N))%8; /* 0..7 */
      /* which cluster */
      stm=cli; /* 0..M-1 */

      /* exact expression for derivative  
       for Gaussian \sum( y_i - f_i(\theta))^2
         2 real( vec^H(residual_this_baseline) 
            * vec(-J_{pm}C_{pqm} J_{qm}^H)
        where m: chosen cluster
        J_{pm},J_{qm} Jones matrices for baseline p-q
        depending on the parameter, J ==> E 
        E: zero matrix, except 1 at location of m
      \sum( 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 

       for robust \sum( log(1+ (y_i-f_i(\theta))^2/\nu) )
       all calculations are like for the Gaussian case, except
       when taking summation
      \sum( 1/(\nu+(y_i-f_i)^2) 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 
 
      so additonal multiplication by 1/(\nu+(y_i-f_i)^2)
     */
     /* read in residual vector, (real,imag) separately */
     xr[0]=t->x[nb*8];
     xr[1]=t->x[nb*8+1];
     xr[2]=t->x[nb*8+2];
     xr[3]=t->x[nb*8+3];
     xr[4]=t->x[nb*8+4];
     xr[5]=t->x[nb*8+5];
     xr[6]=t->x[nb*8+6];
     xr[7]=t->x[nb*8+7];

     /* read in coherency */
     C[0]=t->coh[4*M*nb+4*stm];
     C[1]=t->coh[4*M*nb+4*stm+1];
     C[2]=t->coh[4*M*nb+4*stm+2];
     C[3]=t->coh[4*M*nb+4*stm+3];

     memset(pp,0,sizeof(double)*8); 
     if (stc==sta1) {
       /* this station parameter gradient */
       pp[stoff]=1.0;
       memset(G1,0,sizeof(complex double)*4); 
       G1[0]=pp[0]+_Complex_I*pp[1];
       G1[1]=pp[2]+_Complex_I*pp[3];
       G1[2]=pp[4]+_Complex_I*pp[5];
       G1[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta2*8;
       G2[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G2[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G2[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+5]);
       G2[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);
     } else if (stc==sta2) {
       memset(G2,0,sizeof(complex double)*4); 
       pp[stoff]=1.0;
       G2[0]=pp[0]+_Complex_I*pp[1];
       G2[1]=pp[2]+_Complex_I*pp[3];
       G2[2]=pp[4]+_Complex_I*pp[5];
       G2[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta1*8;
       G1[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G1[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G1[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+5]);
       G1[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);
     }

     /* T1=G1*C */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* calculate product xr*vec(J_p C J_q^H )/(nu+residual^2) */
     dsum=xr[0]*creal(T2[0])/(nu+xr[0]*xr[0]);
     dsum+=xr[1]*cimag(T2[0])/(nu+xr[1]*xr[1]);
     dsum+=xr[2]*creal(T2[1])/(nu+xr[2]*xr[2]);
     dsum+=xr[3]*cimag(T2[1])/(nu+xr[3]*xr[3]);
     dsum+=xr[4]*creal(T2[2])/(nu+xr[4]*xr[4]);
     dsum+=xr[5]*cimag(T2[2])/(nu+xr[5]*xr[5]);
     dsum+=xr[6]*creal(T2[3])/(nu+xr[6]*xr[6]);
     dsum+=xr[7]*cimag(T2[3])/(nu+xr[7]*xr[7]);

     /* accumulate sum NOTE
     its important to get the sign right,
     depending on res=data-model or res=model-data  */
     t->g[ci]+=-2.0*(dsum);
     }
     }
    }
   }
 }


 return NULL;
}
/* calculate gradient */
/* func: vector function to predict model
   p: parameter values size m x 1 (at which grad is calculated)
   g: gradient size m x 1 
   xo: observed data size n x 1
   robust_nu: nu in T distribution
   n: size of vector function

   noff: offset in data (baselines), nlen: length of data (baselines) : both x8 to match data length
   adata:  additional data passed to the function
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
func_grad_robust_batch(
   void (*func)(double *p, double *hx, int m, int n, int noff, int nlen, void *adata),
   double *p, double *g, double *xo, int m, int n, int noff, int nlen, void *adata) {
  /* numerical gradient for each parameter is
     (||func(p+step*e_i)-x||^2-||func(p-step*e_i)-x||^2)/2*step
    i=0,...,m-1 for all parameters
    e_i: unit vector, 1 only at i-th location
  */

  double *x; /* array to store residual */
  int ci;
  me_data_t *dp=(me_data_t*)adata;

  int Nt=dp->Nt;

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_grad_t *threaddata;

  /* we still needs a full length x vector here, even when we only use noff:..:noff+nlen elements */
  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
/*
  double p0=p[0]; double eps=1e-6;
  p[0]=p0+eps;
  func(p,x,m,n,noff,nlen,adata);
  my_daxpy(nlen*8,&xo[8*noff],-1.0,&x[8*noff]);
  double nrm1=my_dnrm2(nlen*8,&x[8*noff]);
  p[0]=p0-eps;
  func(p,x,m,n,noff,nlen,adata);
  my_daxpy(nlen*8,&xo[8*noff],-1.0,&x[8*noff]);
  double nrm2=my_dnrm2(nlen*8,&x[8*noff]);
printf("Numerical grad= %lf , %lf =%lf\n",nrm1,nrm2,(nrm1*nrm1-nrm2*nrm2)/(2.0*eps));
  p[0]=p0;
*/
  /* evaluate func once, store in x, and create threads */
  /* and calculate the residual x=xo-func */
  func(p,x,m,n,noff,nlen,adata);
  /* calculate x<=x-xo : only in the valid region */
  my_daxpy(nlen*8,&xo[8*noff],-1.0,&x[8*noff]);

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_grad_t*)malloc((size_t)Nt*sizeof(thread_data_grad_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  int nth,Nparm;

  /* parameters per thread */
  Nparm=(m+Nt-1)/Nt;

  /* each thread will calculate derivative of part of 
     parameters */
  ci=0;
  for (nth=0;  nth<Nt; nth++) {
   threaddata[nth].Nbase=dp->Nbase;
   threaddata[nth].tilesz=dp->tilesz;
   threaddata[nth].barr=dp->barr;
   threaddata[nth].carr=dp->carr;
   threaddata[nth].M=dp->M;
   threaddata[nth].N=dp->N;
   threaddata[nth].coh=dp->coh;
   threaddata[nth].m=m;
   threaddata[nth].n=n;
   threaddata[nth].x=x;
   threaddata[nth].p=p;
   threaddata[nth].g=g;
   threaddata[nth].robust_nu=dp->robust_nu;
   threaddata[nth].g_start=ci;
   threaddata[nth].g_end=ci+Nparm-1;
   if (threaddata[nth].g_end>=m) {
    threaddata[nth].g_end=m-1;
   }
   ci=ci+Nparm;
    
   /* which data batch to use offset and length in baselines */
   threaddata[nth].noff=noff;
   threaddata[nth].nlen=nlen;
   pthread_create(&th_array[nth],&attr,cpu_calc_deriv_robust,(void*)(&threaddata[nth]));
   //pthread_create(&th_array[nth],&attr,cpu_calc_deriv_huber,(void*)(&threaddata[nth]));
  }

  /* now wait for threads to finish */
  for(nth=0; nth<Nt; nth++) {
   pthread_join(th_array[nth],NULL);
  }

  pthread_attr_destroy(&attr);

//printf("Grad =%lf\n",g[0]);
  free(th_array);
  free(threaddata);

  free(x);
  return 0;
}
/*************************************** END ROBUST ***************************/

/******************** full minimization *****************************/

/* FIXME: remove this */
/* worker thread function for prediction */
static void *
predict_threadfn_withgain_full(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,sta1,sta2;
 double *pm;
 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 int Ntilebase=(t->Nbase)*(t->tilesz);
 int px;

 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   memset(&(t->x[8*ci]),0,sizeof(double)*8);

   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   for (cm=0; cm<M; cm++) { /* clusters */
     /* gains for this cluster, for sta1,sta2 */
     /* depending on the baseline index, and cluster chunk size,
        select proper parameter addres */
    /* depending on the chunk size and the baseline index,
        select right set of parameters 
       data x=[0,........,Nbase*tilesz]
       divided into nchunk chunks
       p[0] -> x[0.....Nbase*tilesz/nchunk-1]
       p[1] -> x[Nbase*tilesz/nchunk......2*Nbase*tilesz-1]
       ....
       p[last] -> x[(nchunk-1)*Nbase*tilesz/nchunk......Nbase*tilesz]

       so given bindex,  right p[] is bindex/((Nbase*tilesz+nchunk-1)/nchunk)
       */

     px=(ci+t->boff)/((Ntilebase+t->carr[cm].nchunk-1)/t->carr[cm].nchunk);
     //pm=&(t->p[cm*8*N]);
     pm=&(t->p[t->carr[cm].p[px]]);
     G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
     G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
     G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
     G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
     G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
     G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
     G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
     G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);


      /* use pre calculated values */
      C[0]=t->coh[4*M*ci+4*cm];
      C[1]=t->coh[4*M*ci+4*cm+1];
      C[2]=t->coh[4*M*ci+4*cm+2];
      C[3]=t->coh[4*M*ci+4*cm+3];

     /* form G1*C*G2' */
     /* T1=G1*C  */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* add to baseline visibilities */
     t->x[8*ci]+=creal(T2[0]);
     t->x[8*ci+1]+=cimag(T2[0]);
     t->x[8*ci+2]+=creal(T2[1]);
     t->x[8*ci+3]+=cimag(T2[1]);
     t->x[8*ci+4]+=creal(T2[2]);
     t->x[8*ci+5]+=cimag(T2[2]);
     t->x[8*ci+6]+=creal(T2[3]);
     t->x[8*ci+7]+=cimag(T2[3]);
   }
  }
 }

 return NULL;
}


/* minimization function for batch mode */
/* p: size mx1 parameters
   x: size nx1 model being calculated
   noff: offset in data (baselines), nlen: length of data (baselines) : both x8 to match data length
   data: extra info needed */
static void
minimize_viz_full_pth_batch(double *p, double *x, int m, int n, int noff, int nlen, void *data) {

  me_data_t *dp=(me_data_t*)data;
  /* u,v,w : size Nbase*tilesz x 1  x: size Nbase*8*tilesz x 1 */
  /* barr: size Nbase*tilesz x 1 carr: size Mx1 */
  /* pp: size 8*N*M x 1 */
  /* pm: size Mx1 of double */

  int nth,nth1,ci;

  /* no of threads */
  int Nt=(dp->Nt);
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=nlen;//(dp->Nbase)*(dp->tilesz);

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase1+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  /* iterate over threads, allocating baselines per thread */
  ci=noff;//0; correct offset in baselines
  for (nth=0;  nth<Nt && ci<Nbase1+noff; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1+noff) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1+noff-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=dp->barr;
    threaddata[nth].u=&(dp->u[ci]);
    threaddata[nth].v=&(dp->v[ci]);
    threaddata[nth].w=&(dp->w[ci]);
    threaddata[nth].carr=dp->carr;
    threaddata[nth].M=dp->M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].p=p;
    threaddata[nth].N=dp->N;
    threaddata[nth].Nbase=dp->Nbase;
    threaddata[nth].tilesz=dp->tilesz;
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*ci]);
    
    //printf("thread %d predict  data from %d baselines %d\n",nth,8*ci,Nthb);
    pthread_create(&th_array[nth],&attr,predict_threadfn_withgain_full,(void*)(&threaddata[nth]));
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

}


/******************** end full minimization *****************************/



/*************************** begin stochastic version ****************************/
typedef struct wrapper_me_data_batchmode_t_ {
 me_data_t *adata;
 void (*func)(double *p, double *hx, int m, int n, void *adata);
 double *x; /* full data vector nx1 */
 int n;
 /* persistent data struct, where info about batch 
    is also stored - FIXME not needed */
 persistent_data_t *ptdata;

 /* for consensus optimization */
 double *y; /* lagrange multiplier, size equal to p */
 double *z; /* Bz polynomial constraint, size equal to p */
 double *rho; /* regularization, Mt values */
} wrapper_me_data_batchmode_t;

static double
robust_cost_func_batch(double *p, int m, void *adata) {
  wrapper_me_data_batchmode_t *dp=(wrapper_me_data_batchmode_t*)adata;
  me_data_t *d1=(me_data_t *)dp->adata;
  persistent_data_t *ptd=dp->ptdata;

  double *x=dp->x; /* full input data, right offset needed */
  int n=dp->n; /* full size */
  double *xmodel;
  //printf("batch %d:%d full %d\n",ptd->noff,ptd->nlen,n);

  /* we still need to allocate the full length here */
  if ((xmodel=(double*)calloc((size_t)n,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* predict model */
  minimize_viz_full_pth_batch(p, xmodel, m, n, ptd->offset, ptd->nlen, dp->adata);

  //double f0=func_robust(x,xmodel,n, d1->robust_nu, d1->Nt);
  double f0=func_robust(&(x[8*ptd->offset]),&xmodel[8*ptd->offset],8*ptd->nlen, d1->robust_nu, d1->Nt);

  free(xmodel);
  return f0;
}

static void 
robust_grad_func_batch(double *p, double *g, int m, void *adata) {
  wrapper_me_data_batchmode_t *dp=(wrapper_me_data_batchmode_t*)adata;
  double *x=dp->x; /* input data : changed at every batch */
  int n=dp->n; /* size also changed per batch */
  persistent_data_t *ptd=dp->ptdata;

  func_grad_robust_batch(&minimize_viz_full_pth_batch,p,g,x, m, n, ptd->offset, ptd->nlen, dp->adata);
}

int
lbfgs_fit_robust_wrapper_minibatch(
   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads,
 void *adata) {
  /* info about division of data into batches,
     and how to cycle between batches stored in wrapper */
  wrapper_me_data_batchmode_t data1;
  me_data_t *dp=(me_data_t *)adata;
  data1.adata=dp;
  data1.x=x;  /* each batch selects a subset to work on */ 
  data1.n=n;  /* amount of data (n) is lower than this for each batch */

  int Nbatch=5; /* how many minibatches to split the data */
  int Niterperbatch=4; /* how many iterations per minibatch */
  /* persistent memory between batches (y,s) pairs
     and info about online var(||grad||) estimate */
  persistent_data_t ptdata;


  int ntiles=dp->tilesz;
  /* split iterations into mini batches  (counting baselines)
     each batch identified by start (offset): 0... Nbase1-1 and length: 1 ... Nbase
     note data size = 8 x baselines 
  */
  int Nbase1=dp->Nbase*ntiles;
  lbfgs_persist_init(&ptdata,Nbatch,m,Nbase1,M,dp->Nt);
  ptdata.nlen=n; /* length of data =8 x baselines, so reset it here */

  data1.ptdata=&ptdata;

#ifdef DEBUG
  printf("baselines =%d tiles=%d total %d divided to %d\n",dp->Nbase,ntiles,n,Nbatch);
#endif
  /* how many loops (epochs) over batches ? */
  int Nloops=(itmax+Niterperbatch)/Niterperbatch;
  int nl,ci;
#ifdef DEBUG
  printf("loops=%d\n",Nloops);
  for (ci=0; ci<Nbatch; ci++) {
   printf("batch %d : off %d size %d\n",ci,ptdata.offsets[ci],ptdata.lengths[ci]);
  }
#endif

#ifdef DEBUG
  ptdata.offset=0; ptdata.nlen=Nbase1; 
  printf("cost=%g\n",robust_cost_func_batch(p, m,&data1));
#endif
  for (nl=0; nl<Nloops; nl++) {
   /* iterate over each batch */
   for (ci=0; ci<Nbatch; ci++) {
    ptdata.offset=ptdata.offsets[ci];
    ptdata.nlen=ptdata.lengths[ci];
    lbfgs_fit(&robust_cost_func_batch,&robust_grad_func_batch,p,m,Niterperbatch,M,&data1,&ptdata);

    /* only for debugging, full cost */
#ifdef DEBUG
    ptdata.offset=0; ptdata.nlen=Nbase1; printf("cost=%g\n",robust_cost_func_batch(p, m,&data1));
#endif
   }
  }


  /* free persistent memory */
  lbfgs_persist_clear(&ptdata);
  return 0;
}


/**************************************************************************************/
/*** all below are for stochastic calibration, without freq. averaging data ***/
static void *
predict_threadfn_withgain_multifreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;

 int ci,cm,sta1,sta2;
 double *pm;
 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 int Ntilebase=(t->Nbase)*(t->tilesz);
 int nchan, chanoff=4*M*Ntilebase; /* offset for finding the coherencies */
 int px;

 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over channels */
   for (nchan=0; nchan<t->Nchan; nchan++) {

    /* iterate over the sky model and calculate contribution */
    /* for this x[8*ci:8*(ci+1)-1] */
    memset(&(t->x[8*ci+nchan*8*Ntilebase]),0,sizeof(double)*8);
   }

   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   for (cm=0; cm<M; cm++) { /* clusters */
     /* gains for this cluster, for sta1,sta2 */
    /* depending on the baseline index, and cluster chunk size,
        select proper parameter addres */
    /* depending on the chunk size and the baseline index,
        select right set of parameters 
       data x=[0,........,Nbase*tilesz]
       divided into nchunk chunks
       p[0] -> x[0.....Nbase*tilesz/nchunk-1]
       p[1] -> x[Nbase*tilesz/nchunk......2*Nbase*tilesz-1]
       ....
       p[last] -> x[(nchunk-1)*Nbase*tilesz/nchunk......Nbase*tilesz]

       so given bindex,  right p[] is bindex/((Nbase*tilesz+nchunk-1)/nchunk)
       */

     /* gains are independent of freq, so loop order could be changed */
     px=(ci+t->boff)/((Ntilebase+t->carr[cm].nchunk-1)/t->carr[cm].nchunk);
     pm=&(t->p[t->carr[cm].p[px]]);
     G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
     G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
     G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
     G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
     G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
     G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
     G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
     G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);


    for (nchan=0; nchan<t->Nchan; nchan++) {
      /* use pre calculated values */
      C[0]=t->coh[nchan*chanoff+4*M*ci+4*cm];
      C[1]=t->coh[nchan*chanoff+4*M*ci+4*cm+1];
      C[2]=t->coh[nchan*chanoff+4*M*ci+4*cm+2];
      C[3]=t->coh[nchan*chanoff+4*M*ci+4*cm+3];
//printf("n=%d base (%d,%d) clus %d vis %lf,%lf\n",ci+t->boff,sta1,sta2,cm,creal(C[0]),cimag(C[0]));

     /* form G1*C*G2' */
     /* T1=G1*C  */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* add to baseline visibilities */
     t->x[nchan*8*Ntilebase+8*ci]+=creal(T2[0]);
     t->x[nchan*8*Ntilebase+8*ci+1]+=cimag(T2[0]);
     t->x[nchan*8*Ntilebase+8*ci+2]+=creal(T2[1]);
     t->x[nchan*8*Ntilebase+8*ci+3]+=cimag(T2[1]);
     t->x[nchan*8*Ntilebase+8*ci+4]+=creal(T2[2]);
     t->x[nchan*8*Ntilebase+8*ci+5]+=cimag(T2[2]);
     t->x[nchan*8*Ntilebase+8*ci+6]+=creal(T2[3]);
     t->x[nchan*8*Ntilebase+8*ci+7]+=cimag(T2[3]);
   } /* channel iter */
  } /* cluster iter */
  } 
 } /* baseline iter */

 return NULL;
}

/* this function is similar to minimize_viz_full_pth(), except it handles 
   Nchan channels of data */
static void
minimize_viz_full_multifreq(double *p, double *x, int m, int n, void *data) {

  me_data_t *dp=(me_data_t*)data;

  int nth,nth1,ci;

  /* no of threads */
  int Nt=(dp->Nt);
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=(dp->Nbase)*(dp->tilesz);

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase1+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }
    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=dp->barr;
    threaddata[nth].u=&(dp->u[ci]);
    threaddata[nth].v=&(dp->v[ci]);
    threaddata[nth].w=&(dp->w[ci]);
    threaddata[nth].carr=dp->carr;
    threaddata[nth].M=dp->M;
    threaddata[nth].x=&(x[8*ci]); /* data offset for 1st channel */
    threaddata[nth].p=p;
    threaddata[nth].N=dp->N;
    threaddata[nth].Nbase=dp->Nbase;
    threaddata[nth].tilesz=dp->tilesz;
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*ci]); /* coherency offset for 1st channel */
    threaddata[nth].Nchan=dp->Nchan;


    pthread_create(&th_array[nth],&attr,predict_threadfn_withgain_multifreq,(void*)(&threaddata[nth]));
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

}

/* cost function */
static double
robust_cost_func_multifreq(double *p, int m, void *adata) {
  wrapper_me_data_batchmode_t *dp=(wrapper_me_data_batchmode_t*)adata;
  me_data_t *d1=(me_data_t *)dp->adata;

  double *x=dp->x;
  int n=dp->n; /* note n here is total data length (all channels) */ 
  double *xmodel;
  if ((xmodel=(double*)calloc((size_t)n,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* predict model */
  minimize_viz_full_multifreq(p, xmodel, m, n, dp->adata);
  double f0=func_robust(x,xmodel,n,d1->robust_nu,d1->Nt);

  free(xmodel);

  if (!dp->y && !dp->z && !dp->rho){
   return f0;
  } 
  /* else this cost function is used in consensus optimization */
  /* extra cost  y^T (x-z) + rho/2 (x-z)^T(x-z) */
  double *xp;
  if ((xp=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if(m !=d1->Mt*d1->N*8) {
     fprintf(stderr,"%s: %d: incorrect parameter lengths\n",__FILE__,__LINE__);
     exit(1);
  }
  /* find xp=p-z */
  my_dcopy(m,p,1,xp,1);
  my_daxpy(m,dp->z,-1.0,xp);
  double f1=0.0;
  int ci;
  for (ci=0; ci<d1->Mt; ci++) {
    f1=f1+my_ddot(8*d1->N,&xp[8*d1->N*ci],&dp->y[8*d1->N*ci])+dp->rho[ci]*0.5*my_ddot(8*d1->N,&xp[8*d1->N*ci],&xp[8*d1->N*ci]);
  }
  free(xp);
  return f0+f1;
}


static void *
cpu_calc_deriv_multifreq(void *adata) {
 thread_data_grad_t *t=(thread_data_grad_t*)adata;

 int ci,nb;
 int stc,stoff,stm,sta1,sta2;
 int N=t->N; /* stations */
 int M=t->M; /* clusters */


 double xr[8]; /* residuals */
 complex double G1[4],G2[4],C[4],T1[4],T2[4];
 double pp[8];
 double dsum;
 int cli,tpchunk,pstart,nchunk,tilesperchunk,stci,ttile,tptile,poff;
 double nu=t->robust_nu;

 int Ntilebase=t->Nbase*t->tilesz;
 int nchan,chanoff=4*M*Ntilebase;

 /* iterate over each paramter */
 for (ci=t->g_start; ci<=t->g_end; ++ci) {
    t->g[ci]=0.0;
    /* find station and parameter corresponding to this value of ci */
    /* this parameter should correspond to the right baseline (x tilesz)
        to contribute to the derivative */
    cli=0;
    while((cli<M) && (ci<t->carr[cli].p[0] || ci>t->carr[cli].p[0]+8*N*t->carr[cli].nchunk-1)) {
     cli++;
    }
   /* now either cli>=M: cluster not found 
       or cli<M and cli is the right cluster */
   if (cli==M && ci>=t->carr[cli-1].p[0] && ci<=t->carr[cli-1].p[0]+8*N*t->carr[cli-1].nchunk-1) {
    cli--;
   }

   if (cli<M) {
    /* right parameter offset */
    stci=ci-t->carr[cli].p[0];
 
    stc=(stci%(8*N))/8; /* 0..N-1 */
    /* make sure this baseline contribute to this parameter */
    tpchunk=stci/(8*N);
    nchunk=t->carr[cli].nchunk;
    pstart=t->carr[cli].p[0];
    tilesperchunk=(t->tilesz+nchunk-1)/nchunk;


    /* iterate over all baselines and accumulate sum */
    for (nb=0; nb<Ntilebase; ++nb) {
     /* which tile is this ? */
     ttile=nb/t->Nbase;
     /* which chunk this tile belongs to */
     tptile=ttile/tilesperchunk;
     /* now tptile has to match tpchunk, otherwise ignore calculation */
     if (tptile==tpchunk) {

     sta1=t->barr[nb].sta1;
     sta2=t->barr[nb].sta2;
    
     if (((stc==sta1)||(stc==sta2))&& !t->barr[nb].flag) {
      /* this baseline has a contribution */
      /* which paramter of this station */
      stoff=(stci%(8*N))%8; /* 0..7 */
      /* which cluster */
      stm=cli; /* 0..M-1 */

      /* exact expression for derivative  
       for Gaussian \sum( y_i - f_i(\theta))^2
         2 real( vec^H(residual_this_baseline) 
            * vec(-J_{pm}C_{pqm} J_{qm}^H)
        where m: chosen cluster
        J_{pm},J_{qm} Jones matrices for baseline p-q
        depending on the parameter, J ==> E 
        E: zero matrix, except 1 at location of m
      \sum( 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 

       for robust \sum( log(1+ (y_i-f_i(\theta))^2/\nu) )
       all calculations are like for the Gaussian case, except
       when taking summation
      \sum( 1/(\nu+(y_i-f_i)^2) 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 
 
      so additonal multiplication by 1/(\nu+(y_i-f_i)^2)
     */
     memset(pp,0,sizeof(double)*8); 
     if (stc==sta1) {
       /* this station parameter gradient */
       pp[stoff]=1.0;
       memset(G1,0,sizeof(complex double)*4); 
       G1[0]=pp[0]+_Complex_I*pp[1];
       G1[1]=pp[2]+_Complex_I*pp[3];
       G1[2]=pp[4]+_Complex_I*pp[5];
       G1[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta2*8;
       G2[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G2[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G2[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+5]);
       G2[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);
     } else if (stc==sta2) {
       memset(G2,0,sizeof(complex double)*4); 
       pp[stoff]=1.0;
       G2[0]=pp[0]+_Complex_I*pp[1];
       G2[1]=pp[2]+_Complex_I*pp[3];
       G2[2]=pp[4]+_Complex_I*pp[5];
       G2[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta1*8;
       G1[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G1[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G1[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+5]);
       G1[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);
     }

     for (nchan=0; nchan<t->Nchan; nchan++) {
     /* read in residual vector, (real,imag) separately */
     xr[0]=t->x[nchan*8*Ntilebase + nb*8];
     xr[1]=t->x[nchan*8*Ntilebase + nb*8+1];
     xr[2]=t->x[nchan*8*Ntilebase + nb*8+2];
     xr[3]=t->x[nchan*8*Ntilebase + nb*8+3];
     xr[4]=t->x[nchan*8*Ntilebase + nb*8+4];
     xr[5]=t->x[nchan*8*Ntilebase + nb*8+5];
     xr[6]=t->x[nchan*8*Ntilebase + nb*8+6];
     xr[7]=t->x[nchan*8*Ntilebase + nb*8+7];

     /* read in coherency */
     C[0]=t->coh[nchan*chanoff+4*M*nb+4*stm];
     C[1]=t->coh[nchan*chanoff+4*M*nb+4*stm+1];
     C[2]=t->coh[nchan*chanoff+4*M*nb+4*stm+2];
     C[3]=t->coh[nchan*chanoff+4*M*nb+4*stm+3];


     /* T1=G1*C */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* calculate product xr*vec(J_p C J_q^H )/(nu+residual^2) */
     dsum=xr[0]*creal(T2[0])/(nu+xr[0]*xr[0]);
     dsum+=xr[1]*cimag(T2[0])/(nu+xr[1]*xr[1]);
     dsum+=xr[2]*creal(T2[1])/(nu+xr[2]*xr[2]);
     dsum+=xr[3]*cimag(T2[1])/(nu+xr[3]*xr[3]);
     dsum+=xr[4]*creal(T2[2])/(nu+xr[4]*xr[4]);
     dsum+=xr[5]*cimag(T2[2])/(nu+xr[5]*xr[5]);
     dsum+=xr[6]*creal(T2[3])/(nu+xr[6]*xr[6]);
     dsum+=xr[7]*cimag(T2[3])/(nu+xr[7]*xr[7]);

     /* accumulate sum NOTE
     its important to get the sign right,
     depending on res=data-model or res=model-data  */
     t->g[ci]+=-2.0*(dsum);
     }
     }
     }
    }
   }
 }


 return NULL;
}

/* calculate gradient */
/* func: vector function to predict model
   p: parameter values size m x 1 (at which grad is calculated)
   g: gradient size m x 1 
   xo: observed data size n x 1
   robust_nu: nu in T distribution
   n: size of vector function (multi channel)

   adata:  additional data passed to the function

   y,z,rho: null for normal cost, otherwise consensus mode 
*/
static int
func_grad_robust_multifreq(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *g, double *xo, int m, int n, double *y, double *z, double *rho, void *adata) {

  double *x; /* array to store residual */
  int ci;
  me_data_t *dp=(me_data_t*)adata;

  int Nt=dp->Nt;

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_grad_t *threaddata;

  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }

  /* evaluate func once, store in x, and create threads */
  /* and calculate the residual x=xo-func */
  func(p,x,m,n,adata);
  /* calculate x<=x-xo : only in the valid region */
  my_daxpy(n,xo,-1.0,x);

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_grad_t*)malloc((size_t)Nt*sizeof(thread_data_grad_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  int nth,Nparm;

  /* parameters per thread */
  Nparm=(m+Nt-1)/Nt;

  /* each thread will calculate derivative of part of 
     parameters */
  ci=0;
  for (nth=0;  nth<Nt; nth++) {
   threaddata[nth].Nbase=dp->Nbase;
   threaddata[nth].tilesz=dp->tilesz;
   threaddata[nth].barr=dp->barr;
   threaddata[nth].carr=dp->carr;
   threaddata[nth].M=dp->M;
   threaddata[nth].N=dp->N;
   threaddata[nth].coh=dp->coh;
   threaddata[nth].m=m;
   threaddata[nth].n=n;
   threaddata[nth].x=x;
   threaddata[nth].p=p;
   threaddata[nth].g=g;
   threaddata[nth].robust_nu=dp->robust_nu;
   threaddata[nth].g_start=ci;
   threaddata[nth].g_end=ci+Nparm-1;
   if (threaddata[nth].g_end>=m) {
    threaddata[nth].g_end=m-1;
   }
  
   threaddata[nth].Nchan=dp->Nchan;
   ci=ci+Nparm;
    
   pthread_create(&th_array[nth],&attr,cpu_calc_deriv_multifreq,(void*)(&threaddata[nth]));
   //pthread_create(&th_array[nth],&attr,cpu_calc_deriv_huber,(void*)(&threaddata[nth]));
  }

  /* now wait for threads to finish */
  for(nth=0; nth<Nt; nth++) {
   pthread_join(th_array[nth],NULL);
  }

  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);

  free(x);

  /* gradient needs more terms y+rho(x-z) in consensus mode */
  if (y && z && rho) {
   double *xp;
   if(m !=dp->Mt*dp->N*8) {
     fprintf(stderr,"%s: %d: incorrect parameter lengths\n",__FILE__,__LINE__);
     exit(1);
   }
   if ((xp=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   /* find xp=p-z */
   my_dcopy(m,p,1,xp,1);
   my_daxpy(m,z,-1.0,xp);
   /* now multiply by rho */
   for (ci=0; ci<dp->Mt; ci++) {
    my_dscal(8*dp->N,rho[ci],&xp[8*dp->N*ci]);
   }

   /* now add y + rho(p-z) to g (-ve for -ve g)*/
   my_daxpy(m,y,-1.0,g);
   my_daxpy(m,xp,-1.0,g);
   free(xp);
  }

  return 0;
}

/* gradient function */
static void
robust_grad_func_multifreq(double *p, double *g, int m, void *adata) {
  wrapper_me_data_batchmode_t *dp=(wrapper_me_data_batchmode_t*)adata;
  double *x=dp->x; /* input data : changed at every batch */
  int n=dp->n; /* size also changed per batch */
  func_grad_robust_multifreq(&minimize_viz_full_multifreq,p,g,x, m, n, dp->y, dp->z, dp->rho, dp->adata);
}

/* caller function for minibatch mode */
/* note that tilesz used here will be normally smaller than the orignal full batch size 
   and x/coh are for Nchan channels, instead of 1 channel */
int
bfgsfit_minibatch_visibilities(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {

  /* FIXME: need absolute tile offset and total number of tiles also 
      for hybrid solutions */
  me_data_t lmdata;

  /*  no. of true parameters */
  int m=N*Mt*8;
  /* no of data */
  int n=Nbase*tilesz*Nf*8;

  /* setup the ME data */
  lmdata.u=u;
  lmdata.v=v;
  lmdata.w=w;
  lmdata.Nbase=Nbase;
  lmdata.tilesz=tilesz;
  lmdata.N=N;
  lmdata.barr=barr;
  lmdata.carr=carr;
  lmdata.M=M;
  lmdata.Mt=Mt;
  lmdata.Nt=Nt;
  lmdata.coh=coh;
  lmdata.Nchan=Nf; /* multichannel data */

  /* starting guess of robust nu */
  lmdata.robust_nu=robust_nu;

  /* wrapper struct to pass input */
  wrapper_me_data_batchmode_t data1;
  data1.x=x;
  data1.n=n;
  data1.adata=&lmdata;
  /* other fiels are none */
  data1.y=data1.z=data1.rho=0;
 
  /* the following fields in persistent data are not used here FIXME */
  indata->offset=0; /* no offset */
  indata->nlen=0 ; /* full data length */
  *res_0=robust_cost_func_multifreq(p, m, &data1);
  lbfgs_fit(&robust_cost_func_multifreq,&robust_grad_func_multifreq,p,m,max_lbfgs,lbfgs_m,&data1,indata);

  *res_1=robust_cost_func_multifreq(p, m, &data1);
  double invn=(double)1.0/n;
  *res_0 *=invn;
  *res_1 *=invn;

  return 0;
}





/* minibatch mode with consensus */
int
bfgsfit_minibatch_consensus(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, double *y, double *z, double *rho, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {

  me_data_t lmdata;

  /*  no. of true parameters */
  int m=N*Mt*8;
  /* no of data */
  int n=Nbase*tilesz*Nf*8;

  /* setup the ME data */
  lmdata.u=u;
  lmdata.v=v;
  lmdata.w=w;
  lmdata.Nbase=Nbase;
  lmdata.tilesz=tilesz;
  lmdata.N=N;
  lmdata.barr=barr;
  lmdata.carr=carr;
  lmdata.M=M;
  lmdata.Mt=Mt;
  lmdata.Nt=Nt;
  lmdata.coh=coh;
  lmdata.Nchan=Nf; /* multichannel data */

  /* starting guess of robust nu */
  lmdata.robust_nu=robust_nu;

  /* wrapper struct to pass input */
  wrapper_me_data_batchmode_t data1;
  data1.x=x;
  data1.n=n;
  data1.adata=&lmdata;

  /* other fiels */
  data1.y=y;
  data1.z=z;
  data1.rho=rho;

  /******************************/
  /* check gradient */
/*  double p0=p[0]; double eps=1e-6;
  p[0]=p0+eps;
  double f00=robust_cost_func_multifreq(p, m, &data1);
  p[0]=p0-eps;
  double f11=robust_cost_func_multifreq(p, m, &data1);
  p[0]=p0;
  double *gg;
   if ((gg=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
  robust_grad_func_multifreq(p,gg,m,&data1);
  
  printf("Numerical grad= %lf , %lf =%lf, analytical=%lf\n",f00,f11,(f00-f11)/(2.0*eps),gg[0]);
  free(gg);
*/
  /******************************/

  /* the following fields in persistent data are not used here FIXME */
  indata->offset=0; /* no offset */
  indata->nlen=0 ; /* full data length */


  *res_0=robust_cost_func_multifreq(p, m, &data1);
  lbfgs_fit(&robust_cost_func_multifreq,&robust_grad_func_multifreq,p,m,max_lbfgs,lbfgs_m,&data1,indata);

  *res_1=robust_cost_func_multifreq(p, m, &data1);
  double invn=(double)1.0/n;
  *res_0 *=invn;
  *res_1 *=invn;

  return 0;
}
