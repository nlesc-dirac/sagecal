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
 int Nbase=(t->Nbase)*(t->tilesz);


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
    for (nb=0; nb<Nbase; ++nb) {
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
     t->g[ci]+=2.0*(dsum);
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
   adata:  additional data passed to the function
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
func_grad_robust(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *g, double *xo, int m, int n, void *adata) {
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


  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  /* evaluate func once, store in x, and create threads */
  /* and calculate the residual x=xo-func */
  func(p,x,m,n,adata);
  /* calculate x<=x-xo */
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
   ci=ci+Nparm;
   pthread_create(&th_array[nth],&attr,cpu_calc_deriv_robust,(void*)(&threaddata[nth]));
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
/*************************************** END ROBUST ***************************/



/* worker thread for a cpu */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
cpu_calc_deriv(void *adata) {
 thread_data_grad_t *t=(thread_data_grad_t*)adata;

 int ci,nb;
 int stc,stoff,stm,sta1,sta2;
 int N=t->N; /* stations */
 int M=t->M; /* clusters */
 int Nbase=(t->Nbase)*(t->tilesz);


 complex double xr[4]; /* residuals */
 complex double G1[4],G2[4],C[4],T1[4],T2[4];
 double pp[8];
 complex double csum;
 int cli,tpchunk,pstart,nchunk,tilesperchunk,stci,ttile,tptile,poff;

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
    for (nb=0; nb<Nbase; ++nb) {
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
         2 real( vec^H(residual_this_baseline) 
            * vec(-J_{pm}C_{pqm} J_{qm}^H)
        where m: chosen cluster
        J_{pm},J_{qm} Jones matrices for baseline p-q
        depending on the parameter, J ==> E 
        E: zero matrix, except 1 at location of m
   
       residual : in x[8*nb:8*nb+7]
       C coh: in coh[8*M*nb+m*8:8*M*nb+m*8+7] (double storage)
           coh[4*M*nb+4*m:4*M*nb+4*m+3] (complex storage)
       J_p,J_q: in p[sta1*8+m*8*N: sta1*8+m*8*N+7]
        and p[sta2*8+m*8*N: sta2*8+m*8*N+ 7]
     */
     /* read in residual vector, conjugated */
     xr[0]=(t->x[nb*8])-_Complex_I*(t->x[nb*8+1]);
     xr[1]=(t->x[nb*8+2])-_Complex_I*(t->x[nb*8+3]);
     xr[2]=(t->x[nb*8+4])-_Complex_I*(t->x[nb*8+5]);
     xr[3]=(t->x[nb*8+6])-_Complex_I*(t->x[nb*8+7]);

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

     /* calculate product xr*vec(J_p C J_q^H ) */
     csum=xr[0]*T2[0];
     csum+=xr[1]*T2[1];
     csum+=xr[2]*T2[2];
     csum+=xr[3]*T2[3];

     /* accumulate sum */
     t->g[ci]+=-2.0*creal(csum);
     }
     }
    }
   }
 }


 return NULL;
}

#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
func_grad(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *g, double *xo, int m, int n,void *adata) {
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


  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  /* evaluate func once, store in x, and create threads */
  /* and calculate the residual x=xo-func */
  func(p,x,m,n,adata);
  /* calculate x<=x-xo */
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

  int nth,nth1,Nparm;

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
   threaddata[nth].g_start=ci;
   threaddata[nth].g_end=ci+Nparm-1;
   if (threaddata[nth].g_end>=m) {
    threaddata[nth].g_end=m-1;
   }
   ci=ci+Nparm;
   pthread_create(&th_array[nth],&attr,cpu_calc_deriv,(void*)(&threaddata[nth]));
  }

  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);

  free(x);
  return 0;
}





typedef struct wrapper_me_data_t_ {
 me_data_t *adata;
 void (*func)(double *p, double *hx, int m, int n, void *adata);
 double *x; /* nx1 data vector */
 int n;
} wrapper_me_data_t;

static double
cost_func(double *p, int m, void *adata) {
  wrapper_me_data_t *dp=(wrapper_me_data_t*)adata;
  double *x=dp->x; /* input data */
  int n=dp->n;
  double *xmodel;

  if ((xmodel=(double*)calloc((size_t)n,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* predict model */
  minimize_viz_full_pth(p, xmodel, m, n, dp->adata);
  /* find ||x-xmodel||^2 */ 
  my_daxpy(n,x,-1.0,xmodel);
  double f0=my_dnrm2(n,xmodel);
  f0*=f0;

  free(xmodel);
  return f0;
}

static void 
grad_func(double *p, double *g, int m, void *adata) {
  wrapper_me_data_t *dp=(wrapper_me_data_t*)adata;
  double *x=dp->x; /* input data */
  int n=dp->n;

  func_grad(&minimize_viz_full_pth,p,g,x, m, n, dp->adata);
}


static double
robust_cost_func(double *p, int m, void *adata) {
  wrapper_me_data_t *dp=(wrapper_me_data_t*)adata;
  me_data_t *d1=(me_data_t *)dp->adata;
  double *x=dp->x; /* input data */
  int n=dp->n;
  double *xmodel;

  if ((xmodel=(double*)calloc((size_t)n,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* predict model */
  minimize_viz_full_pth(p, xmodel, m, n, dp->adata);

  double f0=func_robust(x,xmodel,n, d1->robust_nu, d1->Nt);

  free(xmodel);
  return f0;
}

static void 
robust_grad_func(double *p, double *g, int m, void *adata) {
  wrapper_me_data_t *dp=(wrapper_me_data_t*)adata;
  double *x=dp->x; /* input data */
  int n=dp->n;

  func_grad_robust(&minimize_viz_full_pth,p,g,x, m, n, dp->adata);
}

int
lbfgs_fit_wrapper(
   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads,
 void *adata) {
  wrapper_me_data_t data1;
  data1.adata=(me_data_t *)adata;
  data1.x=x; 
  data1.n=n;

   
  lbfgs_fit(&cost_func,&grad_func,p,m,itmax,M,&data1,NULL);
  return 0;
}

int
lbfgs_fit_robust_wrapper(
   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads,
 void *adata) {
  wrapper_me_data_t data1;
  data1.adata=(me_data_t *)adata;
  data1.x=x; 
  data1.n=n;

   
  lbfgs_fit(&robust_cost_func,&robust_grad_func,p,m,itmax,M,&data1,NULL);
  return 0;
}
