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
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "Dirac.h"

//#define DEBUG

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

/********************** sage minimization ***************************/
/* worker thread function for prediction */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
predict_threadfn_withgain(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,sta1,sta2;
 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 cm=(t->clus);
 int Ntilebase=(t->Nbase)*(t->tilesz);
 int px;
 double *pm;

 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   memset(&(t->x[8*ci]),0,sizeof(double)*8);

      /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   px=(ci+t->boff)/((Ntilebase+t->carr[cm].nchunk-1)/t->carr[cm].nchunk);
   pm=&(t->p[t->carr[cm].p[px]]);

     /* gains for this cluster, for sta1,sta2 */
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

 return NULL;
}


/* minimization function (multithreaded) */
/* p: size mx1 parameters
   x: size nx1 data calculated
   data: extra info needed */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
mylm_fit_single_pth(double *p, double *x, int m, int n, void *data) {

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

  int Nbase=(dp->Nbase);
  int tilesz=(dp->tilesz);

  int Nbase1=Nbase*tilesz;

  /* calculate min baselines a thread can handle */
  //Nthb0=ceil((double)Nbase1/(double)Nt);
  Nthb0=(Nbase1+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
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
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].N=dp->N;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].p=p;
    threaddata[nth].clus=(dp->clus);
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*ci]);
    
    //printf("thread %d predict  data from %d baselines %d\n",nth,8*ci,Nthb);
    pthread_create(&th_array[nth],&attr,predict_threadfn_withgain,(void*)(&threaddata[nth]));
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

 return;
}


/* worker thread function for prediction */
/* assuming no hybrid parameters */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
predict_threadfn_withgain0(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,sta1,sta2;
 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 cm=(t->clus);
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   memset(&(t->x[8*ci]),0,sizeof(double)*8);

      /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
     /* gains for this cluster, for sta1,sta2 */
     G1[0]=(t->p[sta1*8])+_Complex_I*(t->p[sta1*8+1]);
     G1[1]=(t->p[sta1*8+2])+_Complex_I*(t->p[sta1*8+3]);
     G1[2]=(t->p[sta1*8+4])+_Complex_I*(t->p[sta1*8+5]);
     G1[3]=(t->p[sta1*8+6])+_Complex_I*(t->p[sta1*8+7]);
     G2[0]=(t->p[sta2*8])+_Complex_I*(t->p[sta2*8+1]);
     G2[1]=(t->p[sta2*8+2])+_Complex_I*(t->p[sta2*8+3]);
     G2[2]=(t->p[sta2*8+4])+_Complex_I*(t->p[sta2*8+5]);
     G2[3]=(t->p[sta2*8+6])+_Complex_I*(t->p[sta2*8+7]);


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

 return NULL;
}


/* minimization function (multithreaded) : not considering 
  hybrid parameter space */
/* p: size mx1 parameters
   x: size nx1 data calculated
   data: extra info needed */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
mylm_fit_single_pth0(double *p, double *x, int m, int n, void *data) {

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

  int Nbase1=(dp->Nbase)*(dp->tilesz);
  int boff=(dp->Nbase)*(dp->tileoff);

  /* calculate min baselines a thread can handle */
  //Nthb0=ceil((double)Nbase1/(double)Nt);
  Nthb0=(Nbase1+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
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

    threaddata[nth].boff=ci+boff;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=dp->barr;
    threaddata[nth].u=&(dp->u[ci]);
    threaddata[nth].v=&(dp->v[ci]);
    threaddata[nth].w=&(dp->w[ci]);
    threaddata[nth].carr=dp->carr;
    threaddata[nth].M=dp->M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].N=dp->N;
    threaddata[nth].p=p; /* note the difference: here p assumes no hybrid */
    threaddata[nth].clus=(dp->clus);
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*(ci+boff)]);
    
    //printf("thread %d predict  data from %d baselines %d\n",nth,8*ci,Nthb);
    pthread_create(&th_array[nth],&attr,predict_threadfn_withgain0,(void*)(&threaddata[nth]));
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

 return;
}



/* worker thread function for prediction */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
jacobian_threadfn(void *data) {
 thread_data_jac_t *t=(thread_data_jac_t*)data;
 
 int ci,cm,cn,sta1,sta2,col;
 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 double pp1[8],pp2[8];
 int M=(t->M);
 cm=(t->clus);
 int stc,stoff;

 /* Loop order to minimize cache misses */
 /* we calculate the jacobian (nxm) columns [startc...endc] */
 for (col=t->start_col; col<=t->end_col; col++) {
 /* iterate over row */
 for (ci=0; ci<t->Nb; ci++) {

   /* if this baseline is flagged,
     or if this parameter does not belong to sta1 or sta2
     we do not compute */
   stc=col/8; /* 0..N-1 */
   /* stations for this baseline */
   sta1=t->barr[ci].sta1;
   sta2=t->barr[ci].sta2;

   /* change order for checking condition to minimize cache misses 
     since sta2 will appear more, first check that ??? */
   if ( ((stc==sta2)||(stc==sta1)) && (!t->barr[ci].flag) ) {

      /* use pre calculated values */
      C[0]=t->coh[4*M*ci+4*cm];
      C[1]=t->coh[4*M*ci+4*cm+1];
      C[2]=t->coh[4*M*ci+4*cm+2];
      C[3]=t->coh[4*M*ci+4*cm+3];

     /* which parameter exactly 0..7 */
     stoff=col%8;
     //printf("sta1=%d,sta2=%d,stc=%d,off=%d,col=%d,param=%d\n",sta1,sta2,stc,col%8,col,stc*8+stoff);
     if (stc==sta1) {
      for (cn=0; cn<8; cn++) {
       pp1[cn]=0.0;
       pp2[cn]=t->p[sta2*8+cn];
      }
      pp1[stoff]=1.0;
     } else if (stc==sta2) {
      for (cn=0; cn<8; cn++) {
       pp2[cn]=0.0;
       pp1[cn]=t->p[sta1*8+cn];
      }
      pp2[stoff]=1.0;
     }
     /* gains for this cluster, for sta1,sta2 */
     G1[0]=pp1[0]+_Complex_I*pp1[1];
     G1[1]=pp1[2]+_Complex_I*pp1[3];
     G1[2]=pp1[4]+_Complex_I*pp1[5];
     G1[3]=pp1[6]+_Complex_I*pp1[7];
     G2[0]=pp2[0]+_Complex_I*pp2[1];
     G2[1]=pp2[2]+_Complex_I*pp2[3];
     G2[2]=pp2[4]+_Complex_I*pp2[5];
     G2[3]=pp2[6]+_Complex_I*pp2[7];

     /* form G1*C*G2' */
     /* T1=G1*C  */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* add to baseline visibilities */
     /* NOTE: row major order */
     t->jac[col+(t->m)*8*ci]=creal(T2[0]);
     t->jac[col+(t->m)*(8*ci+1)]=cimag(T2[0]);
     t->jac[col+(t->m)*(8*ci+2)]=creal(T2[1]);
     t->jac[col+(t->m)*(8*ci+3)]=cimag(T2[1]);
     t->jac[col+(t->m)*(8*ci+4)]=creal(T2[2]);
     t->jac[col+(t->m)*(8*ci+5)]=cimag(T2[2]);
     t->jac[col+(t->m)*(8*ci+6)]=creal(T2[3]);
     t->jac[col+(t->m)*(8*ci+7)]=cimag(T2[3]);

   } 
   }
 }

 return NULL;
}

/* jacobian function (multithreaded) */
/* p: size mx1 parameters
   jac: size nxm jacobian to be calculated (row major)
   data: extra info needed */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
mylm_jac_single_pth(double *p, double *jac, int m, int n, void *data) {

  me_data_t *dp=(me_data_t*)data;
  /* u,v,w : size Nbase*tilesz x 1  x: size Nbase*8*tilesz x 1 */
  /* barr: size Nbase*tilesz x 1 carr: size Mx1 */
  /* pp: size 8*N*M x 1 */
  /* pm: size Mx1 of double */

  int nth,ci;

  /* no of threads */
  int Nt=(dp->Nt);
  int Nthcol;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_jac_t *threaddata;

  int Nbase=(dp->Nbase)*(dp->tilesz);
  int boff=(dp->Nbase)*(dp->tileoff);

  /* calculate min columns of the jacobian one thread can handle */
  Nthcol=(m+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_jac_t*)malloc((size_t)Nt*sizeof(thread_data_jac_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }
  /* set jacobian to all zeros */
  memset(jac,0,sizeof(double)*n*m);
  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt; nth++) {
    /* this thread will handle columns [ci:min(m-1,ci+Nthcol0-1)] */
    threaddata[nth].Nb=Nbase; /* n=Nbase*8 */
    threaddata[nth].n=n; /* n=Nbase*8 */
    threaddata[nth].m=m; /* no of parameters */
    threaddata[nth].barr=&(dp->barr[boff]);
    threaddata[nth].u=dp->u;
    threaddata[nth].v=dp->v;
    threaddata[nth].w=dp->w;
    threaddata[nth].carr=dp->carr;
    threaddata[nth].M=dp->M;
    threaddata[nth].jac=jac; /* NOTE: jacobian is in row major order */
    threaddata[nth].N=dp->N;
    threaddata[nth].p=p;
    threaddata[nth].clus=(dp->clus);
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*(boff)]);
    threaddata[nth].start_col=ci;
    threaddata[nth].end_col=ci+Nthcol-1;
    if (threaddata[nth].end_col>=m) {
     threaddata[nth].end_col=m-1;
    }
    
    //printf("thread %d calculate cols %d to %d\n",nth,threaddata[nth].start_col, threaddata[nth].end_col);
    pthread_create(&th_array[nth],&attr,jacobian_threadfn,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthcol;
  }

  /* now wait for threads to finish */
  for(nth=0; nth<Nt; nth++) {
   pthread_join(th_array[nth],NULL);
  }

 pthread_attr_destroy(&attr);

 free(th_array);
 free(threaddata);

 return;
}



/******************** end sage minimization *****************************/

void
print_levmar_info(double e_0, double e_final,int itermax, int info, int fnum, int jnum, int lnum) {
 printf("\nOptimization terminated with %d iterations, reason: ",itermax);
 switch(info) {
  case 1:
   printf("stopped by small gradient J^T e.\n");
   break;
  case 2:
   printf("stopped by small Dp.\n");
   break;
  case 3:
   printf("stopped by itmax.\n");
   break;
  case 4:
   printf("singular matrix. Restart from current p with increased mu.\n");
   break;
  case 5:
   printf("no further error reduction is possible. Restart with increased mu.\n");
   break;
  case 6:
   printf("stopped by small ||e||_2.\n");
   break;
  case 7:
   printf("stopped by invalid (i.e. NaN or Inf) \"func\" values. This is a user error.\n");
   break;
  default:
   printf("Unknown.\n");
   break;
 }
 printf("Error from %lf to %lf, Evaluations: %d functions %d Jacobians %d Linear systems\n",sqrt(e_0), sqrt(e_final),fnum,jnum,lnum);
}


/******************** full minimization *****************************/
/* worker thread function for prediction */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
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

#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
void
minimize_viz_full_pth(double *p, double *x, int m, int n, void *data) {

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

  int Nbase1=(dp->Nbase)*(dp->tilesz);

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase1+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
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
int
sagefit_visibilities(double *u, double *v, double *w, double *x, int N,   
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize,  double *mean_nu, double *res_0, double *res_1) {
  /* u,v,w : size Nbase*tilesz x 1  x: size Nbase*8*tilesz x 1 */
  /* barr: size Nbase*tilesz x 1 carr: size Mx1 */
  /* pp: size 8*N*M x 1 */
  /* pm: size Mx1 of double */


  int  ci,cj,ck,tcj;
  double *p; // parameters: m x 1
  int m, n;
  double opts[CLM_OPTS_SZ], info[CLM_INFO_SZ];
  me_data_t lmdata;

  double *xdummy,*xsub;
  double *nerr; /* array to store cost reduction per cluster */
  double *robust_nuM;
  int weighted_iter,this_itermax,total_iter;
  double total_err;

  int ntiles,tilechunk;
  double init_res,final_res;

  opts[0]=CLM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-20;
  opts[4]=-CLM_DIFF_DELTA;

  /*  no. of true parameters */
  m=N*Mt*8;
  /* no of data */
  n=Nbase*tilesz*8;

  /* use full parameter space */
  p=pp;
  lmdata.clus=-1;
  /* setup data for lmfit */
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
  lmdata.freq0=&freq0;
  lmdata.Nt=Nt;
  lmdata.coh=coh;

  /* starting guess of robust nu */
  double robust_nu0=nulow;
  lmdata.robust_nu=robust_nu0;

  if ((xsub=(double*)calloc((size_t)(n),sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((xdummy=(double*)calloc((size_t)(n),sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((nerr=(double*)calloc((size_t)(M),sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if (solver_mode==SM_OSLM_OSRLM_RLBFGS || solver_mode==SM_RLM_RLBFGS || solver_mode==SM_RTR_OSRLM_RLBFGS || solver_mode==SM_NSD_RLBFGS) {
   if ((robust_nuM=(double*)calloc((size_t)(M),sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
   } 
  } else {
    robust_nuM=0;
  } 
  /* remember for each partition how much the cost function decreases
     in the next EM iteration, we allocate more LM iters to partitions
     where const function significantly decreases. So two stages
     1) equal LM iters (find the decrease) 2) weighted LM iters */
  weighted_iter=0;
  total_iter=M*max_iter; /* total iterations per EM */
  /* calculate current model and subtract from data */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata);
  memcpy(xdummy,x,(size_t)(n)*sizeof(double));
  my_daxpy(n, xsub, -1.0, xdummy);
  *res_0=my_dnrm2(n,xdummy)/(double)n;

  int iter_bar=(int)ceil((0.80/(double)M)*((double)total_iter));
  for (ci=0; ci<max_emiter; ci++) {
#ifdef DEBUG
printf("\n\nEM %d\n",ci);
#endif
    for (cj=0; cj<M; cj++) { /* iter per cluster */
     /* calculate max LM iter for this cluster */
     if (weighted_iter) {
       this_itermax=(int)((0.20*nerr[cj])*((double)total_iter))+iter_bar;
     } else {
       this_itermax=max_iter;
     }
#ifdef DEBUG
printf("\n\ncluster %d iter=%d\n",cj,this_itermax);
#endif
     if (this_itermax>0) {
     /* calculate contribution from hidden data, subtract from x
       actually, add the current model for this cluster to residual */
     lmdata.clus=cj;
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata);
     my_daxpy(n, xsub, 1.0, xdummy);
 
     tilechunk=(tilesz+carr[cj].nchunk-1)/carr[cj].nchunk;
     tcj=0;
     init_res=final_res=0.0;
    /* loop through hybrid parameter space */
     for (ck=0; ck<carr[cj].nchunk; ck++) {
       if (tcj+tilechunk<tilesz) {
         ntiles=tilechunk;
       } else {
         ntiles=tilesz-tcj;
       }

       lmdata.tilesz=ntiles;
       lmdata.tileoff=tcj;
       if(solver_mode==SM_OSLM_LBFGS) {
         if (ci==max_emiter-1){
           clevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, (void*)&lmdata);  
         } else {
           oslevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, randomize, (void*)&lmdata);  
         }
       } else if (solver_mode==SM_LM_LBFGS) {
         clevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, (void*)&lmdata);  
       } else if (solver_mode==SM_RLM_RLBFGS) {
         /* only the last EM iteration is robust */
         if (ci==max_emiter-1){
          lmdata.robust_nu=robust_nu0;
          rlevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, NULL, info, linsolv, Nt, nulow, nuhigh, (void*)&lmdata);  
          /* get updated value of robust_nu */
          robust_nuM[cj]+=lmdata.robust_nu;
          } else {
           oslevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, randomize, (void*)&lmdata);  
         }
       } else if (solver_mode==SM_OSLM_OSRLM_RLBFGS) {
         /* only the last EM iteration is robust */
         if (ci==max_emiter-1){
          lmdata.robust_nu=robust_nu0;
          osrlevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, NULL, info, linsolv, Nt,  nulow, nuhigh, randomize,  (void*)&lmdata);  
          /* get updated value of robust_nu */
          robust_nuM[cj]+=lmdata.robust_nu;
          } else {
           oslevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, randomize, (void*)&lmdata);  
         }
       } else if (solver_mode==SM_RTR_OSLM_LBFGS) { /* RTR */
            /* RSD+RTR */
           double Delta0=0.01; /* since previous timeslot used LM, use a very small TR radius because this solution will not be too far off */
           rtr_solve_nocuda(&p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+5, this_itermax+10, Delta0, Delta0*0.125, info, &lmdata);
       } else if (solver_mode==SM_RTR_OSRLM_RLBFGS) { /* RTR + Robust */
            /* RSD+RTR */
           if (!ci){
            lmdata.robust_nu=robust_nu0;
           } 
           double Delta0=0.01; /* since previous timeslot used LM, use a very small TR radius because this solution will not be too far off */
           rtr_solve_nocuda_robust(&p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+5, this_itermax+10, Delta0, Delta0*0.125, nulow, nuhigh, info, &lmdata);
           if (ci==max_emiter-1){
            robust_nuM[cj]+=lmdata.robust_nu;
           }
       } else if (solver_mode==SM_NSD_RLBFGS) { /* Nesterov's */
            /* NSD */
           if (!ci){
            lmdata.robust_nu=robust_nu0;
           } 
           nsd_solve_nocuda_robust(&p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+15, nulow, nuhigh, info, &lmdata);
           if (ci==max_emiter-1){
            robust_nuM[cj]+=lmdata.robust_nu;
           }
       } else { /* not used */
#ifndef USE_MIC
        fprintf(stderr,"%s: %d: undefined solver mode\n",__FILE__,__LINE__);
#endif
        exit(1);
       }
       init_res+=info[0];
       final_res+=info[1];

       tcj=tcj+tilechunk;
     }
#ifdef DEBUG
printf("residual init=%lf final=%lf\n\n",init_res,final_res);
#endif
     lmdata.tilesz=tilesz;
     /* catch -ve value here */
     if (init_res>0.0) {
      nerr[cj]=(init_res-final_res)/init_res;
      if (nerr[cj]<0.0) { nerr[cj]=0.0; }
     } else {
      nerr[cj]=0.0;
     }
     /* subtract current model */
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata);
     my_daxpy(n, xsub, -1.0, xdummy);
     /* if robust LM, calculate average nu over hybrid clusters */
     if ((solver_mode==SM_OSLM_OSRLM_RLBFGS || solver_mode==SM_RLM_RLBFGS || solver_mode==SM_RTR_OSRLM_RLBFGS || solver_mode==SM_NSD_RLBFGS) && (ci==max_emiter-1)) {
      robust_nuM[cj]/=(double)carr[cj].nchunk;
     }
    }
   }

   /* normalize nerr array so that the sum is 1 */
   total_err=my_dasum(M,nerr);
   if (total_err>0.0) {
    my_dscal(M, 1.0/total_err, nerr);
   }

   /* flip weighting flag */
   if (randomize) {
    weighted_iter=!weighted_iter;
   }
 }
  free(nerr);
  free(xdummy);
  if (solver_mode==SM_OSLM_OSRLM_RLBFGS || solver_mode==SM_RLM_RLBFGS || solver_mode==SM_RTR_OSRLM_RLBFGS || solver_mode==SM_NSD_RLBFGS) {
    /* calculate mean robust_nu over all clusters */
    robust_nu0=my_dasum(M,robust_nuM)/(double)M;
#ifdef DEBUG
    for (ci=0; ci<M; ci++) {
     printf("clus %d nu %lf\n",ci,robust_nuM[ci]);
    }
    printf("mean nu=%lf\n",robust_nu0);
#endif
    free(robust_nuM);
    if (robust_nu0<nulow) {
     robust_nu0=nulow;
    } else if (robust_nu0>nuhigh) {
     robust_nu0=nuhigh;
    }
  }

  if (max_lbfgs>0) {
#ifdef USE_MIC
  lmdata.Nt=32; /* FIXME increase threads for MIC */
#endif
  /* use LBFGS */
   if (solver_mode==SM_OSLM_OSRLM_RLBFGS || solver_mode==SM_RLM_RLBFGS ||  solver_mode==SM_RTR_OSRLM_RLBFGS || solver_mode==SM_NSD_RLBFGS) {
    lmdata.robust_nu=robust_nu0;
    if (lbfgs_m>0) {
     lbfgs_fit_robust_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
    } else if (lbfgs_m<0) { /* batch mode LBFGS if memory size is -ve */
     lbfgs_fit_robust_wrapper_minibatch(p, x, m, n, max_lbfgs, -lbfgs_m, gpu_threads, (void*)&lmdata);
    }
   } else {
    lbfgs_fit_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
   }
#ifdef USE_MIC
  lmdata.Nt=Nt; /* reset threads for MIC */
#endif
  }
  /* final residual calculation */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata);
  my_daxpy(n, xsub, -1.0, x);

  *mean_nu=robust_nu0;
  *res_1=my_dnrm2(n,x)/(double)n;

  free(xsub);
 /* if final residual > initial residual, 
    return -1, else 0
 */
 if (*res_1>*res_0) {
   return -1;
 }
 return 0;
}



/* struct and function for qsort */
typedef struct w_n_ {
 int i;
 double w;
} w_n;
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
weight_compare(const void *a, const void *b) {
  w_n *aw,*bw;
  aw=(w_n*)a;
  bw=(w_n*)b;
  if (aw->w>bw->w) return -1;
  if (aw->w==bw->w) return 0;

  return 1;
}

/* generate a random permutation of given integers */
/* note: free returned value after use */
/* n: no of entries, 
   weighter_iter: if 1, take weight into account
                  if 0, only generate a random permutation
   w: weights (size nx1): sort them in descending order and 
      give permutation accordingly
*/
int* 
random_permutation(int n, int weighted_iter, double *w) {
  int *p;
  int i;
  if ((p=(int*)malloc((size_t)(n)*sizeof(int)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if (!weighted_iter) {
   for (i = 0; i < n; ++i) {
    int j = rand() % (i + 1);
    p[i] = p[j];
    p[j] = i;
   }
  } else {
   /* we take weight into account */
   w_n *wn_arr;
   if ((wn_arr=(w_n*)malloc((size_t)(n)*sizeof(w_n)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
   }
   for (i=0; i<n; ++i) {
    wn_arr[i].i=i;
    wn_arr[i].w=w[i];
   }
   qsort(wn_arr,n,sizeof(w_n),weight_compare);
   /* now copy back indices in sorted order */
   for (i=0; i<n; ++i) {
     p[i]=wn_arr[i].i;
   }
   free(wn_arr);
  }
  return p;
}




int
bfgsfit_visibilities(double *u, double *v, double *w, double *x, int N,   
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double mean_nu, double *res_0, double *res_1) {

  double *p; // parameters: m x 1
  int m, n;
  me_data_t lmdata;

  double *xsub,*xdummy;


  /*  no. of true parameters */
  m=N*Mt*8;
  /* no of data */
  n=Nbase*tilesz*8;

  /* use full parameter space */
  p=pp;
  lmdata.clus=-1;
  /* setup data for lmfit */
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
  lmdata.freq0=&freq0;
  lmdata.Nt=Nt;
  lmdata.coh=coh;

  /* starting guess of robust nu */
  lmdata.robust_nu=mean_nu;

  if ((xsub=(double*)calloc((size_t)(n),sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((xdummy=(double*)calloc((size_t)(n),sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  /* calculate current model and subtract from data */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata);
  memcpy(xdummy,x,(size_t)(n)*sizeof(double));
  my_daxpy(n, xsub, -1.0, xdummy);
  *res_0=my_dnrm2(n,xdummy)/(double)n;

  if (max_lbfgs>0) {
#ifdef USE_MIC
  lmdata.Nt=64; /* increase threads for MIC */
#endif
  /* use LBFGS */
   if (solver_mode==SM_RLM_RLBFGS || solver_mode==SM_OSLM_OSRLM_RLBFGS || solver_mode==SM_RTR_OSRLM_RLBFGS || solver_mode==SM_NSD_RLBFGS ) {
    lmdata.robust_nu=mean_nu;
    lbfgs_fit_robust_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata); 
   } else {
    lbfgs_fit_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
   }
#ifdef USE_MIC
  lmdata.Nt=Nt; /* reset threads for MIC */
#endif
  }
  /* final residual calculation */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata);
  my_daxpy(n, xsub, -1.0, x);

  *res_1=my_dnrm2(n,x)/(double)n;

  free(xsub);
  free(xdummy);
 /* if final residual > initial residual, 
    return -1, else 0
 */
 if (*res_1>*res_0) {
   return -1;
 }
 return 0;
}


#ifdef USE_MIC
/* wrapper function with bitwise copyable carr[] for MIC */
/* nchunks: Mx1 array of chunk sizes for each cluster */
/* pindex: Mt x 1 array of index of solutions for each cluster  in pp */
int
sagefit_visibilities_mic(double *u, double *v, double *w, double *x, int N,   
   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *mean_nu, double *res_0, double *res_1) {

  clus_source_t *carr;
  /* create a dummy carr[] structure to pass on */
  if ((carr=(clus_source_t*)calloc((size_t)M,sizeof(clus_source_t)))==0) {
    exit(1);
  }
  int ci,cj,retval; 
  /* only need two fields in carr[] */
  cj=0;
  for (ci=0; ci<M; ci++)  {
    /* fill dummy values for not needed fields */
    carr[ci].N=1;
    carr[ci].id=ci;
    carr[ci].ll=carr[ci].mm=carr[ci].nn=carr[ci].sI=NULL;
    carr[ci].sI0=carr[ci].f0=carr[ci].spec_idx=carr[ci].spec_idx1=carr[ci].spec_idx2=NULL;

    carr[ci].nchunk=nchunks[ci];
    /* just point to original array for index */
    carr[ci].p=&(pindex[cj]);
    cj+=carr[ci].nchunk;
  }

  retval=sagefit_visibilities(u, v, w, x, N, Nbase, tilesz,  barr,  carr, coh, M, Mt, freq0, fdelta, pp, uvmin, Nt, max_emiter, max_iter, max_lbfgs, lbfgs_m, gpu_threads, linsolv,solver_mode,nulow, nuhigh,randomize, mean_nu, res_0, res_1);

  /* free dummy carr[] */
  free(carr);

  return retval;
} 


/* wrapper function with bitwise copyable carr[] for MIC */
/* nchunks: Mx1 array of chunk sizes for each cluster */
/* pindex: Mt x 1 array of index of solutions for each cluster  in pp */
int
bfgsfit_visibilities_mic(double *u, double *v, double *w, double *x, int N,   
   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode,double nu_mean, double *res_0, double *res_1) {

  clus_source_t *carr;
  /* create a dummy carr[] structure to pass on */
  if ((carr=(clus_source_t*)calloc((size_t)M,sizeof(clus_source_t)))==0) {
    exit(1);
  }
  int ci,cj,retval; 
  /* only need two fields in carr[] */
  cj=0;
  for (ci=0; ci<M; ci++)  {
    /* fill dummy values for not needed fields */
    carr[ci].N=1;
    carr[ci].id=ci;
    carr[ci].ll=carr[ci].mm=carr[ci].nn=carr[ci].sI=NULL;
    carr[ci].sI0=carr[ci].f0=carr[ci].spec_idx=carr[ci].spec_idx1=carr[ci].spec_idx2=NULL;

    carr[ci].nchunk=nchunks[ci];
    /* just point to original array for index */
    carr[ci].p=&(pindex[cj]);
    cj+=carr[ci].nchunk;
  }

  retval=bfgsfit_visibilities(u, v, w, x, N, Nbase, tilesz,  barr,  carr, coh, M, Mt, freq0, fdelta, pp, uvmin, Nt, max_lbfgs, lbfgs_m, gpu_threads, solver_mode, nu_mean, res_0, res_1);

  /* free dummy carr[] */
  free(carr);

  return retval;
}
#endif /* USE_MIC */
