/*
 *
 Copyright (C) 2006-2012 Sarod Yatawatta <sarod@users.sf.net>  
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


#define _GNU_SOURCE /* for sincos() */
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "Radio.h"

/* Jones matrix multiplication 
   C=A*B
*/
static void
amb(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=(a[0]*b[0]+a[1]*b[2]);
 c[1]=(a[0]*b[1]+a[1]*b[3]);
 c[2]=(a[2]*b[0]+a[3]*b[2]);
 c[3]=(a[2]*b[1]+a[3]*b[3]);
}


/* Jones matrix multiplication 
   C=A*B^H
*/
static void
ambt(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=a[0]*conj(b[0])+a[1]*conj(b[1]);
 c[1]=a[0]*conj(b[2])+a[1]*conj(b[3]);
 c[2]=a[2]*conj(b[0])+a[3]*conj(b[1]);
 c[3]=a[2]*conj(b[2])+a[3]*conj(b[3]);
}


/* worker thread function for subtraction
  also correct residual with solutions for cluster id 0 */
static void *
residual_threadfn_nointerpolation(void *data) {
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
   /* if this baseline is flagged, we do not compute */

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   for (cm=0; cm<M; cm++) { /* clusters */
      /* check if cluster id >=0 to do a subtraction */
     if (t->carr[cm].id>=0) {
     /* gains for this cluster, for sta1,sta2 */
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
     //printf("base %d, cluster %d, parm off %d abs %d\n",t->bindex[ci],cm,px,t->carr[cm].p[px]);
     //pm=&(t->p0[cm*8*N]);
     pm=&(t->p0[t->carr[cm].p[px]]);
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

     /* subtract from baseline visibilities */
     t->x[8*ci]-=creal(T2[0]);
     t->x[8*ci+1]-=cimag(T2[0]);
     t->x[8*ci+2]-=creal(T2[1]);
     t->x[8*ci+3]-=cimag(T2[1]);
     t->x[8*ci+4]-=creal(T2[2]);
     t->x[8*ci+5]-=cimag(T2[2]);
     t->x[8*ci+6]-=creal(T2[3]);
     t->x[8*ci+7]-=cimag(T2[3]);
     }
   }
   if (t->pinv) {
    cm=t->ccid;
    /* now do correction, if any */
    C[0]=t->x[8*ci]+_Complex_I*t->x[8*ci+1];
    C[1]=t->x[8*ci+2]+_Complex_I*t->x[8*ci+3];
    C[2]=t->x[8*ci+4]+_Complex_I*t->x[8*ci+5];
    C[3]=t->x[8*ci+6]+_Complex_I*t->x[8*ci+7];
    px=(ci+t->boff)/((Ntilebase+t->carr[cm].nchunk-1)/t->carr[cm].nchunk);
    pm=&(t->pinv[8*t->N*px]);
    G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
    G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
    G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
    G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
    G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
    G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
    G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
    G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);
    /* T1=G1*C  */
    amb(G1,C,T1);
    /* T2=T1*G2' */
    ambt(T1,G2,T2);
    t->x[8*ci]=creal(T2[0]);
    t->x[8*ci+1]=cimag(T2[0]);
    t->x[8*ci+2]=creal(T2[1]);
    t->x[8*ci+3]=cimag(T2[1]);
    t->x[8*ci+4]=creal(T2[2]);
    t->x[8*ci+5]=cimag(T2[2]);
    t->x[8*ci+6]=creal(T2[3]);
    t->x[8*ci+7]=cimag(T2[3]);
   }
 }
 return NULL;
}

/* invert matrix xx - 8x1 array
 * store it in   yy - 8x1 array
 */
static int
mat_invert(double xx[8],double yy[8], double rho) {
 complex double a[4];
 complex double det;
 complex double b[4];

 a[0]=xx[0]+xx[1]*_Complex_I+rho;
 a[1]=xx[2]+xx[3]*_Complex_I;
 a[2]=xx[4]+xx[5]*_Complex_I;
 a[3]=xx[6]+xx[7]*_Complex_I+rho;



 det=a[0]*a[3]-a[1]*a[2];
 if (sqrt(cabs(det))<=rho) {
  det+=rho;
 }
 det=1.0/det;
 b[0]=a[3]*det;
 b[1]=-a[1]*det; 
 b[2]=-a[2]*det;
 b[3]=a[0]*det;


 yy[0]=creal(b[0]);
 yy[1]=cimag(b[0]);
 yy[2]=creal(b[1]);
 yy[3]=cimag(b[1]);
 yy[4]=creal(b[2]);
 yy[5]=cimag(b[2]);
 yy[6]=creal(b[3]);
 yy[7]=cimag(b[3]);

 return 0;
}



int
calculate_residuals_interp(double *u,double *v,double *w,double *p0,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, complex double *coh, int M,double freq0,double fdelta,int Nt, int ccid, double rho) {
  int nth,nth1,ci,cj;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=Nbase*tilesz;

  int cm;
  double *pm,*pinv=0;
  cm=-1;
  /* find if any cluster is specified for correction of data */
  for (cj=0; cj<M; cj++) { /* clusters */
    /* check if cluster id == ccid to do a correction */
    if (carr[cj].id==ccid) {
     cm=cj;
     ci=1; /* correction cluster found */
    }
  }
  if (cm>=0) { /* valid cluser for correction */
   /* allocate memory for inverse J */
   if ((pinv=(double*)malloc((size_t)8*N*carr[cm].nchunk*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   for (cj=0; cj<carr[cm].nchunk; cj++) {
    pm=&(p0[carr[cm].p[cj]]); /* start of solutions */
    /* invert N solutions */
    for (ci=0; ci<N; ci++) {
     mat_invert(&pm[8*ci],&pinv[8*ci+8*N*cj], rho);
    }
   }
  } 
    
    
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
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]);
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].p0=p0;
    threaddata[nth].p=p;
    threaddata[nth].pinv=pinv;
    threaddata[nth].ccid=cm;
    threaddata[nth].N=N;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].coh=&(coh[4*(M)*ci]);
    
    //printf("thread %d predict  data from %d baselines %d\n",nth,8*ci,Nthb);
    if (p==p0) {
     pthread_create(&th_array[nth],&attr,residual_threadfn_nointerpolation,(void*)(&threaddata[nth]));
    } else {
     fprintf(stderr,"Warning: interpolation is disabled for the moment\n");
     pthread_create(&th_array[nth],&attr,residual_threadfn_nointerpolation,(void*)(&threaddata[nth]));
     //pthread_create(&th_array[nth],&attr,residual_threadfn_withinterpolation,(void*)(&threaddata[nth]));
    }
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
 free(pinv);

 return 0;

}

/* worker thread function for subtraction
  also correct residual with solutions for cluster id 0 */
static void *
residual_threadfn_onefreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,cn,sta1,sta2;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */
 double *pm;
 double freq0;
 double fdelta2=t->fdelta*0.5;

 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 int Ntilebase=(t->Nbase)*(t->tilesz);
 int px;
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   /* even if this baseline is flagged, we do compute */

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   for (cm=0; cm<M; cm++) { /* clusters */
      /* check if cluster id >=0 to do a subtraction */
     if (t->carr[cm].id>=0) {
     /* gains for this cluster, for sta1,sta2 */
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
     //printf("base %d, cluster %d, parm off %d abs %d\n",t->bindex[ci],cm,px,t->carr[cm].p[px]);
     pm=&(t->p[t->carr[cm].p[px]]);
     G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
     G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
     G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
     G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
     G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
     G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
     G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
     G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);


     /* iterate over frequencies */
      freq0=t->freq0;
/***********************************************/
      /* calculate coherencies for each freq */
      memset(C,0,sizeof(complex double)*4);

     /* setup memory */
     if (posix_memalign((void*)&PHr,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&PHi,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&G,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&II,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&QQ,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&UU,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&VV,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     /* phase (real,imag) parts */
     /* note u=u/c, v=v/c, w=w/c here */
     /* phterm is 2pi(u/c l +v/c m +w/c n) */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       G[cn]=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[cn]+t->v[ci]*t->carr[cm].mm[cn]+t->w[ci]*t->carr[cm].nn[cn]);
     }
     for (cn=0; cn<t->carr[cm].N; cn++) {
       sincos(G[cn]*freq0,&PHi[cn],&PHr[cn]);
     }

     /* term due to shape of source, also multiplied by freq/time smearing */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* freq smearing : extra term delta * sinc(delta/2 * phterm) */
       if (G[cn]!=0.0) {
         double smfac=G[cn]*fdelta2;
         double sinph=sin(smfac)/smfac;
         G[cn]=fabs(sinph);
       } else {
         G[cn]=1.0;
       }
     }

     /* multiply (re,im) phase term with smearing/shape factor */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHr[cn]*=G[cn];
       PHi[cn]*=G[cn];
     }


     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* time smearing TMS eq. 6.81 for EW-array formula */
       //G[cn]*=time_smear(t->carr[cm].ll[cn],t->carr[cm].mm[cn],t->dec0,t->tdelta,t->u[ci],t->v[ci],t->w[ci],t->freq0);

       /* check if source type is not a point source for additional 
          calculations */
       if (t->carr[cm].stype[cn]!=STYPE_POINT) {
        complex double sterm=PHr[cn]+_Complex_I*PHi[cn];
        if (t->carr[cm].stype[cn]==STYPE_SHAPELET) {
         sterm*=shapelet_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_GAUSSIAN) {
         sterm*=gaussian_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_DISK) {
         sterm*=disk_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_RING) {
         sterm*=ring_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        }
        PHr[cn]=creal(sterm);
        PHi[cn]=cimag(sterm);
       }

     }

     /* flux of each source, at each freq */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* coherencies are NOT scaled by 1/2, with spectral index */
       if (t->carr[cm].spec_idx[cn]!=0.0) {
         double fratio=log(freq0/t->carr[cm].f0[cn]);
         double fratio1=fratio*fratio;
         double fratio2=fratio1*fratio;
         double tempfr=t->carr[cm].spec_idx[cn]*fratio+t->carr[cm].spec_idx1[cn]*fratio1+t->carr[cm].spec_idx2[cn]*fratio2;
         /* catch -ve and 0 sI */
         if (t->carr[cm].sI0[cn]>0.0) {
          II[cn]=exp(log(t->carr[cm].sI0[cn])+tempfr);
         } else {
          II[cn]=(t->carr[cm].sI0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sI0[cn])+tempfr));
         }
         if (t->carr[cm].sQ0[cn]>0.0) {
          QQ[cn]=exp(log(t->carr[cm].sQ0[cn])+tempfr);
         } else {
          QQ[cn]=(t->carr[cm].sQ0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sQ0[cn])+tempfr));
         }
         if (t->carr[cm].sU0[cn]>0.0) {
          UU[cn]=exp(log(t->carr[cm].sU0[cn])+tempfr);
         } else {
          UU[cn]=(t->carr[cm].sU0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sU0[cn])+tempfr));
         }
         if (t->carr[cm].sV0[cn]>0.0) {
          VV[cn]=exp(log(t->carr[cm].sV0[cn])+tempfr);
         } else {
          VV[cn]=(t->carr[cm].sV0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sV0[cn])+tempfr));
         }
       } else {
         II[cn]=t->carr[cm].sI[cn];
         QQ[cn]=t->carr[cm].sQ[cn];
         UU[cn]=t->carr[cm].sU[cn];
         VV[cn]=t->carr[cm].sV[cn];
       }
     }
 

     /* add up terms together */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       complex double Ph,IIl,QQl,UUl,VVl;
       Ph=(PHr[cn]+_Complex_I*PHi[cn]);
       IIl=Ph*II[cn];
       QQl=Ph*QQ[cn];
       UUl=Ph*UU[cn];
       VVl=Ph*VV[cn];
       C[0]+=IIl+QQl;
       C[1]+=UUl+_Complex_I*VVl;
       C[2]+=UUl-_Complex_I*VVl;
       C[3]+=IIl-QQl;
     }

     free(PHr);
     free(PHi);
     free(G);
     free(II);
     free(QQ);
     free(UU);
     free(VV);

/***********************************************/
      /* form G1*C*G2' */
      /* T1=G1*C  */
      amb(G1,C,T1);
      /* T2=T1*G2' */
      ambt(T1,G2,T2);

      /* subtract from baseline visibilities */
      t->x[8*ci]-=creal(T2[0]);
      t->x[8*ci+1]-=cimag(T2[0]);
      t->x[8*ci+2]-=creal(T2[1]);
      t->x[8*ci+3]-=cimag(T2[1]);
      t->x[8*ci+4]-=creal(T2[2]);
      t->x[8*ci+5]-=cimag(T2[2]);
      t->x[8*ci+6]-=creal(T2[3]);
      t->x[8*ci+7]-=cimag(T2[3]);
     }
   }
   if (t->pinv) {
    cm=t->ccid;
    px=(ci+t->boff)/((Ntilebase+t->carr[cm].nchunk-1)/t->carr[cm].nchunk);
    pm=&(t->pinv[8*t->N*px]);
    G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
    G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
    G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
    G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
    G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
    G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
    G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
    G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);

     /* now do correction, if any */
     C[0]=t->x[8*ci]+_Complex_I*t->x[8*ci+1];
     C[1]=t->x[8*ci+2]+_Complex_I*t->x[8*ci+3];
     C[2]=t->x[8*ci+4]+_Complex_I*t->x[8*ci+5];
     C[3]=t->x[8*ci+6]+_Complex_I*t->x[8*ci+7];
     /* T1=G1*C  */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);
     t->x[8*ci]=creal(T2[0]);
     t->x[8*ci+1]=cimag(T2[0]);
     t->x[8*ci+2]=creal(T2[1]);
     t->x[8*ci+3]=cimag(T2[1]);
     t->x[8*ci+4]=creal(T2[2]);
     t->x[8*ci+5]=cimag(T2[2]);
     t->x[8*ci+6]=creal(T2[3]);
     t->x[8*ci+7]=cimag(T2[3]);
   }
 }
 return NULL;
}


int
calculate_residuals(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double freq0, double fdelta,double tdelta,double dec0,int Nt, int ccid, double rho) {
  int nth,nth1,ci,cj;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=Nbase*tilesz;

  int cm;
  double *pm,*pinv=0;
  cm=-1;
  /* find if any cluster is specified for correction of data */
  for (cj=0; cj<M; cj++) { /* clusters */
    /* check if cluster id == ccid to do a correction */
    if (carr[cj].id==ccid) {
     cm=cj;
     ci=1; /* correction cluster found */
    }
  }
  if (cm>=0) { /* valid cluser for correction */
   /* allocate memory for inverse J */
   if ((pinv=(double*)malloc((size_t)8*N*carr[cm].nchunk*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   for (cj=0; cj<carr[cm].nchunk; cj++) {
    pm=&(p[carr[cm].p[cj]]); /* start of solutions */
    /* invert N solutions */
    for (ci=0; ci<N; ci++) {
     mat_invert(&pm[8*ci],&pinv[8*ci+8*N*cj], rho);
    }
   }
  } 
    
    
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
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]);
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].p=p;
    threaddata[nth].pinv=pinv;
    threaddata[nth].ccid=cm;
    threaddata[nth].N=N;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].freq0=freq0;
    threaddata[nth].fdelta=fdelta;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
   
    
    pthread_create(&th_array[nth],&attr,residual_threadfn_onefreq,(void*)(&threaddata[nth]));
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
 free(pinv);

 return 0;

}



/* worker thread function for subtraction
  also correct residual with solutions for cluster id 0 */
static void *
residual_threadfn_multifreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,cf,cn,sta1,sta2;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */
 double *pm;
 double freq0;
 double fdelta2=t->fdelta*0.5;

 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 int Ntilebase=(t->Nbase)*(t->tilesz);
 int px;
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   /* if this baseline is flagged, we do not compute */

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   for (cm=0; cm<M; cm++) { /* clusters */
      /* check if cluster id >=0 to do a subtraction */
     if (t->carr[cm].id>=0) {
     /* gains for this cluster, for sta1,sta2 */
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
     //printf("base %d, cluster %d, parm off %d abs %d\n",t->bindex[ci],cm,px,t->carr[cm].p[px]);
     pm=&(t->p[t->carr[cm].p[px]]);
     G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
     G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
     G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
     G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
     G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
     G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
     G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
     G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);


     /* iterate over frequencies */
     for (cf=0; cf<t->Nchan; cf++) {
      freq0=t->freqs[cf];
/***********************************************/
      /* calculate coherencies for each freq */
      memset(C,0,sizeof(complex double)*4);

     /* setup memory */
     if (posix_memalign((void*)&PHr,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&PHi,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&G,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&II,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&QQ,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&UU,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&VV,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }

     /* phase (real,imag) parts */
     /* note u=u/c, v=v/c, w=w/c here */
     /* phterm is 2pi(u/c l +v/c m +w/c n) */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       G[cn]=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[cn]+t->v[ci]*t->carr[cm].mm[cn]+t->w[ci]*t->carr[cm].nn[cn]);
     }
     for (cn=0; cn<t->carr[cm].N; cn++) {
       sincos(G[cn]*freq0,&PHi[cn],&PHr[cn]);
     }

     /* term due to shape of source, also multiplied by freq/time smearing */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* freq smearing : extra term delta * sinc(delta/2 * phterm) */
       if (G[cn]!=0.0) {
         double smfac=G[cn]*fdelta2;
         double sinph=sin(smfac)/smfac;
         G[cn]=fabs(sinph);
       } else {
         G[cn]=1.0;
       }
     }

     /* multiply (re,im) phase term with smearing/shape factor */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHr[cn]*=G[cn];
       PHi[cn]*=G[cn];
     }


     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* check if source type is not a point source for additional 
          calculations */
       if (t->carr[cm].stype[cn]!=STYPE_POINT) {
        complex double sterm=PHr[cn]+_Complex_I*PHi[cn];
        if (t->carr[cm].stype[cn]==STYPE_SHAPELET) {
         sterm*=shapelet_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_GAUSSIAN) {
         sterm*=gaussian_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_DISK) {
         sterm*=disk_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_RING) {
         sterm*=ring_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        }
        PHr[cn]=creal(sterm);
        PHi[cn]=cimag(sterm);
       }

     }


     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* coherencies are NOT scaled by 1/2, with spectral index */
       if (t->carr[cm].spec_idx[cn]!=0.0) {
         double fratio=log(freq0/t->carr[cm].f0[cn]);
         double fratio1=fratio*fratio;
         double fratio2=fratio1*fratio;
         double tempfr=t->carr[cm].spec_idx[cn]*fratio+t->carr[cm].spec_idx1[cn]*fratio1+t->carr[cm].spec_idx2[cn]*fratio2;
         /* catch -ve and 0 sI */
         if (t->carr[cm].sI0[cn]>0.0) {
          II[cn]=exp(log(t->carr[cm].sI0[cn])+tempfr);
         } else {
          II[cn]=(t->carr[cm].sI0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sI0[cn])+tempfr));
         }
         if (t->carr[cm].sQ0[cn]>0.0) {
          QQ[cn]=exp(log(t->carr[cm].sQ0[cn])+tempfr);
         } else {
          QQ[cn]=(t->carr[cm].sQ0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sQ0[cn])+tempfr));
         }
         if (t->carr[cm].sU0[cn]>0.0) {
          UU[cn]=exp(log(t->carr[cm].sU0[cn])+tempfr);
         } else {
          UU[cn]=(t->carr[cm].sU0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sU0[cn])+tempfr));
         }
         if (t->carr[cm].sV0[cn]>0.0) {
          VV[cn]=exp(log(t->carr[cm].sV0[cn])+tempfr);
         } else {
          VV[cn]=(t->carr[cm].sV0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sV0[cn])+tempfr));
         }
       } else {
         II[cn]=t->carr[cm].sI[cn];
         QQ[cn]=t->carr[cm].sQ[cn];
         UU[cn]=t->carr[cm].sU[cn];
         VV[cn]=t->carr[cm].sV[cn];
       }
     }

     /* add up terms together */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       complex double Ph,IIl,QQl,UUl,VVl;
       Ph=(PHr[cn]+_Complex_I*PHi[cn]);
       IIl=Ph*II[cn];
       QQl=Ph*QQ[cn];
       UUl=Ph*UU[cn];
       VVl=Ph*VV[cn];
       C[0]+=IIl+QQl;
       C[1]+=UUl+_Complex_I*VVl;
       C[2]+=UUl-_Complex_I*VVl;
       C[3]+=IIl-QQl;
     }

     free(PHr);
     free(PHi);
     free(G);
     free(II);
     free(QQ);
     free(UU);
     free(VV);


/***********************************************/
      /* form G1*C*G2' */
      /* T1=G1*C  */
      amb(G1,C,T1);
      /* T2=T1*G2' */
      ambt(T1,G2,T2);

      /* subtract from baseline visibilities */
      t->x[8*ci+cf*Ntilebase*8]-=creal(T2[0]);
      t->x[8*ci+1+cf*Ntilebase*8]-=cimag(T2[0]);
      t->x[8*ci+2+cf*Ntilebase*8]-=creal(T2[1]);
      t->x[8*ci+3+cf*Ntilebase*8]-=cimag(T2[1]);
      t->x[8*ci+4+cf*Ntilebase*8]-=creal(T2[2]);
      t->x[8*ci+5+cf*Ntilebase*8]-=cimag(T2[2]);
      t->x[8*ci+6+cf*Ntilebase*8]-=creal(T2[3]);
      t->x[8*ci+7+cf*Ntilebase*8]-=cimag(T2[3]);
     }
     }
   }
   if (t->pinv) {
    cm=t->ccid;
    px=(ci+t->boff)/((Ntilebase+t->carr[cm].nchunk-1)/t->carr[cm].nchunk);
    pm=&(t->pinv[8*t->N*px]);
    G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
    G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
    G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
    G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
    G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
    G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
    G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
    G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);

    /* iterate over frequencies */
    for (cf=0; cf<t->Nchan; cf++) {
     /* now do correction, if any */
     C[0]=t->x[8*ci+cf*Ntilebase*8]+_Complex_I*t->x[8*ci+1+cf*Ntilebase*8];
     C[1]=t->x[8*ci+2+cf*Ntilebase*8]+_Complex_I*t->x[8*ci+3+cf*Ntilebase*8];
     C[2]=t->x[8*ci+4+cf*Ntilebase*8]+_Complex_I*t->x[8*ci+5+cf*Ntilebase*8];
     C[3]=t->x[8*ci+6+cf*Ntilebase*8]+_Complex_I*t->x[8*ci+7+cf*Ntilebase*8];
     /* T1=G1*C  */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);
     t->x[8*ci+cf*Ntilebase*8]=creal(T2[0]);
     t->x[8*ci+1+cf*Ntilebase*8]=cimag(T2[0]);
     t->x[8*ci+2+cf*Ntilebase*8]=creal(T2[1]);
     t->x[8*ci+3+cf*Ntilebase*8]=cimag(T2[1]);
     t->x[8*ci+4+cf*Ntilebase*8]=creal(T2[2]);
     t->x[8*ci+5+cf*Ntilebase*8]=cimag(T2[2]);
     t->x[8*ci+6+cf*Ntilebase*8]=creal(T2[3]);
     t->x[8*ci+7+cf*Ntilebase*8]=cimag(T2[3]);
    }
   }
 }
 return NULL;
}


int
calculate_residuals_multifreq(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,int Nt, int ccid, double rho, int phase_only) {
  int nth,nth1,ci,cj;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=Nbase*tilesz;

  int cm;
  double *pm,*pinv=0,*pphase=0;
  cm=-1;
  /* find if any cluster is specified for correction of data */
  for (cj=0; cj<M; cj++) { /* clusters */
    /* check if cluster id == ccid to do a correction */
    if (carr[cj].id==ccid) {
     cm=cj;
     ci=1; /* correction cluster found */
    }
  }
  if (cm>=0) { /* valid cluser for correction */
   /* allocate memory for inverse J */
   if ((pinv=(double*)malloc((size_t)8*N*carr[cm].nchunk*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   if (!phase_only) {
    for (cj=0; cj<carr[cm].nchunk; cj++) {
     pm=&(p[carr[cm].p[cj]]); /* start of solutions */
     /* invert N solutions */
     for (ci=0; ci<N; ci++) {
      mat_invert(&pm[8*ci],&pinv[8*ci+8*N*cj], rho);
     }
    }
   } else {
    /* joint diagonalize solutions and get only phases before inverting */
    if ((pphase=(double*)malloc((size_t)8*N*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    for (cj=0; cj<carr[cm].nchunk; cj++) {
      pm=&(p[carr[cm].p[cj]]); /* start of solutions */
      /* extract phase of pm, output to pphase */
      extract_phases(pm,pphase,N,10);
      /* invert N solutions */
      for (ci=0; ci<N; ci++) {
       mat_invert(&pphase[8*ci],&pinv[8*ci+8*N*cj], rho);
      }
    }
    free(pphase);
   }
  } 
    
    
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
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]);
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].p=p;
    threaddata[nth].pinv=pinv;
    threaddata[nth].ccid=cm;
    threaddata[nth].N=N;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].freqs=freqs;
    threaddata[nth].Nchan=Nchan;
    threaddata[nth].fdelta=fdelta/(double)Nchan;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
    
    pthread_create(&th_array[nth],&attr,residual_threadfn_multifreq,(void*)(&threaddata[nth]));
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
 free(pinv);

 return 0;

}

/* worker thread function for prediction 
   */
static void *
visibilities_threadfn_multifreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 

 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */
 int ci,cm,cf,cn;
 double freq0;
 double fdelta2=t->fdelta*0.5;

 complex double C[4];
 int M=(t->M);
 int Ntilebase=(t->Nbase)*(t->tilesz);
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* if this baseline is flagged, we do not compute */
   for (cm=0; cm<M; cm++) { /* clusters */
      /* iterate over frequencies */
     for (cf=0; cf<t->Nchan; cf++) {
      freq0=t->freqs[cf];
/***********************************************/
      /* calculate coherencies for each freq */
      memset(C,0,sizeof(complex double)*4);
     /* FIXME: use arrays Nx1 to try to vectorize this part */

     /* setup memory */
     if (posix_memalign((void*)&PHr,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&PHi,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&G,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&II,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&QQ,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&UU,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&VV,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }

     /* phase (real,imag) parts */
     /* note u=u/c, v=v/c, w=w/c here */
     /* phterm is 2pi(u/c l +v/c m +w/c n) */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       G[cn]=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[cn]+t->v[ci]*t->carr[cm].mm[cn]+t->w[ci]*t->carr[cm].nn[cn]);
     }
     for (cn=0; cn<t->carr[cm].N; cn++) {
       sincos(G[cn]*freq0,&PHi[cn],&PHr[cn]);
     }

     /* term due to shape of source, also multiplied by freq/time smearing */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* freq smearing : extra term delta * sinc(delta/2 * phterm) */
       if (G[cn]!=0.0) {
         double smfac=G[cn]*fdelta2;
         double sinph=sin(smfac)/smfac;
         G[cn]=fabs(sinph);
       } else {
         G[cn]=1.0;
       }
     }

     /* multiply (re,im) phase term with smearing/shape factor */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHr[cn]*=G[cn];
       PHi[cn]*=G[cn];
     }


     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* check if source type is not a point source for additional 
          calculations */
       if (t->carr[cm].stype[cn]!=STYPE_POINT) {
        complex double sterm=PHr[cn]+_Complex_I*PHi[cn];
        if (t->carr[cm].stype[cn]==STYPE_SHAPELET) {
         sterm*=shapelet_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_GAUSSIAN) {
         sterm*=gaussian_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_DISK) {
         sterm*=disk_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_RING) {
         sterm*=ring_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        }
        PHr[cn]=creal(sterm);
        PHi[cn]=cimag(sterm);
       }

     }


     /* flux of each source, at each freq */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* coherencies are NOT scaled by 1/2, with spectral index */
       if (t->carr[cm].spec_idx[cn]!=0.0) {
         double fratio=log(freq0/t->carr[cm].f0[cn]);
         double fratio1=fratio*fratio;
         double fratio2=fratio1*fratio;
         double tempfr=t->carr[cm].spec_idx[cn]*fratio+t->carr[cm].spec_idx1[cn]*fratio1+t->carr[cm].spec_idx2[cn]*fratio2;
         /* catch -ve and 0 sI */ 
         if (t->carr[cm].sI0[cn]>0.0) {
          II[cn]=exp(log(t->carr[cm].sI0[cn])+tempfr);
         } else {
          II[cn]=(t->carr[cm].sI0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sI0[cn])+tempfr));
         }
         if (t->carr[cm].sQ0[cn]>0.0) {
          QQ[cn]=exp(log(t->carr[cm].sQ0[cn])+tempfr);
         } else {
          QQ[cn]=(t->carr[cm].sQ0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sQ0[cn])+tempfr));
         }
         if (t->carr[cm].sU0[cn]>0.0) {
          UU[cn]=exp(log(t->carr[cm].sU0[cn])+tempfr);
         } else {
          UU[cn]=(t->carr[cm].sU0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sU0[cn])+tempfr));
         }
         if (t->carr[cm].sV0[cn]>0.0) {
          VV[cn]=exp(log(t->carr[cm].sV0[cn])+tempfr);
         } else {
          VV[cn]=(t->carr[cm].sV0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sV0[cn])+tempfr));
         }
       } else {
         II[cn]=t->carr[cm].sI[cn];
         QQ[cn]=t->carr[cm].sQ[cn];
         UU[cn]=t->carr[cm].sU[cn];
         VV[cn]=t->carr[cm].sV[cn];
       }
     }


     /* add up terms together */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       complex double Ph,IIl,QQl,UUl,VVl;
       Ph=(PHr[cn]+_Complex_I*PHi[cn]);
       IIl=Ph*II[cn];
       QQl=Ph*QQ[cn];
       UUl=Ph*UU[cn];
       VVl=Ph*VV[cn];
       C[0]+=IIl+QQl;
       C[1]+=UUl+_Complex_I*VVl;
       C[2]+=UUl-_Complex_I*VVl;
       C[3]+=IIl-QQl;
     }

     free(PHr);
     free(PHi);
     free(G);
     free(II);
     free(QQ);
     free(UU);
     free(VV);

/***********************************************/
      /* add to baseline visibilities */
      t->x[8*ci+cf*Ntilebase*8]+=creal(C[0]);
      t->x[8*ci+1+cf*Ntilebase*8]+=cimag(C[0]);
      t->x[8*ci+2+cf*Ntilebase*8]+=creal(C[1]);
      t->x[8*ci+3+cf*Ntilebase*8]+=cimag(C[1]);
      t->x[8*ci+4+cf*Ntilebase*8]+=creal(C[2]);
      t->x[8*ci+5+cf*Ntilebase*8]+=cimag(C[2]);
      t->x[8*ci+6+cf*Ntilebase*8]+=creal(C[3]);
      t->x[8*ci+7+cf*Ntilebase*8]+=cimag(C[3]);
     }
   }

 }
 return NULL;
}




int
predict_visibilities_multifreq(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,int Nt, int add_to_data) {
  int nth,nth1,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=Nbase*tilesz;

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

  if (add_to_data==SIMUL_ONLY) {
   /* set output column to zero */
   memset(x,0,sizeof(double)*8*Nbase*tilesz*Nchan);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]);
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].p=0;
    threaddata[nth].pinv=0;
    threaddata[nth].ccid=-1;
    threaddata[nth].N=N;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].freqs=freqs;
    threaddata[nth].Nchan=Nchan;
    threaddata[nth].fdelta=fdelta/(double)Nchan;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
    threaddata[nth].add_to_data=add_to_data;
    
   
    pthread_create(&th_array[nth],&attr,visibilities_threadfn_multifreq,(void*)(&threaddata[nth]));
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


/* worker thread function for prediction with solutions
   */
static void *
predictwithgain_threadfn_multifreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,cf,cn,sta1,sta2;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */
 double *pm;
 double freq0;
 double fdelta2=t->fdelta*0.5;

 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 int Ntilebase=(t->Nbase)*(t->tilesz);
 int px;
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   /* if this baseline is flagged, we do not compute */
   if (t->add_to_data==SIMUL_ONLY) { /* only model is written as output */
    for (cf=0; cf<t->Nchan; cf++) {
     memset(&t->x[8*ci+cf*Ntilebase*8],0,sizeof(double)*8);
    }
   }

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   for (cm=0; cm<M; cm++) { /* clusters */
      /* check if cluster id not in ignore list to do a prediction */
     if (!t->ignlist[cm]) {
     /* gains for this cluster, for sta1,sta2 */
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
     //printf("base %d, cluster %d, parm off %d abs %d\n",t->boff+ci,cm,px,t->carr[cm].p[px]);
     pm=&(t->p[t->carr[cm].p[px]]);
     G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
     G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
     G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
     G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
     G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
     G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
     G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
     G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);


     /* iterate over frequencies */
     for (cf=0; cf<t->Nchan; cf++) {
      freq0=t->freqs[cf];
/***********************************************/
      /* calculate coherencies for each freq */
      memset(C,0,sizeof(complex double)*4);
     /* setup memory */
     if (posix_memalign((void*)&PHr,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&PHi,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&G,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&II,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&QQ,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&UU,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (posix_memalign((void*)&VV,sizeof(double),((size_t)t->carr[cm].N*sizeof(double)))!=0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }

     /* phase (real,imag) parts */
     /* note u=u/c, v=v/c, w=w/c here */
     /* phterm is 2pi(u/c l +v/c m +w/c n) */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       G[cn]=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[cn]+t->v[ci]*t->carr[cm].mm[cn]+t->w[ci]*t->carr[cm].nn[cn]);
     }
     for (cn=0; cn<t->carr[cm].N; cn++) {
       sincos(G[cn]*freq0,&PHi[cn],&PHr[cn]);
     }

     /* term due to shape of source, also multiplied by freq/time smearing */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* freq smearing : extra term delta * sinc(delta/2 * phterm) */
       if (G[cn]!=0.0) {
         double smfac=G[cn]*fdelta2;
         double sinph=sin(smfac)/smfac;
         G[cn]=fabs(sinph);
       } else {
         G[cn]=1.0;
       }
     }

     /* multiply (re,im) phase term with smearing/shape factor */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHr[cn]*=G[cn];
       PHi[cn]*=G[cn];
     }


     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* check if source type is not a point source for additional 
          calculations */
       if (t->carr[cm].stype[cn]!=STYPE_POINT) {
        complex double sterm=PHr[cn]+_Complex_I*PHi[cn];
        if (t->carr[cm].stype[cn]==STYPE_SHAPELET) {
         sterm*=shapelet_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_GAUSSIAN) {
         sterm*=gaussian_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_DISK) {
         sterm*=disk_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        } else if (t->carr[cm].stype[cn]==STYPE_RING) {
         sterm*=ring_contrib(t->carr[cm].ex[cn],t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0);
        }
        PHr[cn]=creal(sterm);
        PHi[cn]=cimag(sterm);
       }

     }


     /* flux of each source, at each freq */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* coherencies are NOT scaled by 1/2, with spectral index */
       if (t->carr[cm].spec_idx[cn]!=0.0) {
         double fratio=log(freq0/t->carr[cm].f0[cn]);
         double fratio1=fratio*fratio;
         double fratio2=fratio1*fratio;
         double tempfr=t->carr[cm].spec_idx[cn]*fratio+t->carr[cm].spec_idx1[cn]*fratio1+t->carr[cm].spec_idx2[cn]*fratio2;
         /* catch -ve and 0 sI */
         if (t->carr[cm].sI0[cn]>0.0) {
          II[cn]=exp(log(t->carr[cm].sI0[cn])+tempfr);
         } else {
          II[cn]=(t->carr[cm].sI0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sI0[cn])+tempfr));
         }
         if (t->carr[cm].sQ0[cn]>0.0) {
          QQ[cn]=exp(log(t->carr[cm].sQ0[cn])+tempfr);
         } else {
          QQ[cn]=(t->carr[cm].sQ0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sQ0[cn])+tempfr));
         }
         if (t->carr[cm].sU0[cn]>0.0) {
          UU[cn]=exp(log(t->carr[cm].sU0[cn])+tempfr);
         } else {
          UU[cn]=(t->carr[cm].sU0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sU0[cn])+tempfr));
         }
         if (t->carr[cm].sV0[cn]>0.0) {
          VV[cn]=exp(log(t->carr[cm].sV0[cn])+tempfr);
         } else {
          VV[cn]=(t->carr[cm].sV0[cn]==0.0?0.0:-exp(log(-t->carr[cm].sV0[cn])+tempfr));
         }
       } else {
         II[cn]=t->carr[cm].sI[cn];
         QQ[cn]=t->carr[cm].sQ[cn];
         UU[cn]=t->carr[cm].sU[cn];
         VV[cn]=t->carr[cm].sV[cn];
       }
     }

     /* add up terms together */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       complex double Ph,IIl,QQl,UUl,VVl;
       Ph=(PHr[cn]+_Complex_I*PHi[cn]);
       IIl=Ph*II[cn];
       QQl=Ph*QQ[cn];
       UUl=Ph*UU[cn];
       VVl=Ph*VV[cn];
       C[0]+=IIl+QQl;
       C[1]+=UUl+_Complex_I*VVl;
       C[2]+=UUl-_Complex_I*VVl;
       C[3]+=IIl-QQl;
     }

     free(PHr);
     free(PHi);
     free(G);
     free(II);
     free(QQ);
     free(UU);
     free(VV);


/***********************************************/
      /* form G1*C*G2' */
      /* T1=G1*C  */
      amb(G1,C,T1);
      /* T2=T1*G2' */
      ambt(T1,G2,T2);
      if (t->add_to_data==SIMUL_ONLY || t->add_to_data==SIMUL_ADD) {
        /* add to baseline visibilities */
        t->x[8*ci+cf*Ntilebase*8]+=creal(T2[0]);
        t->x[8*ci+1+cf*Ntilebase*8]+=cimag(T2[0]);
        t->x[8*ci+2+cf*Ntilebase*8]+=creal(T2[1]);
        t->x[8*ci+3+cf*Ntilebase*8]+=cimag(T2[1]);
        t->x[8*ci+4+cf*Ntilebase*8]+=creal(T2[2]);
        t->x[8*ci+5+cf*Ntilebase*8]+=cimag(T2[2]);
        t->x[8*ci+6+cf*Ntilebase*8]+=creal(T2[3]);
        t->x[8*ci+7+cf*Ntilebase*8]+=cimag(T2[3]);
      } else if (t->add_to_data==SIMUL_SUB) {
        /* subtract from baseline visibilities */
        t->x[8*ci+cf*Ntilebase*8]-=creal(T2[0]);
        t->x[8*ci+1+cf*Ntilebase*8]-=cimag(T2[0]);
        t->x[8*ci+2+cf*Ntilebase*8]-=creal(T2[1]);
        t->x[8*ci+3+cf*Ntilebase*8]-=cimag(T2[1]);
        t->x[8*ci+4+cf*Ntilebase*8]-=creal(T2[2]);
        t->x[8*ci+5+cf*Ntilebase*8]-=cimag(T2[2]);
        t->x[8*ci+6+cf*Ntilebase*8]-=creal(T2[3]);
        t->x[8*ci+7+cf*Ntilebase*8]-=cimag(T2[3]);
      }
     }
     }
   }
  /* if valid cluster is given, correct with its solutions */
   if (t->pinv) {
    cm=t->ccid;
    px=(ci+t->boff)/((Ntilebase+t->carr[cm].nchunk-1)/t->carr[cm].nchunk);
    pm=&(t->pinv[8*t->N*px]);
    G1[0]=(pm[sta1*8])+_Complex_I*(pm[sta1*8+1]);
    G1[1]=(pm[sta1*8+2])+_Complex_I*(pm[sta1*8+3]);
    G1[2]=(pm[sta1*8+4])+_Complex_I*(pm[sta1*8+5]);
    G1[3]=(pm[sta1*8+6])+_Complex_I*(pm[sta1*8+7]);
    G2[0]=(pm[sta2*8])+_Complex_I*(pm[sta2*8+1]);
    G2[1]=(pm[sta2*8+2])+_Complex_I*(pm[sta2*8+3]);
    G2[2]=(pm[sta2*8+4])+_Complex_I*(pm[sta2*8+5]);
    G2[3]=(pm[sta2*8+6])+_Complex_I*(pm[sta2*8+7]);

     /* now do correction, if any */
     C[0]=t->x[8*ci]+_Complex_I*t->x[8*ci+1];
     C[1]=t->x[8*ci+2]+_Complex_I*t->x[8*ci+3];
     C[2]=t->x[8*ci+4]+_Complex_I*t->x[8*ci+5];
     C[3]=t->x[8*ci+6]+_Complex_I*t->x[8*ci+7];
     /* T1=G1*C  */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);
     t->x[8*ci]=creal(T2[0]);
     t->x[8*ci+1]=cimag(T2[0]);
     t->x[8*ci+2]=creal(T2[1]);
     t->x[8*ci+3]=cimag(T2[1]);
     t->x[8*ci+4]=creal(T2[2]);
     t->x[8*ci+5]=cimag(T2[2]);
     t->x[8*ci+6]=creal(T2[3]);
     t->x[8*ci+7]=cimag(T2[3]);
   }
 }
 return NULL;
}

int
predict_visibilities_multifreq_withsol(double *u,double *v,double *w,double *p,double *x,int *ignlist,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,int Nt, int add_to_data, int ccid, double rho, int phase_only) {
  int nth,nth1,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=Nbase*tilesz;


  int cm,cj;
  double *pm,*pinv=0,*pphase=0;
  cm=-1;
  /* find if any cluster is specified for correction of data */
  for (cj=0; cj<M; cj++) { /* clusters */
    /* check if cluster id == ccid to do a correction */
    if (carr[cj].id==ccid) {
     cm=cj;
     ci=1; /* correction cluster found */
    }
  }
  if (cm>=0) { /* valid cluser for correction */
   /* allocate memory for inverse J */
   if ((pinv=(double*)malloc((size_t)8*N*carr[cm].nchunk*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   if (!phase_only) {
    for (cj=0; cj<carr[cm].nchunk; cj++) {
     pm=&(p[carr[cm].p[cj]]); /* start of solutions */
     /* invert N solutions */
     for (ci=0; ci<N; ci++) {
      mat_invert(&pm[8*ci],&pinv[8*ci+8*N*cj], rho);
     }
    }
   } else {
    /* joint diagonalize solutions and get only phases before inverting */
    if ((pphase=(double*)malloc((size_t)8*N*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    for (cj=0; cj<carr[cm].nchunk; cj++) {
      pm=&(p[carr[cm].p[cj]]); /* start of solutions */
      /* extract phase of pm, output to pphase */
      extract_phases(pm,pphase,N,10);
      /* invert N solutions */
      for (ci=0; ci<N; ci++) {
       mat_invert(&pphase[8*ci],&pinv[8*ci+8*N*cj], rho);
      }
    }
    free(pphase);
   }
  }
    
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
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]);
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].p=p;
    threaddata[nth].ignlist=ignlist;
    threaddata[nth].N=N;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].freqs=freqs;
    threaddata[nth].Nchan=Nchan;
    threaddata[nth].fdelta=fdelta/(double)Nchan;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
    threaddata[nth].add_to_data=add_to_data;
    /* for correction of data */
    threaddata[nth].pinv=pinv;
    threaddata[nth].ccid=cm;

    
    pthread_create(&th_array[nth],&attr,predictwithgain_threadfn_multifreq,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

 pthread_attr_destroy(&attr);

 if (pinv) free(pinv);
 free(th_array);
 free(threaddata);

 return 0;

}
