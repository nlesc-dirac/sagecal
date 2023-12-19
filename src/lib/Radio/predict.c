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


#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "Dirac_radio.h"



/************* extended source contributions ************/

complex double
gaussian_contrib(void*dd, double u, double v, double w) {
  exinfo_gaussian *dp=(exinfo_gaussian*)dd;
  double up,vp,a,b,ut,vt,cosph,sinph;

  /* first the rotation due to projection */
  if (dp->use_projection) {
   up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
   vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);
  } else {
   up=u;
   vp=v;
  }

  /* linear transformations, if any */
  a=dp->eX;
  b=dp->eY;
  //cosph=cos(dp->eP);
  //sinph=sin(dp->eP);
  sincos(dp->eP,&sinph,&cosph);
  ut=a*(cosph*up-sinph*vp);
  vt=b*(sinph*up+cosph*vp);

  /* Fourier TF is normalized with integrated flux,
    so to get the peak value right, scale the flux */
  //return 0.5*exp(-(ut*ut+vt*vt))/sqrt(2.0*a*b);
  return exp(-2.0*M_PI*M_PI*(ut*ut+vt*vt));
}

complex double
ring_contrib(void*dd, double u, double v, double w) {
  exinfo_ring *dp=(exinfo_ring*)dd;
  double up,vp,a,b;

  /* first the rotation due to projection */
  up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
  vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);

  a=dp->eX; /* diameter */ 
  b=sqrt(up*up+vp*vp)*a*2.0*M_PI; 

  return j0(b);
}

complex double
disk_contrib(void*dd, double u, double v, double w) {
  exinfo_disk *dp=(exinfo_disk*)dd;
  double up,vp,a,b;

  /* first the rotation due to projection */
  up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
  vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);

  a=dp->eX; /* diameter */ 
  b=sqrt(up*up+vp*vp)*a*2.0*M_PI; 

  return j1(b);
}

/* time smearing TMS eq. 6.80 for EW-array formula 
  note u,v,w: meter/c so multiply by freq. to get wavelength */
double
time_smear(double ll,double mm,double dec0,double tdelta,double u,double v,double w,double freq0) {
  /* baseline length in lambda */
  double bl=sqrt(u*u+v*v+w*w)*freq0;
  /* source dist */
  double ds=sin(dec0)*mm;
  double r1=sqrt(ll*ll+ds*ds);
  /* earch angular vel x time x baseline length x source dist  */
  double prod=7.2921150e-5*tdelta*bl*r1;
  if (prod>CLM_EPSILON) {
   return 1.0645*erf(0.8326*prod)/prod;
  } else {
   return 1.0;
  }
}
 
/* worker thread function for prediction */
static void *
predict_threadfn(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,cn;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */

 complex double C[4];
 double freq0=t->freq0;
 double fdelta2=t->fdelta*0.5;
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   memset(&(t->x[8*ci]),0,sizeof(double)*8);
   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {
    for (cm=0; cm<(t->M); cm++) { /* clusters */

     memset(C,0,sizeof(complex double)*4);
/*****************************************************************/
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
     /* replacing sincos() with sin() and cos() to enable vectorization */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHi[cn]=sin(G[cn]*freq0);
     }
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHr[cn]=cos(G[cn]*freq0);
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
       II[cn]=t->carr[cm].sI[cn];
       QQ[cn]=t->carr[cm].sQ[cn];
       UU[cn]=t->carr[cm].sU[cn];
       VV[cn]=t->carr[cm].sV[cn];
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

/*****************************************************************/
     /* add to baseline visibilities */
     t->x[8*ci]+=creal(C[0]);
     t->x[8*ci+1]+=cimag(C[0]);
     t->x[8*ci+2]+=creal(C[1]);
     t->x[8*ci+3]+=cimag(C[1]);
     t->x[8*ci+4]+=creal(C[2]);
     t->x[8*ci+5]+=cimag(C[2]);
     t->x[8*ci+6]+=creal(C[3]);
     t->x[8*ci+7]+=cimag(C[3]);
    }
   }
 }

 return NULL;
}

int
predict_visibilities(double *u, double *v, double *w, double *x, int N,   
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, int Nt) {
  /* u,v,w : size Nbase*tilesz x 1  x: size Nbase*8*tilesz x 1 */
  /* barr: size Nbase*tilesz x 1 carr: size Mx1 */

  int nth,nth1,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=Nbase*tilesz;

  /* calculate min baselines a thread can handle */
  //Nthb0=ceil((double)Nbase1/(double)Nt);
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
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]); 
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].freq0=freq0;
    threaddata[nth].fdelta=fdelta;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
#ifdef DEBUG
    printf("thread %d writing to data from %d baselines %d\n",nth,8*ci,Nthb);
#endif
    
    
    pthread_create(&th_array[nth],&attr,predict_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for precalculation*/
static void *
precal_threadfn(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 /* memory ordering: x[0:4M-1] baseline 0
                     x[4M:2*4M-1] baseline 1 ... */
 int ci,cm,cn;
 int M=(t->M);
 double uvdist;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */
 complex double C[4];
 double freq0=t->freq0;
 double fdelta2=t->fdelta*0.5;

 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   memset(&(t->coh[4*M*ci]),0,sizeof(complex double)*4*M);
   /* even if this baseline is flagged, we do compute */
    for (cm=0; cm<M; cm++) { /* clusters */
     memset(C,0,sizeof(complex double)*4);
/*****************************************************************/
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
     /* replacing sincos() with sin() and cos() to enable vectorization */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHi[cn]=sin(G[cn]*freq0);
     }
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHr[cn]=cos(G[cn]*freq0);
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
       II[cn]=t->carr[cm].sI[cn];
       QQ[cn]=t->carr[cm].sQ[cn];
       UU[cn]=t->carr[cm].sU[cn];
       VV[cn]=t->carr[cm].sV[cn];
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

/*****************************************************************/
     /* add to baseline visibilities */
     t->coh[4*M*ci+4*cm]=C[0];
     t->coh[4*M*ci+4*cm+1]=C[1];
     t->coh[4*M*ci+4*cm+2]=C[2];
     t->coh[4*M*ci+4*cm+3]=C[3];
    }
    if (!t->barr[ci+t->boff].flag) {
    /* change the flag to 2 if baseline length is < uvmin or > uvmax */
    uvdist=sqrt(t->u[ci]*t->u[ci]+t->v[ci]*t->v[ci])*t->freq0;
    if (uvdist<t->uvmin || uvdist>t->uvmax) {
      t->barr[ci+t->boff].flag=2;
    }
   }
 }

 return NULL;
}


int
precalculate_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int Nt) {
  /* u,v,w : size Nbasex 1  x: size Nbase*4*M x 1 */
  /* barr: size Nbasex 1 carr: size Mx1 */
  /* ordering of x: [0,4M-1] coherencies for baseline 0
                    [4M,2*4M-1] coherencies for baseline 1 ... */

  int nth,nth1,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

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
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]); 
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].uvmin=uvmin;
    threaddata[nth].uvmax=uvmax;
    threaddata[nth].coh=&(x[4*M*ci]);
    threaddata[nth].freq0=freq0;
    threaddata[nth].fdelta=fdelta;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
    pthread_create(&th_array[nth],&attr,precal_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for precalculation*/
static void *
precal_threadfn_multifreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 /* memory ordering: x[0:4M-1] x[4 M Nbase:4 M Nbase+4M-1] baseline 0
                     x[4M:2*4M-1] x[4 M Nbase+4M:4 M Nbase+2*4M-1] baseline 1 ... 
  for each channel, offset is 4 M Nbase */
 int ci,cm,cn;
 int M=(t->M);
 double uvdist;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */
 complex double C[4];
 double fdelta2=t->fdelta*0.5;
 int nchan, chanoff=4*M*t->Nbase;
 
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   for (nchan=0; nchan<t->Nchan; nchan++) {
    memset(&(t->coh[4*M*ci+nchan*chanoff]),0,sizeof(complex double)*4*M);
    double freq0=t->freqs[nchan];
   /* even if this baseline is flagged, we do compute */
    for (cm=0; cm<M; cm++) { /* clusters */
     memset(C,0,sizeof(complex double)*4);
/*****************************************************************/
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
     /* replacing sincos() with sin() and cos() to enable vectorization */
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHi[cn]=sin(G[cn]*freq0);
     }
     for (cn=0; cn<t->carr[cm].N; cn++) {
       PHr[cn]=cos(G[cn]*freq0);
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
       II[cn]=t->carr[cm].sI[cn];
       QQ[cn]=t->carr[cm].sQ[cn];
       UU[cn]=t->carr[cm].sU[cn];
       VV[cn]=t->carr[cm].sV[cn];
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

/*****************************************************************/
     /* add to baseline visibilities, with right channel offset */
     t->coh[nchan*chanoff+4*M*ci+4*cm]=C[0];
     t->coh[nchan*chanoff+4*M*ci+4*cm+1]=C[1];
     t->coh[nchan*chanoff+4*M*ci+4*cm+2]=C[2];
     t->coh[nchan*chanoff+4*M*ci+4*cm+3]=C[3];
     } /* end cluster loop */
    } /* end channel loop */
    if (!t->barr[ci+t->boff].flag) {
    /* change the flag to 2 if baseline length is < uvmin or > uvmax */
    uvdist=sqrt(t->u[ci]*t->u[ci]+t->v[ci]*t->v[ci])*t->freqs[0];
    if (uvdist<t->uvmin || uvdist*t->freqs[t->Nchan-1]>t->uvmax*t->freqs[0]) {
      t->barr[ci+t->boff].flag=2;
    }
   }
 }

 return NULL;
}




int
precalculate_coherencies_multifreq(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double *freqs, int Nchan, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int Nt) {

  int nth,nth1,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

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
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].Nbase=Nbase; /* needed for calculating offset for each channel */
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]); 
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].uvmin=uvmin;
    threaddata[nth].uvmax=uvmax;
    threaddata[nth].coh=&(x[4*M*ci]); /* offset for the 1st channel here */
    threaddata[nth].freqs=freqs;
    threaddata[nth].Nchan=Nchan;
    threaddata[nth].fdelta=fdelta/(double)Nchan;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
    pthread_create(&th_array[nth],&attr,precal_threadfn_multifreq,(void*)(&threaddata[nth]));
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
