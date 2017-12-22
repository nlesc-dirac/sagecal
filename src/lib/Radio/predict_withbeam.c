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
#include "Radio.h"

/* worker thread function for precalculation*/
static void *
precal_threadfn(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 /* memory ordering: x[0:4M-1] baseline 0
                     x[4M:2*4M-1] baseline 2 ... */
 int ci,cm,cn,sta1,sta2,tslot;
 int M=(t->M);
 double uvdist;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */

 complex double C[4];
 double freq0=t->freq0;
 double fdelta2=t->fdelta*0.5;
 for (ci=0; ci<t->Nb; ci++) {
   /* get station ids */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   /* get timeslot */
   tslot=(ci+t->boff)/t->Nbase;
#ifdef DEBUG
   if (tslot>t->tilesz) {
    fprintf(stderr,"%s: %d: timeslot exceed available timeslots\n",__FILE__,__LINE__);
    exit(1);
   }
#endif
   /* reset memory only for initial cluster */
   if (t->clus==0) {
    memset(&(t->coh[4*M*ci]),0,sizeof(complex double)*4*M);
   }
   /* even if this baseline is flagged, we do compute */
   cm=t->clus;  /* predict for only 1 cluster */
   memset(C,0,sizeof(complex double)*4);
   /* iterate over the sky model and calculate contribution */
/************************************************************/
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
       /* get array factor for these 2 stations, at given time */
       double af1=t->arrayfactor[cn*(t->N*t->tilesz)+tslot*t->N+sta1];
       double af2=t->arrayfactor[cn*(t->N*t->tilesz)+tslot*t->N+sta2];

       /* freq smearing : extra term delta * sinc(delta/2 * phterm) */
       if (G[cn]!=0.0) {
         double smfac=G[cn]*fdelta2;
         double sinph=sin(smfac)/smfac;
         G[cn]=fabs(sinph);
       } else {
         G[cn]=1.0;
       }
       G[cn]*=af1*af2;
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


/************************************************************/
    /* add to baseline visibilities */
    t->coh[4*M*ci+4*cm]+=C[0];
    t->coh[4*M*ci+4*cm+1]+=C[1];
    t->coh[4*M*ci+4*cm+2]+=C[2];
    t->coh[4*M*ci+4*cm+3]+=C[3];
    if (t->clus==0) {
     if (!t->barr[ci+t->boff].flag) {
     /* change the flag to 2 if baseline length is < uvmin or > uvmax */
     uvdist=sqrt(t->u[ci]*t->u[ci]+t->v[ci]*t->v[ci])*t->freq0;
     if (uvdist<t->uvmin || uvdist>t->uvmax) {
       t->barr[ci+t->boff].flag=2;
     }
    }
   }
 }

 return NULL;
}

/* worker thread function for precalculation of array factor */
static void *
precalbeam_threadfn(void *data) {
 thread_data_arrayfac_t *t=(thread_data_arrayfac_t*)data;
 
 int cm,cn,ct,cf;
 /* ordering of beamgain : Nstationxtime x source */
 cm=t->cid;  /* predict for only this cluster */
 for (cn=t->soff; cn<t->soff+t->Ns; cn++) {
  //printf("clus=%d src=%d total=%d freq=%d %e %e \n",cm,cn,t->Ns,t->Nf,t->carr[cm].ra[cn],t->carr[cm].sI[cn]);
  /* iterate over frequencies */
  for (cf=0; cf<t->Nf; cf++) {
   /* iterate over all timeslots */
   for (ct=0;ct<t->Ntime;ct++) {
    arraybeam(t->carr[cm].ra[cn], t->carr[cm].dec[cn], t->ra0, t->dec0, t->freqs[cf], t->freq0, t->N, t->longitude, t->latitude, t->time_utc[ct], t->Nelem, t->xx, t->yy, t->zz, &(t->beamgain[cn*(t->N*t->Ntime*t->Nf)+cf*(t->N*t->Ntime)+ct*t->N]));
   }
  }
 }

 return NULL;
}



int
precalculate_coherencies_withbeam(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tilesz, int *Nelem, double **xx, double **yy, double **zz, int Nt) {

  int nth,ci,ncl;

  int Nthb0,Nthb,nth1,Ns0;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;
  double *beamgain;
  thread_data_arrayfac_t *beamdata;

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
  if ((beamdata=(thread_data_arrayfac_t*)malloc((size_t)Nt*sizeof(thread_data_arrayfac_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* set common parameters, and split baselines to threads */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
     /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
     /* determine actual no. of baselines */
     if (ci+Nthb0<Nbase) {
      Nthb=Nthb0;
     } else {
      Nthb=Nbase-ci;
     }

     threaddata[nth].N=N;
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
     threaddata[nth].tilesz=tilesz;
     threaddata[nth].Nbase=N*(N-1)/2;
    
     /* next baseline set */
     ci=ci+Nthb;
  }

/******************** loop over clusters *****************************/
  for (ncl=0; ncl<M; ncl++) {
   /* first precalculate arrayfactor for all sources in this cluster */
   /* for each source : N*tilesz beams, so for a cluster with M: M*N*tilesz */
   if ((beamgain=(double*)calloc((size_t)N*tilesz*carr[ncl].N,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
   }

   Ns0=(carr[ncl].N+Nt-1)/Nt; /* sources per thread */

    ci=0;
    for (nth1=0;  nth1<Nt && ci<carr[ncl].N; nth1++) {
     /* determine actual no. of sources */
     if (ci+Ns0<carr[ncl].N) {
      Nthb=Ns0;
     } else {
      Nthb=carr[ncl].N-ci;
     }
//printf("cluster %d th %d sources %d start %d\n",ncl,nth1,carr[ncl].N,ci);
     beamdata[nth1].Ns=Nthb;
     beamdata[nth1].soff=ci;
     beamdata[nth1].Ntime=tilesz;
     beamdata[nth1].time_utc=time_utc;
     beamdata[nth1].N=N;
     beamdata[nth1].longitude=longitude;
     beamdata[nth1].latitude=latitude;
     beamdata[nth1].ra0=ph_ra0;
     beamdata[nth1].dec0=ph_dec0;
     beamdata[nth1].freq0=ph_freq0;
     beamdata[nth1].Nf=1;
     beamdata[nth1].freqs=&freq0; /* only 1 freq */
   
     beamdata[nth1].Nelem=Nelem;
     beamdata[nth1].xx=xx;
     beamdata[nth1].yy=yy;
     beamdata[nth1].zz=zz;
     beamdata[nth1].carr=carr;
     beamdata[nth1].cid=ncl;
     beamdata[nth1].barr=barr;
  
     beamdata[nth1].beamgain=beamgain;
     pthread_create(&th_array[nth1],&attr,precalbeam_threadfn,(void*)(&beamdata[nth1]));
     
     ci=ci+Nthb;
   }
   /* now wait for threads to finish */
   for(ci=0; ci<nth1; ci++) {
    pthread_join(th_array[ci],NULL);
   }

 
 
   /* iterate over threads, allocating baselines per thread */
   for(ci=0; ci<nth; ci++) {
     threaddata[ci].clus=ncl;
     threaddata[ci].arrayfactor=beamgain;
     pthread_create(&th_array[ci],&attr,precal_threadfn,(void*)(&threaddata[ci]));
   }

   /* now wait for threads to finish */
   for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
   }

   free(beamgain);
  }
/******************** end loop over clusters *****************************/

 pthread_attr_destroy(&attr);

 free(th_array);
 free(threaddata);
 free(beamdata);


 return 0;
}



/* worker thread function for prediction 
   */
static void *
visibilities_threadfn_multifreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,cf,cn,sta1,sta2,tslot;
 double freq0;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */

 complex double C[4];
 int Ntilebase=(t->Nbase)*(t->tilesz);
 double fdelta2=t->fdelta*0.5;
 for (ci=0; ci<t->Nb; ci++) {
   /* get station ids */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   /* get timeslot */
   tslot=(ci+t->boff)/t->Nbase;
#ifdef DEBUG
   if (tslot>t->tilesz) {
    fprintf(stderr,"%s: %d: timeslot exceed available timeslots\n",__FILE__,__LINE__);
    exit(1);
   }
#endif

   /* iterate over the sky model and calculate contribution */
   /* if this baseline is flagged, we do not compute */
   cm=t->clus; /* only 1 cluster */
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
       /* get array factor for these 2 stations, at given time */
       double af1=t->arrayfactor[cn*(t->N*t->tilesz*t->Nchan)+cf*(t->N*t->tilesz)+tslot*t->N+sta1];
       double af2=t->arrayfactor[cn*(t->N*t->tilesz*t->Nchan)+cf*(t->N*t->tilesz)+tslot*t->N+sta2];
       G[cn] *=af1*af2;
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
      if (t->add_to_data==SIMUL_ONLY || t->add_to_data==SIMUL_ADD) {
        /* add to baseline visibilities */
        t->x[8*ci+cf*Ntilebase*8]+=creal(C[0]);
        t->x[8*ci+1+cf*Ntilebase*8]+=cimag(C[0]);
        t->x[8*ci+2+cf*Ntilebase*8]+=creal(C[1]);
        t->x[8*ci+3+cf*Ntilebase*8]+=cimag(C[1]);
        t->x[8*ci+4+cf*Ntilebase*8]+=creal(C[2]);
        t->x[8*ci+5+cf*Ntilebase*8]+=cimag(C[2]);
        t->x[8*ci+6+cf*Ntilebase*8]+=creal(C[3]);
        t->x[8*ci+7+cf*Ntilebase*8]+=cimag(C[3]);
      } else if (t->add_to_data==SIMUL_SUB) {
        /* subtract from baseline visibilities */
        t->x[8*ci+cf*Ntilebase*8]-=creal(C[0]);
        t->x[8*ci+1+cf*Ntilebase*8]-=cimag(C[0]);
        t->x[8*ci+2+cf*Ntilebase*8]-=creal(C[1]);
        t->x[8*ci+3+cf*Ntilebase*8]-=cimag(C[1]);
        t->x[8*ci+4+cf*Ntilebase*8]-=creal(C[2]);
        t->x[8*ci+5+cf*Ntilebase*8]-=cimag(C[2]);
        t->x[8*ci+6+cf*Ntilebase*8]-=creal(C[3]);
        t->x[8*ci+7+cf*Ntilebase*8]-=cimag(C[3]);
      }
     }

 }
 return NULL;
}




int
predict_visibilities_multifreq_withbeam(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int *Nelem, double **xx, double **yy, double **zz, int Nt, int add_to_data) {
  int nth,nth1,ci,ncl,Ns0;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;
  double *beamgain;
  thread_data_arrayfac_t *beamdata;

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
  if ((beamdata=(thread_data_arrayfac_t*)malloc((size_t)Nt*sizeof(thread_data_arrayfac_t)))==0) {
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

    threaddata[nth].N=N;
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
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].freqs=freqs;
    threaddata[nth].Nchan=Nchan;
    threaddata[nth].fdelta=fdelta/(double)Nchan;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
    threaddata[nth].add_to_data=add_to_data;
    
   
    /* next baseline set */
    ci=ci+Nthb;
  }

/******************** loop over clusters *****************************/
  for (ncl=0; ncl<M; ncl++) {
   /* first precalculate arrayfactor for all sources in this cluster */
   /* for each source : N*tilesz*Nchan beams, so for a cluster with M: M*N*Nchan*tilesz */
   if ((beamgain=(double*)calloc((size_t)N*tilesz*carr[ncl].N*Nchan,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
   }

   Ns0=(carr[ncl].N+Nt-1)/Nt; /* sources per thread */
    ci=0;
    for (nth1=0;  nth1<Nt && ci<carr[ncl].N; nth1++) {
     /* determine actual no. of sources */
     if (ci+Ns0<carr[ncl].N) {
      Nthb=Ns0;
     } else {
      Nthb=carr[ncl].N-ci;
     }
//printf("cluster %d th %d sources %d start %d\n",ncl,nth1,carr[ncl].N,ci);
     beamdata[nth1].Ns=Nthb;
     beamdata[nth1].soff=ci;
     beamdata[nth1].Ntime=tilesz;
     beamdata[nth1].time_utc=time_utc;
     beamdata[nth1].N=N;
     beamdata[nth1].longitude=longitude;
     beamdata[nth1].latitude=latitude;
     beamdata[nth1].ra0=ph_ra0;
     beamdata[nth1].dec0=ph_dec0;
     beamdata[nth1].freq0=ph_freq0;
     beamdata[nth1].Nf=Nchan;
     beamdata[nth1].freqs=freqs; 

     beamdata[nth1].Nelem=Nelem;
     beamdata[nth1].xx=xx;
     beamdata[nth1].yy=yy;
     beamdata[nth1].zz=zz;
     beamdata[nth1].carr=carr;
     beamdata[nth1].cid=ncl;
     beamdata[nth1].barr=barr;

     beamdata[nth1].beamgain=beamgain;
     pthread_create(&th_array[nth1],&attr,precalbeam_threadfn,(void*)(&beamdata[nth1]));

     ci=ci+Nthb;
   }
   /* now wait for threads to finish */
   for(ci=0; ci<nth1; ci++) {
    pthread_join(th_array[ci],NULL);
   }


   for(ci=0; ci<nth; ci++) {
     threaddata[ci].clus=ncl;
     threaddata[ci].arrayfactor=beamgain;
     pthread_create(&th_array[ci],&attr,visibilities_threadfn_multifreq,(void*)(&threaddata[ci]));
   }

   /* now wait for threads to finish */
   for(ci=0; ci<nth; ci++) {
     pthread_join(th_array[ci],NULL);
   }


   free(beamgain);
  }
/******************** end loop over clusters *****************************/


 pthread_attr_destroy(&attr);

 free(th_array);
 free(threaddata);
 free(beamdata);

 return 0;

}

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

//printf("A=[%lf+j*(%lf) %lf+j*(%lf)\n %lf+j*(%lf) %lf+j*(%lf)];\n",creal(a[0]),cimag(a[0]),creal(a[2]),cimag(a[2]),creal(a[1]),cimag(a[1]),creal(a[3]),cimag(a[3]));

 det=a[0]*a[3]-a[1]*a[2];
 if (sqrt(cabs(det))<=rho) {
  det+=rho;
 }
 det=1.0/det;
 b[0]=a[3]*det;
 b[1]=-a[1]*det; 
 b[2]=-a[2]*det;
 b[3]=a[0]*det;

//printf("B=[%lf+j*(%lf) %lf+j*(%lf)\n %lf+j*(%lf) %lf+j*(%lf)];\n",creal(b[0]),cimag(b[0]),creal(b[2]),cimag(b[2]),creal(b[1]),cimag(b[1]),creal(b[3]),cimag(b[3]));

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



/* worker thread function for subtraction
  also correct residual with solutions for cluster id 0 */
static void *
residual_threadfn_multifreq(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,cf,cn,sta1,sta2,tslot;
 double *pm;
 double freq0;
 double *PHr=0,*PHi=0,*G=0,*II=0,*QQ=0,*UU=0,*VV=0; /* arrays to store calculations */
 double fdelta2=t->fdelta*0.5;

 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int Ntilebase=(t->Nbase)*(t->tilesz);
 int px;
 for (ci=0; ci<t->Nb; ci++) {
   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
   /* get timeslot */
   tslot=(ci+t->boff)/t->Nbase;
#ifdef DEBUG
   if (tslot>t->tilesz) {
    fprintf(stderr,"%s: %d: timeslot exceed available timeslots\n",__FILE__,__LINE__);
    exit(1);
   }
#endif

     int cmt=t->clus; /* only 1 cluster, assumed positive */
      /* check if cluster id >=0 to do a subtraction */
     if (cmt>=0 && t->carr[cmt].id>=0) {
       cm=cmt;
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
     px=(ci+t->boff)/((Ntilebase+t->carr[cmt].nchunk-1)/t->carr[cmt].nchunk);
     //printf("base %d, cluster %d, parm off %d abs %d\n",t->bindex[ci],cm,px,t->carr[cm].p[px]);
     pm=&(t->p[t->carr[cmt].p[px]]);
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
       /* get array factor for these 2 stations, at given time */
       double af1=t->arrayfactor[cn*(t->N*t->tilesz*t->Nchan)+cf*(t->N*t->tilesz)+tslot*t->N+sta1];
       double af2=t->arrayfactor[cn*(t->N*t->tilesz*t->Nchan)+cf*(t->N*t->tilesz)+tslot*t->N+sta2];
       G[cn] *=af1*af2;
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
   /* if -ve cluster is given (only once), final correction is done */
   if (cmt<0 && t->pinv) {
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
calculate_residuals_multifreq_withbeam(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,
double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, int Nt, int ccid, double rho, int phase_only) {
  int nth,nth1,ci,cj,ncl,Ns0;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;
  double *beamgain;
  thread_data_arrayfac_t *beamdata;

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
  if ((beamdata=(thread_data_arrayfac_t*)malloc((size_t)Nt*sizeof(thread_data_arrayfac_t)))==0) {
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

    threaddata[nth].N=N;
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
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].tilesz=tilesz;
    threaddata[nth].freqs=freqs;
    threaddata[nth].Nchan=Nchan;
    threaddata[nth].fdelta=fdelta/(double)Nchan;
    threaddata[nth].tdelta=tdelta;
    threaddata[nth].dec0=dec0;
    
    /* next baseline set */
    ci=ci+Nthb;
  }

/******************** loop over clusters *****************************/
  for (ncl=0; ncl<M; ncl++) {
   /* first precalculate arrayfactor for all sources in this cluster */
   /* for each source : N*tilesz*Nchan beams, so for a cluster with M: M*N*Nchan*tilesz */
   if ((beamgain=(double*)calloc((size_t)N*tilesz*carr[ncl].N*Nchan,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
   }
   Ns0=(carr[ncl].N+Nt-1)/Nt; /* sources per thread */

    ci=0;
    for (nth1=0;  nth1<Nt && ci<carr[ncl].N; nth1++) {
     /* determine actual no. of sources */
     if (ci+Ns0<carr[ncl].N) {
      Nthb=Ns0;
     } else {
      Nthb=carr[ncl].N-ci;
     }
//printf("cluster %d th %d sources %d start %d\n",ncl,nth1,carr[ncl].N,ci);
     beamdata[nth1].Ns=Nthb;
     beamdata[nth1].soff=ci;
     beamdata[nth1].Ntime=tilesz;
     beamdata[nth1].time_utc=time_utc;
     beamdata[nth1].N=N;
     beamdata[nth1].longitude=longitude;
     beamdata[nth1].latitude=latitude;
     beamdata[nth1].ra0=ph_ra0;
     beamdata[nth1].dec0=ph_dec0;
     beamdata[nth1].freq0=ph_freq0;
     beamdata[nth1].Nf=Nchan;
     beamdata[nth1].freqs=freqs;
     beamdata[nth1].Nelem=Nelem;
     beamdata[nth1].xx=xx;
     beamdata[nth1].yy=yy;
     beamdata[nth1].zz=zz;
     beamdata[nth1].carr=carr;
     beamdata[nth1].cid=ncl;
     beamdata[nth1].barr=barr;

     beamdata[nth1].beamgain=beamgain;
     pthread_create(&th_array[nth1],&attr,precalbeam_threadfn,(void*)(&beamdata[nth1]));

     ci=ci+Nthb;
   }
   /* now wait for threads to finish */
   for(ci=0; ci<nth1; ci++) {
    pthread_join(th_array[ci],NULL);
   }


   for(ci=0; ci<nth; ci++) {
     threaddata[ci].clus=ncl;
     threaddata[ci].arrayfactor=beamgain;
     pthread_create(&th_array[ci],&attr,residual_threadfn_multifreq,(void*)(&threaddata[ci]));
   }
   /* now wait for threads to finish */
   for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
   }

   free(beamgain);
  }

  /* now run with a -ve cluster id if correction is needed */
  for(ci=0; ci<nth; ci++) {
     threaddata[ci].clus=-1;
     threaddata[ci].arrayfactor=0;
     pthread_create(&th_array[ci],&attr,residual_threadfn_multifreq,(void*)(&threaddata[ci]));
  }
   /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
  }

/******************** end loop over clusters *****************************/

 pthread_attr_destroy(&attr);

 free(th_array);
 free(threaddata);
 free(pinv);
 free(beamdata);

 return 0;

}



typedef struct thread_data_precess_t_ {
  clus_source_t *carr;
  int Ns; /* how many clusters */
  int soff; /* offset of cluster index */
  double *Tr; /* transform params to precess from J2000 */
} thread_data_precess_t;



static void *
precess_threadfn(void *data) {
  thread_data_precess_t *t=(thread_data_precess_t*)data;
  int ncl,ci;
  double newra,newdec;
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
   for (ci=0; ci<t->carr[ncl].N; ci++) {
     precession(t->carr[ncl].ra[ci], t->carr[ncl].dec[ci],t->Tr,&newra,&newdec);
     t->carr[ncl].ra[ci]=newra;
     t->carr[ncl].dec[ci]=newdec;
   }
  } 
  return NULL;
}

int
precess_source_locations(double jd_tdb, clus_source_t *carr, int M, double *ra_beam, double *dec_beam, int Nt) {

  int nth,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_precess_t *threaddata;

  /* calculate min clusters thread can handle */
  Nthb0=(M+Nt-1)/Nt;

  /* setup threads : note: Ngpu is no of GPUs used */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_precess_t*)malloc((size_t)Nt*sizeof(thread_data_precess_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  double Tr[9];
  get_precession_params(jd_tdb,Tr);

  /* set common parameters, and split clusters to threads */
  ci=0;
  for (nth=0;  nth<Nt && ci<M; nth++) {
     if (ci+Nthb0<M) {
      Nthb=Nthb0;
     } else {
      Nthb=M-ci;
     }
      
     threaddata[nth].carr=carr;
     threaddata[nth].Tr=Tr;

     threaddata[nth].Ns=Nthb;
     threaddata[nth].soff=ci;

    
     pthread_create(&th_array[nth],&attr,precess_threadfn,(void*)(&threaddata[nth]));
     /* next source set */
     ci=ci+Nthb;
  }

     

  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
  }



 pthread_attr_destroy(&attr);
 free(threaddata);
 free(th_array);


 /* change beam pointing direction as well */
 double newra,newdec;
 precession(*ra_beam, *dec_beam,Tr,&newra,&newdec);
 *ra_beam=newra;
 *dec_beam=newdec;

 return 0;
}
