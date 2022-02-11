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
#include "Dirac.h"
#include "Dirac_common.h"

/* worker thread function for rearranging coherencies*/
static void *
rearrange_threadfn(void *data) {
 thread_data_coharr_t *t=(thread_data_coharr_t*)data;
 
 /* NOTE: t->ddcoh must have all zeros intially (using calloc) */
 int ci,cj;
 double *realcoh=(double*)t->coh;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
   if (!t->barr[ci].flag) {
     t->ddbase[2*ci]=(short)t->barr[ci].sta1;
     t->ddbase[2*ci+1]=(short)t->barr[ci].sta2;
     /* loop over directions and copy coherencies */
     for (cj=0; cj<t->M; cj++) {
       memcpy(&(t->ddcoh[cj*(t->Nbase)*8+8*ci]),&realcoh[8*cj+8*(t->M)*ci],8*sizeof(double));
     }
   } else {
     t->ddbase[2*ci]=t->ddbase[2*ci+1]=-1;
   }
 }

 return NULL;
}

/* rearranges coherencies for GPU use later */
/* barr: 2*Nbase x 1
   coh: M*Nbase*4 x 1 complex
   ddcoh: M*Nbase*8 x 1
   ddbase: 2*Nbase x 1 (sta1,sta2) == -1 if flagged
*/
int
rearrange_coherencies(int Nbase, baseline_t *barr, complex double *coh, double *ddcoh, short *ddbase, int M, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_coharr_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_coharr_t*)malloc((size_t)Nt*sizeof(thread_data_coharr_t)))==0) {
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
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].M=M;
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].barr=barr;
    threaddata[nth].coh=coh;
    threaddata[nth].ddcoh=ddcoh;
    threaddata[nth].ddbase=ddbase;
    pthread_create(&th_array[nth],&attr,rearrange_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for rearranging baselines*/
static void *
rearrange_base_threadfn(void *data) {
 thread_data_coharr_t *t=(thread_data_coharr_t*)data;
 
 int ci;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
   if (!t->barr[ci].flag) {
     t->ddbase[2*ci]=(short)t->barr[ci].sta1;
     t->ddbase[2*ci+1]=(short)t->barr[ci].sta2;
   } else {
     t->ddbase[2*ci]=t->ddbase[2*ci+1]=-1;
   }
 }

 return NULL;
}

/* rearranges baselines for GPU use later */
/* barr: 2*Nbase x 1
   ddbase: 2*Nbase x 1
*/
int
rearrange_baselines(int Nbase, baseline_t *barr, short *ddbase, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_coharr_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_coharr_t*)malloc((size_t)Nt*sizeof(thread_data_coharr_t)))==0) {
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
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].barr=barr;
    threaddata[nth].ddbase=ddbase;
    pthread_create(&th_array[nth],&attr,rearrange_base_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for preflag data */
static void *
preflag_threadfn(void *data) {
 thread_data_preflag_t *t=(thread_data_preflag_t*)data;
 
 int ci;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
   if (t->flag[ci]>0.0) { /* flagged data */
     t->barr[ci].flag=1;
     /* set data points to 0 */
     t->x[8*ci]=0.0;
     t->x[8*ci+1]=0.0;
     t->x[8*ci+2]=0.0;
     t->x[8*ci+3]=0.0;
     t->x[8*ci+4]=0.0;
     t->x[8*ci+5]=0.0;
     t->x[8*ci+6]=0.0;
     t->x[8*ci+7]=0.0;
   } else {
     t->barr[ci].flag=0;
   }
 }

 return NULL;
}

/* update baseline flags, also make data zero if flagged
  this is needed for solving (calculate error) ignore flagged data */
/* Nbase: total actual data points = Nbasextilesz
   flag: flag array Nbasex1
   barr: baseline array Nbasex1
   x: data Nbase*8 x 1 ( 8 value per baseline ) 
   Nt: no of threads 
*/
int
preset_flags_and_data(int Nbase, double *flag, baseline_t *barr, double *x, int Nt){
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_preflag_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_preflag_t*)malloc((size_t)Nt*sizeof(thread_data_preflag_t)))==0) {
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
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].flag=flag;
    threaddata[nth].barr=barr;
    threaddata[nth].x=x;
    pthread_create(&th_array[nth],&attr,preflag_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for type conversion */
static void *
double_to_float_threadfn(void *data) {
 thread_data_typeconv_t *t=(thread_data_typeconv_t*)data;
 
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   t->farr[ci]=(float)t->darr[ci];
 }
 return NULL;
}
static void *
float_to_double_threadfn(void *data) {
 thread_data_typeconv_t *t=(thread_data_typeconv_t*)data;
 
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   t->darr[ci]=(double)t->farr[ci];
 }
 return NULL;
}

/* convert types */
/* both arrays size nx1 
*/
int
double_to_float(float *farr, double *darr,int n, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_typeconv_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(n+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_typeconv_t*)malloc((size_t)Nt*sizeof(thread_data_typeconv_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<n; nth++) {
    if (ci+Nthb0<n) {
     Nthb=Nthb0;
    } else {
     Nthb=n-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].farr=farr;
    threaddata[nth].darr=darr;
    pthread_create(&th_array[nth],&attr,double_to_float_threadfn,(void*)(&threaddata[nth]));
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

/* convert types */
/* both arrays size nx1 
*/
int
float_to_double(double *darr, float *farr,int n, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_typeconv_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(n+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_typeconv_t*)malloc((size_t)Nt*sizeof(thread_data_typeconv_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<n; nth++) {
    if (ci+Nthb0<n) {
     Nthb=Nthb0;
    } else {
     Nthb=n-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].farr=farr;
    threaddata[nth].darr=darr;
    pthread_create(&th_array[nth],&attr,float_to_double_threadfn,(void*)(&threaddata[nth]));
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


static void *
baselinegen_threadfn(void *data) {
 thread_data_baselinegen_t *t=(thread_data_baselinegen_t*)data;
 
 int ci,cj,sta1,sta2;
 for (ci=t->starti; ci<=t->endi; ci++) {
   sta1=0; sta2=sta1+1;
   for (cj=0; cj<t->Nbase; cj++) {
    t->barr[ci*t->Nbase+cj].sta1=sta1;
    t->barr[ci*t->Nbase+cj].sta2=sta2;
    if(sta2<(t->N-1)) {
     sta2=sta2+1;
    } else {
      if (sta1<(t->N-2)) {
      sta1=sta1+1;
      sta2=sta1+1;
      } else {
       sta1=0;
       sta2=sta1+1;
      }
    }
   }
 }
 return NULL;
}

/* generte baselines -> sta1,sta2 pairs for later use */
/* barr: Nbasextilesz
   N : stations
   Nt : threads 
*/
int
generate_baselines(int Nbase, int tilesz, int N, baseline_t *barr,int Nt) {
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_baselinegen_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(tilesz+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_baselinegen_t*)malloc((size_t)Nt*sizeof(thread_data_baselinegen_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<tilesz; nth++) {
    if (ci+Nthb0<tilesz) {
     Nthb=Nthb0;
    } else {
     Nthb=tilesz-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].N=N;
    threaddata[nth].barr=barr;
    pthread_create(&th_array[nth],&attr,baselinegen_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for counting */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
threadfn_fns_fcount(void *data) {
 thread_data_count_t *t=(thread_data_count_t*)data;

 int ci,sta1,sta2;
 for (ci=0; ci<t->Nb; ci++) {
   /* if this baseline is flagged, we do not compute */

   /* stations for this baseline */
   sta1=(int)t->ddbase[2*(ci+t->boff)];
   sta2=(int)t->ddbase[2*(ci+t->boff)+1];

   if (sta1!=-1 && sta2!=-1) {
   pthread_mutex_lock(&t->mx_array[sta1]);
   t->bcount[sta1]+=1;
   pthread_mutex_unlock(&t->mx_array[sta1]);

   pthread_mutex_lock(&t->mx_array[sta2]);
   t->bcount[sta2]+=1;
   pthread_mutex_unlock(&t->mx_array[sta2]);
   }
 }

 return NULL;
}


/* cont how many baselines contribute to each station */
int
count_baselines(int Nbase, int N, float *iw, short *ddbase, int Nt) {
 pthread_attr_t attr;
 pthread_t *th_array;
 thread_data_count_t *threaddata;
 pthread_mutex_t *mx_array;

 int *bcount;

 int ci,nth1,nth;
 int Nthb0,Nthb;

 Nthb0=(Nbase+Nt-1)/Nt;

 pthread_attr_init(&attr);
 pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

 if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }

 if ((threaddata=(thread_data_count_t*)malloc((size_t)Nt*sizeof(thread_data_count_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
 }

 if ((mx_array=(pthread_mutex_t*)malloc((size_t)N*sizeof(pthread_mutex_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }
 for (ci=0; ci<N; ci++) {
   pthread_mutex_init(&mx_array[ci],NULL);
 }

 if ((bcount=(int*)calloc((size_t)N,sizeof(int)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].ddbase=ddbase;
    threaddata[nth].bcount=bcount; /* note this should be 0 first */
    threaddata[nth].mx_array=mx_array;

    pthread_create(&th_array[nth],&attr,threadfn_fns_fcount,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

  /* calculate inverse count */
  for (nth1=0; nth1<N; nth1++) {
   iw[nth1]=(bcount[nth1]>0?1.0f/(float)bcount[nth1]:0.0f);
  }

 /* scale back weight such that max value is 1 */
 nth1=my_isamax(N,iw,1);
 float maxw=iw[nth1-1]; /* 1 based index */
 if (maxw>0.0f) { /* all baselines are flagged */
  my_sscal(N,1.0f/maxw,iw);
 } 

 for (ci=0; ci<N; ci++) {
   pthread_mutex_destroy(&mx_array[ci]);
 }
 pthread_attr_destroy(&attr);
 free(bcount);
 free(th_array);
 free(mx_array);
 free(threaddata);
 return 0;
}

/* worker thread function for array initializing */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
threadfn_setwt(void *data) {
 thread_data_setwt_t *t=(thread_data_setwt_t*)data;

 int ci;
 for (ci=0; ci<t->Nb; ci++) {
  t->b[ci+t->boff]=t->a;
 }

 return NULL;
}



/* initialize array b (size Nx1) to given value a */
void
setweights(int N, double *b, double a, int Nt) {
 pthread_attr_t attr;
 pthread_t *th_array;
 thread_data_setwt_t *threaddata;

 int ci,nth1,nth;
 int Nthb0,Nthb;

 Nthb0=(N+Nt-1)/Nt;

 pthread_attr_init(&attr);
 pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

 if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }

 if ((threaddata=(thread_data_setwt_t*)malloc((size_t)Nt*sizeof(thread_data_setwt_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
 }


  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<N; nth++) {
    if (ci+Nthb0<N) {
     Nthb=Nthb0;
    } else {
     Nthb=N-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].a=a; 
    threaddata[nth].Nb=Nthb; 
    threaddata[nth].b=b; 

    pthread_create(&th_array[nth],&attr,threadfn_setwt,(void*)(&threaddata[nth]));
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


/* worker thread function for one/zero vector */
static void *
onezero_threadfn(void *data) {
 thread_data_onezero_t *t=(thread_data_onezero_t*)data;
 
 /* NOTE: t->ddcoh must have all zeros intially (using calloc) */
 int ci;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
     if (t->ddbase[3*ci+2]) {
      t->x[8*ci]=1.0f;
      t->x[8*ci+1]=1.0f;
      t->x[8*ci+2]=1.0f;
      t->x[8*ci+3]=1.0f;
      t->x[8*ci+4]=1.0f;
      t->x[8*ci+5]=1.0f;
      t->x[8*ci+6]=1.0f;
      t->x[8*ci+7]=1.0f;
     } 
 }

 return NULL;
}

/* create a vector with 1's at flagged data points */
/* 
   ddbase: 3*Nbase x 1 (sta1,sta2,flag)
   x: 8*Nbase (set to 0's and 1's)
*/
int
create_onezerovec(int Nbase, short *ddbase, float *x, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_onezero_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

  memset(x,0,sizeof(float)*8*Nbase);

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_onezero_t*)malloc((size_t)Nt*sizeof(thread_data_onezero_t)))==0) {
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
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].ddbase=ddbase;
    threaddata[nth].x=x;
    pthread_create(&th_array[nth],&attr,onezero_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for finding sums */
static void *
findsumprod_threadfn(void *data) {
 thread_data_findsumprod_t *t=(thread_data_findsumprod_t*)data;
 
 int ci;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
     float xabs=fabsf(t->x[ci]);
     t->sum1 +=xabs;
     t->sum2 +=xabs*fabsf(t->y[ci]);
 }

 return NULL;
}

/* 
  find sum1=sum(|x|), and sum2=y^T |x|
  x,y: nx1 arrays
*/
int
find_sumproduct(int N, float *x, float *y, float *sum1, float *sum2, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_findsumprod_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(N+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_findsumprod_t*)malloc((size_t)Nt*sizeof(thread_data_findsumprod_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<N; nth++) {
    /* determine actual no. of baselines */
    if (ci+Nthb0<N) {
     Nthb=Nthb0;
    } else {
     Nthb=N-ci;
    }
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].x=x;
    threaddata[nth].y=y;
    threaddata[nth].sum1=0.0f;
    threaddata[nth].sum2=0.0f;
    pthread_create(&th_array[nth],&attr,findsumprod_threadfn,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  *sum1=*sum2=0.0f;
  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
   *sum1+=threaddata[nth1].sum1;
   *sum2+=threaddata[nth1].sum2;
  }

 pthread_attr_destroy(&attr);

 free(th_array);
 free(threaddata);


 return 0;
}
