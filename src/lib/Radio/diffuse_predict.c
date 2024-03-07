/*
 *
 Copyright (C) 2023 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <stdlib.h>
#include <string.h>

#include "Dirac.h"
#include "Dirac_radio.h"
#ifdef HAVE_CUDA
#include "Dirac_GPUtune.h"
#endif
#include <math.h>

typedef struct thread_data_shap_ {
  int Nb; /* no of baselines this thread handles */
  int N; /* no of stations */
  int boff; /* baseline offset per thread */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *u,*v,*w; /* pointers to uwv arrays,size Nbx1 */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  int cid; /* which cluster id to predict */
  int sid; /* which source id to predict 0,1,..max sources */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  /* following used for freq/time smearing calculation */
  double freq0;
  double fdelta;

  /* shapelet model info */
  complex double *modes; /* all modes, 4*n0*n0*stations^2 (only half is calculated) */
  int modes_n0; /* basis n0xn0 */
  double modes_beta; /* scale */

#ifdef HAVE_CUDA
  int tid; /* this thread id */
  taskhist *hst; /* for load balancing GPUs */
#endif /* HAVE_CUDA */
} thread_data_shap_t;


typedef struct thread_data_stat_ {
  int off; /* offset in station count */
  int Nstat; /* num. stations this thread */
  int N; /* total stations */

  int sL,sM,sN; /* shapelet model orders (L^2,M^2,N^2) */
  double alpha,beta,gamma; /* shapelet model scales */
  complex double *h_arr,*f_arr,*g_arr; /* arrays for mode coefficients L^2,M^2,N^2 size */
  double *Cf; /* product tensor */
  int hermitian; /* if 1, find h = f x g^H  else, h = f x g */
} thread_data_stat_t;

static void *
shapelet_pred_threadfn(void *data) {
  thread_data_shap_t *t=(thread_data_shap_t*)data;

  complex double coh[4];
  int cm=t->cid;
  int sid=t->sid;
  double freq0=t->freq0;

  double fdelta2=t->fdelta*0.5;

  /* we only predict for cluster id cm, and for only one source sid */
  for (int ci=0; ci<t->Nb; ci++) {
   int stat1=t->barr[ci+t->boff].sta1;
   int stat2=t->barr[ci+t->boff].sta2;
   double Gn=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[sid]+t->v[ci]*t->carr[cm].mm[sid]+t->w[ci]*t->carr[cm].nn[sid]);
   double sin_n,cos_n;
   sincos(freq0*Gn,&sin_n,&cos_n);
   /* freq smearing */
   if (Gn!=0.0) {
     double smfac=Gn*fdelta2;
     Gn =fabs(sin(smfac)/smfac);
   } else {
     Gn =1.0;
   }
   /* multiply (re, im) phase term with smear factor */
   sin_n*=Gn;
   cos_n*=Gn;

   /* shapelet contribution */
   /* modes: n0*n0*4 values &Jp_C_Jq[4*n0*n0*(stat1*t->N+stat2)] */
   complex double *modes=&(t->modes[4*t->modes_n0*t->modes_n0*(stat1*t->N+stat2)]);
   shapelet_contrib_vector(modes,t->modes_n0,t->modes_beta,t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0,coh);
   complex double phterm=cos_n+_Complex_I*sin_n;

   coh[0]*=phterm;
   coh[1]*=phterm;
   coh[2]*=phterm;
   coh[3]*=phterm;
   /* add or replace coherencies for this cluster */
   if (t->sid==0) {
     /* first source will replace, resetting the accumulation to start from 0 */
     t->coh[4*t->M*ci+4*t->cid]=coh[0];
     t->coh[4*t->M*ci+4*t->cid+1]=coh[1];
     t->coh[4*t->M*ci+4*t->cid+2]=coh[2];
     t->coh[4*t->M*ci+4*t->cid+3]=coh[3];
   } else {
     t->coh[4*t->M*ci+4*t->cid]+=coh[0];
     t->coh[4*t->M*ci+4*t->cid+1]+=coh[1];
     t->coh[4*t->M*ci+4*t->cid+2]+=coh[2];
     t->coh[4*t->M*ci+4*t->cid+3]+=coh[3];
   }
  }

  return NULL;
}

/*****************************************************************************/
#ifdef HAVE_CUDA
#define CUDA_DEBUG
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

static void *
shapelet_pred_threadfn_cuda(void *data) {
  thread_data_shap_t *t=(thread_data_shap_t*)data;

  /* kernel will spawn threads to cover baselines t->Nb,
   * which can be lower than the total baselines */
  int card;
  card=select_work_gpu(MAX_GPU_ID,t->hst);

  cudaError_t err;
  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  /* allocate device mem */
  float *modesd; /* shapelet modes 4*n0*n0 complex (float) values */
  double *cohd; /* coherencies 8x1 */
  float *factd;
  
  err=cudaMalloc((void**) &cohd, 8*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &factd, t->modes_n0*sizeof(float));
  checkCudaError(err,__FILE__,__LINE__);

  complex double coh[4];
  int cm=t->cid;
  int sid=t->sid;
  double freq0=t->freq0;

  double fdelta2=t->fdelta*0.5;

  /* set up factorial array */
  float *fact;
  if ((fact=(float *)calloc((size_t)(t->modes_n0),sizeof(float)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  fact[0]=1.0f;
  for (int ci=1; ci<(t->modes_n0); ci++) {
    fact[ci]=((float)ci)*fact[ci-1];
  }
  err=cudaMemcpy(factd, fact, t->modes_n0*sizeof(float), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  free(fact);

  /* we only predict for cluster id cm, and for only one source sid,
   * so source flux and l,m,n coords are scalars */
   /* we copy u,v,w,l,m,n values to GPU and perform calculation per-baseline,
    * CUDA threads parallelize over the modes : n0xn0 ~ large value */
  for (int ci=0; ci<t->Nb; ci++) {
   int stat1=t->barr[ci+t->boff].sta1;
   int stat2=t->barr[ci+t->boff].sta2;

   double Gn=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[sid]+t->v[ci]*t->carr[cm].mm[sid]+t->w[ci]*t->carr[cm].nn[sid]);
   double sin_n,cos_n;
   sincos(freq0*Gn,&sin_n,&cos_n);
   /* freq smearing */
   if (Gn!=0.0) {
     double smfac=Gn*fdelta2;
     Gn =fabs(sin(smfac)/smfac);
   } else {
     Gn =1.0;
   }
   /* multiply (re, im) phase term with smear factor */
   sin_n*=Gn;
   cos_n*=Gn;

   /* shapelet contribution */
   /* modes: n0*n0*4 values &Jp_C_Jq[4*n0*n0*(stat1*t->N+stat2)] */
   complex double *modes=&(t->modes[4*t->modes_n0*t->modes_n0*(stat1*t->N+stat2)]);
   //shapelet_contrib_vector(modes,t->modes_n0,t->modes_beta,t->u[ci]*freq0,t->v[ci]*freq0,t->w[ci]*freq0,coh);


   dtofcopy(8*t->modes_n0*t->modes_n0,&modesd,(double*)modes);
   cudakernel_calculate_shapelet_coherencies((float)t->u[ci]*freq0,(float)t->v[ci]*freq0,modesd,factd,t->modes_n0,(float)t->modes_beta,cohd);
   err=cudaFree(modesd);
   checkCudaError(err,__FILE__,__LINE__);

   err=cudaMemcpy((double*)coh, cohd, sizeof(double)*8, cudaMemcpyDeviceToHost);
   checkCudaError(err,__FILE__,__LINE__);


   complex double phterm=cos_n+_Complex_I*sin_n;

   coh[0]*=phterm;
   coh[1]*=phterm;
   coh[2]*=phterm;
   coh[3]*=phterm;

   /* add or replace coherencies for this cluster */
   if (t->sid==0) {
     /* first source will replace, resetting the accumulation to start from 0 */
     t->coh[4*t->M*ci+4*t->cid]=coh[0];
     t->coh[4*t->M*ci+4*t->cid+1]=coh[1];
     t->coh[4*t->M*ci+4*t->cid+2]=coh[2];
     t->coh[4*t->M*ci+4*t->cid+3]=coh[3];
   } else {
     t->coh[4*t->M*ci+4*t->cid]+=coh[0];
     t->coh[4*t->M*ci+4*t->cid+1]+=coh[1];
     t->coh[4*t->M*ci+4*t->cid+2]+=coh[2];
     t->coh[4*t->M*ci+4*t->cid+3]+=coh[3];
   }
  }

  cudaDeviceSynchronize();

  err=cudaFree(cohd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(factd);
  checkCudaError(err,__FILE__,__LINE__);

  /* reset error state */
  err=cudaGetLastError();

  return NULL;
}
#endif /* HAVE_CUDA */
/*****************************************************************************/
static void *
shapelet_prod_one_threadfn(void *data) {
  thread_data_stat_t *t=(thread_data_stat_t*)data;

  for (int stat=0; stat<t->Nstat; stat++) {
     //shapelet_product_jones(sp->n0,sp->n0,sh_n0,sp->beta,sp->beta,sh_beta,&C_Jq[4*sp->n0*sp->n0*stat],s_coh,&Zt[4*G*stat],Cf,1);
     shapelet_product_jones(t->sL,t->sM,t->sN,t->alpha,t->beta,t->gamma,&(t->h_arr[4*t->sL*t->sL*(stat+t->off)]),t->f_arr,&(t->g_arr[4*t->sN*t->sN*(stat+t->off)]),t->Cf,t->hermitian);
     // For debugging, C_Jq <= s_coh
     //memcpy(&(t->h_arr[4*t->sL*t->sL*(stat+t->off)]),t->f_arr,t->sL*t->sL*4*sizeof(complex double));
  }

  return NULL;
}


static void *
shapelet_prod_two_threadfn(void *data) {
  thread_data_stat_t *t=(thread_data_stat_t*)data;

  for (int stat1=0; stat1<t->Nstat; stat1++) {
      // use the fact C_pq = C_qp^H
      for (int stat2=stat1+t->off; stat2<t->N; stat2++) {
          //shapelet_product_jones(sp->n0,sh_n0,sp->n0,sp->beta,sh_beta,sp->beta,&Jp_C_Jq[4*sp->n0*sp->n0*(stat1*N+stat2)],&Zt[4*G*stat1],&C_Jq[4*sp->n0*sp->n0*stat2],Cf,0);
          shapelet_product_jones(t->sL,t->sM,t->sN,t->alpha,t->beta,t->gamma,&t->h_arr[4*t->sL*t->sL*((stat1+t->off)*t->N+stat2)],&t->f_arr[4*t->sM*t->sM*(stat1+t->off)],&t->g_arr[4*t->sN*t->sN*stat2],t->Cf,t->hermitian);
          // For debugging, Jp_C_Jq <= C_Jq
          //memcpy(&t->h_arr[4*t->sL*t->sL*((stat1+t->off)*t->N+stat2)],&t->g_arr[4*t->sN*t->sN*stat2],t->sL*t->sL*4*sizeof(complex double));
      }
  }
  return NULL;
}

/* cid: diffuse_cluster: ordinal cluster id, 0...M, not the cluster id in the cluster file 
 *
 * Z: 2N x 2G shapelet models, for the given freq0
 */
int
recalculate_diffuse_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int cid, int sh_n0, double sh_beta, complex double *Z, int Nt, int use_cuda) {


      /* thread setup - divide baselines */
      int Nthb0,Nthb;
      pthread_attr_t attr;
      pthread_t *th_array;
      thread_data_shap_t *threaddata;
      Nthb0=(Nbase+Nt-1)/Nt;
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

#ifdef HAVE_CUDA
      taskhist thst;
      init_task_hist(&thst);
#endif /* HAVE_CUDA */

      if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
       fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      if ((threaddata=(thread_data_shap_t*)malloc((size_t)Nt*sizeof(thread_data_shap_t)))==0) {
       fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      int tci=0;
      int nth;
      for (nth=0; nth<Nt && tci<Nbase; nth++) {
        if (tci+Nthb0<Nbase) {
          Nthb=Nthb0;
        } else {
          Nthb=Nbase-tci;
        }
        threaddata[nth].boff=tci;
        threaddata[nth].Nb=Nthb;
        threaddata[nth].barr=barr;
        threaddata[nth].u=&u[tci];
        threaddata[nth].v=&v[tci];
        threaddata[nth].w=&w[tci];
        threaddata[nth].carr=carr;
        threaddata[nth].M=M;
        threaddata[nth].N=N;
        threaddata[nth].cid=cid;
        threaddata[nth].coh=&(x[4*M*tci]);
        threaddata[nth].freq0=freq0;
        threaddata[nth].fdelta=fdelta;
        tci=tci+Nthb;
      }

      /* thread setup - divide stations */
      int Nthst;
      int Nthst0=(N+Nt-1)/Nt;
      thread_data_stat_t *threaddata_stat;
      if ((threaddata_stat=(thread_data_stat_t*)malloc((size_t)Nt*sizeof(thread_data_stat_t)))==0) {
       fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      tci=0;
      int nth_s;
      for (nth_s=0; nth_s<Nt && tci<N; nth_s++) {
        if (tci+Nthst0<N) {
          Nthst=Nthst0;
        } else {
          Nthst=N-tci;
        }
        /* station range stat=off+0 ... < off+Nstat */
        threaddata_stat[nth_s].off=tci;
        threaddata_stat[nth_s].Nstat=Nthst;
        threaddata_stat[nth_s].N=N;
        
         
        tci=tci+Nthst;
      }


      /* spatial model modes */
      int G=sh_n0*sh_n0;
      /* Transpose storage of Z: 2Nx2G to 4GxN, so each column has modes of one station */
      complex double *Zt=0;
      if ((Zt=(complex double*)calloc((size_t)(2*N*2*G),sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
      }
      /* copy rows of Z as columns of Zt */
      for (int ci=0; ci<N; ci++) {
        my_ccopy(2*G,&Z[ci*2],2*N,&Zt[ci*4*G],2);
        my_ccopy(2*G,&Z[ci*2+1],2*N,&Zt[ci*4*G+1],2);
      }

      /* storage for product tensor */
      double *Cf;

      if (cid<0 || cid>=M) {
        fprintf(stderr,"%s: %d: invalid cluster id\n",__FILE__,__LINE__);
        exit(1);
      }
      /* find info about the given cluster */
      for (int ci=0; ci<carr[cid].N; ci++) {
        if (carr[cid].stype[ci]!=STYPE_SHAPELET) {
           fprintf(stderr,"%s: %d: invalid source type, must be shapelet\n",__FILE__,__LINE__);
           exit(1);
        }

        /* get shapelet info */
        exinfo_shapelet *sp=(exinfo_shapelet*) carr[cid].ex[ci];

        /* create tensor : product out (sp->n0,sp->beta),
         * product in (sp->n0,sp->beta) (sh_n0,sh_beta) */
        if ((Cf=(double*)calloc((size_t)(sp->n0*sp->n0*sh_n0),sizeof(double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        shapelet_product_tensor(sp->n0,sp->n0,sh_n0,sp->beta,sp->beta,sh_beta,Cf);

        /* allocate memory to store the product C J_q^H, 
         * n0*n0*2x2  for each station */
        complex double *C_Jq=0;
        if ((C_Jq=(complex double*)calloc((size_t)(2*N*2*sp->n0*sp->n0),sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }

      /* for each source, form N x N shapelet products, S_p x S_k x S_q^H where
       * S_p:spatial model for station p,
       * S_k:shapelet model for source k,
       * S_q^H:spatial model for station q, Hermitian
       * Since S_k is not always a scalar value, it can be like 
       * [I 0; 0 I] but [Q 0; 0 -Q] or [0 U; U 0] etc.,
       * We have to honor the order S_p x x S_k x S_q^H
       * S_k : form the visibilities from coherencies here
       */

        /* Shapelet model: sp->modes array of scalars, find coherencies combining Stokes IQUV */
        complex double *s_coh=0;
        if ((s_coh=(complex double*)calloc((size_t)(2*2*sp->n0*sp->n0),sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        complex double XXYY[4]={
           carr[cid].sI[ci]+carr[cid].sQ[ci],
           carr[cid].sU[ci]+_Complex_I*carr[cid].sV[ci],
           carr[cid].sU[ci]-_Complex_I*carr[cid].sV[ci],
           carr[cid].sI[ci]-carr[cid].sQ[ci]
        };

        for (int nmode=0; nmode<sp->n0*sp->n0; nmode++) {
          s_coh[4*nmode]=XXYY[0]*sp->modes[nmode]; /* XX */
          s_coh[4*nmode+1]=XXYY[1]*sp->modes[nmode]; /* XY */
          s_coh[4*nmode+2]=XXYY[2]*sp->modes[nmode]; /* YX */
          s_coh[4*nmode+3]=XXYY[3]*sp->modes[nmode]; /* YY */
        }


        /* C_Jq = C J_q^H for each station */
        /*for (int stat=0; stat<N; stat++) {
          shapelet_product_jones(sp->n0,sp->n0,sh_n0,sp->beta,sp->beta,sh_beta,&C_Jq[4*sp->n0*sp->n0*stat],s_coh,&Zt[4*G*stat],Cf,1);
        } */
        for (int nth1=0; nth1<nth_s; nth1++) {
          threaddata_stat[nth1].sL=sp->n0;
          threaddata_stat[nth1].sM=sp->n0;
          threaddata_stat[nth1].sN=sh_n0;
          threaddata_stat[nth1].alpha=sp->beta;
          threaddata_stat[nth1].beta=sp->beta;
          threaddata_stat[nth1].gamma=sh_beta;
        
          threaddata_stat[nth1].h_arr=C_Jq;
          threaddata_stat[nth1].f_arr=s_coh;
          threaddata_stat[nth1].g_arr=Zt;
          threaddata_stat[nth1].Cf=Cf;
          threaddata_stat[nth1].hermitian=1;

          // shapelet_product_jones(sp->n0,sp->n0,sh_n0,sp->beta,sp->beta,sh_beta,&C_Jq[4*sp->n0*sp->n0*stat],s_coh,&Zt[4*sh_n0*sh_n0*stat],Cf,1);
          pthread_create(&th_array[nth1],&attr,shapelet_prod_one_threadfn,(void*)(&threaddata_stat[nth1]));
        }
        for (int nth1=0; nth1<nth_s; nth1++) {
          pthread_join(th_array[nth1],NULL);
        }

        free(Cf);
        free(s_coh);


        /* create tensor : product out (sp->n0,sp->beta),
         * product in (sh_n0,sh_beta) (sp->n0,sp->beta) */
        if ((Cf=(double*)calloc((size_t)(sp->n0*sp->n0*sp->n0),sizeof(double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        shapelet_product_tensor(sp->n0,sh_n0,sp->n0,sp->beta,sh_beta,sp->beta,Cf);

        complex double *Jp_C_Jq=0;
        if ((Jp_C_Jq=(complex double*)calloc((size_t)(2*N*N*2*sp->n0*sp->n0),sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }



        /* C' = J_1 (C J_2^H) = J_1 Z_2 */
        /*for (int stat1=0; stat1<N; stat1++) {
         // use the fact C_pq = C_qp^H
         for (int stat2=stat1; stat2<N; stat2++) {
          shapelet_product_jones(sp->n0,sh_n0,sp->n0,sp->beta,sh_beta,sp->beta,&Jp_C_Jq[4*sp->n0*sp->n0*(stat1*N+stat2)],&Zt[4*G*stat1],&C_Jq[4*sp->n0*sp->n0*stat2],Cf,0);
         }
        } */

        for (int nth1=0; nth1<nth_s; nth1++) {
          /* station range, stat1=off+0 ... < off+Nstat, stat2=stat1 ... N */
          threaddata_stat[nth1].sL=sp->n0;
          threaddata_stat[nth1].sM=sh_n0;
          threaddata_stat[nth1].sN=sp->n0;
          threaddata_stat[nth1].alpha=sp->beta;
          threaddata_stat[nth1].beta=sh_beta;
          threaddata_stat[nth1].gamma=sp->beta;

          threaddata_stat[nth1].h_arr=Jp_C_Jq;
          threaddata_stat[nth1].f_arr=Zt;
          threaddata_stat[nth1].g_arr=C_Jq;
          threaddata_stat[nth1].Cf=Cf;
          threaddata_stat[nth1].hermitian=0;
 
           // shapelet_product_jones(sp->n0,sh_n0,sp->n0,sp->beta,sh_beta,sp->beta,&Jp_C_Jq[4*sp->n0*sp->n0*(stat1*N+stat2)],&Zt[4*sh_n0*sh_n0*stat1],&C_Jq[4*sp->n0*sp->n0*stat2],Cf,0);

          pthread_create(&th_array[nth1],&attr,shapelet_prod_two_threadfn,(void*)(&threaddata_stat[nth1]));
        }
        for (int nth1=0; nth1<nth_s; nth1++) {
          pthread_join(th_array[nth1],NULL);
        }

        free(Cf);
        free(C_Jq);

        /* predict visibilities - use threads only using CPU prediction
         * otherwise, do sequential prediction */
#ifndef HAVE_CUDA
        for (int nth1=0; nth1<nth; nth1++) {
          /* set the source id */
          threaddata[nth1].sid=ci;
          threaddata[nth1].modes=Jp_C_Jq;
          threaddata[nth1].modes_n0=sp->n0;
          threaddata[nth1].modes_beta=sp->beta;
          pthread_create(&th_array[nth1],&attr,shapelet_pred_threadfn,(void*)(&threaddata[nth1]));
        }
        for (int nth1=0; nth1<nth; nth1++) {
          pthread_join(th_array[nth1],NULL);
        }
#else /* HAVE_CUDA */
        use_cuda=0;
        if (!use_cuda) {
         for (int nth1=0; nth1<nth; nth1++) {
          /* set the source id */
          threaddata[nth1].sid=ci;
          threaddata[nth1].modes=Jp_C_Jq;
          threaddata[nth1].modes_n0=sp->n0;
          threaddata[nth1].modes_beta=sp->beta;
          pthread_create(&th_array[nth1],&attr,shapelet_pred_threadfn,(void*)(&threaddata[nth1]));
         }
         for (int nth1=0; nth1<nth; nth1++) {
          pthread_join(th_array[nth1],NULL);
         }
        } else {
          for (int nth1=0; nth1<nth; nth1++) {
            threaddata[nth1].sid=ci;
            threaddata[nth1].modes=Jp_C_Jq;
            threaddata[nth1].modes_n0=sp->n0;
            threaddata[nth1].modes_beta=sp->beta;
            threaddata[nth1].hst=&thst;
            threaddata[nth1].tid=nth1;
            shapelet_pred_threadfn_cuda((void*)&threaddata[nth1]);
          }
        }
#endif  /* HAVE_CUDA */

        free(Jp_C_Jq);
      }


#ifdef HAVE_CUDA
      destroy_task_hist(&thst);
#endif

      pthread_attr_destroy(&attr);
      free(th_array);
      free(threaddata);
      free(threaddata_stat);
      free(Zt);
  return 0;
}
