/*
 *
 Copyright (C) 2006-2015 Sarod Yatawatta <sarod@users.sf.net>  
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
static void
ambt(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=a[0]*conj(b[0])+a[1]*conj(b[1]);
 c[1]=a[0]*conj(b[2])+a[1]*conj(b[3]);
 c[2]=a[2]*conj(b[0])+a[3]*conj(b[1]);
 c[3]=a[2]*conj(b[2])+a[3]*conj(b[3]);
}

/********************** sage minimization ***************************/
/* worker thread function for prediction */
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
/* p: size ??x1 parameters, not all belong to 
    this cluster
   x: size nx1 data calculated
   data: extra info needed */
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





/******************** CUP version *****************************/


int
sagefit_visibilities_admm(double *u, double *v, double *w, double *x, int N,   
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *admm_rho, double *mean_nu, double *res_0, double *res_1) {

  int  ci,cj,ck,tcj;
  double *p; // parameters: m x 1
  int m, n;
  double info[CLM_INFO_SZ];
  me_data_t lmdata;

  double *xdummy,*xsub;
  double *nerr; /* array to store cost reduction per cluster */
  double *robust_nuM;
  int weighted_iter,this_itermax,total_iter;
  double total_err;

  int ntiles,tilechunk;
  double init_res,final_res;

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
  if ((robust_nuM=(double*)calloc((size_t)(M),sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
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


       /* Only RSD+RTR */
       if (!ci){
         lmdata.robust_nu=robust_nu0;
       } 
        /* use a reasonable TR radius because cost function has extra 
       regularization NB: ADMM very sensitive to this */
       double Delta0=2.0; 
       rtr_solve_nocuda_robust_admm(&p[carr[cj].p[ck]], &Y[carr[cj].p[ck]], &BZ[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+5, this_itermax+10, Delta0, Delta0*0.125, admm_rho[cj], nulow, nuhigh, info, &lmdata);
       if (ci==max_emiter-1){
            robust_nuM[cj]+=lmdata.robust_nu;
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
     robust_nuM[cj]/=(double)carr[cj].nchunk;
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




#ifdef HAVE_CUDA
/****************************************************************************/
/* slave thread 2GPU function */
static void *
pipeline_slave_code_admm_flt(void *data)
{
 slave_tdata *td=(slave_tdata*)data;
 gbdatafl_admm *gd=(gbdatafl_admm*)(td->pline->data);
 int tid=td->tid;

 while(1) {
  sync_barrier(&(td->pline->gate1)); /* stop at gate 1*/
  if(td->pline->terminate) break; /* if flag is set, break loop */
  sync_barrier(&(td->pline->gate2)); /* stop at gate 2 */
  /* do work : only one solver */
  //printf("state=%d, thread %d\n",gd->status[tid],tid);
  if (gd->status[tid]==PT_DO_WORK_RRTR || gd->status[tid]==PT_DO_WORK_NSD) {
/************************* work *********************/
  me_data_t *t=(me_data_t *)gd->lmdata[tid];
  /* divide the tiles into chunks tilesz/nchunk */
  int tilechunk=(t->tilesz+t->carr[t->clus].nchunk-1)/t->carr[t->clus].nchunk;


  int ci;

  int cj=0; 
  int ntiles;
  double init_res,final_res;
  init_res=final_res=0.0;
  if (tid<2) {
   /* for GPU, the cost func and jacobian are not used */
   /* loop over each chunk, with right parameter set and data set */
   for (ci=0; ci<t->carr[t->clus].nchunk; ci++) {
     /* divide the tiles into chunks tilesz/nchunk */
     if (cj+tilechunk<t->tilesz) {
      ntiles=tilechunk;
     } else {
      ntiles=t->tilesz-cj;
     }

      if (gd->status[tid]==PT_DO_WORK_NSD) {
       nsd_solve_cuda_robust_admm_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->Y[tid][ci*(gd->M[tid])], &gd->Z[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid]/8, ntiles*t->Nbase, gd->itermax[tid]+15, gd->admm_rho[tid], gd->nulow, gd->nuhigh, gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], cj, ntiles, (void*)gd->lmdata[tid]);
      } else {
       /* max trust region radius: keep reasonable */
       float Delta0=2.0f;
       rtr_solve_cuda_robust_admm_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->Y[tid][ci*(gd->M[tid])], &gd->Z[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid]/8, ntiles*t->Nbase, gd->itermax[tid]+10, Delta0, Delta0*0.125f, gd->admm_rho[tid], gd->nulow, gd->nuhigh, gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], cj, ntiles, (void*)gd->lmdata[tid]);
      }

     init_res+=gd->info[tid][0];
     final_res+=gd->info[tid][1];
     cj=cj+tilechunk;
   }

  } 

  gd->info[tid][0]=init_res;
  gd->info[tid][1]=final_res;
 
/************************* work *********************/
  } else if (gd->status[tid]==PT_DO_AGPU) {
   /* no cula needed: 0 at end */
   attach_gpu_to_thread2(select_work_gpu(MAX_GPU_ID,td->pline->thst),&gd->cbhandle[tid],&gd->solver_handle[tid],&gd->gWORK[tid],gd->data_size,0);
  } else if (gd->status[tid]==PT_DO_DGPU) {
   detach_gpu_from_thread2(gd->cbhandle[tid],gd->solver_handle[tid],gd->gWORK[tid],0);
  } else if (gd->status[tid]==PT_DO_MEMRESET) {
   reset_gpu_memory((double*)gd->gWORK[tid],gd->data_size);
  } else if (gd->status[tid]!=PT_DO_NOTHING) { /* catch error */
    fprintf(stderr,"%s: %d: invalid mode for slave tid=%d status=%d\n",__FILE__,__LINE__,tid,gd->status[tid]);
    exit(1);
  }
 }
 return NULL;
}

/* initialize the pipeline
  and start the slaves rolling */
static void
init_pipeline_admm_flt(th_pipeline *pline,
     void *data)
{
 slave_tdata *t0,*t1;
 pthread_attr_init(&(pline->attr));
 pthread_attr_setdetachstate(&(pline->attr),PTHREAD_CREATE_JOINABLE);

 init_th_barrier(&(pline->gate1),3); /* 3 threads, including master */
 init_th_barrier(&(pline->gate2),3); /* 3 threads, including master */
 pline->terminate=0;
 pline->data=data; /* data should have pointers to t1 and t2 */

 if ((t0=(slave_tdata*)malloc(sizeof(slave_tdata)))==0) {
    fprintf(stderr,"no free memory\n");
    exit(1);
 }
 if ((t1=(slave_tdata*)malloc(sizeof(slave_tdata)))==0) {
    fprintf(stderr,"no free memory\n");
    exit(1);
 }
 if ((pline->thst=(taskhist*)malloc(sizeof(taskhist)))==0) {
    fprintf(stderr,"no free memory\n");
    exit(1);
 }

 init_task_hist(pline->thst);
 t0->pline=t1->pline=pline;
 t0->tid=0;
 t1->tid=1; /* link back t1, t2 to data so they could be freed */
 pline->sd0=t0;
 pline->sd1=t1;
 pthread_create(&(pline->slave0),&(pline->attr),pipeline_slave_code_admm_flt,(void*)t0);
 pthread_create(&(pline->slave1),&(pline->attr),pipeline_slave_code_admm_flt,(void*)t1);

}

/* destroy the pipeline */
/* need to kill the slaves first */
static void
destroy_pipeline_admm_flt(th_pipeline *pline)
{

 pline->terminate=1;
 sync_barrier(&(pline->gate1));
 pthread_join(pline->slave0,NULL);
 pthread_join(pline->slave1,NULL);
 destroy_th_barrier(&(pline->gate1));
 destroy_th_barrier(&(pline->gate2));
 pthread_attr_destroy(&(pline->attr));
 destroy_task_hist(pline->thst);
 free(pline->thst);
 free(pline->sd0);
 free(pline->sd1);
 pline->data=NULL;
}

//#define DEBUG
int
sagefit_visibilities_admm_dual_pt_flt(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize,double *admm_rho, double *mean_nu, double *res_0, double *res_1) {


  int  ci,cj;
  double *p; // parameters: m x 1
  int m, n;
  double opts[CLM_OPTS_SZ], info0[CLM_INFO_SZ], info1[CLM_INFO_SZ];
  me_data_t lmdata0,lmdata1;
  int Nbase1;

  double *xdummy0,*xdummy1,*xsub,*xo;
  double *nerr; /* array to store cost reduction per cluster */
  int weighted_iter,this_itermax0,this_itermax1,total_iter;
  double total_err;

  /* rearraged memory for GPU use */
  double *ddcoh;
  short *ddbase;

  int *cr=0; /* array for random permutation of clusters */
  int c0,c1;

  //opts[0]=LM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-20;
  opts[0]=CLM_INIT_MU; opts[1]=1E-9; opts[2]=1E-9; opts[3]=1E-9;
  opts[4]=-CLM_DIFF_DELTA;

  /* robust */
  double robust_nu0;
  double *robust_nuM;

  /*  no. of parameters >= than the no of clusters*8N */
  m=N*Mt*8;
  /* no of data */
  n=Nbase*tilesz*8;

  /* true no of baselines */
  Nbase1=Nbase*tilesz;

  float *ddcohf, *pf, *Yf, *Zf, *xdummy0f, *xdummy1f;
/********* thread data ******************/
  /* barrier */
  th_pipeline tp;
  gbdatafl_admm tpg;
/****************************************/

  /* use full parameter space */
  p=pp;
  lmdata0.clus=lmdata1.clus=-1;
  /* setup data for lmfit */
  lmdata0.u=lmdata1.u=u;
  lmdata0.v=lmdata1.v=v;
  lmdata0.w=lmdata1.w=w;
  lmdata0.Nbase=lmdata1.Nbase=Nbase;
  lmdata0.tilesz=lmdata1.tilesz=tilesz;
  lmdata0.N=lmdata1.N=N;
  lmdata0.barr=lmdata1.barr=barr;
  lmdata0.carr=lmdata1.carr=carr;
  lmdata0.M=lmdata1.M=M;
  lmdata0.Mt=lmdata1.Mt=Mt;
  lmdata0.freq0=lmdata1.freq0=&freq0;
  lmdata0.Nt=lmdata1.Nt=Nt;
  lmdata0.coh=lmdata1.coh=coh;
  /* rearrange coh for GPU use */
  if ((ddcoh=(double*)calloc((size_t)(M*Nbase1*8),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((ddcohf=(float*)calloc((size_t)(M*Nbase1*8),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((ddbase=(short*)calloc((size_t)(Nbase1*2),sizeof(short)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  rearrange_coherencies(Nbase1, barr, coh, ddcoh, ddbase, M, Nt);
  lmdata0.ddcoh=lmdata1.ddcoh=ddcoh;
  lmdata0.ddbase=lmdata1.ddbase=ddbase;

  /* ddcohf (float) << ddcoh (double) */
  double_to_float(ddcohf,ddcoh,M*Nbase1*8,Nt);
  lmdata0.ddcohf=lmdata1.ddcohf=ddcohf;

  if ((xsub=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xo=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy0=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy1=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((nerr=(double*)calloc((size_t)(M),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy0f=(float*)calloc((size_t)(n),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy1f=(float*)calloc((size_t)(n),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((pf=(float*)calloc((size_t)(Mt*8*N),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((Yf=(float*)calloc((size_t)(Mt*8*N),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((Zf=(float*)calloc((size_t)(Mt*8*N),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((robust_nuM=(double*)calloc((size_t)(M),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* starting guess of robust nu */
  robust_nu0=nulow;

  double_to_float(pf,p,Mt*8*N,Nt);
  double_to_float(Yf,Y,Mt*8*N,Nt);
  double_to_float(Zf,BZ,Mt*8*N,Nt);
  /* remember for each partition how much the cost function decreases
     in the next EM iteration, we allocate more LM iters to partitions
     where const function significantly decreases. So two stages
     1) equal LM iters (find the decrease) 2) weighted LM iters */
  weighted_iter=0;
  total_iter=M*max_iter; /* total iterations per EM */
/********** setup threads *******************************/
  init_pipeline_admm_flt(&tp,&tpg);
  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
  /* also calculate the total storage needed to be allocated on a GPU */
   /* determine total size for memory allocation */
   int64_t data_sz=0;
   /* size for RTR/NSD (float), 128 is the ThreadsPerBlock   
      NSD is a bit lower
      Use dummy data size
   */
  if (solver_mode==SM_NSD_RLBFGS) {
   //data_sz=(8*N*(7+(Nbase1+128-1)/128)+N+8*Nbase1*2+3*Nbase1)*sizeof(float);
   data_sz=8*sizeof(float);
  } else { /* default is RTR */
   //data_sz=(8*N*(11+(Nbase1+128-1)/128)+N+8*Nbase1*2+3*Nbase1)*sizeof(float);
   data_sz=8*sizeof(float);
  }

  tpg.data_size=data_sz;
  tpg.nulow=nulow;
  tpg.nuhigh=nuhigh;
  tpg.randomize=randomize;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/

  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/

/********** done setup threads *******************************/

  /* initial residual calculation
       subtract full model from data  */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata0);
  memcpy(xo,x,(size_t)(n)*sizeof(double));
  my_daxpy(n, xsub, -1.0, xo);
  *res_0=my_dnrm2(n,xo)/(double)n;

  int iter_bar=(int)ceil((0.80/(double)M)*((double)total_iter));
  for (ci=0; ci<max_emiter; ci++) {
  /**************** EM iteration ***********************/
    if (randomize && M>1) {
     /* find a random permutation of clusters */
     cr=random_permutation(M,weighted_iter,nerr);
    } else {
     cr=NULL;
    }

    for (cj=0; cj<M/2; cj++) { /* iter per cluster pairs */
     if (randomize) {
      c0=cr[2*cj];
      c1=cr[2*cj+1];
     } else {
      c0=2*cj;
      c1=2*cj+1;
     }
     /* calculate max LM iter for this cluster */
     if (weighted_iter) {
       /* assume permutation gives a sorted pair 
       with almost equal nerr[] values */
       this_itermax0=(int)((0.20*nerr[c0])*((double)total_iter))+iter_bar;
       this_itermax1=(int)((0.20*nerr[c1])*((double)total_iter))+iter_bar;
     } else {
       this_itermax0=this_itermax1=max_iter;
     }
#ifdef DEBUG
     printf("Cluster pair %d(iter=%d,wt=%lf),%d(iter=%d,wt=%lf)\n",c0,this_itermax0,nerr[c0],c1,this_itermax1,nerr[c1]);
#endif
     if (this_itermax0>0 || this_itermax1>0) {
     /* calculate contribution from hidden data, subtract from x */
     /* since x has already subtracted this model, just add
        the ones we are solving for */

  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     memcpy(xdummy0,xo,(size_t)(n)*sizeof(double));
     memcpy(xdummy1,xo,(size_t)(n)*sizeof(double));
     lmdata0.clus=c0;
     lmdata1.clus=c1;

     /* NOTE: conditional mean x^i = s^i + 0.5 * residual^i */
     /* so xdummy=0.5 ( 2*model + residual ) */
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata0);
     my_daxpy(n, xsub, 2.0, xdummy0);
     my_dscal(n, 0.5, xdummy0);
     my_daxpy(n, xsub, 1.0, xo);
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata1);
     my_daxpy(n, xsub, 2.0, xdummy1);
     my_dscal(n, 0.5, xdummy1);
     my_daxpy(n, xsub, 1.0, xo);
/**************************************************************************/
     /* xdummy*f (float) << xdummy* (double) */
     double_to_float(xdummy0f,xdummy0,n,Nt);
     double_to_float(xdummy1f,xdummy1,n,Nt);
     /* run this from a separate thread */
     tpg.p[0]=&pf[carr[c0].p[0]]; /* length carr[c0].nchunk times */
     tpg.Y[0]=&Yf[carr[c0].p[0]]; /* length carr[c0].nchunk times */
     tpg.Z[0]=&Zf[carr[c0].p[0]]; /* length carr[c0].nchunk times */
     tpg.admm_rho[0]=(float)admm_rho[c0];
     tpg.x[0]=xdummy0f;
     tpg.M[0]=8*N; /* even though size of p is > M, dont change this */
     tpg.N[0]=n; /* Nbase*tilesz*8 */
     tpg.itermax[0]=this_itermax0;
     tpg.opts[0]=opts;
     tpg.info[0]=info0;
     tpg.linsolv=linsolv;
     tpg.lmdata[0]=&lmdata0;

     tpg.p[1]=&pf[carr[c1].p[0]]; /* length carr[c1].nchunk times */
     tpg.Y[1]=&Yf[carr[c1].p[0]]; /* length carr[c1].nchunk times */
     tpg.Z[1]=&Zf[carr[c1].p[0]]; /* length carr[c1].nchunk times */
     tpg.admm_rho[1]=(float)admm_rho[c1];
     tpg.x[1]=xdummy1f;
     tpg.M[1]=8*N; /* even though size of p is > M, dont change this */
     tpg.N[1]=n; /* Nbase*tilesz*8 */
     tpg.itermax[1]=this_itermax1;
     tpg.opts[1]=opts;
     tpg.info[1]=info1;
     tpg.linsolv=linsolv;
     tpg.lmdata[1]=&lmdata1;
/**************************************************************************/

     /* both threads do work */
     if (solver_mode==SM_NSD_RLBFGS) {
      tpg.status[0]=tpg.status[1]=PT_DO_WORK_NSD;
     } else {
      tpg.status[0]=tpg.status[1]=PT_DO_WORK_RRTR;
     }
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
#ifdef DEBUG
printf("1: %lf -> %lf 2: %lf -> %lf\n\n\n",info0[0],info0[1],info1[0],info1[1]);
#endif
     /* catch -ve value here */
     if (info0[0]>0.0) {
      nerr[c0]=(info0[0]-info0[1])/info0[0];
      if (nerr[c0]<0.0) { nerr[c0]=0.0; }
     } else {
      nerr[c0]=0.0;
     }
     if (info1[0]>0.0) {
      nerr[c1]=(info1[0]-info1[1])/info1[0];
      if (nerr[c1]<0.0) { nerr[c1]=0.0; }
     } else {
      nerr[c1]=0.0;
     }
     /* update robust_nu */
     robust_nuM[c0]+=lmdata0.robust_nu;
     robust_nuM[c1]+=lmdata1.robust_nu;
     /* p (double) << pf (float) */
     float_to_double(&p[carr[c0].p[0]],&pf[carr[c0].p[0]],carr[c0].nchunk*8*N,Nt);
     float_to_double(&p[carr[c1].p[0]],&pf[carr[c1].p[0]],carr[c1].nchunk*8*N,Nt);
     /* once again subtract solved model from data */
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata0);
     my_daxpy(n, xsub, -1.0, xo);
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata1);
     my_daxpy(n, xsub, -1.0, xo);

    }
   }
   /* odd cluster out, if M is odd */
   if (M%2) {
     if (randomize && M>1) {
      c0=cr[M-1];
     } else {
      c0=M-1;
     }
     /* calculate max LM iter for this cluster */
     if (weighted_iter) {
       this_itermax0=(int)((0.20*nerr[c0])*((double)total_iter))+iter_bar;
     } else {
       this_itermax0=max_iter;
     }
#ifdef DEBUG
    printf("Cluster %d(iter=%d, wt=%lf)\n",c0,this_itermax0,nerr[c0]);
#endif
     if (this_itermax0>0) {
/**************************************************************************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     /* calculate contribution from hidden data, subtract from x */
     memcpy(xdummy0,xo,(size_t)(n)*sizeof(double));
     lmdata0.clus=c0;
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata0);
     my_daxpy(n, xsub, 1.0, xdummy0);
     my_daxpy(n, xsub, 1.0, xo);

     double_to_float(xdummy0f,xdummy0,n,Nt);
     /* run this from a separate thread */
     tpg.p[0]=&pf[carr[c0].p[0]];
     tpg.Y[0]=&Yf[carr[c0].p[0]];
     tpg.Z[0]=&Zf[carr[c0].p[0]];
     tpg.admm_rho[0]=(float)admm_rho[c0];
     tpg.x[0]=xdummy0f;
     tpg.M[0]=8*N;
     tpg.N[0]=n;
     tpg.itermax[0]=this_itermax0;
     tpg.opts[0]=opts;
     tpg.info[0]=info0;
     tpg.linsolv=linsolv;
     tpg.lmdata[0]=&lmdata0;

     if (solver_mode==SM_NSD_RLBFGS) {
      tpg.status[0]=PT_DO_WORK_NSD;
     } else {
      tpg.status[0]=PT_DO_WORK_RRTR;
     }

     tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
/**************************************************************************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */

#ifdef DEBUG
printf("1: %lf -> %lf\n\n\n",info0[0],info0[1]);
#endif
     /* catch -ve value here */
     if (info0[0]>0.0) {
      nerr[c0]=(info0[0]-info0[1])/info0[0];
      if (nerr[c0]<0.0) { nerr[c0]=0.0; }
     } else {
      nerr[c0]=0.0;
     }
     /* update robust_nu */
     robust_nuM[c0]+=lmdata0.robust_nu;
     /* once again subtract solved model from data */
     float_to_double(&p[carr[c0].p[0]],&pf[carr[c0].p[0]],carr[c0].nchunk*8*N,Nt);
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata0);
     my_daxpy(n, xsub, -1.0, xo);
     }
   }
   /* normalize nerr array so that the sum is 1 */
   total_err=my_dasum(M,nerr);
   if (total_err>0.0) {
    my_dscal(M, 1.0/total_err, nerr);
   }
   if (randomize && M>1) { /* nothing to randomize if only 1 direction */
    /* flip weighting flag */
    weighted_iter=!weighted_iter;
    free(cr);
   }
  /**************** End EM iteration ***********************/
 }
  free(nerr);
  free(xo);
  free(xdummy0);
  free(xdummy1);
  free(ddcoh);
  free(ddbase);
  free(xdummy0f);
  free(xdummy1f);
  free(pf);
  free(Yf);
  free(Zf);
  free(ddcohf);
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

  /******** free threads ***************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/


  destroy_pipeline_admm_flt(&tp);
  /******** done free threads ***************/

  /* final residual calculation */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata0);
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


int
sagefit_visibilities_admm_dual_pt_flt_one(double *u, double *v, double *w, double *x, int N, 
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize,double *admm_rho, double *mean_nu, double *res_0, double *res_1) {


  int  ci,cj;
  double *p; // parameters: m x 1
  int m, n;
  double opts[CLM_OPTS_SZ], info0[CLM_INFO_SZ];
  me_data_t lmdata0,lmdata1;
  int Nbase1;

  double *xdummy0,*xdummy1,*xsub,*xo;
  double *nerr; /* array to store cost reduction per cluster */
  int weighted_iter,this_itermax0,total_iter;
  double total_err;

  /* rearraged memory for GPU use */
  double *ddcoh;
  short *ddbase;

  int *cr=0; /* array for random permutation of clusters */
  int c0;

  //opts[0]=LM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-20;
  opts[0]=CLM_INIT_MU; opts[1]=1E-9; opts[2]=1E-9; opts[3]=1E-9;
  opts[4]=-CLM_DIFF_DELTA;

  /* robust */
  double robust_nu0;
  double *robust_nuM;

  /*  no. of parameters >= than the no of clusters*8N */
  m=N*Mt*8;
  /* no of data */
  n=Nbase*tilesz*8;

  /* true no of baselines */
  Nbase1=Nbase*tilesz;

  float *ddcohf, *pf, *Yf, *Zf, *xdummy0f, *xdummy1f;
/********* thread data ******************/
  /* barrier */
  th_pipeline tp;
  gbdatafl_admm tpg;
/****************************************/

  /* use full parameter space */
  p=pp;
  lmdata0.clus=lmdata1.clus=-1;
  /* setup data for lmfit */
  lmdata0.u=lmdata1.u=u;
  lmdata0.v=lmdata1.v=v;
  lmdata0.w=lmdata1.w=w;
  lmdata0.Nbase=lmdata1.Nbase=Nbase;
  lmdata0.tilesz=lmdata1.tilesz=tilesz;
  lmdata0.N=lmdata1.N=N;
  lmdata0.barr=lmdata1.barr=barr;
  lmdata0.carr=lmdata1.carr=carr;
  lmdata0.M=lmdata1.M=M;
  lmdata0.Mt=lmdata1.Mt=Mt;
  lmdata0.freq0=lmdata1.freq0=&freq0;
  lmdata0.Nt=lmdata1.Nt=Nt;
  lmdata0.coh=lmdata1.coh=coh;
  /* rearrange coh for GPU use */
  if ((ddcoh=(double*)calloc((size_t)(M*Nbase1*8),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((ddcohf=(float*)calloc((size_t)(M*Nbase1*8),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((ddbase=(short*)calloc((size_t)(Nbase1*2),sizeof(short)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  rearrange_coherencies(Nbase1, barr, coh, ddcoh, ddbase, M, Nt);
  lmdata0.ddcoh=lmdata1.ddcoh=ddcoh;
  lmdata0.ddbase=lmdata1.ddbase=ddbase;

  /* ddcohf (float) << ddcoh (double) */
  double_to_float(ddcohf,ddcoh,M*Nbase1*8,Nt);
  lmdata0.ddcohf=lmdata1.ddcohf=ddcohf;

  if ((xsub=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xo=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy0=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy1=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((nerr=(double*)calloc((size_t)(M),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy0f=(float*)calloc((size_t)(n),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xdummy1f=(float*)calloc((size_t)(n),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((pf=(float*)calloc((size_t)(Mt*8*N),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((Yf=(float*)calloc((size_t)(Mt*8*N),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((Zf=(float*)calloc((size_t)(Mt*8*N),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((robust_nuM=(double*)calloc((size_t)(M),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* starting guess of robust nu */
  robust_nu0=nulow;

  double_to_float(pf,p,Mt*8*N,Nt);
  double_to_float(Yf,Y,Mt*8*N,Nt);
  double_to_float(Zf,BZ,Mt*8*N,Nt);
  /* remember for each partition how much the cost function decreases
     in the next EM iteration, we allocate more LM iters to partitions
     where const function significantly decreases. So two stages
     1) equal LM iters (find the decrease) 2) weighted LM iters */
  weighted_iter=0;
  total_iter=M*max_iter; /* total iterations per EM */
/********** setup threads *******************************/
  init_pipeline_admm_flt(&tp,&tpg);
  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
  /* also calculate the total storage needed to be allocated on a GPU */
   /* determine total size for memory allocation */
   int64_t data_sz=0;
   /* size for RTR/NSD (float), 128 is the ThreadsPerBlock   
      NSD is a bit lower, but use the same
   */
  //data_sz=(8*N*(11+(Nbase1+128-1)/128)+N+8*Nbase1*2+3*Nbase1)*sizeof(float);
  data_sz=8*sizeof(float);

  tpg.data_size=data_sz;
  tpg.nulow=nulow;
  tpg.nuhigh=nuhigh;
  tpg.randomize=randomize;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/

  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/

/********** done setup threads *******************************/

  /* initial residual calculation
       subtract full model from data  */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata0);
  memcpy(xo,x,(size_t)(n)*sizeof(double));
  my_daxpy(n, xsub, -1.0, xo);
  *res_0=my_dnrm2(n,xo)/(double)n;

  int iter_bar=(int)ceil((0.80/(double)M)*((double)total_iter));
  for (ci=0; ci<max_emiter; ci++) {
  /**************** EM iteration ***********************/
    if (randomize && M>1) {
     /* find a random permutation of clusters */
     cr=random_permutation(M,weighted_iter,nerr);
    } else {
     cr=NULL;
    }

   /* only one cluster at a time */
   for (cj=0; cj<M; cj++) { /* iter per cluster pairs */
     if (randomize && M>1) {
      c0=cr[cj];
     } else {
      c0=cj;
     }
     /* calculate max LM iter for this cluster */
     if (weighted_iter) {
       this_itermax0=(int)((0.20*nerr[c0])*((double)total_iter))+iter_bar;
     } else {
       this_itermax0=max_iter;
     }
#ifdef DEBUG
    printf("Cluster %d(iter=%d, wt=%lf)\n",c0,this_itermax0,nerr[c0]);
#endif
     if (this_itermax0>0) {
/**************************************************************************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     /* calculate contribution from hidden data, subtract from x */
     memcpy(xdummy0,xo,(size_t)(n)*sizeof(double));
     lmdata0.clus=c0;
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata0);
     my_daxpy(n, xsub, 1.0, xdummy0);
     my_daxpy(n, xsub, 1.0, xo);

     double_to_float(xdummy0f,xdummy0,n,Nt);
     /* run this from a separate thread */
     tpg.p[0]=&pf[carr[c0].p[0]];
     tpg.Y[0]=&Yf[carr[c0].p[0]];
     tpg.Z[0]=&Zf[carr[c0].p[0]];
     tpg.admm_rho[0]=(float)admm_rho[c0];
     tpg.x[0]=xdummy0f;
     tpg.M[0]=8*N;
     tpg.N[0]=n;
     tpg.itermax[0]=this_itermax0;
     tpg.opts[0]=opts;
     tpg.info[0]=info0;
     tpg.linsolv=linsolv;
     tpg.lmdata[0]=&lmdata0;

     tpg.status[0]=PT_DO_WORK_RRTR;

     tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
/**************************************************************************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */

#ifdef DEBUG
printf("1: %lf -> %lf\n\n\n",info0[0],info0[1]);
#endif
     /* catch -ve value here */
     if (info0[0]>0.0) {
      nerr[c0]=(info0[0]-info0[1])/info0[0];
      if (nerr[c0]<0.0) { nerr[c0]=0.0; }
     } else {
      nerr[c0]=0.0;
     }
     /* update robust_nu */
     robust_nuM[c0]+=lmdata0.robust_nu;
     /* once again subtract solved model from data */
     float_to_double(&p[carr[c0].p[0]],&pf[carr[c0].p[0]],carr[c0].nchunk*8*N,Nt);
     mylm_fit_single_pth(p, xsub, 8*N, n, (void*)&lmdata0);
     my_daxpy(n, xsub, -1.0, xo);
     }
   }
   /* normalize nerr array so that the sum is 1 */
   total_err=my_dasum(M,nerr);
   if (total_err>0.0) {
    my_dscal(M, 1.0/total_err, nerr);
   }
   if (randomize && M>1) { /* nothing to randomize if only 1 direction */
    /* flip weighting flag */
    weighted_iter=!weighted_iter;
    free(cr);
   }
  /**************** End EM iteration ***********************/
 }
  free(nerr);
  free(xo);
  free(xdummy0);
  free(xdummy1);
  free(ddcoh);
  free(ddbase);
  free(xdummy0f);
  free(xdummy1f);
  free(pf);
  free(Yf);
  free(Zf);
  free(ddcohf);
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


  /******** free threads ***************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/


  destroy_pipeline_admm_flt(&tp);
  /******** done free threads ***************/

  /* final residual calculation */
  minimize_viz_full_pth(p, xsub, m, n, (void*)&lmdata0);
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
#endif /* HAVE_CUDA */
