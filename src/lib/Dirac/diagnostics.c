/*
 *
 Copyright (C) 2014 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>


static void
checkCudaError(cudaError_t err, const char *file, int line)
{
#ifdef CUDA_DEBUG
    if(!err)
        return;
    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
#endif
}

static void
checkCublasError(cublasStatus_t cbstatus, char *file, int line)
{
#ifdef CUDA_DEBUG
   if (cbstatus!=CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"%s: %d: CUBLAS failure\n",file,line);
    exit(EXIT_FAILURE);
   }
#endif
}


/* find for one cluster J (J^T W J+ eW)^-1 J^T  and extract diagonal as output
  p: parameters M x 1
  rd: residual vector N x 1 (on the device, invarient)
  x: (output) diagonal of leverage matrix 
 
  cbhandle,gWORK: BLAS/storage  pointers

  tileoff: need for hybrid parameters

  adata: has all additional info: coherency,baselines,flags
*/
static int
calculate_leverage(float *p, float *rd, float *x, int M, int N, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle, float *gWORK, int tileoff, int ntiles, me_data_t *dp) {

 /* p needs to be copied to device and x needs to be copied back from device
  rd always remains in the device (select part with the right offset) 
  N will change in hybrid mode, so copy back to x with right offset */

 int Nbase=(dp->Nbase)*(ntiles); /* note: we do not use the total tile size */
 float *jacd,*xd,*jacTjacd,*pd,*cohd,*Ud,*VTd,*Sd;
 unsigned long int moff=0;
 short *bbd;

 cudaError_t err;

 /* total storage N+M*N+M*M+M+Nbase*8+M*M+M*M+M+M+Nbase*3(short)/(float) */ 
 xd=&gWORK[moff];
 moff+=N;
 jacd=&gWORK[moff];
 moff+=M*N;
 jacTjacd=&gWORK[moff];
 moff+=M*M;
 pd=&gWORK[moff];
 moff+=M;
 cohd=&gWORK[moff];
 moff+=Nbase*8;
 Ud=&gWORK[moff];
 moff+=M*M;
 VTd=&gWORK[moff];
 moff+=M*M;
 Sd=&gWORK[moff];
 moff+=M;

 bbd=(short*)&gWORK[moff];
 moff+=(Nbase*3*sizeof(short))/sizeof(float);

 err=cudaMemcpyAsync(pd, p, M*sizeof(float), cudaMemcpyHostToDevice,0);
 checkCudaError(err,__FILE__,__LINE__);
 /* need to give right offset for coherencies */
 /* offset: cluster offset+time offset */
 err=cudaMemcpyAsync(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice,0);
 checkCudaError(err,__FILE__,__LINE__);
 /* correct offset for baselines */
 err=cudaMemcpyAsync(bbd, &(dp->ddbase[3*(dp->Nbase)*(tileoff)]), Nbase*3*sizeof(short), cudaMemcpyHostToDevice,0);
 checkCudaError(err,__FILE__,__LINE__);
 cudaDeviceSynchronize();

 int ThreadsPerBlock=DEFAULT_TH_PER_BK;
 int ci,Mi;

 /* extra storage for cusolver */
 int work_size=0;
 int *devInfo;
 err=cudaMalloc((void**)&devInfo, sizeof(int));
 checkCudaError(err,__FILE__,__LINE__);
 float *work;
 float *rwork;
 cusolverDnSgesvd_bufferSize(solver_handle, M, M, &work_size);
 err=cudaMalloc((void**)&work, work_size*sizeof(float));
 checkCudaError(err,__FILE__,__LINE__);
 err=cudaMalloc((void**)&rwork, 5*M*sizeof(float));
 checkCudaError(err,__FILE__,__LINE__);


 /* set mem to 0 */
 cudaMemset(xd, 0, N*sizeof(float));

 /* calculate J^T, not taking flags into account */
 cudakernel_jacf_fl2(pd, jacd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
 
 /* calculate JTJ=(J^T J - [e] [W]) */
 //status=culaDeviceSgemm('N','T',M,M,N,1.0f,jacd,M,jacd,M,0.0f,jacTjacd,M);
 //checkStatus(status,__FILE__,__LINE__);
 cublasStatus_t cbstatus=CUBLAS_STATUS_SUCCESS;
 float cone=1.0f; float czero=0.0f;
 cbstatus=cublasSgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_T,M,M,N,&cone,jacd,M,jacd,M,&czero,jacTjacd,M);


 /* add mu * I to JTJ */
 cudakernel_diagmu_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, jacTjacd, 1e-9f);
 
 /* calculate inv(JTJ) using SVD */
 /* inv(JTJ) = Ud x Sid x VTd : we take into account that JTJ is symmetric */
 //status=culaDeviceSgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M);
 //checkStatus(status,__FILE__,__LINE__);
 cusolverDnSgesvd(solver_handle,'A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,work,work_size,rwork,devInfo);
 cudaDeviceSynchronize();


 /* find Sd= 1/sqrt(Sd) of the singular values (positive singular values) */
 cudakernel_sqrtdiv_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, 1e-9f, Sd);

 /* multiply Ud with Sid (diagonal) Ud <= Ud Sid (columns modified) */
 cudakernel_diagmult_fl(ThreadsPerBlock, (M*M+ThreadsPerBlock-1)/ThreadsPerBlock, M, Ud, Sd);
 /* now multiply Ud VTd to get the square root */
 //status=culaDeviceSgemm('N','N',M,M,M,1.0f,Ud,M,VTd,M,0.0f,jacTjacd,M);
 //checkStatus(status,__FILE__,__LINE__);
 cbstatus=cublasSgemm(cbhandle,CUBLAS_OP_N,CUBLAS_OP_N,M,M,M,&cone,Ud,M,VTd,M,&czero,jacTjacd,M);

 /* calculate J^T, without taking flags into account (use same storage as previous J^T) */
 cudakernel_jacf_fl2(pd, jacd, M, N, cohd, bbd, Nbase, dp->M, dp->N);

 /* multiply (J^T)^T sqrt(B)  == sqrt(B)^T J^T, taking M columns at a time */
 for (ci=0; ci<(N+M-1)/M;ci++) {
  if (ci*M+M<N) {
   Mi=M;
  } else {
   Mi=N-ci*M;
  }
  //status=culaDeviceSgemm('T','N',M,Mi,M,1.0f,jacTjacd,M,&jacd[ci*M*M],M,0.0f,VTd,M);
  //checkStatus(status,__FILE__,__LINE__);
  cbstatus=cublasSgemm(cbhandle,CUBLAS_OP_T,CUBLAS_OP_N,M,Mi,M,&cone,jacTjacd,M,&jacd[ci*M*M],M,&czero,VTd,M);

  err=cudaMemcpy(&jacd[ci*M*M],VTd,Mi*M*sizeof(float),cudaMemcpyDeviceToDevice);
  checkCudaError(err,__FILE__,__LINE__);
 }

 /* xd[i] <= ||J[i,:]||^2 */
 cudakernel_jnorm_fl(ThreadsPerBlock, (N+ThreadsPerBlock-1)/ThreadsPerBlock, jacd, N, M, xd);

 /* output x <=xd */
 err=cudaMemcpyAsync(x, xd, N*sizeof(float), cudaMemcpyDeviceToHost,0);
 cudaDeviceSynchronize();
 checkCudaError(err,__FILE__,__LINE__);
 checkCublasError(cbstatus,__FILE__,__LINE__);

 return 0;
}

/******************** pipeline functions **************************/
typedef struct gb_data_dg_ {
  int status[2]; 
  float *p[2]; /* pointer to parameters being used by each thread (depends on cluster) */
  float *xo; /* residual vector (copied to device) */
  float *x[2]; /* output leverage values from each thread */
  int M[2]; /* no. of parameters (per cluster,hybrid) */
  int N[2]; /* no. of visibilities (might change in hybrid mode) */
  me_data_t *lmdata[2]; /* two for each thread */

  /* GPU related info */
  cublasHandle_t cbhandle[2]; /* CUBLAS handles */
  cusolverDnHandle_t solver_handle[2]; 
  float *rd[2]; /* residual vector on the device (invarient) */
  float *gWORK[2]; /* GPU buffers */
  int64_t data_size; /* size of buffer (bytes) */

} gbdatadg;


/* slave thread 2GPU function */
static void *
pipeline_slave_code_dg(void *data)
{
 slave_tdata *td=(slave_tdata*)data;
 gbdatadg *gd=(gbdatadg*)(td->pline->data);
 int tid=td->tid;

 while(1) {
  sync_barrier(&(td->pline->gate1)); /* stop at gate 1*/
  if(td->pline->terminate) break; /* if flag is set, break loop */
  sync_barrier(&(td->pline->gate2)); /* stop at gate 2 */
 /* do work */
  if (gd->status[tid]==PT_DO_CDERIV) {
    me_data_t *t=(me_data_t *)gd->lmdata[tid];
    /* divide the tiles into chunks tilesz/nchunk */
    int tilechunk=(t->tilesz+t->carr[t->clus].nchunk-1)/t->carr[t->clus].nchunk;

    int ci;
    int cj=0;
    int ntiles;

    /* loop over chunk, righ set of parameters and residual vector */
    for (ci=0; ci<t->carr[t->clus].nchunk; ci++) {
     /* divide the tiles into chunks tilesz/nchunk */
     if (cj+tilechunk<t->tilesz) {
      ntiles=tilechunk;
     } else {
      ntiles=t->tilesz-cj;
     }

    /* right offset for rd[] and x[] needed and since no overlap,
       can wait for all chunks to complete  */
    calculate_leverage(&gd->p[tid][ci*(gd->M[tid])],&gd->rd[tid][8*cj*t->Nbase],&gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], cj, ntiles, gd->lmdata[tid]);

    cj=cj+tilechunk;
   }

  } else if (gd->status[tid]==PT_DO_AGPU) {
    attach_gpu_to_thread2(tid,&gd->cbhandle[tid],&gd->solver_handle[tid],&gd->gWORK[tid],gd->data_size,1);
    
    /* copy residual vector to device */
    cudaError_t err;
    me_data_t *t=(me_data_t *)gd->lmdata[tid];
    err=cudaMalloc((void**)&gd->rd[tid], (size_t)8*t->tilesz*t->Nbase*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);

    err=cudaMemcpy(gd->rd[tid], gd->xo, 8*t->tilesz*t->Nbase*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);
  } else if (gd->status[tid]==PT_DO_DGPU) {
    cudaFree(gd->rd[tid]);
    detach_gpu_from_thread2(gd->cbhandle[tid],gd->solver_handle[tid],gd->gWORK[tid],1);
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
init_pipeline_dg(th_pipeline *pline,
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
 pthread_create(&(pline->slave0),&(pline->attr),pipeline_slave_code_dg,(void*)t0);
 pthread_create(&(pline->slave1),&(pline->attr),pipeline_slave_code_dg,(void*)t1);
}



/* destroy the pipeline */
/* need to kill the slaves first */
static void
destroy_pipeline_dg(th_pipeline *pline)
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
/******************** end pipeline functions **************************/



/*  Calculate St.Laurent-Cook Jacobian leverage
  xo: residual  (modified)
  flags: 2 for flags based on uvcut, 1 for normal flags
  coh: coherencies are calculated for all baselines, regardless of flag
  diagmode: 1: replace residual, 2: calc noise/leverage ratio
 */
int
calculate_diagnostics(double *u,double *v,double *w,double *p,double *xo,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, complex double *coh, int M,int Mt,int diagmode, int Nt) {


  int cj;
  int n;
  me_data_t lmdata0,lmdata1;
  int Nbase1;

  /* no of data */
  n=Nbase*tilesz*8;

  /* true no of baselines */
  Nbase1=Nbase*tilesz;

  double *ddcoh;
  short *ddbase;

  int c0,c1;

  float *ddcohf, *pf, *xdummy0f, *xdummy1f, *res0, *dgf;
/********* thread data ******************/
  /* barrier */
  th_pipeline tp;
  gbdatadg tpg;
/****************************************/

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
  lmdata0.freq0=lmdata1.freq0=NULL; /* not used */
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
  if ((ddbase=(short*)calloc((size_t)(Nbase1*3),sizeof(short)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  rearrange_coherencies2(Nbase1, barr, coh, ddcoh, ddbase, M, Nt);
  lmdata0.ddcoh=lmdata1.ddcoh=ddcoh;
  lmdata0.ddbase=lmdata1.ddbase=ddbase;
  /* ddcohf (float) << ddcoh (double) */
  double_to_float(ddcohf,ddcoh,M*Nbase1*8,Nt);
  lmdata0.ddcohf=lmdata1.ddcohf=ddcohf;

  if ((pf=(float*)calloc((size_t)(Mt*8*N),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  double_to_float(pf,p,Mt*8*N,Nt);
  /* residual */
  if ((res0=(float*)calloc((size_t)(n),sizeof(float)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  double_to_float(res0,xo,n,Nt);

  /* sum of diagonal values of leverage */
  if ((dgf=(float*)calloc((size_t)(n),sizeof(float)))==0) {
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
/********** setup threads *******************************/
  /* also calculate the total storage needed to be allocated on a GPU */
   /* determine total size for memory allocation 
     residual = n (separately allocated)
     diagonal = n
    For one cluster,
     Jacobian = nxm,  J^T J = mxm, (also inverse)
   */
   int Mm=8*N; /* no of parameters */
   int64_t data_sz=0;
   data_sz=(int64_t)(n+Mm*n+3*Mm*Mm+3*Mm+Nbase1*8)*sizeof(float)+(int64_t)Nbase1*3*sizeof(short);
  tpg.data_size=data_sz;
  tpg.lmdata[0]=&lmdata0;
  tpg.lmdata[1]=&lmdata1;
  tpg.xo=res0; /* residual */

  init_pipeline_dg(&tp,&tpg);
  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/

  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/

/********** done setup threads *******************************/
     tpg.x[0]=xdummy0f;
     tpg.M[0]=8*N; /* even though size of p is > M, dont change this */
     tpg.N[0]=n; /* Nbase*tilesz*8 */
     tpg.x[1]=xdummy1f;
     tpg.M[1]=8*N; /* even though size of p is > M, dont change this */
     tpg.N[1]=n; /* Nbase*tilesz*8 */

    for (cj=0; cj<M/2; cj++) { /* iter per cluster pairs */
      c0=2*cj;
      c1=2*cj+1;
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     lmdata0.clus=c0;
     lmdata1.clus=c1;

     /* run this from a separate thread */
     tpg.p[0]=&pf[carr[c0].p[0]]; /* length carr[c0].nchunk times */
     tpg.p[1]=&pf[carr[c1].p[0]]; /* length carr[c1].nchunk times */
     tpg.status[0]=tpg.status[1]=PT_DO_CDERIV;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
    /* add result to the sum */
    my_saxpy(n, xdummy0f, 1.0f, dgf);
    my_saxpy(n, xdummy1f, 1.0f, dgf);
   }
   /* odd cluster out, if M is odd */
   if (M%2) {
      c0=M-1;
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     tpg.p[0]=&pf[carr[c0].p[0]];
     lmdata0.clus=c0;

     tpg.status[0]=PT_DO_CDERIV;
     tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
/**************************************************************************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1 */
     tpg.status[0]=tpg.status[1]=PT_DO_NOTHING;
  sync_barrier(&(tp.gate2)); /* sync at gate 2 */
    my_saxpy(n, xdummy0f, 1.0f, dgf);
  }
  free(pf);
  free(ddcohf);
  free(xdummy1f);
  free(res0);
  free(ddcoh);

  /******** free threads ***************/
  sync_barrier(&(tp.gate1)); /* sync at gate 1*/
  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
  sync_barrier(&(tp.gate2)); /* sync at gate 2*/
  destroy_pipeline_dg(&tp);
  /******** done free threads ***************/

  /* now add 1's to locations with flagged data */
  /* create array for adding */
  create_onezerovec(Nbase1, ddbase, xdummy0f, Nt);
  my_saxpy(n, xdummy0f, 1.0f, dgf);
  free(xdummy0f);
  free(ddbase);
  /* output */
//  for (cj=0; cj<n; cj++) {
//   printf("%d %f\n",cj,dgf[cj]);
//  }
  if (diagmode==1) {
  /* copy back to output */
  float_to_double(xo,dgf,n,Nt);
  } else { 
    /* solve system of  equations a * leverage + b * 1 = |residual|
      to find a,b scalars, and just print them as output */
     /* find  1^T |r| = sum (|residual|) and  lev^T |r|  */
     float sum1,sum2;
     find_sumproduct(n, res0, dgf, &sum1, &sum2, Nt);
     //printf("sum|res|=%f sum(lev^T |res|)=%f\n",sum1,sum2);
     float a00,a01,a11;
     a00=my_fnrm2(n,dgf); /* lev^T lev */
     a01=my_fasum(n,dgf); /* = a10 = sum|leverage| */
     a00=a00*a00;
     a11=(float)n; /* sum( 1 ) */
     float r00,r01;
     r00=sum1;
     r01=sum2;
     //printf("A=[\n %f %f;\n %f %f];\n b=[\n %f\n %f\n]\n",a00,a01,a01,a11,r00,r01);
     /* solve A [a b]^T = r */
     float alpha,beta,denom;
     denom=(a00*a11-a01*a01);
     //printf("denom=%f\n",denom);
     if (denom>1e-6f) { /* can be solved */
      alpha=(r00*a11-r01*a01)/denom;
     } else {
      alpha=0.0f;
     }
     beta=(r00-a00*alpha)/a01; 
     printf("Error Noise/Model %e/%e\n",beta,alpha);
  }
  free(dgf);
 return 0;
}
