/*
 *
 Copyright (C) 2006-2016 Sarod Yatawatta <sarod@users.sf.net>  
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



#include "Radio.h"
#include "GPUtune.h"

//#define CUDA_DEBUG
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

/* invert matrix xx - 8x1 array
 *  * store it in   yy - 8x1 array
 * FIXME: remove duplication of this  */
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

/* struct to pass data to worker threads attached to GPUs */
typedef struct thread_data_pred_t_ {
  int tid; /* this thread id */
  taskhist *hst; /* for load balancing GPUs */
  double *u,*v,*w; /* uvw coords */
  complex double *coh; /* coherencies for M clusters */
  double *x; /* data/residual vector */
  int N; /* stations */
  int Nbase; /* total baselines (N-1)N/2 x tilesz */
  baseline_t *barr; /* baseline info */
  clus_source_t *carr;  /* Mx1 cluster data */
  int M; /* no of clusters */
  int Nf; /* of of freqs */
  double *freqs; /*  Nfx1 freqs  for prediction */
  double fdelta; /* bandwidth */
  double tdelta; /* integration time */
  double dec0; /* phase center dec */

  /* following used in beam prediction */
  int dobeam;
  double ph_ra0,ph_dec0; /* beam pointing */
  double ph_freq0; /* beam central freq */
  double *longitude,*latitude; /* Nx1 array of station locations */
  double *time_utc; /* tileszx1 array of time */
  int tilesz;
  int *Nelem; /* Nx1 array of station sizes */
  double **xx,**yy,**zz; /* Nx1 arrays of station element coords */

  int Ns; /* total no of sources (clusters) per thread */
  int soff; /* starting source for this thread */

  /* following are only used while predict with gain */
  double *p; /* parameter array, size could be 8*N*Mx1 (full) or 8*Nx1 (single)*/

  /* following only used to ignore some clusters while predicting */
  int *ignlist;

} thread_data_pred_t;

/* struct for passing data to threads for correcting data */
typedef struct thread_data_corr_t_ {
  int tid; /* this thread id */
  taskhist *hst; /* for load balancing GPUs */

  double *x; /* residual vector to be corrected : 8*Nbase*Nf*/
  int N; /* stations */
  int Nbase; /* total baselines (N-1)N/2 x tilesz */
  int boff; /* offset of x in for each thread (multiple columns if Nf>1) */
  int Nb; /* no of baselines handeled by this thread */
  baseline_t *barr; /* baseline info */
  int Nf; /* of of freqs */

  double *pinv; /* inverse of solutions to apply: size 8N*nchunk */
  int nchunk; /* how many solutions */
} thread_data_corr_t;

/* copy Nx1 double array x to device as float
   first allocate device memory */
static void
dtofcopy(int N, float **x_d, double *x) {
  float *xhost;
  cudaError_t err;
  /* first alloc pinned temp buffer */
  err=cudaMallocHost((void**)&xhost,sizeof(float)*N);
  checkCudaError(err,__FILE__,__LINE__);
  /* double to float */
  int ci;
  for (ci=0; ci<N; ci++) {
    xhost[ci]=(float)x[ci];
  }

  float *xc;
  /* device alloc */
  err=cudaMalloc((void**)&xc, N*sizeof(float));
  checkCudaError(err,__FILE__,__LINE__);

  /* copy memory */
  err=cudaMemcpy(xc,xhost,N*sizeof(float),cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* free host buffer */
  err=cudaFreeHost(xhost);
  checkCudaError(err,__FILE__,__LINE__);

  *x_d=xc;

}

static void *
precalcoh_threadfn(void *data) {
  thread_data_pred_t *t=(thread_data_pred_t*)data;
  /* first, select a GPU, if total clusters < MAX_GPU_ID
    use random selection, elese use this thread id */
  int card;
  if (t->M<=MAX_GPU_ID) {
   card=select_work_gpu(MAX_GPU_ID,t->hst);
  } else {
   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
  }
  cudaError_t err;
  int ci,ncl,cj;

  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  double *ud,*vd,*wd;
  double *cohd;
  baseline_t *barrd;
  double *freqsd;
  float *longd=0,*latd=0; double *timed;
  int *Nelemd;
  float **xx_p=0,**yy_p=0,**zz_p=0;
  float **xxd,**yyd,**zzd;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nbase*8*sizeof(double)); /* coherencies only for 1 cluster, Nf=1 */
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  
  /* u,v,w and l,m,n coords need to be double for precision */
  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  if (t->dobeam) {
  dtofcopy(t->N,&longd,t->longitude);
  dtofcopy(t->N,&latd,t->latitude);
  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* convert time jd to GMST angle */
  cudakernel_convert_time(t->tilesz,timed);

  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* jagged arrays for element locations */
  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  /* allocate host memory to store pointers */
  if ((xx_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((yy_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((zz_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  for (ci=0; ci<t->N; ci++) {
    err=cudaMalloc((void**)&xx_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&yy_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&zz_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }
  /* now copy data */
  for (ci=0; ci<t->N; ci++) {
    dtofcopy(t->Nelem[ci],&xx_p[ci],t->xx[ci]);
    dtofcopy(t->Nelem[ci],&yy_p[ci],t->yy[ci]);
    dtofcopy(t->Nelem[ci],&zz_p[ci],t->zz[ci]);
  }
  /* now copy pointer locations to device */
  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  }


  float *beamd;
  double *lld,*mmd,*nnd;
  double *sId; float *rad,*decd;
  double *sQd,*sUd,*sVd;
  unsigned char *styped;
  double *sI0d,*f0d,*spec_idxd,*spec_idx1d,*spec_idx2d;
  double *sQ0d,*sU0d,*sV0d;
  int **host_p,**dev_p;

  complex double *tempdcoh;
  err=cudaMallocHost((void**)&tempdcoh,sizeof(complex double)*(size_t)t->Nbase*4);
  checkCudaError(err,__FILE__,__LINE__);

/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {

     if (t->dobeam) {
      /* allocate memory for this clusters beam */
      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
      checkCudaError(err,__FILE__,__LINE__);
     } else {
      beamd=0;
     }

     /* copy cluster details to GPU */
     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->Nf==1) {
     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     }



     if (t->dobeam) {
      dtofcopy(t->carr[ncl].N,&rad,t->carr[ncl].ra);
      dtofcopy(t->carr[ncl].N,&decd,t->carr[ncl].dec);
     } else {
       rad=0;
       decd=0;
     }
     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* for multi channel data - FIXME: remove this part because Nf==1 always */
     if (t->Nf>1) {
     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     }

     /* extra info for source, if any */
     if ((host_p=(int**)calloc((size_t)t->carr[ncl].N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
     checkCudaError(err,__FILE__,__LINE__);


     for (cj=0; cj<t->carr[ncl].N; cj++) {

        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
          host_p[cj]=0;
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d=(exinfo_shapelet*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
          checkCudaError(err,__FILE__,__LINE__);
          double *modes;
          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          exinfo_gaussian *d=(exinfo_gaussian*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          exinfo_disk *d=(exinfo_disk*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          exinfo_ring *d=(exinfo_ring*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        }
        

     }
     /* now copy pointer locations to device */
     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->dobeam) {
     /* now calculate beam for all sources in this cluster */
      cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd);
     }


     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd,
     lld,mmd,nnd,sId,sQd,sUd,sVd,styped,sI0d,sQ0d,sU0d,sV0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,t->fdelta,t->tdelta,t->dec0,cohd,t->dobeam);
    
     /* copy back coherencies to host, 
        coherencies on host have 8M stride, on device have 8 stride */
     err=cudaMemcpy((double*)tempdcoh, cohd, sizeof(double)*t->Nbase*8, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     /* now copy this with right offset and stride */
     my_ccopy(t->Nbase,&tempdcoh[0],4,&(t->coh[4*ncl]),4*t->M);
     my_ccopy(t->Nbase,&tempdcoh[1],4,&(t->coh[4*ncl+1]),4*t->M);
     my_ccopy(t->Nbase,&tempdcoh[2],4,&(t->coh[4*ncl+2]),4*t->M);
     my_ccopy(t->Nbase,&tempdcoh[3],4,&(t->coh[4*ncl+3]),4*t->M);


     for (cj=0; cj<t->carr[ncl].N; cj++) {
        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          double *modes=0;
          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
          err=cudaFree(modes);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        }
     }
     free(host_p);

     err=cudaFree(dev_p);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->dobeam) {
      err=cudaFree(beamd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(lld);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(mmd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(nnd);
     checkCudaError(err,__FILE__,__LINE__);
     if (t->Nf==1) {
     err=cudaFree(sId);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sQd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sUd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sVd);
     checkCudaError(err,__FILE__,__LINE__);
     }
     if (t->dobeam) {
      err=cudaFree(rad);
      checkCudaError(err,__FILE__,__LINE__);
      err=cudaFree(decd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(styped);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->Nf>1) {
     err=cudaFree(sI0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(f0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idxd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx1d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx2d);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaFree(sQ0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sU0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sV0d);
     checkCudaError(err,__FILE__,__LINE__);
     }
  }
/******************* end loop over clusters **************************/

  /* free memory */
  err=cudaFreeHost(tempdcoh);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaFree(ud);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(vd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(wd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(cohd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(freqsd);
  checkCudaError(err,__FILE__,__LINE__);
  
  if (t->dobeam) {
  err=cudaFree(longd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(latd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(timed);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(Nelemd);
  checkCudaError(err,__FILE__,__LINE__);

  for (ci=0; ci<t->N; ci++) {
    err=cudaFree(xx_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(yy_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(zz_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
  }

  err=cudaFree(xxd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(yyd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(zzd);
  checkCudaError(err,__FILE__,__LINE__);

  free(xx_p);
  free(yy_p);
  free(zz_p);
  }

  cudaDeviceSynchronize();

  /* reset error state */
  err=cudaGetLastError(); 

  return NULL;

}

/* worker thread function to (re)set flags */
static void *
resetflags_threadfn(void *data) {
   thread_data_base_t *t=(thread_data_base_t*)data;
   int ci;
   for (ci=0; ci<t->Nb; ci++) {
    if (!t->barr[ci+t->boff].flag) {
     /* change the flag to 2 if baseline length is < uvmin or > uvmax */
     double uvdist=sqrt(t->u[ci]*t->u[ci]+t->v[ci]*t->v[ci])*t->freq0;
     if (uvdist<t->uvmin || uvdist>t->uvmax) {
      t->barr[ci+t->boff].flag=2;
     }
    }
   }
   return NULL;
}

int
precalculate_coherencies_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tilesz, int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt) {

  int nth,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_pred_t *threaddata;
  taskhist thst;
  init_task_hist(&thst);

  int Ngpu=MAX_GPU_ID+1;

  /* calculate min clusters thread can handle */
  Nthb0=(M+Ngpu-1)/Ngpu;

  /* setup threads : note: Ngpu is no of GPUs used */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* set common parameters, and split clusters to threads */
  ci=0;
  for (nth=0;  nth<Ngpu && ci<M; nth++) {
     if (ci+Nthb0<M) {
      Nthb=Nthb0;
     } else {
      Nthb=M-ci;
     }
      
     threaddata[nth].hst=&thst;  /* for load balancing */
     threaddata[nth].tid=nth;
     
     threaddata[nth].u=u; 
     threaddata[nth].v=v;
     threaddata[nth].w=w;
     threaddata[nth].coh=x;
     threaddata[nth].x=0; /* no input data */

     threaddata[nth].N=N;
     threaddata[nth].Nbase=Nbase; /* total baselines: actually Nbasextilesz (from input) */
     threaddata[nth].barr=barr;
     threaddata[nth].carr=carr;
     threaddata[nth].M=M;
     threaddata[nth].Nf=1;
     threaddata[nth].freqs=&freq0; /* only 1 freq */
     threaddata[nth].fdelta=fdelta;
     threaddata[nth].tdelta=tdelta;
     threaddata[nth].dec0=dec0;

     threaddata[nth].dobeam=dobeam; 
     threaddata[nth].ph_ra0=ph_ra0;
     threaddata[nth].ph_dec0=ph_dec0;
     threaddata[nth].ph_freq0=ph_freq0;
     threaddata[nth].longitude=longitude;
     threaddata[nth].latitude=latitude;
     threaddata[nth].time_utc=time_utc;
     threaddata[nth].tilesz=tilesz;
     threaddata[nth].Nelem=Nelem;
     threaddata[nth].xx=xx;
     threaddata[nth].yy=yy;
     threaddata[nth].zz=zz;

     /* no parameters */
     threaddata[nth].p=0;

     threaddata[nth].Ns=Nthb;
     threaddata[nth].soff=ci;

    
     pthread_create(&th_array[nth],&attr,precalcoh_threadfn,(void*)(&threaddata[nth]));
     /* next source set */
     ci=ci+Nthb;
  }

     

  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
  }



 free(threaddata);
 destroy_task_hist(&thst);
 free(th_array);


  thread_data_base_t *threaddata1;
  /* now do some book keeping (setting proper flags) using CPU only */
  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;
  
  if ((threaddata1=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }


  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
     /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
     /* determine actual no. of baselines */
     if (ci+Nthb0<Nbase) {
      Nthb=Nthb0;
     } else {
      Nthb=Nbase-ci;
     }

     threaddata1[nth].boff=ci;
     threaddata1[nth].Nb=Nthb;
     threaddata1[nth].barr=barr;
     threaddata1[nth].u=&(u[ci]);
     threaddata1[nth].v=&(v[ci]);
     threaddata1[nth].w=&(w[ci]);
     threaddata1[nth].uvmin=uvmin;
     threaddata1[nth].uvmax=uvmax;
     threaddata1[nth].freq0=freq0;

     pthread_create(&th_array[nth],&attr,resetflags_threadfn,(void*)(&threaddata1[nth]));

     /* next baseline set */
     ci=ci+Nthb;
  }


  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
   pthread_join(th_array[ci],NULL);
  }



 free(threaddata1);
 pthread_attr_destroy(&attr);
 free(th_array);

 return 0;
}




static void *
predictvis_threadfn(void *data) {
  thread_data_pred_t *t=(thread_data_pred_t*)data;
  /* first, select a GPU, if total clusters < MAX_GPU_ID
    use random selection, elese use this thread id */
  int card;
  if (t->M<=MAX_GPU_ID) {
   card=select_work_gpu(MAX_GPU_ID,t->hst);
  } else {
   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
  }
  cudaError_t err;
  int ci,ncl,cj;


  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);


  double *ud,*vd,*wd;
  double *cohd;
  baseline_t *barrd;
  double *freqsd;
  float *longd=0,*latd=0; double *timed;
  int *Nelemd;
  float **xx_p=0,**yy_p=0,**zz_p=0;
  float **xxd,**yyd,**zzd;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(double)); /* coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  
  /* copy to device */
  /* u,v,w and l,m,n coords need to be double for precision */
  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  /* check if beam is actually calculated */
  if (t->dobeam) {
  dtofcopy(t->N,&longd,t->longitude);
  dtofcopy(t->N,&latd,t->latitude);
  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* convert time jd to GMST angle */
  cudakernel_convert_time(t->tilesz,timed);

  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* jagged arrays for element locations */
  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  /* allocate host memory to store pointers */
  if ((xx_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((yy_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((zz_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  for (ci=0; ci<t->N; ci++) {
    err=cudaMalloc((void**)&xx_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&yy_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&zz_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }
  /* now copy data */
  for (ci=0; ci<t->N; ci++) {
    dtofcopy(t->Nelem[ci],&xx_p[ci],t->xx[ci]);
    dtofcopy(t->Nelem[ci],&yy_p[ci],t->yy[ci]);
    dtofcopy(t->Nelem[ci],&zz_p[ci],t->zz[ci]);
  }
  /* now copy pointer locations to device */
  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  }


  float *beamd;
  double *lld,*mmd,*nnd;
  double *sId; float *rad,*decd;
  double *sQd,*sUd,*sVd;
  unsigned char *styped;
  double *sI0d,*f0d,*spec_idxd,*spec_idx1d,*spec_idx2d;
  double *sQ0d,*sU0d,*sV0d;
  int **host_p,**dev_p;

  double *xlocal;
  err=cudaMallocHost((void**)&xlocal,sizeof(double)*(size_t)t->Nbase*8*t->Nf);
  checkCudaError(err,__FILE__,__LINE__);

/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
     /* allocate memory for this clusters beam */
     if (t->dobeam) {
      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
      checkCudaError(err,__FILE__,__LINE__);
     } else {
       beamd=0;
     }


     /* copy cluster details to GPU */
     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->Nf==1) {
     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     }


     if (t->dobeam) {
      dtofcopy(t->carr[ncl].N,&rad,t->carr[ncl].ra);
      dtofcopy(t->carr[ncl].N,&decd,t->carr[ncl].dec);
     } else {
       rad=0;
       decd=0;
     }
     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* for multi channel data */
     if (t->Nf>1) {
     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     }



     /* extra info for source, if any */
     if ((host_p=(int**)calloc((size_t)t->carr[ncl].N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
     checkCudaError(err,__FILE__,__LINE__);


     for (cj=0; cj<t->carr[ncl].N; cj++) {

        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
          host_p[cj]=0;
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d=(exinfo_shapelet*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
          checkCudaError(err,__FILE__,__LINE__);
          double *modes;
          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          exinfo_gaussian *d=(exinfo_gaussian*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          exinfo_disk *d=(exinfo_disk*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          exinfo_ring *d=(exinfo_ring*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        }
        

     }
     /* now copy pointer locations to device */
     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->dobeam) {
      /* now calculate beam for all sources in this cluster */
      cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd);
     }


     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd,
     lld,mmd,nnd,sId,sQd,sUd,sVd,styped,sI0d,sQ0d,sU0d,sV0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,t->fdelta,t->tdelta,t->dec0,cohd,t->dobeam);
    
     /* copy back coherencies to host, 
        coherencies on host have 8M stride, on device have 8 stride */
     err=cudaMemcpy(xlocal, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     my_daxpy(t->Nbase*8*t->Nf,xlocal,1.0,t->x);

     for (cj=0; cj<t->carr[ncl].N; cj++) {
        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          double *modes=0;
          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
          err=cudaFree(modes);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        }
     }
     free(host_p);

     err=cudaFree(dev_p);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->dobeam) {
      err=cudaFree(beamd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(lld);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(mmd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(nnd);
     checkCudaError(err,__FILE__,__LINE__);
     if (t->Nf==1) {
     err=cudaFree(sId);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sQd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sUd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sVd);
     checkCudaError(err,__FILE__,__LINE__);
     }
     if (t->dobeam) {
      err=cudaFree(rad);
      checkCudaError(err,__FILE__,__LINE__);
      err=cudaFree(decd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(styped);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->Nf>1) {
     err=cudaFree(sI0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(f0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idxd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx1d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx2d);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaFree(sQ0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sU0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sV0d);
     checkCudaError(err,__FILE__,__LINE__);
     }
  }
/******************* end loop over clusters **************************/

  /* free memory */
  err=cudaFreeHost(xlocal);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaFree(ud);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(vd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(wd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(cohd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(freqsd);
  checkCudaError(err,__FILE__,__LINE__);

  if (t->dobeam) {
  err=cudaFree(longd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(latd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(timed);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(Nelemd);
  checkCudaError(err,__FILE__,__LINE__);


  for (ci=0; ci<t->N; ci++) {
    err=cudaFree(xx_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(yy_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(zz_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
  }

  err=cudaFree(xxd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(yyd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(zzd);
  checkCudaError(err,__FILE__,__LINE__);

  free(xx_p);
  free(yy_p);
  free(zz_p);
  }

  cudaDeviceSynchronize();
  /* reset error state */
  err=cudaGetLastError(); 
  return NULL;

}

int
predict_visibilities_multifreq_withbeam_gpu(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt, int add_to_data) {

  int nth,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_pred_t *threaddata;
  taskhist thst;
  init_task_hist(&thst);

  /* oversubsribe GPU */
  int Ngpu=MAX_GPU_ID+1;

  /* calculate min clusters thread can handle */
  Nthb0=(M+Ngpu-1)/Ngpu;

  /* setup threads : note: Ngpu is no of GPUs used */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* arrays to store result */
  double *xlocal;
  if ((xlocal=(double*)calloc((size_t)Nbase*8*tilesz*Nchan*Ngpu,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  if (add_to_data==SIMUL_ONLY) {
   /* set input column to zero */
   memset(x,0,sizeof(double)*Nbase*8*tilesz*Nchan);
  }



  /* set common parameters, and split clusters to threads */
  ci=0;
  for (nth=0;  nth<Ngpu && ci<M; nth++) {
     if (ci+Nthb0<M) {
      Nthb=Nthb0;
     } else {
      Nthb=M-ci;
     }
      
     threaddata[nth].hst=&thst;  /* for load balancing */
     threaddata[nth].tid=nth;

     threaddata[nth].u=u; 
     threaddata[nth].v=v;
     threaddata[nth].w=w;
     threaddata[nth].coh=0;
     threaddata[nth].x=&xlocal[nth*Nbase*8*tilesz*Nchan]; /* distinct arrays to get back the result */

     threaddata[nth].N=N;
     threaddata[nth].Nbase=Nbase*tilesz; /* total baselines: actually Nbasextilesz */
     threaddata[nth].barr=barr;
     threaddata[nth].carr=carr;
     threaddata[nth].M=M;
     threaddata[nth].Nf=Nchan;
     threaddata[nth].freqs=freqs;
     threaddata[nth].fdelta=fdelta;
     threaddata[nth].tdelta=tdelta;
     threaddata[nth].dec0=dec0;

     threaddata[nth].dobeam=dobeam;
     threaddata[nth].ph_ra0=ph_ra0;
     threaddata[nth].ph_dec0=ph_dec0;
     threaddata[nth].ph_freq0=ph_freq0;
     threaddata[nth].longitude=longitude;
     threaddata[nth].latitude=latitude;
     threaddata[nth].time_utc=time_utc;
     threaddata[nth].tilesz=tilesz;
     threaddata[nth].Nelem=Nelem;
     threaddata[nth].xx=xx;
     threaddata[nth].yy=yy;
     threaddata[nth].zz=zz;

     /* no parameters */
     threaddata[nth].p=0;

     threaddata[nth].Ns=Nthb;
     threaddata[nth].soff=ci;

    
     pthread_create(&th_array[nth],&attr,predictvis_threadfn,(void*)(&threaddata[nth]));
     /* next source set */
     ci=ci+Nthb;
  }

     

  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
    /* add or copy xlocal back to x */
    if (add_to_data==SIMUL_ONLY || add_to_data==SIMUL_ADD) {
     my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,1.0,x);
    } else { /* subtract */
     my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,-1.0,x);
    }
  }



  free(xlocal);
  free(threaddata);
  destroy_task_hist(&thst);
  free(th_array);
  pthread_attr_destroy(&attr);

  return 0;
}



static void *
residual_threadfn(void *data) {
  thread_data_pred_t *t=(thread_data_pred_t*)data;
  /* first, select a GPU, if total clusters < MAX_GPU_ID
    use random selection, elese use this thread id */
  int card;
  if (t->M<=MAX_GPU_ID) {
   card=select_work_gpu(MAX_GPU_ID,t->hst);
  } else {
   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
  }
  cudaError_t err;
  int ci,ncl,cj;


  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  double *ud,*vd,*wd;
  double *cohd;
  baseline_t *barrd;
  double *freqsd;
  float *longd=0,*latd=0; double *timed;
  int *Nelemd;
  float **xx_p=0,**yy_p=0,**zz_p=0;
  float **xxd,**yyd,**zzd;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(double)); /* coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  
  /* copy to device */
  /* u,v,w and l,m,n coords need to be double for precision */
  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  /* check if beam is actually calculated */
  if (t->dobeam) {
  dtofcopy(t->N,&longd,t->longitude);
  dtofcopy(t->N,&latd,t->latitude);
  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* convert time jd to GMST angle */
  cudakernel_convert_time(t->tilesz,timed);

  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* jagged arrays for element locations */
  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  /* allocate host memory to store pointers */
  if ((xx_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((yy_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((zz_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  for (ci=0; ci<t->N; ci++) {
    err=cudaMalloc((void**)&xx_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&yy_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&zz_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }
  /* now copy data */
  for (ci=0; ci<t->N; ci++) {
    dtofcopy(t->Nelem[ci],&xx_p[ci],t->xx[ci]);
    dtofcopy(t->Nelem[ci],&yy_p[ci],t->yy[ci]);
    dtofcopy(t->Nelem[ci],&zz_p[ci],t->zz[ci]);
  }
  /* now copy pointer locations to device */
  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  }


  float *beamd;
  double *lld,*mmd,*nnd;
  double *sId; float *rad,*decd;
  double *sQd,*sUd,*sVd;
  unsigned char *styped;
  double *sI0d,*f0d,*spec_idxd,*spec_idx1d,*spec_idx2d;
  double *sQ0d,*sU0d,*sV0d;
  int **host_p,**dev_p;

  double *xlocal;
  err=cudaMallocHost((void**)&xlocal,sizeof(double)*(size_t)t->Nbase*8*t->Nf);
  checkCudaError(err,__FILE__,__LINE__);

  double *pd; /* parameter array per cluster */

/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
     /* check if cluster id >=0 to do a subtraction */
     if (t->carr[ncl].id>=0) {
     /* allocate memory for this clusters beam */
     if (t->dobeam) {
      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
      checkCudaError(err,__FILE__,__LINE__);
     } else {
       beamd=0;
     }


     /* copy cluster details to GPU */
     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* parameter vector size may change depending on hybrid parameter */
     err=cudaMalloc((void**) &pd, t->N*8*t->carr[ncl].nchunk*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(pd, &(t->p[t->carr[ncl].p[0]]), t->N*8*t->carr[ncl].nchunk*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->Nf==1) {
     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     }


     if (t->dobeam) {
      dtofcopy(t->carr[ncl].N,&rad,t->carr[ncl].ra);
      dtofcopy(t->carr[ncl].N,&decd,t->carr[ncl].dec);
     } else {
       rad=0;
       decd=0;
     }
     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* for multi channel data */
     if (t->Nf>1) {
     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     }



     /* extra info for source, if any */
     if ((host_p=(int**)calloc((size_t)t->carr[ncl].N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
     checkCudaError(err,__FILE__,__LINE__);


     for (cj=0; cj<t->carr[ncl].N; cj++) {

        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
          host_p[cj]=0;
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d=(exinfo_shapelet*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
          checkCudaError(err,__FILE__,__LINE__);
          double *modes;
          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          exinfo_gaussian *d=(exinfo_gaussian*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          exinfo_disk *d=(exinfo_disk*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          exinfo_ring *d=(exinfo_ring*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        }
        

     }
     /* now copy pointer locations to device */
     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->dobeam) {
      /* now calculate beam for all sources in this cluster */
      cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd);
     }


     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_residuals(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,pd,t->carr[ncl].nchunk,barrd,freqsd,beamd,
     lld,mmd,nnd,sId,sQd,sUd,sVd,styped,sI0d,sQ0d,sU0d,sV0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,t->fdelta,t->tdelta,t->dec0,cohd,t->dobeam);
    
     /* copy back coherencies to host, 
        coherencies on host have 8M stride, on device have 8 stride */
     err=cudaMemcpy(xlocal, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     my_daxpy(t->Nbase*8*t->Nf,xlocal,1.0,t->x);

     for (cj=0; cj<t->carr[ncl].N; cj++) {
        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          double *modes=0;
          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
          err=cudaFree(modes);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        }
     }
     free(host_p);

     err=cudaFree(dev_p);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->dobeam) {
      err=cudaFree(beamd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(lld);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(mmd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(nnd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(pd);
     checkCudaError(err,__FILE__,__LINE__);
     if (t->Nf==1) {
     err=cudaFree(sId);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sQd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sUd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sVd);
     checkCudaError(err,__FILE__,__LINE__);
     }
     if (t->dobeam) {
      err=cudaFree(rad);
      checkCudaError(err,__FILE__,__LINE__);
      err=cudaFree(decd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(styped);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->Nf>1) {
     err=cudaFree(sI0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(f0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idxd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx1d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx2d);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaFree(sQ0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sU0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sV0d);
     checkCudaError(err,__FILE__,__LINE__);
     }
   }
  }
/******************* end loop over clusters **************************/

  /* free memory */
  err=cudaFreeHost(xlocal);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaFree(ud);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(vd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(wd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(cohd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(freqsd);
  checkCudaError(err,__FILE__,__LINE__);

  if (t->dobeam) {
  err=cudaFree(longd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(latd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(timed);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(Nelemd);
  checkCudaError(err,__FILE__,__LINE__);


  for (ci=0; ci<t->N; ci++) {
    err=cudaFree(xx_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(yy_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(zz_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
  }

  err=cudaFree(xxd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(yyd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(zzd);
  checkCudaError(err,__FILE__,__LINE__);

  free(xx_p);
  free(yy_p);
  free(zz_p);
  }


  cudaDeviceSynchronize();
  /* reset error state */
  err=cudaGetLastError(); 
  return NULL;

}


static void *
correct_threadfn(void *data) {
  thread_data_corr_t *t=(thread_data_corr_t*)data;
  /* first, select a GPU, if total threads > MAX_GPU_ID
    use random selection, elese use this thread id */
  int card;
  if (t->tid>MAX_GPU_ID) {
   card=select_work_gpu(MAX_GPU_ID,t->hst);
  } else {
   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
  }
  cudaError_t err;
  int cf;

  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  double *xd;
  baseline_t *barrd;
  double *pd;

  /* allocate memory in GPU */
  err=cudaMalloc((void**) &xd, t->Nb*8*t->Nf*sizeof(double)); /* coherencies only for Nb baselines, Nf freqs */
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nb*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &pd, t->N*8*t->nchunk*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);

  /* copy with right offset */
  err=cudaMemcpy(barrd, &(t->barr[t->boff]), t->Nb*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(pd, t->pinv, t->N*8*t->nchunk*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  
  for (cf=0; cf<t->Nf; cf++) {
    err=cudaMemcpy(&xd[cf*8*t->Nb], &(t->x[cf*8*t->Nbase+t->boff*8]), t->Nb*8*sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);
  }

  /* run kernel */
  cudakernel_correct_residuals(t->Nbase,t->N,t->Nb,t->boff,t->Nf,t->nchunk,xd,pd,barrd);
 
  /* copy back corrected result */
  for (cf=0; cf<t->Nf; cf++) {
    err=cudaMemcpy(&(t->x[cf*8*t->Nbase+t->boff*8]), &xd[cf*8*t->Nb], t->Nb*8*sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaError(err,__FILE__,__LINE__);
  }

  err=cudaFree(xd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(pd);
  checkCudaError(err,__FILE__,__LINE__);


  cudaDeviceSynchronize();
  /* reset error state */
  err=cudaGetLastError();
  return NULL;
}

/* p: 8NMx1 parameter array, but M is 'effective' clusters, need to use carr to find the right offset
   ccid: which cluster to use as correction
   rho: MMSE robust parameter J+rho I inverted

   phase_only: if >0, and if there is any correction done, use only phase of diagonal elements for correction 
*/
int
calculate_residuals_multifreq_withbeam_gpu(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt, int ccid, double rho, int phase_only) {

  int nth,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_pred_t *threaddata;
  thread_data_corr_t *threaddata_corr;
  taskhist thst;
  init_task_hist(&thst);

  /* oversubsribe GPU */
  int Ngpu=MAX_GPU_ID+1;

  /* calculate min clusters thread can handle */
  Nthb0=(M+Ngpu-1)/Ngpu;

  /* setup threads : note: Ngpu is no of GPUs used */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* arrays to store result */
  double *xlocal;
  if ((xlocal=(double*)calloc((size_t)Nbase*8*tilesz*Nchan*Ngpu,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  /* set common parameters, and split clusters to threads */
  ci=0;
  for (nth=0;  nth<Ngpu && ci<M; nth++) {
     if (ci+Nthb0<M) {
      Nthb=Nthb0;
     } else {
      Nthb=M-ci;
     }
      
     threaddata[nth].hst=&thst;  /* for load balancing */
     threaddata[nth].tid=nth;

     threaddata[nth].u=u; 
     threaddata[nth].v=v;
     threaddata[nth].w=w;
     threaddata[nth].coh=0;
     threaddata[nth].x=&xlocal[nth*Nbase*8*tilesz*Nchan]; /* distinct arrays to get back the result */

     threaddata[nth].N=N;
     threaddata[nth].Nbase=Nbase*tilesz; /* total baselines: actually Nbasextilesz */
     threaddata[nth].barr=barr;
     threaddata[nth].carr=carr;
     threaddata[nth].M=M;
     threaddata[nth].Nf=Nchan;
     threaddata[nth].freqs=freqs;
     threaddata[nth].fdelta=fdelta;
     threaddata[nth].tdelta=tdelta;
     threaddata[nth].dec0=dec0;

     threaddata[nth].dobeam=dobeam;
     threaddata[nth].ph_ra0=ph_ra0;
     threaddata[nth].ph_dec0=ph_dec0;
     threaddata[nth].ph_freq0=ph_freq0;
     threaddata[nth].longitude=longitude;
     threaddata[nth].latitude=latitude;
     threaddata[nth].time_utc=time_utc;
     threaddata[nth].tilesz=tilesz;
     threaddata[nth].Nelem=Nelem;
     threaddata[nth].xx=xx;
     threaddata[nth].yy=yy;
     threaddata[nth].zz=zz;

     /* parameters */
     threaddata[nth].p=p;

     threaddata[nth].Ns=Nthb;
     threaddata[nth].soff=ci;

    
     pthread_create(&th_array[nth],&attr,residual_threadfn,(void*)(&threaddata[nth]));
     /* next source set */
     ci=ci+Nthb;
  }

     

  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
    /* subtract to find residual */
    my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,-1.0,x);
  }



  free(xlocal);
  free(threaddata);

  /* find if any cluster is specified for correction of data */
  int cm,cj;
  cm=-1;
  for (cj=0; cj<M; cj++) { /* clusters */
    /* check if cluster id == ccid to do a correction */
    if (carr[cj].id==ccid) {
     /* valid cluster found */
     cm=cj;
    }
  }
  /* correction of data, if valid cluster is given */
  if (cm>=0) {
   double *pm,*pinv,*pphase=0;
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

   /* divide x[] over GPUs */
   int Nbase1=Nbase*tilesz;
   Nthb0=(Nbase1+Ngpu-1)/Ngpu;

   if ((threaddata_corr=(thread_data_corr_t*)malloc((size_t)Ngpu*sizeof(thread_data_corr_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
   }
   ci=0;
   for (nth=0;  nth<Ngpu && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata_corr[nth].hst=&thst;
    threaddata_corr[nth].tid=nth;
    threaddata_corr[nth].boff=ci;
    threaddata_corr[nth].Nb=Nthb;
    threaddata_corr[nth].barr=barr;
    threaddata_corr[nth].x=x;
    threaddata_corr[nth].N=N;
    threaddata_corr[nth].Nbase=Nbase1;
    threaddata_corr[nth].Nf=Nchan;
    /* for correction of data */
    threaddata_corr[nth].pinv=pinv;
    threaddata_corr[nth].nchunk=carr[cm].nchunk;


    pthread_create(&th_array[nth],&attr,correct_threadfn,(void*)(&threaddata_corr[nth]));
    /* next baseline set */
    ci=ci+Nthb;
   }

   /* now wait for threads to finish */
   for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
   }

   
   free(pinv);
   free(threaddata_corr);
  }


  free(th_array);
  pthread_attr_destroy(&attr);

  destroy_task_hist(&thst);

  return 0;
}



static void *
predict_threadfn(void *data) {
  thread_data_pred_t *t=(thread_data_pred_t*)data;
  /* first, select a GPU, if total clusters < MAX_GPU_ID
    use random selection, elese use this thread id */
  int card;
  if (t->M<=MAX_GPU_ID) {
   card=select_work_gpu(MAX_GPU_ID,t->hst);
  } else {
   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
  }
  cudaError_t err;
  int ci,ncl,cj;


  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  double *ud,*vd,*wd;
  double *cohd;
  baseline_t *barrd;
  double *freqsd;
  float *longd=0,*latd=0; double *timed;
  int *Nelemd;
  float **xx_p=0,**yy_p=0,**zz_p=0;
  float **xxd,**yyd,**zzd;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(double)); /* coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  
  /* copy to device */
  /* u,v,w and l,m,n coords need to be double for precision */
  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  /* check if beam is actually calculated */
  if (t->dobeam) {
  dtofcopy(t->N,&longd,t->longitude);
  dtofcopy(t->N,&latd,t->latitude);
  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* convert time jd to GMST angle */
  cudakernel_convert_time(t->tilesz,timed);

  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* jagged arrays for element locations */
  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  /* allocate host memory to store pointers */
  if ((xx_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((yy_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((zz_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  for (ci=0; ci<t->N; ci++) {
    err=cudaMalloc((void**)&xx_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&yy_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&zz_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }
  /* now copy data */
  for (ci=0; ci<t->N; ci++) {
    dtofcopy(t->Nelem[ci],&xx_p[ci],t->xx[ci]);
    dtofcopy(t->Nelem[ci],&yy_p[ci],t->yy[ci]);
    dtofcopy(t->Nelem[ci],&zz_p[ci],t->zz[ci]);
  }
  /* now copy pointer locations to device */
  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  }


  float *beamd;
  double *lld,*mmd,*nnd;
  double *sId; float *rad,*decd;
  double *sQd,*sUd,*sVd;
  unsigned char *styped;
  double *sI0d,*f0d,*spec_idxd,*spec_idx1d,*spec_idx2d;
  double *sQ0d,*sU0d,*sV0d;
  int **host_p,**dev_p;

  double *xlocal;
  err=cudaMallocHost((void**)&xlocal,sizeof(double)*(size_t)t->Nbase*8*t->Nf);
  checkCudaError(err,__FILE__,__LINE__);

  double *pd; /* parameter array per cluster */

/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
     /* check if cluster id is not part of the ignore list */
     if (!t->ignlist[ncl]) {
     /* allocate memory for this clusters beam */
     if (t->dobeam) {
      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
      checkCudaError(err,__FILE__,__LINE__);
     } else {
       beamd=0;
     }


     /* copy cluster details to GPU */
     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* parameter vector size may change depending on hybrid parameter */
     err=cudaMalloc((void**) &pd, t->N*8*t->carr[ncl].nchunk*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(pd, &(t->p[t->carr[ncl].p[0]]), t->N*8*t->carr[ncl].nchunk*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->Nf==1) {
     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     }


     if (t->dobeam) {
      dtofcopy(t->carr[ncl].N,&rad,t->carr[ncl].ra);
      dtofcopy(t->carr[ncl].N,&decd,t->carr[ncl].dec);
     } else {
       rad=0;
       decd=0;
     }
     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* for multi channel data */
     if (t->Nf>1) {
     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     }



     /* extra info for source, if any */
     if ((host_p=(int**)calloc((size_t)t->carr[ncl].N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
     checkCudaError(err,__FILE__,__LINE__);


     for (cj=0; cj<t->carr[ncl].N; cj++) {

        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
          host_p[cj]=0;
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d=(exinfo_shapelet*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
          checkCudaError(err,__FILE__,__LINE__);
          double *modes;
          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          exinfo_gaussian *d=(exinfo_gaussian*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          exinfo_disk *d=(exinfo_disk*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          exinfo_ring *d=(exinfo_ring*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        }
        

     }
     /* now copy pointer locations to device */
     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->dobeam) {
      /* now calculate beam for all sources in this cluster */
      cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd);
     }


     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_residuals(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,pd,t->carr[ncl].nchunk,barrd,freqsd,beamd,
     lld,mmd,nnd,sId,sQd,sUd,sVd,styped,sI0d,sQ0d,sU0d,sV0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,t->fdelta,t->tdelta,t->dec0,cohd,t->dobeam);
    
     /* copy back coherencies to host, 
        coherencies on host have 8M stride, on device have 8 stride */
     err=cudaMemcpy(xlocal, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     my_daxpy(t->Nbase*8*t->Nf,xlocal,1.0,t->x);

     for (cj=0; cj<t->carr[ncl].N; cj++) {
        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          double *modes=0;
          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
          err=cudaFree(modes);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        }
     }
     free(host_p);

     err=cudaFree(dev_p);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->dobeam) {
      err=cudaFree(beamd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(lld);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(mmd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(nnd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(pd);
     checkCudaError(err,__FILE__,__LINE__);
     if (t->Nf==1) {
     err=cudaFree(sId);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sQd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sUd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sVd);
     checkCudaError(err,__FILE__,__LINE__);
     }
     if (t->dobeam) {
      err=cudaFree(rad);
      checkCudaError(err,__FILE__,__LINE__);
      err=cudaFree(decd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(styped);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->Nf>1) {
     err=cudaFree(sI0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(f0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idxd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx1d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx2d);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaFree(sQ0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sU0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sV0d);
     checkCudaError(err,__FILE__,__LINE__);
     }
   }
  }
/******************* end loop over clusters **************************/

  /* free memory */
  err=cudaFreeHost(xlocal);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaFree(ud);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(vd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(wd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(cohd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(freqsd);
  checkCudaError(err,__FILE__,__LINE__);

  if (t->dobeam) {
  err=cudaFree(longd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(latd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(timed);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(Nelemd);
  checkCudaError(err,__FILE__,__LINE__);


  for (ci=0; ci<t->N; ci++) {
    err=cudaFree(xx_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(yy_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(zz_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
  }

  err=cudaFree(xxd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(yyd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(zzd);
  checkCudaError(err,__FILE__,__LINE__);

  free(xx_p);
  free(yy_p);
  free(zz_p);
  }


  cudaDeviceSynchronize();
  /* reset error state */
  err=cudaGetLastError(); 
  return NULL;

}

/* 
  almost same as calculate_residuals_multifreq_withbeam_gpu(), but ignorelist is used to ignore prediction of some clusters, and predict/add/subtract options based on simulation mode add_to_data

  int * ignorelist: Mx1 array
  int add_to_data: predict/add/subtract mode
*/
int
predict_visibilities_withsol_withbeam_gpu(double *u,double *v,double *w,double *p,double *x, int *ignorelist, int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt, int add_to_data, int ccid, double rho, int phase_only) {

  int nth,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_pred_t *threaddata;
  thread_data_corr_t *threaddata_corr;
  taskhist thst;
  init_task_hist(&thst);

  /* oversubsribe GPU */
  int Ngpu=MAX_GPU_ID+1;

  /* calculate min clusters thread can handle */
  Nthb0=(M+Ngpu-1)/Ngpu;

  /* setup threads : note: Ngpu is no of GPUs used */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* arrays to store result */
  double *xlocal;
  if ((xlocal=(double*)calloc((size_t)Nbase*8*tilesz*Nchan*Ngpu,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  /* set common parameters, and split clusters to threads */
  ci=0;
  for (nth=0;  nth<Ngpu && ci<M; nth++) {
     if (ci+Nthb0<M) {
      Nthb=Nthb0;
     } else {
      Nthb=M-ci;
     }
      
     threaddata[nth].hst=&thst;  /* for load balancing */
     threaddata[nth].tid=nth;

     threaddata[nth].u=u; 
     threaddata[nth].v=v;
     threaddata[nth].w=w;
     threaddata[nth].coh=0;
     threaddata[nth].x=&xlocal[nth*Nbase*8*tilesz*Nchan]; /* distinct arrays to get back the result */

     threaddata[nth].N=N;
     threaddata[nth].Nbase=Nbase*tilesz; /* total baselines: actually Nbasextilesz */
     threaddata[nth].barr=barr;
     threaddata[nth].carr=carr;
     threaddata[nth].M=M;
     threaddata[nth].Nf=Nchan;
     threaddata[nth].freqs=freqs;
     threaddata[nth].fdelta=fdelta;
     threaddata[nth].tdelta=tdelta;
     threaddata[nth].dec0=dec0;

     threaddata[nth].dobeam=dobeam;
     threaddata[nth].ph_ra0=ph_ra0;
     threaddata[nth].ph_dec0=ph_dec0;
     threaddata[nth].ph_freq0=ph_freq0;
     threaddata[nth].longitude=longitude;
     threaddata[nth].latitude=latitude;
     threaddata[nth].time_utc=time_utc;
     threaddata[nth].tilesz=tilesz;
     threaddata[nth].Nelem=Nelem;
     threaddata[nth].xx=xx;
     threaddata[nth].yy=yy;
     threaddata[nth].zz=zz;

     /* parameters */
     threaddata[nth].p=p;

     threaddata[nth].Ns=Nthb;
     threaddata[nth].soff=ci;

     /* which clusters to ignore */
     threaddata[nth].ignlist=ignorelist;
    
     pthread_create(&th_array[nth],&attr,predict_threadfn,(void*)(&threaddata[nth]));
     /* next source set */
     ci=ci+Nthb;
  }

     

  if (add_to_data==SIMUL_ONLY) {
    /* set input to zero */
    memset(x,0,sizeof(double)*8*Nbase*tilesz*Nchan);
  }
  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
    if (add_to_data==SIMUL_ONLY || add_to_data==SIMUL_ADD) {
     /* add to input value */
     my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,1.0,x);
    } else {
     /* subtract from input value */
     my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,-1.0,x);
    }
  }



  free(xlocal);
  free(threaddata);

  /* find if any cluster is specified for correction of data */
  int cm,cj;
  cm=-1;
  for (cj=0; cj<M; cj++) { /* clusters */
    /* check if cluster id == ccid to do a correction */
    if (carr[cj].id==ccid) {
     /* valid cluster found */
     cm=cj;
    }
  }
  /* correction of data, if valid cluster is given */
  if (cm>=0) {
   double *pm,*pinv,*pphase=0;
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

   /* divide x[] over GPUs */
   int Nbase1=Nbase*tilesz;
   Nthb0=(Nbase1+Ngpu-1)/Ngpu;

   if ((threaddata_corr=(thread_data_corr_t*)malloc((size_t)Ngpu*sizeof(thread_data_corr_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
   }
   ci=0;
   for (nth=0;  nth<Ngpu && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata_corr[nth].hst=&thst;
    threaddata_corr[nth].tid=nth;
    threaddata_corr[nth].boff=ci;
    threaddata_corr[nth].Nb=Nthb;
    threaddata_corr[nth].barr=barr;
    threaddata_corr[nth].x=x;
    threaddata_corr[nth].N=N;
    threaddata_corr[nth].Nbase=Nbase1;
    threaddata_corr[nth].Nf=Nchan;
    /* for correction of data */
    threaddata_corr[nth].pinv=pinv;
    threaddata_corr[nth].nchunk=carr[cm].nchunk;


    pthread_create(&th_array[nth],&attr,correct_threadfn,(void*)(&threaddata_corr[nth]));
    /* next baseline set */
    ci=ci+Nthb;
   }

   /* now wait for threads to finish */
   for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
   }

   
   free(pinv);
   free(threaddata_corr);
  }


  free(th_array);
  pthread_attr_destroy(&attr);

  destroy_task_hist(&thst);

  return 0;
}



static void *
precalcoh_multifreq_threadfn(void *data) {
  thread_data_pred_t *t=(thread_data_pred_t*)data;
  /* first, select a GPU, if total clusters < MAX_GPU_ID
    use random selection, elese use this thread id */
  int card;
  if (t->M<=MAX_GPU_ID) {
   card=select_work_gpu(MAX_GPU_ID,t->hst);
  } else {
   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
  }
  cudaError_t err;
  int ci,ncl,cj;

  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  double *ud,*vd,*wd;
  double *cohd;
  baseline_t *barrd;
  double *freqsd;
  float *longd=0,*latd=0; double *timed;
  int *Nelemd;
  float **xx_p=0,**yy_p=0,**zz_p=0;
  float **xxd,**yyd,**zzd;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nf*t->Nbase*8*sizeof(double)); /* coherencies only for 1 cluster, Nf>=1 */
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  
  /* u,v,w and l,m,n coords need to be double for precision */
  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);


  if (t->dobeam) {
  dtofcopy(t->N,&longd,t->longitude);
  dtofcopy(t->N,&latd,t->latitude);
  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* convert time jd to GMST angle */
  cudakernel_convert_time(t->tilesz,timed);

  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* jagged arrays for element locations */
  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
  checkCudaError(err,__FILE__,__LINE__);
  /* allocate host memory to store pointers */
  if ((xx_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((yy_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((zz_p=(float**)calloc((size_t)t->N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  for (ci=0; ci<t->N; ci++) {
    err=cudaMalloc((void**)&xx_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&yy_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&zz_p[ci], t->Nelem[ci]*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }
  /* now copy data */
  for (ci=0; ci<t->N; ci++) {
    dtofcopy(t->Nelem[ci],&xx_p[ci],t->xx[ci]);
    dtofcopy(t->Nelem[ci],&yy_p[ci],t->yy[ci]);
    dtofcopy(t->Nelem[ci],&zz_p[ci],t->zz[ci]);
  }
  /* now copy pointer locations to device */
  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  }


  float *beamd;
  double *lld,*mmd,*nnd;
  double *sId; float *rad,*decd;
  double *sQd,*sUd,*sVd;
  unsigned char *styped;
  double *sI0d,*f0d,*spec_idxd,*spec_idx1d,*spec_idx2d;
  double *sQ0d,*sU0d,*sV0d;
  int **host_p,**dev_p;

  complex double *tempdcoh;
  err=cudaMallocHost((void**)&tempdcoh,sizeof(complex double)*(size_t)t->Nbase*4*t->Nf);
  checkCudaError(err,__FILE__,__LINE__);

/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {

     if (t->dobeam) {
      /* allocate memory for this clusters beam */
      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
      checkCudaError(err,__FILE__,__LINE__);
     } else {
      beamd=0;
     }

     /* copy cluster details to GPU */
     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     if (t->Nf==1) {
     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     }



     if (t->dobeam) {
      dtofcopy(t->carr[ncl].N,&rad,t->carr[ncl].ra);
      dtofcopy(t->carr[ncl].N,&decd,t->carr[ncl].dec);
     } else {
       rad=0;
       decd=0;
     }
     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* for multi channel data */
     if (t->Nf>1) {
     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);


     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     }

     /* extra info for source, if any */
     if ((host_p=(int**)calloc((size_t)t->carr[ncl].N,sizeof(int*)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
     checkCudaError(err,__FILE__,__LINE__);


     for (cj=0; cj<t->carr[ncl].N; cj++) {

        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
          host_p[cj]=0;
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d=(exinfo_shapelet*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
          checkCudaError(err,__FILE__,__LINE__);
          double *modes;
          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          exinfo_gaussian *d=(exinfo_gaussian*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          exinfo_disk *d=(exinfo_disk*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          exinfo_ring *d=(exinfo_ring*)t->carr[ncl].ex[cj];
          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
          checkCudaError(err,__FILE__,__LINE__);
        }
        

     }
     /* now copy pointer locations to device */
     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->dobeam) {
     /* now calculate beam for all sources in this cluster */
      cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd);
     }


     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd,
     lld,mmd,nnd,sId,sQd,sUd,sVd,styped,sI0d,sQ0d,sU0d,sV0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,t->fdelta,t->tdelta,t->dec0,cohd,t->dobeam);
    
     /* copy back coherencies to host, 
        coherencies on host have 8M stride, on device have 8 stride */
     err=cudaMemcpy((double*)tempdcoh, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     /* now copy this with right offset and stride */
     int nchan;
     for (nchan=0; nchan<t->Nf; nchan++) {
      my_ccopy(t->Nbase,&tempdcoh[0+4*t->Nbase*nchan],4,&(t->coh[4*ncl+4*t->Nbase*t->M*nchan]),4*t->M);
      my_ccopy(t->Nbase,&tempdcoh[1+4*t->Nbase*nchan],4,&(t->coh[4*ncl+1+4*t->Nbase*t->M*nchan]),4*t->M);
      my_ccopy(t->Nbase,&tempdcoh[2+4*t->Nbase*nchan],4,&(t->coh[4*ncl+2+4*t->Nbase*t->M*nchan]),4*t->M);
      my_ccopy(t->Nbase,&tempdcoh[3+4*t->Nbase*nchan],4,&(t->coh[4*ncl+3+4*t->Nbase*t->M*nchan]),4*t->M);
     }


     for (cj=0; cj<t->carr[ncl].N; cj++) {
        if (t->carr[ncl].stype[cj]==STYPE_POINT) {
        } else if (t->carr[ncl].stype[cj]==STYPE_SHAPELET) {
          exinfo_shapelet *d_p=(exinfo_shapelet *)host_p[cj];
          double *modes=0;
          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
          err=cudaFree(modes);
          checkCudaError(err,__FILE__,__LINE__);
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_GAUSSIAN) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_DISK) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        } else if (t->carr[ncl].stype[cj]==STYPE_RING) {
          err=cudaFree(host_p[cj]);
          checkCudaError(err,__FILE__,__LINE__);
        }
     }
     free(host_p);

     err=cudaFree(dev_p);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->dobeam) {
      err=cudaFree(beamd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(lld);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(mmd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(nnd);
     checkCudaError(err,__FILE__,__LINE__);
     if (t->Nf==1) {
     err=cudaFree(sId);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sQd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sUd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sVd);
     checkCudaError(err,__FILE__,__LINE__);
     }
     if (t->dobeam) {
      err=cudaFree(rad);
      checkCudaError(err,__FILE__,__LINE__);
      err=cudaFree(decd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     err=cudaFree(styped);
     checkCudaError(err,__FILE__,__LINE__);

     if (t->Nf>1) {
     err=cudaFree(sI0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(f0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idxd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx1d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(spec_idx2d);
     checkCudaError(err,__FILE__,__LINE__);

     err=cudaFree(sQ0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sU0d);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sV0d);
     checkCudaError(err,__FILE__,__LINE__);
     }
  }
/******************* end loop over clusters **************************/

  /* free memory */
  err=cudaFreeHost(tempdcoh);
  checkCudaError(err,__FILE__,__LINE__);

  err=cudaFree(ud);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(vd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(wd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(cohd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(freqsd);
  checkCudaError(err,__FILE__,__LINE__);
  
  if (t->dobeam) {
  err=cudaFree(longd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(latd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(timed);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(Nelemd);
  checkCudaError(err,__FILE__,__LINE__);

  for (ci=0; ci<t->N; ci++) {
    err=cudaFree(xx_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(yy_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(zz_p[ci]);
    checkCudaError(err,__FILE__,__LINE__);
  }

  err=cudaFree(xxd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(yyd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(zzd);
  checkCudaError(err,__FILE__,__LINE__);

  free(xx_p);
  free(yy_p);
  free(zz_p);
  }

  cudaDeviceSynchronize();

  /* reset error state */
  err=cudaGetLastError(); 

  return NULL;

}

int
precalculate_coherencies_multifreq_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double *freqs, int Nchan, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tilesz, int *Nelem, double **xx, double **yy, double **zz, int dobeam, int Nt) {

  int nth,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_pred_t *threaddata;
  taskhist thst;
  init_task_hist(&thst);

  int Ngpu=MAX_GPU_ID+1;

  /* calculate min clusters thread can handle */
  Nthb0=(M+Ngpu-1)/Ngpu;

  /* setup threads : note: Ngpu is no of GPUs used */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* set common parameters, and split clusters to threads */
  ci=0;
  for (nth=0;  nth<Ngpu && ci<M; nth++) {
     if (ci+Nthb0<M) {
      Nthb=Nthb0;
     } else {
      Nthb=M-ci;
     }
      
     threaddata[nth].hst=&thst;  /* for load balancing */
     threaddata[nth].tid=nth;
     
     threaddata[nth].u=u; 
     threaddata[nth].v=v;
     threaddata[nth].w=w;
     threaddata[nth].coh=x;
     threaddata[nth].x=0; /* no input data */

     threaddata[nth].N=N;
     threaddata[nth].Nbase=Nbase; /* total baselines: actually Nbasextilesz (from input) */
     threaddata[nth].barr=barr;
     threaddata[nth].carr=carr;
     threaddata[nth].M=M;
     threaddata[nth].Nf=Nchan;
     threaddata[nth].freqs=freqs; /* > 1 freq */
     threaddata[nth].fdelta=fdelta;
     threaddata[nth].tdelta=tdelta;
     threaddata[nth].dec0=dec0;

     threaddata[nth].dobeam=dobeam; 
     threaddata[nth].ph_ra0=ph_ra0;
     threaddata[nth].ph_dec0=ph_dec0;
     threaddata[nth].ph_freq0=ph_freq0;
     threaddata[nth].longitude=longitude;
     threaddata[nth].latitude=latitude;
     threaddata[nth].time_utc=time_utc;
     threaddata[nth].tilesz=tilesz;
     threaddata[nth].Nelem=Nelem;
     threaddata[nth].xx=xx;
     threaddata[nth].yy=yy;
     threaddata[nth].zz=zz;

     /* no parameters */
     threaddata[nth].p=0;

     threaddata[nth].Ns=Nthb;
     threaddata[nth].soff=ci;

    
     pthread_create(&th_array[nth],&attr,precalcoh_multifreq_threadfn,(void*)(&threaddata[nth]));
     /* next source set */
     ci=ci+Nthb;
  }

     

  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
  }



 free(threaddata);
 destroy_task_hist(&thst);
 free(th_array);


  thread_data_base_t *threaddata1;
  /* now do some book keeping (setting proper flags) using CPU only */
  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;
  
  if ((threaddata1=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }


  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
     /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
     /* determine actual no. of baselines */
     if (ci+Nthb0<Nbase) {
      Nthb=Nthb0;
     } else {
      Nthb=Nbase-ci;
     }

     threaddata1[nth].boff=ci;
     threaddata1[nth].Nb=Nthb;
     threaddata1[nth].barr=barr;
     threaddata1[nth].u=&(u[ci]);
     threaddata1[nth].v=&(v[ci]);
     threaddata1[nth].w=&(w[ci]);
     threaddata1[nth].uvmin=uvmin;
     threaddata1[nth].uvmax=uvmax;
     threaddata1[nth].freq0=freqs[0]; /* use lowest freq */

     pthread_create(&th_array[nth],&attr,resetflags_threadfn,(void*)(&threaddata1[nth]));

     /* next baseline set */
     ci=ci+Nthb;
  }


  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
   pthread_join(th_array[ci],NULL);
  }



 free(threaddata1);
 pthread_attr_destroy(&attr);
 free(th_array);

 return 0;
}
