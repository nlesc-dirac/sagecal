/*
 *
 Copyright (C) 2006-2026 Sarod Yatawatta <sarod@users.sf.net>  
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



#include "Dirac_radio.h"
#include "Dirac_GPUtune.h"

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
  int Nf; /* no of freqs in one MS */
  double *freqs; /*  Nfx1 freqs  for prediction */
  double fdelta; /* bandwidth */
  double tdelta; /* integration time */
  double dec0; /* phase center dec */

  /* following used in beam prediction */
  int dobeam; /* which part of beam to apply */
  int bf_type; /* station beam type: STAT_NONE, STAT_SINGLE or STAT_TILE */
  double b_ra0,b_dec0; /* tile beam pointing */
  double ph_ra0,ph_dec0; /* full beam pointing */
  double ph_freq0; /* beam central freq */
  double *longitude,*latitude; /* Nx1 array of station locations */
  double *time_utc; /* tileszx1 array of time */
  int tilesz;
  int *Nelem; /* Nx1 array of station sizes */
  double **xx,**yy,**zz; /* Nx1 arrays of station element coords,
   ci-th element will have a pointer to Nelem[ci] values (+HBA_TILE_SIZE depending on bf_type) */
  elementcoeff *ecoeff; /* element beam coefficients */

  int Ns; /* total no of sources (clusters) per thread */
  int soff; /* starting source for this thread */

  /* following are only used while predict with gain */
  double *p; /* parameter array, size could be 8*N*Mx1 (full) or 8*Nx1 (single)*/

  /* following only used to ignore some clusters while predicting */
  int *ignlist;

  /* following needed for consensus polynomial related stuff */
  double *Bpoly; /* Npoly x 1 basis functions */
  double *Binv; /* Mt x Npoly x Npoly inverse basis */
  double *rho; /* Mx1 regularization values */
  int Mt;
  int Npoly;
  int Nms; /* = all MS subbands != Nf */

} thread_data_pred_t;

static void *
model_residual_threadfn(void *data) {
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
  double *modeld;
  baseline_t *barrd;
  double *freqsd;
  float *longd=0,*latd=0; double *timed;
  int *Nelemd;
  float **xx_p=0,**yy_p=0,**zz_p=0;
  float **xxd,**yyd,**zzd;
  /* storage for element beam coefficients */
  float *pattern_phid=0, *pattern_thetad=0, *preambled=0;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(double)); /* C coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &modeld, t->Nbase*8*t->Nf*sizeof(double)); /* J C J^H model only for 1 cluster, Nf freq, used to store sum of clusters*/
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
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
  if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL || t->dobeam==DOBEAM_ELEMENT
      ||t->dobeam==DOBEAM_ARRAY_WB || t->dobeam==DOBEAM_FULL_WB || t->dobeam==DOBEAM_ELEMENT_WB) {
  dtofcopy(t->N,&longd,t->longitude);
  dtofcopy(t->N,&latd,t->latitude);
  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* convert time jd to GMST angle */
  cudakernel_convert_time(t->tilesz,timed);
  }
  if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL
      ||t->dobeam==DOBEAM_ARRAY_WB || t->dobeam==DOBEAM_FULL_WB) {
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);

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
    err=cudaMalloc((void**)&xx_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&yy_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMalloc((void**)&zz_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
  }
  /* now copy data */
  for (ci=0; ci<t->N; ci++) {
    dtofcopy(t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0),&xx_p[ci],t->xx[ci]);
    dtofcopy(t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0),&yy_p[ci],t->yy[ci]);
    dtofcopy(t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0),&zz_p[ci],t->zz[ci]);
  }
  /* now copy pointer locations to device */
  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  }
  if (t->dobeam==DOBEAM_ELEMENT || t->dobeam==DOBEAM_FULL
      ||t->dobeam==DOBEAM_ELEMENT_WB ||t->dobeam==DOBEAM_FULL_WB) {
  dtofcopy(2*t->ecoeff->Nmodes*t->ecoeff->Nf,&pattern_phid,(double*)t->ecoeff->pattern_phi);
  dtofcopy(2*t->ecoeff->Nmodes*t->ecoeff->Nf,&pattern_thetad,(double*)t->ecoeff->pattern_theta);
  dtofcopy(t->ecoeff->Nmodes,&preambled,t->ecoeff->preamble);
  }

  float *beamd; /* array beam */
  float *elementd; /* element beam */
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

  double *cohlocal;
  err=cudaMallocHost((void**)&cohlocal,sizeof(double)*(size_t)t->Nbase*8*t->Nf);
  checkCudaError(err,__FILE__,__LINE__);

  double *pd; /* parameter array per cluster */

/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
     /* we doe not check if cluster id >=0 to do a subtraction,
      * because we use the residual for influence calculation only */
     /* allocate memory for this clusters beam */
     if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL
         ||t->dobeam==DOBEAM_ARRAY_WB ||t->dobeam==DOBEAM_FULL_WB) {
      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
      checkCudaError(err,__FILE__,__LINE__);
     } else {
       beamd=0;
     }
     if (t->dobeam==DOBEAM_ELEMENT || t->dobeam==DOBEAM_FULL
         ||t->dobeam==DOBEAM_ELEMENT_WB ||t->dobeam==DOBEAM_FULL_WB) {
      err=cudaMalloc((void**)&elementd, t->N*8*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
      checkCudaError(err,__FILE__,__LINE__);
     } else {
       elementd=0;
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


     if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL || t->dobeam==DOBEAM_ELEMENT
         ||t->dobeam==DOBEAM_ARRAY_WB ||t->dobeam==DOBEAM_FULL_WB ||t->dobeam==DOBEAM_ELEMENT_WB) {
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


     if (t->dobeam==DOBEAM_ARRAY) {
      /* now calculate array beam for all sources in this cluster */
      if (t->bf_type==STAT_TILE) {
        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
      } else {
        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
      }
     } else if (t->dobeam==DOBEAM_ARRAY_WB) {
      if (t->bf_type==STAT_TILE) {
        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
      } else {
        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
      }
     } else if (t->dobeam==DOBEAM_ELEMENT) {
      /* calculate element beam for all sources in this cluster */
      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
     } else if (t->dobeam==DOBEAM_ELEMENT_WB) {
      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
     } else if (t->dobeam==DOBEAM_FULL) {
      /* calculate array+element beam for all sources in this cluster */
      if (t->bf_type==STAT_TILE) {
        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
      } else {
        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
      }
      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
     } else if (t->dobeam==DOBEAM_FULL_WB) {
      if (t->bf_type==STAT_TILE) {
        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
      } else {
        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
      }
      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
     }


     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_coherencies_and_residuals(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,pd,t->carr[ncl].nchunk,barrd,freqsd,beamd, elementd,
     lld,mmd,nnd,sId,sQd,sUd,sVd,styped,sI0d,sQ0d,sU0d,sV0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,t->fdelta,t->tdelta,t->dec0,modeld,cohd,t->dobeam);
    
     /* copy back coherencies to host, for the specific cluster 
      * also copy back model to calculate residual */
     err=cudaMemcpy(xlocal, modeld, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     my_daxpy(t->Nbase*8*t->Nf,xlocal,1.0,t->x);
     err=cudaMemcpy(cohlocal, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     my_dcopy(t->Nbase*8*t->Nf,cohlocal,1,(double*)&t->coh[ncl*t->Nbase*4*t->Nf],1);

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

     if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL
         ||t->dobeam==DOBEAM_ARRAY_WB ||t->dobeam==DOBEAM_FULL_WB) {
      err=cudaFree(beamd);
      checkCudaError(err,__FILE__,__LINE__);
     }
     if (t->dobeam==DOBEAM_ELEMENT || t->dobeam==DOBEAM_FULL
         ||t->dobeam==DOBEAM_ELEMENT_WB ||t->dobeam==DOBEAM_FULL_WB) {
      err=cudaFree(elementd);
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
     if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL || t->dobeam==DOBEAM_ELEMENT
         ||t->dobeam==DOBEAM_ARRAY_WB ||t->dobeam==DOBEAM_FULL_WB ||t->dobeam==DOBEAM_ELEMENT_WB) {
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
  err=cudaFreeHost(cohlocal);
  checkCudaError(err,__FILE__,__LINE__);
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
  err=cudaFree(modeld);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(freqsd);
  checkCudaError(err,__FILE__,__LINE__);

  if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL || t->dobeam==DOBEAM_ELEMENT
      ||t->dobeam==DOBEAM_ARRAY_WB ||t->dobeam==DOBEAM_FULL_WB ||t->dobeam==DOBEAM_ELEMENT_WB) {
  err=cudaFree(longd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(latd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(timed);
  checkCudaError(err,__FILE__,__LINE__);
  }
  if (t->dobeam==DOBEAM_ARRAY || t->dobeam==DOBEAM_FULL
      ||t->dobeam==DOBEAM_ARRAY_WB ||t->dobeam==DOBEAM_FULL_WB) {
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
  if (t->dobeam==DOBEAM_ELEMENT || t->dobeam==DOBEAM_FULL
      ||t->dobeam==DOBEAM_ELEMENT_WB ||t->dobeam==DOBEAM_FULL_WB) {
   err=cudaFree(pattern_phid);
   checkCudaError(err,__FILE__,__LINE__);
   err=cudaFree(pattern_thetad);
   checkCudaError(err,__FILE__,__LINE__);
   err=cudaFree(preambled);
   checkCudaError(err,__FILE__,__LINE__);
  }


  cudaDeviceSynchronize();
  /* reset error state */
  err=cudaGetLastError(); 
  return NULL;

}

static void *
hessian_influence_threadfn(void *data) {
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
  int ncl;


  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  float *resd=0;
  float *hessd;
  baseline_t *barrd;

  dtofcopy(t->Nbase*8*t->Nf,&resd,t->x);

  /* Hessian for one cluster, 4N x 4N complex matrix */
  err=cudaMalloc((void**) &hessd, t->N*4*t->N*4*2*sizeof(float)); /* Hessian for one cluster */
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);

  float *hess;
  err=cudaMallocHost((void**)&hess,sizeof(float)*(size_t)t->N*4*t->N*4*2);
  checkCudaError(err,__FILE__,__LINE__);
  /* copy data */
  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  double *pd; /* parameter array per cluster */

  /* calculate Hessian addition (common part), based only on the polynomial basis*/
  int hess_add_flag=(t->rho && t->Bpoly && t->Binv ? 1: 0);

  /* storage to calculate (Jq C^H)^T -> p-th block, 4NxNbase x 2 (re,im) x 8 (for 8 values of 
   * 2x2 complex matrix) */
  float *AdVd=0;
  err=cudaMalloc((void**) &AdVd, t->N*4*t->Nbase*2*8*sizeof(float)); /* Hessian for one cluster */
  checkCudaError(err,__FILE__,__LINE__);
 
/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
    float *cohd=0;
    /* note dtofcopy() will allocate cohd */
    dtofcopy(t->Nbase*8*t->Nf,&cohd,(double*)&t->coh[ncl*t->Nbase*4*t->Nf]);

    /* initialize hessian to 0 */
    err=cudaMemset(hessd, 0, t->N*4*t->N*4*2*sizeof(float));
    checkCudaError(err,__FILE__,__LINE__);

    /* parameter vector size may change depending on hybrid parameter */
    err=cudaMalloc((void**) &pd, t->N*8*t->carr[ncl].nchunk*sizeof(double));
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaMemcpy(pd, &(t->p[t->carr[ncl].p[0]]), t->N*8*t->carr[ncl].nchunk*sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError(err,__FILE__,__LINE__);


    /* run kernel, which will calculate hessian for this cluster */
    cudakernel_hessian(t->Nbase,t->N,t->tilesz,t->Nf,barrd,pd,t->carr[ncl].nchunk,cohd,resd,hessd);

    /* copy result back */
    err=cudaMemcpy(hess,hessd,sizeof(float)*(size_t)t->N*4*t->N*4*2,cudaMemcpyDeviceToHost);
    checkCudaError(err,__FILE__,__LINE__);

    /* also add component based on spectral basis to the Hessian */
    /* this part done on the CPU */
    if (hess_add_flag) {
      /* code at analysis_uvwdir.m ln 170-180 */
     /* Bpoly: 1 x Npoly, row vector
      * Bf = kron(Bpoly, I_2N) : 2N x 2N*Npoly 
     * P = kron(Binv, I_2N) x Bf^T : (2N*Npoly x 2N*Npoly) (2N*Npoly x 2N): 2N*Npoly x 2*N 
     * BfP= Bf x P : 2N x 2N,
    * BfP=kron(Bpoly,I_2N) x kron(Binv, I_2N) x kron(Bpoly,I_2N)^T
    * =kron(BpolyxBinvxBpoly^T,I_2N) : 2N x 2N
    * F = I_2N - BfP : 2N x 2N, note that rho does not appear (cancel out) 
    so we only need to calculate Bpoly x Binv x Bpoly^T in full matrix operations */
     double *Bibf;
     if ((Bibf=(double*)calloc((size_t)t->Npoly,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     /* Binv x Bpoly^T : at ncl offset, might differ if M!=Mt, but since same basis is used for all directions, ok */
     my_dgemv('N',t->Npoly,t->Npoly,1.0,&t->Binv[ncl*t->Npoly*t->Npoly],t->Npoly,t->Bpoly,1,0.0,Bibf,1);
     /* Bpoly x (Binv x Bpoly) */
     double bfBibf=my_ddot(t->Npoly,t->Bpoly,Bibf);
     /* diagonal value of F=I_2N-BfP */
     double Fd=1.0-bfBibf;
     /* diagonal of F^H F (2N values)*/
     double Fdd = Fd*Fd;
     /* F'*F*(I_2N + pinv(I_2N-F'*F)*F'*F) (2N values) */
     double Fd1=Fdd*(1+Fdd/(1-Fdd));
     /* hessian addition = 0.5*rho*kron(I_2,diag(Fd1)) (4N x 4N) */
     /* = 2x2N diagonal terms =0.5*rho*Fd1 */

     float hfactor=(float)0.5*t->rho[ncl]*Fd1;
     printf("clus %d base=%d stat=%d tile=%d freq=%d Fdd %lf Hadd %f\n",ncl,t->Nbase,t->N,t->tilesz,t->Nf,Fd1,hfactor);
     free(Bibf);

     /* add to diagonal of hessian (real part) */
     for (int ci=0; ci<4*t->N; ci++) {
       hess[ci*2*4*t->N+ci*2]+=hfactor;
     }
    }


    /* per cluster, Dsolutions_uvw(), fill matrix AdV (4N x B),
     * by adding (J_q C^H)^T at p-th column block for baseline p-q */
     cudakernel_d_solutions(t->Nbase,t->N,t->tilesz,t->Nf,barrd,pd,t->carr[ncl].nchunk,cohd,AdVd);
  /* my_cgels() to find inv(Hessian) (AdV) */

  /* accumulate Dresidual_uvw() for this cluster in t->x of size t->Nbase*8*t->Nf */

    err=cudaFree(cohd);
    checkCudaError(err,__FILE__,__LINE__);
    err=cudaFree(pd);
    checkCudaError(err,__FILE__,__LINE__);
  }
/******************* end loop over clusters **************************/


  err=cudaFree(resd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(hessd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(barrd);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFreeHost(hess);
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaFree(AdVd);
  checkCudaError(err,__FILE__,__LINE__);

  cudaDeviceSynchronize();
  /* reset error state */
  err=cudaGetLastError(); 
  return NULL;
}

/* p: 8NMtx1 parameter array, but Mt is 'effective' clusters, need to use carr to find the right offset, note M<= Mt
 * rho: Mx1 regularization values =NULL for calibration without consensus
 * Bpoly: Npoly x 1 basis functions (to match the frequency) =NULL for calibration without consensus
 * Bi: Mt x Npoly x Npoly inverse basis =NULL for calibration without consensus
 * Nf != Nchan, Nf: total freqs, Nchan: channels for one MS
*/
int
calculate_diagnostics_gpu(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int dobeam, double *rho, double *Bpoly, double *Bi, int Mt, int Npoly, int Nf, int Nt) {

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

  complex double *coh;
  printf("Coherencies baselines=%d tilesize=%d chan=%d GPU=%d clus=%d trueclus=%d\n",Nbase,tilesz,Nchan,Ngpu,M,Mt);
  if ((coh=(complex double*)calloc((size_t)Nbase*4*tilesz*Nchan*M,sizeof(complex double)))==0) {
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
     // Note: pass coherencies without any offset, as the offset
     // will be determined by the cluster x (Nbase*8*tilesz*Nchan)
     threaddata[nth].coh=coh;
     threaddata[nth].x=&xlocal[nth*Nbase*8*tilesz*Nchan]; /* distinct arrays to get back the result */

     threaddata[nth].N=N;
     threaddata[nth].Nbase=Nbase*tilesz; /* total baselines: actually Nbasextilesz */
     threaddata[nth].barr=barr;
     threaddata[nth].carr=carr;
     threaddata[nth].M=M;
     threaddata[nth].Nf=Nchan;
     threaddata[nth].freqs=freqs;
     threaddata[nth].fdelta=fdelta/(double)Nchan;
     threaddata[nth].tdelta=tdelta;
     threaddata[nth].dec0=dec0;

     threaddata[nth].dobeam=dobeam;
     threaddata[nth].bf_type=bf_type;
     threaddata[nth].b_ra0=b_ra0;
     threaddata[nth].b_dec0=b_dec0;
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

     threaddata[nth].ecoeff=ecoeff;

     threaddata[nth].rho=rho;
     threaddata[nth].Bpoly=Bpoly;
     threaddata[nth].Binv=Bi;
     threaddata[nth].Mt=Mt;
     threaddata[nth].Npoly=Npoly;
     threaddata[nth].Nms=Nf;

     pthread_create(&th_array[nth],&attr,model_residual_threadfn,(void*)(&threaddata[nth]));
     /* next source set */
     ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
    /* subtract to find residual */
    my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,-1.0,x);
  }

  /* loop over clusters */
  for(ci=0; ci<nth; ci++) {
     /* copy residual back to each thread */
     my_dcopy(Nbase*8*tilesz*Nchan,x,1,threaddata[ci].x,1);
     pthread_create(&th_array[ci],&attr,hessian_influence_threadfn,(void*)(&threaddata[ci]));
  }

  /* reset x to zero */
  memset(x,0,sizeof(double)*Nbase*8*tilesz*Nchan);

  for(ci=0; ci<nth; ci++) {
    pthread_join(th_array[ci],NULL);
    /* accumulate result of each thread back to residual column */
    my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,1.0,x);
  }

  free(coh);
  free(xlocal);
  free(threaddata);

  free(th_array);
  pthread_attr_destroy(&attr);

  destroy_task_hist(&thst);

  return 0;
}
