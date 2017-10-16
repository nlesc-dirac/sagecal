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

} thread_data_pred_t;


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
  err=cudaMemcpyAsync(xc,xhost,N*sizeof(float),cudaMemcpyHostToDevice,0);
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
   card=t->tid;
  }
  cudaError_t err;
  int ci,ncl,cj;

  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  float *ud,*vd,*wd,*cohd;
  baseline_t *barrd;
  float *freqsd;

  float *longd,*latd; double *timed, *copyoftimed;
  // This is needed to write times to file. 
  // Because of the convert_time applied to timed, we cannot use t->time_utc as input to fwrite.
  if ((copyoftimed=(double*)calloc((size_t) t->tilesz, sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }

  int *Nelemd;
  float **xx_p,**yy_p,**zz_p;
  float **xxd,**yyd,**zzd;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nbase*8*sizeof(float)); /* coherencies only for 1 cluster, Nf=1 */
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  
  /* copy to device */
  dtofcopy(t->Nbase,&ud,t->u);
  dtofcopy(t->Nbase,&vd,t->v);
  dtofcopy(t->Nbase,&wd,t->w);
  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  dtofcopy(t->Nf,&freqsd,t->freqs);
  dtofcopy(t->N,&longd,t->longitude);
  dtofcopy(t->N,&latd,t->latitude);
  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);
  /* convert time jd to GMST angle */
  cudakernel_convert_time(t->tilesz,timed);

  // Fill copyoftimed.
  err=cudaMemcpy((double*)copyoftimed, timed, t->tilesz*sizeof(double), cudaMemcpyDeviceToHost);

  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  /* temp host storage to copy coherencies */
  complex float *tempcoh;
  if ((tempcoh=(complex float*)calloc((size_t)t->Nbase*4,sizeof(complex float)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }

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

  float *beamd;
  /* temp host storage for beam */
  float *tempbeam;

  float *lld,*mmd,*nnd,*sId,*rad,*decd;
  unsigned char *styped;
  float *sI0d,*f0d,*spec_idxd,*spec_idx1d,*spec_idx2d;
  int **host_p,**dev_p;
/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
     /* allocate memory for this clusters beam */
     err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
     checkCudaError(err,__FILE__,__LINE__);

     /* copy cluster details to GPU */
     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
     checkCudaError(err,__FILE__,__LINE__);

     dtofcopy(t->carr[ncl].N,&lld,t->carr[ncl].ll);
     dtofcopy(t->carr[ncl].N,&mmd,t->carr[ncl].mm);
     dtofcopy(t->carr[ncl].N,&nnd,t->carr[ncl].nn);
     dtofcopy(t->carr[ncl].N,&sId,t->carr[ncl].sI);
     dtofcopy(t->carr[ncl].N,&rad,t->carr[ncl].ra);
     dtofcopy(t->carr[ncl].N,&decd,t->carr[ncl].dec);
     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* for multi channel data */
     dtofcopy(t->carr[ncl].N,&sI0d,t->carr[ncl].sI0);
     dtofcopy(t->carr[ncl].N,&f0d,t->carr[ncl].f0);
     dtofcopy(t->carr[ncl].N,&spec_idxd,t->carr[ncl].spec_idx);
     dtofcopy(t->carr[ncl].N,&spec_idx1d,t->carr[ncl].spec_idx1);
     dtofcopy(t->carr[ncl].N,&spec_idx2d,t->carr[ncl].spec_idx2);

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

     if ((tempbeam=(float*)calloc((size_t) t->N*t->tilesz*t->carr[ncl].N*t->Nf,sizeof(float)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
     }

     /* now copy pointer locations to device */
     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     FILE *t_N;
     t_N=fopen("t_N.bin","wb");
     fwrite(&t->N, sizeof(int), sizeof(t->N)/sizeof(int), t_N);
     fclose(t_N);

     FILE *t_tilesz;
     t_tilesz=fopen("t_tilesz.bin","wb");
     fwrite(&t->tilesz, sizeof(int), sizeof(t->tilesz)/sizeof(int), t_tilesz);
     fclose(t_tilesz);

     FILE *t_carr_ncl_N;
     t_carr_ncl_N=fopen("t_carr_ncl_N.bin","wb");
     fwrite(&t->carr[ncl].N, sizeof(int), sizeof(t->carr[ncl].N)/sizeof(int), t_carr_ncl_N);
     fclose(t_carr_ncl_N);

     FILE *t_Nf;
     t_Nf=fopen("t_Nf.bin","wb");
     fwrite(&t->Nf, sizeof(int), sizeof(t->Nf)/sizeof(int), t_Nf);
     fclose(t_Nf);

     FILE *freq_sd;
     freq_sd=fopen("freq_sd.bin","wb");
     fwrite(t->freqs, sizeof(double), t->Nf, freq_sd);
     fclose(freq_sd);

     FILE *long_d;
     long_d=fopen("long_d.bin","wb");
     fwrite(t->longitude, sizeof(double), t->N, long_d);
     fclose(long_d);

     FILE *lat_d;
     lat_d=fopen("lat_d.bin","wb");
     fwrite(t->latitude, sizeof(double), t->N, lat_d);
     fclose(lat_d);

     FILE *time_d;
     time_d=fopen("time_d.bin","wb");
     fwrite(&copyoftimed, sizeof(double), t->tilesz, time_d);
     fclose(time_d);

     FILE *Nelem_d;
     Nelem_d=fopen("Nelem_d.bin","wb");
     fwrite(t->Nelem, sizeof(int), t->N, Nelem_d);
     fclose(Nelem_d);

     FILE *xx_d;
     xx_d=fopen("xx_d.bin","wb");
     int j;
     for (j=0; j<t->N; j++){
         fwrite(t->xx[j], sizeof(double), t->Nelem[j], xx_d);
     }
     fclose(xx_d);

     FILE *yy_d;
     yy_d=fopen("yy_d.bin","wb");
     for (j=0; j<t->N; j++){
         fwrite(t->yy[j], sizeof(double), t->Nelem[j], yy_d);
     }
     fclose(yy_d);

     FILE *zz_d;
     zz_d=fopen("zz_d.bin","wb");
     for (j=0; j<t->N; j++){
         fwrite(t->zz[j], sizeof(double), t->Nelem[j], zz_d);
     }
     fclose(zz_d);

     FILE *ra_d;
     ra_d=fopen("ra_d.bin","wb");
     fwrite(t->carr[ncl].ra, t->carr[ncl].N, sizeof(double), ra_d);
     fclose(ra_d);

     FILE *dec_d;
     dec_d=fopen("dec_d.bin","wb");
     fwrite(t->carr[ncl].dec, t->carr[ncl].N, sizeof(double), dec_d);
     fclose(dec_d);

     FILE *t_ph_ra0;
     t_ph_ra0=fopen("t_ph_ra0.bin","wb");
     fwrite((double *)&t->ph_ra0, 1, sizeof(double), t_ph_ra0);
     fclose(t_ph_ra0);

     FILE *t_ph_dec0;
     t_ph_dec0=fopen("t_ph_dec0.bin","wb");
     fwrite((double *)&t->ph_dec0, 1, sizeof(double), t_ph_dec0);
     fclose(t_ph_dec0);

     FILE *t_ph_freq0;
     t_ph_freq0=fopen("t_ph_freq0.bin","wb");
     fwrite((double *)&t->ph_freq0, 1, sizeof(double), t_ph_freq0);
     fclose(t_ph_freq0);

     /* now calculate beam for all sources in this cluster */
     cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd);

     FILE *beam_d;
     beam_d=fopen("beam_d.bin","wb");
     err=cudaMemcpy((float*)tempbeam, beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float), cudaMemcpyDeviceToHost);

     for (j=0; j<t->N*t->tilesz*t->carr[ncl].N*t->Nf; j++){
       fwrite(&tempbeam[j], sizeof(float), 1, beam_d); 
     }
     fclose(beam_d);

     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd,
     lld,mmd,nnd,sId,styped,sI0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,(float)t->fdelta,(float)t->tdelta,(float)t->dec0,cohd,t->dobeam);
    
     /* copy back coherencies to host, 
        coherencies on host have 8M stride, on device have 8 stride */
     err=cudaMemcpy((float*)tempcoh, cohd, sizeof(float)*t->Nbase*8, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     complex double *tempdcoh;
     if ((tempdcoh=(complex double*)calloc((size_t)t->Nbase*4,sizeof(complex double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
     }
     int di;
     double *dcp=(double*)tempdcoh;
     float *fcp=(float*)tempcoh;
     for (di=0; di<t->Nbase; di++) {
       dcp[8*di]=(double)fcp[8*di];
       dcp[8*di+1]=(double)fcp[8*di+1];
       dcp[8*di+2]=(double)fcp[8*di+2];
       dcp[8*di+3]=(double)fcp[8*di+3];
       dcp[8*di+4]=(double)fcp[8*di+4];
       dcp[8*di+5]=(double)fcp[8*di+5];
       dcp[8*di+6]=(double)fcp[8*di+6];
       dcp[8*di+7]=(double)fcp[8*di+7];
     }
     /* now copy this with right offset and stride */
     my_ccopy(t->Nbase,&tempdcoh[0],4,&(t->coh[4*ncl]),4*t->M);
     my_ccopy(t->Nbase,&tempdcoh[1],4,&(t->coh[4*ncl+1]),4*t->M);
     my_ccopy(t->Nbase,&tempdcoh[2],4,&(t->coh[4*ncl+2]),4*t->M);
     my_ccopy(t->Nbase,&tempdcoh[3],4,&(t->coh[4*ncl+3]),4*t->M);
     free(tempdcoh);


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


     err=cudaFree(beamd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(lld);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(mmd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(nnd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sId);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(rad);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(decd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(styped);
     checkCudaError(err,__FILE__,__LINE__);

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
  }
/******************* end loop over clusters **************************/

  free(tempcoh);
  free(tempbeam);
  free(copyoftimed);

  /* free memory */
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

  int Ngpu;
  if (M<4) {
   Ngpu=2;
  } else {
   Ngpu=4;
  }

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
     threaddata[nth].x=0; /* no data */

     threaddata[nth].N=N;
     threaddata[nth].Nbase=Nbase; /* total baselines: actually Nbasextilesz */
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
   card=t->tid;
  }
  cudaError_t err;
  int ci,ncl,cj;

  err=cudaSetDevice(card);
  checkCudaError(err,__FILE__,__LINE__);

  float *ud,*vd,*wd,*cohd;
  baseline_t *barrd;
  float *freqsd;
  float *longd,*latd; double *timed;
  int *Nelemd;
  float **xx_p,**yy_p,**zz_p;
  float **xxd,**yyd,**zzd;
  /* allocate memory in GPU */
  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(float)); /* coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
  checkCudaError(err,__FILE__,__LINE__);
  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
  checkCudaError(err,__FILE__,__LINE__);
  
  /* copy to device */
  dtofcopy(t->Nbase,&ud,t->u);
  dtofcopy(t->Nbase,&vd,t->v);
  dtofcopy(t->Nbase,&wd,t->w);
  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
  checkCudaError(err,__FILE__,__LINE__);

  dtofcopy(t->Nf,&freqsd,t->freqs);
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


  float *beamd;
  float *lld,*mmd,*nnd,*sId,*rad,*decd;
  unsigned char *styped;
  float *sI0d,*f0d,*spec_idxd,*spec_idx1d,*spec_idx2d;
  int **host_p,**dev_p;
/******************* begin loop over clusters **************************/
  for (ncl=t->soff; ncl<t->soff+t->Ns; ncl++) {
     /* allocate memory for this clusters beam */
     err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
     checkCudaError(err,__FILE__,__LINE__);


     /* copy cluster details to GPU */
     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
     checkCudaError(err,__FILE__,__LINE__);

     dtofcopy(t->carr[ncl].N,&lld,t->carr[ncl].ll);
     dtofcopy(t->carr[ncl].N,&mmd,t->carr[ncl].mm);
     dtofcopy(t->carr[ncl].N,&nnd,t->carr[ncl].nn);
     dtofcopy(t->carr[ncl].N,&sId,t->carr[ncl].sI);
     dtofcopy(t->carr[ncl].N,&rad,t->carr[ncl].ra);
     dtofcopy(t->carr[ncl].N,&decd,t->carr[ncl].dec);
     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
     checkCudaError(err,__FILE__,__LINE__);

     /* for multi channel data */
     dtofcopy(t->carr[ncl].N,&sI0d,t->carr[ncl].sI0);
     dtofcopy(t->carr[ncl].N,&f0d,t->carr[ncl].f0);
     dtofcopy(t->carr[ncl].N,&spec_idxd,t->carr[ncl].spec_idx);
     dtofcopy(t->carr[ncl].N,&spec_idx1d,t->carr[ncl].spec_idx1);
     dtofcopy(t->carr[ncl].N,&spec_idx2d,t->carr[ncl].spec_idx2);

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


     /* now calculate beam for all sources in this cluster */
     cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd);


     /* calculate coherencies for all sources in this cluster, add them up */
     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd,
     lld,mmd,nnd,sId,styped,sI0d,f0d,spec_idxd,spec_idx1d,spec_idx2d,dev_p,(float)t->fdelta,(float)t->tdelta,(float)t->dec0,cohd,t->dobeam);
    
     /* copy back coherencies to host, 
        coherencies on host have 8M stride, on device have 8 stride */
     float *tempx;
     if ((tempx=(float*)calloc((size_t)t->Nbase*8*t->Nf,sizeof(float)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     err=cudaMemcpy(tempx, cohd, sizeof(float)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
     checkCudaError(err,__FILE__,__LINE__);
     /* copy back as double */
     int di;
     for (di=0; di<t->Nbase*t->Nf; di++) {
      t->x[8*di]=(double)tempx[8*di];
      t->x[8*di+1]=(double)tempx[8*di+1];
      t->x[8*di+2]=(double)tempx[8*di+2];
      t->x[8*di+3]=(double)tempx[8*di+3];
      t->x[8*di+4]=(double)tempx[8*di+4];
      t->x[8*di+5]=(double)tempx[8*di+5];
      t->x[8*di+6]=(double)tempx[8*di+6];
      t->x[8*di+7]=(double)tempx[8*di+7];
     }
     free(tempx);


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


     err=cudaFree(beamd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(lld);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(mmd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(nnd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(sId);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(rad);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(decd);
     checkCudaError(err,__FILE__,__LINE__);
     err=cudaFree(styped);
     checkCudaError(err,__FILE__,__LINE__);

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
  }
/******************* end loop over clusters **************************/

  /* free memory */
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

  int Ngpu;
  if (M<4) {
   Ngpu=2;
  } else {
   Ngpu=4;
  }

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

  if (!add_to_data) {
   /* set output column to zero */
   memset(x,0,sizeof(double)*Nbase*8*tilesz*Nchan);
  }



  /* set common parameters, and split baselines to threads */
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
    my_daxpy(Nbase*8*tilesz*Nchan,threaddata[ci].x,1.0,x);
  }



  free(xlocal);
  free(threaddata);
  destroy_task_hist(&thst);
  free(th_array);
  pthread_attr_destroy(&attr);

  return 0;
}
