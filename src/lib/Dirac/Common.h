/*
 *
 Copyright (C) 2018 Sarod Yatawatta <sarod@users.sf.net>  
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

#ifndef Common_H
#define Common_H
#ifdef __cplusplus
        extern "C" {
#endif

/* simulation options */
#define SIMUL_ONLY 1 /* only predict model */
#define SIMUL_ADD 2 /* add to input */
#define SIMUL_SUB 3 /* subtract from input */

/* structures to store extra source info for extended sources */
typedef struct exinfo_gaussian_ {
  double eX,eY,eP; /* major,minor,PA */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_gaussian;

typedef struct exinfo_disk_ {
  double eX; /* diameter */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_disk;

typedef struct exinfo_ring_ {
  double eX; /* diameter */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_ring;

typedef struct exinfo_shapelet_ {
  int n0; /* model order, no of modes=n0*n0 */
  double beta; /* scale*/
  double *modes; /* array of n0*n0 x 1 values */
  double eX,eY,eP; /* linear transform parameters */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_shapelet;


/* when to project l,m coordinates */
#ifndef PROJ_CUT
#define PROJ_CUT 0.998
#endif


/* struct for a cluster GList item */
typedef struct clust_t_{
 int id; /* cluster id */
 int nchunk; /* no of chunks the data is divided for solving */
 GList *slist; /* list of sources in this cluster (string)*/
} clust_t;

typedef struct clust_n_{
 char *name; /* source name (string)*/
} clust_n;

/* struct to store source info in hash table */
typedef struct sinfo_t_ {
 double ll,mm,ra,dec,sI[4]; /* sI:4x1 for I,Q,U,V, note sI is updated for central freq (ra,dec) for Az,El */
 unsigned char stype; /* source type */
 void *exdata; /* pointer to carry additional data, if needed */
 double sI0[4],f0,spec_idx,spec_idx1,spec_idx2; /* for multi channel data, original sI,Q,U,V, f0 and spectral index */
} sinfo_t;

/* struct for array of the sky model, with clusters */
typedef struct clus_source_t_ {
 int N; /* no of source in this cluster */
 int id; /* cluster id */
 double *ll,*mm,*nn,*sI,*sQ,*sU,*sV; /* arrays Nx1 of source info, note: sI is at reference freq of data */
 /* nn=sqrt(1-ll^2-mm^2)-1 */
 double *ra,*dec; /* arrays Nx1 for Az,El calculation */
 unsigned char *stype; /* source type array Nx1 */
 void **ex; /* array for extra source information Nx1 */

 int nchunk; /* no of chunks the data is divided for solving */
 int *p; /* array nchunkx1 points to parameter array indices */


 double *sI0,*sQ0,*sU0,*sV0,*f0,*spec_idx,*spec_idx1,*spec_idx2; /* for multi channel data, original sI, f0 and spectral index */
} clus_source_t;

/* strutct to store baseline to station mapping */
typedef struct baseline_t_ {
 int sta1,sta2;
 unsigned char flag; /* if this baseline is flagged, set to 1, otherwise 0: 
             special case: 2 if baseline is not used in solution, but will be
              subtracted */
} baseline_t;


/* structure for worker threads for various function calculations */
typedef struct thread_data_base_ {
  int Nb; /* no of baselines this handle */
  int boff; /* baseline offset per thread */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *u,*v,*w; /* pointers to uwv arrays,size Nbx1 */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  double *x; /* output vector Nbx8 array re,im,re,im .... */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  /* following are only used while predict with gain */
  double *p; /* parameter array, size could be 8*N*Mx1 (full) or 8*Nx1 (single)*/
  int N; /* no of stations */
  int clus; /* which cluster to process, 0,1,...,M-1 if -1 all clusters */
  double uvmin; /* baseline length sqrt(u^2+v^2) lower limit, below this is not 
                 included in calibration, but will be subtracted */
  double uvmax;
  /* following used for freq/time smearing calculation */
  double freq0;
  double fdelta;
  double tdelta; /* integration time for time smearing */
  double dec0; /* declination for time smearing */

  /* following used for interpolation */
  double *p0; /* old parameters, same as p */
  int tilesz; /* tile size */
  int Nbase; /* total no of baselines */
  /* following for correction of data */
  double *pinv; /* inverted solution array, if null no correction */
  int ccid; /* which cluster id (not user specified id) for correction, >=0 */

  /* following for ignoring clusters in simulation */
  int *ignlist; /* Mx1 array, if any value 1, that cluster will not be simulated */
  /* flag for adding/subtracting model to data */
  int add_to_data; /* see SIMUL* defs */

  /* following used for multifrequency (channel) data */
  double *freqs;
  int Nchan;

  /* following used for calculating beam */
  double *arrayfactor; /* storage for precomputed beam */
  /* if clus==0, reset memory before adding */

} thread_data_base_t;

/* structure for worker threads for 
   precalculating beam array factor */
typedef struct thread_data_arrayfac_ {
  int Ns; /* total no of sources per thread */
  int soff; /* starting source */
  int Ntime; /* total timeslots */
  double *time_utc; /* Ntimex1 array */
  int N; /* no. of stations */
  double *longitude, *latitude;

  double ra0,dec0,freq0; /* reference pointing and freq */
  int Nf; /* no. of frequencies to calculate */
  double *freqs; /* Nfx1 array */

  int *Nelem; /* Nx1 array of element counts */
  double **xx,**yy,**zz; /* Nx1 arrays to element coords of each station, size Nelem[]x1 */
  
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int cid; /* cluster id to calculate beam */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *beamgain; /* output */
} thread_data_arrayfac_t;


/* structure for worker threads for presetting
   flagged data before solving */
typedef struct thread_data_preflag_ {
  int Nbase; /* total no of baselines */
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *x; /* data */
  double *flag; /* flag array 0 or 1 */
} thread_data_preflag_t;


/* structure for worker threads for arranging coherencies for GPU use */
typedef struct thread_data_coharr_ {
  int M; /* no of clusters */
  int Nbase; /* no of baselines */
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  double *ddcoh; /* coherencies, rearranged for easy copying to GPU, also real,imag instead of complex */
  short *ddbase; /* baseline to station maps, same as barr, assume no of stations < 32k, if flagged set to -1 OR (sta1,sta2,flag) 3 values for each baseline */
} thread_data_coharr_t;

/* structure for worker threads for type conversion */
typedef struct thread_data_typeconv_{
  int starti; /* starting baseline */
  int endi; /* ending baseline */
  double *darr; /* double array */
  float *farr; /* float array */
} thread_data_typeconv_t;

/* structure for worker threads for baseline generation */
typedef struct thread_data_baselinegen_{
  int starti; /* starting tile */
  int endi; /* ending tile */
  baseline_t *barr; /* baseline array */
  int N; /* stations */
  int Nbase; /* baselines */
} thread_data_baselinegen_t;

/* structure for counting baselines for each station (RTR)*/
typedef struct thread_data_count_ {
 int Nb; /* no of baselines this handle */
 int boff; /* baseline offset per thread */

 short *ddbase;

 int *bcount;

  /* mutexs: N x 1, one for each station */
  pthread_mutex_t *mx_array;
} thread_data_count_t;


/* structure for initializing an array */
typedef struct thread_data_setwt_ {
 int Nb; /* no of baselines this handle */
 int boff; /* baseline offset per thread */

 double *b;
 double a;

} thread_data_setwt_t;

/* structure for weight calculation for baselines */
typedef struct thread_data_baselinewt_ {
 int Nb; /* no of baselines this handle */
 int boff; /* baseline offset per thread */

 double *wt; /* 8 values per baseline */
 double *u,*v;
 double freq0;

} thread_data_baselinewt_t;



/* structure for worker threads for jacobian calculation */
typedef struct thread_data_jac_ {
  int Nb; /* no of baselines this handle */
  int n; /* function dimension n=8*Nb  is implied */
  int m; /* no of parameters */
  baseline_t *barr; /* pointer to baseline-> stations mapping array */
  double *u,*v,*w; /* pointers to uwv arrays,size Nbx1 */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  double *jac; /* output jacobian Nbx8 rows, re,im,re,im .... */
  complex double *coh; /* output vector in complex form, (not used always) size 4*M*Nb */
  /* following are only used while predict with gain */
  double *p; /* parameter array, size could be 8*N*Mx1 (full) or 8*Nx1 (single)*/
  int N; /* no of stations */
  int clus; /* which cluster to process, 0,1,...,M-1 if -1 all clusters */
  int start_col;
  int end_col; /* which column of jacobian we calculate */
} thread_data_jac_t;


/* structure for levmar */
typedef struct me_data_t_ {
  int clus; /* which cluster 0,1,...,M-1 if -1 all clusters */
  double *u,*v,*w; /* uvw coords size Nbase*tilesz x 1 */
  int Nbase; /* no of baselines */
  int tilesz; /* tile size */
  int N; /* no of stations */
  baseline_t *barr; /* baseline->station mapping, size Nbase*tilesz x 1 */
  clus_source_t *carr; /* sky model, with clusters size Mx1 */
  int M; /* no of clusters */
  int Mt; /* apparent no of clusters, due to hybrid solving, Mt>=M */
  double *freq0; /* frequency */
  int Nt; /* no of threads */

  complex double *coh; /* pre calculated cluster coherencies, per cluster 4xNbase values, total size 4*M*Nbase*tilesz x 1 */
  /* following only used by CPU LM */
  int tileoff; /* tile offset for hybrid solution */

  /* following only used by GPU LM version */
  double *ddcoh; /* coherencies, rearranged for easy copying to GPU, also real,imag instead of complex */
  short *ddbase; /* baseline to station maps, same as barr, size 2*Nbase*tilesz x 1, assume no of stations < 32k, if flagged set to -1 */
  /* following used only by LBFGS */
  short *hbb; /*  baseline to station maps, same as ddbase size 2*Nbase*tilesz x 1, assume no of stations < 32k, if flagged set to -1 */
  int *ptoclus; /* param no -> cluster mapping, size 2*M x 1 
      for each cluster : chunk size, start param index */

  /* following used only by mixed precision solver */
  float *ddcohf; /* float version of ddcoh */

  /* following used only by robust T cost/grad functions */
  double robust_nu;

  /* following used only by RTR */
} me_data_t;


/* structure for gpu driver threads for LBFGS */
typedef struct thread_gpu_data_t {
  int ThreadsPerBlock;
  int BlocksPerGrid;
  int card; /* which gpu ? */
   
  int Nbase; /* no of baselines */
  int tilesz; /* tile size */
  baseline_t *barr; /* baseline->station mapping, size Nbase*tilesz x 1 */
  int M; /* no of clusters */
  int N; /* no of stations */
  complex double *coh; /* pre calculated cluster coherencies, per cluster 4xNbase values, total size 4*M*Nbase*tilesz x 1 */
  int m; /* no of parameters */
  int n; /* no of observations */
  double *xo; /* observed data size n x 1 */
  double *p;/* parameter vectors size m x 1 */
  double *g; /* gradient vector (output) size m x 1*/
  int g_start; /* at which point in g do we start calculation */
  int g_end; /* at which point in g do we end calculation */

  short *hbb; /*  baseline to station maps, same as ddbase size 2*Nbase*tilesz x 1, assume no of stations < 32k, if flagged set to -1 */
  int *ptoclus; /* param no -> cluster mapping, size 2*M x 1 
      for each cluster : chunk size, start param index */

  /* only used in robust LBFGS */
  double robust_nu;
} thread_gpu_data;


/* structure for driver threads to evaluate gradient */
typedef struct thread_data_grad_ {
  int Nbase; /* no of baselines */
  int tilesz; /* tile size */
  baseline_t *barr; /* baseline->station mapping, size Nbase*tilesz x 1 */
  clus_source_t *carr; /* sky model, with clusters Mx1 */
  int M; /* no of clusters */
  int N; /* no of stations */
  complex double *coh; /* pre calculated cluster coherencies, per cluster 4xNbase values, total size 4*M*Nbase*tilesz x 1 */
  int m; /* no of parameters */
  int n; /* no of observations */
  double *x; /* residual data size n x 1 x=observed-func*/
  double *p;/* parameter vectors size m x 1 */
  double *g; /* gradient vector (output) size m x 1*/
  int g_start; /* at which point in g do we start calculation */
  int g_end; /* at which point in g do we end calculation */

  /* only used in robust version */
  double robust_nu;
} thread_data_grad_t;

/* structure for weight product calculation in robust LM  */
typedef struct thread_data_vec_{
  int starti,endi;
  double *ed;
  double *wtd;
} thread_data_vec_t;

/* structure for weight calculation + nu update in robust LM  */
typedef struct thread_data_vecnu_{
  int starti,endi;
  double *ed;
  double *wtd;
  double *q;
  double nu0;
  double sumq;
  double nulow,nuhigh;
} thread_data_vecnu_t;


/* structure for worker threads for setting 1/0 */
typedef struct thread_data_onezero_ {
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  short *ddbase; /* baseline to station maps, (sta1,sta2,flag) */
  float *x; /* data vector */
} thread_data_onezero_t;


/* structure for worker threads for finding sum(|x|) and y^T |x| */
typedef struct thread_data_findsumprod_ {
  int startbase; /* starting baseline */
  int endbase; /* ending baseline */
  float *x; /* can be -ve*/
  float *y;
  float sum1; /* sum(|x|) */
  float sum2; /* y^T |x| */
} thread_data_findsumprod_t;

#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* Common_H */
