/*
 *
 Copyright (C) 2019 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <pthread.h>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

//#define DEBUG

int
bfgsfit_minibatch_visibilities(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata) {

  me_data_t lmdata;

  /*  no. of true parameters */
  int m=N*Mt*8;
  /* no of data */
  int n=Nbase*tilesz*Nf*8;

  /* setup the ME data */
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
  lmdata.Nt=Nt;
  lmdata.coh=coh;
  lmdata.Nchan=Nf; /* multichannel data */

  /* starting guess of robust nu */
  lmdata.robust_nu=robust_nu;

  /* call lbfgs_fit_cuda() with proper cost/grad functions */
  return 0;
}





/* minibatch mode with consensus */
int
bfgsfit_minibatch_consensus(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, double *y, double *z, double *rho, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata) {

  me_data_t lmdata;

  /*  no. of true parameters */
  int m=N*Mt*8;
  /* no of data */
  int n=Nbase*tilesz*Nf*8;

  /* setup the ME data */
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
  lmdata.Nt=Nt;
  lmdata.coh=coh;
  lmdata.Nchan=Nf; /* multichannel data */

  /* starting guess of robust nu */
  lmdata.robust_nu=robust_nu;

  /* call lbfgs_fit_cuda() with proper cost/grad functions */
  return 0;
}
