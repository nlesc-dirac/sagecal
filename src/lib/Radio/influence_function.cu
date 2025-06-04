/*
 *
 Copyright (C) 2025 Sarod Yatawatta <sarod@users.sf.net>  
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

#include "cuda.h"
#include <cuComplex.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "Dirac_radio.h"

/* enable this for checking for kernel failure */
#define CUDA_DBG

/* matrix multiplications */
/* C=A*B */
static __device__ void
amb(const cuFloatComplex*__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],b[0]),cuCmulf(a[1],b[2]));
 c[1]=cuCaddf(cuCmulf(a[0],b[1]),cuCmulf(a[1],b[3]));
 c[2]=cuCaddf(cuCmulf(a[2],b[0]),cuCmulf(a[3],b[2]));
 c[3]=cuCaddf(cuCmulf(a[2],b[1]),cuCmulf(a[3],b[3]));
} 
/* C=A*B^H */
static __device__ void
ambt(const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],cuConjf(b[0])),cuCmulf(a[1],cuConjf(b[1])));
 c[1]=cuCaddf(cuCmulf(a[0],cuConjf(b[2])),cuCmulf(a[1],cuConjf(b[3])));
 c[2]=cuCaddf(cuCmulf(a[2],cuConjf(b[0])),cuCmulf(a[3],cuConjf(b[1])));
 c[3]=cuCaddf(cuCmulf(a[2],cuConjf(b[2])),cuCmulf(a[3],cuConjf(b[3])));
}

__global__ void
kernel_hessian(int B, int N, int T, int F, baseline_t *barr,
    const float *__restrict__ coh, const float *__restrict__ res, float *hess) {

  /* only work with the first freq, so F==1 taken */
  /* x: baseline */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;
  /* y: station, column block of Hessian, upper triangle */
  unsigned int m=threadIdx.y+blockDim.y*blockIdx.y;

  if (n<B) {
    int sta1=barr[n].sta1;
    int sta2=barr[n].sta2;
    if ((sta1>0 && sta2>0) && (m==sta1 or m==sta2)) {

    cuFloatComplex C[4],R[4];
    C[0].x=(coh[8*n]);
    C[0].y=(coh[8*n+1]);
    C[1].x=(coh[8*n+2]);
    C[1].y=(coh[8*n+3]);
    C[2].x=(coh[8*n+4]);
    C[2].y=(coh[8*n+5]);
    C[3].x=(coh[8*n+6]);
    C[3].y=(coh[8*n+7]);
    R[0].x=(res[8*n]);
    R[0].y=(res[8*n+1]);
    R[1].x=(res[8*n+2]);
    R[1].y=(res[8*n+3]);
    R[2].x=(res[8*n+4]);
    R[2].y=(res[8*n+5]);
    R[3].x=(res[8*n+6]);
    R[3].y=(res[8*n+7]);
    }

  }
  __syncthreads();

  /* copy upper triangle to lower triangle map column block m to
   row block m */


}
 

/* only use extern if calling code is C */
extern "C"
{

#ifdef CUDA_DBG
static void
checkCudaError(cudaError_t err, const char *file, int line)
{
    if(!err)
        return;
    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
}
#endif


void
cudakernel_hessian(int B, int N, int T, int F, baseline_t *barr, float *coh, float *res, float *hess) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  /* spawn threads to handle baselines, these threads will loop over sources */
  /* thread x : baseline, thread y: station */
  dim3 threadsPerBlock(16,8); 
  dim3 numBlocks((B+threadsPerBlock.x-1)/threadsPerBlock.x,
         (N+threadsPerBlock.y-1)/threadsPerBlock.y);
  kernel_hessian<<<numBlocks,threadsPerBlock>>>(B, N, T, F, barr,
    coh, res, hess);
  cudaDeviceSynchronize();
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
}

} /* extern "C" */
