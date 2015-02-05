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

#include "cuda.h"
#include <cuComplex.h>
#include <stdio.h>

/* enable this for checking for kernel failure */
#define CUDA_DBG

__global__ void 
kernel_sqrtdiv_fl(int M, float eps, float *__restrict__ x){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<M) {
    if (x[tid]>eps) {
      x[tid]=1.0f/sqrtf(x[tid]);
    } else {
      x[tid]=0.0f;
    }
  }
}

__global__ void 
kernel_diagmult_fl(int M, float *__restrict__ U, const float *__restrict__ D) {

  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* which column this tid operates on */
  unsigned int col = tid/M;
  if (tid<M*M) {
     U[tid]=U[tid]*D[col];
  }

}


__global__ void 
kernel_jnorm_fl(int N, int M, const float *__restrict__ J, float *__restrict__ d) {
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* each thread handles one row */  
  if (tid<N) {
    d[tid]=0.0f;
    for (int ci=0; ci<M; ci++) {
     /* J is transposed, so read each column */
     d[tid]=d[tid]+J[tid*M+ci]*J[tid*M+ci]; 
    }
  }
}

__global__ void 
kernel_jacf_fl2(int Nbase, int M, float *__restrict__ jac, const float *__restrict__ coh, const float *__restrict__ p, const char *__restrict__ bb, int N){
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  /* which parameter:0...M */
  unsigned int m = threadIdx.y + blockDim.y*blockIdx.y;

  if(n<Nbase && m<M) {
    int sta1=(int)bb[3*n];
    int sta2=(int)bb[3*n+1];
    /* condition for calculating this baseline sum is 
     If this baseline is flagged,
     or if this parameter does not belong to sta1 or sta2
     we do not compute
    */
    int stc=m>>3; /* 0...Ns-1 (because M=total par= 8 * Nstations */
    /* flags are not taken into account */
    if (((stc==sta2)||(stc==sta1))) {   

     cuFloatComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* which parameter exactly 0..7 */
     int stoff=m-stc*8;
     float pp1[8]; 
     float pp2[8]; 
     if (stc==sta1) {
      for (int cn=0; cn<8; cn++) {
       pp1[cn]=0.0f;
       pp2[cn]=p[sta2*8+cn];
      }
      pp1[stoff]=1.0f;
     } else if (stc==sta2) {
      for (int cn=0; cn<8; cn++) {
       pp2[cn]=0.0f;
       pp1[cn]=p[sta1*8+cn];
      }
      pp2[stoff]=1.0f;
     }


     cuFloatComplex G1[4];
     G1[0].x=pp1[0];
     G1[0].y=pp1[1];
     G1[1].x=pp1[2];
     G1[1].y=pp1[3];
     G1[2].x=pp1[4];
     G1[2].y=pp1[5];
     G1[3].x=pp1[6];
     G1[3].y=pp1[7];
     
     cuFloatComplex T1[4];
     /* T=G1*C */
     T1[0]=cuCaddf(cuCmulf(G1[0],C[0]),cuCmulf(G1[1],C[2]));
     T1[1]=cuCaddf(cuCmulf(G1[0],C[1]),cuCmulf(G1[1],C[3]));
     T1[2]=cuCaddf(cuCmulf(G1[2],C[0]),cuCmulf(G1[3],C[2]));
     T1[3]=cuCaddf(cuCmulf(G1[2],C[1]),cuCmulf(G1[3],C[3]));

     cuFloatComplex G2[4];
     /* conjugate this */
     G2[0].x=pp2[0];
     G2[0].y=-pp2[1];
     G2[2].x=pp2[2];
     G2[2].y=-pp2[3];
     G2[1].x=pp2[4];
     G2[1].y=-pp2[5];
     G2[3].x=pp2[6];
     G2[3].y=-pp2[7];

     cuFloatComplex T2[4];
     T2[0]=cuCaddf(cuCmulf(T1[0],G2[0]),cuCmulf(T1[1],G2[2]));
     T2[1]=cuCaddf(cuCmulf(T1[0],G2[1]),cuCmulf(T1[1],G2[3]));
     T2[2]=cuCaddf(cuCmulf(T1[2],G2[0]),cuCmulf(T1[3],G2[2]));
     T2[3]=cuCaddf(cuCmulf(T1[2],G2[1]),cuCmulf(T1[3],G2[3]));
     /* update jacobian */
     /* NOTE: row major order */
     jac[m+M*8*n]=T2[0].x;
     jac[m+M*(8*n+1)]=T2[0].y;
     jac[m+M*(8*n+2)]=T2[1].x;
     jac[m+M*(8*n+3)]=T2[1].y;
     jac[m+M*(8*n+4)]=T2[2].x;
     jac[m+M*(8*n+5)]=T2[2].y;
     jac[m+M*(8*n+6)]=T2[3].x;
     jac[m+M*(8*n+7)]=T2[3].y;

    } 
   }

}


/* only use extern if calling code is C */
extern "C"
{


/* cuda driver for calculating jacf() */
/* p: params (Mx1), jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
void
cudakernel_jacf_fl2(float *p, float *jac, int M, int N, float *coh, char *bbh, int Nbase, int Mclus, int Nstations) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  /* NOTE: use small value for ThreadsPerBlock here, like 8 */
  dim3 threadsPerBlock(16, 8);
  /* jacobian: Nbase x Nstations (proportional to N), so */
  dim3 numBlocks((Nbase+threadsPerBlock.x-1)/threadsPerBlock.x, 
               (M+threadsPerBlock.y-1)/threadsPerBlock.y);
  /* set memory of jac to zero */
  cudaMemset(jac, 0, N*M*sizeof(float));
 // printf("Kernel Jax data size=%d, params=%d, block=%d,%d, thread=%d,%d, baselines=%d\n",N, M, numBlocks.x,numBlocks.y, threadsPerBlock.x, threadsPerBlock.y, Nbase);
  kernel_jacf_fl2<<< numBlocks, threadsPerBlock>>>(Nbase,  M, jac, coh, p, bbh, Nstations);

  cudaDeviceSynchronize();
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

}

/* invert sqrt(singular values)  1/Sd[]  for Sd[]> eps */
void
cudakernel_sqrtdiv_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float eps, float *Sd) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_sqrtdiv_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(M, eps, Sd);
  cudaDeviceSynchronize();
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

}


/* U <= U D, 
   U : MxM
   D : Mx1, diagonal matrix
*/
void
cudakernel_diagmult_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float *U, float *D) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_diagmult_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(M, U, D);
  cudaDeviceSynchronize();
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

}


/* diag(J^T J)
   d[i] = J[i,:] * J[i,:]
   J: NxM (in row major order, so J[i,:] is actually J[:,i]
   d: Nx1
*/
void
cudakernel_jnorm_fl(int ThreadsPerBlock, int BlocksPerGrid, float *J, int N, int M, float *d) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_jnorm_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(N,M,J,d);
  cudaDeviceSynchronize();
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif
}

}
