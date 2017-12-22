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
//#define CUDA_DBG

__global__ void 
kernel_func_wt_fl(int Nbase, float *x, float *coh, float *p, short *bb, float *wt, int N){
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;

  /* this thread works on 
    x[8*n:8*n+7], coh[8*M*n:8*M*n+8*M-1]
    bb[2*n:2*n+1] (sta1,sta2)
    organization of p (N stations and M clusters)
             sta 0          sta 1           sta 2        ....  sta N-1 
  clus 0   0...7            8...15          16...23      ...   8N-8     8N-1
  clus 1   8N..8N+7         8N+8..8N+15     8N+16..8N+23 ....  8N+8N-8...8N+8N-1
  ......
  clus M-1 (M-1)N..(M-1)N+7 (M-1)N+8..(M-1)N+15....  ...(M-1)N+8N-8 (M-1)N+8N-1

    organization of coherencies (coh)
        [0, 8*M-1] : baseline 0
        [8*M, 8*M+8*M-1]: baseline 1
        [n*8*M, n*8*M+8*M-1]: baseline n
        ......
        [n*8*M+cm*8, n*8*M+cm*8+7]  cluster cm, baseline n

    residual error stored at sum[n]
  */ 

  if(n<Nbase) {
    int sta1=(int)bb[2*n];
    int sta2=(int)bb[2*n+1];

    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
    */
    if (sta1>=0 && sta2>=0) {   
     cuComplex G1[4];
     float pp[8]; 
     pp[0]=p[sta1*8];
     pp[1]=p[sta1*8+1];
     pp[2]=p[sta1*8+2];
     pp[3]=p[sta1*8+3];
     pp[4]=p[sta1*8+4];
     pp[5]=p[sta1*8+5];
     pp[6]=p[sta1*8+6];
     pp[7]=p[sta1*8+7];
     G1[0].x=pp[0];
     G1[0].y=pp[1];
     G1[1].x=pp[2];
     G1[1].y=pp[3];
     G1[2].x=pp[4];
     G1[2].y=pp[5];
     G1[3].x=pp[6];
     G1[3].y=pp[7];
     

     cuComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     cuComplex T1[4];
     /* T=G1*C */
     T1[0]=cuCaddf(cuCmulf(G1[0],C[0]),cuCmulf(G1[1],C[2]));
     T1[1]=cuCaddf(cuCmulf(G1[0],C[1]),cuCmulf(G1[1],C[3]));
     T1[2]=cuCaddf(cuCmulf(G1[2],C[0]),cuCmulf(G1[3],C[2]));
     T1[3]=cuCaddf(cuCmulf(G1[2],C[1]),cuCmulf(G1[3],C[3]));

     cuComplex G2[4];
     /* conjugate this */
     pp[0]=p[sta2*8];
     pp[1]=-p[sta2*8+1];
     pp[2]=p[sta2*8+2];
     pp[3]=-p[sta2*8+3];
     pp[4]=p[sta2*8+4];
     pp[5]=-p[sta2*8+5];
     pp[6]=p[sta2*8+6];
     pp[7]=-p[sta2*8+7];
     G2[0].x=pp[0];
     G2[0].y=pp[1];
     G2[2].x=pp[2];
     G2[2].y=pp[3];
     G2[1].x=pp[4];
     G2[1].y=pp[5];
     G2[3].x=pp[6];
     G2[3].y=pp[7];

     cuComplex T2[4];
     T2[0]=cuCaddf(cuCmulf(T1[0],G2[0]),cuCmulf(T1[1],G2[2]));
     T2[1]=cuCaddf(cuCmulf(T1[0],G2[1]),cuCmulf(T1[1],G2[3]));
     T2[2]=cuCaddf(cuCmulf(T1[2],G2[0]),cuCmulf(T1[3],G2[2]));
     T2[3]=cuCaddf(cuCmulf(T1[2],G2[1]),cuCmulf(T1[3],G2[3]));
     /* update model vector, with weights */
     x[8*n]=wt[8*n]*T2[0].x;
     x[8*n+1]=wt[8*n+1]*T2[0].y;
     x[8*n+2]=wt[8*n+2]*T2[1].x;
     x[8*n+3]=wt[8*n+3]*T2[1].y;
     x[8*n+4]=wt[8*n+4]*T2[2].x;
     x[8*n+5]=wt[8*n+5]*T2[2].y;
     x[8*n+6]=wt[8*n+6]*T2[3].x;
     x[8*n+7]=wt[8*n+7]*T2[3].y;

    } 
   }

}

__global__ void 
kernel_jacf_wt_fl(int Nbase, int M, float *jac, float *coh, float *p, short *bb, float *wt, int N){
  /* global thread index : equal to the baseline */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  /* which parameter:0...M */
  unsigned int m = threadIdx.y + blockDim.y*blockIdx.y;

  /* this thread works on 
    x[8*n:8*n+7], coh[8*M*n:8*M*n+8*M-1]
    bb[2*n:2*n+1] (sta1,sta2)
    organization of p (N stations and M clusters)
             sta 0          sta 1           sta 2        ....  sta N-1 
  clus 0   0...7            8...15          16...23      ...   8N-8     8N-1
  clus 1   8N..8N+7         8N+8..8N+15     8N+16..8N+23 ....  8N+8N-8...8N+8N-1
  ......
  clus M-1 (M-1)N..(M-1)N+7 (M-1)N+8..(M-1)N+15....  ...(M-1)N+8N-8 (M-1)N+8N-1

    organization of coherencies (coh)
        [0, 8*M-1] : baseline 0
        [8*M, 8*M+8*M-1]: baseline 1
        [n*8*M, n*8*M+8*M-1]: baseline n
        ......
        [n*8*M+cm*8, n*8*M+cm*8+7]  cluster cm, baseline n

    residual error stored at sum[n]
  */ 

  if(n<Nbase && m<M) {
    int sta1=(int)bb[2*n];
    int sta2=(int)bb[2*n+1];
    /* condition for calculating this baseline sum is 
     If this baseline is flagged,
     or if this parameter does not belong to sta1 or sta2
     we do not compute
    */
    //int stc=m/8; /* 0...Ns-1 (because M=total par= 8 * Nstations */
    int stc=m>>3; /* 0...Ns-1 (because M=total par= 8 * Nstations */

    if (((stc==sta2)||(stc==sta1)) && sta1>=0 && sta2>=0 ) {   

     cuComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     /* which parameter exactly 0..7 */
     //int stoff=m%8;
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


     cuComplex G1[4];
     G1[0].x=pp1[0];
     G1[0].y=pp1[1];
     G1[1].x=pp1[2];
     G1[1].y=pp1[3];
     G1[2].x=pp1[4];
     G1[2].y=pp1[5];
     G1[3].x=pp1[6];
     G1[3].y=pp1[7];
     
     cuComplex T1[4];
     /* T=G1*C */
     T1[0]=cuCaddf(cuCmulf(G1[0],C[0]),cuCmulf(G1[1],C[2]));
     T1[1]=cuCaddf(cuCmulf(G1[0],C[1]),cuCmulf(G1[1],C[3]));
     T1[2]=cuCaddf(cuCmulf(G1[2],C[0]),cuCmulf(G1[3],C[2]));
     T1[3]=cuCaddf(cuCmulf(G1[2],C[1]),cuCmulf(G1[3],C[3]));

     cuComplex G2[4];
     /* conjugate this */
     G2[0].x=pp2[0];
     G2[0].y=-pp2[1];
     G2[2].x=pp2[2];
     G2[2].y=-pp2[3];
     G2[1].x=pp2[4];
     G2[1].y=-pp2[5];
     G2[3].x=pp2[6];
     G2[3].y=-pp2[7];

     cuComplex T2[4];
     T2[0]=cuCaddf(cuCmulf(T1[0],G2[0]),cuCmulf(T1[1],G2[2]));
     T2[1]=cuCaddf(cuCmulf(T1[0],G2[1]),cuCmulf(T1[1],G2[3]));
     T2[2]=cuCaddf(cuCmulf(T1[2],G2[0]),cuCmulf(T1[3],G2[2]));
     T2[3]=cuCaddf(cuCmulf(T1[2],G2[1]),cuCmulf(T1[3],G2[3]));
     /* update jacobian , with row weights */
     /* NOTE: row major order */
     jac[m+M*8*n]=wt[8*n]*T2[0].x;
     jac[m+M*(8*n+1)]=wt[8*n+1]*T2[0].y;
     jac[m+M*(8*n+2)]=wt[8*n+2]*T2[1].x;
     jac[m+M*(8*n+3)]=wt[8*n+3]*T2[1].y;
     jac[m+M*(8*n+4)]=wt[8*n+4]*T2[2].x;
     jac[m+M*(8*n+5)]=wt[8*n+5]*T2[2].y;
     jac[m+M*(8*n+6)]=wt[8*n+6]*T2[3].x;
     jac[m+M*(8*n+7)]=wt[8*n+7]*T2[3].y;

    } 
   }

}

__global__ void 
kernel_setweights_fl(int N, float *wt, float alpha){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<N) {
     wt[tid]=alpha;
  }
}

__global__ void 
kernel_hadamard_fl(int N, float *wt, float *x){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<N) {
     x[tid]*=wt[tid];
  }
}

__global__ void 
kernel_updateweights_fl(int N, float *wt, float *x, float *q, float nu){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<N) {
     wt[tid]=((nu+1.0f)/(nu+x[tid]*x[tid]));
     q[tid]=wt[tid]-logf(wt[tid]); /* so that its +ve */
  }
}

__global__ void 
kernel_sqrtweights_fl(int N, float *wt){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<N) {
     wt[tid]=sqrtf(wt[tid]); 
  }
}


__device__ float 
digamma_fl(float x) {
  float result = 0.0f, xx, xx2, xx4;
  for ( ; x < 7.0f; ++x) { /* reduce x till x<7 */
    result -= 1.0f/x;
  }
  x -= 1.0f/2.0f;
  xx = 1.0f/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += logf(x)+(1.0f/24.0f)*xx2-(7.0f/960.0f)*xx4+(31.0f/8064.0f)*xx4*xx2-(127.0f/30720.0f)*xx4*xx4;
  return result;
}

__global__ void 
kernel_evaluatenu_fl(int Nd, float qsum, float *q, float deltanu,float nulow) {
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<Nd) {
   float thisnu=(nulow+((float)tid)*deltanu);
   float dgm=digamma_fl(thisnu*0.5f+0.5f);
   q[tid]=dgm-logf((thisnu+1.0f)*0.5f); /* psi((nu+1)/2)-log((nu+1)/2) */
   dgm=digamma_fl(thisnu*0.5f);
   q[tid]+=-dgm+logf((thisnu)*0.5f); /* -psi((nu)/2)+log((nu)/2) */
   q[tid]+=-qsum+1.0f; /* -(-sum(ln(w_i))/N+sum(w_i)/N)+1 */
  }
}

__global__ void 
kernel_evaluatenu_fl_eight(int Nd, float qsum, float *q, float deltanu,float nulow, float nu0) {
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* each block calculte  psi((nu+8)/2)-log((nu+8)/2) */
  /* actually p=2, so psi((nu+2)/2)-log((nu+2)/2) */
  float dgm0;
  if (threadIdx.x==0) {
   dgm0=digamma_fl(nu0*0.5f+1.0f);
   dgm0=dgm0-logf((nu0+2.0f)*0.5f); /* psi((nu0+8)/2)-log((nu0+8)/2) */
  }
  __syncthreads();
  if (tid<Nd) {
   float thisnu=(nulow+((float)tid)*deltanu);
   q[tid]=dgm0; /* psi((nu0+8)/2)-log((nu0+8)/2) */
   float dgm=digamma_fl(thisnu*0.5f);
   q[tid]+=-dgm+logf((thisnu)*0.5f); /* -psi((nu)/2)+log((nu)/2) */
   q[tid]+=-qsum+1.0f; /* -(-sum(ln(w_i))/N+sum(w_i)/N)+1 */
  }
}

/* only use extern if calling code is C */
extern "C"
{

/* set initial weights to 1 by a cuda kernel */
void
cudakernel_setweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float alpha) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_setweights_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt, alpha);
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

/* hadamard product by a cuda kernel x<= x*wt */
void
cudakernel_hadamard_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_hadamard_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt, x);
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

/* update weights by a cuda kernel */
void
cudakernel_updateweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x, float *q, float robust_nu) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_updateweights_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt, x, q, robust_nu);
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

/* update weights by a cuda kernel */
void
cudakernel_sqrtweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_sqrtweights_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt);
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

/* evaluate expression for finding optimum nu for 
  a range of nu values */
void
cudakernel_evaluatenu_fl(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_evaluatenu_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(Nd, qsum, q, deltanu,nulow);
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


/* evaluate expression for finding optimum nu for 
  a range of nu values, using AECM (p=8 before, but now p=2)
  nu0: current value of robust_nu*/
void
cudakernel_evaluatenu_fl_eight(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow, float nu0) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_evaluatenu_fl_eight<<< BlocksPerGrid, ThreadsPerBlock >>>(Nd, qsum, q, deltanu,nulow, nu0);
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

/* cuda driver for calculating wt \odot f() */
/* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
void
cudakernel_func_wt_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cudaMemset(x, 0, N*sizeof(float));
//  printf("Kernel data size=%d, block=%d, thread=%d, baselines=%d\n",N,BlocksPerGrid, ThreadsPerBlock,Nbase);
  kernel_func_wt_fl<<< BlocksPerGrid, ThreadsPerBlock >>>(Nbase,  x, coh, p, bbh, wt, Nstations);
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

/* cuda driver for calculating wt \odot jacf() */
/* p: params (Mx1), jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
void
cudakernel_jacf_wt_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations, int clus) {

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
  kernel_jacf_wt_fl<<< numBlocks, threadsPerBlock>>>(Nbase,  M, jac, coh, p, bbh, wt, Nstations);

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
