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



__global__ void kernel_product(int N, int M, int Ns, double *x, double *coh, double *p, int *bb, double *sum, double *tsum, int stc, int stm, int ci, double pci){
  /* to store partial sum per thread (buffer size given by kernel call) */
  extern __shared__ double sdata[];
  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  /* thread idx in this block */
  unsigned int tid = threadIdx.x;


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

  sdata[tid]=0.0;
  if(n<N) {
    int sta1=bb[2*n];
    int sta2=bb[2*n+1];
    /* condition for calculating this baseline sum is 
      1) its not flagged (sta1,sta2)>=0
      2) we update all paratmeters stc==-1
      3) updated parameter value is stc==sta1 or stc==sta2
    */
    if (((stc==-1)||(stc==sta1)||(stc==sta2)) && sta1>=0 && sta2>=0) {   

    double temp1,temp2,sumn;
    sumn=0.0;
    double c=0.0;
    for (unsigned int cm=0; cm<M; cm++) {

     cuDoubleComplex G1[4];
     double pp[8]; 
     pp[0]=p[sta1*8+cm*8*Ns];
     pp[1]=p[sta1*8+cm*8*Ns+1];
     pp[2]=p[sta1*8+cm*8*Ns+2];
     pp[3]=p[sta1*8+cm*8*Ns+3];
     pp[4]=p[sta1*8+cm*8*Ns+4];
     pp[5]=p[sta1*8+cm*8*Ns+5];
     pp[6]=p[sta1*8+cm*8*Ns+6];
     pp[7]=p[sta1*8+cm*8*Ns+7];
     if (stc==sta1 && stm==cm) {  
      pp[ci]=pci;
     }
     G1[0].x=pp[0];
     G1[0].y=pp[1];
     G1[1].x=pp[2];
     G1[1].y=pp[3];
     G1[2].x=pp[4];
     G1[2].y=pp[5];
     G1[3].x=pp[6];
     G1[3].y=pp[7];
     

     cuDoubleComplex C[4];
     C[0].x=coh[8*n*M+8*cm];
     C[0].y=coh[8*n*M+8*cm+1];
     C[1].x=coh[8*n*M+8*cm+2];
     C[1].y=coh[8*n*M+8*cm+3];
     C[2].x=coh[8*n*M+8*cm+4];
     C[2].y=coh[8*n*M+8*cm+5];
     C[3].x=coh[8*n*M+8*cm+6];
     C[3].y=coh[8*n*M+8*cm+7]; 
 
     cuDoubleComplex T1[4];
     /* T=G1*C */
     T1[0]=cuCadd(cuCmul(G1[0],C[0]),cuCmul(G1[1],C[2]));
     T1[1]=cuCadd(cuCmul(G1[0],C[1]),cuCmul(G1[1],C[3]));
     T1[2]=cuCadd(cuCmul(G1[2],C[0]),cuCmul(G1[3],C[2]));
     T1[3]=cuCadd(cuCmul(G1[2],C[1]),cuCmul(G1[3],C[3]));

     cuDoubleComplex G2[4];
     /* conjugate this */
     pp[0]=p[sta2*8+cm*8*Ns];
     pp[1]=-p[sta2*8+cm*8*Ns+1];
     pp[2]=p[sta2*8+cm*8*Ns+2];
     pp[3]=-p[sta2*8+cm*8*Ns+3];
     pp[4]=p[sta2*8+cm*8*Ns+4];
     pp[5]=-p[sta2*8+cm*8*Ns+5];
     pp[6]=p[sta2*8+cm*8*Ns+6];
     pp[7]=-p[sta2*8+cm*8*Ns+7];
     if (stc==sta2 && stm==cm) {  
      pp[ci]=pci;
     }
     G2[0].x=pp[0];
     G2[0].y=pp[1];
     G2[2].x=pp[2];
     G2[2].y=pp[3];
     G2[1].x=pp[4];
     G2[1].y=pp[5];
     G2[3].x=pp[6];
     G2[3].y=pp[7];


     cuDoubleComplex T2[4];
     T2[0]=cuCadd(cuCmul(T1[0],G2[0]),cuCmul(T1[1],G2[2]));
     T2[1]=cuCadd(cuCmul(T1[0],G2[1]),cuCmul(T1[1],G2[3]));
     T2[2]=cuCadd(cuCmul(T1[2],G2[0]),cuCmul(T1[3],G2[2]));
     T2[3]=cuCadd(cuCmul(T1[2],G2[1]),cuCmul(T1[3],G2[3]));

     /* calculate |e|^2 */
     /* use Kahan Summation */
     double tt,yy;
     temp1=x[8*n]-T2[0].x; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=x[8*n+1]-T2[0].y; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=x[8*n+2]-T2[1].x; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=x[8*n+3]-T2[1].y; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=x[8*n+4]-T2[2].x; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=x[8*n+5]-T2[2].y; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=x[8*n+6]-T2[3].x; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     temp1=x[8*n+7]-T2[3].y; temp2=temp1*temp1; yy=temp2-c; tt=sumn+yy; c=(tt-sumn)-yy; sumn=tt;
     }
     sdata[tid]=sumn;
     /* also update global memory if stc==-1 */
     if (stc==-1) {
      tsum[n]=sdata[tid];
     }
    } else if (sta1>=0 && sta2>=0) {
    /* this parameter not updated, so read from global memory */
     sdata[tid]=tsum[n];
    }
    __syncthreads();

  // do reduction in shared mem. stride over the block and
  // reduce the parts until we get down to a single value (sdata[0])
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
     if (tid < s) {
      sdata[tid] += sdata[tid + s];
     }
     __syncthreads();
  } 
  }   

  if (tid == 0) {
   sum[blockIdx.x]=sdata[0];
  }

}


__global__ void kernel_sum(double *sum,double *total){
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  /* we have only one block */
  sdata[tid]=0.0;
  if (blockIdx.x==0) {
   sdata[tid]=sum[tid];
   __syncthreads();

   for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
     if (tid < s) {
      sdata[tid] += sdata[tid + s];
     }
     __syncthreads();
   }  

   if (tid==0) {
    *total=sdata[0];
   }
  }
}



__global__ void kernel_deriv(int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, char *bb, int *ptoclus, double *grad){
  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  /* parameter number of this thread */
  unsigned int np=n+goff;


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

  if (n<Nparam) {
    /* this thread will calculate derivative for parameter np,
      and store it in grad[n] */
    double gsum=0.0;

    /* find which cluster this parameter belongs to */
    /* ptoclus[0,1]  are nchunk, and p[start index] for each cluster */
    int cli=0;
    /* np should be within ptoclus[2*cli+1]....ptoclus[2*cli+1]+ptoclus[2*cli]*8*Ns-1 */
    while ((cli<M) && (np<ptoclus[2*cli+1] || np>ptoclus[2*cli+1]+ptoclus[2*cli]*8*Ns-1)) { cli++; }
    /* now either ci>=M: cluster not found 
       or ci<M and ci is the right cluster */
    if ((cli==M) && np>=ptoclus[2*cli-1] && np<=ptoclus[2*cli-1]+ptoclus[2*cli-2]*8*Ns-1) {
     cli--;
    }
   
    if (cli<M) {
      int pstart=ptoclus[2*cli+1];
      int nchunk=ptoclus[2*cli];
 
      /* find station and which parameter for this thread (parameter) */
      /* this cluster has parameters ptoclus[2*cli+1] ..... +ptoclus[2*cli]*8*Ns-1 */
      unsigned int np_s=(np-pstart)%(8*Ns);
      unsigned int stc=np_s/8; /* this is the station of this param */
      /* which chunk does this parameter belong to */
      unsigned int tpchunk=(np-pstart)/(8*Ns);
      int tilesperchunk=(tilesz+nchunk-1)/nchunk;

      /* total baselines in one tile */
      int Nbase0=(Ns-1)*Ns/2;
      for(unsigned int nb=0; nb<Nbase; nb++) {
        /* which tile is this ? */
        int ttile=nb/Nbase0;
        /* which chunk this tile belongs to */
        int tptile=ttile/tilesperchunk;
        /* now tptile has to match tpchunk, otherwise ignore calculation */
        if (tptile==tpchunk) {
        int sta1=(int)bb[2*nb];
        int sta2=(int)bb[2*nb+1];
        /* only calculate deriv if baseline corresponds
          to this station and baseline is not flagged */
        /* flagged baselines will have sta1==sta2==-1 */
        if (((stc==sta1)||(stc==sta2)) && sta1>=0 && sta2>=0) {
         /* which parameter 0..7 */
         unsigned int stoff=np_s-stc*8; 
         /* which cluster 0..M-1 */
         unsigned int stm=cli;

         /* read residual vector, conjugated */
         cuDoubleComplex xr[4];
         xr[0].x=x[nb*8];
         xr[0].y=-x[nb*8+1];
         xr[1].x=x[nb*8+2];
         xr[1].y=-x[nb*8+3];
         xr[2].x=x[nb*8+4];
         xr[2].y=-x[nb*8+5];
         xr[3].x=x[nb*8+6];
         xr[3].y=-x[nb*8+7];

         /* read in coherency */
         cuDoubleComplex C[4];
         C[0].x=coh[8*nb*M+8*stm];
         C[0].y=coh[8*nb*M+8*stm+1];
         C[1].x=coh[8*nb*M+8*stm+2];
         C[1].y=coh[8*nb*M+8*stm+3];
         C[2].x=coh[8*nb*M+8*stm+4];
         C[2].y=coh[8*nb*M+8*stm+5];
         C[3].x=coh[8*nb*M+8*stm+6];
         C[3].y=coh[8*nb*M+8*stm+7];
         
         cuDoubleComplex G1[4];
         cuDoubleComplex G2[4];
         double pp[8]; 
         pp[0]=0.0;
         pp[1]=0.0;
         pp[2]=0.0;
         pp[3]=0.0;
         pp[4]=0.0;
         pp[5]=0.0;
         pp[6]=0.0;
         pp[7]=0.0;

         if(stc==sta1) {
           pp[stoff]=1.0;
           G1[0].x=pp[0];
           G1[0].y=pp[1];
           G1[1].x=pp[2];
           G1[1].y=pp[3];
           G1[2].x=pp[4];
           G1[2].y=pp[5];
           G1[3].x=pp[6];
           G1[3].y=pp[7];
 
           /* conjugate and transpose G2 */
           G2[0].x=p[pstart+tpchunk*8*Ns+sta2*8];
           G2[0].y=-p[pstart+tpchunk*8*Ns+sta2*8+1];
           G2[2].x=p[pstart+tpchunk*8*Ns+sta2*8+2];
           G2[2].y=-p[pstart+tpchunk*8*Ns+sta2*8+3];
           G2[1].x=p[pstart+tpchunk*8*Ns+sta2*8+4];
           G2[1].y=-p[pstart+tpchunk*8*Ns+sta2*8+5];
           G2[3].x=p[pstart+tpchunk*8*Ns+sta2*8+6];
           G2[3].y=-p[pstart+tpchunk*8*Ns+sta2*8+7];
         } else if (stc==sta2) {
           pp[stoff]=1.0;
           /* conjugate and transpose G2 */
           G2[0].x=pp[0];
           G2[0].y=-pp[1];
           G2[2].x=pp[2];
           G2[2].y=-pp[3];
           G2[1].x=pp[4];
           G2[1].y=-pp[5];
           G2[3].x=pp[6];
           G2[3].y=-pp[7];
 
           /* conjugate and transpose G2 */
           G1[0].x=p[pstart+tpchunk*8*Ns+sta1*8];
           G1[0].y=p[pstart+tpchunk*8*Ns+sta1*8+1];
           G1[1].x=p[pstart+tpchunk*8*Ns+sta1*8+2];
           G1[1].y=p[pstart+tpchunk*8*Ns+sta1*8+3];
           G1[2].x=p[pstart+tpchunk*8*Ns+sta1*8+4];
           G1[2].y=p[pstart+tpchunk*8*Ns+sta1*8+5];
           G1[3].x=p[pstart+tpchunk*8*Ns+sta1*8+6];
           G1[3].y=p[pstart+tpchunk*8*Ns+sta1*8+7];
         }
         cuDoubleComplex T1[4];
         /* T1=G1*C */
         T1[0]=cuCadd(cuCmul(G1[0],C[0]),cuCmul(G1[1],C[2]));
         T1[1]=cuCadd(cuCmul(G1[0],C[1]),cuCmul(G1[1],C[3]));
         T1[2]=cuCadd(cuCmul(G1[2],C[0]),cuCmul(G1[3],C[2]));
         T1[3]=cuCadd(cuCmul(G1[2],C[1]),cuCmul(G1[3],C[3]));

         cuDoubleComplex T2[4];
         /* T2=T1*G2 , G2 conjugate transposed */
         T2[0]=cuCadd(cuCmul(T1[0],G2[0]),cuCmul(T1[1],G2[2]));
         T2[1]=cuCadd(cuCmul(T1[0],G2[1]),cuCmul(T1[1],G2[3]));
         T2[2]=cuCadd(cuCmul(T1[2],G2[0]),cuCmul(T1[3],G2[2]));
         T2[3]=cuCadd(cuCmul(T1[2],G2[1]),cuCmul(T1[3],G2[3]));

         /* calculate product xr*vec(J_p C J_q^H ) */
         cuDoubleComplex csum;
         csum=cuCmul(xr[0],T2[0]);
         csum=cuCadd(csum,cuCmul(xr[1],T2[1]));
         csum=cuCadd(csum,cuCmul(xr[2],T2[2]));
         csum=cuCadd(csum,cuCmul(xr[3],T2[3]));



        gsum+=-2.0*csum.x;     
      } 

     }

    }
    }

    
    grad[n]=gsum;
  }   

}

__global__ void kernel_diagdiv(int M, double eps, double *y,double *x){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<M) {
    if (x[tid]>eps) {
      y[tid]=y[tid]/x[tid];
    } else {
      y[tid]=0.0;
    }
  }
}

__global__ void kernel_diagmu(int M, double *A,double mu){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<M) {
    A[tid*(M+1)]=A[tid*(M+1)]+mu;
  } 
}


__global__ void kernel_func(int Nbase, double *x, double *coh, double *p, char *bb, int N){
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
     cuDoubleComplex G1[4];
     double pp[8]; 
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
     

     cuDoubleComplex C[4];
     C[0].x=coh[8*n];
     C[0].y=coh[8*n+1];
     C[1].x=coh[8*n+2];
     C[1].y=coh[8*n+3];
     C[2].x=coh[8*n+4];
     C[2].y=coh[8*n+5];
     C[3].x=coh[8*n+6];
     C[3].y=coh[8*n+7]; 
 
     cuDoubleComplex T1[4];
     /* T=G1*C */
     T1[0]=cuCadd(cuCmul(G1[0],C[0]),cuCmul(G1[1],C[2]));
     T1[1]=cuCadd(cuCmul(G1[0],C[1]),cuCmul(G1[1],C[3]));
     T1[2]=cuCadd(cuCmul(G1[2],C[0]),cuCmul(G1[3],C[2]));
     T1[3]=cuCadd(cuCmul(G1[2],C[1]),cuCmul(G1[3],C[3]));

     cuDoubleComplex G2[4];
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

     cuDoubleComplex T2[4];
     T2[0]=cuCadd(cuCmul(T1[0],G2[0]),cuCmul(T1[1],G2[2]));
     T2[1]=cuCadd(cuCmul(T1[0],G2[1]),cuCmul(T1[1],G2[3]));
     T2[2]=cuCadd(cuCmul(T1[2],G2[0]),cuCmul(T1[3],G2[2]));
     T2[3]=cuCadd(cuCmul(T1[2],G2[1]),cuCmul(T1[3],G2[3]));
     /* update model vector */
     x[8*n]=T2[0].x;
     x[8*n+1]=T2[0].y;
     x[8*n+2]=T2[1].x;
     x[8*n+3]=T2[1].y;
     x[8*n+4]=T2[2].x;
     x[8*n+5]=T2[2].y;
     x[8*n+6]=T2[3].x;
     x[8*n+7]=T2[3].y;

    } 
   }

}

__global__ void kernel_jacf(int Nbase, int M, double *jac, double *coh, double *p, char *bb, int N){
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

     cuDoubleComplex C[4];
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
     double pp1[8]; 
     double pp2[8]; 
     if (stc==sta1) {
      for (int cn=0; cn<8; cn++) {
       pp1[cn]=0.0;
       pp2[cn]=p[sta2*8+cn];
      }
      pp1[stoff]=1.0;
     } else if (stc==sta2) {
      for (int cn=0; cn<8; cn++) {
       pp2[cn]=0.0;
       pp1[cn]=p[sta1*8+cn];
      }
      pp2[stoff]=1.0;
     }


     cuDoubleComplex G1[4];
     G1[0].x=pp1[0];
     G1[0].y=pp1[1];
     G1[1].x=pp1[2];
     G1[1].y=pp1[3];
     G1[2].x=pp1[4];
     G1[2].y=pp1[5];
     G1[3].x=pp1[6];
     G1[3].y=pp1[7];
     
     cuDoubleComplex T1[4];
     /* T=G1*C */
     T1[0]=cuCadd(cuCmul(G1[0],C[0]),cuCmul(G1[1],C[2]));
     T1[1]=cuCadd(cuCmul(G1[0],C[1]),cuCmul(G1[1],C[3]));
     T1[2]=cuCadd(cuCmul(G1[2],C[0]),cuCmul(G1[3],C[2]));
     T1[3]=cuCadd(cuCmul(G1[2],C[1]),cuCmul(G1[3],C[3]));

     cuDoubleComplex G2[4];
     /* conjugate this */
     G2[0].x=pp2[0];
     G2[0].y=-pp2[1];
     G2[2].x=pp2[2];
     G2[2].y=-pp2[3];
     G2[1].x=pp2[4];
     G2[1].y=-pp2[5];
     G2[3].x=pp2[6];
     G2[3].y=-pp2[7];

     cuDoubleComplex T2[4];
     T2[0]=cuCadd(cuCmul(T1[0],G2[0]),cuCmul(T1[1],G2[2]));
     T2[1]=cuCadd(cuCmul(T1[0],G2[1]),cuCmul(T1[1],G2[3]));
     T2[2]=cuCadd(cuCmul(T1[2],G2[0]),cuCmul(T1[3],G2[2]));
     T2[3]=cuCadd(cuCmul(T1[2],G2[1]),cuCmul(T1[3],G2[3]));
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


/* highly divergent jacobian with minimum no of computations:
   16 branches
*/
__global__ void kernel_jacf_bad(int Nbase, int M, double *jac, double *coh, double *p, char *bb){
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

     /* which parameter exactly 0..7 */
     //int stoff=m%8;
     int stoff=m-stc*8;
     int brid=(stoff/2)%2;

     cuDoubleComplex C[4];
     cuDoubleComplex G[4];
     cuDoubleComplex A[2];
     cuDoubleComplex T[4];
     /* set to zero */
     T[0].x=T[0].y=T[1].x=T[1].y=T[2].x=T[2].y=T[3].x=T[3].y=0.0;
     /* branching cases */
     if (stc==sta1) { /* E x C x G2^H */
         /* G2^H */
         G[0].x=p[sta2*8];
         G[0].y=-p[sta2*8+1];
         G[1].x=p[sta2*8+2];
         G[1].y=-p[sta2*8+3];
         G[2].x=p[sta2*8+4];
         G[2].y=-p[sta2*8+5];
         G[3].x=p[sta2*8+6];
         G[3].y=-p[sta2*8+7];

       if (brid==0) { /* stoff is 0, 1  or 4,5 */
         /* first row of C x G2^H */
         C[0].x=coh[8*n];
         C[0].y=coh[8*n+1];
         C[1].x=coh[8*n+2];
         C[1].y=coh[8*n+3];

         A[0]=cuCadd(cuCmul(C[0],G[0]),cuCmul(C[1],G[1]));
         A[1]=cuCadd(cuCmul(C[0],G[2]),cuCmul(C[1],G[3]));

       } else  { /*  stoff is 2, 3 or 6,7  */
         /* second row of C x G2^H */
         C[2].x=coh[8*n+4];
         C[2].y=coh[8*n+5];
         C[3].x=coh[8*n+6];
         C[3].y=coh[8*n+7];

         A[0]=cuCadd(cuCmul(C[2],G[0]),cuCmul(C[3],G[1]));
         A[1]=cuCadd(cuCmul(C[2],G[2]),cuCmul(C[3],G[3]));
         
       }  

         if (stoff==0 || stoff==2) {
           T[0]=A[0];
           T[1]=A[1];
         } else if (stoff==1 || stoff==3) {
           /* mult by j */
           T[0].x=-A[0].y;
           T[0].y=A[0].x;
           T[1].x=-A[1].y;
           T[1].y=A[1].x;
         } else if (stoff==4 || stoff==6) {
           T[2]=A[0];
           T[3]=A[1];
         } else { /* stoff==5  or 7 */
           /* mult by j */
           T[2].x=-A[0].y;
           T[2].y=A[0].x;
           T[3].x=-A[1].y;
           T[3].y=A[1].x;
         }

     } else { /* stc==sta2 G1 x C x E^H */
         /* G1 */
         G[0].x=p[sta1*8];
         G[0].y=p[sta1*8+1];
         G[1].x=p[sta1*8+2];
         G[1].y=p[sta1*8+3];
         G[2].x=p[sta1*8+4];
         G[2].y=p[sta1*8+5];
         G[3].x=p[sta1*8+6];
         G[3].y=p[sta1*8+7];

       if (brid==0) { /* stoff is 0, 1  or 4,5 */
         /* first column of G1 x C */
         C[0].x=coh[8*n];
         C[0].y=coh[8*n+1];
         C[2].x=coh[8*n+4];
         C[2].y=coh[8*n+5];

         A[0]=cuCadd(cuCmul(G[0],C[0]),cuCmul(G[1],C[2]));
         A[1]=cuCadd(cuCmul(G[2],C[0]),cuCmul(G[3],C[2]));

       } else  { /*  stoff is 2, 3 or 6,7  */
         /* second column of G1 x C */
         C[1].x=coh[8*n+2];
         C[1].y=coh[8*n+3];
         C[3].x=coh[8*n+6];
         C[3].y=coh[8*n+7];

         A[0]=cuCadd(cuCmul(G[0],C[1]),cuCmul(G[1],C[3]));
         A[1]=cuCadd(cuCmul(G[2],C[1]),cuCmul(G[3],C[3]));
         
       }  

         if (stoff==0 || stoff==2) {
           T[0]=A[0];
           T[2]=A[1];
         } else if (stoff==1 || stoff==3) {
           /* mult by -j */
           T[0].x=A[0].y;
           T[0].y=-A[0].x;
           T[2].x=A[1].y;
           T[2].y=-A[1].x;
         } else if (stoff==4 || stoff==6) {
           T[1]=A[0];
           T[3]=A[1];
         } else { /* stoff==5  or 7 */
           /* mult by -j */
           T[1].x=A[0].y;
           T[1].y=-A[0].x;
           T[3].x=A[1].y;
           T[3].y=-A[1].x;
         }
     }

     /* update jacobian */
     /* NOTE: row major order */
     jac[m+M*8*n]=T[0].x;
     jac[m+M*(8*n+1)]=T[0].y;
     jac[m+M*(8*n+2)]=T[1].x;
     jac[m+M*(8*n+3)]=T[1].y;
     jac[m+M*(8*n+4)]=T[2].x;
     jac[m+M*(8*n+5)]=T[2].y;
     jac[m+M*(8*n+6)]=T[3].x;
     jac[m+M*(8*n+7)]=T[3].y;

    } 
   }

}

/* only use extern if calling code is C */
extern "C"
{

/* cuda driver for kernel */
/* ThreadsPerBlock: keep <= 128
   BlocksPerGrid: depends on the threads/baselines> Threads*Blocks approx baselines
   Nbase: no of baselines (total, including tilesz >1)
   tilesz: tile size
   M: no of clusters
   Ns: no of stations
   Nparam: no of actual parameters  <=total 
   goff: starting point of gradient calculation 0..Nparams
   x: N*8 x 1 residual
   coh: N*8*M x 1
   p: M*Ns*8 x 1
   bb: 2*N x 1
   ptoclus: 2*M x 1

   grad: Nparamsx1 gradient values
*/
void cudakernel_lbfgs(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, char *bb, int *ptoclus, double *grad){
 
  cudaError_t error;
  /* invoke device on this block/thread grid (last argument is buffer size in bytes) */
  kernel_deriv<<< BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock*sizeof(double) >>> (Nbase, tilesz, M, Ns, Nparam, goff, x, coh, p, bb, ptoclus, grad);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }

  cudaDeviceSynchronize();
}


/* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps */
void cudakernel_diagdiv(int ThreadsPerBlock, int BlocksPerGrid, int M, double eps, double *Dpd, double *Sd) {

  cudaError_t error;
  kernel_diagdiv<<< BlocksPerGrid, ThreadsPerBlock >>>(M, eps, Dpd, Sd);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }

  cudaDeviceSynchronize();
}

/* cuda driver for calculating
  A<= A+mu I, adding mu to diagonal entries of A
  A: size MxM
  ThreadsPerBlock, BlocksPerGrid calculated to meet M
*/
void
cudakernel_diagmu(int ThreadsPerBlock, int BlocksPerGrid, int M, double *A, double mu) {
  cudaError_t error;
  kernel_diagmu<<< BlocksPerGrid, ThreadsPerBlock >>>(M, A, mu);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
  cudaDeviceSynchronize();
}


/* cuda driver for calculating f() */
/* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
void
cudakernel_func(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, char *bbh, int Nbase, int Mclus, int Nstations) {

  cudaError_t error;
  cudaMemset(x, 0, N*sizeof(double));
//  printf("Kernel data size=%d, block=%d, thread=%d, baselines=%d\n",N,BlocksPerGrid, ThreadsPerBlock,Nbase);
  kernel_func<<< BlocksPerGrid, ThreadsPerBlock >>>(Nbase,  x, coh, p, bbh, Nstations);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }

  cudaDeviceSynchronize();
}

/* cuda driver for calculating jacf() */
/* p: params (Mx1), jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
void
cudakernel_jacf(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, char *bbh, int Nbase, int Mclus, int Nstations) {

  cudaError_t error;
  /* NOTE: use small value for ThreadsPerBlock here, like 8 */
  dim3 threadsPerBlock(16, 8);
  /* jacobian: Nbase x Nstations (proportional to N), so */
  dim3 numBlocks((Nbase+threadsPerBlock.x-1)/threadsPerBlock.x, 
               (M+threadsPerBlock.y-1)/threadsPerBlock.y);
  /* set memory of jac to zero */
  cudaMemset(jac, 0, N*M*sizeof(double));
 // printf("Kernel Jax data size=%d, params=%d, block=%d,%d, thread=%d,%d, baselines=%d\n",N, M, numBlocks.x,numBlocks.y, threadsPerBlock.x, threadsPerBlock.y, Nbase);
  kernel_jacf<<< numBlocks, threadsPerBlock>>>(Nbase,  M, jac, coh, p, bbh, Nstations);

  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }

  cudaDeviceSynchronize();
}

}
