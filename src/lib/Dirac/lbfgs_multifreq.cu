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

#include "cuda.h"
#include <cuComplex.h>
#include <stdio.h>
#include "GPUtune.h"

/* enable this for checking for kernel failure */
//#define CUDA_DBG

/* note x is residual, not data */
__global__ void 
kernel_deriv_r_robust(int Nbase, int tilesz, int Nchan, int M, int Ns, int Nparam, int Nbasetotal, int boff, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ grad, double robust_nu){
  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  /* parameter number of this thread */
  unsigned int np=n;


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
      /* total baselines in one tile */
      int Nbase0=(Ns-1)*Ns/2;

      //int tilesperchunk=(tilesz+nchunk-1)/nchunk;
      int tilesperchunk=(Nbasetotal/Nbase0+nchunk-1)/nchunk;

      for(unsigned int nb=0; nb<Nbase; nb++) {

        /* which tile is this ? */
        int ttile=(nb+boff)/Nbase0;
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

         cuDoubleComplex G1[4];
         cuDoubleComplex G2[4];
         cuDoubleComplex T1[4];
         cuDoubleComplex T2[4];

         G1[0].x=p[pstart+tpchunk*8*Ns+sta1*8];
         G1[0].y=p[pstart+tpchunk*8*Ns+sta1*8+1];
         G1[1].x=p[pstart+tpchunk*8*Ns+sta1*8+2];
         G1[1].y=p[pstart+tpchunk*8*Ns+sta1*8+3];
         G1[2].x=p[pstart+tpchunk*8*Ns+sta1*8+4];
         G1[2].y=p[pstart+tpchunk*8*Ns+sta1*8+5];
         G1[3].x=p[pstart+tpchunk*8*Ns+sta1*8+6];
         G1[3].y=p[pstart+tpchunk*8*Ns+sta1*8+7];
         /* conjugate and transpose G2 */
         G2[0].x=p[pstart+tpchunk*8*Ns+sta2*8];
         G2[0].y=-p[pstart+tpchunk*8*Ns+sta2*8+1];
         G2[2].x=p[pstart+tpchunk*8*Ns+sta2*8+2];
         G2[2].y=-p[pstart+tpchunk*8*Ns+sta2*8+3];
         G2[1].x=p[pstart+tpchunk*8*Ns+sta2*8+4];
         G2[1].y=-p[pstart+tpchunk*8*Ns+sta2*8+5];
         G2[3].x=p[pstart+tpchunk*8*Ns+sta2*8+6];
         G2[3].y=-p[pstart+tpchunk*8*Ns+sta2*8+7];

         double pp[8]; 
         pp[0]=0.0;
         pp[1]=0.0;
         pp[2]=0.0;
         pp[3]=0.0;
         pp[4]=0.0;
         pp[5]=0.0;
         pp[6]=0.0;
         pp[7]=0.0;

         pp[stoff]=1.0;
         if(stc==sta1) {
           G1[0].x=pp[0];
           G1[0].y=pp[1];
           G1[1].x=pp[2];
           G1[1].y=pp[3];
           G1[2].x=pp[4];
           G1[2].y=pp[5];
           G1[3].x=pp[6];
           G1[3].y=pp[7];
         } else if (stc==sta2) {
           /* conjugate and transpose G2 */
           G2[0].x=pp[0];
           G2[0].y=-pp[1];
           G2[2].x=pp[2];
           G2[2].y=-pp[3];
           G2[1].x=pp[4];
           G2[1].y=-pp[5];
           G2[3].x=pp[6];
           G2[3].y=-pp[7];
         }

         for (int fi=0; fi<Nchan; fi++) {
         /* read residual vector */
         double xr[8];
         xr[0]=x[nb*8  +fi*8*Nbase];
         xr[1]=x[nb*8+1+fi*8*Nbase];
         xr[2]=x[nb*8+2+fi*8*Nbase];
         xr[3]=x[nb*8+3+fi*8*Nbase];
         xr[4]=x[nb*8+4+fi*8*Nbase];
         xr[5]=x[nb*8+5+fi*8*Nbase];
         xr[6]=x[nb*8+6+fi*8*Nbase];
         xr[7]=x[nb*8+7+fi*8*Nbase];

         /* read in coherency */
         cuDoubleComplex C[4];
         C[0].x=coh[8*nb*M+8*stm  +8*M*Nbase*fi];
         C[0].y=coh[8*nb*M+8*stm+1+8*M*Nbase*fi];
         C[1].x=coh[8*nb*M+8*stm+2+8*M*Nbase*fi];
         C[1].y=coh[8*nb*M+8*stm+3+8*M*Nbase*fi];
         C[2].x=coh[8*nb*M+8*stm+4+8*M*Nbase*fi];
         C[2].y=coh[8*nb*M+8*stm+5+8*M*Nbase*fi];
         C[3].x=coh[8*nb*M+8*stm+6+8*M*Nbase*fi];
         C[3].y=coh[8*nb*M+8*stm+7+8*M*Nbase*fi];
         
         /* T1=G1*C */
         T1[0]=cuCadd(cuCmul(G1[0],C[0]),cuCmul(G1[1],C[2]));
         T1[1]=cuCadd(cuCmul(G1[0],C[1]),cuCmul(G1[1],C[3]));
         T1[2]=cuCadd(cuCmul(G1[2],C[0]),cuCmul(G1[3],C[2]));
         T1[3]=cuCadd(cuCmul(G1[2],C[1]),cuCmul(G1[3],C[3]));

         /* T2=T1*G2 , G2 conjugate transposed */
         T2[0]=cuCadd(cuCmul(T1[0],G2[0]),cuCmul(T1[1],G2[2]));
         T2[1]=cuCadd(cuCmul(T1[0],G2[1]),cuCmul(T1[1],G2[3]));
         T2[2]=cuCadd(cuCmul(T1[2],G2[0]),cuCmul(T1[3],G2[2]));
         T2[3]=cuCadd(cuCmul(T1[2],G2[1]),cuCmul(T1[3],G2[3]));


         /* calculate product xr*vec(J_p C J_q^H )/(nu+residual^2) */
         double dsum;
         dsum=xr[0]*T2[0].x/(robust_nu+xr[0]*xr[0]);
         dsum+=xr[1]*T2[0].y/(robust_nu+xr[1]*xr[1]);
         dsum+=xr[2]*T2[1].x/(robust_nu+xr[2]*xr[2]);
         dsum+=xr[3]*T2[1].y/(robust_nu+xr[3]*xr[3]);
         dsum+=xr[4]*T2[2].x/(robust_nu+xr[4]*xr[4]);
         dsum+=xr[5]*T2[2].y/(robust_nu+xr[5]*xr[5]);
         dsum+=xr[6]*T2[3].x/(robust_nu+xr[6]*xr[6]);
         dsum+=xr[7]*T2[3].y/(robust_nu+xr[7]*xr[7]);
       /* accumulate sum NOTE
       its important to get the sign right,
      depending on res=data-model or res=model-data  */
        gsum+=-2.0*dsum;
        }

      } 

     }

    }
    }

    
    grad[n]=gsum;
  }  

}


__global__ void 
kernel_residual_multifreq(int Nbase, int Nchan, int boff, int M, int Ns, int Nbasetotal, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ ed){

  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x; /* baseline */
  unsigned int fi= threadIdx.y + blockDim.y*blockIdx.y; /* channel */

  if (n<Nbase && fi<Nchan) {
    int sta1=(int)bb[2*n];
    int sta2=(int)bb[2*n+1];

    /* only calculate deriv if baseline corresponds
    to this station and baseline is not flagged */
    /* flagged baselines will have sta1==sta2==-1 */
    if (sta1>=0 && sta2>=0) {
      /* read data vector */
      cuDoubleComplex xr[4];
      xr[0].x=x[n*8  +fi*8*Nbase];
      xr[0].y=x[n*8+1+fi*8*Nbase];
      xr[1].x=x[n*8+2+fi*8*Nbase];
      xr[1].y=x[n*8+3+fi*8*Nbase];
      xr[2].x=x[n*8+4+fi*8*Nbase];
      xr[2].y=x[n*8+5+fi*8*Nbase];
      xr[3].x=x[n*8+6+fi*8*Nbase];
      xr[3].y=x[n*8+7+fi*8*Nbase];

      for (int cm=0; cm<M; cm++) {
       int pstart=ptoclus[2*cm+1];
       int nchunk=ptoclus[2*cm];
       /* read in coherency */
       cuDoubleComplex C[4];
       C[0].x=coh[8*n*M+8*cm  +8*M*Nbase*fi];
       C[0].y=coh[8*n*M+8*cm+1+8*M*Nbase*fi];
       C[1].x=coh[8*n*M+8*cm+2+8*M*Nbase*fi];
       C[1].y=coh[8*n*M+8*cm+3+8*M*Nbase*fi];
       C[2].x=coh[8*n*M+8*cm+4+8*M*Nbase*fi];
       C[2].y=coh[8*n*M+8*cm+5+8*M*Nbase*fi];
       C[3].x=coh[8*n*M+8*cm+6+8*M*Nbase*fi];
       C[3].y=coh[8*n*M+8*cm+7+8*M*Nbase*fi];
         
       cuDoubleComplex G1[4];
       cuDoubleComplex G2[4];
       cuDoubleComplex T1[4];
       cuDoubleComplex T2[4];

       int px=(n+boff)/((Nbasetotal+nchunk-1)/nchunk);

       G1[0].x=p[pstart+px*8*Ns+sta1*8];
       G1[0].y=p[pstart+px*8*Ns+sta1*8+1];
       G1[1].x=p[pstart+px*8*Ns+sta1*8+2];
       G1[1].y=p[pstart+px*8*Ns+sta1*8+3];
       G1[2].x=p[pstart+px*8*Ns+sta1*8+4];
       G1[2].y=p[pstart+px*8*Ns+sta1*8+5];
       G1[3].x=p[pstart+px*8*Ns+sta1*8+6];
       G1[3].y=p[pstart+px*8*Ns+sta1*8+7];
 
       /* conjugate and transpose G2 */
       G2[0].x=p[pstart+px*8*Ns+sta2*8];
       G2[0].y=-p[pstart+px*8*Ns+sta2*8+1];
       G2[2].x=p[pstart+px*8*Ns+sta2*8+2];
       G2[2].y=-p[pstart+px*8*Ns+sta2*8+3];
       G2[1].x=p[pstart+px*8*Ns+sta2*8+4];
       G2[1].y=-p[pstart+px*8*Ns+sta2*8+5];
       G2[3].x=p[pstart+px*8*Ns+sta2*8+6];
       G2[3].y=-p[pstart+px*8*Ns+sta2*8+7];
 
       /* T1=G1*C */
       T1[0]=cuCadd(cuCmul(G1[0],C[0]),cuCmul(G1[1],C[2]));
       T1[1]=cuCadd(cuCmul(G1[0],C[1]),cuCmul(G1[1],C[3]));
       T1[2]=cuCadd(cuCmul(G1[2],C[0]),cuCmul(G1[3],C[2]));
       T1[3]=cuCadd(cuCmul(G1[2],C[1]),cuCmul(G1[3],C[3]));

       /* T2=T1*G2 , G2 conjugate transposed */
       T2[0]=cuCadd(cuCmul(T1[0],G2[0]),cuCmul(T1[1],G2[2]));
       T2[1]=cuCadd(cuCmul(T1[0],G2[1]),cuCmul(T1[1],G2[3]));
       T2[2]=cuCadd(cuCmul(T1[2],G2[0]),cuCmul(T1[3],G2[2]));
       T2[3]=cuCadd(cuCmul(T1[2],G2[1]),cuCmul(T1[3],G2[3]));

       /* find residul V - J_p C J_q^H */
       xr[0]=cuCsub(xr[0],T2[0]);
       xr[1]=cuCsub(xr[1],T2[1]);
       xr[2]=cuCsub(xr[2],T2[2]);
       xr[3]=cuCsub(xr[3],T2[3]);
      }
      /* store residual */

      ed[n*8  +fi*8*Nbase]=xr[0].x;
      ed[n*8+1+fi*8*Nbase]=xr[0].y;
      ed[n*8+2+fi*8*Nbase]=xr[1].x;
      ed[n*8+3+fi*8*Nbase]=xr[1].y;
      ed[n*8+4+fi*8*Nbase]=xr[2].x;
      ed[n*8+5+fi*8*Nbase]=xr[2].y;
      ed[n*8+6+fi*8*Nbase]=xr[3].x;
      ed[n*8+7+fi*8*Nbase]=xr[3].y;

    }
  } 

}


__global__ void 
kernel_fcost_multifreq_robust(int Nbase, int Nchan, int boff, int M, int Ns, int Nbasetotal, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ ed, double inv_robust_nu){
  /* shared memory */
  extern __shared__ double ek[];

  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x; /* baseline */
  unsigned int fi = threadIdx.y + blockDim.y*blockIdx.y; /* channel */
  int tid=threadIdx.x+blockDim.x*threadIdx.y; /* local 2D thread mapped to a vector */
  ek[tid]=0.0;

  if (n<Nbase && fi<Nchan) {
    double gsum=0.0;
    int sta1=(int)bb[2*n];
    int sta2=(int)bb[2*n+1];

    /* only calculate deriv if baseline corresponds
    to this station and baseline is not flagged */
    /* flagged baselines will have sta1==sta2==-1 */
    if (sta1>=0 && sta2>=0) {
       /* read data vector */
       cuDoubleComplex xr[4];
       xr[0].x=x[n*8  +fi*8*Nbase];
       xr[0].y=x[n*8+1+fi*8*Nbase];
       xr[1].x=x[n*8+2+fi*8*Nbase];
       xr[1].y=x[n*8+3+fi*8*Nbase];
       xr[2].x=x[n*8+4+fi*8*Nbase];
       xr[2].y=x[n*8+5+fi*8*Nbase];
       xr[3].x=x[n*8+6+fi*8*Nbase];
       xr[3].y=x[n*8+7+fi*8*Nbase];


      for (int cm=0; cm<M; cm++) {
       int pstart=ptoclus[2*cm+1];
       int nchunk=ptoclus[2*cm];
         
       cuDoubleComplex G1[4];
       cuDoubleComplex G2[4];
       cuDoubleComplex T1[4];
       cuDoubleComplex T2[4];

       int px=(n+boff)/((Nbasetotal+nchunk-1)/nchunk);

       G1[0].x=p[pstart+px*8*Ns+sta1*8];
       G1[0].y=p[pstart+px*8*Ns+sta1*8+1];
       G1[1].x=p[pstart+px*8*Ns+sta1*8+2];
       G1[1].y=p[pstart+px*8*Ns+sta1*8+3];
       G1[2].x=p[pstart+px*8*Ns+sta1*8+4];
       G1[2].y=p[pstart+px*8*Ns+sta1*8+5];
       G1[3].x=p[pstart+px*8*Ns+sta1*8+6];
       G1[3].y=p[pstart+px*8*Ns+sta1*8+7];
 
       /* conjugate and transpose G2 */
       G2[0].x=p[pstart+px*8*Ns+sta2*8];
       G2[0].y=-p[pstart+px*8*Ns+sta2*8+1];
       G2[2].x=p[pstart+px*8*Ns+sta2*8+2];
       G2[2].y=-p[pstart+px*8*Ns+sta2*8+3];
       G2[1].x=p[pstart+px*8*Ns+sta2*8+4];
       G2[1].y=-p[pstart+px*8*Ns+sta2*8+5];
       G2[3].x=p[pstart+px*8*Ns+sta2*8+6];
       G2[3].y=-p[pstart+px*8*Ns+sta2*8+7];

       /* read in coherency  -- per each chan */
       cuDoubleComplex C[4];
       C[0].x=coh[8*n*M+8*cm  +8*M*Nbase*fi];
       C[0].y=coh[8*n*M+8*cm+1+8*M*Nbase*fi];
       C[1].x=coh[8*n*M+8*cm+2+8*M*Nbase*fi];
       C[1].y=coh[8*n*M+8*cm+3+8*M*Nbase*fi];
       C[2].x=coh[8*n*M+8*cm+4+8*M*Nbase*fi];
       C[2].y=coh[8*n*M+8*cm+5+8*M*Nbase*fi];
       C[3].x=coh[8*n*M+8*cm+6+8*M*Nbase*fi];
       C[3].y=coh[8*n*M+8*cm+7+8*M*Nbase*fi];

 
       /* T1=G1*C */
       T1[0]=cuCadd(cuCmul(G1[0],C[0]),cuCmul(G1[1],C[2]));
       T1[1]=cuCadd(cuCmul(G1[0],C[1]),cuCmul(G1[1],C[3]));
       T1[2]=cuCadd(cuCmul(G1[2],C[0]),cuCmul(G1[3],C[2]));
       T1[3]=cuCadd(cuCmul(G1[2],C[1]),cuCmul(G1[3],C[3]));

       /* T2=T1*G2 , G2 conjugate transposed */
       T2[0]=cuCadd(cuCmul(T1[0],G2[0]),cuCmul(T1[1],G2[2]));
       T2[1]=cuCadd(cuCmul(T1[0],G2[1]),cuCmul(T1[1],G2[3]));
       T2[2]=cuCadd(cuCmul(T1[2],G2[0]),cuCmul(T1[3],G2[2]));
       T2[3]=cuCadd(cuCmul(T1[2],G2[1]),cuCmul(T1[3],G2[3]));

       /* find residul V - J_p C J_q^H  
         -- per channel */
       xr[0]=cuCsub(xr[0],T2[0]);
       xr[1]=cuCsub(xr[1],T2[1]);
       xr[2]=cuCsub(xr[2],T2[2]);
       xr[3]=cuCsub(xr[3],T2[3]);
      }
      /* log(squared) error */
      gsum+=log(1.0+xr[0].x*xr[0].x*inv_robust_nu);
      gsum+=log(1.0+xr[0].y*xr[0].y*inv_robust_nu);
      gsum+=log(1.0+xr[1].x*xr[1].x*inv_robust_nu);
      gsum+=log(1.0+xr[1].y*xr[1].y*inv_robust_nu);
      gsum+=log(1.0+xr[2].x*xr[2].x*inv_robust_nu);
      gsum+=log(1.0+xr[2].y*xr[2].y*inv_robust_nu);
      gsum+=log(1.0+xr[3].x*xr[3].x*inv_robust_nu);
      gsum+=log(1.0+xr[3].y*xr[3].y*inv_robust_nu);

    }
 
    /* robust cost is log( 1 + error^2/nu ) -- per channel */
    ek[tid]=gsum;
  }

  __syncthreads();
  // Build summation tree over elements, assuming blockDim.x*blockDim.y is power of 2.
  for(int s=blockDim.x*blockDim.y/2; s>0; s=s/2) {
    if(tid < s) ek[tid] += ek[tid + s];
   __syncthreads();
  }

  /* copy back to global array */
  if(tid==0) {
   ed[blockIdx.x+gridDim.x*blockIdx.y]=ek[0];
  }

}


/* sum up all N elements of vector input 
 and save (per block) in output (size > number of blocks) */
__global__ void
plus_reduce_multi_mf(const double *__restrict__ input, int N, int blockDim_2, double *__restrict__ output) {
 // Each block loads its elements into shared memory
 extern __shared__ double x[];
 int tid = threadIdx.x;
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 x[tid] = (i<N) ? input[i] : 0.0; // last block may pad with 0’s
 __syncthreads();
 // Build summation tree over elements, handling case where B is not a power of two.
  int nTotalThreads = blockDim_2; // Total number of threads, rounded up to the next power of two
  while(nTotalThreads > 1) {
   int halfPoint = (nTotalThreads >> 1); // divide by two
    if (tid < halfPoint) {
     int thread2 = tid + halfPoint;
     if (thread2 < blockDim.x) { // Skipping the fictitious threads blockDim.x ... blockDim_2-1
      x[tid] = x[tid]+x[thread2];
     }
    }
    __syncthreads();
    nTotalThreads = halfPoint; // Reducing the binary tree size by two
 }

 /* add back to total */
 if( tid == 0 ) {
  output[blockIdx.x]=x[tid];
 }
}

/* sum up all N elements of vector input 
 NOTE: only 1 block should be used */
__global__ void
plus_reduce_mf(const double *__restrict__ input, int N, int blockDim_2, double *total) {
 // Each block loads its elements into shared memory
 extern __shared__ double x[];
 int tid = threadIdx.x;
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 x[tid] = (i<N) ? input[i] : 0.0; // last block may pad with 0’s
 __syncthreads();
 // Build summation tree over elements, handling case where B is not a power of two.
  int nTotalThreads = blockDim_2; // Total number of threads, rounded up to the next power of two
  while(nTotalThreads > 1) {
   int halfPoint = (nTotalThreads >> 1); // divide by two
    if (tid < halfPoint) {
     int thread2 = tid + halfPoint;
     if (thread2 < blockDim.x) { // Skipping the fictitious threads blockDim.x ... blockDim_2-1
      x[tid] = x[tid]+x[thread2];
     }
    }
    __syncthreads();
    nTotalThreads = halfPoint; // Reducing the binary tree size by two
 }

 /* add back to total */
 if( tid == 0 ) {
  *total=*total+x[tid];
 }
}


/* only use extern if calling code is C */
extern "C"
{


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

/* need power of 2 for tree reduction to work */
static int
NearestPowerOf2 (int n){
  if (!n) return n;  //(0 == 2^0)

  int x = 1;
  while(x < n) {
      x <<= 1;
  }
  return x;
}


/* 
   Nbase: no of baselines (total, including tilesz >1)
   tilesz: tile size
   M: no of clusters
   Ns: no of stations
   bb: 2*Nbase x 1
   ptoclus: 2*M x 1

   Nbasetotal: the baselines for the full dataset
   boff: gives the offset of this minibatch in the full batch data
   coh: includes Nchan channels, instead of 1 : Nbase*8*M*Nchan x 1  
   x: data size Nbase*8*Nchan x 1
    ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
    and repeating this for each channel


   grad: mx1 gradient values
   p: mx1 parameters, m :already defined by other variables, this is just to safeguard
   p: M*Ns*8 x 1, can also be Mt*Ns*8x1 for hybrid solutions
*/
void 
cudakernel_lbfgs_multifreq_r_robust(int Nbase, int tilesz, int Nchan, int M, int Ns, int Nbasetotal, int boff, double *x, double *coh, double *p, int m, short *bb, int *ptoclus, double *grad, double robust_nu){
 
  cudaError_t error;
  /* invoke kernel to calculate residuals, per baseline and channel */
  double *eo;
  if((error=cudaMalloc((void**)&eo, Nbase*8*Nchan*sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  cudaMemset(eo, 0, sizeof(double)*Nbase*8*Nchan);
  checkCudaError(error,__FILE__,__LINE__);
  dim3 threadsPerBlock(16,4);
  dim3 blocksPerGrid((Nbase+threadsPerBlock.x-1)/threadsPerBlock.x,(Nchan+threadsPerBlock.y-1)/threadsPerBlock.y);

#ifdef CUDA_DBG
  error = cudaGetLastError(); /* reset all previous errors */
#endif

  kernel_residual_multifreq<<< blocksPerGrid, threadsPerBlock >>> (Nbase, Nchan, boff, M, Ns, Nbasetotal, x, coh, p, bb, ptoclus, eo);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  int ThreadsPerBlock=32;
  int BlocksPerGrid=(m+ThreadsPerBlock-1)/ThreadsPerBlock;

  /* invoke device on this block/thread grid (last argument is buffer size in bytes) */
  kernel_deriv_r_robust<<< BlocksPerGrid, ThreadsPerBlock >>> (Nbase, tilesz, Nchan, M, Ns, m, Nbasetotal, boff, eo, coh, p, bb, ptoclus, grad, robust_nu);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  cudaFree(eo);
}

/* 
   Nbase: baselines for this minibatch (already multiplied by the tilesz)
   Nchan: total channels of data
   Nbasetotal: the baselines for the full dataset
   boff: gives the offset of this minibatch in the full batch data
   coh: includes Nchan channels, instead of 1 : Nbase*8*M*Nchan x 1  
   x: data size Nbase*8*Nchan x 1
    ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
    and repeating this for each channel

   p: mx1 parameters, m :already defined by other variables, this is just to safeguard
 */
double 
cudakernel_lbfgs_multifreq_cost_robust(int Nbase, int Nchan, int M, int Ns, int Nbasetotal, int boff, double *x, double *coh, double *p, int m, short *bb, int *ptoclus, double robust_nu){
 
  double *ed;
  cudaError_t error;

  /* how many blocks needed to cover the minibatch baselines, and frequencies */
  dim3 threadsPerBlock(16,4);
  dim3 blocksPerGrid((Nbase+threadsPerBlock.x-1)/threadsPerBlock.x,(Nchan+threadsPerBlock.y-1)/threadsPerBlock.y);

  int blocksPerGridXY=blocksPerGrid.x*blocksPerGrid.y;
  /* Note that we need 2D ed and shared memory */
  if((error=cudaMalloc((void**)&ed, sizeof(double)*blocksPerGridXY))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  cudaMemset(ed, 0, sizeof(double)*blocksPerGridXY);
  kernel_fcost_multifreq_robust<<< blocksPerGrid, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(double) >>> (Nbase, Nchan, boff, M, Ns, Nbasetotal, x, coh, p, bb, ptoclus, ed, 1.0/robust_nu);

#ifdef CUDA_DBG
  error = cudaGetLastError();
  checkCudaError(error,__FILE__,__LINE__);
#endif

  /* the summing up is done using 1D launches */
  int T=DEFAULT_TH_PER_BK; 
  double *totald,total;
  if((error=cudaMalloc((void**)&totald, sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  cudaMemset(totald, 0, sizeof(double));
  checkCudaError(error,__FILE__,__LINE__);
  if (T>blocksPerGridXY) {
    /* one kernel launch is enough */
    plus_reduce_mf<<< 1, blocksPerGridXY, sizeof(double)*blocksPerGridXY>>>(ed, blocksPerGridXY, NearestPowerOf2(blocksPerGridXY), totald);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    checkCudaError(error,__FILE__,__LINE__);
#endif
  } else {
    /* multiple kernel launches */
    int L=(blocksPerGridXY+T-1)/T;
    double *eo;
    if((error=cudaMalloc((void**)&eo, L*sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    plus_reduce_multi_mf<<< L, T, sizeof(double)*T>>>(ed, blocksPerGridXY, NearestPowerOf2(T), eo);

#ifdef CUDA_DBG
    error = cudaGetLastError();
    checkCudaError(error,__FILE__,__LINE__);
#endif
    plus_reduce_mf<<< 1, L, sizeof(double)*L>>>(eo, L, NearestPowerOf2(L), totald);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    checkCudaError(error,__FILE__,__LINE__);
#endif
    cudaFree(eo);
  }
  cudaMemcpy(&total,totald,sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(totald);
  cudaFree(ed);

  return total;
}


}
