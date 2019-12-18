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

/* enable this for kernel failure detection */
//#define CUDA_DBG

__global__ void kernel_deriv_robust(int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double robust_nu, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad){
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

         /* read residual vector, real,imag separate*/
         double xr[8];
         xr[0]=x[nb*8];
         xr[1]=x[nb*8+1];
         xr[2]=x[nb*8+2];
         xr[3]=x[nb*8+3];
         xr[4]=x[nb*8+4];
         xr[5]=x[nb*8+5];
         xr[6]=x[nb*8+6];
         xr[7]=x[nb*8+7];

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
         if(stc==sta1) {
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
        gsum+=2.0*dsum;     
      } 

     }

    }
    }

    
    grad[n]=gsum;
  }   

}


__global__ void kernel_func_wt(int Nbase, double *x, double *coh, double *p, short *bb, double *wt, int N){
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

__global__ void kernel_jacf_wt(int Nbase, int M, double *jac, double *coh, double *p, short *bb, double *wt, int N){
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

__global__ void kernel_setweights(int N, double *wt, double alpha){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only N threads */
  if (tid<N) {
     wt[tid]=alpha;
  }
}

__global__ void kernel_hadamard(int N, double *wt, double *x){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only N threads */
  if (tid<N) {
     x[tid]*=wt[tid];
  }
}


__global__ void kernel_hadamard_sum(int N, double *y, double *x, double *w){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only N threads */
  if (tid<N) {
     y[tid]+=x[tid]*w[tid];
  }
}

__global__ void kernel_updateweights(int N, double *wt, double *x, double *q, double nu){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only N threads */
  if (tid<N) {
     wt[tid]=((nu+1.0)/(nu+x[tid]*x[tid]));
     q[tid]=wt[tid]-log(wt[tid]); /* so that its +ve */
  }
}

__global__ void kernel_sqrtweights(int N, double *wt){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only N threads */
  if (tid<N) {
     wt[tid]=sqrt(wt[tid]); 
  }
}


__device__ double
digamma(double x) {
  double result = 0.0, xx, xx2, xx4;
  for ( ; x < 7.0; ++x) { /* reduce x till x<7 */
    result -= 1.0/x;
  }
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;
}

__global__ void kernel_evaluatenu(int Nd, double qsum, double *q, double deltanu,double nulow) {
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid<Nd) {
   double thisnu=(nulow+((double)tid)*deltanu);
   double dgm=digamma(thisnu*0.5+0.5);
   q[tid]=dgm-log((thisnu+1.0)*0.5); /* psi((nu+1)/2)-log((nu+1)/2) */
   dgm=digamma(thisnu*0.5);
   q[tid]+=-dgm+log((thisnu)*0.5); /* -psi((nu)/2)+log((nu)/2) */
   q[tid]+=-qsum+1.0; /* -(-sum(ln(w_i))/N+sum(w_i)/N)+1 */
  }
}

/* only use extern if calling code is C */
extern "C"
{

/* set initial weights to 1 by a cuda kernel */
void
cudakernel_setweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double alpha) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_setweights<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt, alpha);
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
cudakernel_hadamard(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_hadamard<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt, x);
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


/* sum hadamard product by a cuda kernel y=y+x.*w (x.*w elementwise) */
void
cudakernel_hadamard_sum(int ThreadsPerBlock, int BlocksPerGrid, int N, double *y, double *x, double *w) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_hadamard_sum<<< BlocksPerGrid, ThreadsPerBlock >>>(N, y, x, w);
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
cudakernel_updateweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x, double *q, double robust_nu) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_updateweights<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt, x, q, robust_nu);
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
cudakernel_sqrtweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_sqrtweights<<< BlocksPerGrid, ThreadsPerBlock >>>(N, wt);
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
cudakernel_evaluatenu(int ThreadsPerBlock, int BlocksPerGrid, int Nd, double qsum, double *q, double deltanu,double nulow) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_evaluatenu<<< BlocksPerGrid, ThreadsPerBlock >>>(Nd, qsum, q, deltanu, nulow);
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
cudakernel_func_wt(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cudaMemset(x, 0, N*sizeof(double));
//  printf("Kernel data size=%d, block=%d, thread=%d, baselines=%d\n",N,BlocksPerGrid, ThreadsPerBlock,Nbase);
  kernel_func_wt<<< BlocksPerGrid, ThreadsPerBlock >>>(Nbase,  x, coh, p, bbh, wt, Nstations);
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
cudakernel_jacf_wt(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations, int clus) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  /* NOTE: use small value for ThreadsPerBlock here, like 8 */
  dim3 threadsPerBlock(16, 8);
  /* jacobian: Nbase x Nstations (proportional to N), so */
  dim3 numBlocks((Nbase+threadsPerBlock.x-1)/threadsPerBlock.x, 
               (M+threadsPerBlock.y-1)/threadsPerBlock.y);
  /* set memory of jac to zero */
  cudaMemset(jac, 0, N*M*sizeof(double));
 // printf("Kernel Jax data size=%d, params=%d, block=%d,%d, thread=%d,%d, baselines=%d\n",N, M, numBlocks.x,numBlocks.y, threadsPerBlock.x, threadsPerBlock.y, Nbase);
  kernel_jacf_wt<<< numBlocks, threadsPerBlock>>>(Nbase,  M, jac, coh, p, bbh, wt, Nstations);

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

/* cuda driver for kernel */
/* ThreadsPerBlock: keep <= 128 ???
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
void cudakernel_lbfgs_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double robust_nu, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad){

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  /* invoke device on this block/thread grid (last argument is buffer size in bytes) */
  kernel_deriv_robust<<< BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock*sizeof(double) >>> (Nbase, tilesz, M, Ns, Nparam, goff, robust_nu, x, coh, p, bb, ptoclus, grad);
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
