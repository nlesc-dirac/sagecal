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
#include "GPUtune.h"

/* enable this for checking for kernel failure */
//#define CUDA_DBG


__global__ void 
kernel_deriv(int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ grad){
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


/* note x is residual, not data */
__global__ void 
kernel_deriv_r_robust(int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ grad, double robust_nu){
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

         /* read residual vector */
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

    
    grad[n]=gsum;
  }  

}


/* note x is residual, not data */
__global__ void 
kernel_deriv_r(int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ grad){
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


         /* calculate product xr*vec(J_p C J_q^H ) */
         cuDoubleComplex csum;
         csum=cuCmul(xr[0],T2[0]);
         csum=cuCadd(csum,cuCmul(xr[1],T2[1]));
         csum=cuCadd(csum,cuCmul(xr[2],T2[2]));
         csum=cuCadd(csum,cuCmul(xr[3],T2[3]));


        /* notice no -ve sign */
        gsum+=2.0*csum.x;     
      } 

     }

    }
    }

    
    grad[n]=gsum;
  }   

}


__global__ void 
kernel_residual(int Nbase, int M, int Ns, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ ed){

  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;

  if (n<Nbase) {
    int sta1=(int)bb[2*n];
    int sta2=(int)bb[2*n+1];

    /* only calculate deriv if baseline corresponds
    to this station and baseline is not flagged */
    /* flagged baselines will have sta1==sta2==-1 */
    if (sta1>=0 && sta2>=0) {
      /* read data vector */
      cuDoubleComplex xr[4];
      xr[0].x=x[n*8];
      xr[0].y=x[n*8+1];
      xr[1].x=x[n*8+2];
      xr[1].y=x[n*8+3];
      xr[2].x=x[n*8+4];
      xr[2].y=x[n*8+5];
      xr[3].x=x[n*8+6];
      xr[3].y=x[n*8+7];

      for (int cm=0; cm<M; cm++) {
       int pstart=ptoclus[2*cm+1];
       int nchunk=ptoclus[2*cm];
       /* read in coherency */
       cuDoubleComplex C[4];
       C[0].x=coh[8*n*M+8*cm];
       C[0].y=coh[8*n*M+8*cm+1];
       C[1].x=coh[8*n*M+8*cm+2];
       C[1].y=coh[8*n*M+8*cm+3];
       C[2].x=coh[8*n*M+8*cm+4];
       C[2].y=coh[8*n*M+8*cm+5];
       C[3].x=coh[8*n*M+8*cm+6];
       C[3].y=coh[8*n*M+8*cm+7];
         
       cuDoubleComplex G1[4];
       cuDoubleComplex G2[4];
       cuDoubleComplex T1[4];
       cuDoubleComplex T2[4];

       int px=(n)/((Nbase+nchunk-1)/nchunk);

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

      ed[n*8]=xr[0].x;
      ed[n*8+1]=xr[0].y;
      ed[n*8+2]=xr[1].x;
      ed[n*8+3]=xr[1].y;
      ed[n*8+4]=xr[2].x;
      ed[n*8+5]=xr[2].y;
      ed[n*8+6]=xr[3].x;
      ed[n*8+7]=xr[3].y;

    }
  } 

}


__global__ void 
kernel_fcost_robust(int Nbase, int boff, int M, int Ns, int Nbasetotal, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ ed, double inv_robust_nu){
  /* shared memory */
  extern __shared__ double ek[];

  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int tid=threadIdx.x;
  ek[tid]=0.0;

  if (n<Nbase) {
    double gsum=0.0;
    int sta1=(int)bb[2*n];
    int sta2=(int)bb[2*n+1];

    /* only calculate deriv if baseline corresponds
    to this station and baseline is not flagged */
    /* flagged baselines will have sta1==sta2==-1 */
    if (sta1>=0 && sta2>=0) {
       /* read data vector */
       cuDoubleComplex xr[4];
       xr[0].x=x[n*8];
       xr[0].y=x[n*8+1];
       xr[1].x=x[n*8+2];
       xr[1].y=x[n*8+3];
       xr[2].x=x[n*8+4];
       xr[2].y=x[n*8+5];
       xr[3].x=x[n*8+6];
       xr[3].y=x[n*8+7];


      for (int cm=0; cm<M; cm++) {
       int pstart=ptoclus[2*cm+1];
       int nchunk=ptoclus[2*cm];
       /* read in coherency */
       cuDoubleComplex C[4];
       C[0].x=coh[8*n*M+8*cm];
       C[0].y=coh[8*n*M+8*cm+1];
       C[1].x=coh[8*n*M+8*cm+2];
       C[1].y=coh[8*n*M+8*cm+3];
       C[2].x=coh[8*n*M+8*cm+4];
       C[2].y=coh[8*n*M+8*cm+5];
       C[3].x=coh[8*n*M+8*cm+6];
       C[3].y=coh[8*n*M+8*cm+7];
         
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
      /* squared error */
      gsum+=xr[0].x*xr[0].x+xr[0].y*xr[0].y;
      gsum+=xr[1].x*xr[1].x+xr[1].y*xr[1].y;
      gsum+=xr[2].x*xr[2].x+xr[2].y*xr[2].y;
      gsum+=xr[3].x*xr[3].x+xr[3].y*xr[3].y;
    }
 
    /* robust cost is log( 1 + error^2/nu ) */
    ek[tid]=log(1.0+gsum*inv_robust_nu);
  } 

  __syncthreads();
  // Build summation tree over elements, assuming blockDim.x is power of 2.
  for(int s=blockDim.x/2; s>0; s=s/2) {
    if(tid < s) ek[tid] += ek[tid + s];
   __syncthreads();
  }

  /* copy back to global array */
  if(tid==0) {
   ed[blockIdx.x]=ek[0];
  }

}


__global__ void 
kernel_fcost(int Nbase, int boff, int M, int Ns, int Nbasetotal, const double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, const int *__restrict__ ptoclus, double *__restrict__ ed){
  /* shared memory */
  extern __shared__ double ek[];

  /* global thread index */
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int tid=threadIdx.x;
  ek[tid]=0.0;

  if (n<Nbase) {
    double gsum=0.0;
    int sta1=(int)bb[2*n];
    int sta2=(int)bb[2*n+1];

    /* only calculate deriv if baseline corresponds
    to this station and baseline is not flagged */
    /* flagged baselines will have sta1==sta2==-1 */
    if (sta1>=0 && sta2>=0) {
       /* read data vector */
       cuDoubleComplex xr[4];
       xr[0].x=x[n*8];
       xr[0].y=x[n*8+1];
       xr[1].x=x[n*8+2];
       xr[1].y=x[n*8+3];
       xr[2].x=x[n*8+4];
       xr[2].y=x[n*8+5];
       xr[3].x=x[n*8+6];
       xr[3].y=x[n*8+7];


      for (int cm=0; cm<M; cm++) {
       int pstart=ptoclus[2*cm+1];
       int nchunk=ptoclus[2*cm];
       /* read in coherency */
       cuDoubleComplex C[4];
       C[0].x=coh[8*n*M+8*cm];
       C[0].y=coh[8*n*M+8*cm+1];
       C[1].x=coh[8*n*M+8*cm+2];
       C[1].y=coh[8*n*M+8*cm+3];
       C[2].x=coh[8*n*M+8*cm+4];
       C[2].y=coh[8*n*M+8*cm+5];
       C[3].x=coh[8*n*M+8*cm+6];
       C[3].y=coh[8*n*M+8*cm+7];
         
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
      /* squared error */
      gsum+=xr[0].x*xr[0].x+xr[0].y*xr[0].y;
      gsum+=xr[1].x*xr[1].x+xr[1].y*xr[1].y;
      gsum+=xr[2].x*xr[2].x+xr[2].y*xr[2].y;
      gsum+=xr[3].x*xr[3].x+xr[3].y*xr[3].y;

    }
 
    ek[tid]=gsum;
  } 

  __syncthreads();
  // Build summation tree over elements, assuming blockDim.x is power of 2.
  for(int s=blockDim.x/2; s>0; s=s/2) {
    if(tid < s) ek[tid] += ek[tid + s];
   __syncthreads();
  }

  /* copy back to global array */
  if(tid==0) {
   ed[blockIdx.x]=ek[0];
  }

}


__global__ void 
kernel_diagdiv(int M, double eps, double *__restrict__ y,const double *__restrict__ x){
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

__global__ void 
kernel_diagmu(int M, double *__restrict__ A,double mu){
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  /* make sure to use only M threads */
  if (tid<M) {
    A[tid*(M+1)]=A[tid*(M+1)]+mu;
  } 
}


__global__ void 
kernel_func(int Nbase, double *__restrict__ x, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, int N){
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

__global__ void 
kernel_jacf(int Nbase, int M, double *__restrict__ jac, const double *__restrict__ coh, const double *__restrict__ p, const short *__restrict__ bb, int N){
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


/* sum up all N elements of vector input 
 and save (per block) in output (size > number of blocks) */
__global__ void
plus_reduce_multi(const double *__restrict__ input, int N, int blockDim_2, double *__restrict__ output) {
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
plus_reduce(const double *__restrict__ input, int N, int blockDim_2, double *total) {
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
void 
cudakernel_lbfgs(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad){
 
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  /* invoke device on this block/thread grid (last argument is buffer size in bytes) */
  kernel_deriv<<< BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock*sizeof(double) >>> (Nbase, tilesz, M, Ns, Nparam, goff, x, coh, p, bb, ptoclus, grad);
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

void 
cudakernel_lbfgs_r_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad, double robust_nu){
 
  cudaError_t error;
  /* invoke kernel to calculate residuals first */
  double *eo;
  if((error=cudaMalloc((void**)&eo, Nbase*8*sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  cudaMemset(eo, 0, sizeof(double)*Nbase*8);
  checkCudaError(error,__FILE__,__LINE__);

  int L=(Nbase+ThreadsPerBlock-1)/ThreadsPerBlock;
#ifdef CUDA_DBG
  error = cudaGetLastError(); /* reset all previous errors */
#endif

  kernel_residual<<< L, ThreadsPerBlock >>> (Nbase, M, Ns, x, coh, p, bb, ptoclus, eo);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* invoke device on this block/thread grid (last argument is buffer size in bytes) */
  kernel_deriv_r_robust<<< BlocksPerGrid, ThreadsPerBlock >>> (Nbase, tilesz, M, Ns, Nparam, goff, eo, coh, p, bb, ptoclus, grad, robust_nu);
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

void 
cudakernel_lbfgs_r(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad){
 
  cudaError_t error;
  /* invoke kernel to calculate residuals first */
  double *eo;
  if((error=cudaMalloc((void**)&eo, Nbase*8*sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  cudaMemset(eo, 0, sizeof(double)*Nbase*8);
  checkCudaError(error,__FILE__,__LINE__);
  int L=(Nbase+ThreadsPerBlock-1)/ThreadsPerBlock;

#ifdef CUDA_DBG
  error = cudaGetLastError(); /* reset all previous errors */
#endif

  kernel_residual<<< L, ThreadsPerBlock >>> (Nbase, M, Ns, x, coh, p, bb, ptoclus, eo);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  if(error != cudaSuccess) {
    // print the CUDA error message and exit
    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
    exit(-1);
  }
#endif

  /* invoke device on this block/thread grid (last argument is buffer size in bytes) */
  kernel_deriv_r<<< BlocksPerGrid, ThreadsPerBlock >>> (Nbase, tilesz, M, Ns, Nparam, goff, eo, coh, p, bb, ptoclus, grad);
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

/* note x,coh and bb are with the right offset */
double 
cudakernel_lbfgs_cost_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus, double robust_nu){
 
  double *ed;
  cudaError_t error;
  if((error=cudaMalloc((void**)&ed, sizeof(double)*BlocksPerGrid))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  cudaMemset(ed, 0, sizeof(double)*BlocksPerGrid);
  kernel_fcost_robust<<< BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock*sizeof(double) >>> (Nbase, boff, M, Ns, Nbasetotal, x, coh, p, bb, ptoclus, ed, 1.0/robust_nu);

#ifdef CUDA_DBG
  error = cudaGetLastError();
  checkCudaError(error,__FILE__,__LINE__);
#endif

  int T=DEFAULT_TH_PER_BK; 
  double *totald,total;
  if((error=cudaMalloc((void**)&totald, sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  cudaMemset(totald, 0, sizeof(double));
  checkCudaError(error,__FILE__,__LINE__);


  if (T>BlocksPerGrid) {
    /* one kernel launch is enough */
    plus_reduce<<< 1, BlocksPerGrid, sizeof(double)*BlocksPerGrid>>>(ed, BlocksPerGrid, NearestPowerOf2(BlocksPerGrid), totald);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    checkCudaError(error,__FILE__,__LINE__);
#endif
  } else {
    /* multiple kernel launches */
    int L=(BlocksPerGrid+T-1)/T;
    double *eo;
    if((error=cudaMalloc((void**)&eo, L*sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    plus_reduce_multi<<< L, T, sizeof(double)*T>>>(ed, BlocksPerGrid, NearestPowerOf2(T), eo);

#ifdef CUDA_DBG
    error = cudaGetLastError();
    checkCudaError(error,__FILE__,__LINE__);
#endif
    plus_reduce<<< 1, L, sizeof(double)*L>>>(eo, L, NearestPowerOf2(L), totald);
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

double 
cudakernel_lbfgs_cost(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus){
  double *ed;
  cudaError_t error;
  if((error=cudaMalloc((void**)&ed, sizeof(double)*BlocksPerGrid))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  cudaMemset(ed, 0, sizeof(double)*BlocksPerGrid);
  kernel_fcost<<< BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock*sizeof(double) >>> (Nbase, boff, M, Ns, Nbasetotal, x, coh, p, bb, ptoclus, ed);
#ifdef CUDA_DBG
  error = cudaGetLastError();
  checkCudaError(error,__FILE__,__LINE__);
#endif

  int T=DEFAULT_TH_PER_BK; 
  double *totald,total;
  if((error=cudaMalloc((void**)&totald, sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  cudaMemset(totald, 0, sizeof(double));
  checkCudaError(error,__FILE__,__LINE__);


  if (T>BlocksPerGrid) {
    /* one kernel launch is enough */
    plus_reduce<<< 1, BlocksPerGrid, sizeof(double)*BlocksPerGrid>>>(ed, BlocksPerGrid, NearestPowerOf2(BlocksPerGrid), totald);

#ifdef CUDA_DBG
    error = cudaGetLastError();
    checkCudaError(error,__FILE__,__LINE__);
#endif
  } else {
    /* multiple kernel launches */
    int L=(BlocksPerGrid+T-1)/T;
    double *eo;
    if((error=cudaMalloc((void**)&eo, L*sizeof(double)))!=cudaSuccess) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    plus_reduce_multi<<< L, T, sizeof(double)*T>>>(ed, BlocksPerGrid, NearestPowerOf2(T), eo);
#ifdef CUDA_DBG
    error = cudaGetLastError();
    checkCudaError(error,__FILE__,__LINE__);
#endif
    plus_reduce<<< 1, L, sizeof(double)*L>>>(eo, L, NearestPowerOf2(L), totald);
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



/* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps */
void 
cudakernel_diagdiv(int ThreadsPerBlock, int BlocksPerGrid, int M, double eps, double *Dpd, double *Sd) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_diagdiv<<< BlocksPerGrid, ThreadsPerBlock >>>(M, eps, Dpd, Sd);
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

/* cuda driver for calculating
  A<= A+mu I, adding mu to diagonal entries of A
  A: size MxM
  ThreadsPerBlock, BlocksPerGrid calculated to meet M
*/
void
cudakernel_diagmu(int ThreadsPerBlock, int BlocksPerGrid, int M, double *A, double mu) {
#ifdef CUDA_DBG
  cudaError_t error;
#endif
  kernel_diagmu<<< BlocksPerGrid, ThreadsPerBlock >>>(M, A, mu);
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


/* cuda driver for calculating f() */
/* p: params (Mx1), x: data (Nx1), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
void
cudakernel_func(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations) {

#ifdef CUDA_DBG
  cudaError_t error;
#endif
  cudaMemset(x, 0, N*sizeof(double));
//  printf("Kernel data size=%d, block=%d, thread=%d, baselines=%d\n",N,BlocksPerGrid, ThreadsPerBlock,Nbase);
  kernel_func<<< BlocksPerGrid, ThreadsPerBlock >>>(Nbase,  x, coh, p, bbh, Nstations);
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

/* cuda driver for calculating jacf() */
/* p: params (Mx1), jac: jacobian (NxM), other data : coh, baseline->stat mapping, Nbase, Mclusters, Nstations */
void
cudakernel_jacf(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations) {

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
  kernel_jacf<<< numBlocks, threadsPerBlock>>>(Nbase,  M, jac, coh, p, bbh, Nstations);
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
