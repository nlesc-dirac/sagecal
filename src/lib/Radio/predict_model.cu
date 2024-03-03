/*
 *
 Copyright (C) 2014 Sarod Yatawatta <sarod@users.sf.net>  
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
//#define CUDA_DBG

/* matrix multiplications */
/* C=A*B */
__device__ void
amb(const cuDoubleComplex *__restrict__ a, const cuDoubleComplex *__restrict__ b, cuDoubleComplex *__restrict__ c) {
 c[0]=cuCadd(cuCmul(a[0],b[0]),cuCmul(a[1],b[2]));
 c[1]=cuCadd(cuCmul(a[0],b[1]),cuCmul(a[1],b[3]));
 c[2]=cuCadd(cuCmul(a[2],b[0]),cuCmul(a[3],b[2]));
 c[3]=cuCadd(cuCmul(a[2],b[1]),cuCmul(a[3],b[3]));
}
/* C=A*B^H */
__device__ void
ambt(const cuDoubleComplex *__restrict__ a, const cuDoubleComplex *__restrict__ b, cuDoubleComplex *__restrict__ c) {
 c[0]=cuCadd(cuCmul(a[0],cuConj(b[0])),cuCmul(a[1],cuConj(b[1])));
 c[1]=cuCadd(cuCmul(a[0],cuConj(b[2])),cuCmul(a[1],cuConj(b[3])));
 c[2]=cuCadd(cuCmul(a[2],cuConj(b[0])),cuCmul(a[3],cuConj(b[1])));
 c[3]=cuCadd(cuCmul(a[2],cuConj(b[2])),cuCmul(a[3],cuConj(b[3])));
}


__device__ void
radec2azel_gmst__(float ra, float dec, float longitude, float latitude, float thetaGMST, float *az, float *el) {
  float thetaLST=thetaGMST+longitude*180.0f/M_PIf;

  float LHA=fmodf(thetaLST-ra*180.0f/M_PIf,360.0f);

  float sinlat,coslat,sindec,cosdec,sinLHA,cosLHA;
  sincosf(latitude,&sinlat,&coslat);
  sincosf(dec,&sindec,&cosdec);
  sincosf(LHA*M_PIf/180.0f,&sinLHA,&cosLHA);

  float tmp=sinlat*sindec+coslat*cosdec*cosLHA;
  float eld=asinf(tmp);

  float sinel,cosel;
  sincosf(eld,&sinel,&cosel);

  float azd=fmodf(atan2f(-sinLHA*cosdec/cosel,(sindec-sinel*sinlat)/(cosel*coslat)),2.0f*M_PIf);
  if (azd<0.0f) {
   azd+=2.0f*M_PIf;
  }
  *el=eld;
  *az=azd;
}

/* generalized Laguerre polynomial L_p^q(x) */
/* for calculating L_{n-|m|/2}^|m| (x) */
__device__ float
L_g1(int p, int q, float x) {
  /* max p: (n-|m|)/2 = n/2 */
  if(p==0) return 1.0f;
  if(p==1) return 1.0f-x+(float)q;
  /* else, use two variables to store past values */
  float L_p=0.0f,L_p_1,L_p_2;
  L_p_2=1.0f;
  L_p_1=1.0f-x+(float)q;
  for (int i=2; i<=p; i++) {
   float p_1=1.0f/(float)i;
   L_p=(2.0f+p_1*((float)q-1.0f-x))*L_p_1-(1.0f+p_1*(q-1))*L_p_2;
   L_p_2=L_p_1;
   L_p_1=L_p;
  }
  return L_p;
}

/* evaluate element value using coefficient arrays */
__device__ float4
eval_elementcoeff(float r, float theta, int M, float beta, const float2 *pattern_theta,
     const float2 *pattern_phi, const float *pattern_preamble) {
  float4 eval={0.f,0.f,0.f,0.f};
  float rb=powf(r/beta,2);
  float ex=expf(-0.5f*rb);

  int idx=0;
  for (int n=0; n<M; n++) {
    for (int m=-n; m<=n; m+=2) {
      int absm=m>0?m:-m; /* |m| */
      float Lg=L_g1((n-absm)/2,absm,rb);
      float rm=powf(M_PI_4f+r,(float)absm);
      float s,c;
      sincosf(-(float)m*theta,&s,&c);
      float pr=rm*Lg*ex*pattern_preamble[idx];
      float2 reim;
      reim.x=pr*c;
      reim.y=pr*s;
      float2 prod1=cuCmulf(pattern_theta[idx],reim);
      float2 prod2=cuCmulf(pattern_phi[idx],reim);
      eval.x+=prod1.x;
      eval.y+=prod1.y;
      eval.z+=prod2.x;
      eval.w+=prod2.y;
      idx++;
    }
  }

  return eval;
}

/* use compiler directives to use/not use shared memory */
/* ARRAY_MAX_ELEM : define this in header file beforehand, if using shared memory */
/* master kernel to calculate beam */
__global__ void 
kernel_array_beam(int N, int T, int K, int F, 
 const double *__restrict__ freqs, const float *__restrict__ longitude, const float *__restrict__ latitude,
 const double *__restrict__ time_utc, const int *__restrict__ Nelem, 
 const float * const *__restrict__ xx, const float * const *__restrict__ yy, const float * const *__restrict__ zz, 
 const float *__restrict__ ra, const float *__restrict__ dec, 
 float ph_ra0, float ph_dec0, float ph_freq0, float *beam, const int wideband) {

    /* global thread index, x-dimension (data) */
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    /* y-dimension is station */
    int istat=blockIdx.y;

    // find respective source,freq,time for this thread
    int n1 = x;
    int isrc=n1/(T*F);
    n1=n1-isrc*(T*F);
    int ifrq=n1/(T);
    n1=n1-ifrq*(T);
    int itm=n1;

    //number of elements for this station
    int Nelems = __ldg(&Nelem[istat]);

    //using shared memory
    #if (ARRAY_USE_SHMEM==1)
      __shared__ float sh_x[ARRAY_MAX_ELEM];
      __shared__ float sh_y[ARRAY_MAX_ELEM];
      __shared__ float sh_z[ARRAY_MAX_ELEM];
      for (int i=threadIdx.x; i<Nelems; i+=blockDim.x) {
        sh_x[i] = __ldg(&xx[istat][i]);
        sh_y[i] = __ldg(&yy[istat][i]);
        sh_z[i] = __ldg(&zz[istat][i]);
      }
      __syncthreads();
    #endif

  float r1,r2,r3;
  //check data limit
  if (x<(K*T*F)) {

/*********************************************************************/
   /* time is already converted to thetaGMST */
   float thetaGMST=(float)__ldg(&time_utc[itm]);
   /* find az,el */
   float az,el,az0,el0,theta,phi,theta0,phi0;
   radec2azel_gmst__(__ldg(&ra[isrc]),__ldg(&dec[isrc]), __ldg(&longitude[istat]), __ldg(&latitude[istat]), thetaGMST, &az, &el);
   radec2azel_gmst__(ph_ra0,ph_dec0, __ldg(&longitude[istat]), __ldg(&latitude[istat]), thetaGMST, &az0, &el0);
   /* transform : theta = 90-el, phi=-az? 45 only needed for element beam */
   theta=M_PI_2f-el;
   phi=-az; /* */
   theta0=M_PI_2f-el0;
   phi0=-az0; /* */
/*********************************************************************/
   if (el>=0.0f) {
   /* 2*PI/C */
   const float tpc=2.0f*M_PIf/CONST_C;
   float sint,cost,sinph,cosph,sint0,cost0,sinph0,cosph0;
   sincosf(theta,&sint,&cost);
   sincosf(phi,&sinph,&cosph);
   sincosf(theta0,&sint0,&cost0);
   sincosf(phi0,&sinph0,&cosph0);

   /*r1=(float)-tpc*(ph_freq0*sint0*cosph0-freqs[ifrq]*sint*cosph);
   r2=(float)-tpc*(ph_freq0*sint0*sinph0-freqs[ifrq]*sint*sinph);
   r3=(float)-tpc*(ph_freq0*cost0-freqs[ifrq]*cost);
   */
   float f=(float)__ldg(&freqs[ifrq]);
   // use channel freq for per-channel beamformer (wideband data)
   float beam_freq=(!wideband?ph_freq0:f);
   float rat1=beam_freq*sint0;
   float rat2=f*sint;
   r1=-tpc*(rat1*cosph0-rat2*cosph);
   r2=-tpc*(rat1*sinph0-rat2*sinph);
   r3=-tpc*(beam_freq*cost0-f*cost);
   

        float ssum = 0.0f;
        float csum = 0.0f;
        for (int i=0; i<Nelems; i++) {
            float ss,cc;
            #if (ARRAY_USE_SHMEM == 0)
            sincosf((r1*__ldg(&xx[istat][i])+r2*__ldg(&yy[istat][i])+r3*__ldg(&zz[istat][i])),&ss,&cc);
            #else
            sincosf(r1*sh_x[i]+r2*sh_y[i]+r3*sh_z[i],&ss,&cc);
            #endif
            ssum += ss;
            csum += cc;
        }



   float Nnor=1.0f/(float)Nelems;
   /* store output (amplitude of beam)*/
   int boffset=itm*N*K*F+isrc*N*F+ifrq*N+istat;
   beam[boffset]=sqrtf(ssum*ssum+csum*csum)*Nnor;
   } else {
   int boffset=itm*N*K*F+isrc*N*F+ifrq*N+istat;
   beam[boffset]=0.0f;
   }
   //printf("thread %d stat %d src %d freq %d time %d : %lf longitude=%lf latitude=%lf time=%lf freq=%lf elem=%d ra=%lf dec=%lf beam=%lf\n",n,istat,isrc,ifrq,itm,time_utc[itm],longitude[istat],latitude[istat],time_utc[itm],freqs[ifrq],Nelem[istat],ra[isrc],dec[isrc],beam[boffset]);
  }

}

/* similar to kernel_array_beam, except using two stage beamformer,
   first the tile beamformer, next the full beamformer using tile centroids */
__global__ void
kernel_tile_array_beam(int N, int T, int K, int F,
 const double *__restrict__ freqs, const float *__restrict__ longitude, const float *__restrict__ latitude,
 const double *__restrict__ time_utc, const int *__restrict__ Nelem,
 const float * const *__restrict__ xx, const float * const *__restrict__ yy, const float * const *__restrict__ zz,
 const float *__restrict__ ra, const float *__restrict__ dec,
 float b_ra0, float b_dec0, float ph_ra0, float ph_dec0, float ph_freq0, float *beam, const int wideband) {

    /* global thread index, x-dimension (data) */
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    /* y-dimension is station */
    int istat=blockIdx.y;

    // find respective source,freq,time for this thread
    int n1 = x;
    int isrc=n1/(T*F);
    n1=n1-isrc*(T*F);
    int ifrq=n1/(T);
    n1=n1-ifrq*(T);
    int itm=n1;

    //number of elements (tiles) for this station
    //note that the first HBA_TILE_SIZE values are the dipole locations,
    //and the rest Nelems are tile locations
    int Nelems = __ldg(&Nelem[istat]);

    //using shared memory
    #if (ARRAY_USE_SHMEM==1)
      __shared__ float sh_x[ARRAY_MAX_ELEM];
      __shared__ float sh_y[ARRAY_MAX_ELEM];
      __shared__ float sh_z[ARRAY_MAX_ELEM];
      for (int i=threadIdx.x; i<Nelems+HBA_TILE_SIZE; i+=blockDim.x) {
        sh_x[i] = __ldg(&xx[istat][i]);
        sh_y[i] = __ldg(&yy[istat][i]);
        sh_z[i] = __ldg(&zz[istat][i]);
      }
      __syncthreads();
    #endif

  float r1,r2,r3;
  //check data limit
  if (x<(K*T*F)) {

/*********************************************************************/
   /* time is already converted to thetaGMST */
   float thetaGMST=(float)__ldg(&time_utc[itm]);
   /* find az,el */
   float az,el,az0,el0,az_b,el_b,theta,phi,theta0,phi0,theta_b,phi_b;
   radec2azel_gmst__(__ldg(&ra[isrc]),__ldg(&dec[isrc]), __ldg(&longitude[istat]), __ldg(&latitude[istat]), thetaGMST, &az, &el);
   radec2azel_gmst__(ph_ra0,ph_dec0, __ldg(&longitude[istat]), __ldg(&latitude[istat]), thetaGMST, &az0, &el0);
   radec2azel_gmst__(b_ra0,b_dec0, __ldg(&longitude[istat]), __ldg(&latitude[istat]), thetaGMST, &az_b, &el_b);
   /* transform : theta = 90-el, phi=-az? 45 only needed for element beam */
   theta=M_PI_2f-el;
   phi=-az; /* */
   theta0=M_PI_2f-el0;
   phi0=-az0; /* */
   theta_b=M_PI_2f-el_b;
   phi_b=-az_b;
/*********************************************************************/
   if (el>=0.0f) {
   /* 2*PI/C */
   const float tpc=2.0f*M_PIf/CONST_C;
   float sint,cost,sinph,cosph,sint0,cost0,sinph0,cosph0;
   sincosf(theta,&sint,&cost);
   sincosf(phi,&sinph,&cosph);
   sincosf(theta0,&sint0,&cost0);
   sincosf(phi0,&sinph0,&cosph0);

   /*r1=(float)-tpc*(ph_freq0*sint0*cosph0-freqs[ifrq]*sint*cosph);
   r2=(float)-tpc*(ph_freq0*sint0*sinph0-freqs[ifrq]*sint*sinph);
   r3=(float)-tpc*(ph_freq0*cost0-freqs[ifrq]*cost);
   */
   float f=(float)__ldg(&freqs[ifrq]);
   // use channel freq for per-channel beamformer (wideband data)
   float beam_freq=(!wideband?ph_freq0:f);
   float rat1=beam_freq*sint0;
   float rat2=f*sint;
   r1=-tpc*(rat1*cosph0-rat2*cosph);
   r2=-tpc*(rat1*sinph0-rat2*sinph);
   r3=-tpc*(beam_freq*cost0-f*cost);

   /* full beamformer using tile centroids */
        float ssum = 0.0f;
        float csum = 0.0f;
        for (int i=0; i<Nelems; i++) {
            float ss,cc;
            #if (ARRAY_USE_SHMEM == 0)
            sincosf((r1*__ldg(&xx[istat][i+HBA_TILE_SIZE])+r2*__ldg(&yy[istat][i+HBA_TILE_SIZE])+r3*__ldg(&zz[istat][i+HBA_TILE_SIZE])),&ss,&cc);
            #else
            sincosf(r1*sh_x[i+HBA_TILE_SIZE]+r2*sh_y[i+HBA_TILE_SIZE]+r3*sh_z[i+HBA_TILE_SIZE],&ss,&cc);
            #endif
            ssum += ss;
            csum += cc;
        }
   /* normalization: num tiles x tile size */
   float Nnor=1.0f/(float)(Nelems*HBA_TILE_SIZE);

   /* tile beamformer using element positions */
   sincosf(theta_b,&sint0,&cost0);
   sincosf(phi_b,&sinph0,&cosph0);
   rat1=beam_freq*sint0;
   r1=-tpc*(rat1*cosph0-rat2*cosph);
   r2=-tpc*(rat1*sinph0-rat2*sinph);
   r3=-tpc*(beam_freq*cost0-f*cost);
        float ssum_b = 0.0f;
        float csum_b = 0.0f;
        for (int i=0; i<HBA_TILE_SIZE; i++) {
            float ss,cc;
            #if (ARRAY_USE_SHMEM == 0)
            sincosf((r1*__ldg(&xx[istat][i])+r2*__ldg(&yy[istat][i])+r3*__ldg(&zz[istat][i])),&ss,&cc);
            #else
            sincosf(r1*sh_x[i]+r2*sh_y[i]+r3*sh_z[i],&ss,&cc);
            #endif
            ssum_b += ss;
            csum_b += cc;
        }

   /* store output (amplitude of beam)*/
   int boffset=itm*N*K*F+isrc*N*F+ifrq*N+istat;
   beam[boffset]=sqrtf(ssum*ssum+csum*csum)*sqrtf(ssum_b*ssum_b+csum_b*csum_b)*Nnor;
   } else {
   int boffset=itm*N*K*F+isrc*N*F+ifrq*N+istat;
   beam[boffset]=0.0f;
   }
  }

}


__global__ void 
kernel_element_beam(int N, int T, int K, int F, 
 const double *__restrict__ freqs, const float *__restrict__ longitude, const float *__restrict__ latitude,
 const double *__restrict__ time_utc,
 const float *__restrict__ ra, const float *__restrict__ dec, 
 int Nmodes, int M, float beta, const float *__restrict__ pattern_phi,
 const float *__restrict__ pattern_theta,const float *__restrict__ pattern_preamble,
 float *beam, const int wideband) {

    /* global thread index, x-dimension (data) */
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    /* y-dimension is station */
    int istat=blockIdx.y;

    // find respective source,freq,time for this thread
    int n1 = x;
    int isrc=n1/(T*F);
    n1=n1-isrc*(T*F);
    int ifrq=n1/(T);
    n1=n1-ifrq*(T);
    int itm=n1;

     //using shared memory (not for wideband model)
    #if (ARRAY_USE_SHMEM==1)
      __shared__ cuFloatComplex sh_phi[ELEMENT_MAX_SIZE];
      __shared__ cuFloatComplex sh_theta[ELEMENT_MAX_SIZE];
      __shared__ float sh_preamble[ELEMENT_MAX_SIZE];
      if (!wideband) {
       for (int i=threadIdx.x; i<Nmodes; i+=blockDim.x) {
        sh_phi[i].x = __ldg(&pattern_phi[2*i]);
        sh_phi[i].y =__ldg(&pattern_phi[2*i+1]);
        sh_theta[i].x = __ldg(&pattern_theta[2*i]);
        sh_theta[i].y = __ldg(&pattern_theta[2*i+1]);
        sh_preamble[i] = __ldg(&pattern_preamble[i]);
       }
      } else {
       // in wideband mode, total is Nmodes*F (which can be too large)
      }
      __syncthreads();
    #endif

  //check data limit
  if (x<(K*T*F)) {

/*********************************************************************/
   /* time is already converted to thetaGMST */
   float thetaGMST=(float)__ldg(&time_utc[itm]);
   /* find az,el */
   float az,el,r,theta;
   radec2azel_gmst__(__ldg(&ra[isrc]),__ldg(&dec[isrc]), __ldg(&longitude[istat]), __ldg(&latitude[istat]), thetaGMST, &az, &el);
   /* transform : r= pi/2-el, phi=az-pi/4 for element beam */
   r=M_PI_2f-el;
   theta=az-M_PI_4f;
/*********************************************************************/
   if (el>=0.0f) {
      float4 evalX,evalY;
      if (!wideband) {
      #if (ARRAY_USE_SHMEM == 1)
      evalX=eval_elementcoeff(r, theta, M, beta, sh_theta,
                sh_phi, sh_preamble);
      evalY=eval_elementcoeff(r, theta+M_PI_2f, M, beta, sh_theta,
                sh_phi, sh_preamble);
      #else
      evalX=eval_elementcoeff(r, theta, M, beta, (float2*)pattern_theta,
                (float2*)pattern_phi, pattern_preamble);
      evalY=eval_elementcoeff(r, theta+M_PI_2f, M, beta, (float2*)pattern_theta,
                (float2*)pattern_phi, pattern_preamble);
      #endif
      } else {
       // in wideband mode, offset by 2*Nmodes*ifrq, not using shared memory
      evalX=eval_elementcoeff(r, theta, M, beta, (float2*)&pattern_theta[2*Nmodes*ifrq],
                (float2*)&pattern_phi[2*Nmodes*ifrq], pattern_preamble);
      evalY=eval_elementcoeff(r, theta+M_PI_2f, M, beta, (float2*)&pattern_theta[2*Nmodes*ifrq],
                (float2*)&pattern_phi[2*Nmodes*ifrq], pattern_preamble);

      }

   /* store output EJones 8 values */ 
   int boffset=itm*N*K*F+isrc*N*F+ifrq*N+istat;
   beam[8*boffset]=evalX.x;
   beam[8*boffset+1]=evalX.y;
   beam[8*boffset+2]=evalX.z;
   beam[8*boffset+3]=evalX.w;
   beam[8*boffset+4]=evalY.x;
   beam[8*boffset+5]=evalY.y;
   beam[8*boffset+6]=evalY.z;
   beam[8*boffset+7]=evalY.w;
   } else {
   int boffset=itm*N*K*F+isrc*N*F+ifrq*N+istat;
   beam[8*boffset]=0.0f;
   beam[8*boffset+1]=0.0f;
   beam[8*boffset+2]=0.0f;
   beam[8*boffset+3]=0.0f;
   beam[8*boffset+4]=0.0f;
   beam[8*boffset+5]=0.0f;
   beam[8*boffset+6]=0.0f;
   beam[8*boffset+7]=0.0f;
   }
  }

}

/***************************************************************************/
__device__ cuDoubleComplex
gaussian_contrib__(int *dd, float u, float v, float w) {
  exinfo_gaussian *dp=(exinfo_gaussian*)dd;
  float up,vp,a,b,ut,vt,cosph,sinph;

  /* first the rotation due to projection */
  if (dp->use_projection) {
   up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
   vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);
  } else {
   up=u;
   vp=v;
  }

  /* linear transformations, if any */
  a=dp->eX;
  b=dp->eY;
  sincosf(dp->eP,&sinph,&cosph);
  ut=a*(cosph*up-sinph*vp);
  vt=b*(sinph*up+cosph*vp);

  return make_cuDoubleComplex((double)(expf(-2.0*M_PI*M_PI*(ut*ut+vt*vt))),0.0);
}



__device__ cuDoubleComplex
ring_contrib__(int *dd, float u, float v, float w) {
  exinfo_ring *dp=(exinfo_ring*)dd;
  float up,vp,a,b;

  /* first the rotation due to projection */
  up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
  vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);

  a=dp->eX; /* diameter */
  b=sqrtf(up*up+vp*vp)*a*2.0f*M_PIf;

  return make_cuDoubleComplex((double)j0f(b),0.0);
}

__device__ cuDoubleComplex
disk_contrib__(int *dd, float u, float v, float w) {
  exinfo_disk *dp=(exinfo_disk*)dd;
  float up,vp,a,b;

  /* first the rotation due to projection */
  up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
  vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);

  a=dp->eX; /* diameter */
  b=sqrtf(up*up+vp*vp)*a*2.0f*M_PIf;

  return make_cuDoubleComplex((double)j1f(b),0.0);
}


/* Hermite polynomial, non recursive version */
__device__ float 
H_e(float x, int n) {
  if(n==0) return 1.0f;
  if(n==1) return 2.0f*x;
  /* else iterate */
  float Hn_1,Hn,Hnp1;
  Hn_1=1.0f;
  Hn=2.0f*x;
  int ci;
  for (ci=1; ci<n; ci++) {
    Hnp1=2.0f*x*Hn-2.0f*((float)ci)*Hn_1;
    Hn_1=Hn;
    Hn=Hnp1;
  }

  return Hn;
}

/* Hermite polynomial, non recursive version, suitable for large n
 scaled down, He/sqrt(2^n * n!) to prevent overflow */
__device__ float
H_e_scaled(float x, int n, float *fact) {
  const float scalefactor=sqrtf((float)(2<<n)*fact[n]);
  if(n==0) return 1.0f/scalefactor;
  if(n==1) return 2.0f*x/scalefactor;
  /* else iterate */
  float Hn_1,Hn,Hnp1;
  Hn_1=1.0f/scalefactor;
  Hn=2.0f*x/scalefactor;
  int ci;
  for (ci=1; ci<n; ci++) {
    Hnp1=2.0f*x*Hn-2.0f*((float)ci)*Hn_1;
    Hn_1=Hn;
    Hn=Hnp1;
  }

  return Hn;
}

#define LARGE_MODE_LIMIT 20
__device__ void
calculate_uv_mode_vectors_scalar(float u, float v, float beta, int n0, float *Av, int *cplx, float *fact, float *shpvl) {

  int xci;

  int n1,n2,start;
  int signval;


  /* start filling in the array from the positive values */
  float xvalu=u*beta;
  float expvalu=__expf(-0.5f*xvalu*xvalu);
  float xvalv=v*beta;
  float expvalv=__expf(-0.5f*xvalv*xvalv);
  if (n0 < LARGE_MODE_LIMIT) {
    for (xci=0; xci<n0; xci++) {
      shpvl[xci]=H_e(xvalu,xci)*expvalu/__fsqrt_rn((float)(2<<xci)*fact[xci]);

      shpvl[xci+n0]=H_e(xvalv,xci)*expvalv/__fsqrt_rn((float)(2<<xci)*fact[xci]);
    }
  } else {
    for (xci=0; xci<n0; xci++) {
      shpvl[xci]=H_e_scaled(xvalu,xci,fact)*expvalu;

      shpvl[xci+n0]=H_e_scaled(xvalv,xci,fact)*expvalv;
    }
  }

  /* now calculate the mode vectors */
  /* each vector is 1 x 1 length and there are n0*n0 of them */

  for (n2=0; n2<(n0); n2++) {
   for (n1=0; n1<(n0); n1++) {
    cplx[n2*n0+n1]=((n1+n2)%2==0?0:1) /* even (real) or odd (imaginary)*/;
    /* sign */
    if (cplx[n2*n0+n1]==0) {
      signval=(((n1+n2)/2)%2==0?1:-1);
    } else {
      signval=(((n1+n2-1)/2)%2==0?1:-1);
    }

    /* fill in 1*1*(zci) to 1*1*(zci+1)-1 */
    start=(n2*(n0)+n1);
    if (signval==-1) {
        Av[start]=-shpvl[n1]*shpvl[n0+n2];
    } else {
        Av[start]=shpvl[n1]*shpvl[n0+n2];
    }
   }
  }

}


#define SMALLEST_SPATIAL_SCALE_FAC 100.0f

__device__ cuDoubleComplex
shapelet_contrib__(int *dd, float u, float v, float w) {
  exinfo_shapelet *dp=(exinfo_shapelet*)dd;
  int *cplx;
  float *Av;
  int ci,M;
  float a,b,ut,vt,up,vp;
  float sinph,cosph;
  float realsum,imagsum;

  /* first the rotation due to projection */
  if (dp->use_projection) {
   up=-u*(dp->cxi)+v*(dp->cphi)*(dp->sxi)-w*(dp->sphi)*(dp->sxi);
   vp=-u*(dp->sxi)-v*(dp->cphi)*(dp->cxi)+w*(dp->sphi)*(dp->cxi);
  } else {
   up=u;
   vp=v;
  }

  /* linear transformations, if any */
  a=1.0f/dp->eX;
  b=1.0f/dp->eY;
  __sincosf((float)dp->eP,&sinph,&cosph);
  ut=a*(cosph*up-sinph*vp);
  vt=b*(sinph*up+cosph*vp);
  /* if u,v is way off the scale (beta) of shapelet modes, the result is almost always zero,
     so check this here and return 0, otherwise spurious nans may result,
   i.e., predict only for spatial scales l,m > beta * scale_factor ~ beta * 0.01 */
  if (__fdiv_rz(SMALLEST_SPATIAL_SCALE_FAC,__fsqrt_rz(ut*ut+vt*vt))<dp->beta) {
   return make_cuDoubleComplex(0.0,0.0);
  }
  /* note: we decompose f(-l,m) so the Fourier transform is F(-u,v)
   so negate the u grid */
  Av=(float*)malloc((size_t)((dp->n0)*(dp->n0))*sizeof(float));
  cplx=(int*)malloc((size_t)((dp->n0)*(dp->n0))*sizeof(int));

  float *fact=0;
  float *shpvl=0;
  /* set up factorial array */
  fact=(float *)malloc((size_t)(dp->n0)*sizeof(float));
  /* setup array to store calculated shapelet value */
  /* need max storage 2 x n0 */
  shpvl=(float*)malloc((size_t)(2*dp->n0)*sizeof(float));

  if (!fact || !Av || !cplx || !shpvl) {
   printf("Error: Device memory allocation failure!! increase heap size.\n");
  }
  fact[0]=1.0f;
  for (ci=1; ci<dp->n0; ci++) {
    fact[ci]=((float)ci)*fact[ci-1];
  }


  calculate_uv_mode_vectors_scalar(-ut, vt, dp->beta, dp->n0, Av, cplx, fact, shpvl);

  free(fact);
  free(shpvl);

  realsum=imagsum=0.0f;
  M=(dp->n0)*(dp->n0);
  for (ci=0; ci<M; ci++) {
    if (cplx[ci]) {
     imagsum+=dp->modes[ci]*Av[ci];
    } else {
     realsum+=dp->modes[ci]*Av[ci];
    }
  }

  free(Av);
  free(cplx);
  realsum*=2.0f*M_PIf*a*b;
  imagsum*=2.0f*M_PIf*a*b;
  /* additional safeguards */
  if ( isnan(realsum) ) { realsum=0.0f; }
  if ( isnan(imagsum) ) { imagsum=0.0f; }
  return make_cuDoubleComplex((double)realsum,(double)imagsum);
}

__device__ void 
compute_prodterm_multifreq(int sta1, int sta2, int N, int K, int T, int F,
double phterm0, double sI0f, double sQ0f, double sU0f, double sV0f, double spec_idxf, double spec_idx1f, double spec_idx2f, double myf0,
 double myfreq, double deltaf, int dobeam, int itm, int k, int cf, const float *__restrict__ beam, const float *__restrict__ element, int **exs, unsigned char stypeT, double u, double v, double w, double *__restrict__ output) {
     /* F>1 is assumed output: 8x1 array */
     double sinph,cosph;
     sincos(phterm0*myfreq,&sinph,&cosph);
     cuDoubleComplex prodterm=make_cuDoubleComplex(cosph,sinph);
     double If,Qf,Uf,Vf;
     If=Qf=Uf=Vf=0.0;
     /* evaluate spectra, only if non-zero spectral indices given */
     int spectra_valid=(spec_idxf!=0.0 || spec_idx1f!=0.0 || spec_idx2f!=0.0);
     double cm=0.0;
     if (spectra_valid) {
      double fratio=log(myfreq/myf0);
      double fratio1=fratio*fratio;
      double fratio2=fratio1*fratio;
      cm=spec_idxf*fratio+spec_idx1f*fratio1+spec_idx2f*fratio2;
     }
     if (spectra_valid) {
     /* catch -ve flux */
     if (sI0f>0.0) {
        If=exp(log(sI0f)+cm);
     } else if (sI0f<0.0) {
        If=-exp(log(-sI0f)+cm);
     } 
     if (sQ0f>0.0) {
        Qf=exp(log(sQ0f)+cm);
     } else if (sI0f<0.0) {
        Qf=-exp(log(-sQ0f)+cm);
     } 
     if (sU0f>0.0) {
        Uf=exp(log(sU0f)+cm);
     } else if (sI0f<0.0) {
        Uf=-exp(log(-sU0f)+cm);
     } 
     if (sV0f>0.0) {
        Vf=exp(log(sV0f)+cm);
     } else if (sI0f<0.0) {
        Vf=-exp(log(-sV0f)+cm);
     }
     } else {
       If=sI0f;
       Qf=sQ0f;
       Uf=sU0f;
       Vf=sV0f;
     }

     /* smearing, beam */
     double scalef=1.0;
     double phterm =(phterm0*0.5*deltaf);
     if (phterm!=0.0) {
      sinph=(sin(phterm)/phterm);
      scalef *=fabs(sinph); /* catch -ve values due to rounding off */
     }

     if (dobeam==DOBEAM_ARRAY || dobeam==DOBEAM_FULL
         ||dobeam==DOBEAM_ARRAY_WB || dobeam==DOBEAM_FULL_WB) {
      /* get beam info */
      int boffset1=itm*N*K*F+k*N*F+cf*N+sta1;
      //  printf("itm=%d, k1=%d, sta1=%d, sta2=%d, boffset1=%d, boffset2=%d\n", itm, k1, sta1, sta2, boffset1, boffset2);
      float beam1=__ldg(&beam[boffset1]);
      //int boffset2=sta2*K*T*F + k1*T*F + cf*T + itm;
      int boffset2=itm*N*K*F+k*N*F+cf*N+sta2;
      float beam2=__ldg(&beam[boffset2]);
      scalef *=(double)(beam1*beam2);
     }


     /* form complex value */
     prodterm.x *=scalef;
     prodterm.y *=scalef;

     /* check for type of source */
     if (stypeT!=STYPE_POINT && !(u==0.0 && v==0.0)) {
      float uscaled=(float)(u*myfreq);
      float vscaled=(float)(v*myfreq);
      float wscaled=(float)(w*myfreq);
      if (stypeT==STYPE_SHAPELET) {
       prodterm=cuCmul(shapelet_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      } else if (stypeT==STYPE_GAUSSIAN) {
       prodterm=cuCmul(gaussian_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      } else if (stypeT==STYPE_DISK) {
       prodterm=cuCmul(disk_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      } else if (stypeT==STYPE_RING) {
       prodterm=cuCmul(ring_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      }
     }

    double Ix,Iy,Qx,Qy,Ux,Uy,Vx,Vy;
    Ix=If*prodterm.x;
    Iy=If*prodterm.y;
    Qx=Qf*prodterm.x;
    Qy=Qf*prodterm.y;
    Ux=Uf*prodterm.x;
    Uy=Uf*prodterm.y;
    Vx=Vf*prodterm.x;
    Vy=Vf*prodterm.y;

    if (dobeam==DOBEAM_ELEMENT || dobeam==DOBEAM_FULL
        ||dobeam==DOBEAM_ELEMENT_WB ||dobeam==DOBEAM_FULL_WB) {
     cuDoubleComplex E1[4], E2[4], C[4], T[4];
     C[0].x=Ix+Qx;
     C[0].y=Iy+Qy;
     C[1].x=Ux-Vy;
     C[1].y=Vx+Uy;
     C[2].x=Ux+Vy;
     C[2].y=-Vx+Uy;
     C[3].x=Ix-Qx;
     C[3].y=Iy-Qy;
     /* Ejones matrices */
     int boffset1=itm*N*K*F+k*N*F+cf*N+sta1;
     int boffset2=itm*N*K*F+k*N*F+cf*N+sta2;
     E1[0].x=(double)__ldg(&element[8*boffset1]);
     E1[0].y=(double)__ldg(&element[8*boffset1+1]);
     E1[1].x=(double)__ldg(&element[8*boffset1+2]);
     E1[1].y=(double)__ldg(&element[8*boffset1+3]);
     E1[2].x=(double)__ldg(&element[8*boffset1+4]);
     E1[2].y=(double)__ldg(&element[8*boffset1+5]);
     E1[3].x=(double)__ldg(&element[8*boffset1+6]);
     E1[3].y=(double)__ldg(&element[8*boffset1+7]);
     E2[0].x=(double)__ldg(&element[8*boffset2]);
     E2[0].y=(double)__ldg(&element[8*boffset2+1]);
     E2[1].x=(double)__ldg(&element[8*boffset2+2]);
     E2[1].y=(double)__ldg(&element[8*boffset2+3]);
     E2[2].x=(double)__ldg(&element[8*boffset2+4]);
     E2[2].y=(double)__ldg(&element[8*boffset2+5]);
     E2[3].x=(double)__ldg(&element[8*boffset2+6]);
     E2[3].y=(double)__ldg(&element[8*boffset2+7]);
     amb(E1,C,T); /* T = E1 x C */
     ambt(T,E2,C); /* C= T x E2^H = (E1 x C) x E2^H */
     output[0]=C[0].x;
     output[1]=C[0].y;
     output[2]=C[1].x;
     output[3]=C[1].y;
     output[4]=C[2].x;
     output[5]=C[2].y;
     output[6]=C[3].x;
     output[7]=C[3].y;
    } else {
     output[0]=Ix+Qx;
     output[1]=Iy+Qy;
     output[2]=Ux-Vy;
     output[3]=Vx+Uy;
     output[4]=Ux+Vy;
     output[5]=-Vx+Uy;
     output[6]=Ix-Qx;
     output[7]=Iy-Qy;
    }
}


__device__ void 
compute_prodterm(int sta1, int sta2, int N, int K, int T,
 double phterm0, double If, double Qf, double Uf, double Vf,
 double myfreq, double deltaf, int dobeam, int itm, int k, const float *__restrict__ beam, const float *__restrict__ element, int **exs, unsigned char stypeT, double u, double v, double w, double *__restrict__ output) {
     /* F==1 is assumed, output: 8x1 array */
     double sinph,cosph;
     sincos(phterm0*myfreq,&sinph,&cosph);
     cuDoubleComplex prodterm=make_cuDoubleComplex(cosph,sinph);

     /* smearing, beam */
     double scalef=1.0;
     double phterm =(phterm0*0.5*deltaf);
     if (phterm!=0.0) {
      sinph=(sin(phterm)/phterm);
      scalef *=fabs(sinph); /* catch -ve values due to rounding off */
     }

     if (dobeam==DOBEAM_ARRAY || dobeam==DOBEAM_FULL
         ||dobeam==DOBEAM_ARRAY_WB ||dobeam==DOBEAM_FULL_WB) {
      /* get beam info */
      int boffset1=itm*N*K+k*N+sta1;
      //  printf("itm=%d, k1=%d, sta1=%d, sta2=%d, boffset1=%d, boffset2=%d\n", itm, k1, sta1, sta2, boffset1, boffset2);
      float beam1=__ldg(&beam[boffset1]);
      int boffset2=itm*N*K+k*N+sta2;
      float beam2=__ldg(&beam[boffset2]);
      scalef *=(double)(beam1*beam2);
     }


     /* form complex value */
     prodterm.x *=scalef;
     prodterm.y *=scalef;

     /* check for type of source */
     if (stypeT!=STYPE_POINT && !(u==0.0 && v==0.0)) {
      float uscaled=(float)(u*myfreq);
      float vscaled=(float)(v*myfreq);
      float wscaled=(float)(w*myfreq);
      if (stypeT==STYPE_SHAPELET) {
       prodterm=cuCmul(shapelet_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      } else if (stypeT==STYPE_GAUSSIAN) {
       prodterm=cuCmul(gaussian_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      } else if (stypeT==STYPE_DISK) {
       prodterm=cuCmul(disk_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      } else if (stypeT==STYPE_RING) {
       prodterm=cuCmul(ring_contrib__(exs[k],uscaled,vscaled,wscaled),prodterm);
      }
     }


    double Ix,Iy,Qx,Qy,Ux,Uy,Vx,Vy;
    Ix=If*prodterm.x;
    Iy=If*prodterm.y;
    Qx=Qf*prodterm.x;
    Qy=Qf*prodterm.y;
    Ux=Uf*prodterm.x;
    Uy=Uf*prodterm.y;
    Vx=Vf*prodterm.x;
    Vy=Vf*prodterm.y;

    if (dobeam==DOBEAM_ELEMENT || dobeam==DOBEAM_FULL
        ||dobeam==DOBEAM_ELEMENT_WB ||dobeam==DOBEAM_FULL_WB) {
     cuDoubleComplex E1[4], E2[4], C[4], T[4];
     C[0].x=Ix+Qx;
     C[0].y=Iy+Qy;
     C[1].x=Ux-Vy;
     C[1].y=Vx+Uy;
     C[2].x=Ux+Vy;
     C[2].y=-Vx+Uy;
     C[3].x=Ix-Qx;
     C[3].y=Iy-Qy;
     /* Ejones matrices */
     int boffset1=itm*N*K+k*N+sta1;
     int boffset2=itm*N*K+k*N+sta2;
     E1[0].x=(double)__ldg(&element[8*boffset1]);
     E1[0].y=(double)__ldg(&element[8*boffset1+1]);
     E1[1].x=(double)__ldg(&element[8*boffset1+2]);
     E1[1].y=(double)__ldg(&element[8*boffset1+3]);
     E1[2].x=(double)__ldg(&element[8*boffset1+4]);
     E1[2].y=(double)__ldg(&element[8*boffset1+5]);
     E1[3].x=(double)__ldg(&element[8*boffset1+6]);
     E1[3].y=(double)__ldg(&element[8*boffset1+7]);
     E2[0].x=(double)__ldg(&element[8*boffset2]);
     E2[0].y=(double)__ldg(&element[8*boffset2+1]);
     E2[1].x=(double)__ldg(&element[8*boffset2+2]);
     E2[1].y=(double)__ldg(&element[8*boffset2+3]);
     E2[2].x=(double)__ldg(&element[8*boffset2+4]);
     E2[2].y=(double)__ldg(&element[8*boffset2+5]);
     E2[3].x=(double)__ldg(&element[8*boffset2+6]);
     E2[3].y=(double)__ldg(&element[8*boffset2+7]);
     amb(E1,C,T); /* T = E1 x C */
     ambt(T,E2,C); /* C= T x E2^H = (E1 x C) x E2^H */
     output[0]=C[0].x;
     output[1]=C[0].y;
     output[2]=C[1].x;
     output[3]=C[1].y;
     output[4]=C[2].x;
     output[5]=C[2].y;
     output[6]=C[3].x;
     output[7]=C[3].y;
    } else {
     output[0]=Ix+Qx;
     output[1]=Iy+Qy;
     output[2]=Ux-Vy;
     output[3]=Vx+Uy;
     output[4]=Ux+Vy;
     output[5]=-Vx+Uy;
     output[6]=Ix-Qx;
     output[7]=Iy-Qy;
    }
}

/* master kernel to calculate coherencies */
__global__ void 
kernel_coherencies(int B, int N, int T, int K, int F,
  const double *__restrict__ u, const double *__restrict__ v, const double *__restrict__ w,
  baseline_t *barr, const double *__restrict__ freqs, const float *__restrict__ beam, const float *__restrict__ element, const double *__restrict__ ll, const double *__restrict__ mm, const double *__restrict__ nn, 
  const double *__restrict__ sI, const double *__restrict__ sQ, const double *__restrict__ sU, const double *__restrict__ sV,
  const unsigned char *__restrict__ stype, const double *__restrict__ sI0, 
const double *__restrict__ sQ0, const double *__restrict__ sU0, const double *__restrict__ sV0,
  const double *__restrict__ f0, const double *__restrict__ spec_idx, const double *__restrict__ spec_idx1, const double *__restrict__ spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh, int dobeam) {

  /* global thread index */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;

  /* each thread will calculate for one baseline, over all sources */
  if (n<B) {
   int sta1=barr[n].sta1;
   int sta2=barr[n].sta2;
   /* find out which time slot this baseline is from */
   int tslot=n/((N*(N-1)/2));

   double u_n=(u[n]);
   double v_n=(v[n]);
   double w_n=(w[n]);

   double l_coh[MODEL_MAX_F][8];

   if (F<=MODEL_MAX_F) {

   for(int cf=0; cf<F; cf++) {
        l_coh[cf][0]=0.0;
        l_coh[cf][1]=0.0;
        l_coh[cf][2]=0.0;
        l_coh[cf][3]=0.0;
        l_coh[cf][4]=0.0;
        l_coh[cf][5]=0.0;
        l_coh[cf][6]=0.0;
        l_coh[cf][7]=0.0;
   }

   // split to two cases, F==1 and F>1
   if (F==1) {
     //use simply for-loop, if K is very large this may be slow and may need further parallelization
     for (int k=0; k<K; k++) {
        //source specific params
        double sIf,sQf,sUf,sVf;
        double phterm0 = (2.0*M_PI*(u_n*__ldg(&ll[k])+v_n*__ldg(&mm[k])+w_n*__ldg(&nn[k])));
        sIf=__ldg(&sI[k]);
        sQf=__ldg(&sQ[k]);
        sUf=__ldg(&sU[k]);
        sVf=__ldg(&sV[k]);

        unsigned char stypeT=__ldg(&stype[k]);

        double llcoh[8];
        compute_prodterm(sta1, sta2, N, K, T, phterm0, sIf, sQf, sUf, sVf,
               __ldg(&(freqs[0])), deltaf, dobeam, tslot, k, beam, element, exs, stypeT, u_n, v_n, w_n, llcoh);

         l_coh[0][0] +=llcoh[0];
         l_coh[0][1] +=llcoh[1];
         l_coh[0][2] +=llcoh[2];
         l_coh[0][3] +=llcoh[3];
         l_coh[0][4] +=llcoh[4];
         l_coh[0][5] +=llcoh[5];
         l_coh[0][6] +=llcoh[6];
         l_coh[0][7] +=llcoh[7];

     }
   } else {
     //use simply for-loop, if K is very large this may be slow and may need further parallelization
     for (int k=0; k<K; k++) {
        //source specific params
        double sI0f,sQ0f,sU0f,sV0f,spec_idxf,spec_idx1f,spec_idx2f,myf0;
        double phterm0 = (2.0*M_PI*(u_n*__ldg(&ll[k])+v_n*__ldg(&mm[k])+w_n*__ldg(&nn[k])));
        sI0f=__ldg(&sI0[k]);
        sQ0f=__ldg(&sQ0[k]);
        sU0f=__ldg(&sU0[k]);
        sV0f=__ldg(&sV0[k]);
        spec_idxf=__ldg(&spec_idx[k]);
        spec_idx1f=__ldg(&spec_idx1[k]);
        spec_idx2f=__ldg(&spec_idx2[k]);
        myf0=__ldg(&f0[k]);

        unsigned char stypeT=__ldg(&stype[k]);

        for(int cf=0; cf<F; cf++) {
            double llcoh[8];
            compute_prodterm_multifreq(sta1, sta2, N, K, T, F, phterm0, sI0f, sQ0f, sU0f, sV0f, spec_idxf, spec_idx1f, spec_idx2f, 
               myf0, __ldg(&(freqs[cf])), deltaf, dobeam, tslot, k, cf, beam, element, exs, stypeT, u_n, v_n, w_n,llcoh);
         l_coh[cf][0] +=llcoh[0];
         l_coh[cf][1] +=llcoh[1];
         l_coh[cf][2] +=llcoh[2];
         l_coh[cf][3] +=llcoh[3];
         l_coh[cf][4] +=llcoh[4];
         l_coh[cf][5] +=llcoh[5];
         l_coh[cf][6] +=llcoh[6];
         l_coh[cf][7] +=llcoh[7];
        }

     }

    }
     //write output with right multi frequency offset
    double *coh1 = &coh[8*n];
    for(int cf=0; cf<F; cf++) {
        coh1[cf*8*B+0] = l_coh[cf][0];
        coh1[cf*8*B+1] = l_coh[cf][1];
        coh1[cf*8*B+2] = l_coh[cf][2];
        coh1[cf*8*B+3] = l_coh[cf][3];
        coh1[cf*8*B+4] = l_coh[cf][4];
        coh1[cf*8*B+5] = l_coh[cf][5];
        coh1[cf*8*B+6] = l_coh[cf][6];
        coh1[cf*8*B+7] = l_coh[cf][7];
    }

    } else {
      /* F> MODEL_MAX_F, need to calculate/write multiple times */
      /* how many calculate/write cycles */
      int fcycle=(F+MODEL_MAX_F-1)/MODEL_MAX_F;
      for (int cff=0; cff<fcycle; cff++) {
        for(int cf=0; cf<MODEL_MAX_F; cf++) {
        l_coh[cf][0]=0.0;
        l_coh[cf][1]=0.0;
        l_coh[cf][2]=0.0;
        l_coh[cf][3]=0.0;
        l_coh[cf][4]=0.0;
        l_coh[cf][5]=0.0;
        l_coh[cf][6]=0.0;
        l_coh[cf][7]=0.0;
        }

        for (int k=0; k<K; k++) {
        //source specific params
        double sI0f,sQ0f,sU0f,sV0f,spec_idxf,spec_idx1f,spec_idx2f,myf0;
        double phterm0 = (2.0*M_PI*(u_n*__ldg(&ll[k])+v_n*__ldg(&mm[k])+w_n*__ldg(&nn[k])));
        sI0f=__ldg(&sI0[k]);
        sQ0f=__ldg(&sQ0[k]);
        sU0f=__ldg(&sU0[k]);
        sV0f=__ldg(&sV0[k]);
        spec_idxf=__ldg(&spec_idx[k]);
        spec_idx1f=__ldg(&spec_idx1[k]);
        spec_idx2f=__ldg(&spec_idx2[k]);
        myf0=__ldg(&f0[k]);

        unsigned char stypeT=__ldg(&stype[k]);
        for(int cf=0; cf<MODEL_MAX_F && cf+cff*MODEL_MAX_F<F; cf++) {
            double llcoh[8];
            compute_prodterm_multifreq(sta1, sta2, N, K, T, F, phterm0, sI0f, sQ0f, sU0f, sV0f, spec_idxf, spec_idx1f, spec_idx2f, 
               myf0, __ldg(&(freqs[cf+cff*MODEL_MAX_F])), deltaf, dobeam, tslot, k, cf+cff*MODEL_MAX_F, beam, element, exs, stypeT, u_n, v_n, w_n,llcoh);
         l_coh[cf][0] +=llcoh[0];
         l_coh[cf][1] +=llcoh[1];
         l_coh[cf][2] +=llcoh[2];
         l_coh[cf][3] +=llcoh[3];
         l_coh[cf][4] +=llcoh[4];
         l_coh[cf][5] +=llcoh[5];
         l_coh[cf][6] +=llcoh[6];
         l_coh[cf][7] +=llcoh[7];
        }


        }
  
        double *coh1 = &coh[8*n];
        for(int cf=0; cf<MODEL_MAX_F && cf+cff*MODEL_MAX_F<F; cf++) {
        coh1[(cf+cff*MODEL_MAX_F)*8*B+0] = l_coh[cf][0];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+1] = l_coh[cf][1];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+2] = l_coh[cf][2];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+3] = l_coh[cf][3];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+4] = l_coh[cf][4];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+5] = l_coh[cf][5];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+6] = l_coh[cf][6];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+7] = l_coh[cf][7];
        }


       }

    }
  
  }

}

/* kernel to calculate residuals */
__global__ void 
kernel_residuals(int B, int N, int T, int K, int F,
  const double *__restrict__ u, const double *__restrict__ v, const double *__restrict__ w,
  const double *__restrict__ p, int nchunk,
  baseline_t *barr, const double *__restrict__ freqs, const float *__restrict__ beam, const float *__restrict__ element, const double *__restrict__ ll, const double *__restrict__ mm, const double *__restrict__ nn, 
  const double *__restrict__ sI, const double *__restrict__ sQ, const double *__restrict__ sU, const double *__restrict__ sV,
  const unsigned char *__restrict__ stype, const double *__restrict__ sI0, 
const double *__restrict__ sQ0, const double *__restrict__ sU0, const double *__restrict__ sV0,
  const double *__restrict__ f0, const double *__restrict__ spec_idx, const double *__restrict__ spec_idx1, const double *__restrict__ spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh, int dobeam) {

  /* global thread index */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;

  /* each thread will calculate for one baseline, over all sources */
  if (n<B) {
   int sta1=barr[n].sta1;
   int sta2=barr[n].sta2;
   /* find out which time slot this baseline is from */
   int tslot=n/((N*(N-1)/2));
   /* find out which chunk to select from p : 0,1..nchunk-1 */
   int chunk=(n)/((B+nchunk-1)/nchunk);
   /* create G1 and G2 Jones matrices from p */
   cuDoubleComplex G1[4],G2[4];
   G1[0].x=__ldg(&p[chunk*8*N+sta1*8]);
   G1[0].y=__ldg(&p[chunk*8*N+sta1*8+1]);
   G1[1].x=__ldg(&p[chunk*8*N+sta1*8+2]);
   G1[1].y=__ldg(&p[chunk*8*N+sta1*8+3]);
   G1[2].x=__ldg(&p[chunk*8*N+sta1*8+4]);
   G1[2].y=__ldg(&p[chunk*8*N+sta1*8+5]);
   G1[3].x=__ldg(&p[chunk*8*N+sta1*8+6]);
   G1[3].y=__ldg(&p[chunk*8*N+sta1*8+7]);
   G2[0].x=__ldg(&p[chunk*8*N+sta2*8]);
   G2[0].y=__ldg(&p[chunk*8*N+sta2*8+1]);
   G2[1].x=__ldg(&p[chunk*8*N+sta2*8+2]);
   G2[1].y=__ldg(&p[chunk*8*N+sta2*8+3]);
   G2[2].x=__ldg(&p[chunk*8*N+sta2*8+4]);
   G2[2].y=__ldg(&p[chunk*8*N+sta2*8+5]);
   G2[3].x=__ldg(&p[chunk*8*N+sta2*8+6]);
   G2[3].y=__ldg(&p[chunk*8*N+sta2*8+7]);

   double u_n=(u[n]);
   double v_n=(v[n]);
   double w_n=(w[n]);

   double l_coh[MODEL_MAX_F][8];
   if (F<=MODEL_MAX_F) {
   for(int cf=0; cf<F; cf++) {
        l_coh[cf][0]=0.0;
        l_coh[cf][1]=0.0;
        l_coh[cf][2]=0.0;
        l_coh[cf][3]=0.0;
        l_coh[cf][4]=0.0;
        l_coh[cf][5]=0.0;
        l_coh[cf][6]=0.0;
        l_coh[cf][7]=0.0;
   }


   // split to two cases, F==1 and F>1
   if (F==1) {
     //use simply for-loop, if K is very large this may be slow and may need further parallelization
     for (int k=0; k<K; k++) {
        //source specific params
        double sIf,sQf,sUf,sVf;
        double phterm0 = (2.0*M_PI*(u_n*__ldg(&ll[k])+v_n*__ldg(&mm[k])+w_n*__ldg(&nn[k])));
        sIf=__ldg(&sI[k]);
        sQf=__ldg(&sQ[k]);
        sUf=__ldg(&sU[k]);
        sVf=__ldg(&sV[k]);

        unsigned char stypeT=__ldg(&stype[k]);

        double llcoh[8];
        compute_prodterm(sta1, sta2, N, K, T, phterm0, sIf, sQf, sUf, sVf,
               __ldg(&(freqs[0])), deltaf, dobeam, tslot, k, beam, element, exs, stypeT, u_n, v_n, w_n, llcoh);

         l_coh[0][0] +=llcoh[0];
         l_coh[0][1] +=llcoh[1];
         l_coh[0][2] +=llcoh[2];
         l_coh[0][3] +=llcoh[3];
         l_coh[0][4] +=llcoh[4];
         l_coh[0][5] +=llcoh[5];
         l_coh[0][6] +=llcoh[6];
         l_coh[0][7] +=llcoh[7];


     }
     cuDoubleComplex L1[4],L2[4];
     L1[0].x=l_coh[0][0];
     L1[0].y=l_coh[0][1];
     L1[1].x=l_coh[0][2];
     L1[1].y=l_coh[0][3];
     L1[2].x=l_coh[0][4];
     L1[2].y=l_coh[0][5];
     L1[3].x=l_coh[0][6];
     L1[3].y=l_coh[0][7];
     /* L2=G1*L1 */
     amb(G1,L1,L2);
     /* L1=L2*G2^H */
     ambt(L2,G2,L1);
     l_coh[0][0]=L1[0].x;
     l_coh[0][1]=L1[0].y;
     l_coh[0][2]=L1[1].x;
     l_coh[0][3]=L1[1].y;
     l_coh[0][4]=L1[2].x;
     l_coh[0][5]=L1[2].y;
     l_coh[0][6]=L1[3].x;
     l_coh[0][7]=L1[3].y;

   } else {
     //use simply for-loop, if K is very large this may be slow and may need further parallelization
     for (int k=0; k<K; k++) {
        //source specific params
        double sI0f,sQ0f,sU0f,sV0f,spec_idxf,spec_idx1f,spec_idx2f,myf0;
        double phterm0 = (2.0*M_PI*(u_n*__ldg(&ll[k])+v_n*__ldg(&mm[k])+w_n*__ldg(&nn[k])));
        sI0f=__ldg(&sI0[k]);
        sQ0f=__ldg(&sQ0[k]);
        sU0f=__ldg(&sU0[k]);
        sV0f=__ldg(&sV0[k]);
        spec_idxf=__ldg(&spec_idx[k]);
        spec_idx1f=__ldg(&spec_idx1[k]);
        spec_idx2f=__ldg(&spec_idx2[k]);
        myf0=__ldg(&f0[k]);

        unsigned char stypeT=__ldg(&stype[k]);

        for(int cf=0; cf<F; cf++) {
            double llcoh[8];
            compute_prodterm_multifreq(sta1, sta2, N, K, T, F, phterm0, sI0f, sQ0f, sU0f, sV0f, spec_idxf, spec_idx1f, spec_idx2f, 
               myf0, __ldg(&(freqs[cf])), deltaf, dobeam, tslot, k, cf, beam, element, exs, stypeT, u_n, v_n, w_n,llcoh);
         l_coh[cf][0] +=llcoh[0];
         l_coh[cf][1] +=llcoh[1];
         l_coh[cf][2] +=llcoh[2];
         l_coh[cf][3] +=llcoh[3];
         l_coh[cf][4] +=llcoh[4];
         l_coh[cf][5] +=llcoh[5];
         l_coh[cf][6] +=llcoh[6];
         l_coh[cf][7] +=llcoh[7];
        }

     }

     cuDoubleComplex L1[4],L2[4];
     for(int cf=0; cf<F; cf++) {
       L1[0].x=l_coh[cf][0];
       L1[0].y=l_coh[cf][1];
       L1[1].x=l_coh[cf][2];
       L1[1].y=l_coh[cf][3];
       L1[2].x=l_coh[cf][4];
       L1[2].y=l_coh[cf][5];
       L1[3].x=l_coh[cf][6];
       L1[3].y=l_coh[cf][7];
       /* L2=G1*L1 */
       amb(G1,L1,L2);
       /* L1=L2*G2^H */
       ambt(L2,G2,L1);
       l_coh[cf][0]=L1[0].x;
       l_coh[cf][1]=L1[0].y;
       l_coh[cf][2]=L1[1].x;
       l_coh[cf][3]=L1[1].y;
       l_coh[cf][4]=L1[2].x;
       l_coh[cf][5]=L1[2].y;
       l_coh[cf][6]=L1[3].x;
       l_coh[cf][7]=L1[3].y;
     }


     }

     //write output with right multi frequency offset
    double *coh1 = &coh[8*n];
    for(int cf=0; cf<F; cf++) {
        coh1[cf*8*B+0] = l_coh[cf][0];
        coh1[cf*8*B+1] = l_coh[cf][1];
        coh1[cf*8*B+2] = l_coh[cf][2];
        coh1[cf*8*B+3] = l_coh[cf][3];
        coh1[cf*8*B+4] = l_coh[cf][4];
        coh1[cf*8*B+5] = l_coh[cf][5];
        coh1[cf*8*B+6] = l_coh[cf][6];
        coh1[cf*8*B+7] = l_coh[cf][7];
    }
   } else {
    /* F> MODEL_MAX_F, need to calculate/write multiple times */
      /* how many calculate/write cycles */
      int fcycle=(F+MODEL_MAX_F-1)/MODEL_MAX_F;
      for (int cff=0; cff<fcycle; cff++) {
     for(int cf=0; cf<MODEL_MAX_F; cf++) {
        l_coh[cf][0]=0.0;
        l_coh[cf][1]=0.0;
        l_coh[cf][2]=0.0;
        l_coh[cf][3]=0.0;
        l_coh[cf][4]=0.0;
        l_coh[cf][5]=0.0;
        l_coh[cf][6]=0.0;
        l_coh[cf][7]=0.0;
     }

     for (int k=0; k<K; k++) {
        //source specific params
        double sI0f,sQ0f,sU0f,sV0f,spec_idxf,spec_idx1f,spec_idx2f,myf0;
        double phterm0 = (2.0*M_PI*(u_n*__ldg(&ll[k])+v_n*__ldg(&mm[k])+w_n*__ldg(&nn[k])));
        sI0f=__ldg(&sI0[k]);
        sQ0f=__ldg(&sQ0[k]);
        sU0f=__ldg(&sU0[k]);
        sV0f=__ldg(&sV0[k]);
        spec_idxf=__ldg(&spec_idx[k]);
        spec_idx1f=__ldg(&spec_idx1[k]);
        spec_idx2f=__ldg(&spec_idx2[k]);
        myf0=__ldg(&f0[k]);

        unsigned char stypeT=__ldg(&stype[k]);

        for(int cf=0; cf<MODEL_MAX_F && cf+cff*MODEL_MAX_F<F; cf++) {
            double llcoh[8];
            compute_prodterm_multifreq(sta1, sta2, N, K, T, F, phterm0, sI0f, sQ0f, sU0f, sV0f, spec_idxf, spec_idx1f, spec_idx2f, 
               myf0, __ldg(&(freqs[cf+cff*MODEL_MAX_F])), deltaf, dobeam, tslot, k, cf+cff*MODEL_MAX_F, beam, element, exs, stypeT, u_n, v_n, w_n,llcoh);
         l_coh[cf][0] +=llcoh[0];
         l_coh[cf][1] +=llcoh[1];
         l_coh[cf][2] +=llcoh[2];
         l_coh[cf][3] +=llcoh[3];
         l_coh[cf][4] +=llcoh[4];
         l_coh[cf][5] +=llcoh[5];
         l_coh[cf][6] +=llcoh[6];
         l_coh[cf][7] +=llcoh[7];
        }

     }

     cuDoubleComplex L1[4],L2[4];
     for(int cf=0; cf<MODEL_MAX_F; cf++) {
       L1[0].x=l_coh[cf][0];
       L1[0].y=l_coh[cf][1];
       L1[1].x=l_coh[cf][2];
       L1[1].y=l_coh[cf][3];
       L1[2].x=l_coh[cf][4];
       L1[2].y=l_coh[cf][5];
       L1[3].x=l_coh[cf][6];
       L1[3].y=l_coh[cf][7];
       /* L2=G1*L1 */
       amb(G1,L1,L2);
       /* L1=L2*G2^H */
       ambt(L2,G2,L1);
       l_coh[cf][0]=L1[0].x;
       l_coh[cf][1]=L1[0].y;
       l_coh[cf][2]=L1[1].x;
       l_coh[cf][3]=L1[1].y;
       l_coh[cf][4]=L1[2].x;
       l_coh[cf][5]=L1[2].y;
       l_coh[cf][6]=L1[3].x;
       l_coh[cf][7]=L1[3].y;
     }



   
    double *coh1 = &coh[8*n];
    for(int cf=0; cf<MODEL_MAX_F && cf+cff*MODEL_MAX_F<F; cf++) {
        coh1[(cf+cff*MODEL_MAX_F)*8*B+0] = l_coh[cf][0];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+1] = l_coh[cf][1];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+2] = l_coh[cf][2];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+3] = l_coh[cf][3];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+4] = l_coh[cf][4];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+5] = l_coh[cf][5];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+6] = l_coh[cf][6];
        coh1[(cf+cff*MODEL_MAX_F)*8*B+7] = l_coh[cf][7];
    }

     }


   }

  
  }

}


/* kernel to correct residuals */
__global__ void
kernel_correct_residuals(int B, int N, int Nb, int boff, int F, int nchunk, double *__restrict__ x, const double *__restrict__ p, baseline_t *barr)  {

  /* relative baseline index */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;

  if (n<Nb) {
   int sta1=barr[n].sta1;
   int sta2=barr[n].sta2;
   /* find out which chunk to select from p : 0,1..nchunk-1 */
   int chunk=(n+boff)/((B+nchunk-1)/nchunk);
   /* create G1 and G2 Jones matrices from p */
   cuDoubleComplex G1[4],G2[4];
   G1[0].x=__ldg(&p[chunk*8*N+sta1*8]);
   G1[0].y=__ldg(&p[chunk*8*N+sta1*8+1]);
   G1[1].x=__ldg(&p[chunk*8*N+sta1*8+2]);
   G1[1].y=__ldg(&p[chunk*8*N+sta1*8+3]);
   G1[2].x=__ldg(&p[chunk*8*N+sta1*8+4]);
   G1[2].y=__ldg(&p[chunk*8*N+sta1*8+5]);
   G1[3].x=__ldg(&p[chunk*8*N+sta1*8+6]);
   G1[3].y=__ldg(&p[chunk*8*N+sta1*8+7]);
   G2[0].x=__ldg(&p[chunk*8*N+sta2*8]);
   G2[0].y=__ldg(&p[chunk*8*N+sta2*8+1]);
   G2[1].x=__ldg(&p[chunk*8*N+sta2*8+2]);
   G2[1].y=__ldg(&p[chunk*8*N+sta2*8+3]);
   G2[2].x=__ldg(&p[chunk*8*N+sta2*8+4]);
   G2[2].y=__ldg(&p[chunk*8*N+sta2*8+5]);
   G2[3].x=__ldg(&p[chunk*8*N+sta2*8+6]);
   G2[3].y=__ldg(&p[chunk*8*N+sta2*8+7]);

   for (int cf=0; cf<F; cf++) {
      cuDoubleComplex L1[4],L2[4];

      L1[0].x=x[cf*8*Nb+8*n];
      L1[0].y=x[cf*8*Nb+8*n+1];
      L1[1].x=x[cf*8*Nb+8*n+2];
      L1[1].y=x[cf*8*Nb+8*n+3];
      L1[2].x=x[cf*8*Nb+8*n+4];
      L1[2].y=x[cf*8*Nb+8*n+5];
      L1[3].x=x[cf*8*Nb+8*n+6];
      L1[3].y=x[cf*8*Nb+8*n+7];

      /* L2=G1*L1 */
      amb(G1,L1,L2);
      /* L1=L2*G2^H */
      ambt(L2,G2,L1);

      x[cf*8*Nb+8*n]=L1[0].x;
      x[cf*8*Nb+8*n+1]=L1[0].y;
      x[cf*8*Nb+8*n+2]=L1[1].x;
      x[cf*8*Nb+8*n+3]=L1[1].y;
      x[cf*8*Nb+8*n+4]=L1[2].x;
      x[cf*8*Nb+8*n+5]=L1[2].y;
      x[cf*8*Nb+8*n+6]=L1[3].x;
      x[cf*8*Nb+8*n+7]=L1[3].y;

   }
  }
}

/* kernel to convert time (JD) to GMST angle*/
__global__ void 
kernel_convert_time(int T, double *time_utc) {

  /* global thread index */
  unsigned int n=threadIdx.x+blockDim.x*blockIdx.x;
  if (n<T) {
   /* convert time */
   double t_ut1=(__ldg(&time_utc[n])-2451545.0)/36525.0;
   /* use Horners rule */
   double theta=67310.54841 + t_ut1*((876600.0*3600.0 + 8640184.812866) + t_ut1*(0.093104-(6.2*10e-6)*(t_ut1)));
   double thetaGMST=fmod((fmod(theta,86400.0*(theta/fabs(theta)))/240.0),360.0);
   time_utc[n]=thetaGMST;
  }

}


__global__ void
kernel_fns_shapelet_coh(float u, float v, const float *__restrict__ modes, const float *__restrict__ fact, int n0, float beta, float *J_C_J) {
  extern __shared__ float Jprod[]; /* 8*threads shared mem per block */
  /* global thread index : equal to the mode [0,n0*n0-1]*/
  unsigned int n = threadIdx.x + blockDim.x*blockIdx.x;
  int tid = threadIdx.x;

  /* separate mode to n1,n2 */
  unsigned int n1=n%n0;
  unsigned int n2=n/n0;
  float uu=u*beta;
  float vv=v*beta;

  int val_finite=__fdiv_rz(SMALLEST_SPATIAL_SCALE_FAC,__fsqrt_rz(uu*uu+vv*vv))>beta?1:0; 
  if (val_finite && n<n0*n0 && n1<n0 && n2<n0) {
   float basis=H_e(uu,n1)/__fsqrt_rn((float)(2<<n1)*fact[n1])*__expf(-0.5f*uu*uu)
      *H_e(vv,n2)/__fsqrt_rn((float)(2<<n2)*fact[n2])*__expf(-0.5f*vv*vv);

   if((n1+n2)%2==0)  {/* even (basis is real) or odd (basis is imaginary)*/;
    /* multiply 8 values of modes[] */
    Jprod[8*tid]=basis*modes[8*n];
    Jprod[8*tid+1]=basis*modes[8*n+1];
    Jprod[8*tid+2]=basis*modes[8*n+2];
    Jprod[8*tid+3]=basis*modes[8*n+3];
    Jprod[8*tid+4]=basis*modes[8*n+4];
    Jprod[8*tid+5]=basis*modes[8*n+5];
    Jprod[8*tid+6]=basis*modes[8*n+6];
    Jprod[8*tid+7]=basis*modes[8*n+7];
   } else {
    Jprod[8*tid+1]=basis*modes[8*n];
    Jprod[8*tid]=-basis*modes[8*n+1];
    Jprod[8*tid+3]=basis*modes[8*n+2];
    Jprod[8*tid+2]=-basis*modes[8*n+3];
    Jprod[8*tid+5]=basis*modes[8*n+4];
    Jprod[8*tid+4]=-basis*modes[8*n+5];
    Jprod[8*tid+7]=basis*modes[8*n+6];
    Jprod[8*tid+6]=-basis*modes[8*n+7];
   }
  } else {
   Jprod[8*tid]=Jprod[8*tid+1]=Jprod[8*tid+2]=Jprod[8*tid+3]=
    Jprod[8*tid+4]=Jprod[8*tid+5]=Jprod[8*tid+6]=Jprod[8*tid+7]=0.0f;
  }
  __syncthreads();
  // Build summation tree over elements, assuming blockDim.x is power of 2.
  for(int s=blockDim.x/2; s>0; s=s/2) {
    if(tid < s) {
      Jprod[8*tid] += Jprod[8*(tid + s)];
      Jprod[8*tid+1] += Jprod[8*(tid + s)+1];
      Jprod[8*tid+2] += Jprod[8*(tid + s)+2];
      Jprod[8*tid+3] += Jprod[8*(tid + s)+3];
      Jprod[8*tid+4] += Jprod[8*(tid + s)+4];
      Jprod[8*tid+5] += Jprod[8*(tid + s)+5];
      Jprod[8*tid+6] += Jprod[8*(tid + s)+6];
      Jprod[8*tid+7] += Jprod[8*(tid + s)+7];
    }
   __syncthreads();
  }

  /* copy back the sum to proper location in ed */
  if(tid==0) {
   J_C_J[8*blockIdx.x]=Jprod[0];
   J_C_J[8*blockIdx.x+1]=Jprod[1];
   J_C_J[8*blockIdx.x+2]=Jprod[2];
   J_C_J[8*blockIdx.x+3]=Jprod[3];
   J_C_J[8*blockIdx.x+4]=Jprod[4];
   J_C_J[8*blockIdx.x+5]=Jprod[5];
   J_C_J[8*blockIdx.x+6]=Jprod[6];
   J_C_J[8*blockIdx.x+7]=Jprod[7];
  }
}


__global__ void
plus_reduce(const float *__restrict__ input, int N, int blockDim_2, double *coh) {
 // Each block loads its 8 elements into shared memory
 extern __shared__ float x[];
 int tid = threadIdx.x;
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 if (i<N) {
  x[8*tid] =input[8*i];
  x[8*tid+1] =input[8*i+1];
  x[8*tid+2] =input[8*i+2];
  x[8*tid+3] =input[8*i+3];
  x[8*tid+4] =input[8*i+4];
  x[8*tid+5] =input[8*i+5];
  x[8*tid+6] =input[8*i+6];
  x[8*tid+7] =input[8*i+7];
 } else {
  x[8*tid] =0.0f;
  x[8*tid+1] =0.0f;
  x[8*tid+2] =0.0f;
  x[8*tid+3] =0.0f;
  x[8*tid+4] =0.0f;
  x[8*tid+5] =0.0f;
  x[8*tid+6] =0.0f;
  x[8*tid+7] =0.0f;
 }
 __syncthreads();
 // Build summation tree over elements, handling case where B is not a power of two.
  int nTotalThreads = blockDim_2; // Total number of threads, rounded up to the next power of two
  while(nTotalThreads > 1) {
   int halfPoint = (nTotalThreads >> 1); // divide by two
    if (tid < halfPoint) {
     int thread2 = tid + halfPoint;
     if (thread2 < blockDim.x) { // Skipping the fictitious threads blockDim.x ... blockDim_2-1
      x[8*tid] = x[8*tid]+x[8*thread2];
      x[8*tid+1] = x[8*tid+1]+x[8*thread2+1];
      x[8*tid+2] = x[8*tid+2]+x[8*thread2+2];
      x[8*tid+3] = x[8*tid+3]+x[8*thread2+3];
      x[8*tid+4] = x[8*tid+4]+x[8*thread2+4];
      x[8*tid+5] = x[8*tid+5]+x[8*thread2+5];
      x[8*tid+6] = x[8*tid+6]+x[8*thread2+6];
      x[8*tid+7] = x[8*tid+7]+x[8*thread2+7];
     }
    }
    __syncthreads();
    nTotalThreads = halfPoint; // Reducing the binary tree size by two
 }

 /* add back to total */
 if( tid == 0 ) {
  coh[0]=(double)x[tid];
  coh[1]=(double)x[tid+1];
  coh[2]=(double)x[tid+2];
  coh[3]=(double)x[tid+3];
  coh[4]=(double)x[tid+4];
  coh[5]=(double)x[tid+5];
  coh[6]=(double)x[tid+6];
  coh[7]=(double)x[tid+7];
 }
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


/* 
  precalculate station beam:
  N: no of stations
  T: no of time slots
  K: no of sources
  F: no of frequencies
  freqs: frequencies Fx1
  longitude, latitude: Nx1 station locations
  time_utc: Tx1 time
  Nelem: Nx1 array of no. of elements
  xx,yy,zz: Nx1 arrays of Nelem[] dipole locations
  ra,dec: Kx1 source positions
  beam: output beam values NxTxKxF values
  ph_ra0,ph_dec0: beam pointing direction
  ph_freq0: beam referene freq
  wideband: 0: use freq0 for beamformer freq, 1: use each freqs[] for beamformer freq
*/
void
cudakernel_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, int *Nelem, float **xx, float **yy, float **zz, float *ra, float *dec, float ph_ra0, float ph_dec0, float ph_freq0, float *beam, int wideband) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* note: make sure we do not exceed max no of blocks available, otherwise (too many sources, loop over source id) */

  /* 2D grid of threads: x dim->data, y dim-> stations */
  dim3 grid(1, 1, 1);
  grid.x = (int)ceilf((K*T*F) / (float)ThreadsPerBlock);
  grid.y = N;

  kernel_array_beam<<<grid,ThreadsPerBlock>>>(N,T,K,F,freqs,longitude,latitude,time_utc,Nelem,xx,yy,zz,ra,dec,ph_ra0,ph_dec0,ph_freq0,beam,wideband);
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

/*
  precalculate station beam, same as cudakernel_array_beam,
  but with a two stage beamformer, first the tile beam is calculated,
  next the final beam using tile centroids
  additional paramters:
  b_ra0,b_dec0: tile beam pointing direction

  xx,yy,zz: Nx1 arrays of HBA_TILE_SIZE+Nelem[]
  first HBA_TILE_SIZE values are the rotated dipole locations in a tile
  next Nelem[] values are the rotated tile centroid locations
 */
void
cudakernel_tile_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, int *Nelem, float **xx, float **yy, float **zz, float *ra, float *dec, float b_ra0, float b_dec0, float ph_ra0, float ph_dec0, float ph_freq0, float *beam, int wideband) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* note: make sure we do not exceed max no of blocks available, otherwise (too many sources, loop over source id) */

  /* 2D grid of threads: x dim->data, y dim-> stations */
  dim3 grid(1, 1, 1);
  grid.x = (int)ceilf((K*T*F) / (float)ThreadsPerBlock);
  grid.y = N;

  kernel_tile_array_beam<<<grid,ThreadsPerBlock>>>(N,T,K,F,freqs,longitude,latitude,time_utc,Nelem,xx,yy,zz,ra,dec,b_ra0,b_dec0,ph_ra0,ph_dec0,ph_freq0,beam,wideband);
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


/* element beam parameters
  N: no of stations
  T: no of time slots
  K: no of sources
  F: no of frequencies
  freqs: frequencies Fx1
  longitude, latitude: Nx1 station locations
  time_utc: Tx1 time
  ra,dec: Kx1 source positions
 * Nmodes : M(M+1)/2
 * M: model order
 * beta: scale
 * pattern_phi, pattern_theta: Nmodes x 1 complex, 2Nmodes x 1 float arrays
 * pattern_preamble: Nmodes x 1 float array
  beam: output element beam values 8*NxTxKxF values
  wideband: 0: use freq0 for beamformer freq, 1: use each freqs[] for beamformer freq
 */
void
cudakernel_element_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, float *ra, float *dec, int Nmodes, int M, float beta, float *pattern_phi, float *pattern_theta, float *pattern_preamble, float *beam, int wideband) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif
  // Set a heap size of 128 megabytes. Note that this must
  // be done before any kernel is launched. 
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
  // for an array of max 24*16 x 2  double, the default 8MB is ok

  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* note: make sure we do not exceed max no of blocks available, otherwise (too many sources, loop over source id) */

  /* 2D grid of threads: x dim->data, y dim-> stations */
  dim3 grid(1, 1, 1);
  grid.x = (int)ceilf((K*T*F) / (float)ThreadsPerBlock);
  grid.y = N;

  kernel_element_beam<<<grid,ThreadsPerBlock>>>(N,T,K,F,freqs,longitude,latitude,time_utc,ra,dec,Nmodes,M,beta,pattern_phi,pattern_theta,pattern_preamble,beam,wideband);
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



/* 
  calculate coherencies:
  B: total baselines (could be more than one timeslot)
  N: no of stations
  T: no of time slots
  K: no of sources
  F: no of frequencies
  u,v,w: Bx1 uvw coords
  barr: Bx1 array of baseline/flag info
  freqs: Fx1 frequencies
  beam: NxTxKxF array beam gain (or NULL)
  element: 8NxTxKxF element beam gain (or NULL)
  ll,mm,nn : Kx1 source coordinates
  sI: Kx1 source flux at reference freq
  stype: Kx1 source type info
  sI0: Kx1 original source referene flux
  f0: Kx1 source reference freq for calculating flux 
  spec_idx,spec_idx1,spec_idx2: Kx1 spectra info 
  exs: Kx1 array of pointers to extended source info
  deltaf,deltat: freq/time smearing integration interval
  dec0: phace reference dec
  coh: coherency Bx8 values, all K sources are added together

  dobeam: enable array, element or array+element beam if >0
*/
void
cudakernel_coherencies(int B, int N, int T, int K, int F, double *u, double *v, double *w,baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
  unsigned char *stype, double *sI0, double *sQ0, double *sU0, double *sV0, double *f0, double *spec_idx, double *spec_idx1, double *spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh,int dobeam) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  /* spawn threads to handle baselines, these threads will spawn threads for sources */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* note: make sure we do not exceed max no of blocks available, 
   otherwise (too many baselines, loop over source id) */
  int BlocksPerGrid=(B+ThreadsPerBlock-1)/ThreadsPerBlock;
  kernel_coherencies<<<BlocksPerGrid,ThreadsPerBlock>>>(B, N, T, K, F,u,v,w,barr,freqs, beam, element, ll, mm, nn, sI, sQ, sU, sV,
    stype, sI0, sQ0, sU0, sV0, f0, spec_idx, spec_idx1, spec_idx2, exs, deltaf, deltat, dec0, coh, dobeam);
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


/* p : parameters 8Nxnchunk values */
void
cudakernel_residuals(int B, int N, int T, int K, int F, double *u, double *v, double *w, double *p, int nchunk, baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
  unsigned char *stype, double *sI0, double *sQ0, double *sU0, double *sV0, double *f0, double *spec_idx, double *spec_idx1, double *spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh,int dobeam) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  /* spawn threads to handle baselines, these threads will loop over sources */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* note: make sure we do not exceed max no of blocks available, 
   otherwise (too many baselines, loop over source id) */
  int BlocksPerGrid=(B+ThreadsPerBlock-1)/ThreadsPerBlock;
  kernel_residuals<<<BlocksPerGrid,ThreadsPerBlock>>>(B, N, T, K, F,u,v,w,p,nchunk,barr,freqs, beam, element, ll, mm, nn, sI, sQ, sU, sV,
    stype, sI0, sQ0, sU0, sV0, f0, spec_idx, spec_idx1, spec_idx2, exs, deltaf, deltat, dec0, coh, dobeam);
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
cudakernel_correct_residuals(int B, int N, int Nb, int boff, int F, int nchunk, double *x, double *p, baseline_t *barr) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  /* spawn threads to handle baselines */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int BlocksPerGrid=(Nb+ThreadsPerBlock-1)/ThreadsPerBlock;
  kernel_correct_residuals<<<BlocksPerGrid,ThreadsPerBlock>>>(B, N, Nb, boff, F, nchunk, x, p, barr);
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

/* convert time JD to GMST angle
  store result at the same location */
void
cudakernel_convert_time(int T, double *time_utc) {
#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  /* note: make sure we do not exceed max no of blocks available, 
   otherwise (too many baselines, loop over source id) */
  int BlocksPerGrid=(T+ThreadsPerBlock-1)/ThreadsPerBlock;
  kernel_convert_time<<<BlocksPerGrid,ThreadsPerBlock>>>(T,time_utc);
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

#define CUDA_DBG
/* calculate visibilites for shapelet model at u,v,
modes: (device memory) n0*n0*(2x2)x2 double, n0*n0*(2x2) complex double
fact: (device memory) n0 factorial array
coh: (device memory) 8x1 double, 4x1 complex double
*/
void
cudakernel_calculate_shapelet_coherencies(float u, float v, float *modes, float *fact, int n0, float beta, double *coh) {

#ifdef CUDA_DBG
  cudaError_t error;
  error = cudaGetLastError();
#endif

  /* split n0*n0 modes into threads */
  int ThreadsPerBlock=DEFAULT_TH_PER_BK;
  int BlocksPerGrid=(n0*n0+ThreadsPerBlock-1)/ThreadsPerBlock;
  /* each thread computes basis for that mode (n1,n2), finds the product
     of basis with modes (2x2 complex),
     thereafter, summation over each block, and result written back to global mem */

  /* global mem to store summation per block */
  float *J_C_J;
  cudaMalloc((void**)&J_C_J, 8*sizeof(float)*BlocksPerGrid);
  cudaMemset(J_C_J, 0, 8*sizeof(float)*BlocksPerGrid);
  /* shared mem: 8*ThreadsPerBlock */
  kernel_fns_shapelet_coh<<< BlocksPerGrid, ThreadsPerBlock, 8*sizeof(float)*ThreadsPerBlock >>>(-u, v, modes, fact, n0, beta, J_C_J);

  /* launch 1 block, threads=BlocksPerGrid */
  plus_reduce<<< 1, BlocksPerGrid, 8*sizeof(float)*BlocksPerGrid>>>(J_C_J, BlocksPerGrid, NearestPowerOf2(BlocksPerGrid), coh);
  
  cudaFree(J_C_J);
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
