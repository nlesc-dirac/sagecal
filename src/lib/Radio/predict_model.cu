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
#include "Radio.h"

/* enable this for checking for kernel failure */
//#define CUDA_DBG

//Max no. of frequencies for a single kernel to work on
//Make this large ~64 for handling data with many channels
#ifndef MODEL_MAX_F
#define MODEL_MAX_F 16
#endif

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
  float thetaLST=thetaGMST+longitude*180.0f/M_PI;

  float LHA=fmodf(thetaLST-ra*180.0f/M_PI,360.0f);

  float sinlat,coslat,sindec,cosdec,sinLHA,cosLHA;
  sincosf(latitude,&sinlat,&coslat);
  sincosf(dec,&sindec,&cosdec);
  sincosf(LHA*M_PI/180.0f,&sinLHA,&cosLHA);

  float tmp=sinlat*sindec+coslat*cosdec*cosLHA;
  float eld=asinf(tmp);

  float sinel,cosel;
  sincosf(eld,&sinel,&cosel);

  float azd=fmodf(atan2f(-sinLHA*cosdec/cosel,(sindec-sinel*sinlat)/(cosel*coslat)),2.0f*M_PI);
  if (azd<0.0f) {
   azd+=2.0f*M_PI;
  }
  *el=eld;
  *az=azd;
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
 float ph_ra0, float ph_dec0, float ph_freq0, float *beam) {

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
   theta=M_PI_2-el;
   phi=-az; /* */
   theta0=M_PI_2-el0;
   phi0=-az0; /* */
/*********************************************************************/

   /* 2*PI/C */
   const float tpc=2.0f*M_PI/CONST_C;
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
   float rat1=ph_freq0*sint0;
   float rat2=f*sint;
   r1=-tpc*(rat1*cosph0-rat2*cosph);
   r2=-tpc*(rat1*sinph0-rat2*sinph);
   r3=-tpc*(ph_freq0*cost0-f*cost);
   

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
   ssum*=Nnor;
   csum*=Nnor;
   /* store output (amplitude of beam)*/
   int boffset=itm*N*K*F+isrc*N*F+ifrq*N+istat;
   beam[boffset]=sqrtf(ssum*ssum+csum*csum);
   //printf("thread %d stat %d src %d freq %d time %d : %lf longitude=%lf latitude=%lf time=%lf freq=%lf elem=%d ra=%lf dec=%lf beam=%lf\n",n,istat,isrc,ifrq,itm,time_utc[itm],longitude[istat],latitude[istat],time_utc[itm],freqs[ifrq],Nelem[istat],ra[isrc],dec[isrc],beam[boffset]);
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

  return make_cuDoubleComplex((double)(0.5f*M_PI*expf(-(ut*ut+vt*vt))),0.0);
}



__device__ cuDoubleComplex
ring_contrib__(int *dd, float u, float v, float w) {
  exinfo_ring *dp=(exinfo_ring*)dd;
  float up,vp,a,b;

  /* first the rotation due to projection */
  up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
  vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);

  a=dp->eX; /* diameter */
  b=sqrtf(up*up+vp*vp)*a*2.0f*M_PI;

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
  b=sqrtf(up*up+vp*vp)*a*2.0f*M_PI;

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

__device__ void
calculate_uv_mode_vectors_scalar00(float u, float v, float beta, int n0, float *Av, int *cplx) {

  int xci,zci,Ntot;

  float **shpvl, *fact;
  int n1,n2,start;
  float xval;
  int signval;

  Ntot=2; /* u,v seperately */
  /* set up factorial array */
  fact=(float *)malloc((size_t)(n0)*sizeof(float));
  fact[0]=1.0f;
  for (xci=1; xci<(n0); xci++) {
    fact[xci]=(xci+1.0f)*fact[xci-1];
  }

  /* setup array to store calculated shapelet value */
  /* need max storage Ntot x n0 */
  shpvl=(float**)malloc((size_t)(Ntot)*sizeof(float*));
  for (xci=0; xci<Ntot; xci++) {
   shpvl[xci]=(float*)malloc((size_t)(n0)*sizeof(float));
  }


  /* start filling in the array from the positive values */
  zci=0;
  xval=u*beta;
  float expval=__expf(-0.5f*(float)xval*xval);
  for (xci=0; xci<n0; xci++) {
    shpvl[zci][xci]=H_e(xval,xci)*expval/__fsqrt_rn((float)(2<<xci)*fact[xci]);
  }
  zci=1;
  xval=v*beta;
  expval=exp(-0.5f*xval*xval);
  for (xci=0; xci<n0; xci++) {
    shpvl[zci][xci]=H_e(xval,xci)*expval/__fsqrt_rn((float)(2<<xci)*fact[xci]);
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
        Av[start]=-shpvl[0][n1]*shpvl[1][n2];
    } else {
        Av[start]=shpvl[0][n1]*shpvl[1][n2];
    }
   }
  }

  free(fact);
  for (xci=0; xci<Ntot; xci++) {
   free(shpvl[xci]);
  }
  free(shpvl);
}


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
  for (xci=0; xci<n0; xci++) {
    shpvl[xci]=H_e(xvalu,xci)*expvalu/__fsqrt_rn((float)(2<<xci)*fact[xci]);

    shpvl[xci+n0]=H_e(xvalv,xci)*expvalv/__fsqrt_rn((float)(2<<xci)*fact[xci]);
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
     so check this here and return 0, otherwise spurious nans may result */
  if (__fdiv_rz(100.0f,__fsqrt_rz(ut*ut+vt*vt))<dp->beta) {
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
    fact[ci]=((float)ci+1.0f)*fact[ci-1];
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
  realsum*=2.0f*M_PI*a*b;
  imagsum*=2.0f*M_PI*a*b;
  /* additional safeguards */
  if ( isnan(realsum) ) { realsum=0.0f; }
  if ( isnan(imagsum) ) { imagsum=0.0f; }
  return make_cuDoubleComplex((double)realsum,(double)imagsum);
}

__device__ void 
compute_prodterm_multifreq(int sta1, int sta2, int N, int K, int T, int F,
double phterm0, double sI0f, double sQ0f, double sU0f, double sV0f, double spec_idxf, double spec_idx1f, double spec_idx2f, double myf0,
 double myfreq, double deltaf, int dobeam, int itm, int k, int cf, const float *__restrict__ beam, int **exs, unsigned char stypeT, double u, double v, double w, double *__restrict__ output) {
     /* F>1 is assumed output: 8x1 array */
     double sinph,cosph;
     sincos(phterm0*myfreq,&sinph,&cosph);
     cuDoubleComplex prodterm=make_cuDoubleComplex(cosph,sinph);
     double If,Qf,Uf,Vf;
     If=Qf=Uf=Vf=0.0;
     /* evaluate spectra */
     double fratio=log(myfreq/myf0);
     double fratio1=fratio*fratio;
     double fratio2=fratio1*fratio;
     double cm=spec_idxf*fratio+spec_idx1f*fratio1+spec_idx2f*fratio2;
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

     /* smearing, beam */
     double scalef=1.0;
     double phterm =(phterm0*0.5*deltaf);
     if (phterm!=0.0) {
      sinph=(sin(phterm)/phterm);
      scalef *=fabs(sinph); /* catch -ve values due to rounding off */
     }

     if (dobeam) {
      /* get beam info */
      //int boffset1=sta1*K*T*F + k1*T*F + cf*T + itm;

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



    output[0]=Ix+Qx;
    output[1]=Iy+Qy;
    output[2]=Ux-Vy;
    output[3]=Vx+Uy;
    output[4]=Ux+Vy;
    output[5]=-Vx+Uy;
    output[6]=Ix-Qx;
    output[7]=Iy-Qy;

}


__device__ void 
compute_prodterm(int sta1, int sta2, int N, int K, int T,
 double phterm0, double If, double Qf, double Uf, double Vf,
 double myfreq, double deltaf, int dobeam, int itm, int k, const float *__restrict__ beam, int **exs, unsigned char stypeT, double u, double v, double w, double *__restrict__ output) {
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

     if (dobeam) {
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



    output[0]=Ix+Qx;
    output[1]=Iy+Qy;
    output[2]=Ux-Vy;
    output[3]=Vx+Uy;
    output[4]=Ux+Vy;
    output[5]=-Vx+Uy;
    output[6]=Ix-Qx;
    output[7]=Iy-Qy;

}

/* master kernel to calculate coherencies */
__global__ void 
kernel_coherencies(int B, int N, int T, int K, int F,
  const double *__restrict__ u, const double *__restrict__ v, const double *__restrict__ w,
  baseline_t *barr, const double *__restrict__ freqs, const float *__restrict__ beam, const double *__restrict__ ll, const double *__restrict__ mm, const double *__restrict__ nn, 
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
               __ldg(&(freqs[0])), deltaf, dobeam, tslot, k, beam, exs, stypeT, u_n, v_n, w_n, llcoh);

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
               myf0, __ldg(&(freqs[cf])), deltaf, dobeam, tslot, k, cf, beam, exs, stypeT, u_n, v_n, w_n,llcoh);
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
               myf0, __ldg(&(freqs[cf+cff*MODEL_MAX_F])), deltaf, dobeam, tslot, k, cf+cff*MODEL_MAX_F, beam, exs, stypeT, u_n, v_n, w_n,llcoh);
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
  baseline_t *barr, const double *__restrict__ freqs, const float *__restrict__ beam, const double *__restrict__ ll, const double *__restrict__ mm, const double *__restrict__ nn, 
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
               __ldg(&(freqs[0])), deltaf, dobeam, tslot, k, beam, exs, stypeT, u_n, v_n, w_n, llcoh);

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
               myf0, __ldg(&(freqs[cf])), deltaf, dobeam, tslot, k, cf, beam, exs, stypeT, u_n, v_n, w_n,llcoh);
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
               myf0, __ldg(&(freqs[cf+cff*MODEL_MAX_F])), deltaf, dobeam, tslot, k, cf+cff*MODEL_MAX_F, beam, exs, stypeT, u_n, v_n, w_n,llcoh);
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
  xx,yy,zz: Nx1 arrays of Nelem[] station locations
  ra,dec: Kx1 source positions
  beam: output beam values NxTxKxF values
  ph_ra0,ph_dec0: beam pointing direction
  ph_freq0: beam referene freq
*/

void
cudakernel_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, int *Nelem, float **xx, float **yy, float **zz, float *ra, float *dec, float ph_ra0, float ph_dec0, float ph_freq0, float *beam) {
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

  kernel_array_beam<<<grid,ThreadsPerBlock>>>(N,T,K,F,freqs,longitude,latitude,time_utc,Nelem,xx,yy,zz,ra,dec,ph_ra0,ph_dec0,ph_freq0,beam);
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
  beam: NxTxKxF beam gain
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

  dobeam: enable beam if >0
*/
void
cudakernel_coherencies(int B, int N, int T, int K, int F, double *u, double *v, double *w,baseline_t *barr, double *freqs, float *beam, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
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
  kernel_coherencies<<<BlocksPerGrid,ThreadsPerBlock>>>(B, N, T, K, F,u,v,w,barr,freqs, beam, ll, mm, nn, sI, sQ, sU, sV,
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
cudakernel_residuals(int B, int N, int T, int K, int F, double *u, double *v, double *w, double *p, int nchunk, baseline_t *barr, double *freqs, float *beam, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
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
  kernel_residuals<<<BlocksPerGrid,ThreadsPerBlock>>>(B, N, T, K, F,u,v,w,p,nchunk,barr,freqs, beam, ll, mm, nn, sI, sQ, sU, sV,
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


} /* extern "C" */
