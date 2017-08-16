/*
 *
 Copyright (C) 2010- Sarod Yatawatta <sarod@users.sf.net>  
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


#include "buildsky.h"

/* fitting function  - single point  (ref flux, spectral index)
 p: mx1 format sI0,sP0
 x: nx1 pixel fluxes, from model, pixels ordered with freq
*/ 
/* 3rd order spectra */
static void
mylm_fit_single_sipf(double *p, double *x, int m, int n, void *data) {
 fit_double_point_dataf *dp=(fit_double_point_dataf*)data;
 int cj;
 double fratio,fratio2,fratio3;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 for (cj=0; cj<dp->Nf; ++cj) { /* freq iteration */
   fratio=log(dp->freqs[cj]/dp->ref_freq);
   fratio2=fratio*fratio;
   fratio3=fratio2*fratio;
   x[cj]=exp(log(p[0])+p[1]*fratio+p[2]*fratio2+p[3]*fratio3);
 }
}

/* 2nd order spectra */
static void
mylm_fit_single_sipf_2d(double *p, double *x, int m, int n, void *data) {
 fit_double_point_dataf *dp=(fit_double_point_dataf*)data;
 int cj;
 double fratio,fratio2;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 for (cj=0; cj<dp->Nf; ++cj) { /* freq iteration */
   fratio=log(dp->freqs[cj]/dp->ref_freq);
   fratio2=fratio*fratio;
   x[cj]=exp(log(p[0])+p[1]*fratio+p[2]*fratio2);
 }
}

/* 1 order spectra */
static void
mylm_fit_single_sipf_1d(double *p, double *x, int m, int n, void *data) {
 fit_double_point_dataf *dp=(fit_double_point_dataf*)data;
 int cj;
 double fratio;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 for (cj=0; cj<dp->Nf; ++cj) { /* freq iteration */
   fratio=log(dp->freqs[cj]/dp->ref_freq);
   x[cj]=exp(log(p[0])+p[1]*fratio);
 }
}


double
fit_single_point0_f(hpixelf *parr, int npix, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, int  maxiter, double *ll1, double *mm1, double *sI1, double *sP){

 double xm,sumI,*peakIneg,*peakIpos,*peakI;
 int ci,cj;
 double l,m,ll,mm;
 double alpha;
 double dpow;


 if ((peakIneg=(double*)calloc((size_t)(Nf),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((peakIpos=(double*)calloc((size_t)(Nf),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }

 xm=sumI=0.0;
 ll=mm=0.0;
 for (ci=0; ci<Nf; ++ci) {
  peakIneg[ci]=INFINITY_L;
  peakIpos[ci]=-INFINITY_L; 
 }
 for (cj=0; cj<Nf; ++cj ){
   for (ci=0; ci<npix; ci++) {
       sumI+=parr[ci].sI[cj]; /* FIX negative values */
       if (peakIpos[cj]<parr[ci].sI[cj]) { /* detect -ve or +ve peak */
         peakIpos[cj]=parr[ci].sI[cj];
       }
       if (peakIneg[cj]>parr[ci].sI[cj]) { /* detect -ve or +ve peak */
         peakIneg[cj]=parr[ci].sI[cj];
       }
       ll+=parr[ci].sI[cj]*parr[ci].l;
       mm+=parr[ci].sI[cj]*parr[ci].m;
   }
 }
 ll/=sumI;
 mm/=sumI;
 /* now decide if -ve or +ve peak is used for model, choose 
  one with highest magnitude */
 if (fabs(peakIneg[0])> fabs(peakIpos[0])) {
  peakI=peakIneg;
 } else {
  peakI=peakIpos;
 }
 /* estimate spectral index, ignore frequencies equal to f0 */
 /* alpha = log(I/I_0)/log(f/f_0) --> ideal */
 alpha=0.0;
 for (cj=1; cj<Nf; ++cj) {
   alpha+=log(peakI[cj]/peakI[0])/(log(freqs[cj]/freqs[0])+TOL_L);
 }
 alpha=alpha/(double)(Nf-1);

 /* calculate error */
 sumI=0.0;
 for (cj=0; cj<Nf; ++cj ){
  dpow=peakI[0]*pow(freqs[cj]/freqs[0],alpha);
  for (ci=0; ci<npix; ci++) {
   /* rotate by beam pa */
   l=(-(parr[ci].l-ll)*sin(bpa[cj])+(parr[ci].m-mm)*cos(bpa[cj]))/bmaj[cj];
   m=(-(parr[ci].l-ll)*cos(bpa[cj])-(parr[ci].m-mm)*sin(bpa[cj]))/bmin[cj];
   xm=parr[ci].sI[cj]-dpow*exp(-(l*l+m*m));
   /* squared error */
   sumI+=xm*xm;
  }
 }

 *sI1=peakI[0];
 free(peakIneg);
 free(peakIpos);
#ifdef DEBUG
 printf("Error = %lf\n",sumI);
#endif
 /* AIC=2*k+N*ln(sum_error) */
 /* k=4, flux, 2 positions, spec index */
 sumI=2*4+npix*Nf*log(sumI);
 *ll1=ll;
 *mm1=mm;
 *sP=alpha;
 return sumI;
}


/* fitting function  - single point  (no position)
 p: mx1 format sI0,sP0
 x: nx1 pixel fluxes, from model, pixels ordered with freq
*/ 
/* 3rd order spectra */
static void
mylm_fit_single_pf(double *p, double *x, int m, int n, void *data) {
 fit_double_point_dataf *dp=(fit_double_point_dataf*)data;
 int ci,cj,ck;

 /* reference freq is first freq */
 double ll,mm,dpow;
 double sbpa;
 double cbpa;
 double invbmaj;
 double invbmin;
 double fratio,fratio2,fratio3;
 /* actual pixels per freq */
 int N=n/dp->Nf;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 ck=0;
 for (cj=0; cj<dp->Nf; ++cj) { /* freq iteration */
  //dpow=p[0]*pow(dp->freqs[cj]/dp->ref_freq,p[1]);
  fratio=log(dp->freqs[cj]/dp->ref_freq);
  fratio2=fratio*fratio;
  fratio3=fratio2*fratio;
  dpow=exp(log(p[0])+p[1]*fratio+p[2]*fratio2+p[3]*fratio3);
  sincos(dp->bpa[cj],&sbpa,&cbpa);
  invbmaj=1.0/dp->bmaj[cj];
  invbmin=1.0/dp->bmin[cj];
  for (ci=0; ci<N; ++ci) { /* pixel iteration */
   ll=(-(dp->parr[ci].l-*(dp->ll))*sbpa+(dp->parr[ci].m-*(dp->mm))*cbpa)*invbmaj;
   mm=(-(dp->parr[ci].l-*(dp->ll))*cbpa-(dp->parr[ci].m-*(dp->mm))*sbpa)*invbmin;
   /* sI=I_0 *(f/f0)^{sP} */
   x[ck++]+=dpow*exp(-(ll*ll+mm*mm));
  }
 }
}

/* 2nd order spectra */
static void
mylm_fit_single_pf_2d(double *p, double *x, int m, int n, void *data) {
 fit_double_point_dataf *dp=(fit_double_point_dataf*)data;
 int ci,cj,ck;

 /* reference freq is first freq */
 double ll,mm,dpow;
 double sbpa;
 double cbpa;
 double invbmaj;
 double invbmin;
 double fratio,fratio2;
 /* actual pixels per freq */
 int N=n/dp->Nf;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 ck=0;
 for (cj=0; cj<dp->Nf; ++cj) { /* freq iteration */
  //dpow=p[0]*pow(dp->freqs[cj]/dp->ref_freq,p[1]);
  fratio=log(dp->freqs[cj]/dp->ref_freq);
  fratio2=fratio*fratio;
  dpow=exp(log(p[0])+p[1]*fratio+p[2]*fratio2);
  sincos(dp->bpa[cj],&sbpa,&cbpa);
  invbmaj=1.0/dp->bmaj[cj];
  invbmin=1.0/dp->bmin[cj];
  for (ci=0; ci<N; ++ci) { /* pixel iteration */
   ll=(-(dp->parr[ci].l-*(dp->ll))*sbpa+(dp->parr[ci].m-*(dp->mm))*cbpa)*invbmaj;
   mm=(-(dp->parr[ci].l-*(dp->ll))*cbpa-(dp->parr[ci].m-*(dp->mm))*sbpa)*invbmin;
   /* sI=I_0 *(f/f0)^{sP} */
   x[ck++]+=dpow*exp(-(ll*ll+mm*mm));
  }
 }
}

/* 1 order spectra */
static void
mylm_fit_single_pf_1d(double *p, double *x, int m, int n, void *data) {
 fit_double_point_dataf *dp=(fit_double_point_dataf*)data;
 int ci,cj,ck;

 /* reference freq is first freq */
 double ll,mm,dpow;
 double sbpa;
 double cbpa;
 double invbmaj;
 double invbmin;
 double fratio;
 /* actual pixels per freq */
 int N=n/dp->Nf;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 ck=0;
 for (cj=0; cj<dp->Nf; ++cj) { /* freq iteration */
  //dpow=p[0]*pow(dp->freqs[cj]/dp->ref_freq,p[1]);
  fratio=log(dp->freqs[cj]/dp->ref_freq);
  dpow=exp(log(p[0])+p[1]*fratio);
  sincos(dp->bpa[cj],&sbpa,&cbpa);
  invbmaj=1.0/dp->bmaj[cj];
  invbmin=1.0/dp->bmin[cj];
  for (ci=0; ci<N; ++ci) { /* pixel iteration */
   ll=(-(dp->parr[ci].l-*(dp->ll))*sbpa+(dp->parr[ci].m-*(dp->mm))*cbpa)*invbmaj;
   mm=(-(dp->parr[ci].l-*(dp->ll))*cbpa-(dp->parr[ci].m-*(dp->mm))*sbpa)*invbmin;
   /* sI=I_0 *(f/f0)^{sP} */
   x[ck++]+=dpow*exp(-(ll*ll+mm*mm));
  }
 }
}

double
fit_single_point_f(hpixelf *parr, int npix, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, int  maxiter, double *ll1, double *mm1, double *sI1, double *sP1){
 int ci,cj;
 double *p,*p1,*p2, // params m x 1
     *x; // observed data n x 1, the image pixel fluxes
 int m,n;

 double opts[CLM_OPTS_SZ], info[CLM_INFO_SZ];
 double *peakI,*peakIneg,*peakIpos;
 
 double sumI;
 double ll,mm;
 double xm,le,me,dpow;

 /* calculation of mean error */
 double mean_bmaj,mean_bmin,mean_bpa,mean_flux,mean_error;

 fit_double_point_dataf lmdata;

 opts[0]=CLM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-15;
  opts[4]=-CLM_DIFF_DELTA; 

 m=4; /* 1x1 flux, 3x1 spec index */
 n=Nf; /* no of pixels: use only peak value */

 if ((p=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((p1=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((p2=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((x=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }

 if ((peakIneg=(double*)calloc((size_t)(Nf),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((peakIpos=(double*)calloc((size_t)(Nf),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }

 sumI=0.0;
 ll=mm=0.0;
 for (ci=0; ci<Nf; ++ci) {
  peakIneg[ci]=INFINITY_L;
  peakIpos[ci]=-INFINITY_L; 
 }
 for (cj=0; cj<Nf; ++cj ){
   for (ci=0; ci<npix; ci++) {
       sumI+=parr[ci].sI[cj]; /* FIX negative values */
       if (peakIpos[cj]<parr[ci].sI[cj]) { /* detect -ve or +ve peak */
         peakIpos[cj]=parr[ci].sI[cj];
       }
       if (peakIneg[cj]>parr[ci].sI[cj]) { /* detect -ve or +ve peak */
         peakIneg[cj]=parr[ci].sI[cj];
       }
       ll+=parr[ci].sI[cj]*parr[ci].l;
       mm+=parr[ci].sI[cj]*parr[ci].m;
   }
 }

 ll/=sumI;
 mm/=sumI;
 /* now decide if -ve or +ve peak is used for model, choose 
  one with highest magnitude */
 if (fabs(peakIneg[0])> fabs(peakIpos[0])) {
  peakI=peakIneg;
 } else {
  peakI=peakIpos;
 }

 for (cj=0; cj<Nf; ++cj) {
  x[cj]=peakI[cj];
 }

 mean_bmaj=mean_bmin=mean_bpa=mean_flux=0.0;
 for (cj=0; cj<Nf; ++cj) {
  mean_bmaj+=bmaj[cj];
  mean_bmin+=bmin[cj];
  mean_bpa+=bpa[cj];
  mean_flux+=peakI[cj];
 }
 mean_bmaj/=(double)Nf;
 mean_bmin/=(double)Nf;
 mean_bpa/=(double)Nf;
 mean_flux/=(double)Nf;


 /* initial values */ 
 p[0]=p1[0]=p2[0]=mean_flux;
 p[1]=p[2]=p[3]=0.0;
 p1[1]=p1[2]=p1[3]=0.0;
 p2[1]=p2[2]=p2[3]=0.0;


 lmdata.Nf=Nf;
 lmdata.parr=parr;
 lmdata.freqs=freqs;
 lmdata.bmaj=bmaj;
 lmdata.bmin=bmin;
 lmdata.bpa=bpa;
 lmdata.ll=&ll; /* pointer to positions */
 lmdata.mm=&mm;
 lmdata.ref_freq=ref_freq;

 double aic1,aic2,aic3;
 //ret=dlevmar_dif(mylm_fit_single_sipf, p, x, m, n, maxiter, opts, info, NULL, NULL, (void*)&lmdata);  // no Jacobian
 clevmar_der_single_nocuda(mylm_fit_single_sipf, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
 /* penalize only 1/10 of parameters */
 aic3=0.3+log(info[1]);
 clevmar_der_single_nocuda(mylm_fit_single_sipf_2d, NULL, p2, x, m-1, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
 aic2=0.2+log(info[1]);
 clevmar_der_single_nocuda(mylm_fit_single_sipf_1d, NULL, p1, x, m-2, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
 aic1=0.1+log(info[1]);
 /* choose one with minimum error */
 if (aic3<aic2) {
     if (aic3<aic1) {
       /* 3d */
     } else {
       /* 1d */
       p[0]=p1[0];
       p[1]=p1[1];
       p[2]=p1[2];
       p[3]=p1[3];
     }
 } else {
     if (aic2<aic1) {
       /* 2d */
       p[0]=p2[0];
       p[1]=p2[1];
       p[2]=p2[2];
       p[3]=p2[3];
     } else {
       /* 1d */
       p[0]=p1[0];
       p[1]=p1[1];
       p[2]=p1[2];
       p[3]=p1[3];
     }
 }



#ifdef DEBUG
  print_levmar_info(info[0],info[1],(int)info[5], (int)info[6], (int)info[7], (int)info[8], (int)info[9]);
  printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
#endif


 /* re-calculate mean error */
 mean_error=0.0;
 for (ci=0; ci<npix; ci++) {
  le=(-(parr[ci].l-ll)*sin(mean_bpa)+(parr[ci].m-mm)*cos(mean_bpa))/mean_bmaj;
  me=(-(parr[ci].l-ll)*cos(mean_bpa)-(parr[ci].m-mm)*sin(mean_bpa))/mean_bmin;
  xm=0.0;
  for (cj=0; cj<Nf; ++cj ){
   /* average pixel value */
   xm+=parr[ci].sI[cj];
  }
  xm=xm/(double)Nf-mean_flux*exp(-(le*le+me*me));
  /* squared error */
  mean_error+=xm*xm;
 }

 double fratio,fratio2,fratio3;
 sumI=0.0;
 for (cj=0; cj<Nf; ++cj ){
  fratio=log(freqs[cj]/ref_freq);
  fratio2=fratio*fratio;
  fratio3=fratio2*fratio;
  dpow=exp(log(p[0])+p[1]*fratio+p[2]*fratio2+p[3]*fratio3);

  for (ci=0; ci<npix; ci++) {
   /* rotate by beam pa */
   le=(-(parr[ci].l-ll)*sin(bpa[cj])+(parr[ci].m-mm)*cos(bpa[cj]))/bmaj[cj];
   me=(-(parr[ci].l-ll)*cos(bpa[cj])-(parr[ci].m-mm)*sin(bpa[cj]))/bmin[cj];
   xm=parr[ci].sI[cj]-dpow*exp(-(le*le+me*me));
   /* squared error */
   sumI+=xm*xm;
  }
 }


 *sI1=p[0];
 free(peakIneg);
 free(peakIpos);
 *ll1=ll;
 *mm1=mm;
 sP1[0]=p[1];
 sP1[1]=p[2];
 sP1[2]=p[3];

 free(p);
 free(p1);
 free(p2);
 free(x);

 /* AIC, 4 parms */
 return 2*4+Nf*npix*(log(sumI)+log(mean_error));
}



/* N>1 */
double
fit_N_point_em_f(hpixelf *parr, int npix, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, int maxiter, int max_em_iter, double *ll, double *mm, double *sI, double *sP, int N, int Nh, hpoint *hull){

 int ci,cj,ck;
 double *p,*p1,*p2, // params m x 1
     *x; // observed data n x 1, the image pixel fluxes
 double *xdummy, *xsub; //extra arrays
 int m,n;
 double *b; /* affine combination */

 double opts[CLM_OPTS_SZ], info[CLM_INFO_SZ];

 double l_min,l_max,m_min,m_max,sumI;
 double fraction;

 double penalty; /* penalty for solutions with components out of pixel range */

 fit_double_point_dataf lmdata;

 /* for initial average pixel fit */
 hpixel *parrav;
 double mean_bmaj,mean_bmin,mean_bpa,mean_err;


 /***** first do a fit for average pixesl, using average PSF ******/
 if ((parrav=(hpixel*)calloc((size_t)(npix),sizeof(hpixel)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 for (ci=0; ci<npix; ++ci) {
   parrav[ci].x=parr[ci].x;
   parrav[ci].y=parr[ci].y;
   parrav[ci].l=parr[ci].l;
   parrav[ci].m=parr[ci].m;
   parrav[ci].ra=parr[ci].ra;
   parrav[ci].dec=parr[ci].dec;
   parrav[ci].sI=0.0;
   for (cj=0; cj<Nf; ++cj) {
    parrav[ci].sI+=parr[ci].sI[cj];
   }
   parrav[ci].sI/=(double)Nf;
 }
 mean_bmaj=mean_bmin=mean_bpa=0.0;
 for (cj=0; cj<Nf; ++cj) {
  mean_bmaj+=bmaj[cj];
  mean_bmin+=bmin[cj];
  mean_bpa+=bpa[cj];
 }
 mean_bmaj/=(double)Nf;
 mean_bmin/=(double)Nf;
 mean_bpa/=(double)Nf;
 

 /* we get mean_err=2*3*N+npix*log(error) */
 mean_err=fit_N_point_em(parrav, npix, mean_bmaj, mean_bmin, mean_bpa, maxiter, max_em_iter, ll, mm, sI, N, Nh, hull);

 free(parrav);


 opts[0]=CLM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-15;
  opts[4]=-CLM_DIFF_DELTA; 

 m=4; /* 1x1 flux, 3x1 spec index for each component */
 n=Nf*npix; /* no of pixels */

 if ((p=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((p1=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((p2=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((x=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((xsub=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((xdummy=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((b=(double*)calloc((size_t)(N),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }



 l_min=m_min=INFINITY_L;
 l_max=m_max=-INFINITY_L;
 sumI=0.0;
 /* only use valid pixels for initial conditions */
 for (ci=0; ci<npix; ci++) {
   if (parr[ci].ra!=-1 && parr[ci].m!=-1) {
       if (l_min>parr[ci].l) {
          l_min=parr[ci].l;
       }
       if (l_max<parr[ci].l) {
         l_max=parr[ci].l;
       }
       if (m_min>parr[ci].m) {
          m_min=parr[ci].m;
       }
       if (m_max<parr[ci].m) {
         m_max=parr[ci].m;
       }
   }
 }
 ck=0;
 for (cj=0; cj<Nf; ++cj) {
  for (ci=0; ci<npix; ci++) {
       x[ck++]=parr[ci].sI[cj];
       sumI+=parr[ci].sI[cj];
  }
 }
 sumI/=(double)n;

 fraction=1.0;///(double)N;
/**********************************/
 for (ci=0; ci<N; ci++) {
  sP[ci]=0.0;
  sP[ci+N]=0.0;
  sP[ci+2*N]=0.0;

  b[ci]=fraction;    
 }
 
  lmdata.Nf=Nf;
  lmdata.parr=parr;
  lmdata.freqs=freqs;
  lmdata.bmaj=bmaj;
  lmdata.bmin=bmin;
  lmdata.bpa=bpa;
  lmdata.ref_freq=ref_freq;

 double aic1,aic2,aic3;
 for (ci=0; ci<max_em_iter; ci++) {
   for (cj=0; cj<N; cj++) {
     /* calculate contribution from hidden data, subtract from x */
    memcpy(xdummy,x,(size_t)(n)*sizeof(double));
    for (ck=0; ck<N; ck++) {
     if (ck!=cj) {
       lmdata.ll=&ll[ck]; /* pointer to positions */
       lmdata.mm=&mm[ck];
       p[0]=sI[ck];
       p[1]=sP[ck];
       p[2]=sP[ck+N];
       p[3]=sP[ck+2*N];

       mylm_fit_single_pf(p, xsub, m, n, (void*)&lmdata);
       /* xdummy=xdummy-b*xsub */
       my_daxpy(n, xsub, -b[ck], xdummy);
     }
    }

    lmdata.ll=&ll[cj]; /* pointer to positions */
    lmdata.mm=&mm[cj];
    p[0]=p1[0]=p2[0]=sI[cj];
    p[1]=p1[1]=p2[1]=sP[cj];
    p[2]=p2[2]=sP[cj+N]; p1[2]=0.0;
    p[3]=sP[cj+2*N]; p1[3]=p2[3]=0.0;

    //ret=dlevmar_dif(mylm_fit_single_pf, p, xdummy, m, n, maxiter, opts, info, NULL, NULL, (void*)&lmdata);  // no Jacobian
    clevmar_der_single_nocuda(mylm_fit_single_pf, NULL, p, xdummy, m, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
/* penalize only 1/10 of parameters */
    aic3=0.3+log(info[1]);
    clevmar_der_single_nocuda(mylm_fit_single_pf_2d, NULL, p2, xdummy, m-1, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
    aic2=0.2+log(info[1]);
    clevmar_der_single_nocuda(mylm_fit_single_pf_1d, NULL, p1, xdummy, m-2, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
    aic1=0.1+log(info[1]);
    /* choose one with minimum error */
    if (aic3<aic2) {
     if (aic3<aic1) {
       /* 3d */
       sI[cj]=p[0];
       sP[cj]=p[1];
       sP[cj+N]=p[2];
       sP[cj+2*N]=p[3];
     } else {
       /* 1d */
       sI[cj]=p1[0];
       sP[cj]=p1[1];
       sP[cj+N]=p1[2];
       sP[cj+2*N]=p1[3];
     }
   } else {
     if (aic2<aic1) {
       /* 2d */
       sI[cj]=p2[0];
       sP[cj]=p2[1];
       sP[cj+N]=p2[2];
       sP[cj+2*N]=p2[3];
     } else {
       /* 1d */
       sI[cj]=p1[0];
       sP[cj]=p1[1];
       sP[cj+N]=p1[2];
       sP[cj+2*N]=p1[3];
     }
   }
  }
 }
/**********************************/



#ifdef DEBUG
  print_levmar_info(info[0],info[1],(int)info[5], (int)info[6], (int)info[7], (int)info[8], (int)info[9]);
  printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
#endif
 /* check for solutions such that l_min <= ll <= l_max and m_min <= mm <= m_max */
 penalty=0.0;
 for (ci=0; ci<N; ci++) {
   /* position out of range */
   if (ll[ci]<l_min || ll[ci]>l_max || mm[ci]<m_min || mm[ci]>m_max) {
    penalty+=INFINITY_L;
   }
   /* spec index too high to be true */
   if (fabs(sP[ci])>20.0) {
    penalty+=INFINITY_L;
   }
 }

 /* calculate error */
 memcpy(xdummy,x,(size_t)(n)*sizeof(double));
 for (cj=0; cj<N; cj++) {
    for (ck=0; ck<N; ck++) {
       lmdata.ll=&ll[ck]; /* pointer to positions */
       lmdata.mm=&mm[ck];
       p[0]=sI[ck];
       p[1]=sP[ck];
       p[2]=sP[ck+N];
       p[3]=sP[ck+2*N];

       mylm_fit_single_pf(p, xsub, m, n, (void*)&lmdata);
       /* xdummy=xdummy-b*xsub */
       my_daxpy(n, xsub, -1.0, xdummy);
    }
 }
 /*sumI=0.0;
 for (ci=0; ci<n; ++ci ){
  sumI+=xdummy[ci]*xdummy[ci];
 } */
 sumI=my_dnrm2(n,xdummy);
 sumI=sumI*sumI;
 free(p);
 free(p1);
 free(p2);
 free(x);
 free(xdummy);
 free(xsub);
 free(b);

 /* AIC, 4*N parms */
 //return 2*4*N+npix*Nf*log(sumI)+penalty;
 return 2*4*N+Nf*(mean_err-2*3*N)+log(sumI)*npix*Nf+penalty;
}


