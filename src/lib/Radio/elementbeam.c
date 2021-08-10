/*
 *
 Copyright (C) 2021 Sarod Yatawatta <sarod@users.sf.net>  
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



#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <unistd.h>
#include <string.h>

#include "Radio.h"
#include "elementcoeff.h"

/* get beam type LBA/HBA and frequency
   return beam pattern coeff vectors for theta/phi patterns 
   element_type: ELEM_LBA or ELEM_HBA
   freq: in Hz
*/
int 
set_elementcoeffs(int element_type,  double frequency, elementcoeff *ecoeff) {
  /* common to all beam types */
  ecoeff->M=BEAM_ELEM_MODES; /* model order 1,2.. */
  ecoeff->Nmodes=ecoeff->M*(ecoeff->M+1)/2;
  ecoeff->beta=BEAM_ELEM_BETA;

  if ((ecoeff->pattern_phi=(complex double*)calloc((size_t)ecoeff->Nmodes,sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((ecoeff->pattern_theta=(complex double*)calloc((size_t)ecoeff->Nmodes,sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((ecoeff->preamble=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* convert frequency to GHz */
  double myfreq=frequency/1e9;

  /* pointers to freq and pattern arrays */
  int Nfreq=0;
  double *freqs;
  complex double *phi, *theta;
  switch(element_type) {
     case ELEM_LBA:
       Nfreq=LBA_FREQS;
       phi=(complex double *)&lba_beam_elem_phi[0][0];
       theta=(complex double *)&lba_beam_elem_theta[0][0];
       freqs=(double *)lba_beam_elem_freqs;
       //printf("ELEM LBA\n");
       break;
     case ELEM_HBA:
       Nfreq=HBA_FREQS;
       phi=(complex double *)&hba_beam_elem_phi[0][0];
       theta=(complex double *)&hba_beam_elem_theta[0][0];
       freqs=(double *)hba_beam_elem_freqs;
       //printf("ELEM HBA\n");
       break;

     default:
      fprintf(stderr,"%s: %d: undefined element beam type\n",__FILE__,__LINE__);
      exit(1);
  }

  //printf("myfreq=%lf\n",myfreq);
  //printf("%lf %lf\n",creal(lba_beam_elem_phi[0][0]),cimag(lba_beam_elem_phi[0][0]));
  //printf("%lf %lf\n",creal(phi[0]),cimag(phi[0]));
  /* find correct freq interval */
  int idl,idh=0;
  while(idh<Nfreq && myfreq>freqs[idh]) { idh++; }
  if (idh==Nfreq) { /* higher edge */
    idl=idh=Nfreq-1;
  } else if (idh==0) { /* lower edge */
    idl=idh;
  } else {
    idl=idh-1;
  }
  /* now freqs[idl]<myfreq<=freqs[idh] */
  //printf("%d %lf < %lf <= %d %lf\n",idl,freqs[idl],myfreq,idh,freqs[idh]);
  if (idh==idl) { /* edge case, just copy this value */
    my_ccopy(ecoeff->Nmodes, &phi[idl*ecoeff->Nmodes], 1, ecoeff->pattern_phi, 1);
    my_ccopy(ecoeff->Nmodes, &theta[idl*ecoeff->Nmodes], 1, ecoeff->pattern_theta, 1);
  } else {
    /* interpolate */
    /* find interpolation weights */
    double wl=myfreq-freqs[idl];
    double wh=freqs[idh]-myfreq;
    double w1=wl/(wl+wh);
    
    /* first copy to temp buffers */
    complex double *xh;
    if ((xh=(complex double*)calloc((size_t)ecoeff->Nmodes,sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
    }
    my_ccopy(ecoeff->Nmodes, &phi[idh*ecoeff->Nmodes], 1, xh, 1);
    my_ccopy(ecoeff->Nmodes, &phi[idl*ecoeff->Nmodes], 1, ecoeff->pattern_phi, 1);
    my_cscal(ecoeff->Nmodes,1.0-w1+_Complex_I*0.0,ecoeff->pattern_phi);
    my_caxpy(ecoeff->Nmodes,xh,w1+_Complex_I*0.0,ecoeff->pattern_phi);

    my_ccopy(ecoeff->Nmodes, &theta[idh*ecoeff->Nmodes], 1, xh, 1);
    my_ccopy(ecoeff->Nmodes, &theta[idl*ecoeff->Nmodes], 1, ecoeff->pattern_theta, 1);
    my_cscal(ecoeff->Nmodes,1.0-w1+_Complex_I*0.0,ecoeff->pattern_theta);
    my_caxpy(ecoeff->Nmodes,xh,w1+_Complex_I*0.0,ecoeff->pattern_theta);
    free(xh);
  }
#ifdef DEBUG
  for (int i=0; i<ecoeff->Nmodes; i++) {
   printf("%d %lf %lf %lf %lf\n",i,creal(ecoeff->pattern_phi[i]),cimag(ecoeff->pattern_phi[i]),creal(ecoeff->pattern_theta[i]),cimag(ecoeff->pattern_theta[i]));
  }
#endif
 
  /* factorial array */
  int *factorial;
  if ((factorial=(int*)calloc((size_t)ecoeff->Nmodes,sizeof(int)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  factorial[0]=1; /* 0! */
  for (int i=1; i<ecoeff->Nmodes; i++) {
    factorial[i]=factorial[i-1]*i;
  }

  /* calculate preamble for basis calculation */
  int idx=0;
  for (int n=0; n<ecoeff->M; n++) {
    for (int m=-n; m<=n; m+=2) {
      int absm=m>=0?m:-m; /* |m| */
      /* sqrt { ((n-|m|)/2)! / pi ((n+|m|)/2)! } */
      ecoeff->preamble[idx]=sqrt(M_1_PI*(double)factorial[(n-absm)/2]/(double)factorial[(n+absm)/2]);
      /* (-1)^(n-|m|)/2 */
      if (((n-absm)/2)%2) {ecoeff->preamble[idx]=-ecoeff->preamble[idx];}
      /* 1/beta^{1+|m|} */
      ecoeff->preamble[idx] *=pow(ecoeff->beta,-1.0-absm);
      //printf("n=%d m=%d |m|=%d %d %lf\n",n,m,absm,idx,ecoeff->preamble[idx]);
      idx++;
    }
  }

  free(factorial);
  return 0;
}


int 
free_elementcoeffs(elementcoeff ecoeff) {

  free(ecoeff.pattern_phi);
  free(ecoeff.pattern_theta);
  free(ecoeff.preamble);
  return 0;
}


/* generalized Laguerre polynomial L_p^q(x) */
/* for calculating L_{n-|m|/2}^|m| (x) */
static double
L_g1(int p, int q, double x) {
  /* max p: (n-|m|)/2 = n/2 */
  if(p==0) return 1.0;
  if(p==1) return 1.0-x+(double)q;
  /* else, use two variables to store past values */
  double L_p=0.0,L_p_1,L_p_2;
  L_p_2=1.0;
  L_p_1=1.0-x+(double)q;
  for (int i=2; i<=p; i++) {
   double p_1=1.0/(double)i;
   L_p=(2.0+p_1*((double)q-1.0-x))*L_p_1-(1.0+p_1*(q-1))*L_p_2;
   L_p_2=L_p_1;
   L_p_1=L_p;
  }
  return L_p;
}


elementval
eval_elementcoeffs(double r, double theta, elementcoeff *ecoeff) {
  /* evaluate r^2/beta^2 */
  double rb=pow(r/ecoeff->beta,2);
  /* evaluate e^(-r^2/2beta^2) */
  double ex=exp(-0.5*rb);
 
  elementval eval;
  eval.phi=0.0+_Complex_I*0.0;
  eval.theta=0.0+_Complex_I*0.0;
  int idx=0;
  for (int n=0; n<ecoeff->M; n++) {
    for (int m=-n; m<=n; m+=2) {
     int absm=m>=0?m:-m; /* |m| */
     /* evaluate L_((n-|m|)/2)^|m| ( . ) */
     double Lg=L_g1((n-absm)/2,absm,rb);
     /* evaluate r^|m| (with pi/4 offset) */
     double rm=pow(M_PI_4+r,(double)absm);
     /* evaluate exp(-j*m*theta) */
     double s,c;
     sincos(-(double)m*theta,&s,&c);

     /* find product of real terms (including the preamble) */
     double pr=rm*Lg*ex*ecoeff->preamble[idx];
     double re,im;
     /* basis function re+j*im */
     re=pr*c;
     im=pr*s; 

     eval.phi+=ecoeff->pattern_phi[idx]*(re+_Complex_I*im);
     eval.theta+=ecoeff->pattern_theta[idx]*(re+_Complex_I*im);
     idx++;
    }
  }
  

 return eval;
}
