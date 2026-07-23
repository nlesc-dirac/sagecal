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
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include "Dirac_radio.h"
#include "elementcoeff.h"
#include "elementcoeff_ALO.h"

/* get beam type LBA/HBA and frequency
   return beam pattern coeff vectors for theta/phi patterns 
   element_type: ELEM_LBA or ELEM_HBA
   freq: in Hz
*/
int 
set_elementcoeffs(int element_type,  double frequency, elementcoeff *ecoeff) {
  /* common to all beam types */
  if (element_type==ELEM_LBA || element_type==ELEM_HBA) {
   ecoeff->M=BEAM_ELEM_MODES; /* model order 1,2.. */
   ecoeff->beta=BEAM_ELEM_BETA;
  } else if (element_type==ELEM_ALO) {
   ecoeff->M=ALO_BEAM_ELEM_MODES; /* model order 1,2.. */
   ecoeff->beta=ALO_BEAM_ELEM_BETA;
  } else {
   fprintf(stderr,"%s: %d: undefined element beam type\n",__FILE__,__LINE__);
   exit(1);
  }
  ecoeff->Ns=1; /* by default, only 1 pattern for all stations */
  ecoeff->Nmodes=ecoeff->M*(ecoeff->M+1)/2;
  ecoeff->Nf=1;
#ifdef HAVE_CUDA
  /* print warning (GPU mode) so shared memory is not exceeded */
  if (ecoeff->Nmodes > ELEMENT_MAX_SIZE) {
   fprintf(stderr,"%s: %d: Warning: requested element beam model order is too large for shared memory.\n Increase ELEMENT_MAX_SIZE (currently %d) and re-compile if GPU beam model prediction is done",__FILE__,__LINE__,ELEMENT_MAX_SIZE);
   exit(1);
  }
#endif

  if ((ecoeff->pattern_phi=(complex double*)calloc((size_t)ecoeff->Nmodes*ecoeff->Ns,sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((ecoeff->pattern_theta=(complex double*)calloc((size_t)ecoeff->Nmodes*ecoeff->Ns,sizeof(complex double)))==0) {
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
       break;
     case ELEM_HBA:
       Nfreq=HBA_FREQS;
       phi=(complex double *)&hba_beam_elem_phi[0][0];
       theta=(complex double *)&hba_beam_elem_theta[0][0];
       freqs=(double *)hba_beam_elem_freqs;
       break;
     case ELEM_ALO:
       Nfreq=ALO_FREQS;
       phi=(complex double *)&alo_beam_elem_phi[0][0];
       theta=(complex double *)&alo_beam_elem_theta[0][0];
       freqs=(double *)alo_beam_elem_freqs;
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
  double *factorial;
  if ((factorial=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  factorial[0]=1.0; /* 0! */
  for (int i=1; i<ecoeff->Nmodes; i++) {
    factorial[i]=factorial[i-1]*(double)i;
  }

  /* calculate preamble for basis calculation */
  int idx=0;
  for (int n=0; n<ecoeff->M; n++) {
    for (int m=-n; m<=n; m+=2) {
      int absm=m>=0?m:-m; /* |m| */
      /* sqrt { ((n-|m|)/2)! / pi ((n+|m|)/2)! } */
      ecoeff->preamble[idx]=sqrt(M_1_PI*factorial[(n-absm)/2]/factorial[(n+absm)/2]);
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

/* get beam type LBA/HBA and for each frequency in frequencies array
   return beam pattern coeff vectors for theta/phi patterns
   element_type: ELEM_LBA or ELEM_HBA
   frequencies: in Hz, Nf x 1 array
*/
int
set_elementcoeffs_wb(int element_type,  double *frequencies, int Nf,  elementcoeff *ecoeff) {
  /* common to all beam types */
  if (element_type==ELEM_LBA || element_type==ELEM_HBA) {
   ecoeff->M=BEAM_ELEM_MODES; /* model order 1,2.. */
   ecoeff->beta=BEAM_ELEM_BETA;
  } else if (element_type==ELEM_ALO) {
   ecoeff->M=ALO_BEAM_ELEM_MODES; /* model order 1,2.. */
   ecoeff->beta=ALO_BEAM_ELEM_BETA;
  } else {
   fprintf(stderr,"%s: %d: undefined element beam type\n",__FILE__,__LINE__);
   exit(1);
  }
  ecoeff->Ns=1; /* by default, only 1 pattern for all stations */
  ecoeff->Nmodes=ecoeff->M*(ecoeff->M+1)/2;
  ecoeff->Nf=Nf;
  /* no need to check GPU shared memory here because we dont use it in wideband mode */

  if ((ecoeff->pattern_phi=(complex double*)calloc((size_t)ecoeff->Nmodes*ecoeff->Nf*ecoeff->Ns,sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((ecoeff->pattern_theta=(complex double*)calloc((size_t)ecoeff->Nmodes*ecoeff->Nf*ecoeff->Ns,sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((ecoeff->preamble=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

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
     case ELEM_ALO:
       Nfreq=ALO_FREQS;
       phi=(complex double *)&alo_beam_elem_phi[0][0];
       theta=(complex double *)&alo_beam_elem_theta[0][0];
       freqs=(double *)alo_beam_elem_freqs;
       break;
     default:
      fprintf(stderr,"%s: %d: undefined element beam type\n",__FILE__,__LINE__);
      exit(1);
  }

  for (int cf=0; cf<ecoeff->Nf; cf++) {
  /* convert frequency to GHz */
  double myfreq=frequencies[cf]/1e9;

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
    my_ccopy(ecoeff->Nmodes, &phi[idl*ecoeff->Nmodes], 1, &ecoeff->pattern_phi[cf*ecoeff->Nmodes], 1);
    my_ccopy(ecoeff->Nmodes, &theta[idl*ecoeff->Nmodes], 1, &ecoeff->pattern_theta[cf*ecoeff->Nmodes], 1);
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
    my_ccopy(ecoeff->Nmodes, &phi[idl*ecoeff->Nmodes], 1, &ecoeff->pattern_phi[cf*ecoeff->Nmodes], 1);
    my_cscal(ecoeff->Nmodes,1.0-w1+_Complex_I*0.0,&ecoeff->pattern_phi[cf*ecoeff->Nmodes]);
    my_caxpy(ecoeff->Nmodes,xh,w1+_Complex_I*0.0,&ecoeff->pattern_phi[cf*ecoeff->Nmodes]);

    my_ccopy(ecoeff->Nmodes, &theta[idh*ecoeff->Nmodes], 1, xh, 1);
    my_ccopy(ecoeff->Nmodes, &theta[idl*ecoeff->Nmodes], 1, &ecoeff->pattern_theta[cf*ecoeff->Nmodes], 1);
    my_cscal(ecoeff->Nmodes,1.0-w1+_Complex_I*0.0,&ecoeff->pattern_theta[cf*ecoeff->Nmodes]);
    my_caxpy(ecoeff->Nmodes,xh,w1+_Complex_I*0.0,&ecoeff->pattern_theta[cf*ecoeff->Nmodes]);
    free(xh);
  }
  }
#ifdef DEBUG
  for (int i=0; i<ecoeff->Nmodes*ecoeff->Nf; i++) {
   printf("%d %lf %lf %lf %lf\n",i,creal(ecoeff->pattern_phi[i]),cimag(ecoeff->pattern_phi[i]),creal(ecoeff->pattern_theta[i]),cimag(ecoeff->pattern_theta[i]));
  }
#endif
 
  /* factorial array */
  double *factorial;
  if ((factorial=(double *)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  factorial[0]=1.0; /* 0! */
  for (int i=1; i<ecoeff->Nmodes; i++) {
    factorial[i]=factorial[i-1]*(double)i;
  }

  /* calculate preamble for basis calculation */
  int idx=0;
  for (int n=0; n<ecoeff->M; n++) {
    for (int m=-n; m<=n; m+=2) {
      int absm=m>=0?m:-m; /* |m| */
      /* sqrt { ((n-|m|)/2)! / pi ((n+|m|)/2)! } */
      ecoeff->preamble[idx]=sqrt(M_1_PI*factorial[(n-absm)/2]/factorial[(n+absm)/2]);
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
set_elementcoeffs_textfile(int element_type,  int n_stat, double frequency, const char *model_dir, elementcoeff *ecoeff) {
   /* check if directory exists */
   struct stat statbuf;
   if (!(!stat(model_dir,&statbuf) && S_ISDIR(statbuf.st_mode))) {
     fprintf(stderr,"%s: %d: Invalid directory %s\n",__FILE__,__LINE__,model_dir);
     exit(1);
   }

   /* parse directory and determine the model parameters */
   /* only handle dipole only beam types, do a check if necessary */
   DIR *dir=0;
   struct dirent *entry;
   if ((dir=opendir(model_dir)) == 0) {
     fprintf(stderr,"%s: %d: Invalid directory %s\n",__FILE__,__LINE__,model_dir);
     exit(1);
   }

   ecoeff->Ns=n_stat; /* by default, use n_stat as number of expected patterns */
   ecoeff->Nf=1;
   /* other data of ecoeff will be set while reading */

   const char *modfile="/beam.model\0";
   int first_ant=0;
   int n_ant=0;
   /* open directory contents, parse the files as well */
   while((entry = readdir(dir)) != NULL) {
     if (entry->d_type == DT_DIR) {
       if (strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")) {
         char *endptr;
         long val = strtol(entry->d_name, &endptr, 10);
         if (endptr == entry->d_name || *endptr != '\0') {
               /* Not a valid integer */
               fprintf(stderr,"Warning: Skipping non-numeric directory '%s'\n", entry->d_name);
               continue;
         }
         if (val < 0 || val >= n_stat) {
               fprintf(stderr,"Warning: Skipping station %ld (out of range 0-%d)\n", val, n_stat-1);
               continue;
         }
         int ant = (int)val;
         /* only handle a valid station id */
         /* open ./beam.model in this directory */
         char fullname[PATH_MAX];
         int n = snprintf(fullname, sizeof(fullname), "%s/%s%s",
                  model_dir, entry->d_name, modfile);
         if (n < 0 || n >= sizeof(fullname)) {
               fprintf(stderr,"%s: %d: Path too long\n",__FILE__,__LINE__);
               continue;  // Skip this station, don't crash
         }
         read_element_coeffs(fullname,ecoeff,frequency,ant,!first_ant);
         n_ant++;
         first_ant=1;
       }
     }
   }
   closedir(dir);

#ifdef HAVE_CUDA
  /* print warning (GPU mode) so shared memory is not exceeded */
  if (ecoeff->Nmodes > ELEMENT_MAX_SIZE) {
   fprintf(stderr,"%s: %d: Warning: requested element beam model order is too large for shared memory.\n Increase ELEMENT_MAX_SIZE (currently %d) and re-compile if GPU beam model prediction is done",__FILE__,__LINE__,ELEMENT_MAX_SIZE);
  }
#endif

   /* check number of antennas = number of sub directories */
   if (n_ant < n_stat) {
     fprintf(stderr,"%s: %d: Invalid number of antenna models: expected %d, got %d, copying the model of antenna 0 to all others\n",__FILE__,__LINE__,n_stat,n_ant);

     for (int ant=n_ant; ant < n_stat; ant++) {
       my_ccopy(ecoeff->Nmodes, &ecoeff->pattern_phi[0], 1, &ecoeff->pattern_phi[ant*ecoeff->Nmodes],1);
       my_ccopy(ecoeff->Nmodes, &ecoeff->pattern_theta[0], 1, &ecoeff->pattern_theta[ant*ecoeff->Nmodes],1);
     }
   }

   /* factorial array */
   double *factorial;
   if ((factorial=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   factorial[0]=1.0; /* 0! */
   for (int i=1; i<ecoeff->Nmodes; i++) {
     factorial[i]=factorial[i-1]*(double)i;
   }

   /* calculate preamble for basis calculation */
   int idx=0;
   for (int n=0; n<ecoeff->M; n++) {
     for (int m=-n; m<=n; m+=2) {
      int absm=m>=0?m:-m; /* |m| */
      /* sqrt { ((n-|m|)/2)! / pi ((n+|m|)/2)! } */
      ecoeff->preamble[idx]=sqrt(M_1_PI*factorial[(n-absm)/2]/factorial[(n+absm)/2]);
      /* (-1)^(n-|m|)/2 */
      if (((n-absm)/2)%2) {ecoeff->preamble[idx]=-ecoeff->preamble[idx];}
      /* 1/beta^{1+|m|} */
      ecoeff->preamble[idx] *=pow(ecoeff->beta,-1.0-absm);
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
eval_elementcoeffs(double r, double theta, elementcoeff *ecoeff, int station) {
  /* evaluate r^2/beta^2 */
  double rb=pow(r/ecoeff->beta,2);
  /* evaluate e^(-r^2/2beta^2) */
  double ex=exp(-0.5*rb);
 
  elementval eval;
  eval.phi=0.0+_Complex_I*0.0;
  eval.theta=0.0+_Complex_I*0.0;
  int idx=0;
  int offset=0;
  /* when separate models are available for each station */
  if (ecoeff->Ns > 1 && station > 0 && station < ecoeff->Ns) { 
    offset=ecoeff->Nmodes*station;
  }
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

     eval.phi+=ecoeff->pattern_phi[offset+idx]*(re+_Complex_I*im);
     eval.theta+=ecoeff->pattern_theta[offset+idx]*(re+_Complex_I*im);
     idx++;
    }
  }
  

 return eval;
}

elementval
eval_elementcoeffs_wb(double r, double theta, elementcoeff *ecoeff, int findex) {
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

     eval.phi+=ecoeff->pattern_phi[findex*ecoeff->Nmodes+idx]*(re+_Complex_I*im);
     eval.theta+=ecoeff->pattern_theta[findex*ecoeff->Nmodes+idx]*(re+_Complex_I*im);
     idx++;
    }
  }
  

 return eval;
}

/* Legendre function P(l,m,x) */
static double
P(int l, int m, double x) {
 double pmm, somx2, fact, pmmp1, pll;
 int i;
 pmm=1.0;
 if (m>0) {
   somx2=sqrt((1.0-x)*(1.0+x));
   fact=1.0;
   for (i=1; i<=m; i++) {
     pmm *=(-fact)*somx2;
     fact +=2.0;
   }
 }
 if (l==m) return pmm;

 pmmp1=x*(2.0*m+1.0)*pmm;
 
 if(l==m+1) return pmmp1;

 pll=0.0;
 for (i=m+2; i<=l; ++i) {
   pll=((2.0*i-1.0)*x*pmmp1-(i+m-1.0)*pmm )/(i-m);
   pmm=pmmp1;
   pmmp1=pll;
 }

 return pll;
} 

/* spherical harmonic basis functions
 * n0: max modes, starts from 1,2,...
 l=0,1,2,....,n0-1 : total n0
 m=(0),(-1,0,1),(-2,-1,0,1,2),....(-l,-l+1,...,l-1,l) : total 2*l+1
 total no of modes=(n0)^2
 * th,ph: array of theta,phi values, both of size Nt (not a grid)
 * range th: 0..pi/2, ph: 0..2*pi
 * output: n0^2 (per each mode) x Nt vector
 */
int
sharmonic_modes(int n0,double *th, double *ph, int Nt, complex double *output) {
 double *fact;
 double **pream;
 complex double **phgrid;
 double ***Lg;
 complex double ***M;
 int l,m,zci, xlen, npm;
 int nmodes=n0*n0;

 /* factorial array, store from 0! to (2*(n0-1))! at the 
 * respective position of the array - length 2*n0
 * size  2*n0
 */
 if ((fact=(double*)calloc((size_t)(2*n0),sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 fact[0]=1.0;
 for (l=1; l<=(2*n0-1); l++) {
    fact[l]=((double)l)*fact[l-1];
 }


#ifdef DEBUG
 printf("Theta= ");
 for (l=0; l<Nt; l++) {
  printf("%lf ",th[l]);
 } 
 printf("\nPhi= ");
 for (l=0; l<Nt; l++) {
  printf("%lf ",ph[l]);
 } 
 printf("\n");
#endif

#ifdef DEBUG
 printf("Fact = ");
 for(l=0; l<2*n0; l++) {
  printf("%lf ",fact[l]);
 }
 printf("\n");
#endif

 /* storage to calculate preamble, dependent only on l and |m| */
 /* size n0 x [1,3,5,...] varying */
 /* only for positive values of m */
 if ((pream=(double**)calloc((size_t)(n0),sizeof(double*)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 for (l=0; l<(n0); l++) {
  /* for each l, |m| goes from 0,1,...,l*/
  xlen=l+1;
  if ((pream[l]=(double*)calloc((size_t)(xlen),sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (m=0; m<=l; m++) {
   pream[l][m]=0.5*sqrt((2.0*l+1.0)/M_PI*fact[l-m]/fact[l+m]);
  }
 }

 free(fact);
#ifdef DEBUG
 printf("Pream=\n");
 for (l=0; l<(n0); l++) {
  for (m=0; m<=l; m++) {
   printf("%lf ",pream[l][m]);
  }
  printf("\n");
 }
#endif

 /* storage to calculate exp(j*m*phi) for all possible values */
 /* only calculate m from 0 .. to n0-1 because negative is conjugate */
 /* size Nt x (n0) */
 if ((phgrid=(complex double**)calloc((size_t)(Nt),sizeof(complex double*)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 for (l=0; l<(Nt); l++) {
  if ((phgrid[l]=(complex double*)calloc((size_t)(n0),sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (m=0; m<(n0); m++) {
   phgrid[l][m]=cos(m*ph[l])+_Complex_I*sin(m*ph[l]);
  }
 }

#ifdef DEBUG
 printf("Phigrid=\n");
 for (l=0; l<(Nt); l++) {
  for (m=0; m<(n0); m++) {
   printf("(%lf %lf) ",creal(phgrid[l][m]),cimag(phgrid[l][m]));
  }
  printf("\n");
 }
#endif
 
 /* storage to calculate Legendre polynomical P(l,m,theta)
  for given l,|m|,theta */
 /* l in [0,..,n0-1] m in [0,..,l]
  size n0 x [l+1] x  Nt */
 if ((Lg=(double***)calloc((size_t)(n0),sizeof(double**)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }

 for (l=0; l<(n0); l++) {
  /* for each l, |m| goes from 0,...,l*/
  xlen=l+1;
  if ((Lg[l]=(double**)calloc((size_t)(xlen),sizeof(double*)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (m=0; m<=l; m++) {
    if ((Lg[l][m]=(double*)calloc((size_t)(Nt),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    } 

    for (npm=0; npm<Nt; npm++) {
      Lg[l][m][npm]=P(l,m,cos(th[npm]));
    }
  }
 }
 


#ifdef DEBUG
 for (l=0; l<(n0); l++) {
  for (m=0; m<=l; m++) {
    for (npm=0; npm<Nt; npm++) {
     printf("%lf ",Lg[l][m][npm]);
    }
    printf("\n");
   }
 }
#endif

 /* now form the product of pream(l,|m|) ,  Lg(l,|m|,theta), 
 *  (m positive) and phgrid(l, m, phi) (take conjugate for m<0)
 * size: n0 x [1, 3, 5, 7, 9, ..] x Nt 
 */
 if ((M=(complex double***)calloc((size_t)(n0),sizeof(complex double**)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 for(l=0; l<n0; l++) {
   if ((M[l]=(complex double**)calloc((size_t)(2*l+1),sizeof(complex double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   for (m=0; m<=2*l; m++) {
     if ((M[l][m]=(complex double*)calloc((size_t)(Nt),sizeof(complex double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
     }
   }
 }

 for(l=0; l<n0; l++) {
  for (m=l; m<=2*l;m++) { /* index for positive value of m 0,1,2,...*/
     npm=m-l;/* true value of m, for index m */
     for (zci=0; zci<Nt; zci++) {
        M[l][m][zci]=pream[l][npm]*phgrid[zci][npm]*Lg[l][npm][zci];
     }
  }
  /* now, negative m take conjugate */
  for (m=0; m<l; m++) { /* index for negative values of m ..,-2,-1 */
     npm=2*l-m;/* true value of m, for index m */
     for (zci=0; zci<Nt; zci++) {
          M[l][m][zci]=conj(M[l][npm][zci]);
     }
  }

 }

 for(l=0; l<n0; l++) {
  free(pream[l]);
 }
 free(pream);
 for(l=0; l<Nt; l++) {
  free(phgrid[l]);
 }
 free(phgrid);
 for(l=0; l<n0; l++) {
   xlen=l+1;
   for(m=0; m<xlen; m++) {
    free(Lg[l][m]);
   }
   free(Lg[l]);
 }
 free(Lg);


 for (zci=0; zci<Nt; zci++) {
    int idx=0;
    for(l=0; l<n0; l++) {
     for (m=0; m<=2*l; m++) {
      output[nmodes*zci+idx]=M[l][m][zci];
      idx++;
     }
    }
 }


 for(l=0; l<n0; l++) {
   for (m=0; m<=2*l; m++) {
     free(M[l][m]);
   } 
  free(M[l]);
 }
 free(M);

 return 0;
}

int
read_element_coeffs(const char *coeff_file, elementcoeff *ecoeff, const double freq, const int stat, const int initialize) {

  FILE *cfp;
  int c,buff_len;
  if ((cfp=fopen(coeff_file,"r"))==0) {
    fprintf(stderr,"%s: %d: Cannot open file: %s (errno: %d)\n",__FILE__,__LINE__,coeff_file,errno);
    exit(1);
  }
  /* Check file is readable and not empty */
  if (fseek(cfp, 0, SEEK_END) != 0 || ftell(cfp) == 0) {
    fprintf(stderr,"%s: %d: File empty or unreadable: %s\n",__FILE__,__LINE__,coeff_file);
    fclose(cfp);
    exit(1);
  }
  rewind(cfp);

  char *buff;
  /* allocate memory for buffer */
  buff_len = 128;
  if((buff = (char*)malloc(sizeof(char)*(size_t)(buff_len)))==NULL) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
  }

  c=skip_lines(cfp);
  int ord, n_freq;
  double scale;
  /* model order, num freq, scale */
  c=fscanf(cfp,"%d %d %lf",&ord,&n_freq,&scale);
  if (initialize) {
     ecoeff->M=ord;
     ecoeff->Nmodes=ord*(ord+1)/2;
     ecoeff->beta=scale;
     if ((ecoeff->pattern_phi=(complex double*)calloc((size_t)ecoeff->Nmodes*ecoeff->Ns,sizeof(complex double)))==0) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
     }
     if ((ecoeff->pattern_theta=(complex double*)calloc((size_t)ecoeff->Nmodes*ecoeff->Ns,sizeof(complex double)))==0) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
     }
     if ((ecoeff->preamble=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
     }
  } else {
    /* sanity check all models have same order */
    if ((ecoeff->M != ord) || (ecoeff->beta != scale)) {
        fprintf(stderr,"%s: %d: All element beam models should have order and scale\n",__FILE__,__LINE__);
        exit(1);
    }
  }

  c=skip_lines(cfp);
  /* read all frequencies to determine which beam values to use */ 
  double *freqs;
  if ((freqs=(double*)calloc((size_t)n_freq,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
 
  double myfreq=freq/1e9; // GHz
  int nf=0;
  while (c!=1 && nf < n_freq) {
    /* we have a new value */
    memset(buff,0,buff_len);
    c=read_next_string(&buff,&buff_len,cfp);
    sscanf(buff,"%lf",&freqs[nf]);
    nf++;
  }
  if (c==1 && nf < n_freq) {
        fprintf(stderr,"%s: %d: Expected %d frequencies, but got %d\n",__FILE__,__LINE__,n_freq,nf);
        exit(1);
  }

  /* find which freq indices to read */
  int idl,idh=0;
  while(idh < n_freq && myfreq > freqs[idh]) { idh++; }
  if (idh == n_freq) {
    idl=idh=n_freq-1;
  } else if (idh==0) {
    idl=idh;
  } else {
    idl=idh-1;
  }

  double *phi_low_re,*phi_low_im,*phi_high_re,*phi_high_im;
  double *theta_low_re,*theta_low_im,*theta_high_re,*theta_high_im;
  if ((phi_low_re=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((phi_low_im=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((phi_high_re=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((phi_high_im=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((theta_low_re=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((theta_low_im=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((theta_high_re=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((theta_high_im=(double*)calloc((size_t)ecoeff->Nmodes,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  
  /* read file until correct lower freq is read */
  for (int li=0; li <= idl; li++) {
     c=skip_lines(cfp);
     /* read ecoff->Nmodes lines */
     for (int nmode=0; nmode <ecoeff->Nmodes; nmode++) {
       c=fscanf(cfp,"%lf %lf %lf %lf",&theta_low_re[nmode],&theta_low_im[nmode],&phi_low_re[nmode],&phi_low_im[nmode]);
       if (c != 4) {
         fprintf(stderr,"%s: %d: parse error reading coefficients\n",__FILE__,__LINE__);
         exit(1);
       }
     }
  }
  /* now read the higer freq */
  if (idh != idl) {
     c=skip_lines(cfp);
     for (int nmode=0; nmode <ecoeff->Nmodes; nmode++) {
       c=fscanf(cfp,"%lf %lf %lf %lf",&theta_high_re[nmode],&theta_high_im[nmode],&phi_high_re[nmode],&phi_high_im[nmode]);
       if (c != 4) {
         fprintf(stderr,"%s: %d: parse error reading coefficients\n",__FILE__,__LINE__);
         exit(1);
       }
     }
  } else {
    /* copy lower freq value to higher freq value */
    memcpy(theta_high_re,theta_low_re,sizeof(double)*ecoeff->Nmodes);
    memcpy(theta_high_im,theta_low_im,sizeof(double)*ecoeff->Nmodes);
    memcpy(phi_high_re,phi_low_re,sizeof(double)*ecoeff->Nmodes);
    memcpy(phi_high_im,phi_low_im,sizeof(double)*ecoeff->Nmodes);
  }

  /* interpolate */
  double wl=myfreq-freqs[idl];
  double wh=freqs[idh]-myfreq;
  double w1=wl/(wl+wh);

  for (int nmode=0; nmode< ecoeff->Nmodes; nmode++) {
    ecoeff->pattern_theta[stat*ecoeff->Nmodes+nmode]=theta_low_re[nmode]*w1+theta_high_re[nmode]*(1-w1)+_Complex_I*(theta_low_im[nmode]*w1+theta_high_im[nmode]*(1-w1));
    ecoeff->pattern_phi[stat*ecoeff->Nmodes+nmode]=phi_low_re[nmode]*w1+phi_high_re[nmode]*(1-w1)+_Complex_I*(phi_low_im[nmode]*w1+phi_high_im[nmode]*(1-w1));
  }
 
  free(theta_low_re);
  free(theta_low_im);
  free(theta_high_re);
  free(theta_high_im);
  free(phi_low_re);
  free(phi_low_im);
  free(phi_high_re);
  free(phi_high_im);
  free(freqs);
  free(buff);
  fclose(cfp);
  return 0;
}
