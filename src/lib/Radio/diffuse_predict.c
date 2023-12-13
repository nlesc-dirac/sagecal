/*
 *
 Copyright (C) 2023 Sarod Yatawatta <sarod@users.sf.net>  
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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Dirac.h"
#include "Dirac_radio.h"


int
recalculate_diffuse_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int diffuse_cluster, int sh_n0, double sh_beta, complex double *Z, int Nt) {


        /* create tensor */
      double *Cf;
      int sL=4,sM=3,sN=2;
      double s_alpha=0.8, s_beta=0.9, s_gamma=1.1;
      if ((Cf=(double*)calloc((size_t)(sL*sM*sN),sizeof(double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      double *s_h,*s_f,*s_g;
      if ((s_h=(double*)calloc((size_t)(sL*sL),sizeof(double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      if ((s_f=(double*)calloc((size_t)(sM*sM),sizeof(double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      if ((s_g=(double*)calloc((size_t)(sN*sN),sizeof(double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      s_f[0]=-0.2; s_f[1]=0.11; s_f[2]=0.5; s_f[3]=1.1; s_f[4]=-1; s_f[5]=0.3;
      s_f[6]=0.01; s_f[7]=-0.2; s_f[8]=0.2;
      s_g[0]=0.3; s_g[1]=0.1; s_g[2]=-0.5; s_g[3]=-0.4;

      shapelet_product_tensor(sL,sM,sN,s_alpha,s_beta,s_gamma,Cf);
      shapelet_product(sL,sM,sN,s_alpha,s_beta,s_gamma,s_h,s_f,s_g,Cf);
      free(Cf);
      free(s_h);
      free(s_f);
      free(s_g);


  return 0;
}

