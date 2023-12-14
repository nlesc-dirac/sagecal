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


/* cid: diffuse_cluster: ordinal cluster id, 0...M, not the cluster id in the cluster file 
 *
 * Z: 2N x 2G shapelet models, for the given freq0
 */
int
recalculate_diffuse_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int cid, int sh_n0, double sh_beta, complex double *Z, int Nt) {


      int G=sh_n0*sh_n0;
      /* Transpose storage of Z: 2Nx2G to 4GxN, so each column has modes of one station */
      complex double *Zt=0;
      if ((Zt=(complex double*)calloc((size_t)(2*N*2*G),sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
      }
      /* copy rows of Z as columns of Zt */
      for (int ci=0; ci<N; ci++) {
        my_ccopy(2*G,&Z[ci*2],2*N,&Zt[ci*4*G],2);
        my_ccopy(2*G,&Z[ci*2+1],2*N,&Zt[ci*4*G+1],2);
      }

      FILE *dfp;
      if ((dfp=fopen("debug.m","w+"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      exit(1);
      }
      fclose(dfp);

      /* storage for tensor */
      double *Cf;


      if (cid<0 || cid>=M) {
        fprintf(stderr,"%s: %d: invalid cluster id\n",__FILE__,__LINE__);
        exit(1);
      }
      /* find info about the given cluster */
      printf("cluster %d has %d sources\n",cid,carr[cid].N);
      for (int ci=0; ci<carr[cid].N; ci++) {
        printf("source %d type %d\n",ci,carr[cid].stype[ci]);
        if (carr[cid].stype[ci]!=STYPE_SHAPELET) {
           fprintf(stderr,"%s: %d: invalid source type, must be shapelet\n",__FILE__,__LINE__);
           exit(1);
        }

        /* get shapelet info */
        exinfo_shapelet *sp=(exinfo_shapelet*) carr[cid].ex[ci];

        printf("shapelet n0=%d beta=%lf\n",sp->n0,sp->beta);
        /* create tensor : product out (sp->n0,sp->beta),
         * product in (sp->n0,sp->beta) (sh_n0,sh_beta) */
        if ((Cf=(double*)calloc((size_t)(sp->n0*sp->n0*sh_n0),sizeof(double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        shapelet_product_tensor(sp->n0,sp->n0,sh_n0,sp->beta,sp->beta,sh_beta,Cf);

        int station=1;
        /* allocate memory to store the product, n0*n0*2x2  for each station */
        complex double *Zout=0;
        if ((Zout=(complex double*)calloc((size_t)(2*N*2*sp->n0*sp->n0),sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }

        /* Shapelet model: sp->modes array of scalars, find coherencies combining Stokes IQUV */
        complex double *s_coh=0;
        if ((s_coh=(complex double*)calloc((size_t)(2*2*sp->n0*sp->n0),sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        complex double XXYY[4]={
           carr[cid].sI[ci]+carr[cid].sQ[ci],
           carr[cid].sU[ci]+_Complex_I*carr[cid].sV[ci],
           carr[cid].sU[ci]-_Complex_I*carr[cid].sV[ci],
           carr[cid].sI[ci]-carr[cid].sQ[ci]
        };

        for (int nmode=0; nmode<sp->n0*sp->n0; nmode++) {
          s_coh[4*nmode]=XXYY[0]*sp->modes[nmode]; /* XX */
          s_coh[4*nmode+1]=XXYY[1]*sp->modes[nmode]; /* XY */
          s_coh[4*nmode+2]=XXYY[2]*sp->modes[nmode]; /* YX */
          s_coh[4*nmode+3]=XXYY[3]*sp->modes[nmode]; /* YY */
        }


        shapelet_product_jones(sp->n0,sp->n0,sh_n0,sp->beta,sp->beta,sh_beta,&Zout[4*sp->n0*sp->n0*station],s_coh,&Zt[4*G*station],Cf);


        free(Cf);
        free(Zout);
        free(s_coh);
      }

      /* for each source, form N x N shapelet products, S_p x S_k x S_q^H where
       * S_p:spatial model for station p,
       * S_k:shapelet model for source k,
       * S_q^H:spatial model for station q, Hermitian
       * Since S_k is not always a scalar value, it can be like 
       * [I 0; 0 I] but [Q 0; 0 -Q] or [0 U; U 0] etc.,
       * We have to honor the order S_p x x S_k x S_q^H
       */


      free(Zt);
  return 0;
}

