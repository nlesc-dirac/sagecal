/*
 *
 Copyright (C) 2017 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <string.h>
#include "Dirac.h"


//#define DEBUG
/*
  change polynomial order from Kstart to Kfinish
  evaluate Z for each poly order, then find MDL
   N: stations
   M: clusters
   F: frequencies
   J: weightxrhoxJ solutions (note: not true J, but J scaled by each slaves' rho), 8NMxF blocks
   rho: regularization, no weighting applied, Mx1 
   freqs: frequencies, Fx1
   freq0: reference freq
   weight: weight for each freq, based on flagged data, Fx1
   polytype: type of polynomial
  Kstart, Kfinish: range of order of polynomials to calculate the MDL
   Nt: no. of threads
*/
int
minimum_description_length(int N, int M, int F, double *J, double *rho, double *freqs, double freq0, double *weight, int polytype, int Kstart, int Kfinish, int Nt) {


 double *Z,*z,*B,*Bi;
 int Npoly;
 int ci,cm,cf,p;

 /* array to store DL for all poly setups */
 double *mdl,*aic;
 int idx;
 if ((mdl=(double*)calloc((size_t)(Kfinish-Kstart+1),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((aic=(double*)calloc((size_t)(Kfinish-Kstart+1),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }

#ifdef DEBUG
    FILE *dfp;
    int m,q;
    if ((dfp=fopen("debug.m","w+"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      exit(1);
    }

         for(cm=0; cm<F; cm++) {
          for(m=0; m<M; m++) {
           fprintf(dfp,"%%%%%%%%%%%% (rho J) slave=%d dir=%d\n",cm,m);
           fprintf(dfp,"J_%d_%d=[\n",cm,m);
           for (p=0; p<N; p++) {
            int off=cm*N*8*M+m*8*N+p*8;
            fprintf(dfp,"%e+j*(%e), %e+j*(%e);\n%e+j*(%e), %e+j*(%e);\n",J[off],J[off+1],J[off+2],J[off+3],J[off+4],J[off+5],J[off+6],J[off+7]);
           }
           fprintf(dfp,"];\n");
          }
         }
#endif
 /* loop over different polynomial orders */
 /* for each poly setup, 
     - estimate Z
     - find sum_F ||rho J - rho B Z||^2  (primal error)
     - find MDL
 */
 idx=0;
 for (Npoly=Kstart; Npoly<=Kfinish; Npoly++) {
      /* Z: 2Nx2 x Npoly x M */
    /* keep ordered by M (one direction together) */
    if ((Z=(double*)calloc((size_t)N*8*Npoly*M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* z : 2Nx2 x M x Npoly vector, so each block is 8NM */
    if ((z=(double*)calloc((size_t)N*8*Npoly*M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* Npoly terms, for each frequency, so Npoly x F */
    if ((B=(double*)calloc((size_t)Npoly*F,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* pseudoinverse */
    if ((Bi=(double*)calloc((size_t)Npoly*Npoly,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* for constant poly, only use type 1 */
    setup_polynomials(B, Npoly, F, freqs, freq0, (Npoly==1?1:polytype));
    /* find sum fratio[i] * B(:,i)B(:,i)^T, and its pseudoinverse */
    find_prod_inverse(B,Bi,Npoly,F,weight);
#ifdef DEBUG
    fprintf(dfp,"B=[\n");
    for (p=0; p<Npoly; p++) {
     for (q=0; q<F; q++) {
       fprintf(dfp,"%lf ",B[p*Npoly+q]);
     }
     fprintf(dfp,";\n");
    }
    fprintf(dfp,"];\n");
    fprintf(dfp,"Bi=[\n");
    for (p=0; p<Npoly; p++) {
     for (q=0; q<Npoly; q++) {
       fprintf(dfp,"%lf ",Bi[p*Npoly+q]);
     }
     fprintf(dfp,";\n");
    }
    fprintf(dfp,"];\n");
#endif


    /* find Z */
    /* add to 8NM vector, multiplied by Npoly different scalars, F times */
    for (ci=0; ci<Npoly; ci++) {
           my_dcopy(8*N*M,J,1,&z[ci*8*N*M],1); /* starting with J[0], 8NM values */
           my_dscal(8*N*M,B[ci],&z[ci*8*N*M]);
    }
    for (cm=1; cm<F; cm++) {
           for (ci=0; ci<Npoly; ci++) {
            /* Note: no weighting of J is needed, because slave has already weighted their rho (we have rho J here) */
            my_daxpy(8*N*M, &J[cm*8*N*M], B[cm*Npoly+ci], &z[ci*8*N*M]);
           }
    }
    /* also scale by 1/rho, only if rho>0, otherwise set it to 0.0*/
    for (cm=0; cm<M; cm++) {
          double invscale=0.0;
          if (rho[cm]>0.0) {
           invscale=1.0/rho[cm];
          }
          for (ci=0; ci<Npoly; ci++) {
            my_dscal(8*N,invscale,&z[8*N*M*ci+8*N*cm]);
          }
    }

    /* find Z */
    update_global_z(Z,N,M,Npoly,z,Bi);

#ifdef DEBUG
          for(m=0; m<M; m++) {
           for (ci=0;ci<Npoly; ci++) {
            fprintf(dfp,"%%%%%%%%%%%% Z dir=%d poly=%d\n",m,ci);
            fprintf(dfp,"Z_%d_%d=[\n",m,ci);
            for (p=0; p<N; p++) {
             int off=m*8*N*Npoly+ci*8*N+p*8;
             fprintf(dfp,"%lf+j*(%lf), %lf+j*(%lf);\n%lf+j*(%lf), %lf+j*(%lf);\n",Z[off],Z[off+1],Z[off+2],Z[off+3],Z[off+4],Z[off+5],Z[off+6],Z[off+7]);
            }
            fprintf(dfp,"];\n");
           }
          }
#endif


    double sumnrm=0.0;
    for (cf=0; cf<F; cf++) {
      for (p=0; p<M; p++) {
         memset(&z[8*N*p],0,sizeof(double)*(size_t)N*8);
         for (ci=0; ci<Npoly; ci++) {
             my_daxpy(8*N, &Z[p*8*N*Npoly+ci*8*N], B[cf*Npoly+ci], &z[8*N*p]);
         }
      }
      /* now z has B Z for the freq 'cm' , but J is weight x rho x J*/
      /* so need to find J - weight x rho x (B Z), multiply z with weightxrho */
      for (cm=0; cm<M; cm++) {
          for (ci=0; ci<Npoly; ci++) {
            my_dscal(8*N,rho[cm]*weight[cf],&z[8*N*M*ci+8*N*cm]);
          }
      }
#ifdef DEBUG
          for(m=0; m<M; m++) {
           fprintf(dfp,"%%%%%%%%%%%% consensus for slave=%d dir=%d\n",cf,m);
           fprintf(dfp,"BZ_%d_%d=[\n",cf,m);
           for (p=0; p<N; p++) {
            int off=m*8*N+p*8;
            fprintf(dfp,"%lf+j*(%lf), %lf+j*(%lf);\n%lf+j*(%lf), %lf+j*(%lf);\n",z[off],z[off+1],z[off+2],z[off+3],z[off+4],z[off+5],z[off+6],z[off+7]);
           }
           fprintf(dfp,"];\n");
          }
#endif
      /* find residual */
      my_daxpy(8*N*M, &J[cf*8*N*M], -1.0, z);
      /* rescale back residual */
      for (cm=0; cm<M; cm++) {
          double invscale=rho[cm]*weight[cf];
          invscale=(invscale>0.0?1.0/invscale:0.0); 
          for (ci=0; ci<Npoly; ci++) {
            my_dscal(8*N,invscale,&z[8*N*M*ci+8*N*cm]);
          }
      }

#ifdef DEBUG
          for(m=0; m<M; m++) {
           fprintf(dfp,"%%%%%%%%%%%% residual for slave=%d dir=%d\n",cf,m);
           fprintf(dfp,"E_%d_%d=[\n",cf,m);
           for (p=0; p<N; p++) {
            int off=m*8*N+p*8;
            fprintf(dfp,"%lf+j*(%lf), %lf+j*(%lf);\n%lf+j*(%lf), %lf+j*(%lf);\n",z[off],z[off+1],z[off+2],z[off+3],z[off+4],z[off+5],z[off+6],z[off+7]);
           }
           fprintf(dfp,"];\n");
          }
#endif
      double mnrm=my_dnrm2(8*N*M, z);
      sumnrm+=mnrm*mnrm;
    }
    /* RSS: residual sum of squares: per data point */
    double RSS=sumnrm/(double)(8*N*M);
    //printf("%d AIC=%lf+%lf MDL=%lf+%lf\n",idx,((double)F)*log(RSS/(double)(F)),(double)(2*Npoly),(0.5*(double)F)*log(RSS/(double)(F)),0.5*(double)(Npoly)*log((double)F));

    /* Data points: F, polynomial degree Npoly */
    /* AIC = F log (RSS/F) + 2 Npoly*/
    aic[idx]=((double)F)*log(RSS/(double)(F))+(double)(2*Npoly);
    /* MDL =  F/2 log(RSS/F) + Npoly/2 log(F) */
    mdl[idx++]=(0.5*(double)F)*log(RSS/(double)(F))+0.5*(double)(Npoly)*log((double)F);

#ifdef DEBUG
   fclose(dfp);
#endif

   free(Z);
   free(z);
   free(B);
   free(Bi);
 }

  /* find MDL from possible values */
  int minidx=0,minaicidx=0;
  double minval=mdl[minidx];
  double minaic=aic[minaicidx];
  idx=0;
  for (Npoly=Kstart; Npoly<=Kfinish; Npoly++) {
    if (mdl[idx]<minval) {
      minval=mdl[idx];
      minidx=idx;
    }
    if (aic[idx]<minaic) {
      minaic=aic[idx];
      minaicidx=idx;
    }

    idx++;
  }
  

 printf("Finding best fitting polynomials: MDL %lf for polynomial terms=%d,",minval,Kstart+minidx);
 printf(" AIC %lf for polynomial terms=%d\n",minaic,Kstart+minaicidx);
 free(mdl);
 free(aic);
 return 0;
}
