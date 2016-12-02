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

#include "sagecal.h"
#include <math.h>
#include <stdio.h>

//#define DEBUG
/* build matrix with polynomial terms
  B : Npoly x Nf, each row is one basis function
  Npoly : total basis functions
  Nf: frequencies
  freqs: Nfx1 array freqs
  freq0: reference freq
  type : 
  0 :[1 ((f-fo)/fo) ((f-fo)/fo)^2 ...] basis functions
  1 : normalize each row such that norm is 1
  2 : Bernstein poly \sum N_C_r x^r (1-x)^r where x in [0,1] : use min,max values of freq to normalize
     Note: freqs might not be in sorted order, so need to search array to find min,max values
  3: [1 ((f-fo)/fo) (fo/f-1) ((f-fo)/fo)^2 (fo/f-1)^2 ... ] basis, for this case odd Npoly  preferred
*/
int
setup_polynomials(double *B, int Npoly, int Nf, double *freqs, double freq0, int type) {

  if (type==0 || type==1) {
  double frat,dsum;
  double invf=1.0/freq0;
  int ci,cm;
  for (ci=0; ci<Nf; ci++) {
     B[ci*Npoly]=1.0;
     frat=(freqs[ci]-freq0)*invf;
     for (cm=1; cm<Npoly; cm++) {
      B[ci*Npoly+cm]=B[ci*Npoly+cm-1]*frat;
     }
  }
#ifdef DEBUG
  int cj;
  printf("BT=[\n");
  for(cj=0; cj<Npoly; cj++) {
   for (ci=0; ci<Nf; ci++) {
    printf("%lf ",B[ci*Npoly+cj]); 
   }
   printf("\n");
  }
  printf("];\n");
#endif
  if (type==1) {
   /* normalize each row such that norm is 1 */
   for (cm=0; cm<Npoly; cm++) {
     dsum=0.0;
     for (ci=0; ci<Nf; ci++) {
      dsum+=B[ci*Npoly+cm]*B[ci*Npoly+cm];
     }
     if (dsum>0.0) {
      invf=1.0/sqrt(dsum);
     } else {
      invf=0.0;
     }
     for (ci=0; ci<Nf; ci++) {
      B[ci*Npoly+cm] *=invf;
     }
   }
  }
  } else if (type==2) {
   /* Bernstein polynomials */
   int idmax=my_idamax(Nf, freqs, 1);
   int idmin=my_idamin(Nf, freqs, 1);
   double fmax=freqs[idmax-1];
   double fmin=freqs[idmin-1];
   double *fact; /* factorial array */
   double *px,*p1x; /* arrays for powers of x and (1+x) */
   if ((fact=(double*)calloc((size_t)Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
   }
   if ((px=(double*)calloc((size_t)Npoly*Nf,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
   }
   if ((p1x=(double*)calloc((size_t)Npoly*Nf,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
   }


   fact[0]=1.0;
   int ci,cj;
   for (ci=1; ci<Npoly; ci++) {
     fact[ci]=fact[ci-1]*(double)ci;
   }
   double invf=1.0/(fmax-fmin);
   double frat;
   for (ci=0; ci<Nf; ci++) {
     /* normalize coordinates */
     frat=(freqs[ci]-fmin)*invf;
     px[ci]=1.0;
     p1x[ci]=1.0;
     px[ci+Nf]=frat;
     p1x[ci+Nf]=1.0-frat;
   }
   for (cj=2; cj<Npoly; cj++) {
    for (ci=0; ci<Nf; ci++) {
     px[cj*Nf+ci]=px[(cj-1)*Nf+ci]*px[Nf+ci]; 
     p1x[cj*Nf+ci]=p1x[(cj-1)*Nf+ci]*p1x[Nf+ci]; 
    }
   }
   for (cj=0; cj<Npoly; cj++) { /* ci: freq, cj: poly order */
     frat=fact[Npoly-1]/(fact[Npoly-cj-1]*fact[cj]);
     for (ci=0; ci<Nf; ci++) {
      B[ci*Npoly+cj]=frat*px[cj*Nf+ci]*p1x[(Npoly-cj-1)*Nf+ci];
     }
   }

#ifdef DEBUG
   printf("BT=[\n");
   for(cj=0; cj<Npoly; cj++) {
    for (ci=0; ci<Nf; ci++) {
    printf("%lf ",B[ci*Npoly+cj]); 
   }
   printf("\n");
   }
   printf("];\n");
#endif
   free(fact);
   free(px);
   free(p1x);
  } else if (type==3) { /* [1 (f-fo)/fo (fo/f-1) ... */
   double frat;
   double invf=1.0/freq0;
   int ci,cm;
   for (ci=0; ci<Nf; ci++) {
     B[ci*Npoly]=1.0;
     frat=(freqs[ci]-freq0)*invf;
     double lastval=frat;
     for (cm=1; cm<Npoly; cm+=2) { /* odd values 1,3,5,... */
      B[ci*Npoly+cm]=lastval;
      lastval*=frat;
     }
     frat=(freq0/freqs[ci]-1.0);
     lastval=frat;
     for (cm=2; cm<Npoly; cm+=2) { /* even values 2,4,6,... */
      B[ci*Npoly+cm]=lastval;
      lastval*=frat;
     }
   }
#ifdef DEBUG
  int cj;
  printf("BT=[\n");
  for(cj=0; cj<Npoly; cj++) {
   for (ci=0; ci<Nf; ci++) {
    printf("%lf ",B[ci*Npoly+cj]); 
   }
   printf("\n");
  }
  printf("];\n");
#endif

  } else {
    fprintf(stderr,"%s : %d: undefined polynomial type\n",__FILE__,__LINE__);
  }
  return 0;
}



/* build matrix with polynomial terms
  B : Npoly x Nf, each row is one basis function
  Bi: Npoly x Npoly pseudo inverse of sum( B(:,col) x B(:,col)' )
  Npoly : total basis functions
  Nf: frequencies
  fratio: Nfx1 array of weighing factors depending on the flagged data of each freq
  Sum taken is a weighted sum, using weights in fratio
*/
int
find_prod_inverse(double *B, double *Bi, int Npoly, int Nf, double *fratio) {

  int ci,status,lwork=0;
  double w[1],*WORK,*U,*S,*VT;
  /* set Bi to zero */
  memset(Bi,0,sizeof(double)*Npoly*Npoly);
  /* find sum */
  for (ci=0; ci<Nf; ci++) { 
   /* outer product */
   my_dgemm('N','T',Npoly,Npoly,1,fratio[ci],&B[ci*Npoly],Npoly,&B[ci*Npoly],Npoly,1.0,Bi,Npoly);
  }
#ifdef DEBUG
  int cj;
  printf("BT=[\n");
  for (ci=0; ci<Nf; ci++) {
   for(cj=0; cj<Npoly; cj++) {
    printf("%lf ",B[ci*Npoly+cj]); 
   }
   printf("\n");
  }
  printf("];\nBi=[\n");
  for (ci=0; ci<Npoly; ci++) {
   for(cj=0; cj<Npoly; cj++) {
    printf("%lf ",Bi[ci*Npoly+cj]); 
   }
   printf("\n");
  }
  printf("];\n");
#endif

  if ((U=(double*)calloc((size_t)Npoly*Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((VT=(double*)calloc((size_t)Npoly*Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((S=(double*)calloc((size_t)Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* memory for SVD */
  status=my_dgesvd('A','A',Npoly,Npoly,Bi,Npoly,S,U,Npoly,VT,Npoly,w,-1);
  if (!status) {
    lwork=(int)w[0];
  } else {
    printf("%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
    exit(1);
  }
  if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  status=my_dgesvd('A','A',Npoly,Npoly,Bi,Npoly,S,U,Npoly,VT,Npoly,WORK,lwork);
  if (status) {
    printf("%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
    exit(1);
  }

  /* find 1/singular values, and multiply columns of U with new singular values */
  for (ci=0; ci<Npoly; ci++) {
   if (S[ci]>CLM_EPSILON) {
    S[ci]=1.0/S[ci];
   } else {
    S[ci]=0.0;
   }
   my_dscal(Npoly,S[ci],&U[ci*Npoly]);
  }

  /* find product U 1/S V^T */
  my_dgemm('N','N',Npoly,Npoly,Npoly,1.0,U,Npoly,VT,Npoly,0.0,Bi,Npoly);

#ifdef DEBUG
  printf("Bii=[\n");
  for (ci=0; ci<Npoly; ci++) {
   for(cj=0; cj<Npoly; cj++) {
    printf("%lf ",Bi[ci*Npoly+cj]); 
   }
   printf("\n");
  }
  printf("];\n");
#endif

  free(U);
  free(S);
  free(VT);
  free(WORK);
  return 0;
}



/* update Z
   Z: 8N Npoly x M double array (real and complex need to be updated separate)
   N : stations
   M : clusters
   Npoly: no of basis functions
   z : right hand side 8NM Npoly x 1 (note the different ordering from Z)
   Bi : NpolyxNpoly matrix, Bi^T=Bi assumed
*/
int 
update_global_z(double *Z,int N,int M,int Npoly,double *z,double *Bi) { 
 /* one block of Z for one direction 2Nx2xNpoly (complex)
    and 8NxNpoly  real values : select one column : 2NxNpoly (complex)
    select real,imag : 2NxNpoly each (vector)
    reshape each to 2NxNpoly matrix => Q
    Bi : NpolyxNpoly matrix = B^T
    
    for each direction (M values)
    select 2N,2N,... : 2Nx Npoly complex values from z (ordered by M)
    select real,imag: size 2NxNpoly, 2NxNpoly vectors
    reshape to 2NxNpoly => R
    reshape to 2NxNpoly => I (imag)
    
    then Q=([R I] Bi^T) for each column
    Q=[R_1^T I_1^T R_2^T I_2^T]^T Bi^T for 2 columns
    R_1,I_1,R_2,I_2 : size 2NxNpoly 
    R : (2N 4) x Npoly
    so find Q
 */
 double *R,*Q;
 if ((R=(double*)calloc((size_t)2*N*Npoly*4,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 if ((Q=(double*)calloc((size_t)2*N*Npoly*4,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 
 int ci,np;
 for (ci=0; ci<M; ci++) {
  for (np=0; np<Npoly; np++) {
   /* select 2N */
   my_dcopy(2*N, &z[8*N*ci+np*8*N*M], 4, &R[np*8*N], 1); /* R_1 */
   my_dcopy(2*N, &z[8*N*ci+np*8*N*M+1], 4, &R[np*8*N+2*N], 1); /* I_1 */
   my_dcopy(2*N, &z[8*N*ci+np*8*N*M+2], 4, &R[np*8*N+2*2*N], 1); /* R_2 */
   my_dcopy(2*N, &z[8*N*ci+np*8*N*M+3], 4, &R[np*8*N+3*2*N], 1); /* I_2 */
  }
  /* find Q=R B^T */
  memset(Q,0,sizeof(double)*2*N*Npoly*4);
  my_dgemm('N','N',8*N,Npoly,Npoly,1.0,R,8*N,Bi,Npoly,1.0,Q,8*N);
  /* copy back to Z */ 
  for (np=0; np<Npoly; np++) {
   my_dcopy(2*N, &Q[np*8*N], 1, &Z[8*N*Npoly*ci+8*N*np], 4); 
   my_dcopy(2*N, &Q[np*8*N+2*N], 1, &Z[8*N*Npoly*ci+8*N*np+1], 4); 
   my_dcopy(2*N, &Q[np*8*N+2*2*N], 1, &Z[8*N*Npoly*ci+8*N*np+2], 4); 
   my_dcopy(2*N, &Q[np*8*N+3*2*N], 1, &Z[8*N*Npoly*ci+8*N*np+3], 4); 
  }
   
 }

 free(R);
 free(Q);
 return 0;
}
