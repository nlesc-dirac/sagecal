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

#include "Dirac.h"
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
   /* sanity check */
   if (idmax<1) idmax=1;
   if (idmax>Nf) idmax=Nf;
   if (idmin<1) idmin=1;
   if (idmin>Nf) idmin=Nf;
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

  int ci,status,lwork=1;
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



typedef struct thread_data_prod_inv_ {
 int startM;
 int endM;
 int M;
 int Nf;
 int Npoly;
 double *B;
 double *Bi;
 double *rho;
 double *alpha; /* Mx1 array, only used with fed. averaging and spatial regularization */
} thread_data_prod_inv_t;


/* worker thread function for calculating sum and inverse */ 
static void*
sum_inv_threadfn(void *data) {
 thread_data_prod_inv_t *t=(thread_data_prod_inv_t*)data;
 double w[1],*WORK,*U,*S,*VT;

 int k,ci,status,lwork=1;
 int Np2=t->Npoly*t->Npoly;
 /* allocate memory for the SVD here */
  if ((U=(double*)calloc((size_t)Np2,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((VT=(double*)calloc((size_t)Np2,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((S=(double*)calloc((size_t)t->Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* memory for SVD: use first location of Bi */
  status=my_dgesvd('A','A',t->Npoly,t->Npoly,&(t->Bi[t->startM*Np2]),t->Npoly,S,U,t->Npoly,VT,t->Npoly,w,-1);
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



 /* iterate over clusters */
 for (k=t->startM; k<=t->endM; k++) {
   memset(&(t->Bi[k*Np2]),0,sizeof(double)*Np2);
   /* find sum */
   for (ci=0; ci<t->Nf; ci++) {
    /* outer product */
    my_dgemm('N','T',t->Npoly,t->Npoly,1,t->rho[k+ci*t->M],&t->B[ci*t->Npoly],t->Npoly,&t->B[ci*t->Npoly],t->Npoly,1.0,&(t->Bi[k*Np2]),t->Npoly);
   }
   /* find SVD */
   status=my_dgesvd('A','A',t->Npoly,t->Npoly,&(t->Bi[k*Np2]),t->Npoly,S,U,t->Npoly,VT,t->Npoly,WORK,lwork);
   if (status) {
    printf("%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
    exit(1);
   }

   /* find 1/singular values, and multiply columns of U with new singular values */
   for (ci=0; ci<t->Npoly; ci++) {
    if (S[ci]>CLM_EPSILON) {
     S[ci]=1.0/S[ci]; 
    } else {
     S[ci]=0.0;
    }
    my_dscal(t->Npoly,S[ci],&U[ci*t->Npoly]);
   }

   /* find product U 1/S V^T */
   my_dgemm('N','N',t->Npoly,t->Npoly,t->Npoly,1.0,U,t->Npoly,VT,t->Npoly,0.0,&(t->Bi[k*Np2]),t->Npoly);

 }

 free(U);
 free(VT);
 free(S);
 free(WORK);
 return NULL;
}

/* worker thread function for calculating sum and inverse
  using also fed. averaging info */ 
static void*
sum_inv_fed_threadfn(void *data) {
 thread_data_prod_inv_t *t=(thread_data_prod_inv_t*)data;
 double w[1],*WORK,*U,*S,*VT;

 int k,ci,status,lwork=1;
 int Np2=t->Npoly*t->Npoly;
 /* allocate memory for the SVD here */
  if ((U=(double*)calloc((size_t)Np2,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((VT=(double*)calloc((size_t)Np2,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((S=(double*)calloc((size_t)t->Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* memory for SVD: use first location of Bi */
  status=my_dgesvd('A','A',t->Npoly,t->Npoly,&(t->Bi[t->startM*Np2]),t->Npoly,S,U,t->Npoly,VT,t->Npoly,w,-1);
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



 /* iterate over clusters */
 for (k=t->startM; k<=t->endM; k++) {
   memset(&(t->Bi[k*Np2]),0,sizeof(double)*Np2);
   /* find sum */
   for (ci=0; ci<t->Nf; ci++) {
    /* outer product */
    my_dgemm('N','T',t->Npoly,t->Npoly,1,t->rho[k+ci*t->M],&t->B[ci*t->Npoly],t->Npoly,&t->B[ci*t->Npoly],t->Npoly,1.0,&(t->Bi[k*Np2]),t->Npoly);
   }
   /* find SVD */
   status=my_dgesvd('A','A',t->Npoly,t->Npoly,&(t->Bi[k*Np2]),t->Npoly,S,U,t->Npoly,VT,t->Npoly,WORK,lwork);
   if (status) {
    printf("%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
    exit(1);
   }

   /* find 1/singular values, and multiply columns of U with new singular values */
   for (ci=0; ci<t->Npoly; ci++) {
    if (S[ci]>CLM_EPSILON) {
     S[ci]=1.0/(S[ci]+t->alpha[k]); /* 1.0/(S[]+alpha) for inverting (B^T B + alpha I) */
    } else {
     S[ci]=1.0/(t->alpha[k]);
    }
    my_dscal(t->Npoly,S[ci],&U[ci*t->Npoly]);
   }

   /* find product U 1/S V^T */
   my_dgemm('N','N',t->Npoly,t->Npoly,t->Npoly,1.0,U,t->Npoly,VT,t->Npoly,0.0,&(t->Bi[k*Np2]),t->Npoly);

 }

 free(U);
 free(VT);
 free(S);
 free(WORK);
 return NULL;
}

/* build matrix with polynomial terms
  B : Npoly x Nf, each row is one basis function
  Bi: Npoly x Npoly pseudo inverse of sum( B(:,col) x B(:,col)' ) : M times
  Npoly : total basis functions
  Nf: frequencies
  M: clusters
  rho: NfxM array of regularization factors (for each freq, M values)
  Sum taken is a weighted sum, using weights in rho, rho is assumed to change for each freq,cluster pair 

  Nt: no. of threads
*/
int
find_prod_inverse_full(double *B, double *Bi, int Npoly, int Nf, int M, double *rho, int Nt) {

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_prod_inv_t *threaddata;

  int ci,Nthb0,Nthb,nth,nth1;
  /* clusters per thread */
  Nthb0=(M+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_prod_inv_t*)malloc((size_t)Nt*sizeof(thread_data_prod_inv_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  ci=0;
  for (nth=0;  nth<Nt && ci<M; nth++) {
    if (ci+Nthb0<M) { 
     Nthb=Nthb0;
    } else {
     Nthb=M-ci;
    }
    threaddata[nth].B=B;
    threaddata[nth].Bi=Bi;
    threaddata[nth].rho=rho;
    threaddata[nth].Npoly=Npoly;
    threaddata[nth].Nf=Nf;
    threaddata[nth].M=M;
    threaddata[nth].startM=ci;
    threaddata[nth].endM=ci+Nthb-1;

    pthread_create(&th_array[nth],&attr,sum_inv_threadfn,(void*)(&threaddata[nth]));
    ci=ci+Nthb;
  }

  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }
  
  pthread_attr_destroy(&attr);


  free(th_array);
  free(threaddata);

#ifdef DEBUG
  int k,cj;
  for (k=0; k<M; k++) {
    printf("dir_%d=",k);
    for (cj=0; cj<Nf; cj++) {
      printf("%lf ",rho[k+cj*M]);
    }
    printf("\n");
  }
  for (k=0; k<M; k++) {
  printf("Bii_%d=[\n",k);
  for (ci=0; ci<Npoly; ci++) {
   for(cj=0; cj<Npoly; cj++) {
    printf("%lf ",Bi[k*Npoly*Npoly+ci*Npoly+cj]);
   }
   printf("\n");
  }
  printf("];\n");

  }
#endif


  return 0;
}

/* same as above, but add alphaxI to B^T B before inversion */
int
find_prod_inverse_full_fed(double *B, double *Bi, int Npoly, int Nf, int M, double *rho, double *alpha, int Nt) {

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_prod_inv_t *threaddata;

  int ci,Nthb0,Nthb,nth,nth1;
  /* clusters per thread */
  Nthb0=(M+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_prod_inv_t*)malloc((size_t)Nt*sizeof(thread_data_prod_inv_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  ci=0;
  for (nth=0;  nth<Nt && ci<M; nth++) {
    if (ci+Nthb0<M) { 
     Nthb=Nthb0;
    } else {
     Nthb=M-ci;
    }
    threaddata[nth].B=B;
    threaddata[nth].Bi=Bi;
    threaddata[nth].rho=rho;
    threaddata[nth].alpha=alpha;
    threaddata[nth].Npoly=Npoly;
    threaddata[nth].Nf=Nf;
    threaddata[nth].M=M;
    threaddata[nth].startM=ci;
    threaddata[nth].endM=ci+Nthb-1;

    pthread_create(&th_array[nth],&attr,sum_inv_fed_threadfn,(void*)(&threaddata[nth]));
    ci=ci+Nthb;
  }

  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }
  
  pthread_attr_destroy(&attr);


  free(th_array);
  free(threaddata);

#ifdef DEBUG
  int k,cj;
  for (k=0; k<M; k++) {
    printf("dir_%d=",k);
    for (cj=0; cj<Nf; cj++) {
      printf("%lf ",rho[k+cj*M]);
    }
    printf("\n");
  }
  for (k=0; k<M; k++) {
  printf("Bii_%d=[\n",k);
  for (ci=0; ci<Npoly; ci++) {
   for(cj=0; cj<Npoly; cj++) {
    printf("%lf ",Bi[k*Npoly*Npoly+ci*Npoly+cj]);
   }
   printf("\n");
  }
  printf("];\n");

  }
#endif


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


typedef struct thread_data_update_z_ {
 int startM;
 int endM;
 int N;
 int M;
 int Npoly;
 double *Z;
 double *z;
 double *Bi;
} thread_data_update_z_t;


/* worker thread function for updating z */
static void*
update_z_threadfn(void *data) {
  thread_data_update_z_t *t=(thread_data_update_z_t*)data;

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
 if ((R=(double*)calloc((size_t)2*t->N*t->Npoly*4,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 if ((Q=(double*)calloc((size_t)2*t->N*t->Npoly*4,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 
 int ci,np;
 for (ci=t->startM; ci<=t->endM; ci++) {
  for (np=0; np<t->Npoly; np++) {
   /* select 2N */
   my_dcopy(2*t->N, &t->z[8*t->N*ci+np*8*t->N*t->M], 4, &R[np*8*t->N], 1); /* R_1 */
   my_dcopy(2*t->N, &t->z[8*t->N*ci+np*8*t->N*t->M+1], 4, &R[np*8*t->N+2*t->N], 1); /* I_1 */
   my_dcopy(2*t->N, &t->z[8*t->N*ci+np*8*t->N*t->M+2], 4, &R[np*8*t->N+2*2*t->N], 1); /* R_2 */
   my_dcopy(2*t->N, &t->z[8*t->N*ci+np*8*t->N*t->M+3], 4, &R[np*8*t->N+3*2*t->N], 1); /* I_2 */
  }
  /* find Q=R B^T */
  memset(Q,0,sizeof(double)*2*t->N*t->Npoly*4);
  my_dgemm('N','N',8*t->N,t->Npoly,t->Npoly,1.0,R,8*t->N,&t->Bi[ci*t->Npoly*t->Npoly],t->Npoly,1.0,Q,8*t->N);
  /* copy back to Z */ 
  for (np=0; np<t->Npoly; np++) {
   my_dcopy(2*t->N, &Q[np*8*t->N], 1, &t->Z[8*t->N*t->Npoly*ci+8*t->N*np], 4); 
   my_dcopy(2*t->N, &Q[np*8*t->N+2*t->N], 1, &t->Z[8*t->N*t->Npoly*ci+8*t->N*np+1], 4); 
   my_dcopy(2*t->N, &Q[np*8*t->N+2*2*t->N], 1, &t->Z[8*t->N*t->Npoly*ci+8*t->N*np+2], 4); 
   my_dcopy(2*t->N, &Q[np*8*t->N+3*2*t->N], 1, &t->Z[8*t->N*t->Npoly*ci+8*t->N*np+3], 4); 
  }
   
 }

 free(R);
 free(Q);

 return NULL;
}

/* update Z
   Z: 8N Npoly x M double array (real and complex need to be updated separate)
   N : stations
   M : clusters
   Npoly: no of basis functions
   z : right hand side 8NM Npoly x 1 (note the different ordering from Z)
   Bi : M values of NpolyxNpoly matrices, Bi^T=Bi assumed

   Nt: no. of threads
*/
int 
update_global_z_multi(double *Z,int N,int M,int Npoly,double *z,double *Bi, int Nt) {
   pthread_attr_t attr;
   pthread_t *th_array;
   thread_data_update_z_t *threaddata;

  int ci,Nthb0,Nthb,nth,nth1;
  /* clusters per thread */
  Nthb0=(M+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_update_z_t*)malloc((size_t)Nt*sizeof(thread_data_update_z_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  ci=0;
  for (nth=0;  nth<Nt && ci<M; nth++) {
    if (ci+Nthb0<M) {
     Nthb=Nthb0;
    } else {
     Nthb=M-ci;
    }
    threaddata[nth].z=z;
    threaddata[nth].Z=Z;
    threaddata[nth].Bi=Bi;
    threaddata[nth].N=N;
    threaddata[nth].M=M;
    threaddata[nth].Npoly=Npoly;
    threaddata[nth].startM=ci;
    threaddata[nth].endM=ci+Nthb-1;

    pthread_create(&th_array[nth],&attr,update_z_threadfn,(void*)(&threaddata[nth]));
    ci=ci+Nthb;
  }

  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

  pthread_attr_destroy(&attr);


  free(th_array);
  free(threaddata);
 
 return 0;
}

/* generate a random integer in the range 0,1,...,maxval */
int
random_int(int maxval) {
  double rat=(double)random()/(double)RAND_MAX;
  double y=rat*(double)(maxval+1);
  int x=(int)floor(y);
  return x;
}


typedef struct thread_data_rho_bb_ {
 int startM;
 int endM;
 int offset;
 int M;
 int N;
 double *rho;
 double *rhoupper;
 double *deltaY;
 double *deltaJ;
 clus_source_t *carr;
} thread_data_rho_bb_t;


/* worker thread function for calculating sum and inverse */
static void*
rho_bb_threadfn(void *data) {
 thread_data_rho_bb_t *t=(thread_data_rho_bb_t*)data;
 double alphacorrmin=0.2;
 int ci,ck;
 double ip11,ip12,ip22;
 ck=t->offset;
 for (ci=t->startM; ci<=t->endM; ci++) {
   ip12=my_ddot(8*t->N*t->carr[ci].nchunk,&t->deltaY[ck],&t->deltaJ[ck]); /* x^T y */
   /* further computations are only required if there is +ve correlation */
   if (ip12>CLM_EPSILON) {
   /* find the inner products */
   ip11=my_dnrm2(8*t->N*t->carr[ci].nchunk,&t->deltaY[ck]); /* || ||_2 */
   ip22=my_dnrm2(8*t->N*t->carr[ci].nchunk,&t->deltaJ[ck]); /* || ||_2 */
   /* square the norm to get dot prod */
   ip11*=ip11;
   ip22*=ip22;
   /* only try to do an update if the 'delta's are finite, also 
     there is tangible correlation between the two deltas */
#ifdef DEBUG
   printf("%d ip11=%lf ip12=%lf ip22=%lf\n",ci,ip11,ip12,ip22);
#endif
   if (ip11>CLM_EPSILON && ip22>CLM_EPSILON) {
     double alphacorr=ip12/sqrt(ip11*ip22);
     /* decide on whether to do further calculations only if there is sufficient correlation 
        between the deltas */
     if (alphacorr>alphacorrmin) {
     double alphaSD=ip11/ip12;
     double alphaMG=ip12/ip22;
     double alphahat;
     if (2.0*alphaMG>alphaSD) {
      alphahat=alphaMG;
     } else {
      alphahat=alphaSD-alphaMG*0.5;
     }
#ifdef DEBUG
     printf("alphacorr=%lf alphaSD=%lf alphaMG=%lf alphahat=%lf rho=%lf\n",alphacorr,alphaSD,alphaMG,alphahat,t->rho[ci]);
#endif
      /* decide on whether to update rho based on heuristics */
      if (alphahat> 0.001 && alphahat<t->rhoupper[ci]) {
#ifdef DEBUG
       printf("updating rho from %lf to %lf\n",t->rho[ci],alphahat);
#endif
       t->rho[ci]=alphahat;
      }
     }
   }
  
   } 
   ck+=t->N*8*t->carr[ci].nchunk;
 }
 return NULL;
}


/* Barzilai & Borwein update of rho [Xu et al] */
/* rho: Mx1 values, to be updated
   rhoupper: Mx1 values, upper limit of rho
   N: no of stations
   M : clusters
   Mt: actual clusters (include hybrid parameter) Mt >= M
   carr: cluster info array, to get hybrid parameters: Mx1
   Yhat: current Yhat : 8*N*Mt 
   Yhat_k0 : old Yhat at previous update of rho : 8*N*Mt
   J: current solution : 8*N*Mt
   J_k0: old solution at previous update of rho : 8*N*Mt
   Nt: no. of threads
*/ 
int
update_rho_bb(double *rho, double *rhoupper, int N, int M, int Mt, clus_source_t *carr, double *Yhat, double *Yhat_k0, double *J, double *J_k0, int Nt) {

 double *deltaY; /* Yhat - Yhat_k0 */
 double *deltaJ; /* J - J_k0 (with J_k0 projected to tangent plane of J)*/
 if ((deltaY=(double*)calloc((size_t)8*N*Mt,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 if ((deltaJ=(double*)calloc((size_t)8*N*Mt,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
 }
 

 my_dcopy(8*N*Mt, Yhat, 1, deltaY, 1); 
 my_daxpy(8*N*Mt, Yhat_k0, -1.0, deltaY);
//no need to remove unitary ambiguity from J-Jold
 my_dcopy(8*N*Mt, J, 1, deltaJ, 1); 
 my_daxpy(8*N*Mt, J_k0,-1.0, deltaJ);

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_rho_bb_t *threaddata;

  int ci,cj,ck,Nthb0,Nthb,nth,nth1;
  /* clusters per thread */
  Nthb0=(M+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_rho_bb_t*)malloc((size_t)Nt*sizeof(thread_data_rho_bb_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }


  ci=0;
  ck=0;
  for (nth=0;  nth<Nt && ci<M; nth++) {
    if (ci+Nthb0<M) {
     Nthb=Nthb0;
    } else {
     Nthb=M-ci;
    }
    threaddata[nth].N=N;
    threaddata[nth].M=M;
    threaddata[nth].offset=ck;
    threaddata[nth].startM=ci;
    threaddata[nth].endM=ci+Nthb-1;
    threaddata[nth].rho=rho;
    threaddata[nth].rhoupper=rhoupper;
    threaddata[nth].deltaY=deltaY;
    threaddata[nth].deltaJ=deltaJ;
    threaddata[nth].carr=carr;
    /* find the right offset too, since ci is not always incremented by 1 need to add up */
    for (cj=ci; cj<ci+Nthb && cj<M; cj++) {
      ck+=N*8*carr[cj].nchunk;
    }


    pthread_create(&th_array[nth],&attr,rho_bb_threadfn,(void*)(&threaddata[nth]));
    ci=ci+Nthb;
  }

  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);


 free(deltaY);
 free(deltaJ);
 return 0;
}



typedef struct thread_data_sthreshold_ {
 int starti;
 int endi;
 double *z;
 double lambda;
} thread_data_sthreshold_t;


/* soft thresholding function */
static void*
sthreshold_threadfn(void *data) {
  thread_data_sthreshold_t *t=(thread_data_sthreshold_t*) data;
  int ci;
  for (ci=t->starti; ci<=t->endi; ci++) {
   /* elementwise soft threshold */
   t->z[ci]=(t->z[ci]<-t->lambda?t->z[ci]+t->lambda:(t->z[ci]>t->lambda?t->z[ci]-t->lambda:0.0));
  }

  return NULL;
}

/* soft threshold elementwise
   z: Nx1 data vector (or matrix) : this is modified
   lambda: threshold
   Nt: no. of threads

   Z_i ={ Z_i-lambda if Z_i > lambda, Z_i+lambda  if Z_i<-lambda, else 0}
*/
int
soft_threshold_z(double *z, int N, double lambda, int Nt) {

  int Nthb,Nthb0,ci,nth,nth1;

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_sthreshold_t *threaddata;


  /* values per thread */
  Nthb0=(N+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_sthreshold_t*)malloc((size_t)Nt*sizeof(thread_data_sthreshold_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  ci=0;
  for (nth=0;  nth<Nt && ci<N; nth++) {
    if (ci+Nthb0<N) {
     Nthb=Nthb0;
    } else {
     Nthb=N-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].lambda=lambda;
    threaddata[nth].z=z;

    pthread_create(&th_array[nth],&attr,sthreshold_threadfn,(void*)(&threaddata[nth]));
    ci=ci+Nthb;
  }

  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);

   return 0;
}



/* 
 * B_f Z Phi_k = J_0, where J_0 = 1_N \kron I_2
 * B_f : 2N x 2Npoly N , Nf values
 * Phi_k : 2G x 2, M values
 * Z : 2Npoly N x 2G
 * Z = (\sum_f B_f^T B_f)^{-1} (\sum_f B_f^T) J_0 (\sum_k Phi_k^H) (\sum_k Phi_k Phi_k^H)^{-1}
 * Note: B: Npoly x  Nf, each row one basis function, use this to create B_f = B(:col) \kron I_2N = b_f^T \kron I_2N
 * phi: MxG vector, Phi_k = I_2 \kron phi_k, G values, k=0..M-1
 *
 * Final product : (Bii_sum_b \kron I_2N) (1_N \kron I_2) (I_2 \kron sum_phi_Phi^T)
 *
 * Z: 2N Npoly x 2G output
 */
int 
find_initial_spatial(double *B, complex double *phi, int Npoly, int N, int Nf, int M, int G, complex double *Z) {

  double w[1],*WORK,*U,*S,*VT;
  int status,lwork=1;
  double *sum_b_f, *Bi, *Binv_b;

  if ((sum_b_f=(double*)calloc((size_t)Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((Bi=(double*)calloc((size_t)Npoly*Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  for (int ci=0; ci<Nf; ci++) {
    my_daxpy(Npoly,&B[ci*Npoly],1.0,sum_b_f);
    /* b_f x b_f^T */
    my_dgemm('N','T',Npoly,Npoly,1,1.0,&B[ci*Npoly],Npoly,&B[ci*Npoly],Npoly,1.0,Bi,Npoly);
  }

#ifdef DEBUG
  printf("BT=[\n");
  for (int ci=0; ci<Nf; ci++) {
   for(int cj=0; cj<Npoly; cj++) {
    printf("%lf ",B[ci*Npoly+cj]);
   }
   printf("\n");
  }
  printf("];\nBi=[\n");
  for (int ci=0; ci<Npoly; ci++) {
   for(int cj=0; cj<Npoly; cj++) {
    printf("%lf ",Bi[ci*Npoly+cj]);
   }
   printf("\n");
  }
  printf("];\nsum_b_f=[");
  for (int ci=0; ci<Npoly; ci++) {
    printf("%lf ",sum_b_f[ci]);
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
  for (int ci=0; ci<Npoly; ci++) {
   if (S[ci]>CLM_EPSILON) {
    S[ci]=1.0/S[ci];
   } else {
    S[ci]=0.0;
   }
   my_dscal(Npoly,S[ci],&U[ci*Npoly]);
  }

  /* find product U 1/S V^T */
  my_dgemm('N','N',Npoly,Npoly,Npoly,1.0,U,Npoly,VT,Npoly,0.0,Bi,Npoly);

  free(U);
  free(S);
  free(VT);
  free(WORK);

#ifdef DEBUG
  printf("Bii=[\n");
  for (int ci=0; ci<Npoly; ci++) {
   for(int cj=0; cj<Npoly; cj++) {
    printf("%lf ",Bi[ci*Npoly+cj]);
   }
   printf("\n");
  }
  printf("];\n");
#endif

  if ((Binv_b=(double*)calloc((size_t)Npoly,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  /* (\sum_f B_f^T B_f)^{-1} (\sum_f B_f^T) 
   * = ( (\sum_f b_f^T b_f)^{-1} \sum_f b_f ) \kron I_2N 
   * so find the product Bi x sum_b_f
   */
  my_dgemv('N',Npoly,Npoly,1.0,Bi,Npoly,sum_b_f,1,0.0,Binv_b,1);

#ifdef DEBUG
  printf("Binv_sum_b_f=[");
  for (int ci=0; ci<Npoly; ci++) {
    printf("%lf ",Binv_b[ci]);
  }
  printf("];\n");
#endif

  complex double *Phi,*sum_phi_k;

  if ((sum_phi_k=(complex double*)calloc((size_t)G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((Phi=(complex double*)calloc((size_t)G*G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  for (int ci=0; ci<M; ci++) {
    /* \sum_k phi_k = conj( \sum_k phi_k ) */
    my_caxpy(G,&phi[ci*G],1.0,sum_phi_k);
    /* phi_k x phi_k^H */
    my_zgemm('N','C',G,G,1,1.0,&phi[ci*G],G,&phi[ci*G],G,1.0,Phi,G);
  }

#ifdef DEBUG
  printf("sum_phi_k=[");
  for (int ci=0; ci<G; ci++) {
    printf("%lf+j*(%lf) ",creal(sum_phi_k[ci]),cimag(sum_phi_k[ci]));
  }
  printf("];\n");
#endif

  /* conjugate sum_phi_k */
  for (int ci=0; ci<G; ci++) {
    sum_phi_k[ci]=creal(sum_phi_k[ci])-_Complex_I*cimag(sum_phi_k[ci]);
  }

#ifdef DEBUG
  printf("sum_phi_k_H=[");
  for (int ci=0; ci<G; ci++) {
    printf("%lf+j*(%lf) ",creal(sum_phi_k[ci]),cimag(sum_phi_k[ci]));
  }
  printf("];\n");
#endif


  complex double cw[1],*cWORK,*cU,*cVT;
  double *RWORK; /* > 5 G */

  if ((cU=(complex double*)calloc((size_t)G*G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((cVT=(complex double*)calloc((size_t)G*G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((S=(double*)calloc((size_t)G,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((RWORK=(double*)calloc((size_t)6*G,sizeof(double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  status=my_zgesvd('A','A',G,G,Phi,G,S,cU,G,cVT,G,cw,-1,RWORK);
  if (!status) {
    lwork=(int)w[0];
  } else {
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
   exit(1);
  }
  if ((cWORK=(complex double*)malloc((size_t)(int)lwork*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }

#ifdef DEBUG
  printf("Phi=[\n");
  for (int ci=0; ci<G; ci++) {
   for(int cj=0; cj<G; cj++) {
    printf("%lf+j*(%lf) ",creal(Phi[ci*G+cj]),cimag(Phi[ci*G+cj]));
   }
   printf("\n");
  }
  printf("];\n");
#endif


  status=my_zgesvd('A','A',G,G,Phi,G,S,cU,G,cVT,G,cWORK,lwork,RWORK);
  if (status) {
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
   exit(1);
  }

  /* find 1/singular values, and multiply columns of U with new singular values */
  for (int ci=0; ci<G; ci++) {
   if (S[ci]>CLM_EPSILON) {
    S[ci]=1.0/S[ci];
   } else {
    S[ci]=0.0;
   }
   my_cscal(G,S[ci],&cU[ci*G]);
  }


  /* find product U 1/S V^T */
  my_zgemm('N','N',G,G,G,1.0,cU,G,cVT,G,0.0,Phi,G);

#ifdef DEBUG
  printf("Phii=[\n");
  for (int ci=0; ci<G; ci++) {
   for(int cj=0; cj<G; cj++) {
    printf("%lf+j*(%lf) ",creal(Phi[ci*G+cj]),cimag(Phi[ci*G+cj]));
   }
   printf("\n");
  }
  printf("];\n");
#endif


  free(cU);
  free(cVT);
  free(S);
  free(RWORK);
  free(cWORK);

  complex double *sum_phi_Phii;
  if ((sum_phi_Phii=(complex double*)calloc((size_t)G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* find product sum_phi_k^H x Phii */
  my_zgemm('C','N',1,G,G,1.0,sum_phi_k,G,Phi,G,0.0,sum_phi_Phii,1);

#ifdef DEBUG
  printf("sum_phi_Phii=[");
  for (int ci=0; ci<G; ci++) {
    printf("%lf+j*(%lf) ",creal(sum_phi_Phii[ci]),cimag(sum_phi_Phii[ci]));
  }
  printf("];\n");
#endif

  /* form I_2 \kron sum_phi_Phii^T :  2 x 2G */
  complex double *Phi_rhs;
  if ((Phi_rhs=(complex double*)calloc((size_t)2*2*G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  my_ccopy(G, sum_phi_Phii, 1, Phi_rhs, 2);
  my_ccopy(G, sum_phi_Phii, 1, &Phi_rhs[2*G+1], 2);

#ifdef DEBUG
  printf("Phi_rhs=[\n");
  for (int ci=0; ci<2; ci++) {
    for (int cj=0; cj<2*G; cj++) {
    printf("%lf+j*(%lf) ",creal(Phi_rhs[2*cj+ci]),cimag(Phi_rhs[2*cj+ci]));
    }
    printf("\n");
  }
  printf("];\n");
#endif

  /* form 1_N \kron I_2 : 2N x 2 */
  complex double *J_0;
  if ((J_0=(complex double*)calloc((size_t)2*N*2,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  for (int ci=0; ci<N; ci++) {
    J_0[2*ci]=1.0;
    J_0[2*N+2*ci+1]=1.0;
  }

#ifdef DEBUG
  printf("J_0=[\n");
  for (int ci=0; ci<2*N; ci++) {
    for (int cj=0; cj<2; cj++) {
    printf("%lf+j*(%lf) ",creal(J_0[2*N*cj+ci]),cimag(J_0[2*N*cj+ci]));
    }
    printf("\n");
  }
  printf("];\n");
#endif

  complex double *J_Phi;
  if ((J_Phi=(complex double*)calloc((size_t)2*N*2*G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* form product (2Nx2) x (2x2G) = 2Nx2G */
  my_zgemm('N','N',2*N,2*G,2,1.0,J_0,2*N,Phi_rhs,2,1.0,J_Phi,2*N);

#ifdef DEBUG
  printf("J_Phi=[\n");
  for (int ci=0; ci<2*N; ci++) {
    for (int cj=0; cj<2*G; cj++) {
    printf("%lf+j*(%lf) ",creal(J_Phi[2*N*cj+ci]),cimag(J_Phi[2*N*cj+ci]));
    }
    printf("\n");
  }
  printf("];\n");
#endif

  /* form product (b_sum \kron I_2N) x J_Phi
   * Npoly elements in b_sum, (Npoly*2N x 2N) x (2N x 2G)
   * out row 0 : b[0] x J_Phi[row 0]
   * out row 1 : b[1] x J_Phi[row 0]
   * ..
   * out row Npoly-1 : b[Npoly-1] x J_Phi[row 0]
   * out row Npoly : b[0] x J_Phi[row 1]
   * ..
   * out row 2Npoly-1 : b[Npoly-1] x J_Phi[row 1] ...
   * out row ci : b[ci%Npoly] x J_Phi[row ci/Npoly], ci = 0...2N*Npoly-1 
   */

  complex double *row;
  if ((row=(complex double*)calloc((size_t)2*G,sizeof(complex double)))==0) {
    printf("%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  for (int ci=0; ci<2*N*Npoly; ci++) {
   my_ccopy(2*G,&J_Phi[ci/Npoly],2*N,row,1);
   my_cscal(2*G,sum_b_f[ci%Npoly],row);
   my_ccopy(2*G,row,1,&Z[ci],2*N*Npoly);
  }

  free(sum_b_f);
  free(Bi);
  free(Binv_b);

  free(Phi);
  free(sum_phi_k);
  free(sum_phi_Phii);
  free(Phi_rhs);
  free(J_0);
  free(J_Phi);
  free(row);
  return 0;
}
