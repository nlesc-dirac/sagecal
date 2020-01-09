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

//#define DEBUG
typedef struct thread_data_manavg_ {
 double *Y;
 int startM;
 int endM;
 int Niter;
 int N;
 int M;
 int Nf;
 int randomize;
} thread_data_manavg_t;

/* worker thread function for manifold average+projection 
  project the solution to the average */
static void*
manifold_average_threadfn(void *data) {
 thread_data_manavg_t *t=(thread_data_manavg_t*)data;
 int ci,cj,iter;
 double *Yl;
 complex double *J3,*Jp;
 /* local storage 2Nx2 x Nf complex values */
 if ((Yl=(double*)malloc((size_t)t->N*8*t->Nf*sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((J3=(complex double*)malloc((size_t)t->N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((Jp=(complex double*)malloc((size_t)t->N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
#ifdef DEBUG
 complex double *Jerr;
 if ((Jerr=(complex double*)malloc((size_t)t->N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
#endif

 complex double *Yc=(complex double*)Yl;
 complex double a=1.0/(double)t->Nf+0.0*_Complex_I;

 /* work for SVD */
 complex double *WORK=0;
 complex double w[1];
 double RWORK[32]; /* size > 5*max_matrix_dimension */
 complex double JTJ[4],U[4],VT[4];
 double S[2];
 
 int status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,w,-1,RWORK);
 if (status!=0) {
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
   exit(1);
 } 
 int lwork=(int)w[0];
 if ((WORK=(complex double*)malloc((size_t)(int)lwork*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }

 for (ci=t->startM; ci<=t->endM; ci++) {
   /* copy to local storage */
   for (cj=0; cj<t->Nf; cj++) {
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N], 8, &Yl[cj*8*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+1], 8, &Yl[cj*8*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+4], 8, &Yl[cj*8*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+5], 8, &Yl[cj*8*t->N+3], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+2], 8, &Yl[cj*8*t->N+4*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+3], 8, &Yl[cj*8*t->N+4*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+6], 8, &Yl[cj*8*t->N+4*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+7], 8, &Yl[cj*8*t->N+4*t->N+3], 4);

   }
   /* first averaging, select random block in [0,Nf-1] to project to */
   int cr; /* remainder always in [0,Nf-1] */
   if (t->randomize) {
    cr=rand()%(t->Nf); /* remainder always in [0,Nf-1] */
   } else {
    cr=0;
   }
   /* J3 <= cr th  block */
   my_ccopy(t->N*4,&Yc[cr*t->N*4],1,J3,1);
   /* project the remainder */
   for (cj=0; cj<cr; cj++) {
      project_procrustes_block(t->N,J3,&Yc[cj*t->N*4]);
   }
   for (cj=cr+1; cj<t->Nf; cj++) {
      project_procrustes_block(t->N,J3,&Yc[cj*t->N*4]);
   }


   /* now each 2, 2N complex vales is one J block */
   /* average values and project to common average */
   for (iter=0; iter<t->Niter; iter++) {
     /* J3 <= 1st block */
     my_ccopy(t->N*4,Yc,1,J3,1); 
     /* add the remainder */
     for (cj=1; cj<t->Nf; cj++) {
     my_caxpy(t->N*4,&Yc[cj*t->N*4],1.0+_Complex_I*0.0,J3);
     }
     my_cscal(t->N*4,a,J3);
     /* now find unitary matrix using Procrustes problem */
     for (cj=0; cj<t->Nf; cj++) {
       /* find product JTJ = J^H J3 */
       my_zgemm('C','N',2,2,2*t->N,1.0+_Complex_I*0.0,&Yc[cj*t->N*4],2*t->N,J3,2*t->N,0.0+_Complex_I*0.0,JTJ,2);
       status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,WORK,lwork,RWORK);
       //printf("%d %d %lf %lf\n",ci,cj,S[0],S[1]);
       /* find JTJ= U V^H */
       my_zgemm('N','N',2,2,2,1.0+_Complex_I*0.0,U,2,VT,2,0.0+_Complex_I*0.0,JTJ,2);
       /* find J*(JTJ) : projected matrix */
       my_zgemm('N','N',2*t->N,2,2,1.0+_Complex_I*0.0,&Yc[cj*t->N*4],2*t->N,JTJ,2,0.0+_Complex_I*0.0,Jp,2*t->N);
       /* copy back */
       my_ccopy(t->N*4,Jp,1,&Yc[cj*t->N*4],1); 
#ifdef DEBUG
     /* calculate error between projected value and global mean */
     my_ccopy(t->N*4,J3,1,Jerr,1); 
     my_caxpy(t->N*4,&Yc[cj*t->N*4],-1.0+_Complex_I*0.0,Jerr);
     printf("Error freq=%d dir=%d iter=%d %lf\n",cj,ci,iter,my_cnrm2(t->N*4,Jerr));
#endif
     }
   }

   /* now get a fresh copy, because we should modify Y only by 
      one unitary matrix  */
   my_ccopy(t->N*4,Yc,1,J3,1);
   /* add the remainder */
   for (cj=1; cj<t->Nf; cj++) {
      my_caxpy(t->N*4,&Yc[cj*t->N*4],1.0+_Complex_I*0.0,J3);
   }
   my_cscal(t->N*4,a,J3);
   for (cj=0; cj<t->Nf; cj++) {
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N], 8, &Yl[cj*8*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+1], 8, &Yl[cj*8*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+4], 8, &Yl[cj*8*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+5], 8, &Yl[cj*8*t->N+3], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+2], 8, &Yl[cj*8*t->N+4*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+3], 8, &Yl[cj*8*t->N+4*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+6], 8, &Yl[cj*8*t->N+4*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+7], 8, &Yl[cj*8*t->N+4*t->N+3], 4);
   }

   for (cj=0; cj<t->Nf; cj++) {
       /* find product JTJ = J^H J3 */
       my_zgemm('C','N',2,2,2*t->N,1.0+_Complex_I*0.0,&Yc[cj*t->N*4],2*t->N,J3,2*t->N,0.0+_Complex_I*0.0,JTJ,2);

       status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,WORK,lwork,RWORK);
       /* find JTJ= U V^H */
       my_zgemm('N','N',2,2,2,1.0+_Complex_I*0.0,U,2,VT,2,0.0+_Complex_I*0.0,JTJ,2);
       /* find J*(JTJ) : projected matrix */
       my_zgemm('N','N',2*t->N,2,2,1.0+_Complex_I*0.0,&Yc[cj*t->N*4],2*t->N,JTJ,2,0.0+_Complex_I*0.0,Jp,2*t->N);
       /* copy back */
       my_ccopy(t->N*4,Jp,1,&Yc[cj*t->N*4],1);
   }

   /* copy back from local storage */
   for (cj=0; cj<t->Nf; cj++) {
     my_dcopy(t->N, &Yl[cj*8*t->N], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+1], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+1], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+2], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+4], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+3], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+5], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+2], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N+1], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+3], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N+2], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+6], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N+3], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+7], 8);

   }
 }

#ifdef DEBUG
 free(Jerr);
#endif
 free(Yl);
 free(J3);
 free(Jp);
 free(WORK);
 return NULL;
}

int
calculate_manifold_average(int N,int M,int Nf,double *Y,int Niter,int randomize,int Nt) {
 /* Y : each 2Nx2xM blocks belong to one freq,
   select one 2Nx2 from this, reorder to J format : Nf blocks
   and average */
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_manavg_t *threaddata;

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
  if ((threaddata=(thread_data_manavg_t*)malloc((size_t)Nt*sizeof(thread_data_manavg_t)))==0) {
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
    threaddata[nth].Y=Y;
    threaddata[nth].N=N;
    threaddata[nth].M=M;
    threaddata[nth].Nf=Nf;
    threaddata[nth].Niter=Niter;
    threaddata[nth].startM=ci;
    threaddata[nth].endM=ci+Nthb-1;
    threaddata[nth].randomize=randomize;
    
    pthread_create(&th_array[nth],&attr,manifold_average_threadfn,(void*)(&threaddata[nth]));
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



int
project_procrustes(int N,double *J,double *J1) {
 /* min ||J - J1 U || find U */
 complex double *X,*Y;
 /* local storage */
 if ((X=(complex double*)malloc((size_t)N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((Y=(complex double*)malloc((size_t)N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 double *Jx=(double*)X;
 double *Jy=(double*)Y;
 /* copy to get correct format */
 my_dcopy(N, &J[0], 8, &Jx[0], 4);
 my_dcopy(N, &J[0+1], 8, &Jx[1], 4);
 my_dcopy(N, &J[0+4], 8, &Jx[2], 4);
 my_dcopy(N, &J[0+5], 8, &Jx[3], 4);
 my_dcopy(N, &J[0+2], 8, &Jx[4*N], 4);
 my_dcopy(N, &J[0+3], 8, &Jx[4*N+1], 4);
 my_dcopy(N, &J[0+6], 8, &Jx[4*N+2], 4);
 my_dcopy(N, &J[0+7], 8, &Jx[4*N+3], 4);
 my_dcopy(N, &J1[0], 8, &Jy[0], 4);
 my_dcopy(N, &J1[0+1], 8, &Jy[1], 4);
 my_dcopy(N, &J1[0+4], 8, &Jy[2], 4);
 my_dcopy(N, &J1[0+5], 8, &Jy[3], 4);
 my_dcopy(N, &J1[0+2], 8, &Jy[4*N], 4);
 my_dcopy(N, &J1[0+3], 8, &Jy[4*N+1], 4);
 my_dcopy(N, &J1[0+6], 8, &Jy[4*N+2], 4);
 my_dcopy(N, &J1[0+7], 8, &Jy[4*N+3], 4);

 /* min ||X - Y U|| find U */

 /* work for SVD */
 complex double *WORK=0;
 complex double w[1];
 double RWORK[32]; /* size > 5*max_matrix_dimension */
 complex double JTJ[4],U[4],VT[4];
 double S[2];
 
 int status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,w,-1,RWORK);
 if (status!=0) {
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
   exit(1);
 } 
 int lwork=(int)w[0];
 if ((WORK=(complex double*)malloc((size_t)(int)lwork*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }

 /* find product JTJ = Y^H X */
 my_zgemm('C','N',2,2,2*N,1.0+_Complex_I*0.0,Y,2*N,X,2*N,0.0+_Complex_I*0.0,JTJ,2);
 /* JTJ = U S V^H */
 status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,WORK,lwork,RWORK);
 /* find JTJ= U V^H */
 my_zgemm('N','N',2,2,2,1.0+_Complex_I*0.0,U,2,VT,2,0.0+_Complex_I*0.0,JTJ,2);
 /* find Y*(JTJ) : projected matrix -> store in X */
 my_zgemm('N','N',2*N,2,2,1.0+_Complex_I*0.0,Y,2*N,JTJ,2,0.0+_Complex_I*0.0,X,2*N);

 my_dcopy(N, &Jx[0], 4, &J1[0], 8);
 my_dcopy(N, &Jx[1], 4, &J1[0+1], 8);
 my_dcopy(N, &Jx[2], 4, &J1[0+4], 8);
 my_dcopy(N, &Jx[3], 4, &J1[0+5], 8);
 my_dcopy(N, &Jx[4*N], 4, &J1[0+2], 8);
 my_dcopy(N, &Jx[4*N+1], 4, &J1[0+3], 8);
 my_dcopy(N, &Jx[4*N+2], 4, &J1[0+6], 8);
 my_dcopy(N, &Jx[4*N+3], 4, &J1[0+7], 8);


 free(WORK);
 free(X);
 free(Y);
 return 0;
}



int
project_procrustes_block(int N,complex double *X,complex double *Y) {
 /* min ||X - Y U || find U */
 complex double *Jlocal;
 /* local storage */
 if ((Jlocal=(complex double*)malloc((size_t)N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }

 /* work for SVD */
 complex double *WORK=0;
 complex double w[1];
 double RWORK[32]; /* size > 5*max_matrix_dimension */
 complex double JTJ[4],U[4],VT[4];
 double S[2];
 
 int status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,w,-1,RWORK);
 if (status!=0) {
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
   exit(1);
 } 
 int lwork=(int)w[0];
 if ((WORK=(complex double*)malloc((size_t)(int)lwork*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }

 /* find product JTJ = Y^H X */
 my_zgemm('C','N',2,2,2*N,1.0+_Complex_I*0.0,Y,2*N,X,2*N,0.0+_Complex_I*0.0,JTJ,2);
 /* JTJ = U S V^H */
 status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,WORK,lwork,RWORK);
 /* find JTJ= U V^H */
 my_zgemm('N','N',2,2,2,1.0+_Complex_I*0.0,U,2,VT,2,0.0+_Complex_I*0.0,JTJ,2);
 /* find Y*(JTJ) : projected matrix -> store in Jlocal */
 my_zgemm('N','N',2*N,2,2,1.0+_Complex_I*0.0,Y,2*N,JTJ,2,0.0+_Complex_I*0.0,Jlocal,2*N);

 /* copy Jlocal -> Y */
 my_dcopy(8*N, (double*)Jlocal, 1, (double*)Y, 1);

 free(WORK);
 free(Jlocal);
 return 0;
}




//#define DEBUG
/* Extract only the phase of diagonal entries from solutions 
   p: 8Nx1 solutions, orders as [(real,imag)vec(J1),(real,imag)vec(J2),...]
   pout: 8Nx1 phases (exp(j*phase)) of solutions, after joint diagonalization of p
   N: no. of 2x2 Jones matrices in p, having common unitary ambiguity
   niter: no of iterations for Jacobi rotation */
int
extract_phases(double *p, double *pout, int N, int niter) {

  /* local storage */
  complex double *J,*Jcopy;
  /* local storage, change ordering of solutions [J_1^T,J_2^T,...]^T  */
  if ((J=(complex double*)malloc((size_t)N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((Jcopy=(complex double*)malloc((size_t)N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }

  double *Jx=(double *)J;
  /* copy to get correct format */
  my_dcopy(N, &p[0], 8, &Jx[0], 4);
  my_dcopy(N, &p[0+1], 8, &Jx[1], 4);
  my_dcopy(N, &p[0+4], 8, &Jx[2], 4);
  my_dcopy(N, &p[0+5], 8, &Jx[3], 4);
  my_dcopy(N, &p[0+2], 8, &Jx[4*N], 4);
  my_dcopy(N, &p[0+3], 8, &Jx[4*N+1], 4);
  my_dcopy(N, &p[0+6], 8, &Jx[4*N+2], 4);
  my_dcopy(N, &p[0+7], 8, &Jx[4*N+3], 4);

  complex double h[3],Hc[9];
  double H[9]; 
  double W[3],Z[3];
  double w[1],*WORK;
  int IWORK[15],IFAIL[3],info;
  int ni,ci;
  complex double c,s,G[4];
  
#ifdef DEBUG
  printf("J=[\n");
  for (ci=0; ci<N; ci++) {
   printf("%lf+j*(%lf), %lf+j*(%lf)\n",p[8*ci],p[8*ci+1],p[8*ci+2],p[8*ci+3]);
   printf("%lf+j*(%lf), %lf+j*(%lf)\n",p[8*ci+4],p[8*ci+5],p[8*ci+6],p[8*ci+7]);
  }
  printf("];\n");
#endif
  /* setup workspace for eigenvalue decomposition */
  info=my_dsyevx('V','I','L',3,H,3,0.0,0.0,3,3,dlamch('S'),1,W,Z,3,w,-1,IWORK,IFAIL);
  if (info) {
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,info);
   exit(1);
  }
  /* get work size */
  int lwork=(int)w[0];
  /* allocate memory */
  if ((WORK=(double*)malloc((size_t)lwork*sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  } 
  /* iteration loop */
  for (ni=0; ni<niter; ni++) {
  
    /************** for element (1,2) **********************/
    /* accumulate h*h^H product */
    memset(Hc,0,9*sizeof(complex double));
    for (ci=0; ci<N; ci++) {
       /* [a_ii-a_jj,a_ij+a_ji,I*(a_ji-a_ij)] */
       h[0]=conj(J[2*ci]-J[2*ci+2*N+1]);
       h[1]=conj(J[2*ci+2*N]+J[2*ci+1]);
       h[2]=conj(_Complex_I*(J[2*ci+1]-J[2*ci+2*N]));
       /* store results onto lower triangle */
       my_zher('L',3,1.0,h,1,Hc,3);
    }
    /* get real part, copy it to lower triangle */
    H[0]=creal(Hc[0]);
    H[1]=creal(Hc[1]);
    H[2]=creal(Hc[2]);
    H[4]=creal(Hc[4]);
    H[5]=creal(Hc[5]);
    H[8]=creal(Hc[8]);
#ifdef DEBUG
    printf("H=[\n");
    printf("%e %e %e\n",H[0],H[1],H[2]);
    printf("%e %e %e\n",H[1],H[4],H[5]);
    printf("%e %e %e\n",H[2],H[5],H[8]);
    printf("];\n");
#endif
    info=my_dsyevx('V','I','L',3,H,3,0.0,0.0,3,3,dlamch('S'),1,W,Z,3,WORK,lwork,IWORK,IFAIL);
    if (info<0) {
     fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,info);
     exit(1);
    }
#ifdef DEBUG
    printf("max eigenvalue=%e\n",W[0]);
    printf("ev=[\n");
    printf("%e\n",Z[0]);
    printf("%e\n",Z[1]);
    printf("%e\n",Z[2]);
    printf("];\n");
#endif

   /* form sin,cos values */
   if (Z[0]>=0.0) {
    c=sqrt(0.5+Z[0]*0.5)+_Complex_I*0.0;
    s=0.5*(Z[1]-_Complex_I*Z[2])/c;
   } else {
    /* flip sign of eigenvector */
    c=sqrt(0.5-Z[0]*0.5)+_Complex_I*0.0;
    s=0.5*(-Z[1]+_Complex_I*Z[2])/c;
   }
   /* form Givens rotation matrix */
   G[0]=c;
   G[1]=-s;
   G[2]=conj(s);
   G[3]=conj(c);
#ifdef DEBUG
   printf("G=[\n");
   printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(G[0]),cimag(G[0]),creal(G[2]),cimag(G[2]));
   printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(G[1]),cimag(G[1]),creal(G[3]),cimag(G[3]));
   printf("];\n");
#endif
   /* rotate J <= J * G^H: Jcopy = 1 x J x G^H  + 0 x Jcopy */
   my_zgemm('N','C',2*N,2,2,1.0+_Complex_I*0.0,J,2*N,G,2,0.0+_Complex_I*0.0,Jcopy,2*N);
   memcpy(J,Jcopy,(size_t)4*N*sizeof(complex double));
#ifdef DEBUG
   printf("JGH=[\n");
   for (ci=0; ci<N; ci++) {
    printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(J[2*ci]),cimag(J[2*ci]),creal(J[2*N+2*ci]),cimag(J[2*N+2*ci]));
    printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(J[2*ci+1]),cimag(J[2*ci+1]),creal(J[2*N+2*ci+1]),cimag(J[2*N+2*ci+1]));
   }
   printf("];\n");
#endif

    /************** for element (2,1) **********************/
    /* accumulate h*h^H product */
    memset(Hc,0,9*sizeof(complex double));
    for (ci=0; ci<N; ci++) {
       /* [a_ii-a_jj,a_ij+a_ji,I*(a_ji-a_ij)] */
       h[0]=conj(J[2*ci+2*N+1]-J[2*ci]);
       h[1]=conj(J[2*ci+1]+J[2*ci+2*N]);
       h[2]=conj(_Complex_I*(J[2*ci+2*N]-J[2*ci+1]));
       /* store results onto lower triangle */
       my_zher('L',3,1.0,h,1,Hc,3);
    }
    /* get real part, copy it to lower triangle */
    H[0]=creal(Hc[0]);
    H[1]=creal(Hc[1]);
    H[2]=creal(Hc[2]);
    H[4]=creal(Hc[4]);
    H[5]=creal(Hc[5]);
    H[8]=creal(Hc[8]);
#ifdef DEBUG
    printf("H=[\n");
    printf("%e %e %e\n",H[0],H[1],H[2]);
    printf("%e %e %e\n",H[1],H[4],H[5]);
    printf("%e %e %e\n",H[2],H[5],H[8]);
    printf("];\n");
#endif
    info=my_dsyevx('V','I','L',3,H,3,0.0,0.0,3,3,dlamch('S'),1,W,Z,3,WORK,lwork,IWORK,IFAIL);
    if (info<0) {
     fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,info);
     exit(1);
    }
#ifdef DEBUG
    printf("max eigenvalue=%e\n",W[0]);
    printf("ev=[\n");
    printf("%e\n",Z[0]);
    printf("%e\n",Z[1]);
    printf("%e\n",Z[2]);
    printf("];\n");
#endif

   /* form sin,cos values */
   if (Z[0]>=0.0) {
    c=sqrt(0.5+Z[0]*0.5)+_Complex_I*0.0;
    s=0.5*(Z[1]-_Complex_I*Z[2])/c;
   } else {
    /* flip sign of eigenvector */
    c=sqrt(0.5-Z[0]*0.5)+_Complex_I*0.0;
    s=0.5*(-Z[1]+_Complex_I*Z[2])/c;
   }
   /* form Givens rotation matrix */
   G[0]=c;
   G[1]=-s;
   G[2]=conj(s);
   G[3]=conj(c);
#ifdef DEBUG
   printf("G=[\n");
   printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(G[0]),cimag(G[0]),creal(G[2]),cimag(G[2]));
   printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(G[1]),cimag(G[1]),creal(G[3]),cimag(G[3]));
   printf("];\n");
#endif
   /* rotate J <= J * G^H: Jcopy = 1 x J x G^H  + 0 x Jcopy */
   my_zgemm('N','C',2*N,2,2,1.0+_Complex_I*0.0,J,2*N,G,2,0.0+_Complex_I*0.0,Jcopy,2*N);
   /* before copying updated result, find residual norm */
   /* J = -Jcopy + J */
   my_caxpy(4*N,Jcopy,-1.0+_Complex_I*0.0,J); 
#ifdef DEBUG
   printf("Iter %d residual=%lf\n",ni,my_cnrm2(4*N,J));
#endif
   memcpy(J,Jcopy,(size_t)4*N*sizeof(complex double));
#ifdef DEBUG
   printf("JGH=[\n");
   for (ci=0; ci<N; ci++) {
    printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(J[2*ci]),cimag(J[2*ci]),creal(J[2*N+2*ci]),cimag(J[2*N+2*ci]));
    printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(J[2*ci+1]),cimag(J[2*ci+1]),creal(J[2*N+2*ci+1]),cimag(J[2*N+2*ci+1]));
   }
   printf("];\n");
#endif

   
  }
  free(WORK);

#ifdef DEBUG
  printf("Jfinal=[\n");
  for (ci=0; ci<N; ci++) {
    printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(J[2*ci]),cimag(J[2*ci]),creal(J[2*N+2*ci]),cimag(J[2*N+2*ci]));
    printf("%lf+j*(%lf), %lf+j*(%lf)\n",creal(J[2*ci+1]),cimag(J[2*ci+1]),creal(J[2*N+2*ci+1]),cimag(J[2*N+2*ci+1]));
  }
  printf("];\n");
#endif


  /* extract phase only from diagonal elements */
  for (ci=0; ci<N; ci++) {
    J[2*ci]=J[2*ci]/cabs(J[2*ci]);
    J[2*ci+2*N+1]=J[2*ci+2*N+1]/cabs(J[2*ci+2*N+1]);
  }

  /* copy back to output (only the diagonal values) */
  memset(pout,0,sizeof(double)*8*N);
  my_dcopy(N, &Jx[0], 4, &pout[0], 8);
  my_dcopy(N, &Jx[1], 4, &pout[0+1], 8);
  my_dcopy(N, &Jx[4*N+2], 4, &pout[0+6], 8);
  my_dcopy(N, &Jx[4*N+3], 4, &pout[0+7], 8);

  free(J);
  free(Jcopy);
  return 0;
}


/* worker thread function for manifold average+projection 
  project the average back to the solution */
static void*
manifold_average_projectback_threadfn(void *data) {
 thread_data_manavg_t *t=(thread_data_manavg_t*)data;
 int ci,cj,iter;
 double *Yl;
 complex double *J3,*Jp;
 /* local storage 2Nx2 x Nf complex values */
 if ((Yl=(double*)malloc((size_t)t->N*8*t->Nf*sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((J3=(complex double*)malloc((size_t)t->N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((Jp=(complex double*)malloc((size_t)t->N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
#ifdef DEBUG
 complex double *Jerr;
 if ((Jerr=(complex double*)malloc((size_t)t->N*4*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }
#endif

 complex double *Yc=(complex double*)Yl;
 complex double a=1.0/(double)t->Nf+0.0*_Complex_I;

 /* work for SVD */
 complex double *WORK=0;
 complex double w[1];
 double RWORK[32]; /* size > 5*max_matrix_dimension */
 complex double JTJ[4],U[4],VT[4];
 double S[2];
 
 int status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,w,-1,RWORK);
 if (status!=0) {
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
   exit(1);
 } 
 int lwork=(int)w[0];
 if ((WORK=(complex double*)malloc((size_t)(int)lwork*sizeof(complex double)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
 }

 for (ci=t->startM; ci<=t->endM; ci++) {
   /* copy to local storage */
   for (cj=0; cj<t->Nf; cj++) {
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N], 8, &Yl[cj*8*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+1], 8, &Yl[cj*8*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+4], 8, &Yl[cj*8*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+5], 8, &Yl[cj*8*t->N+3], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+2], 8, &Yl[cj*8*t->N+4*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+3], 8, &Yl[cj*8*t->N+4*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+6], 8, &Yl[cj*8*t->N+4*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+7], 8, &Yl[cj*8*t->N+4*t->N+3], 4);

   }
   /* first averaging, select random block in [0,Nf-1] to project to */
   int cr; /* remainder always in [0,Nf-1] */
   if (t->randomize) {
    cr=rand()%(t->Nf); /* remainder always in [0,Nf-1] */
   } else {
    cr=0;
   }
   /* J3 <= cr th  block */
   my_ccopy(t->N*4,&Yc[cr*t->N*4],1,J3,1);
   /* project the remainder */
   for (cj=0; cj<cr; cj++) {
      project_procrustes_block(t->N,J3,&Yc[cj*t->N*4]);
   }
   for (cj=cr+1; cj<t->Nf; cj++) {
      project_procrustes_block(t->N,J3,&Yc[cj*t->N*4]);
   }


   /* now each 2, 2N complex vales is one J block */
   /* average values and project to common average */
   for (iter=0; iter<t->Niter; iter++) {
     /* J3 <= 1st block */
     my_ccopy(t->N*4,Yc,1,J3,1); 
     /* add the remainder */
     for (cj=1; cj<t->Nf; cj++) {
     my_caxpy(t->N*4,&Yc[cj*t->N*4],1.0+_Complex_I*0.0,J3);
     }
     my_cscal(t->N*4,a,J3);
     /* now find unitary matrix using Procrustes problem */
     for (cj=0; cj<t->Nf; cj++) {
       /* find product JTJ = J^H J3 */
       my_zgemm('C','N',2,2,2*t->N,1.0+_Complex_I*0.0,&Yc[cj*t->N*4],2*t->N,J3,2*t->N,0.0+_Complex_I*0.0,JTJ,2);
       status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,WORK,lwork,RWORK);
       //printf("%d %d %lf %lf\n",ci,cj,S[0],S[1]);
       /* find JTJ= U V^H */
       my_zgemm('N','N',2,2,2,1.0+_Complex_I*0.0,U,2,VT,2,0.0+_Complex_I*0.0,JTJ,2);
       /* find J*(JTJ) : projected matrix */
       my_zgemm('N','N',2*t->N,2,2,1.0+_Complex_I*0.0,&Yc[cj*t->N*4],2*t->N,JTJ,2,0.0+_Complex_I*0.0,Jp,2*t->N);
       /* copy back */
       my_ccopy(t->N*4,Jp,1,&Yc[cj*t->N*4],1); 
#ifdef DEBUG
     /* calculate error between projected value and global mean */
     my_ccopy(t->N*4,J3,1,Jerr,1); 
     my_caxpy(t->N*4,&Yc[cj*t->N*4],-1.0+_Complex_I*0.0,Jerr);
     printf("Error freq=%d dir=%d iter=%d %lf\n",cj,ci,iter,my_cnrm2(t->N*4,Jerr));
#endif
     }
   }

   /* now get a fresh copy, because we should modify Y <= (J3 U)
      where U unitary matrix  */
   my_ccopy(t->N*4,Yc,1,J3,1);
   /* add the remainder */
   for (cj=1; cj<t->Nf; cj++) {
      my_caxpy(t->N*4,&Yc[cj*t->N*4],1.0+_Complex_I*0.0,J3);
   }
   my_cscal(t->N*4,a,J3);
   /* now J3 is the average, and project this back to Y */
   for (cj=0; cj<t->Nf; cj++) {
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N], 8, &Yl[cj*8*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+1], 8, &Yl[cj*8*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+4], 8, &Yl[cj*8*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+5], 8, &Yl[cj*8*t->N+3], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+2], 8, &Yl[cj*8*t->N+4*t->N], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+3], 8, &Yl[cj*8*t->N+4*t->N+1], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+6], 8, &Yl[cj*8*t->N+4*t->N+2], 4);
     my_dcopy(t->N, &t->Y[cj*8*t->N*t->M+ci*8*t->N+7], 8, &Yl[cj*8*t->N+4*t->N+3], 4);
   }

   for (cj=0; cj<t->Nf; cj++) {
       /* find product JTJ = J3^H J */
       my_zgemm('C','N',2,2,2*t->N,1.0+_Complex_I*0.0,J3,2*t->N,&Yc[cj*t->N*4],2*t->N,0.0+_Complex_I*0.0,JTJ,2);

       status=my_zgesvd('A','A',2,2,JTJ,2,S,U,2,VT,2,WORK,lwork,RWORK);
       /* find JTJ= U V^H */
       my_zgemm('N','N',2,2,2,1.0+_Complex_I*0.0,U,2,VT,2,0.0+_Complex_I*0.0,JTJ,2);
       /* find J3*(JTJ) : projected matrix */
       my_zgemm('N','N',2*t->N,2,2,1.0+_Complex_I*0.0,J3,2*t->N,JTJ,2,0.0+_Complex_I*0.0,Jp,2*t->N);
       /* copy back */
       my_ccopy(t->N*4,Jp,1,&Yc[cj*t->N*4],1);
   }

   /* copy back from local storage */
   for (cj=0; cj<t->Nf; cj++) {
     my_dcopy(t->N, &Yl[cj*8*t->N], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+1], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+1], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+2], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+4], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+3], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+5], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+2], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N+1], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+3], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N+2], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+6], 8);
     my_dcopy(t->N, &Yl[cj*8*t->N+4*t->N+3], 4, &t->Y[cj*8*t->N*t->M+ci*8*t->N+7], 8);

   }
 }

#ifdef DEBUG
 free(Jerr);
#endif
 free(Yl);
 free(J3);
 free(Jp);
 free(WORK);
 return NULL;
}


int
calculate_manifold_average_projectback(int N,int M,int Nf,double *Y,int Niter,int randomize,int Nt) {
 /* Y : each 2Nx2xM blocks belong to one freq,
   select one 2Nx2 from this, reorder to J format : Nf blocks
   and average */
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_manavg_t *threaddata;

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
  if ((threaddata=(thread_data_manavg_t*)malloc((size_t)Nt*sizeof(thread_data_manavg_t)))==0) {
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
    threaddata[nth].Y=Y;
    threaddata[nth].N=N;
    threaddata[nth].M=M;
    threaddata[nth].Nf=Nf;
    threaddata[nth].Niter=Niter;
    threaddata[nth].startM=ci;
    threaddata[nth].endM=ci+Nthb-1;
    threaddata[nth].randomize=randomize;
    
    pthread_create(&th_array[nth],&attr,manifold_average_projectback_threadfn,(void*)(&threaddata[nth]));
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
