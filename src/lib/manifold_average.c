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

//#define DEBUG
typedef struct thread_data_manavg_ {
 double *Y;
 int startM;
 int endM;
 int Niter;
 int N;
 int M;
 int Nf;
} thread_data_manavg_t;

/* worker thread function for manifold average+projection */
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
   int cr=rand()%(t->Nf); /* remainder always in [0,Nf-1] */
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
calculate_manifold_average(int N,int M,int Nf,double *Y,int Niter,int Nt) {
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
