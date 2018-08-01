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

//#define DEBUG
/* Jones matrix multiplication 
   C=A*B
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
amb(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=a[0]*b[0]+a[1]*b[2];
 c[1]=a[0]*b[1]+a[1]*b[3];
 c[2]=a[2]*b[0]+a[3]*b[2];
 c[3]=a[2]*b[1]+a[3]*b[3];
}

/* Jones matrix multiplication 
   C=A*B^H
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
ambt(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=a[0]*conj(b[0])+a[1]*conj(b[1]);
 c[1]=a[0]*conj(b[2])+a[1]*conj(b[3]);
 c[2]=a[2]*conj(b[0])+a[3]*conj(b[1]);
 c[3]=a[2]*conj(b[2])+a[3]*conj(b[3]);
}

/* Jones matrix multiplication 
   C=A^H*B
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
atmb(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=conj(a[0])*b[0]+conj(a[2])*b[2];
 c[1]=conj(a[0])*b[1]+conj(a[2])*b[3];
 c[2]=conj(a[1])*b[0]+conj(a[3])*b[2];
 c[3]=conj(a[1])*b[1]+conj(a[3])*b[3];
}


/* worker thread function for counting */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
threadfn_fns_fcount(void *data) {
 thread_data_rtr_t *t=(thread_data_rtr_t*)data;
 
 int ci,sta1,sta2;
 for (ci=0; ci<t->Nb; ci++) {
   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;

   t->bcount[sta1]+=1;

   t->bcount[sta2]+=1;
   }
 }

 return NULL;
}


/* function to count how many baselines contribute to the calculation of
   grad and hess, so normalization can be made */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
void
fns_fcount(global_data_rtr_t *gdata) {

  me_data_t *dp=(me_data_t*)gdata->medata;

  int nth,nth1,ci;

  /* no of threads */
  int Nt=(dp->Nt);
  int Nthb0,Nthb;
  thread_data_rtr_t *threaddata;

  int Nbase1=(dp->Nbase)*(dp->tilesz);
  int boff=(dp->Nbase)*(dp->tileoff);

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase1+Nt-1)/Nt;

  if ((threaddata=(thread_data_rtr_t*)malloc((size_t)Nt*sizeof(thread_data_rtr_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }
  int *bcount;
  if ((bcount=(int*)calloc((size_t)dp->N,sizeof(int)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci+boff;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=dp->barr;
    /* note bcount[] should be 0 first */
    if ((threaddata[nth].bcount=(int*)calloc((size_t)dp->N,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    threaddata[nth].mx_array=gdata->mx_array;
    
    pthread_create(&gdata->th_array[nth],&gdata->attr,threadfn_fns_fcount,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(gdata->th_array[nth1],NULL);
   /* add all thread bcount[] arrays into bcount[] */
   for (ci=0; ci<dp->N; ci++) {
    bcount[ci]+=threaddata[nth1].bcount[ci];
   }
   free(threaddata[nth1].bcount);
  }
 free(threaddata);

 /* calculate inverse count */
 for (nth1=0; nth1<dp->N; nth1++) {
   gdata->iw[nth1]=(bcount[nth1]>0?1.0/(double)bcount[nth1]:0.0);
 }
 free(bcount);
 /* scale back weight such that max value is 1 */
 nth1=my_idamax(dp->N,gdata->iw,1);
 double maxw=gdata->iw[nth1-1]; /* 1 based index */
 if (maxw>0.0) { /* all baselines can be flagged */
  my_dscal(dp->N,1.0/maxw,gdata->iw);
 }
}



/* worker thread function for cost function calculation */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
threadfn_fns_f(void *data) {
 thread_data_rtr_t *t=(thread_data_rtr_t*)data;
 
 int ci,cm,sta1,sta2;
 complex double C[4],G1[4],G2[4],T1[4],T2[4];
 int M=(t->M);
 cm=(t->clus);
 for (ci=0; ci<t->Nb; ci++) {
   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
     /* gains for this cluster, for sta1,sta2 */
     G1[0]=(t->x[sta1*2]);
     G1[1]=(t->x[sta1*2+2*t->N]);
     G1[2]=(t->x[sta1*2+1]);
     G1[3]=(t->x[sta1*2+2*t->N+1]);
     G2[0]=(t->x[sta2*2]);
     G2[1]=(t->x[sta2*2+2*t->N]);
     G2[2]=(t->x[sta2*2+1]);
     G2[3]=(t->x[sta2*2+2*t->N+1]);

      /* use pre calculated values */
      C[0]=t->coh[4*M*ci+4*cm];
      C[1]=t->coh[4*M*ci+4*cm+1];
      C[2]=t->coh[4*M*ci+4*cm+2];
      C[3]=t->coh[4*M*ci+4*cm+3];


     /* form G1*C*G2' */
     /* T1=G1*C  */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* add to baseline visibilities V->U */
     double r00=t->y[8*ci]-creal(T2[0]);
     double i00=t->y[8*ci+1]-cimag(T2[0]);
     double r01=t->y[8*ci+2]-creal(T2[1]);
     double i01=t->y[8*ci+3]-cimag(T2[1]);
     double r10=t->y[8*ci+4]-creal(T2[2]);
     double i10=t->y[8*ci+5]-cimag(T2[2]);
     double r11=t->y[8*ci+6]-creal(T2[3]);
     double i11=t->y[8*ci+7]-cimag(T2[3]);

     t->fcost+=r00*r00+i00*i00+r01*r01+i01*i01+r10*r10+i10*i10+r11*r11+i11*i11;
   }
 }

 return NULL;
}


/* cost function */
/* x: 2Nx2 solution
   y: visibilities, vectorized V(:)  8*Nbase x 1
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double
fns_f(complex double *x, double *y,  global_data_rtr_t *gdata) {

  me_data_t *dp=(me_data_t*)gdata->medata;

  int nth,nth1,ci;

  /* no of threads */
  int Nt=(dp->Nt);
  int Nthb0,Nthb;
  thread_data_rtr_t *threaddata;

  int Nbase1=(dp->Nbase)*(dp->tilesz);
  int boff=(dp->Nbase)*(dp->tileoff);

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase1+Nt-1)/Nt;

  if ((threaddata=(thread_data_rtr_t*)malloc((size_t)Nt*sizeof(thread_data_rtr_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci+boff;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=dp->barr;
    threaddata[nth].carr=dp->carr;
    threaddata[nth].M=dp->M;
    threaddata[nth].y=&(y[8*ci]);
    threaddata[nth].N=dp->N;
    threaddata[nth].x=x; /* note the difference: here x assumes no hybrid, also ordering different */
    threaddata[nth].clus=(dp->clus);
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*(ci+boff)]);
    threaddata[nth].fcost=0.0;
    
    //printf("thread %d predict  data from %d baselines %d\n",nth,8*ci,Nthb);
    pthread_create(&gdata->th_array[nth],&gdata->attr,threadfn_fns_f,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  double fcost=0.0;
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(gdata->th_array[nth1],NULL);
   fcost+=threaddata[nth1].fcost;
  }

 free(threaddata);

 return fcost;
}


/* inner product (metric) */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double
fns_g(int N,complex double *x, complex double *eta, complex double *gamma) {
 /* 2 x real( trace(eta'*gamma) )
  = 2 x real( eta(:,1)'*gamma(:,1) + eta(:,2)'*gamma(:,2) )
  no need to calculate off diagonal terms
  )*/
 complex double v1=my_cdot(2*N,eta,gamma);
 complex double v2=my_cdot(2*N,&eta[2*N],&gamma[2*N]);

 return 2.0*creal(v1+v2);
}

/* Projection 
   rnew: new value */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
fns_proj(int N,complex double *x, complex double *z,complex double *rnew) {
  /* projection  = Z-X Om, where
   Om X^H X+X^H X Om = X^H  Z - Z^H X
   is solved to find Om */

  /* find X^H X */
  complex double xx00,xx01,xx10,xx11;
  xx00=my_cdot(2*N,x,x);
  xx01=my_cdot(2*N,x,&x[2*N]);
  xx10=conj(xx01);
  xx11=my_cdot(2*N,&x[2*N],&x[2*N]);

  /* find X^H Z (and using this just calculte Z^H X directly) */
  complex double xz00,xz01,xz10,xz11;
  xz00=my_cdot(2*N,x,z);
  xz01=my_cdot(2*N,x,&z[2*N]);
  xz10=my_cdot(2*N,&x[2*N],z);
  xz11=my_cdot(2*N,&x[2*N],&z[2*N]);

  /* find X^H Z - Z^H X */
  complex double rr00,rr01,rr10,rr11;
  rr00=xz00-conj(xz00);
  rr01=xz01-conj(xz10);
  rr10=-conj(rr01);
  rr11=xz11-conj(xz11);

  /* find I_2 kron (X^H X) + (X^H X)^T kron I_2 */
  /* A = [2*xx00  xx01       xx10         0
          xx10    xx11+xx00  0            xx10
          xx01    0          xx11+xx00    xx01
          0       xx01       xx10         2*xx11 ]
  */
  complex double A[16];
  A[0]=2.0*xx00;
  A[5]=A[10]=xx11+xx00;
  A[15]=2.0*xx11;
  A[1]=A[8]=A[11]=A[13]=xx10;
  A[2]=A[4]=A[7]=A[14]=xx01;
  A[3]=A[6]=A[9]=A[12]=0.0;
  complex double b[4];
  b[0]=rr00;
  b[1]=rr10;
  b[2]=rr01;
  b[3]=rr11;

  /* solve A u = b to find u */
  complex double w,*WORK;
  /* workspace query */
  int status=my_zgels('N',4,4,1,A,4,b,4,&w,-1);
  int lwork=(int)w;
  if ((WORK=(complex double*)calloc((size_t)lwork,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  status=my_zgels('N',4,4,1,A,4,b,4,WORK,lwork);
  if (status) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: LAPACK error %d\n",__FILE__,__LINE__,status);
#endif
  }

  free(WORK);
  /* form Z - X * Om, where Om is given by solution b 
    but no need to rearrange b because it is already in col major order */
  my_ccopy(4*N,z,1,rnew,1);
  my_zgemm('N','N',2*N,2,2,-1.0+0.0*_Complex_I,x,2*N,b,2,1.0+0.0*_Complex_I,rnew,2*N);


}


/* Retraction 
   rnew: new value */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
fns_R(int N,complex double *x, complex double *r,complex double *rnew) {
 /* rnew = x + r */
 my_ccopy(4*N,x,1,rnew,1);
 my_caxpy(4*N,r,1.0+_Complex_I*0.0,rnew);
}


/* worker thread function for gradient/hessian weighting */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
threadfn_fns_fscale(void *data) {
 thread_data_rtr_t *t=(thread_data_rtr_t*)data;
 
 int ci,sta1;
 for (ci=0; ci<t->Nb; ci++) {
   sta1=ci+t->boff;
   t->grad[2*sta1]*=t->iw[sta1];
   t->grad[2*sta1+2*t->N]*=t->iw[sta1];
   t->grad[2*sta1+1]*=t->iw[sta1];
   t->grad[2*sta1+2*t->N+1]*=t->iw[sta1];
 }

 return NULL;
}




/* worker thread function for gradient calculation */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
threadfn_fns_fgrad(void *data) {
 thread_data_rtr_t *t=(thread_data_rtr_t*)data;
 
 int ci,cm,sta1,sta2;
 complex double C[4],G1[4],G2[4],T1[4],T2[4],res[4];
 int M=(t->M);
 cm=(t->clus);
 for (ci=0; ci<t->Nb; ci++) {
   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
     /* gains for this cluster, for sta1,sta2 */
     G1[0]=(t->x[sta1*2]);
     G1[1]=(t->x[sta1*2+2*t->N]);
     G1[2]=(t->x[sta1*2+1]);
     G1[3]=(t->x[sta1*2+2*t->N+1]);
     G2[0]=(t->x[sta2*2]);
     G2[1]=(t->x[sta2*2+2*t->N]);
     G2[2]=(t->x[sta2*2+1]);
     G2[3]=(t->x[sta2*2+2*t->N+1]);

      /* use pre calculated values */
      C[0]=t->coh[4*M*ci+4*cm];
      C[1]=t->coh[4*M*ci+4*cm+1];
      C[2]=t->coh[4*M*ci+4*cm+2];
      C[3]=t->coh[4*M*ci+4*cm+3];


   /* G1*C*G2' */
   amb(G1,C,T1);
   ambt(T1,G2,T2);

   /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
   /* V->U */
   res[0]=(t->y[8*ci]+_Complex_I*t->y[8*ci+1])-T2[0];
   res[1]=(t->y[8*ci+2]+_Complex_I*t->y[8*ci+3])-T2[1];
   res[2]=(t->y[8*ci+4]+_Complex_I*t->y[8*ci+5])-T2[2];
   res[3]=(t->y[8*ci+6]+_Complex_I*t->y[8*ci+7])-T2[3];

   /* 
      grad(2*p-1:2*p,:)=grad(2*p-1:2*p,:)+res*x(2*q-1:2*q,:)*C';
      grad(2*q-1:2*q,:)=grad(2*q-1:2*q,:)+res'*x(2*p-1:2*p,:)*C;
   */
   /* res*G2*C' */
   amb(res,G2,T1);
   ambt(T1,C,T2);

   pthread_mutex_lock(&t->mx_array[sta1]);
   t->grad[2*sta1]+=T2[0];
   t->grad[2*sta1+2*t->N]+=T2[1];
   t->grad[2*sta1+1]+=T2[2];
   t->grad[2*sta1+2*t->N+1]+=T2[3];
   pthread_mutex_unlock(&t->mx_array[sta1]);

   /* res'*G1*C */
   atmb(res,G1,T1);
   amb(T1,C,T2);

   pthread_mutex_lock(&t->mx_array[sta2]);
   t->grad[2*sta2]+=T2[0];
   t->grad[2*sta2+2*t->N]+=T2[1];
   t->grad[2*sta2+1]+=T2[2];
   t->grad[2*sta2+2*t->N+1]+=T2[3];
   pthread_mutex_unlock(&t->mx_array[sta2]);

   }
 }

 return NULL;
}



/* gradient function */
/* x: 2Nx2 solution
   fgradx: output, same shape as x
   y: visibilities, vectorized V(:)  8*Nbase x 1
   if negate==1, return -grad, else just grad
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
fns_fgrad(complex double *x, complex double *fgradx, double *y,  global_data_rtr_t *gdata, int negate) {

  me_data_t *dp=(me_data_t*)gdata->medata;

  int nth,nth1,ci;

  /* no of threads */
  int Nt=(dp->Nt);
  int Nthb0,Nthb;
  thread_data_rtr_t *threaddata;

  int Nbase1=(dp->Nbase)*(dp->tilesz);
  int boff=(dp->Nbase)*(dp->tileoff);

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase1+Nt-1)/Nt;

  if ((threaddata=(thread_data_rtr_t*)malloc((size_t)Nt*sizeof(thread_data_rtr_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }
  complex double *grad;
  if ((grad=(complex double*)calloc((size_t)4*dp->N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }


  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci+boff;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=dp->barr;
    threaddata[nth].carr=dp->carr;
    threaddata[nth].M=dp->M;
    threaddata[nth].y=&(y[8*ci]);
    threaddata[nth].N=dp->N;
    threaddata[nth].x=x; /* note the difference: here x assumes no hybrid, also ordering different */
    threaddata[nth].clus=(dp->clus);
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*(ci+boff)]);
    threaddata[nth].grad=grad;
    threaddata[nth].mx_array=gdata->mx_array;
    
    //printf("thread %d predict  data from %d baselines %d\n",nth,8*ci,Nthb);
    pthread_create(&gdata->th_array[nth],&gdata->attr,threadfn_fns_fgrad,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(gdata->th_array[nth1],NULL);
  }
/******************* scale *************/
   Nthb0=(dp->N+Nt-1)/Nt;
   ci=0;
   for (nth=0;  nth<Nt && ci<dp->N; nth++) {
    if (ci+Nthb0<dp->N) {
     Nthb=Nthb0;
    } else {
     Nthb=dp->N-ci;
    }
    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].N=dp->N;
    threaddata[nth].grad=grad;
    threaddata[nth].iw=gdata->iw;
    pthread_create(&gdata->th_array[nth],&gdata->attr,threadfn_fns_fscale,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(gdata->th_array[nth1],NULL);
  }
/******************* scale *************/
 free(threaddata);
 if (negate) {
  my_cscal(4*dp->N,-1.0+0.0*_Complex_I,grad);
 }
 fns_proj(dp->N,x,grad,fgradx);
 free(grad);

}


/* worker thread function for Hessian calculation */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
threadfn_fns_fhess(void *data) {
 thread_data_rtr_t *t=(thread_data_rtr_t*)data;
 
 int ci,cm,sta1,sta2;
 complex double C[4],G1[4],G2[4],T1[4],T2[4],res[4],res1[4],E1[4],E2[4];
 int M=(t->M);
 cm=(t->clus);
 for (ci=0; ci<t->Nb; ci++) {
   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {

   /* stations for this baseline */
   sta1=t->barr[ci+t->boff].sta1;
   sta2=t->barr[ci+t->boff].sta2;
     /* gains for this cluster, for sta1,sta2 */
     G1[0]=(t->x[sta1*2]);
     G1[1]=(t->x[sta1*2+2*t->N]);
     G1[2]=(t->x[sta1*2+1]);
     G1[3]=(t->x[sta1*2+2*t->N+1]);
     G2[0]=(t->x[sta2*2]);
     G2[1]=(t->x[sta2*2+2*t->N]);
     G2[2]=(t->x[sta2*2+1]);
     G2[3]=(t->x[sta2*2+2*t->N+1]);
     E1[0]=t->eta[2*sta1];
     E1[1]=t->eta[2*sta1+2*t->N];
     E1[2]=t->eta[2*sta1+1];
     E1[3]=t->eta[2*sta1+2*t->N+1];
     E2[0]=t->eta[2*sta2];
     E2[1]=t->eta[2*sta2+2*t->N];
     E2[2]=t->eta[2*sta2+1];
     E2[3]=t->eta[2*sta2+2*t->N+1];



      /* use pre calculated values */
      C[0]=t->coh[4*M*ci+4*cm];
      C[1]=t->coh[4*M*ci+4*cm+1];
      C[2]=t->coh[4*M*ci+4*cm+2];
      C[3]=t->coh[4*M*ci+4*cm+3];

   /* G1*C*G2' */
   amb(G1,C,T1);
   ambt(T1,G2,T2);


   /* res=V(2*ck-1:2*ck,:)-x(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)'; */
   /* V->U */
   res[0]=(t->y[8*ci]+_Complex_I*t->y[8*ci+1])-T2[0];
   res[1]=(t->y[8*ci+2]+_Complex_I*t->y[8*ci+3])-T2[1];
   res[2]=(t->y[8*ci+4]+_Complex_I*t->y[8*ci+5])-T2[2];
   res[3]=(t->y[8*ci+6]+_Complex_I*t->y[8*ci+7])-T2[3];


   /*
      res1=x(2*p-1:2*p,:)*C*eta(2*q-1:2*q,:)'+eta(2*p-1:2*p,:)*C*x(2*q-1:2*q,:)';
   */
   /* G1*C*E2' */
   amb(G1,C,T1);
   ambt(T1,E2,T2);
   res1[0]=T2[0];
   res1[1]=T2[1];
   res1[2]=T2[2];
   res1[3]=T2[3];
   /* E1*C*G2' */
   amb(E1,C,T1);
   ambt(T1,G2,T2);
   res1[0]+=T2[0];
   res1[1]+=T2[1];
   res1[2]+=T2[2];
   res1[3]+=T2[3];


   /* 
      hess(2*p-1:2*p,:)=hess(2*p-1:2*p,:)+(res*eta(2*q-1:2*q,:)-res1*x(2*q-1:2*q,:))*C';
   */

   /* (res*E2-res1*G2)*C' */
   amb(res,E2,T1);
   amb(res1,G2,T2);
   T1[0]-=T2[0];
   T1[1]-=T2[1];
   T1[2]-=T2[2];
   T1[3]-=T2[3];
   ambt(T1,C,T2);

   pthread_mutex_lock(&t->mx_array[sta1]);
   t->hess[2*sta1]+=T2[0];
   t->hess[2*sta1+2*t->N]+=T2[1];
   t->hess[2*sta1+1]+=T2[2];
   t->hess[2*sta1+2*t->N+1]+=T2[3];
   pthread_mutex_unlock(&t->mx_array[sta1]);


   /* 
      hess(2*q-1:2*q,:)=hess(2*q-1:2*q,:)+(res'*eta(2*p-1:2*p,:)-res1'*x(2*p-1:2*p,:))*C;
   */

   /* (res'*E1-res1'*G1)*C */
   atmb(res,E1,T1);
   atmb(res1,G1,T2);
   T1[0]-=T2[0];
   T1[1]-=T2[1];
   T1[2]-=T2[2];
   T1[3]-=T2[3];
   amb(T1,C,T2);

   pthread_mutex_lock(&t->mx_array[sta2]);
   t->hess[2*sta2]+=T2[0];
   t->hess[2*sta2+2*t->N]+=T2[1];
   t->hess[2*sta2+1]+=T2[2];
   t->hess[2*sta2+2*t->N+1]+=T2[3];
   pthread_mutex_unlock(&t->mx_array[sta2]);

   }
 }

 return NULL;
}



/* Hessian function */
/* x: 2Nx2 solution
   eta: same shape as x
   fhess: output, same shape as x
   y: visibilities, vectorized V(:)  8*Nbase x 1
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
fns_fhess(complex double *x, complex double *eta,complex double *fhess, double *y,  global_data_rtr_t *gdata) {

  me_data_t *dp=(me_data_t*)gdata->medata;

  int nth,nth1,ci;

  /* no of threads */
  int Nt=(dp->Nt);
  int Nthb0,Nthb;
  thread_data_rtr_t *threaddata;

  int Nbase1=(dp->Nbase)*(dp->tilesz);
  int boff=(dp->Nbase)*(dp->tileoff);

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase1+Nt-1)/Nt;

  if ((threaddata=(thread_data_rtr_t*)malloc((size_t)Nt*sizeof(thread_data_rtr_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }
  complex double *hess;
  if ((hess=(complex double*)calloc((size_t)4*dp->N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }


  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }

    threaddata[nth].boff=ci+boff;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=dp->barr;
    threaddata[nth].carr=dp->carr;
    threaddata[nth].M=dp->M;
    threaddata[nth].y=&(y[8*ci]);
    threaddata[nth].N=dp->N;
    threaddata[nth].x=x; /* note the difference: here x assumes no hybrid, also ordering different */
    threaddata[nth].clus=(dp->clus);
    threaddata[nth].coh=&(dp->coh[4*(dp->M)*(ci+boff)]);
    threaddata[nth].eta=eta;
    threaddata[nth].hess=hess;
    threaddata[nth].mx_array=gdata->mx_array;
    
    //printf("thread %d predict  data from %d baselines %d\n",nth,8*ci,Nthb);
    pthread_create(&gdata->th_array[nth],&gdata->attr,threadfn_fns_fhess,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(gdata->th_array[nth1],NULL);
  }

/******************* scale *************/
   Nthb0=(dp->N+Nt-1)/Nt;
   ci=0;
   for (nth=0;  nth<Nt && ci<dp->N; nth++) {
    if (ci+Nthb0<dp->N) {
     Nthb=Nthb0;
    } else {
     Nthb=dp->N-ci;
    }
    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].N=dp->N;
    threaddata[nth].grad=fhess;
    threaddata[nth].iw=gdata->iw;
    pthread_create(&gdata->th_array[nth],&gdata->attr,threadfn_fns_fscale,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }

  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(gdata->th_array[nth1],NULL);
  }
/******************* scale *************/
 free(threaddata);
 fns_proj(dp->N,x,hess,fhess);
 free(hess);

}


/* truncated conjugate gradient method 
  x, grad, eta, r, z, delta, Hxd  : size 2N x 2  complex 
  so, vector size is 4N complex double

  output: eta
  output: fhess (can be reused in calling func)
  return value: stop_tCG code   

  y: vec(V) visibilities
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
tcg_solve(int N, complex double *x, complex double *grad, complex double *eta, complex double *fhess,
 double Delta, double theta, double kappa, int max_inner, int min_inner, double *y, global_data_rtr_t *gdata) {

  complex double *r,*z,*delta,*Hxd, *rnew;
  double e_Pe, r_r, norm_r, z_r, d_Pd, d_Hd, alpha, e_Pe_new,
     e_Pd, Deltasq, tau, zold_rold, beta, norm_r0;
  int cj, stop_tCG;

  if ((r=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((z=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((delta=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Hxd=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((rnew=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
 /*
  initial values
 % eta = 0*grad; << zero matrix provided
 r = grad;
 e_Pe = 0;
 */
  my_ccopy(4*N,grad,1,r,1);
  e_Pe=0.0;

 /* 
   r_r = fns.g(x,r,r);
   norm_r = sqrt(r_r);
   norm_r0 = norm_r;
 */

  r_r=fns_g(N,x,r,r);
  norm_r=sqrt(r_r);
  norm_r0=norm_r;

 /*
  z = r;
 */
 my_ccopy(4*N,r,1,z,1);

 /*
  % compute z'*r
  z_r = fns.g(x,z,r);
  d_Pd = z_r;
 */
  z_r=fns_g(N,x,z,r);
  d_Pd=z_r;

 /*
   % initial search direction
   delta  = -z;
   e_Pd = fns.g(x,eta,delta);
 */
  memset(delta,0,sizeof(complex double)*N*4);  
  my_caxpy(4*N,z,-1.0+_Complex_I*0.0,delta);
  e_Pd=fns_g(N,x,eta,delta);

 /*
   % pre-assume termination b/c j == end
   stop_tCG = 5;
 */
  stop_tCG=5;

 /* % begin inner/tCG loop
    for j = 1:max_inner,
 */
 for(cj=1; cj<=max_inner; cj++) { 
 /**************************************************/
 /* 
     Hxd = fns.fhess(x,delta);

     % compute curvature
     d_Hd = fns.g(x,delta,Hxd);
 */
    fns_fhess(x,delta,Hxd,y,gdata);
    d_Hd=fns_g(N,x,delta,Hxd);

 /*
      alpha = z_r/d_Hd;
      e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;
 */
     alpha=z_r/d_Hd;
     e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;

 /*

      % check curvature and trust-region radius
      if d_Hd <= 0 || e_Pe_new >= Delta^2,

 */
      Deltasq=Delta*Delta;
      if (d_Hd <= 0.0 || e_Pe_new >= Deltasq) {
  /*
         tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd;

  */
         tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Deltasq-e_Pe)))/d_Pd;
  
  /*
       eta = eta + tau*delta;

  */
       my_caxpy(4*N,delta,tau+_Complex_I*0.0,eta);

  /* NEW  Heta = Heta + tau*Hdelta */
      my_caxpy(4*N,fhess,tau+_Complex_I*0.0,Hxd);

  /*
        if d_Hd <= 0,
            stop_tCG = 1;     % negative curvature
         else
            stop_tCG = 2;     % exceeded trust region
         end
   */
       stop_tCG=(d_Hd<=0.0?1:2);

   /* 
      break (for)  
   */
       break;
   /* 
     end if 
   */
     }


   /*
      % no negative curvature and eta_prop inside TR: accept it
      e_Pe = e_Pe_new;
      eta = eta + alpha*delta;

   */
      e_Pe=e_Pe_new;
      my_caxpy(4*N,delta,alpha+_Complex_I*0.0,eta);

    /* NEW Heta = Heta + alpha*Hdelta */
      my_caxpy(4*N,fhess,alpha+_Complex_I*0.0,Hxd);


   /*
      % update the residual
      r = r + alpha*Hxd;

   */
      my_caxpy(4*N,Hxd,alpha+_Complex_I*0.0,r);

   /*
      % compute new norm of r
      r_r = fns.g(x,r,r);
      norm_r = sqrt(r_r);

   */
      r_r=fns_g(N,x,r,r);
      norm_r=sqrt(r_r);


   /*
      % check kappa/theta stopping criterion
      if j >= min_inner && norm_r <= norm_r0*min(norm_r0^theta,kappa)
   */
      if (cj >= min_inner) {
      double norm_r0pow=pow(norm_r0,theta);
      if (norm_r <= norm_r0*MIN(norm_r0pow,kappa)) {

   /*
         % residual is small enough to quit
         if kappa < norm_r0^theta,
             stop_tCG = 3;  % linear convergence
         else
             stop_tCG = 4;  % superlinear convergence
         end

   */
     stop_tCG=(kappa<norm_r0pow?3:4);

   /* 
     break (for);
   */
     break; 
   /* 
     end (if)
   */
    }
    }


   /*
     % precondition the residual
     z = r;

   */
    my_ccopy(4*N,r,1,z,1);
   /*
      % save the old z'*r
      zold_rold = z_r;
   
   */
   zold_rold=z_r;

   /*
     % compute new z'*r
      z_r = fns.g(x,z,r);
   */
     z_r=fns_g(N,x,z,r);

   /*
      % compute new search direction
      beta = z_r/zold_rold;
      delta = -z + beta*delta;

   */
     beta=z_r/zold_rold;
     my_cscal(4*N,beta,delta);
     my_caxpy(4*N,z,-1.0+_Complex_I*0.0,delta);

   /*
      % update new P-norms and P-dots
      e_Pd = beta*(e_Pd + alpha*d_Pd);
      d_Pd = z_r + beta*beta*d_Pd;

   */
      e_Pd = beta*(e_Pd + alpha*d_Pd);
      d_Pd = z_r + beta*beta*d_Pd;

   /* 
    end for loop 
   */
   }
   /**************************************************/

  free(r);
  free(z);
  free(delta);
  free(Hxd);
  free(rnew);
  return stop_tCG;
}


/* Armijo step calculation,
  output teta: Armijo gradient 
return value: 0 : cost reduced, 1: no cost reduction, so do not run again 
  mincost: minimum value of cost found, if possible
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
armijostep(int N,complex double *x,complex double *teta, double *y, global_data_rtr_t *gdata, complex double *eta, complex double *x_prop, double *mincost) {
 double alphabar=10.0;
 double beta=0.2;
 double sigma=0.5;

 me_data_t *dp=(me_data_t*)gdata->medata;

 double fx=fns_f(x,y,gdata);
 fns_fgrad(x,eta,y,gdata,0);
 double beta0=beta;
 double minfx=fx; double minbeta=beta0;
 double lhs,rhs,metric;
 int m,nocostred=0;
 *mincost=fx;
 double metric0=fns_g(dp->N,x,eta,eta);
 for (m=0; m<50; m++) {
   /* abeta=(beta0)*alphabar*eta; */
   my_ccopy(4*dp->N,eta,1,teta,1);
   my_cscal(4*dp->N,beta0*alphabar+0.0*_Complex_I,teta);
   /* Rx=R(x,teta); */
   fns_R(dp->N,x,teta,x_prop);
   lhs=fns_f(x_prop,y,gdata);
   if (lhs<minfx) {
     minfx=lhs;
     *mincost=minfx;
     minbeta=beta0;
   }
   metric=beta0*alphabar*metric0;
   //rhs=fx+sigma*fns_g(dp->N,x,eta,teta);
   rhs=fx+sigma*metric;
   /* break loop also if no further cost improvement is seen */
   if (lhs<=rhs) {
    minbeta=beta0;
    break;
   }
   beta0=beta0*beta;
 }

 /* if no further cost improvement is seen */
 if (lhs>fx) {
     nocostred=1;
 }

 my_ccopy(4*dp->N,eta,1,teta,1);
 my_cscal(4*dp->N,minbeta*alphabar+0.0*_Complex_I,teta);

 return nocostred;
}


int
rtr_solve_nocuda(
  double *x0,         /* initial values and updated solution at output (size 8*N double) */
  double *y,         /* data vector (size 8*M double) */
  int N,              /* no. of stations */
  int M,              /* no. of constraints */
  int itmax_rsd,          /* maximum number of iterations RSD */
  int itmax_rtr,          /* maximum number of iterations RTR */
  double Delta_bar, double Delta0, /* Trust region radius and initial value */
  double *info, /* initial and final residuals */
  me_data_t *adata) { /* pointer to additional data */

  /* reshape x to make J: 2Nx2 complex double 
  */
  complex double *x;
  if ((x=(complex double*)malloc((size_t)4*N*sizeof(complex double)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  /* map x: [(re,im)J_1(0,0) (re,im)J_1(0,1) (re,im)J_1(1,0) (re,im)J_1(1,1)...]
   to
  J: [J_1(0,0) J_1(1,0) J_2(0,0) J_2(1,0) ..... J_1(0,1) J_1(1,1) J_2(0,1) J_2(1,1)....]
 */
  double *Jd=(double*)x;
  /* re J(0,0) */
  my_dcopy(N, &x0[0], 8, &Jd[0], 4);
  /* im J(0,0) */
  my_dcopy(N, &x0[1], 8, &Jd[1], 4);
  /* re J(1,0) */
  my_dcopy(N, &x0[4], 8, &Jd[2], 4);
  /* im J(1,0) */
  my_dcopy(N, &x0[5], 8, &Jd[3], 4);
  /* re J(0,1) */
  my_dcopy(N, &x0[2], 8, &Jd[4*N], 4);
  /* im J(0,1) */
  my_dcopy(N, &x0[3], 8, &Jd[4*N+1], 4);
  /* re J(1,1) */
  my_dcopy(N, &x0[6], 8, &Jd[4*N+2], 4);
  /* im J(1,1) */
  my_dcopy(N, &x0[7], 8, &Jd[4*N+3], 4);



  int Nt=adata->Nt;
  int ci;
  global_data_rtr_t gdata;

  gdata.medata=adata;
  /* setup threads */
  pthread_attr_init(&gdata.attr);
  pthread_attr_setdetachstate(&gdata.attr,PTHREAD_CREATE_JOINABLE);

  if ((gdata.th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  
  if ((gdata.mx_array=(pthread_mutex_t*)malloc((size_t)N*sizeof(pthread_mutex_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((gdata.iw=(double*)malloc((size_t)N*sizeof(double)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }

  for (ci=0; ci<N; ci++) {
   pthread_mutex_init(&gdata.mx_array[ci],NULL);
  }
 /* count baseline->station contributions 
   NOTE: has to be done here because the baseline offset would change */
 fns_fcount(&gdata);
/***************************************************/
 int min_inner,max_inner,min_outer,max_outer;
 double epsilon,kappa,theta,rho_prime;
 /*
 min_inner =   0;
 max_inner = inf;
 min_outer =   3;
 max_outer = 100;
 epsilon   =   1e-6;
 kappa     =   0.1;
 theta     =   1.0;
 rho_prime =   0.1;
 %Delta_bar =  user must specify
 %Delta0    =  user must specify
 %x0        =  user must specify
 */
 min_inner=1; max_inner=itmax_rtr;//8*N;
 min_outer=3; max_outer=itmax_rtr;
 epsilon=CLM_EPSILON;
 kappa=0.1;
 theta=1.0;
 /* default values 0.25, 0.75, 0.25, 2.0 */
 double eta1=0.0001; double eta2=0.99; double alpha1=0.25; double alpha2=3.5;
 rho_prime=eta1; /* default 0.1 should be <= 0.25 */
 double  rho_regularization; /* default 1e2 use large damping (but less than GPU version) */
 double rho_reg;
 int model_decreased=0;

 complex double *fgradx,*eta,*Heta,*x_prop;
 if ((fgradx=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
 }
 if ((eta=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
 }
 if ((Heta=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
 }
 if ((x_prop=(complex double*)calloc((size_t)4*N,sizeof(complex double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
 }


 double fx,norm_grad,Delta,fx_prop,rhonum,rhoden,rho;
 fx=fns_f(x,y,&gdata);
 double fx0=fx;
 int rsdstat=0;
/***************************************************/
 /* RSD solution */
 for (ci=0; ci<itmax_rsd; ci++) {
  /* Armijo step */
  /* teta=armijostep(V,C,N,x); */
  rsdstat=armijostep(N,x,eta,y,&gdata,fgradx,x_prop,&fx); /* NOTE last two are just storage */
  /* x=R(x,teta); */
  fns_R(N,x,eta,x_prop);
  if(!rsdstat) { 
   my_ccopy(4*N,x_prop,1,x,1);
  } else {/* no cost reduction, break loop */
    break;
  }
 }

 Delta_bar=MIN(fx,0.01);
 Delta0=Delta_bar*0.125;
 rho_regularization=fx*1e-6;
//printf("fx=%g Delta_bar=%g Delta0=%g\n",fx,Delta_bar,Delta0);

/***************************************************/
 /*
 % initialize counters/sentinals
 % allocate storage for dist, counters
 k = 0;  % counter for outer (TR) iteration.
 stop_outer = 0;  % stopping criterion for TR.
 
 x = x0;
fx = fns.f(x);
fgradx = fns.fgrad(x);
norm_grad = sqrt(fns.g(x,fgradx,fgradx));

% initialize trust-region radius
Delta = Delta0;
  */
  int k=0;
  int stop_outer=(itmax_rtr>0?0:1);
  int stop_inner=0;
  // x0 is already copied to x:  my_ccopy(4*N,x0,1,x,1); 
  if(!stop_outer) {
   fns_fgrad(x,fgradx,y,&gdata,1);
   norm_grad = sqrt(fns_g(N,x,fgradx,fgradx));
  }
  Delta=Delta0;

  /* initial residual */
  info[0]=fx;

  /*
   % ** Start of TR loop **
   while stop_outer==0,
  */
   while(!stop_outer) {
   /*  
    % update counter
    k = k+1;
   */
    k++;


    /*
     ** Begin TR Subproblem **
     % determine eta0
      % without randT, 0*fgradx is the only way that we 
      % know how to create a tangent vector
      eta = 0*fgradx;
    */
     memset(eta,0,sizeof(complex double)*N*4);

     
     /*
      % solve TR subproblem
      [eta,numit,stop_inner] = tCG(fns,x,fgradx,eta,Delta,theta,kappa,min_inner,max_inner,useRand,debug);
     */
     stop_inner=tcg_solve(N, x, fgradx, eta, Heta, Delta, theta, kappa, max_inner, min_inner,y,&gdata);

     /*
      norm_eta = sqrt(fns.g(x,eta,eta));
     */
   
     /*
        Heta = fns.fhess(x,eta);
     */
      //OLD fns_fhess(x,eta,Heta,y,&gdata);

     /*
         % compute the retraction of the proposal
   x_prop  = fns.R(x,eta);
      */
      fns_R(N,x,eta,x_prop);

     /*
         % compute function value of the proposal
   fx_prop = fns.f(x_prop);
     */
     fx_prop=fns_f(x_prop,y,&gdata);
  
     /*
          % do we accept the proposed solution or not?
   % compute the Hessian at the proposal
   Heta = fns.fhess(x,eta);
      FIXME: do we need to do this, because Heta is already there
      or change x to x_prop ???
     */
      //Disabled fns_fhess(x,eta,Heta,y,&gdata);

     /*
         % check the performance of the quadratic model
   rhonum = fx-fx_prop;
   rhoden = -fns.g(x,fgradx,eta) - 0.5*fns.g(x,Heta,eta);
     */
      rhonum=fx-fx_prop;
      rhoden=-fns_g(N,x,fgradx,eta)-0.5*fns_g(N,x,Heta,eta);
    /* regularization of rho ratio */
    /* 
    rho_reg = max(1, abs(fx)) * eps * options.rho_regularization;
    rhonum = rhonum + rho_reg;
    rhoden = rhoden + rho_reg;
    */
    rho_reg=MAX(1.0,fx)*rho_regularization; /* no epsilon */
    rhonum+=rho_reg;
    rhoden+=rho_reg;


     /*
        rho =   rhonum  / rhoden;
     */
      rho=rhonum/rhoden;

  

     /*
           % HEURISTIC WARNING:
   % if abs(model change) is relatively zero, we are probably near a critical
   % point. set rho to 1.
   if abs(rhonum/fx) < sqrt(eps),
      small_rhonum = rhonum;
      rho = 1;
   else
      small_rhonum = 0;
   end
     FIXME: use constant for sqrt(eps) 
      */
  /* OLD CODE if (fabs(rhonum/fx) <sqrt_eps) {
     rho=1.0;
   } else {
   } */
     /* model_decreased = (rhoden >= 0); */
     model_decreased=(rhoden>=0.0?1:0);

    /* NOTE: if too many values of rho are -ve, it means TR radius is too big
       so initial TR radius should be reduced */
#ifdef DEBUG
    printf("stop_inner=%d rho_reg=%g rho =%g/%g= %g rho'= %g\n",stop_inner,rho_reg,rhonum,rhoden,rho,rho_prime);
#endif
   
     /*
       % choose new TR radius based on performance
   if rho < 1/4
      Delta = 1/4*Delta;
   elseif rho > 3/4 && (stop_inner == 2 || stop_inner == 1),
      Delta = min(2*Delta,Delta_bar);
   end
      */
    if (!model_decreased || rho<eta1) {
      /* TR radius is too big, reduce it */
      Delta=alpha1*Delta;
    } else if (rho>eta2 && (stop_inner==2 || stop_inner==1)) {
      /* we have a good reduction, so increase TR radius */
      Delta=MIN(alpha2*Delta,Delta_bar);
    }

     /*
          % choose new iterate based on performance
   oldgradx = fgradx;
   if rho > rho_prime,
      accept = true;
      x    = x_prop;
      fx   = fx_prop;
      fgradx = fns.fgrad(x);
      norm_grad = sqrt(fns.g(x,fgradx,fgradx));
   else
      accept = false;
   end
     */
     if (model_decreased && rho>rho_prime) {
      my_ccopy(4*N,x_prop,1,x,1);
      fx=fx_prop;
      fns_fgrad(x,fgradx,y,&gdata,1);
      norm_grad=sqrt(fns_g(N,x,fgradx,fgradx));
     } 


     /*
          % ** Testing for Stop Criteria
   % min_outer is the minimum number of inner iterations
   % before we can exit. this gives randomization a chance to
   % escape a saddle point.
   if norm_grad < epsilon && (~useRand || k > min_outer),
      stop_outer = 1;
   end
      */
   if (norm_grad<epsilon && k>min_outer) {
      stop_outer=1;
   }


     /*
          % stop after max_outer iterations
   if k >= max_outer,
      if (verbosity > 0),
         fprintf('\n*** timed out -- k == %d***\n',k);
      end
      stop_outer = 1;
   end
      */
    if (k>=max_outer) {
      stop_outer=1;
    }

#ifdef DEBUG
    printf("Iter %d cost=%lf\n",k,fx);
#endif
    /* end  of TR loop (counter: k) */
    }

   /* final residual */
   info[1]=fx;

   free(fgradx);
   free(eta);
   free(Heta);
   free(x_prop);
/***************************************************/
  for (ci=0; ci<N; ci++) {
   pthread_mutex_destroy(&gdata.mx_array[ci]);
  }
  pthread_attr_destroy(&gdata.attr);
  free(gdata.th_array);
  free(gdata.mx_array);
  free(gdata.iw);

  if (fx0>fx) { 
  /* copy back solution to x0 */
  /* re J(0,0) */
  my_dcopy(N, &Jd[0], 4, &x0[0], 8);
  /* im J(0,0) */
  my_dcopy(N, &Jd[1], 4, &x0[1], 8);
  /* re J(1,0) */
  my_dcopy(N, &Jd[2], 4, &x0[4], 8);
  /* im J(1,0) */
  my_dcopy(N, &Jd[3], 4, &x0[5], 8);
  /* re J(0,1) */
  my_dcopy(N, &Jd[4*N], 4, &x0[2], 8);
  /* im J(0,1) */
  my_dcopy(N, &Jd[4*N+1], 4, &x0[3], 8);
  /* re J(1,1) */
  my_dcopy(N, &Jd[4*N+2], 4, &x0[6], 8);
  /* im J(1,1) */
  my_dcopy(N, &Jd[4*N+3], 4, &x0[7], 8);
  }

  free(x);
  return 0;
}
