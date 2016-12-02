/*
 *
 Copyright (C) 2006-2008 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <pthread.h>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

/**** repeated code here ********************/
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

/**** end repeated code ********************/
/***************************************************************/
/* worker thread to calculate
   sum ( log(1+ (y_i-f_i)^2/nu) )
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
func_robust_th(void *data) {
  thread_data_logf_t *t=(thread_data_logf_t*)data;
  double inv_nu=1.0/t->nu;
  t->sum=0.0;
  int ci;
  double err;
  for (ci=t->start; ci<=t->end; ci++) {
    err=t->x[ci]-t->f[ci]; 
    err=err*err*inv_nu;
    t->sum+=log(1.0+err);
  } 
  return NULL;
}
/* recalculate log(1+ (y_i-f_i)^2/nu) 
   from function() that calculates f_i 
   y (data)
   f=function()
   x=log(1+(y_i-f_i)^2/nu) output
   all size n x 1
   Nt: no of threads

   return sum(log(..))
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double
func_robust(
   double *f, double *y, int n, double robust_nu, int Nt) {

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_logf_t *threaddata;
  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_logf_t*)malloc((size_t)Nt*sizeof(thread_data_logf_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  int ci,nth,Nparm;
  Nparm=(n+Nt-1)/Nt;

  ci=0;
  for (nth=0; nth<Nt; nth++) {
    threaddata[nth].f=f; /* model */
    threaddata[nth].x=y; /* data */
    threaddata[nth].nu=robust_nu;
    threaddata[nth].start=ci;
    threaddata[nth].sum=0.0;
    threaddata[nth].end=ci+Nparm-1;
    if (threaddata[nth].end >=n) {
     threaddata[nth].end=n-1;
    }
    ci=ci+Nparm; 
    pthread_create(&th_array[nth],&attr,func_robust_th,(void*)(&threaddata[nth]));

  }
  /* now wait for threads to finish */
  double mysum=0.0;
  for(nth=0; nth<Nt; nth++) {
   pthread_join(th_array[nth],NULL);
   mysum+=threaddata[nth].sum;
  }
  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);

  return mysum;
}
/***************************************************************/

/* use algorithm 9.1 to compute pk=Hk gk */
/* pk,gk: size m x 1
   s, y: size mM x 1 
   rho: size M x 1 
   ii: true location of the k th values in s,y */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void
mult_hessian(int m, double *pk, double *gk, double *s, double *y, double *rho, int M, int ii) {
 int ci;
 double *alphai;
 int *idx; /* store sorted locations of s, y here */
 double gamma,beta;

 if ((alphai=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
 }
 if ((idx=(int*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
 }
 if (M>0) {
  /* find the location of k-1 th value */
  if (ii>0) {
   ii=ii-1;
  } else {
   ii=M-1;
  }
 /* s,y will have 0,1,...,ii,ii+1,...M-1 */
 /* map this to  ii+1,ii+2,...,M-1,0,1,..,ii */
  for (ci=0; ci<M-ii-1; ci++){
   idx[ci]=(ii+ci+1);
  }
  for(ci=M-ii-1; ci<M; ci++) {
   idx[ci]=(ci-M+ii+1);
  }
 }

#ifdef DEBUG
 printf("prod M=%d, current ii=%d\n",M,ii);
 for(ci=0; ci<M; ci++) {
  printf("%d->%d ",ci,idx[ci]);
 }
 printf("\n");
#endif
 /* q = grad(f)k : pk<=gk */
 my_dcopy(m,gk,1,pk,1);
 /* this should be done in the right order */
 for (ci=0; ci<M; ci++) {
  /* alphai=rhoi si^T*q */
  alphai[M-ci-1]=rho[idx[M-ci-1]]*my_ddot(m,&s[m*idx[M-ci-1]],pk);
  /* q=q-alphai yi */
  my_daxpy(m,&y[m*idx[M-ci-1]],-alphai[M-ci-1],pk);
 }
 /* r=Hk(0) q : initial hessian */
 /* gamma=s(k-1)^T*y(k-1)/y(k-1)^T*y(k-1)*/
 gamma=1.0;
 if (M>0) {
  gamma=my_ddot(m,&s[m*idx[M-1]],&y[m*idx[M-1]]);
  gamma/=my_ddot(m,&y[m*idx[M-1]],&y[m*idx[M-1]]);
  /* Hk(0)=gamma I, so scale q by gamma */
  /* r= Hk(0) q */
  my_dscal(m,gamma,pk);
 } 

 for (ci=0; ci<M; ci++) {
  /* beta=rhoi yi^T * r */
  beta=rho[idx[ci]]*my_ddot(m,&y[m*idx[ci]],pk);
  /* r = r + (alphai-beta)*si */
  my_daxpy(m,&s[m*idx[ci]],alphai[ci]-beta,pk);
 }

 free(alphai);
 free(idx);
}

/* cubic interpolation in interval [a,b] (a>b is possible)
   to find step that minimizes cost function */
/* func: vector function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   a/b:  interval for interpolation
   x: size n x 1 (storage)
   xp: size m x 1 (storage)
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double 
cubic_interp(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double a, double b, double *x, double *xp,  double *xo, int m, int n, double step, void *adata) {

  me_data_t *dp=(me_data_t*)adata;
  double f0,f1,f0d,f1d; /* function values and derivatives at a,b */
  double p01,p02,z0,fz0;
  double aa,cc;

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,a,xp); /* xp<=xp+(a)*pk */
  func(xp,x,m,n,adata);
  //my_daxpy(n,xo,-1.0,x);
  //f0=my_dnrm2(n,x);
  //f0*=f0;
  f0=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  /* grad(phi_0): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(a+step)*pk */
  func(xp,x,m,n,adata);
  //my_daxpy(n,xo,-1.0,x);
  //p01=my_dnrm2(n,x);
  p01=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(a-step)*pk */
  func(xp,x,m,n,adata);
  //my_daxpy(n,xo,-1.0,x);
  //p02=my_dnrm2(n,x);
  p02=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  f0d=(p01-p02)/(2.0*step);

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,b,xp); /* xp<=xp+(b)*pk */
  func(xp,x,m,n,adata);
  //my_daxpy(n,xo,-1.0,x);
  //f1=my_dnrm2(n,x);
  //f1*=f1;
  f1=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  /* grad(phi_1): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(b+step)*pk */
  func(xp,x,m,n,adata);
  //my_daxpy(n,xo,-1.0,x);
  //p01=my_dnrm2(n,x);
  p01=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(b-step)*pk */
  func(xp,x,m,n,adata);
  //my_daxpy(n,xo,-1.0,x);
  //p02=my_dnrm2(n,x);
  p02=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  f1d=(p01-p02)/(2.0*step);


  //printf("Interp a,f(a),f'(a): (%lf,%lf,%lf) (%lf,%lf,%lf)\n",a,f0,f0d,b,f1,f1d);
  /* cubic poly in [0,1] is f0+f0d z+eta z^2+xi z^3 
    where eta=3(f1-f0)-2f0d-f1d, xi=f0d+f1d-2(f1-f0) 
    derivative f0d+2 eta z+3 xi z^2 => cc+bb z+aa z^2 */
   aa=3.0*(f0-f1)/(b-a)+(f1d-f0d);
   p01=aa*aa-f0d*f1d;
  /* root exist? */
  if (p01>0.0) {
   /* root */
   cc=sqrt(p01);
   z0=b-(f1d+cc-aa)*(b-a)/(f1d-f0d+2.0*cc);
   /* FIXME: check if this is within boundary */
   aa=MAX(a,b);
   cc=MIN(a,b);
   //printf("Root=%lf, in [%lf,%lf]\n",z0,cc,aa);
   if (z0>aa || z0<cc) {
    fz0=f0+f1;
   } else {
    /* evaluate function for this root */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,a+z0*(b-a),xp); /* xp<=xp+(z0)*pk */
    func(xp,x,m,n,adata);
    //my_daxpy(n,xo,-1.0,x);
    //fz0=my_dnrm2(n,x);
    //fz0*=fz0;
    fz0=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
   }

   /* now choose between f0,f1,fz0,fz1 */
   if (f0<f1 && f0<fz0) {
     return a;
   }
   if (f1<fz0) {
     return b;
   }
   /* else */
   return (z0);
  } else { 

   /* find the value from a or b that minimizes func */
   if (f0<f1) {
    return a;
   } else {
    return b;
   }
  }

  return 0;
}


/*************** Fletcher line search **********************************/
/* zoom function for line search */
/* func: vector function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   a/b: bracket interval [a,b] (a>b) is possible
   x: size n x 1 (storage)
   xp: size m x 1 (storage)
   phi_0: phi(0)
   gphi_0: grad(phi(0))
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double 
linesearch_zoom(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double a, double b, double *x, double *xp,  double phi_0, double gphi_0, double sigma, double rho, double t1, double t2, double t3, double *xo, int m, int n, double step, void *adata) {

  me_data_t *dp=(me_data_t*)adata;
  double alphaj,phi_j,phi_aj;
  double gphi_j,p01,p02,aj,bj;
  double alphak=1.0;
  int ci,found_step=0;

  aj=a;
  bj=b;
  ci=0;
  while(ci<10) {
    /* choose alphaj from [a+t2(b-a),b-t3(b-a)] */
    p01=aj+t2*(bj-aj);
    p02=bj-t3*(bj-aj);
    alphaj=cubic_interp(func,xk,pk,p01,p02,x,xp,xo,m,n,step,adata);
    //printf("cubic intep [%lf,%lf]->%lf\n",p01,p02,alphaj);

    /* evaluate phi(alphaj) */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,alphaj,xp); /* xp<=xp+(alphaj)*pk */
    func(xp,x,m,n,adata);
    /* calculate x<=x-xo */
    //my_daxpy(n,xo,-1.0,x);
    //phi_j=my_dnrm2(n,x);
    //phi_j*=phi_j;
    phi_j=func_robust(x,xo,n,dp->robust_nu,dp->Nt);

    /* evaluate phi(aj) */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,aj,xp); /* xp<=xp+(alphaj)*pk */
    func(xp,x,m,n,adata);
    /* calculate x<=x-xo */
    //my_daxpy(n,xo,-1.0,x);
    //phi_aj=my_dnrm2(n,x);
    //phi_aj*=phi_aj;
    phi_aj=func_robust(x,xo,n,dp->robust_nu,dp->Nt);

#ifdef DEBUG
    printf("phi_j=%lf, phi_aj=%lf\n",phi_j,phi_aj);
#endif
    if ((phi_j>phi_0+rho*alphaj*gphi_0) || phi_j>=phi_aj) {
      bj=alphaj; /* aj unchanged */
    } else {
     /* evaluate grad(alphaj) */
     my_dcopy(m,xk,1,xp,1); /* xp<=xk */
     my_daxpy(m,pk,alphaj+step,xp); /* xp<=xp+(alphaj+step)*pk */
     func(xp,x,m,n,adata);
     /* calculate x<=x-xo */
     //my_daxpy(n,xo,-1.0,x);
     //p01=my_dnrm2(n,x);
     p01=func_robust(x,xo,n,dp->robust_nu,dp->Nt);

     my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphaj-step)*pk */
     func(xp,x,m,n,adata);
     /* calculate x<=x-xo */
     //my_daxpy(n,xo,-1.0,x);
     //p02=my_dnrm2(n,x);
     p02=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
     gphi_j=(p01-p02)/(2.0*step);
#ifdef DEBUG
    printf("p01=%lf, p02=%lf\n",p01,p02);
#endif

     /* termination due to roundoff/other errors pp. 38, Fletcher */
     if ((aj-alphaj)*gphi_j<=step) {
      alphak=alphaj;
      found_step=1;
      break;
     }
    
     if (fabs(gphi_j)<=-sigma*gphi_0) {
      alphak=alphaj;
      found_step=1;
      break;
     }
     
     if (gphi_j*(bj-aj)>=0) {
       bj=aj;
     } /* else bj unchanged */
     aj=alphaj;
   }
   ci++;
  }

  if (!found_step) {
   /* use bound to find possible step */
   alphak=alphaj;
  }
   
#ifdef DEBUG
  printf("Found %lf Interval [%lf,%lf]\n",alphak,a,b);
#endif
  return alphak;
}
 
 

/* line search */
/* func: vector function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   alpha1: initial value for step
   sigma,rho,t1,t2,t3: line search parameters (from Fletcher) 
   xo: observed data size n x 1
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double 
linesearch(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *xk, double *pk, double alpha1, double sigma, double rho, double t1, double t2, double t3, double *xo, int m, int n, double step, void *adata) {
 
 /* phi(alpha)=f(xk+alpha pk)
  for vector function func 
   f(xk) =||func(xk)||^2 */
  
  me_data_t *dp=(me_data_t*)adata;
  double *x,*xp;
  double alphai,alphai1;
  double phi_0,phi_alphai,phi_alphai1;
  double p01,p02;
  double gphi_0,gphi_i;
  double alphak;

  double mu;
  double tol; /* lower limit for minimization */

  int ci;

  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((xp=(double*)calloc((size_t)m,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }

  alphak=1.0;
  /* evaluate phi_0 and grad(phi_0) */
  func(xk,x,m,n,adata);
  //my_daxpy(n,xo,-1.0,x);
  //phi_0=my_dnrm2(n,x);
  //phi_0*=phi_0;
  phi_0=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  /* select tolarance 1/100 of current function value */
  tol=MIN(0.01*phi_0,1e-6);


  /* grad(phi_0): evaluate at -step and +step */
  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(0.0+step)*pk */
  func(xp,x,m,n,adata);
  /* calculate x<=x-xo */
  //my_daxpy(n,xo,-1.0,x);
  //p01=my_dnrm2(n,x);
  p01=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(0.0-step)*pk */
  func(xp,x,m,n,adata);
  /* calculate x<=x-xo */
  //my_daxpy(n,xo,-1.0,x);
  //p02=my_dnrm2(n,x);
  p02=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
  gphi_0=(p01-p02)/(2.0*step);


  /* estimate for mu */
  /* mu = (tol-phi_0)/(rho gphi_0) */
  mu=(tol-phi_0)/(rho*gphi_0);
#ifdef DEBUG
  printf("cost=%lf grad=%lf mu=%lf, alpha1=%lf\n",phi_0,gphi_0,mu,alpha1);
#endif

  ci=1;
  alphai=alpha1; /* initial value for alpha(i) : check if 0<alphai<=mu */
  alphai1=0.0;
  phi_alphai1=phi_0;
  while(ci<10) {
   /* evalualte phi(alpha(i))=f(xk+alphai pk) */
   my_dcopy(m,xk,1,xp,1); /* xp<=xk */
   my_daxpy(m,pk,alphai,xp); /* xp<=xp+alphai*pk */
   func(xp,x,m,n,adata);
   /* calculate x<=x-xo */
   //my_daxpy(n,xo,-1.0,x);
   //phi_alphai=my_dnrm2(n,x);
   //phi_alphai*=phi_alphai;
   phi_alphai=func_robust(x,xo,n,dp->robust_nu,dp->Nt);

   if (phi_alphai<tol) {
     alphak=alphai;
#ifdef DEBUG
     printf("Linesearch : Condition 0 met\n");
#endif
     break;
   }

   if ((phi_alphai>phi_0+alphai*gphi_0) || (ci>1 && phi_alphai>=phi_alphai1)) {
      /* ai=alphai1, bi=alphai bracket */
      alphak=linesearch_zoom(func,xk,pk,alphai1,alphai,x,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,xo,m,n,step,adata);
#ifdef DEBUG
      printf("Linesearch : Condition 1 met\n");
#endif
      break;
   } 

   /* evaluate grad(phi(alpha(i))) */
   my_dcopy(m,xk,1,xp,1); /* NOT NEEDED here?? xp<=xk */
   my_daxpy(m,pk,alphai+step,xp); /* xp<=xp+(alphai+step)*pk */
   func(xp,x,m,n,adata);
   /* calculate x<=x-xo */
   //my_daxpy(n,xo,-1.0,x);
   //p01=my_dnrm2(n,x);
   p01=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
   my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphai-step)*pk */
   func(xp,x,m,n,adata);
   /* calculate x<=x-xo */
   //my_daxpy(n,xo,-1.0,x);
   //p02=my_dnrm2(n,x);
   p01=func_robust(x,xo,n,dp->robust_nu,dp->Nt);
   gphi_i=(p01-p02)/(2.0*step);

   if (fabs(gphi_i)<=-sigma*gphi_0) {
     alphak=alphai;
#ifdef DEBUG
     printf("Linesearch : Condition 2 met\n");
#endif
     break;
   }

   if (gphi_i>=0) {
     /* ai=alphai, bi=alphai1 bracket */
     alphak=linesearch_zoom(func,xk,pk,alphai,alphai1,x,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,xo,m,n,step,adata);
#ifdef DEBUG
     printf("Linesearch : Condition 3 met\n");
#endif
     break;
   }

   /* else preserve old values */
   if (mu<=(2*alphai-alphai1)) {
     /* next step */
     alphai1=alphai;
     alphai=mu;
   } else {
     /* choose by interpolation in [2*alphai-alphai1,min(mu,alphai+t1*(alphai-alphai1)] */
     p01=2*alphai-alphai1;
     p02=MIN(mu,alphai+t1*(alphai-alphai1));
     alphai=cubic_interp(func,xk,pk,p01,p02,x,xp,xo,m,n,step,adata);
     //printf("cubic interp [%lf,%lf]->%lf\n",p01,p02,alphai);
   }
   phi_alphai1=phi_alphai;

   ci++;
  }



  free(x);
  free(xp);
#ifdef DEBUG
  printf("Step size=%lf\n",alphak);
#endif
  return alphak;
}
/*************** END Fletcher line search **********************************/

/*************************************** ROBUST ***************************/
/* worker thread for a cpu */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
cpu_calc_deriv_robust(void *adata) {
 thread_data_grad_t *t=(thread_data_grad_t*)adata;

 int ci,nb;
 int stc,stoff,stm,sta1,sta2;
 int N=t->N; /* stations */
 int M=t->M; /* clusters */
 int Nbase=(t->Nbase)*(t->tilesz);


 double xr[8]; /* residuals */
 complex double G1[4],G2[4],C[4],T1[4],T2[4];
 double pp[8];
 double dsum;
 int cli,tpchunk,pstart,nchunk,tilesperchunk,stci,ttile,tptile,poff;
 double nu=t->robust_nu;

 /* iterate over each paramter */
 for (ci=t->g_start; ci<=t->g_end; ++ci) {
    t->g[ci]=0.0;
    /* find station and parameter corresponding to this value of ci */
    /* this parameter should correspond to the right baseline (x tilesz)
        to contribute to the derivative */
    cli=0;
    while((cli<M) && (ci<t->carr[cli].p[0] || ci>t->carr[cli].p[0]+8*N*t->carr[cli].nchunk-1)) {
     cli++;
    }
   /* now either cli>=M: cluster not found 
       or cli<M and cli is the right cluster */
   if (cli==M && ci>=t->carr[cli-1].p[0] && ci<=t->carr[cli-1].p[0]+8*N*t->carr[cli-1].nchunk-1) {
    cli--;
   }

   if (cli<M) {
    /* right parameter offset */
    stci=ci-t->carr[cli].p[0];
 
    stc=(stci%(8*N))/8; /* 0..N-1 */
    /* make sure this baseline contribute to this parameter */
    tpchunk=stci/(8*N);
    nchunk=t->carr[cli].nchunk;
    pstart=t->carr[cli].p[0];
    tilesperchunk=(t->tilesz+nchunk-1)/nchunk;


    /* iterate over all baselines and accumulate sum */
    for (nb=0; nb<Nbase; ++nb) {
     /* which tile is this ? */
     ttile=nb/t->Nbase;
     /* which chunk this tile belongs to */
     tptile=ttile/tilesperchunk;
     /* now tptile has to match tpchunk, otherwise ignore calculation */
     if (tptile==tpchunk) {

     sta1=t->barr[nb].sta1;
     sta2=t->barr[nb].sta2;
    
     if (((stc==sta1)||(stc==sta2))&& !t->barr[nb].flag) {
      /* this baseline has a contribution */
      /* which paramter of this station */
      stoff=(stci%(8*N))%8; /* 0..7 */
      /* which cluster */
      stm=cli; /* 0..M-1 */

      /* exact expression for derivative  
       for Gaussian \sum( y_i - f_i(\theta))^2
         2 real( vec^H(residual_this_baseline) 
            * vec(-J_{pm}C_{pqm} J_{qm}^H)
        where m: chosen cluster
        J_{pm},J_{qm} Jones matrices for baseline p-q
        depending on the parameter, J ==> E 
        E: zero matrix, except 1 at location of m
      \sum( 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 

       for robust \sum( log(1+ (y_i-f_i(\theta))^2/\nu) )
       all calculations are like for the Gaussian case, except
       when taking summation
      \sum( 1/(\nu+(y_i-f_i)^2) 2 (y_i-f_i) * -\partical (f_i)/ \partial\theta 
 
      so additonal multiplication by 1/(\nu+(y_i-f_i)^2)
     */
     /* read in residual vector, (real,imag) separately */
     xr[0]=t->x[nb*8];
     xr[1]=t->x[nb*8+1];
     xr[2]=t->x[nb*8+2];
     xr[3]=t->x[nb*8+3];
     xr[4]=t->x[nb*8+4];
     xr[5]=t->x[nb*8+5];
     xr[6]=t->x[nb*8+6];
     xr[7]=t->x[nb*8+7];

     /* read in coherency */
     C[0]=t->coh[4*M*nb+4*stm];
     C[1]=t->coh[4*M*nb+4*stm+1];
     C[2]=t->coh[4*M*nb+4*stm+2];
     C[3]=t->coh[4*M*nb+4*stm+3];

     memset(pp,0,sizeof(double)*8); 
     if (stc==sta1) {
       /* this station parameter gradient */
       pp[stoff]=1.0;
       memset(G1,0,sizeof(complex double)*4); 
       G1[0]=pp[0]+_Complex_I*pp[1];
       G1[1]=pp[2]+_Complex_I*pp[3];
       G1[2]=pp[4]+_Complex_I*pp[5];
       G1[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta2*8;
       G2[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G2[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G2[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+4]);
       G2[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);

     } else if (stc==sta2) {
       memset(G2,0,sizeof(complex double)*4); 
       pp[stoff]=1.0;
       G2[0]=pp[0]+_Complex_I*pp[1];
       G2[1]=pp[2]+_Complex_I*pp[3];
       G2[2]=pp[4]+_Complex_I*pp[5];
       G2[3]=pp[6]+_Complex_I*pp[7];
       poff=pstart+tpchunk*8*N+sta1*8;
       G1[0]=(t->p[poff])+_Complex_I*(t->p[poff+1]);
       G1[1]=(t->p[poff+2])+_Complex_I*(t->p[poff+3]);
       G1[2]=(t->p[poff+4])+_Complex_I*(t->p[poff+5]);
       G1[3]=(t->p[poff+6])+_Complex_I*(t->p[poff+7]);
     }

     /* T1=G1*C */
     amb(G1,C,T1);
     /* T2=T1*G2' */
     ambt(T1,G2,T2);

     /* calculate product xr*vec(J_p C J_q^H )/(nu+residual^2) */
     dsum=xr[0]*creal(T2[0])/(nu+xr[0]*xr[0]);
     dsum+=xr[1]*cimag(T2[0])/(nu+xr[1]*xr[1]);
     dsum+=xr[2]*creal(T2[1])/(nu+xr[2]*xr[2]);
     dsum+=xr[3]*cimag(T2[1])/(nu+xr[3]*xr[3]);
     dsum+=xr[4]*creal(T2[2])/(nu+xr[4]*xr[4]);
     dsum+=xr[5]*cimag(T2[2])/(nu+xr[5]*xr[5]);
     dsum+=xr[6]*creal(T2[3])/(nu+xr[6]*xr[6]);
     dsum+=xr[7]*cimag(T2[3])/(nu+xr[7]*xr[7]);

     /* accumulate sum NOTE
     its important to get the sign right,
     depending on res=data-model or res=model-data  */
     t->g[ci]+=2.0*(dsum);
     }
     }
    }
   }
 }


 return NULL;
}
/* calculate gradient */
/* func: vector function
   p: parameter values size m x 1 (at which grad is calculated)
   g: gradient size m x 1 
   xo: observed data size n x 1
   robust_nu: nu in T distribution
   n: size of vector function
   step: step size for differencing 
   adata:  additional data passed to the function
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static int
func_grad_robust(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *g, double *xo, int m, int n, double step, void *adata) {
  /* gradient for each parameter is
     (||func(p+step*e_i)-x||^2-||func(p-step*e_i)-x||^2)/2*step
    i=0,...,m-1 for all parameters
    e_i: unit vector, 1 only at i-th location
  */

  double *x; /* array to store residual */
  int ci;
  me_data_t *dp=(me_data_t*)adata;

  int Nt=dp->Nt;

  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_grad_t *threaddata;


  if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  /* evaluate func once, store in x, and create threads */
  /* and calculate the residual x=xo-func */
  func(p,x,m,n,adata);
  /* calculate x<=x-xo */
  my_daxpy(n,xo,-1.0,x);

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_grad_t*)malloc((size_t)Nt*sizeof(thread_data_grad_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  int nth,Nparm;

  /* parameters per thread */
  Nparm=(m+Nt-1)/Nt;

  /* each thread will calculate derivative of part of 
     parameters */
  ci=0;
  for (nth=0;  nth<Nt; nth++) {
   threaddata[nth].Nbase=dp->Nbase;
   threaddata[nth].tilesz=dp->tilesz;
   threaddata[nth].barr=dp->barr;
   threaddata[nth].carr=dp->carr;
   threaddata[nth].M=dp->M;
   threaddata[nth].N=dp->N;
   threaddata[nth].coh=dp->coh;
   threaddata[nth].m=m;
   threaddata[nth].n=n;
   threaddata[nth].x=x;
   threaddata[nth].p=p;
   threaddata[nth].g=g;
   threaddata[nth].robust_nu=dp->robust_nu;
   threaddata[nth].g_start=ci;
   threaddata[nth].g_end=ci+Nparm-1;
   if (threaddata[nth].g_end>=m) {
    threaddata[nth].g_end=m-1;
   }
   ci=ci+Nparm;
   pthread_create(&th_array[nth],&attr,cpu_calc_deriv_robust,(void*)(&threaddata[nth]));
  }

  /* now wait for threads to finish */
  for(nth=0; nth<Nt; nth++) {
   pthread_join(th_array[nth],NULL);
  }

  pthread_attr_destroy(&attr);

  free(th_array);
  free(threaddata);

  free(x);
  return 0;
}

int
lbfgs_fit_robust(
   void (*func)(double *p, double *hx, int m, int n, void *adata),
   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads,
 void *adata) {

  double *gk; /* gradients at both k+1 and k iter */
  double *xk1,*xk; /* parameters at k+1 and k iter */
  double *pk; /* step direction H_k * grad(f) */

  double step=1e-6;
  double *y, *s; /* storage for delta(grad) and delta(p) */
  double *rho; /* storage for 1/yk^T*sk */
  int ci,ck,cm;
  double alphak=1.0;
  

  if ((gk=(double*)calloc((size_t)m,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((xk1=(double*)calloc((size_t)m,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((xk=(double*)calloc((size_t)m,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }

  if ((pk=(double*)calloc((size_t)m,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }


  /* storage size mM x 1*/
  if ((s=(double*)calloc((size_t)m*M,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((y=(double*)calloc((size_t)m*M,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }
  if ((rho=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
     exit(1);
  }

  /* initial value for params xk=p */
  my_dcopy(m,p,1,xk,1);
  /*  gradient gk=grad(f)_k */
  func_grad_robust(func,xk,gk,x,m,n,step,adata);
  double gradnrm=my_dnrm2(m,gk);
  /* if gradient is too small, no need to solve, so stop */
  if (gradnrm<CLM_STOP_THRESH) {
   ck=itmax;
   step=0.0;
  } else {
   ck=0;
   /* step in [1e-6,1e-9] */
   step=MAX(1e-9,MIN(1e-3/gradnrm,1e-6));
  }
#ifdef DEBUG
  printf("||grad||=%g step=%g\n",gradnrm,step);
#endif

  cm=0;
  ci=0;
  
  while (ck<itmax) {
   /* mult with hessian  pk=-H_k*gk */
   if (ck<M) {
    mult_hessian(m,pk,gk,s,y,rho,ck,ci);
   } else {
    mult_hessian(m,pk,gk,s,y,rho,M,ci);
   }
   my_dscal(m,-1.0,pk);

   /* linesearch to find step length */
   /* parameters alpha1=10.0,sigma=0.1, rho=0.01, t1=9, t2=0.1, t3=0.5 */
   alphak=linesearch(func,xk,pk,10.0,0.1,0.01,9,0.1,0.5,x,m,n,step,adata);
   /* parameters c1=1e-4 c2=0.9, alpha1=1.0, alphamax=10.0, step (for alpha)=1e-4*/
   //alphak=linesearch_nw(func_robust,xk,pk,1.0,10.0,1e-4,0.9,x,m,n,1e-4,adata);
   //alphak=1.0;
   /* update parameters xk1=xk+alpha_k *pk */
   my_dcopy(m,xk,1,xk1,1);
   my_daxpy(m,pk,alphak,xk1);
  
   /* calculate sk=xk1-xk and yk=gk1-gk */
   /* sk=xk1 */ 
   my_dcopy(m,xk1,1,&s[cm],1); 
   /* sk=sk-xk */
   my_daxpy(m,xk,-1.0,&s[cm]);
   /* yk=-gk */ 
   my_dcopy(m,gk,1,&y[cm],1); 
   my_dscal(m,-1.0,&y[cm]);

   /* update gradient */
   func_grad_robust(func,xk1,gk,x,m,n,step,adata);
   /* yk=yk+gk1 */
   my_daxpy(m,gk,1.0,&y[cm]);

   /* calculate 1/yk^T*sk */
   rho[ci]=1.0/my_ddot(m,&y[cm],&s[cm]);

   /* update xk=xk1 */
   my_dcopy(m,xk1,1,xk,1); 
  
   //printf("iter %d store %d\n",ck,cm);
   ck++;
   /* increment storage appropriately */
   if (cm<(M-1)*m) {
    /* offset of m */
    cm=cm+m;
    ci++;
   } else {
    cm=ci=0;
   }
  }


 /* copy back solution to p */
 my_dcopy(m,xk,1,p,1);

 /* for (ci=0; ci<m; ci++) {
   printf("grad %d=%lf\n",ci,gk[ci]);
  } */

  free(gk);
  free(xk1);
  free(xk);
  free(pk);
  free(s);
  free(y);
  free(rho);
  return 0;
}
