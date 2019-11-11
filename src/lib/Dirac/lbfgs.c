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


#include "Dirac.h"
#include <pthread.h>

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
 if ((idx=(int*)calloc((size_t)M,sizeof(int)))==0) {
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
/* func: scalar function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   a/b:  interval for interpolation
   xp: size m x 1 (storage)
   step: step size for differencing 
   adata:  additional data passed to the function
*/
static double 
cubic_interp(
   double (*func)(double *p, int m, void *adata),
   double *xk, double *pk, double a, double b, double *xp,  int m, double step, void *adata) {

  double f0,f1,f0d,f1d; /* function values and derivatives at a,b */
  double p01,p02,z0,fz0;
  double aa,cc;

  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,a,xp); /* xp<=xp+(a)*pk */
  f0=func(xp,m,adata);
  /* grad(phi_0): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(a+step)*pk */
  p01=func(xp,m,adata);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(a-step)*pk */
  p02=func(xp,m,adata);
  f0d=(p01-p02)/(2.0*step);

  //FIXME my_dcopy(m,xk,1,xp,1); /* not necessary because xp=xk+(a-step)*pk */
  //FIXME my_daxpy(m,pk,b,xp); /* xp<=xp+(b)*pk */
  my_daxpy(m,pk,-a+step+b,xp); /* xp<=xp+(b)*pk */
  f1=func(xp,m,adata);
  /* grad(phi_1): evaluate at -step and +step */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(b+step)*pk */
  p01=func(xp,m,adata);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(b-step)*pk */
  p02=func(xp,m,adata);
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
    //FIXME my_dcopy(m,xk,1,xp,1); /* not necessary because xp=xk+(b-step)*pk */
    //my_daxpy(m,pk,a+z0*(b-a),xp); /* xp<=xp+(z0)*pk */
    my_daxpy(m,pk,-b+step+a+z0*(b-a),xp); /* xp<=xp+(a+z0*(b-a))*pk */
    fz0=func(xp,m,adata);
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
  /* fallback value */
  return (a+b)*0.5;
}





/*************** Fletcher line search **********************************/
/* zoom function for line search */
/* func: scalar function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   a/b: bracket interval [a,b] (a>b) is possible
   xp: size m x 1 (storage)
   phi_0: phi(0)
   gphi_0: grad(phi(0))
   sigma,rho,t1,t2,t3: line search parameters (from Fletcher) 
   step: step size for differencing 
   adata:  additional data passed to the function
*/
static double 
linesearch_zoom(
   double (*func)(double *p, int m, void *adata),
   double *xk, double *pk, double a, double b, double *xp, double phi_0, double gphi_0, double sigma, double rho, double t1, double t2, double t3, int m, double step, void *adata) {

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
    alphaj=cubic_interp(func,xk,pk,p01,p02,xp,m,step,adata);
    //printf("cubic intep [%lf,%lf]->%lf\n",p01,p02,alphaj);

    /* evaluate phi(alphaj) */
    my_dcopy(m,xk,1,xp,1); /* xp<=xk */
    my_daxpy(m,pk,alphaj,xp); /* xp<=xp+(alphaj)*pk */
    phi_j=func(xp,m,adata);

    /* evaluate phi(aj) */
    //FIXME my_dcopy(m,xk,1,xp,1); /* xp<=xk : not necessary because already copied */
    //FIXME my_daxpy(m,pk,aj,xp); /* xp<=xp+(aj)*pk */
    my_daxpy(m,pk,-alphaj+aj,xp); /* xp<=xp+(aj)*pk */
    phi_aj=func(xp,m,adata);

    if ((phi_j>phi_0+rho*alphaj*gphi_0) || phi_j>=phi_aj) {
      bj=alphaj; /* aj unchanged */
    } else {
     /* evaluate grad(alphaj) */
     //FIXME my_dcopy(m,xk,1,xp,1); /* xp<=xk */
     //FIXME my_daxpy(m,pk,alphaj+step,xp); /* xp<=xp+(alphaj+step)*pk */
     my_daxpy(m,pk,-aj+alphaj+step,xp); /* xp<=xp+(alphaj+step)*pk */
     p01=func(xp,m,adata);
     my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphaj-step)*pk */
     p02=func(xp,m,adata);
     gphi_j=(p01-p02)/(2.0*step);

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
  printf("Found %g Interval [%lf,%lf]\n",alphak,a,b);
#endif
  return alphak;
}
 
 

/* line search */
/* func: scalar function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   alpha1: initial value for step
   sigma,rho,t1,t2,t3: line search parameters (from Fletcher) 
   m: size or parameter vector
   step: step size for differencing 
   adata:  additional data passed to the function
*/
double 
linesearch(
   double (*func)(double *p, int m, void *adata),
   double *xk, double *pk, double alpha1, double sigma, double rho, double t1, double t2, double t3, int m, double step, void *adata) {
 
 /* phi(alpha)=f(xk+alpha pk)
  for vector function func 
   f(xk) =||func(xk)||^2 */
  
  double *xp;
  double alphai,alphai1;
  double phi_0,phi_alphai,phi_alphai1;
  double p01,p02;
  double gphi_0,gphi_i;
  double alphak;

  double mu;
  double tol; /* lower limit for minimization */

  int ci;

  if ((xp=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  alphak=1.0;
  /* evaluate phi_0 and grad(phi_0) */
  phi_0=func(xk,m,adata);
  /* select tolarance 1/100 of current function value */
  tol=MIN(0.01*phi_0,1e-6);

  /* grad(phi_0): evaluate at -step and +step */
  my_dcopy(m,xk,1,xp,1); /* xp<=xk */
  my_daxpy(m,pk,step,xp); /* xp<=xp+(0.0+step)*pk */
  p01=func(xp,m,adata);
  my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(0.0-step)*pk */
  p02=func(xp,m,adata);
  gphi_0=(p01-p02)/(2.0*step);

  /* estimate for mu */
  /* mu = (tol-phi_0)/(rho gphi_0) */
  mu=(tol-phi_0)/(rho*gphi_0);
#ifdef DEBUG
  printf("deltaphi=%lf, mu=%lf, alpha1=%lf\n",gphi_0,mu,alpha1);
#endif
  /* catch if not finite (deltaphi=0 or nan) */
  if (!isnormal(mu)) {
    free(xp);
#ifdef DEBUG
  printf("line interval too small\n");
#endif
    return mu;
  }

  ci=1;
  alphai=alpha1; /* initial value for alpha(i) : check if 0<alphai<=mu */
  alphai1=0.0;
  phi_alphai1=phi_0;
  while(ci<10) {
   /* evalualte phi(alpha(i))=f(xk+alphai pk) */
   my_dcopy(m,xk,1,xp,1); /* xp<=xk */
   my_daxpy(m,pk,alphai,xp); /* xp<=xp+alphai*pk */
   phi_alphai=func(xp,m,adata);

   if (phi_alphai<tol) {
     alphak=alphai;
#ifdef DEBUG
     printf("Linesearch : Condition 0 met\n");
#endif
     break;
   }

   if ((phi_alphai>phi_0+alphai*gphi_0) || (ci>1 && phi_alphai>=phi_alphai1)) {
      /* ai=alphai1, bi=alphai bracket */
      alphak=linesearch_zoom(func,xk,pk,alphai1,alphai,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,m,step,adata);
#ifdef DEBUG
      printf("Linesearch : Condition 1 met\n");
#endif
      break;
   } 

   /* evaluate grad(phi(alpha(i))) */
   //FIXME my_dcopy(m,xk,1,xp,1); /* NOT NEEDED here?? xp<=xk */
   //FIXME my_daxpy(m,pk,alphai+step,xp); /* xp<=xp+(alphai+step)*pk */
   /* note that xp  already is xk+alphai. pk, so only add the missing term */
   my_daxpy(m,pk,step,xp); /* xp<=xp+(alphai+step)*pk */
   p01=func(xp,m,adata);
   my_daxpy(m,pk,-2.0*step,xp); /* xp<=xp+(alphai-step)*pk */
   p02=func(xp,m,adata);
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
     alphak=linesearch_zoom(func,xk,pk,alphai,alphai1,xp,phi_0,gphi_0,sigma,rho,t1,t2,t3,m,step,adata);
#ifdef DEBUG
     printf("Linesearch : Condition 3 met\n");
#endif
     break;
   }

   /* else preserve old values */
   if (mu<=(2.0*alphai-alphai1)) {
     /* next step */
     alphai1=alphai;
     alphai=mu;
   } else {
     /* choose by interpolation in [2*alphai-alphai1,min(mu,alphai+t1*(alphai-alphai1)] */
     p01=2.0*alphai-alphai1;
     p02=MIN(mu,alphai+t1*(alphai-alphai1));
     alphai=cubic_interp(func,xk,pk,p01,p02,xp,m,step,adata);
     //printf("cubic interp [%lf,%lf]->%lf\n",p01,p02,alphai);
   }
   phi_alphai1=phi_alphai;

   ci++;
  }



  free(xp);
#ifdef DEBUG
  printf("Step size=%g\n",alphak);
#endif
  return alphak;
}
/*************** END Fletcher line search **********************************/


/*************** backtracking line search **********************************/
/* func: cost function
   xk: parameter values size m x 1 (at which step is calculated)
   pk: step direction size m x 1 (x(k+1)=x(k)+alphak * pk)
   gk: gradient vector size m x 1
   m: size or parameter vector
   alpha0: initial alpha
   adata:  additional data passed to the function
*/
static double
linesearch_backtrack(
   double (*func)(double *p, int m, void *adata),
   double *xk, double *pk, double *gk, int m, double alpha0, void *adata) {

  /* Armijo condition  f(x+alpha p) <= f(x) + c alpha p^T grad(f(x)) */
  const double c=1e-4;
  double alphak=alpha0;
  double *xk1,fnew,fold,product;
  if ((xk1=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* update parameters xk1=xk+alpha_k *pk */
  my_dcopy(m,xk,1,xk1,1);
  my_daxpy(m,pk,alphak,xk1);

  fnew=func(xk1,m,adata);
  fold=func(xk,m,adata); /* add threshold to make iterations stop at some point FIXME: is this correct/needed? */
  product=c*my_ddot(m,pk,gk);
  int ci=0;
  while (ci<15 && fnew>fold+alphak*product) {
     alphak *=0.5;
     my_dcopy(m,xk,1,xk1,1);
     my_daxpy(m,pk,alphak,xk1);
     fnew=func(xk1,m,adata);
     ci++;
  }

  free(xk1);
  return alphak;
}
/*************** END backtracking line search **********************************/

/* full batch (original) version of LBFGS */
static int
lbfgs_fit_fullbatch(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   /* adata: user supplied data, 
   */
   /* p:mx1 vector, M: memory size */
   double *p, int m, int itmax, int M, void *adata) { 

  double *gk; /* gradients at both k+1 and k iter */
  double *xk1,*xk; /* parameters at k+1 and k iter */
  double *pk; /* step direction H_k * grad(f) */

  double step=1e-6; /* step for interpolation */
  double *y, *s; /* storage for delta(grad) and delta(p) */
  double *rho; /* storage for 1/yk^T*sk */
  int ci,ck,cm;
  double alphak=1.0;
  

  if ((gk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xk1=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  if ((pk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }


  /* storage size mM x 1*/
  if ((s=(double*)calloc((size_t)m*M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((y=(double*)calloc((size_t)m*M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }


  if ((rho=(double*)calloc((size_t)M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* initial value for params xk=p */
  my_dcopy(m,p,1,xk,1);

  /*  gradient gk=grad(f)_k */
  grad_func(xk,gk,m,adata);
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
  
  cm=0; /* cycle in 0..(M-1)m (in strides of m)*/
  ci=0; /* cycle in 0..(M-1) */
  
#ifdef DEBUG
  printf("cost=%g\n",cost_func(xk,m,adata));
#endif
  while (ck<itmax && isnormal(gradnrm) && gradnrm>CLM_STOP_THRESH) {
#ifdef DEBUG
  printf("iter %d gradnrm %g\n",ck,gradnrm);
#endif
   /* mult with hessian  pk=-H_k*gk */
   if (ck<M) {
    mult_hessian(m,pk,gk,s,y,rho,ck,ci);
   } else {
    mult_hessian(m,pk,gk,s,y,rho,M,ci);
   }
   my_dscal(m,-1.0,pk);

   /* linesearch to find step length */
   /* parameters alpha1=10.0,sigma=0.1, rho=0.01, t1=9, t2=0.1, t3=0.5 */
   alphak=linesearch(cost_func,xk,pk,10.0,0.1,0.01,9,0.1,0.5,m,step,adata);
   
   /* check if step size is too small, or nan, then stop */
   if (!isnormal(alphak) || fabs(alphak)<CLM_EPSILON) {
    break;
   }
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


   grad_func(xk1,gk,m,adata);
   gradnrm=my_dnrm2(m,gk);
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

#ifdef DEBUG
  printf("iter %d alpha=%g ||grad||=%g\n",ck,alphak,gradnrm);
  printf("cost=%g\n",cost_func(xk,m,adata));
#endif

  }


 /* copy back solution to p */
 my_dcopy(m,xk,1,p,1);

#ifdef DEBUG
 // for (ci=0; ci<m; ci++) {
 //  printf("grad %d=%lf\n",ci,gk[ci]);
 // } 
#endif

  free(gk);
  free(xk1);
  free(xk);
  free(pk);
  free(s);
  free(y);
  free(rho);
  return 0;
}


typedef struct thread_data_vecdot_t_ {
 int start;
 int length;
 double *y;
 double *a;
 double *b;
} thread_data_vecdot_t;


static void *
outer_product_threadfn(void *data) {
 thread_data_vecdot_t *t=(thread_data_vecdot_t*)data;
 int ci;
 for (ci=t->start; ci<t->start+t->length; ci++) {
  t->y[ci] += t->a[ci]*t->b[ci];
 }
 return NULL;
}

/* y = y + a.* b
   a,b,y: vectors of size m
   Nt: no. of threads 
*/ 
static int
parallel_outer_product(int m,double *__restrict y,double *__restrict a,double *__restrict b,int Nt) {
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_vecdot_t *threaddata;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_vecdot_t*)malloc((size_t)Nt*sizeof(thread_data_vecdot_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* calculate min values a thread can handle */
  Nthb0=(m+Nt-1)/Nt;
  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<m; nth++) {
   if (ci+Nthb0<m) {
     Nthb=Nthb0;
    } else {
     Nthb=m-ci;
    }
    threaddata[nth].start=ci;
    threaddata[nth].length=Nthb;
    threaddata[nth].y=y;
    threaddata[nth].a=a;
    threaddata[nth].b=b;
    pthread_create(&th_array[nth],&attr,outer_product_threadfn,(void*)(&threaddata[nth]));
    /* next data set */
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


static int
lbfgs_fit_minibatch(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   /* iter: iteration number, adata: user supplied data, 
   indata: persistant data that need to be kept between batches */
   /* p:mx1 vector, M: memory size */
   double *p, int m, int itmax, int M, void *adata, persistent_data_t *indata) { /* note indata!=NULL is assumed implicitly*/

  double *gk; /* gradients at both k+1 and k iter */
  double *xk1,*xk; /* parameters at k+1 and k iter */
  double *pk; /* step direction H_k * grad(f) */

  double *g_min_rold, *g_min_rnew; /* temp storage for updating running averages */

  double *y, *s; /* storage for delta(grad) and delta(p) */
  double *rho; /* storage for 1/yk^T*sk */
  int ci,ck,cm;
  double alphak=1.0;
  double alphabar=1.0;
  

  if ((gk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xk1=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  if ((pk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }


  /* use y,s pairs from the previous run */
  /* storage size mM x 1*/
  s=indata->s;
  y=indata->y;
  rho=indata->rho;
  if (!s || !y || !rho) {
     fprintf(stderr,"%s: %d: storage must be pre allocated befor calling this function.\n",__FILE__,__LINE__);
     exit(1);
  }

  /* initial value for params xk=p */
  my_dcopy(m,p,1,xk,1);

  /*  gradient gk=grad(f)_k */
  grad_func(xk,gk,m,adata);
  double gradnrm=my_dnrm2(m,gk);
  /* if gradient is too small, no need to solve, so stop */
  if (gradnrm<CLM_STOP_THRESH) {
   ck=itmax;
  } else {
   ck=0;
  }
#ifdef DEBUG
  printf("||grad||=%g\n",gradnrm);
#endif
  
  ci=indata->vacant; /* cycle in 0..(M-1) */
  cm=m*ci; /* cycle in 0..(M-1)m (in strides of m)*/
  
  while (ck<itmax && isnormal(gradnrm) && gradnrm>CLM_STOP_THRESH) {
#ifdef DEBUG
   printf("iter %d gradnrm %g\n",ck,gradnrm);
#endif
   /* increment global iteration count */
   indata->niter++;
   /* detect if we are at first iteration of a new batch */
   int batch_changed=(indata->niter>1 && ck==0);
   /* if the batch has changed, update running averages */
   if (batch_changed) {
     /* temp vectors : grad-running_avg(old) , grad - running_avg(new) */
     /* running_avg_new = running_avg_old + (grad-running_avg(old))/niter */
     if ((g_min_rold=(double*)calloc((size_t)m,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if ((g_min_rnew=(double*)calloc((size_t)m,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     my_dcopy(m,gk,1,g_min_rold,1); /* g_min_rold <- grad */
     my_daxpy(m,indata->running_avg,-1.0,g_min_rold); /* g_min_rold <- g_min_rold - running_avg(old) */
     my_daxpy(m,g_min_rold,1.0/(double)indata->niter,indata->running_avg); /* running_avg <- running_avg + 1/niter . g_min_rold */

     my_dcopy(m,gk,1,g_min_rnew,1);
     my_daxpy(m,indata->running_avg,-1.0,g_min_rnew); /* g_min_rnew <- g_min_rnew - running_avg(new) */

     /* this loop should be parallelized/vectorized */
     /*for (it=0; it<m; it++) {
       indata->running_avg_sq[it] += g_min_rold[it]*g_min_rnew[it];
     }*/
     parallel_outer_product(m,indata->running_avg_sq,g_min_rold,g_min_rnew,indata->Nt);

     /* estimate online variance 
       Note: for badly initialized cases, might need to increase initial value of alphabar
       because of gradnrm is too large, alphabar becomes too small */
     alphabar=10.0/(1.0+my_dasum(m,indata->running_avg_sq)/((double)(indata->niter-1)*gradnrm)); 
#ifdef DEBUG
     printf("iter=%d running_avg %lf gradnrm %lf alpha=%lf\n",indata->niter,my_dasum(m,indata->running_avg_sq),gradnrm,alphabar);
#endif
     free(g_min_rold);
     free(g_min_rnew);
   }

   /* mult with hessian  pk=-H_k*gk */
   if (indata->nfilled<M) {
    mult_hessian(m,pk,gk,s,y,rho,indata->nfilled,ci);
   } else {
    mult_hessian(m,pk,gk,s,y,rho,M,ci);
   }
   my_dscal(m,-1.0,pk);

   /* linesearch to find step length */
   /* Armijo line search */
   alphak=linesearch_backtrack(cost_func,xk,pk,gk,m,alphabar,adata);
   /* check if step size is too small, or nan, then stop */
   if (!isnormal(alphak) || fabs(alphak)<CLM_EPSILON) {
    break;
   }
   /* update parameters xk1=xk+alpha_k *pk */
   my_dcopy(m,xk,1,xk1,1);
   my_daxpy(m,pk,alphak,xk1);
  
   if (!batch_changed) {
   /* calculate sk=xk1-xk and yk=gk1-gk */
   /* sk=xk1 */ 
   my_dcopy(m,xk1,1,&s[cm],1); 
   /* sk=sk-xk */
   my_daxpy(m,xk,-1.0,&s[cm]);
   /* yk=-gk */ 
   my_dcopy(m,gk,1,&y[cm],1); 
   my_dscal(m,-1.0,&y[cm]);
   }

   grad_func(xk1,gk,m,adata);
   gradnrm=my_dnrm2(m,gk);
   /* do a sanity check here */
   if (!isnormal(gradnrm) || gradnrm<CLM_STOP_THRESH) {
     break;
   }
 
   if (!batch_changed) {
   /* yk=yk+gk1 */
   my_daxpy(m,gk,1.0,&y[cm]);
   
   /* yk = yk + lm0* sk, to create a trust region */
   double lm0=1e-6;
   if (gradnrm>1e3*lm0) {
    my_daxpy(m,&s[cm],lm0,&y[cm]);
   }
   
   /* calculate 1/yk^T*sk */
   rho[ci]=1.0/my_ddot(m,&y[cm],&s[cm]);
   }

   /* update xk=xk1 */
   my_dcopy(m,xk1,1,xk,1); 
  
   //printf("iter %d store %d\n",ck,cm);
   ck++;
  
   if (!batch_changed) {
   indata->nfilled=(indata->nfilled<M?indata->nfilled+1:M);
   /* increment storage appropriately */
   if (cm<(M-1)*m) {
    /* offset of m */
    cm=cm+m;
    ci++;
    indata->vacant++;
   } else {
    cm=ci=0;
    indata->vacant=0;
   }
   }

#ifdef DEBUG
  printf("iter %d alpha=%g ||grad||=%g\n",ck,alphak,gradnrm);
#endif
  }


 /* copy back solution to p */
 my_dcopy(m,xk,1,p,1);

#ifdef DEBUG
//  for (ci=0; ci<m; ci++) {
//   printf("grad %d=%lf\n",ci,gk[ci]);
//  } 
#endif

  free(gk);
  free(xk1);
  free(xk);
  free(pk);
  return 0;
}

/* cost function : return a scalar cost, input : p (mx1) parameters, m: no. of params, adata: additional data
   grad function: return gradient (mx1): input : p (mx1) parameters, g (mx1) gradient vector, m: no. of params, adata: additional data
*/ 
/*
   p: parameters m x 1 (used as initial value, output final value)
   itmax: max iterations
   M: BFGS memory size
   adata: additional data
*/
int
lbfgs_fit(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   /* iter: iteration number, adata: user supplied data, 
   indata: persistant data that need to be kept between batches */
   /* p:mx1 vector, M: memory size */
   double *p, int m, int itmax, int M, void *adata, persistent_data_t *indata) { /* indata=NULL for full batch */

  if (!indata) {
    lbfgs_fit_fullbatch(cost_func,grad_func,p,m,itmax,M,adata);
  } else {
    lbfgs_fit_minibatch(cost_func,grad_func,p,m,itmax,M,adata,indata);
  }
  return 0;
}



/* user routines for setting up and clearing persistent data structure
   for using stochastic LBFGS */
int
lbfgs_persist_init(persistent_data_t *pt, int Nminibatch, int m, int n, int lbfgs_m, int Nt) {

  if ((pt->offsets=(int*)calloc((size_t)Nminibatch,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((pt->lengths=(int*)calloc((size_t)Nminibatch,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((pt->s=(double*)calloc((size_t)m*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((pt->y=(double*)calloc((size_t)m*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((pt->rho=(double*)calloc((size_t)lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  pt->m=m;
  pt->lbfgs_m=lbfgs_m;

  /* storage for calculating on-line variance of gradient */
  if ((pt->running_avg=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((pt->running_avg_sq=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  pt->nfilled=0; /* always 0 when we start */
  pt->vacant=0; /* cycle in 0..M-1 */
  pt->offset=0; /* offset to data */
  pt->nlen=n; /* length of data */
  pt->niter=0; /* cumulative iteration count */
  pt->Nt=Nt; /* no. of threads need to be passed */

  int batchsize=(n+Nminibatch-1)/Nminibatch;
  /* store info about no. of minibatches to use also in persistent data */
  int ci,ck;
  ck=0;
  for (ci=0; ci<Nminibatch; ci++) {
   pt->offsets[ci]=ck;
   if (pt->offsets[ci]+batchsize<=n) {
    pt->lengths[ci]=batchsize;
   } else {
    pt->lengths[ci]=n-pt->offsets[ci];
   }
   ck=ck+pt->lengths[ci];
  }

  return 0;
}

int
lbfgs_persist_clear(persistent_data_t *pt) {
  /* free persistent memory */
  free(pt->s);
  free(pt->y);
  free(pt->rho);
  free(pt->running_avg);
  free(pt->running_avg_sq);

  free(pt->offsets);
  free(pt->lengths);

  return 0;
} 


int
lbfgs_persist_reset(persistent_data_t *pt) {
 
  memset(pt->s,0,sizeof(double)*(size_t)pt->m*pt->lbfgs_m);
  memset(pt->y,0,sizeof(double)*(size_t)pt->m*pt->lbfgs_m);
  memset(pt->rho,0,sizeof(double)*(size_t)pt->lbfgs_m);
  memset(pt->running_avg,0,sizeof(double)*(size_t)pt->m);
  memset(pt->running_avg_sq,0,sizeof(double)*(size_t)pt->m);

  pt->nfilled=0; /* always 0 when we start */
  pt->vacant=0; /* cycle in 0..m-1 */
  pt->niter=0; /* cumulative iteration count */

  return 0;
}
