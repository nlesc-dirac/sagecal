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
#include <math.h>

/* Digamma function
   if x>7 use digamma(x) = digamma(x+1) - 1/x
   for accuracy
   using maple expansion
   series(Psi(x+1/2), x=infinity, 21);
   ln(x)+1/24/x^2-7/960/x^4+31/8064/x^6-127/30720/x^8+511/67584/x^10-1414477/67092480/x^12+8191/98304/x^14-118518239/267386880/x^16+5749691557/1882718208/x^18-91546277357/3460300800/x^20+O(1/x^21)


   based on code by Mark Johnson, 2nd September 2007
*/
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static double
digamma(double x) {
  double result = 0.0, xx, xx2, xx4;
  for ( ; x < 7; ++x) { /* reduce x till x<7 */
    result -= 1.0/x;
  }
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;
}



/* update nu (degrees of freedom)

   nu0: current value of nu
   e: Nx1 residual error
   w: Nx1 weight vector


   psi() : digamma function
   find soltion to
   psi((nu+1)/2)-ln((nu+1)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 = 0
   use ln(gamma()) => lgamma_r
*/
double
update_nu(double nu0, double *w, int N, int Nt, double nulow, double nuhigh) {
  int ci;
  int Nd=30; /* no of samples to estimate nu */
  double dgm,sumq,sumw,deltanu,thisnu,*q;
  if ((q=(double*)calloc((size_t)N,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }

  sumw=my_dasum(N,w)/(double)N; /* sum(w_i)/N */
  /* use q to calculate log(w_i) */
  for (ci=0; ci<N; ci++) {
    q[ci]=log(w[ci]);
    //printf("ci=%d w=%lf logw=%lf\n",ci,w[ci],q[ci]);
  }
  sumq=my_dasum(N,q)/(double)N; /* sum(log(w_i)/N) */

  /* search range [low,high] if nu~=30, its Gaussian */
  deltanu=(double)(nuhigh-nulow)/(double)Nd;
  for (ci=0; ci<Nd; ci++) {
   thisnu=(nulow+ci*deltanu);
   dgm=digamma(thisnu*0.5+0.5);
   q[ci]=dgm-log((thisnu+1.0)*0.5); /* psi((nu+1)/2)-log((nu+1)/2) */
   dgm=digamma(thisnu*0.5);
   q[ci]+=-dgm+log((thisnu)*0.5); /* -psi((nu)/2)+log((nu)/2) */
   q[ci]+=sumq-sumw+1.0; /* sum(ln(w_i))/N-sum(w_i)/N+1 */
   //printf("ci=%d q=%lf\n",ci,q[ci]);
  }
  ci=my_idamin(Nd,q,1);
  thisnu=(nulow+ci*deltanu);

  free(q);
  return thisnu;

}

/* update w<= (nu+1)/(nu+delta^2)
   then q <= w-log(w), so that it is +ve
*/ 
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
w_nu_update_threadfn(void *data) {
 thread_data_vecnu_t *t=(thread_data_vecnu_t*)data;
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   //t->ed[ci]*=t->wtd[ci]; ??
   t->wtd[ci]=(t->nu0+1.0)/(t->nu0+t->ed[ci]*t->ed[ci]);
   t->q[ci]=t->wtd[ci]-log(t->wtd[ci]);
 }
 return NULL;
}

/* update w<= sqrt(w) */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
w_sqrt_threadfn(void *data) {
 thread_data_vecnu_t *t=(thread_data_vecnu_t*)data;
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   t->wtd[ci]=sqrt(t->wtd[ci]);
 }
 return NULL;
}

/* update nu  */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
q_update_threadfn(void *data) {
 thread_data_vecnu_t *t=(thread_data_vecnu_t*)data;
 int ci;
 double thisnu,dgm;
 for (ci=t->starti; ci<=t->endi; ci++) {
   thisnu=(t->nulow+ci*t->nu0); /* deltanu stored in nu0 */
   dgm=digamma(thisnu*0.5+0.5);
   t->q[ci]=dgm-log((thisnu+1.0)*0.5); /* psi((nu+1)/2)-log((nu+1)/2) */
   dgm=digamma(thisnu*0.5);
   t->q[ci]+=-dgm+log((thisnu)*0.5); /* -psi((nu)/2)+log((nu)/2) */
   t->q[ci]+=-t->sumq+1.0; /* q is w-log(w), so -ve: sum(ln(w_i))/N-sum(w_i)/N+1 */
 }
 return NULL;
}

/* update nu (degrees of freedom)
   also update w

   nu0: current value of nu
   w: Nx1 weight vector
   ed: Nx1 residual error


   psi() : digamma function
   find soltion to
   psi((nu+1)/2)-ln((nu+1)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 = 0
   use ln(gamma()) => lgamma_r
*/
double
update_w_and_nu(double nu0, double *w, double *ed, int N, int Nt, double nulow, double nuhigh) {
  int Nd=30; /* no of samples to estimate nu */
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_vecnu_t *threaddata;

  double deltanu,*q,thisnu,sumq;
  if ((q=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
  }
  if ((threaddata=(thread_data_vecnu_t*)malloc((size_t)Nt*sizeof(thread_data_vecnu_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
  }

  /* calculate min values a thread can handle */
  Nthb0=(N+Nt-1)/Nt;
  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<N; nth++) {
    if (ci+Nthb0<N) {
     Nthb=Nthb0;
    } else {
     Nthb=N-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].ed=ed;
    threaddata[nth].wtd=w;
    threaddata[nth].q=q;
    threaddata[nth].nu0=nu0;
    threaddata[nth].nulow=nulow;
    threaddata[nth].nuhigh=nuhigh;
    pthread_create(&th_array[nth],&attr,w_nu_update_threadfn,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }
  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }
  sumq=my_dasum(N,q)/(double)N; /* sum(|w_i-log(w_i)|/N), assume all elements are +ve */
  for(nth1=0; nth1<nth; nth1++) {
    pthread_create(&th_array[nth1],&attr,w_sqrt_threadfn,(void*)(&threaddata[nth1]));
  }
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

  /* search range 2 to 30 because if nu~=30, its Gaussian */
  deltanu=(double)(nuhigh-nulow)/(double)Nd;
  Nthb0=(Nd+Nt-1)/Nt;
  ci=0;
  for (nth=0;  nth<Nt && ci<Nd; nth++) {
    if (ci+Nthb0<Nd) {
     Nthb=Nthb0;
    } else {
     Nthb=Nd-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].q=q;
    threaddata[nth].nu0=deltanu;
    threaddata[nth].sumq=sumq;
    pthread_create(&th_array[nth],&attr,q_update_threadfn,(void*)(&threaddata[nth]));
    /* next baseline set */
    ci=ci+Nthb;
  }
  /* now wait for threads to finish */
  for(nth1=0; nth1<nth; nth1++) {
   pthread_join(th_array[nth1],NULL);
  }

  pthread_attr_destroy(&attr);
  free(th_array);
  free(threaddata);

  ci=my_idamin(Nd,q,1);
  thisnu=(nulow+ci*deltanu);

  free(q);
  return thisnu;

 return 0;
}
