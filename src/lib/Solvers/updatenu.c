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
  /* FIXME catch -ve value as input */
  double result = 0.0, xx, xx2, xx4;
  for ( ; x < 7.0; ++x) { /* reduce x till x<7 */
    result -= 1.0/x;
  }
  x -= 0.5;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;
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
   thisnu=(t->nulow+(double)ci*t->nu0); /* deltanu stored in nu0 */
   dgm=digamma(thisnu*0.5+0.5);
   t->q[ci]=dgm-log((thisnu+1.0)*0.5); /* psi((nu+1)/2)-log((nu+1)/2) */
   dgm=digamma(thisnu*0.5);
   t->q[ci]+=-dgm+log((thisnu)*0.5); /* -psi((nu)/2)+log((nu)/2) */
   t->q[ci]+=-t->sumq+1.0; /* q is w-log(w), so -ve: sum(ln(w_i))/N-sum(w_i)/N+1 */
 }
 return NULL;
}

/* update nu  */
#ifdef USE_MIC
__attribute__ ((target(MIC)))
#endif
static void *
q_update_threadfn_aecm(void *data) {
 thread_data_vecnu_t *t=(thread_data_vecnu_t*)data;
 int ci;
 double thisnu,dgm;
 for (ci=t->starti; ci<=t->endi; ci++) {
   thisnu=(t->nulow+(double)ci*t->nu0); /* deltanu stored in nu0 */
   dgm=digamma(thisnu*0.5);
   t->q[ci]=-dgm+log((thisnu)*0.5); /* -psi((nu)/2)+log((nu)/2) */
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
  /* check for too low number of values per thread, halve the threads */
  if (Nthb0<=2) {
   Nt=Nt/2;
   Nthb0=(Nd+Nt-1)/Nt;
  }
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
  thisnu=(nulow+(double)ci*deltanu);

  free(q);
  return thisnu;

 return 0;
}




/* update nu (degrees of freedom)
   nu_old: old nu
   logsumw = 1/N sum(log(w_i)-w_i)

   use Nd values in [nulow,nuhigh] to find nu


   psi() : digamma function
   find soltion to
   psi((nu_old+p)/2)-ln((nu_old+p)/2)-psi(nu/2)+ln(nu/2)+1/N sum(ln(w_i)-w_i) +1 = 0
   use ln(gamma()) => lgamma_r

   p: 1 or 8
*/
double
update_nu(double logsumw, int Nd, int Nt, double nulow, double nuhigh, int p, double nu_old) {
  int ci,nth,nth1,Nthb,Nthb0;
  double deltanu,thisnu,*q;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_vecnu_t *threaddata;

  if ((q=(double*)calloc((size_t)Nd,sizeof(double)))==0) {
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
  /* calculate psi((nu_old+p)/2)-ln((nu_old+p)/2) */
  double dgm=digamma((nu_old+(double)p)*0.5);
  dgm=dgm-log((nu_old+(double)p)*0.5); /* psi((nu+p)/2)-log((nu+p)/2) */


  deltanu=(double)(nuhigh-nulow)/(double)Nd;
  Nthb0=(Nd+Nt-1)/Nt;
  /* check for too low number of values per thread, halve the threads */
  if (Nthb0<=2) {
   Nt=Nt/2;
   Nthb0=(Nd+Nt-1)/Nt;
  }
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
    threaddata[nth].nulow=nulow;
    threaddata[nth].nuhigh=nuhigh;
    threaddata[nth].sumq=-logsumw-dgm;
    pthread_create(&th_array[nth],&attr,q_update_threadfn_aecm,(void*)(&threaddata[nth]));
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
  thisnu=(nulow+((double)ci)*deltanu);

  free(q);
  return thisnu;
}


/* ud = sqrt(u^2+v^2) */
static double
ncp_weight(double ud) {
/*    fo(x) = 1/(1+alpha*exp(-x/A))
      A ~=30
*/
 if (ud>400.0) return 1.0; /* no effect on long baselines */
 //return 1.0/(1.0+0.4*exp(-0.05*ud)); 
 return 1.0/(1.0+1.8*exp(-0.05*ud)); 
}

static void *
threadfn_setblweight(void *data) {
 thread_data_baselinewt_t *t=(thread_data_baselinewt_t*)data;

 int ci;
 for (ci=0; ci<t->Nb; ci++) {
  /* get sqrt(u^2+v^2) */
  double uu=t->u[ci+t->boff]*t->freq0;
  double vv=t->v[ci+t->boff]*t->freq0;
  double a=ncp_weight(sqrt(uu*uu+vv*vv));
  t->wt[8*(ci+t->boff)]*=a;
  t->wt[8*(ci+t->boff)+1]*=a;
  t->wt[8*(ci+t->boff)+2]*=a;
  t->wt[8*(ci+t->boff)+3]*=a;
  t->wt[8*(ci+t->boff)+4]*=a;
  t->wt[8*(ci+t->boff)+5]*=a;
  t->wt[8*(ci+t->boff)+6]*=a;
  t->wt[8*(ci+t->boff)+7]*=a;
  //printf("%lf %lf %lf\n",uu,vv,a);
 }

 return NULL;
}


/* 
  taper data by weighting based on uv distance (for short baselines)
  for example: use weights as the inverse density function
  1/( 1+f(u,v) ) 
 as u,v->inf, f(u,v) -> 0 so long baselines are not affected 
 x: Nbase*8 x 1 (input,output) data
 u,v : Nbase x 1
 note: u = u/c, v=v/c here, so need freq to convert to wavelengths */
void
whiten_data(int Nbase, double *x, double *u, double *v, double freq0, int Nt) {
 pthread_attr_t attr;
 pthread_t *th_array;
 thread_data_baselinewt_t *threaddata;

 int ci,nth1,nth;
 int Nthb0,Nthb;

 Nthb0=(Nbase+Nt-1)/Nt;

 pthread_attr_init(&attr);
 pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

 if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
#ifndef USE_MIC
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
   exit(1);
 }

 if ((threaddata=(thread_data_baselinewt_t*)malloc((size_t)Nt*sizeof(thread_data_baselinewt_t)))==0) {
#ifndef USE_MIC
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
#endif
    exit(1);
 }


  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }

    threaddata[nth].Nb=Nthb;
    threaddata[nth].boff=ci;
    threaddata[nth].wt=x;
    threaddata[nth].u=u;
    threaddata[nth].v=v;
    threaddata[nth].freq0=freq0;

    pthread_create(&th_array[nth],&attr,threadfn_setblweight,(void*)(&threaddata[nth]));
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
}
