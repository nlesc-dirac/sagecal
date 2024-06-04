/*
 *
 Copyright (C) 2024 Sarod Yatawatta <sarod@users.sf.net>  
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

int 
lbfgsb_persist_init(persistent_lbfgsb_data_t *pt, int n_minibatch, 
    int m, int n, int lbfgs_m, int Nt) {

  pt->lbfgs_m=lbfgs_m;
  pt->m=m;
  pt->Nt=Nt;

  /* W: m x lbfgs_m*2 */
  if ((pt->W=(double*)calloc((size_t)m*2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* Y: m x lbfgs_m */
  if ((pt->Y=(double*)calloc((size_t)m*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* S: m x lbfgs_m */
  if ((pt->S=(double*)calloc((size_t)m*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* M: 2*lbfgs_m x 2*lbfgs_m */
  if ((pt->S=(double*)calloc((size_t)2*lbfgs_m*2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }


  return 0;
}


int 
lbfgsb_persist_clear(persistent_lbfgsb_data_t *pt) {

  free(pt->W);
  free(pt->Y);
  free(pt->S);
  free(pt->M);
  return 0;
}

/* 
 * x: paramters mx1
 * g: gradient mx1
 * x_low,x_high: lower/upper bounds of parameters mx1
 * m: parameters
 *
 * return 
 * optimality
 */
static double
get_optimality(double *x, double *g, double *x_low, double *x_high, int m) {
  double *pg;
  if ((pg=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* projected gradient, pg= x-g */
  my_dcopy(m,x,1,pg,1);
  my_daxpy(m,g,-1.0,pg);
#pragma GCC ivdep
  for (int ci=0; ci<m; ci++) {
    pg[ci]=(pg[ci] < x_low[ci] ? x_low[ci] : (pg[ci] > x_high[ci] ? x_high[ci] : pg[ci]));
  }

  /* pg = pg - x */
  my_daxpy(m,x,-1.0,pg);

  /* return max(abs(pg)) */
  int i_max=my_idamax(m,pg,1);
  double optimality=pg[i_max];
  free(pg);

  return optimality;
}

/* for sorting array of values and return the indices in sorted order */
typedef struct value_index_t_{
 double val;
 int idx;
} value_index_t;

/* comparison */
static int
compare_coordinates(const void *a, const void *b) {
 const value_index_t *da=(const value_index_t *)a;
 const value_index_t *db=(const value_index_t *)b;

 return(da->val>=db->val?1:-1);
}

/* 
 * x: paramters mx1
 * g: gradient mx1
 * x_low,x_high: lower/upper bounds of parameters mx1
 * m: parameters
 *
 * return :
 * t: mx1 breakpoints : initialized to 0
 * d: mx1 search directions
 * F: mx1 indices that sort t from low to high
 */
static int
get_breakpoints(double *x, double *g, double *x_low, double *x_high, int m, double *t, double *d, int *F) {

  /* t <= 0, already done */
  /* d = -g */
  my_dcopy(m,g,1,d,1);
  my_dscal(m,-1.0,d);

#pragma GCC ivdep
  for (int ci=0; ci<m; ci++) {
    t[ci]=(g[ci] < 0.0 ? (x[ci]-x_high[ci])/g[ci] :
        (g[ci] > 0.0 ? (x[ci]-x_low[ci])/g[ci] : CLM_DBL_MAX));
    if (t[ci]<CLM_EPSILON) {
      d[ci]=0.0;
    }
  }

  /* sort t into ascending order */
  /* create array of index,value for sorting t */
  value_index_t *t_idx;
  if ((t_idx=(value_index_t*)calloc((size_t)m,sizeof(value_index_t)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

#pragma GCC ivdep
  for (int ci=0; ci<m; ci++) {
    t_idx[ci].val=t[ci];
    t_idx[ci].idx=ci;
    printf("before %d %lf\n",t_idx[ci].idx,t_idx[ci].val);
  }
  qsort(t_idx,m,sizeof(value_index_t),compare_coordinates);

#pragma GCC ivdep
  for (int ci=0; ci<m; ci++) {
    printf("after %d %lf\n",t_idx[ci].idx,t_idx[ci].val);
    F[ci]=t_idx[ci].idx;
  }

  free(t_idx);
  return 0;
}


/* 
 * x: paramters mx1
 * g: gradient mx1
 * x_low,x_high: lower/upper bounds of parameters mx1
 * m: parameters
 * lbfgs_m : memory size
 * theta: scale factor >0
 * W: m x lbfgs_m
 * M: 2*lbfgs_m x 2*lbfgs_m
 *
 * return :
 * xc: mx1 the generalized cauchy point
 * c: 2mx1 initialization vector for subspace minimization
 */
static int
get_cauchy_point(double *x, double *g, double *x_low, double *x_high, int m, int lbfgs_m, double theta, double *W, double *M, double *xc, double *c) {

  double *tt, *d;
  int *F;
  /* note tt initialized to tt[]=0 */
  if ((tt=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((d=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((F=(int*)calloc((size_t)m,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  get_breakpoints(x, g, x_low, x_high, m, tt, d, F);


  free(tt);
  free(d);
  free(F);
  return 0;
}
 
 
static int
lbfgsb_fit_fullbatch(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   double *p, double *p_low, double *p_high, int m, int itmax, int lbfgs_m, void *adata) {

  /* create own persistent data struct for memory allocation */
  persistent_lbfgsb_data_t pt;
  lbfgsb_persist_init(&pt,1,m,1,lbfgs_m,8);
  double theta=1.0;
  int ck=0;

  double *xk,*gk,*xold,*gold;
  if ((gk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xk=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((gold=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((xold=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }


  /* copy parameters */
  my_dcopy(m,p,1,xk,1);
  /* cost */
  double f=cost_func(xk,m,adata);
  /* gradient */
  grad_func(xk,gk,m,adata);
  /* grad norm */
  double gradnrm=my_dnrm2(m,gk);
  
  double optimality=get_optimality(xk,gk,p_low,p_high,m);
  while (ck<itmax && isnormal(gradnrm) && optimality>CLM_STOP_THRESH) {
    optimality=get_optimality(xk,gk,p_low,p_high,m);

    my_dcopy(m,p,1,xold,1);
    my_dcopy(m,gk,1,gold,1);
    ck++;
  }

  lbfgsb_persist_clear(&pt);
  free(xk);
  free(gk);
  free(xold);
  free(gold);
  return 0;
}


static int
lbfgsb_fit_minibatch(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   double *p, double *p_low, double *p_high, int m, int itmax, int lbfgs_m, void *adata, persistent_lbfgsb_data_t *indata) {

  return 0;
}



int
lbfgsb_fit(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   double *p, double *p_low, double *p_high, int m, int itmax, int lbfgs_m, void *adata, persistent_lbfgsb_data_t *indata) {

  int retval=0;
  if (!indata) {
    retval=lbfgsb_fit_fullbatch(cost_func,grad_func,p,p_low,p_high,m,itmax,lbfgs_m,adata);
  } else {
    retval=lbfgsb_fit_minibatch(cost_func,grad_func,p,p_low,p_high,m,itmax,lbfgs_m,adata,indata);
  }

  return retval;
}
