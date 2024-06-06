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
  if ((pt->M=(double*)calloc((size_t)2*lbfgs_m*2*lbfgs_m,sizeof(double)))==0) {
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
 * return :
 * optimality : scalar
 *
 * get the inf-norm of the projected gradient pp. 17, (6.1)
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
 *
 * compute breakpoints for Cauchy point pp 5-6, (4.1), (4.2), pp. 8, CP initialize \mathcal{F}
 */
static int
get_breakpoints(double *x, double *g, double *x_low, double *x_high, int m, double *t, double *d, int *F) {

  /* Note t <= 0, must be already done */

  /* d <= -g */
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
 * W: m x 2*lbfgs_m
 * M: 2*lbfgs_m x 2*lbfgs_m
 *
 * return :
 * xc: mx1 the generalized cauchy point
 * c: 2*lbfgs_mx1 initialization vector for subspace minimization, will be initialized to 0
 *
 * Generalized Cauchy point pp. 8-9, algorithm CP
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

  /* xc <= x */
  my_dcopy(m,x,1,xc,1);
  /* c <= 0 */
  memset(c,0,sizeof(double)*(size_t)2*lbfgs_m);

  double *p;
  if ((p=(double*)calloc((size_t)2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* p <= W^T d */
  my_dgemv('T',m,2*lbfgs_m,1.0,W,m,d,1,0.0,p,1);
  /* f' = fp <= - d^T d */
  double fp=-my_ddot(m,d,d);
  /* Mp <= M x p */
  double *Mp;
  if ((Mp=(double*)calloc((size_t)2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  my_dgemv('N',2*lbfgs_m,2*lbfgs_m,1.0,M,2*lbfgs_m,p,1,0.0,Mp,1);
  /* pMp <= p^T (M p) */
  double pMp=my_ddot(2*lbfgs_m,p,Mp);
  /* f'' = fpp <= -theta f' - p^T M p */
  double fpp=-theta*fp - pMp;
  /* keep a copy of fpp */
  double fpp0=-theta*fp;
  double dt_min=(fpp !=0.0 ? -fp/fpp : -fp/CLM_EPSILON);
  /* find lowest index i where F[i] is positive (minimum t) */
  int i=0;
  while (i<m && F[i]<0.0) {i++;}
  /* now we have a valid index i where F[i]>=0.0 */
  int b=F[i];
  double t=tt[b];
  double t_old=0.0;
  double dt=t-t_old;

  double *wb;
  if ((wb=(double*)calloc((size_t)2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  while (i<m && dt_min < dt) {
    xc[b]=(d[b]>0.0? x_high[b]: (d[b]<0.0? x_low[b] : xc[b]));
    double zb=xc[b]-x[b];

    /* c <= c + dt*p */
    my_daxpy(2*lbfgs_m,p,dt,c);
    double gb=g[b];
    /* wb(^T) <= b-th row of W, so wb : row as a column vector */
    my_dcopy(2*lbfgs_m,&W[b],m,wb,1);
    /* Mp <= M x c */
    my_dgemv('N',2*lbfgs_m,2*lbfgs_m,1.0,M,2*lbfgs_m,c,1,0.0,Mp,1);
    /* wb^T (M c) */
    double wb_Mc=my_ddot(2*lbfgs_m,wb,Mp);
    /* Mp <= M x p */
    my_dgemv('N',2*lbfgs_m,2*lbfgs_m,1.0,M,2*lbfgs_m,p,1,0.0,Mp,1);
    /* wb^T (M p) */
    double wb_Mp=my_ddot(2*lbfgs_m,wb,Mp);
    /* Mp <= M x wb */
    my_dgemv('N',2*lbfgs_m,2*lbfgs_m,1.0,M,2*lbfgs_m,wb,1,0.0,Mp,1);
    /* wb^T (M wb) */
    double wb_Mb=my_ddot(2*lbfgs_m,wb,Mp);

    /* fp <= fp + dt * fpp + gb*gb + theta*gb*zb - gb *wb^T* (M*c) */
    fp += dt*fpp + gb*gb + theta*gb*zb - gb*wb_Mc;
    /* fpp <= fpp - theta*gb*gb -2*gb*wb^T (M p) - gb*gb*wb^T (M wb) */
    fpp += -theta*gb*gb-2.0*gb*wb_Mp-gb*gb*wb_Mb;
    fpp =MAX(CLM_EPSILON*fpp0,fpp);
    /* p<= p + gb *wb */
    my_daxpy(2*lbfgs_m,wb,gb,p);

    d[b]=0.0;

    dt_min=(fpp !=0.0 ? -fp/fpp : -fp/CLM_EPSILON);
    t_old=t;
    i++;
    if (i<m) {
      b=F[i];
      t=tt[b];
      dt=t-t_old;
    }
  }

  /* perform final update */
  dt_min=MAX(dt_min,0.0);
  t_old+=dt_min;

#pragma GCC ivdep
  for (int cj=i; cj<m; cj++) {
    int idx=F[cj];
    xc[idx]+=t_old*d[idx];
  }
  /* c <= c + dt_min*p */
  my_daxpy(2*lbfgs_m,p,dt_min,c);

  free(wb);
  free(Mp);
  free(p);
  free(tt);
  free(d);
  free(F);
  return 0;
}
 
/* 
 * x_low,x_high: lower/upper bounds of parameters mx1
 * xc: mx1 the generalized cauchy point
 * du: n_free_varsx1 solution for unconstrained minimization
 * free_vars_index: n_free_varsx1 indices in [0,m-1] of the free variables
 * m: parameters
 * n_free_vars: number of free variables
 *
 * return :
 * alpha_star: scalar
 *
 * Equation (5.8), Page 8.
 */
static double
find_alpha(double *x_low, double *x_high, double *xc, double *du, int *free_vars_index, int m, int n_free_vars) {

  double alpha_star=1.0;
#pragma GCC ivdep
  for (int ci=0; ci<n_free_vars; ci++) {
    int idx=free_vars_index[ci];
    alpha_star=(du[ci]>0.0 ? MIN(alpha_star, (x_high[idx]-xc[idx])/du[ci]) 
        : (du[ci]<0.0 ? MIN(alpha_star, (x_low[idx]-xc[idx])/du[ci]) : alpha_star));
  }
  return alpha_star;
}

/* 
 * x: paramters mx1
 * g: gradient mx1
 * x_low,x_high: lower/upper bounds of parameters mx1
 * xc: mx1 the generalized cauchy point
 * c: 2*lbfgs_mx1 initialization vector for subspace minimization
 * m: parameters
 * lbfgs_m : memory size
 * theta: scale factor >0
 * W: m x 2*lbfgs_m
 * M: 2*lbfgs_m x 2*lbfgs_m
 *
 * return :
 * xbar: mx1 minimizer,
 * line_search_flag: bool
 *
 * subspace minimization for the quadratic model over free variables direct primal method, pp 12
 */
static int
subspace_min(double *x, double *g, double *x_low, double *x_high, double *xc, double *c, int m, int lbfgs_m, double theta, double *W, double *M, double *xbar, int *line_search_flag) {

  *line_search_flag=1;

  /* collect indices of free variables */
  int *free_vars_index;
  if ((free_vars_index=(int*)calloc((size_t)m,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  int n_free_vars=0;
  for (int ci=0; ci<m; ci++) {
    if ((x[ci] != x_low[ci]) && (x[ci] != x_high[ci])) {
      free_vars_index[n_free_vars]=ci;
      n_free_vars++;
    }
  }
  /* if no free variables found */
  if (n_free_vars==0) {
    /* xbar <= xc */
    my_dcopy(m,xc,1,xbar,1);
    *line_search_flag=0;
    free(free_vars_index);
    return 0;
  }

  /* WtZ = restriction of W to the free variables, size 2*lbfgs_m x |free_variables| 
   * each column of WtZ (2*lbfgs_m values) = row of i-th free variable in W (2*lbfgs_m values) */

  double *WtZ;
  if ((WtZ=(double*)calloc((size_t)2*lbfgs_m*n_free_vars,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  for (int ci=0; ci<n_free_vars; ci++) {
    my_dcopy(2*lbfgs_m,&W[free_vars_index[ci]],m,&WtZ[ci*2*lbfgs_m],1);
  }
  double *rr;
  if ((rr=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* Mc <= M x c */
  double *Mc;
  if ((Mc=(double*)calloc((size_t)2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  my_dgemv('N',2*lbfgs_m,2*lbfgs_m,1.0,M,2*lbfgs_m,c,1,0.0,Mc,1);
  /* rr <= -W*(Mc) */
  my_dgemv('N',m,2*lbfgs_m,1.0,W,m,Mc,1,0.0,rr,1);
  my_dscal(m,-1.0,rr);
  /* compute the reduced gradient of m(k), the quadratic model, restricted to free variables */
  /* rr <= g + theta*(xc-x) - W*(M*c) */
  my_daxpy(m,g,1.0,rr);
  my_daxpy(m,xc,theta,rr);
  my_daxpy(m,x,-theta,rr);
  free(Mc);

  double *r;
  if ((r=(double*)calloc((size_t)n_free_vars,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  for (int ci=0; ci<n_free_vars; ci++) {
    r[ci]=rr[free_vars_index[ci]];
  }
  free(rr);

  double *WtZr;
  if ((WtZr=(double*)calloc((size_t)2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* WtZr <= WtZ* r */
  my_dgemv('N',2*lbfgs_m,n_free_vars,1.0,WtZ,2*lbfgs_m,r,1,0.0,WtZr,1);
  double *v;
  if ((v=(double*)calloc((size_t)2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* v <= M*(WtZ*r) */
  my_dgemv('N',2*lbfgs_m,2*lbfgs_m,1.0,M,2*lbfgs_m,WtZr,1,0.0,v,1);
  free(WtZr);

  double *N;
  if ((N=(double*)calloc((size_t)2*lbfgs_m*2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  double invtheta=1.0/theta;
  /* N <= invtheta*WtZ*WtZ^T */
  my_dgemm('N','T',2*lbfgs_m,2*lbfgs_m,n_free_vars,invtheta,WtZ,2*lbfgs_m,WtZ,2*lbfgs_m,0.0,N,2*lbfgs_m);
  /* N <= eye(2*lbfgs_m) - M N */
  my_dgemm('N','N',2*lbfgs_m,2*lbfgs_m,2*lbfgs_m,-1.0,M,2*lbfgs_m,N,2*lbfgs_m,0.0,N,2*lbfgs_m);
  /* update diagonal */
  for (int ci=0; ci<2*lbfgs_m; ci++) {
    N[ci+ci*2*lbfgs_m]+=1.0;
  }
  
  /* solve N vv = v for v, replace v <= with solution vv */
  /* workspace query */
  double w[1];
  double *WORK=0;
  int status=my_dgels('N',2*lbfgs_m,2*lbfgs_m,1,N,2*lbfgs_m,v,2*lbfgs_m,w,-1);
  if (status) {
     fprintf(stderr,"%s: %d: LAPACK error\n",__FILE__,__LINE__);
     exit(1);
  }
  int lwork=(int)w[0];
  if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  status=my_dgels('N',2*lbfgs_m,2*lbfgs_m,1,N,2*lbfgs_m,v,2*lbfgs_m,WORK,lwork);
  if (status) {
     fprintf(stderr,"%s: %d: singular matrix\n",__FILE__,__LINE__);
     exit(1);
  }
  free(WORK);
  free(N);

  double *du;
  if ((du=(double*)calloc((size_t)n_free_vars,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* du <= -invtheta^2 * WtZ^T v */
  my_dgemv('T',2*lbfgs_m,n_free_vars,-invtheta*invtheta,WtZ,2*lbfgs_m,v,1,0.0,du,1);
  free(WtZ);
  /* du <= -invtheta*r -invtheta^2 * WtZ^T v */
  my_daxpy(n_free_vars,r,-invtheta,du);
  free(r);
  free(v);

  /* find alpha_star */
  double alpha_star=find_alpha(x_low,x_high,xc,du,free_vars_index,m,n_free_vars);

  /* compute subspace minimization */
  /* d_star = du <= alpha_star * du */
  my_dscal(n_free_vars,alpha_star,du);
  /* xbar <= xc */
  my_dcopy(m,xc,1,xbar,1);
#pragma GCC ivdep
  for (int ci=0; ci<n_free_vars; ci++) {
    xbar[free_vars_index[ci]] += du[ci];
  }

  free(free_vars_index);
  free(du);
  return 0;
}
 
/* 
 * cost_func: cost function R^m -> 1
 * grad_func: gradient function R^m -> R^m
 * adata : additional data needed by cost/grad functions
 * f0: initial cost (scalar)
 * x0: paramters mx1
 * g0: gradient mx1
 * p: mx1 initial search direction
 * m: parameters
 * alpha_lo,alpha_hi : lower/upper limit for alpha
 *
 * return :
 * alpha: zoomed in step size
 *
 * Alg 3.6, pp. 61, Numerical optimization Nocedal & Wright
 */
static double
alpha_zoom(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   void *adata,
   double f0, double *x0, double *g0, double *p, int m, double alpha_lo, double alpha_hi) {
  const double c1=1e-4;
  const double c2=0.9;
  int i=0;
  int max_iters=20;

  double *x,*x_lo,*g_i;
  if ((x=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((x_lo=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((g_i=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  double dphi0=my_ddot(m,g0,p);
  double alpha,alpha_i;
  while (1) {
    alpha_i=0.5*(alpha_lo+alpha_hi);
    alpha=alpha_i;
    /* x <= x0 + alpha_i *p */
    my_dcopy(m,x0,1,x,1);
    my_daxpy(m,p,alpha_i,x);
    double f_i=cost_func(x,m,adata);
    /* x_lo <= x0 + alpha_lo *p */
    my_dcopy(m,x0,1,x_lo,1);
    my_daxpy(m,p,alpha_lo,x_lo);
    double f_lo=cost_func(x_lo,m,adata);
    if ((f_i > f0 + c1*alpha_i*dphi0) || (f_i>=f_lo)){
      alpha_hi=alpha_i;
    } else {
      grad_func(x,g_i,m,adata);
      double dphi=my_ddot(m,g_i,p);
      if ((fabs(dphi) <= -c2*dphi0)) {
        alpha=alpha_i;
        break;
      }
      if (dphi*(alpha_hi-alpha_lo)>=0.0) {
        alpha_hi=alpha_lo;
      }
      alpha_lo=alpha_i;
    }
    i++;
    if (i>max_iters) {
      alpha=alpha_i;
      break;
    }
  }

  free(x);
  free(x_lo);
  free(g_i);

  return alpha;
}

/* 
 * cost_func: cost function R^m -> 1
 * grad_func: gradient function R^m -> R^m
 * adata : additional data needed by cost/grad functions
 * f0: initial cost (scalar)
 * x0: paramters mx1
 * g0: gradient mx1
 * p: mx1 initial search direction
 * m: parameters
 *
 * return :
 * alpha: step size
 *
 * Compute a line search to satisfy the strong Wolfe conditions.
 * Algorithm 3.5. Page 60. "Numerical Optimization". Nocedal & Wright.
 */
static double
strong_wolfe(
   double (*cost_func)(double *p, int m, void *adata),
   void (*grad_func)(double *p, double *g, int m, void *adata),
   void *adata,
   double f0, double *x0, double *g0, double *p, int m) {

  /* constants */
  const double c1=1e-4;
  const double c2=0.9;
  const double alpha_max=2.5;
  double alpha_im1=0.0;
  double alpha_i=1.0;
  double f_im1=f0;
  double dphi0=my_ddot(m,g0,p);
  int i=0;
  int max_iters=20;
  double alpha;

  double *x,*g_i;
  if ((x=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((g_i=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  while (1) {
    /* x <= x0 + alpha_i *p */
    my_dcopy(m,x0,1,x,1);
    my_daxpy(m,p,alpha_i,x);
    double f_i=cost_func(x,m,adata);
    if ((f_i > f0 + c1*dphi0) || ((i>0) && (f_i>f_im1))) {
      alpha=alpha_zoom(cost_func,grad_func,adata,f0,x0,g0,p,m,alpha_im1,alpha_i);
      break;
    }
    grad_func(x,g_i,m,adata);
    double dphi=my_ddot(m,g_i,p);
    if (fabs(dphi)<= -c2*dphi0) {
      alpha=alpha_i;
      break;
    }
    if (dphi>=0.0) {
      alpha=alpha_zoom(cost_func,grad_func,adata,f0,x0,g0,p,m,alpha_i,alpha_im1);
      break;
    }
    alpha_im1=alpha_i;
    f_im1=f_i;
    alpha_i +=0.8*(alpha_max-alpha_i);
    if (i>max_iters) {
      alpha=alpha_i;
      break;
    }
    i++;
  }

  free(x);
  free(g_i);
  return alpha;
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

  double *xc,*c;
  if ((xc=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((c=(double*)calloc((size_t)2*lbfgs_m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  double *xbar;
  if ((xbar=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  double *q;
  if ((q=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  double *s,*y;
  if ((s=(double*)calloc((size_t)m,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((y=(double*)calloc((size_t)m,sizeof(double)))==0) {
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

  int line_search_flag;
  while (ck<itmax && isnormal(gradnrm) && optimality>CLM_STOP_THRESH) {
    optimality=get_optimality(xk,gk,p_low,p_high,m);

    my_dcopy(m,xk,1,xold,1);
    my_dcopy(m,gk,1,gold,1);

    get_cauchy_point(xk, gk, p_low, p_high, m, lbfgs_m, theta, pt.W, pt.M, xc, c);
    subspace_min(xk, gk, p_low, p_high, xc, c, m, lbfgs_m, theta, pt.W, pt.M, xbar, &line_search_flag);

    double alpha=1.0;
    /* find search direction q <= xbar - x*/
    my_dcopy(m,xbar,1,q,1);
    my_daxpy(m,xk,-1.0,q);
    if (line_search_flag) {
      alpha = strong_wolfe(cost_func,grad_func,adata,f,xk,gk,q,m);
    }
    /* update solution x <= x + alpha ( xbar - x )*/
    my_daxpy(m,q,alpha,xk);

    f=cost_func(xk,m,adata);
    grad_func(xk,gk,m,adata);
    /* curvature pair */
    /* s = x-xold */
    my_dcopy(m,xk,1,s,1);
    my_daxpy(m,xold,-1.0,s);
    /* y = g-gold */
    my_dcopy(m,gk,1,y,1);
    my_daxpy(m,gold,-1.0,y);

    double curv=my_ddot(m,s,y);
    if (curv < CLM_STOP_THRESH) {
      fprintf(stderr,"%s: %d: negative curvature detected, skipping\n",__FILE__,__LINE__);
      ck++;
      continue;
    }

    if (ck<lbfgs_m) {
      /* add pair to history */
      my_dcopy(m,y,1,&pt.Y[ck*m],1);
      my_dcopy(m,s,1,&pt.S[ck*m],1);
    } else {
      /* history already full, remove oldest curvature pair, 
       * just move colums 1,...lbfs_m-1 to columns 0...lbfgs_m-2 */
      memmove(pt.Y,&pt.Y[m],m*(lbfgs_m-1)*sizeof(double));
      memmove(pt.S,&pt.S[m],m*(lbfgs_m-1)*sizeof(double));
      my_dcopy(m,y,1,&pt.Y[(lbfgs_m-1)*m],1);
      my_dcopy(m,s,1,&pt.S[(lbfgs_m-1)*m],1);
    }



    ck++;
  }

  lbfgsb_persist_clear(&pt);
  free(xk);
  free(gk);
  free(xold);
  free(gold);
  free(xc);
  free(c);
  free(xbar);
  free(q);
  free(s);
  free(y);
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
