/*
 *
 Copyright (C) 2010- Sarod Yatawatta <sarod@users.sf.net>  
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


#include "buildsky.h"


void
print_levmar_info(double e_0, double e_final,int itermax, int info, int fnum, int jnum, int lnum) {
 printf("\nOptimization terminated with %d iterations, reason: ",itermax);
 switch(info) {
  case 1:
   printf("stopped by small gradient J^T e.\n");
   break;
  case 2:
   printf("stopped by small Dp.\n");
   break;
  case 3:
   printf("stopped by itmax.\n");
   break;
  case 4:
   printf("singular matrix. Restart from current p with increased mu.\n");
   break;
  case 5:
   printf("no further error reduction is possible. Restart with increased mu.\n");
   break;
  case 6:
   printf("stopped by small ||e||_2.\n");
   break;
  case 7:
   printf("stopped by invalid (i.e. NaN or Inf) \"func\" values. This is a user error.\n");
   break;
  default:
   printf("Unknown.\n");
   break;
 }
 printf("Error from %lf to %lf, Evaluations: %d functions %d Jacobians %d Linear systems\n",e_0, e_final,fnum,jnum,lnum);
}

double
fit_single_point0(hpixel *parr, int npix, double bmaj, double bmin, double bpa, int  maxiter, double *ll1, double *mm1, double *sI1){

 double xm,sumI,peakIneg,peakIpos,peakI;
 int ci;
 double l,m,ll,mm;

 xm=sumI=0.0;
 ll=mm=0.0;
 peakIneg=INFINITY_L;
 peakIpos=-INFINITY_L; /* FIX: use proper limits */
 for (ci=0; ci<npix; ci++) {
       sumI+=parr[ci].sI; /* FIX negative values */
       if (peakIpos<parr[ci].sI) { /* detect -ve or +ve peak */
         peakIpos=parr[ci].sI;
       }
       if (peakIneg>parr[ci].sI) { /* detect -ve or +ve peak */
         peakIneg=parr[ci].sI;
       }
       ll+=parr[ci].sI*parr[ci].l;
       mm+=parr[ci].sI*parr[ci].m;
 }
 ll/=sumI;
 mm/=sumI;
 /* now decide if -ve or +ve peak is used for model, choose 
  one with highest magnitude */
 if (fabs(peakIneg)> fabs(peakIpos)) {
  peakI=peakIneg;
 } else {
  peakI=peakIpos;
 }
 /* calculate error */
 sumI=0.0;
 for (ci=0; ci<npix; ci++) {
  /* rotate by beam pa */
  l=(-(parr[ci].l-ll)*sin(bpa)+(parr[ci].m-mm)*cos(bpa))/bmaj;
  m=(-(parr[ci].l-ll)*cos(bpa)-(parr[ci].m-mm)*sin(bpa))/bmin;
  xm=parr[ci].sI-peakI*exp(-(l*l+m*m));
  /* squared error */
  sumI+=xm*xm;
 }

#ifdef DEBUG
 printf("Error = %lf\n",sumI);
#endif
 /* AIC=2*k+N*ln(sum_error) */
 /* k=3, flux, 2 positions */
 sumI=2*3+npix*log(sumI)*2.0;
 *ll1=ll;
 *mm1=mm;
 *sI1=peakI;
 return sumI;
}

/* LS fit for one source */
/* p: mx1, m=3 for one source 
   x: nx1
   maxiter: max LS iterations
*/
static int 
dweighted_ls_fit(double *p, double *x, int m, int n, int maxiter, void *data) {
 fit_double_point_data *dp=(fit_double_point_data*)data;
 /* not p[0]: l, p[1]: m, p[2]: I */

  double *xn;
  double A[9]; /* 3x3 matrix */
  double b[3]; /* RHS vector */
  int ci;
  double xy,x2,y2,logz,z2,xz2,yz2,x2z2,xyz2,y2z2,sumterm,a3,a4,a5;
  double b00,b01,b11,denom;
  double lest,mest,logI,prod1,prod2,prod3;
  double ll,mm;
  double pdelta; /* norm of p update to stop */

  double *WORK=0;
  int lwork=0;
  double w[1];
  int status,niter;

  /* precalculate these !*/
  a3=dp->a3;
  a4=dp->a4;
  a5=dp->a5;

  b00=-2.0*a4; b01=-a3; b11=-2.0*a5; denom=1.0/(b00*b11-b01*b01);
  /* B=[b00, b01; b01, b11] */

  if ((xn=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  memcpy(xn,x,(size_t)n*sizeof(double));
  /* setup memory for LAPACK */
  /* workspace query */
  status=my_dgels('N',3,3,1,A,3,b,3,w,-1);
  if (!status) {
     lwork=(int)w[0];
     if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
  }

  pdelta=INFINITY_L;
  memset(p,0,sizeof(double)*3);
  niter=0;
  while (niter<maxiter && pdelta>TOL_L) {
  memset(b,0,sizeof(double)*3);
  memset(A,0,sizeof(double)*9);
  for (ci=0; ci<n; ci++) {
   if (x[ci]>TOL_L) { /* cannot find log() for negative */
    xy=(dp->parr[ci].l)*(dp->parr[ci].m);
    x2=(dp->parr[ci].l)*(dp->parr[ci].l);
    y2=(dp->parr[ci].m)*(dp->parr[ci].m);
    logz=log(x[ci]);
    z2=xn[ci]*xn[ci];
    xz2=(dp->parr[ci].l)*z2;
    yz2=(dp->parr[ci].m)*z2;
    x2z2=x2*z2;
    xyz2=xy*z2;
    y2z2=y2*z2;
    sumterm=logz-a3*xy-a4*x2-a5*y2;
    A[0]+=z2;
    A[1]+=xz2;
    A[2]+=yz2;
    A[3]+=xz2;
    A[4]+=x2z2;
    A[5]+=xyz2;
    A[6]+=yz2;
    A[7]+=xyz2;
    A[8]+=y2z2;
    b[0]+=z2*sumterm;
    b[1]+=xz2*sumterm;
    b[2]+=yz2*sumterm;
    
   /* ll=(-(dp->parr[ci].l-p[0])*sin(dp->bpa)+(dp->parr[ci].m-p[1])*cos(dp->bpa))/dp->bmaj;
    mm=(-(dp->parr[ci].l-p[0])*cos(dp->bpa)-(dp->parr[ci].m-p[1])*sin(dp->bpa))/dp->bmin;
    x[ci]+=p[2]*exp(-(ll*ll+mm*mm)); */
   }
  }

  //printf("A=[%lf, %lf, %lf; %lf, %lf, %lf; %lf, %lf, %lf];\n",A[0],A[3],A[6],A[1],A[4],A[7],A[2],A[5],A[8]);
  //printf("b=[%lf; %lf; %lf];\n",b[0],b[1],b[2]);
  status=my_dgels('N',3,3,1,A,3,b,3,WORK,lwork);

  //printf("sol=[%lf; %lf; %lf];\n",b[0],b[1],b[2]);

  lest=denom*(b[1]*b11-b[2]*b01);
  mest=denom*(b[2]*b00-b[1]*b01);

  logI=b[0];
  prod1=lest*sin(dp->bpa);
  prod2=mest*cos(dp->bpa);
  prod3=lest*mest*sin(2.0*dp->bpa);
  logI+=(prod1*prod1+prod2*prod2-prod3)/(dp->bmaj*dp->bmaj);
  prod1=lest*cos(dp->bpa);
  prod2=mest*sin(dp->bpa);
  logI+=(prod1*prod1+prod2*prod2+prod3)/(dp->bmin*dp->bmin);
  
  prod2=exp(logI);
  prod1=p[2]-prod2;
  pdelta=prod1*prod1;
  p[2]=prod2;
  prod1=p[0]-lest;
  p[0]=lest;
  pdelta+=prod1*prod1;
  prod1=p[1]-mest;
  p[1]=mest;
  pdelta+=prod1*prod1;
  pdelta=sqrt(pdelta);
  
  //printf("iter %d, est %lf,%lf,%lf, ||p||=%lf\n",niter,p[0],p[1],p[2],pdelta);
    /* update xn with extimated values */
    for (ci=0; ci<n; ci++) {
     ll=(-(dp->parr[ci].l-p[0])*sin(dp->bpa)+(dp->parr[ci].m-p[1])*cos(dp->bpa))/dp->bmaj;
     mm=(-(dp->parr[ci].l-p[0])*cos(dp->bpa)-(dp->parr[ci].m-p[1])*sin(dp->bpa))/dp->bmin;
     xn[ci]=p[2]*exp(-(ll*ll+mm*mm)); 
    }
   niter++;
  }

/*  xy=xv(ci)*yv(ci);
  x2=xv(ci)*xv(ci);
  y2=yv(ci)*yv(ci);
  logz=log(zvdata(ci)); % only this is from data
  z2=zv(ci)*zv(ci);
  xz2=xv(ci)*z2;
  yz2=yv(ci)*z2;
  x2z2=xv(ci)*xz2;
  xyz2=xy*z2;
  y2z2=yv(ci)*yz2;
  sumterm=logz-a3*xy-a4*x2-a5*y2;
  A(1,1)=A(1,1)+z2;
  A(1,2)=A(1,2)+xz2;
  A(1,3)=A(1,3)+yz2;
  A(2,1)=A(2,1)+xz2;
  A(2,2)=A(2,2)+x2z2;
  A(2,3)=A(2,3)+xyz2;
  A(3,1)=A(3,1)+yz2;
  A(3,2)=A(3,2)+xyz2;
  A(3,3)=A(3,3)+y2z2;
  b(1)=b(1)+z2*sumterm;
  b(2)=b(2)+xz2*sumterm;
  b(3)=b(3)+yz2*sumterm;
*/

   free(xn);
   free(WORK);
   return 0;
}

/* fitting function  - single point 
 p: mx1 format l0,m0,sI0
 x: nx1 pixel fluxes, from model
*/ 
static void
mylm_fit_single(double *p, double *x, int m, int n, void *data) {
 fit_double_point_data *dp=(fit_double_point_data*)data;
 int ci;

 double ll,mm;
 double sbpa;
 double cbpa;
 sincos(dp->bpa,&sbpa,&cbpa);
 double invmaj=1.0/dp->bmaj;
 double invmin=1.0/dp->bmin;

 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 for (ci=0; ci<n; ci++) {
  ll=(-(dp->parr[ci].l-p[0])*sbpa+(dp->parr[ci].m-p[1])*cbpa)*invmaj;
  mm=(-(dp->parr[ci].l-p[0])*cbpa-(dp->parr[ci].m-p[1])*sbpa)*invmin;
  x[ci]+=p[2]*exp(-(ll*ll+mm*mm));
 }
}



double
fit_single_point(hpixel *parr, int npix, double bmaj, double bmin, double bpa, int  maxiter, double *ll1, double *mm1, double *sI1){
 int ci;
 double *p, // params m x 1
     *x; // observed data n x 1, the image pixel fluxes
 int m,n;

 double opts[CLM_OPTS_SZ], info[CLM_INFO_SZ];
 
 double sumI;
 double ll,mm;

 fit_double_point_data lmdata;

 opts[0]=CLM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-15;
  opts[4]=-CLM_DIFF_DELTA; 

 m=3; /* 1x2 positions, 1x1 flux */
 n=npix; /* no of pixels */

 if ((p=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((x=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }


 sumI=0.0;
 ll=mm=0.0;
 for (ci=0; ci<npix; ci++) {
       sumI+=parr[ci].sI; /* FIX negative values */
       ll+=parr[ci].sI*parr[ci].l;
       mm+=parr[ci].sI*parr[ci].m;
       x[ci]=parr[ci].sI;
 }
 ll/=sumI;
 mm/=sumI;
 sumI/=(double)npix;

 /* initial values */ 
 p[0]=ll;
 p[1]=mm;
 p[2]=sumI;

 lmdata.parr=parr;
 lmdata.bmaj=bmaj;
 lmdata.bmin=bmin;
 lmdata.bpa=bpa;

 //dlevmar_dif(mylm_fit_single, p, x, 3, n, maxiter, opts, info, NULL, NULL, (void*)&lmdata);  // no Jacobian
 clevmar_der_single_nocuda(mylm_fit_single, NULL, p, x, 3, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian

#ifdef DEBUG
  print_levmar_info(info[0],info[1],(int)info[5], (int)info[6], (int)info[7], (int)info[8], (int)info[9]);
  printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
#endif

 *ll1=p[0];
 *mm1=p[1];
 *sI1=p[2];

 free(p);
 free(x);

 /* AIC, 3 parms */
 return 2*3+npix*log(info[1])*2.0;
}



/* fitting function  - N points
 p: mx1 format l0,m0,sI0, l1,m1,sI1, etc
 x: nx1 pixel fluxes, from model
*/ 
static void
mylm_fit_N(double *p, double *x, int m, int n, void *data) {
 fit_double_point_data *dp=(fit_double_point_data*)data;
 int ci,N,cj;

 double ll,mm;

 N=m/3;
 double sbpa;
 double cbpa;
 sincos(dp->bpa,&sbpa,&cbpa);
 double invmaj=1.0/dp->bmaj;
 double invmin=1.0/dp->bmin;

 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 for (ci=0; ci<n; ci++) {
  for (cj=0; cj<N; cj++) {
  ll=(-(dp->parr[ci].l-p[3*cj])*sbpa+(dp->parr[ci].m-p[3*cj+1])*cbpa)*invmaj;
  mm=(-(dp->parr[ci].l-p[3*cj])*cbpa-(dp->parr[ci].m-p[3*cj+1])*sbpa)*invmin;
  x[ci]+=p[3*cj+2]*exp(-(ll*ll+mm*mm));
  }
 }
}


/* N>1 */
double
fit_N_point_em(hpixel *parr, int npix, double bmaj, double bmin, double bpa, int maxiter, int max_em_iter, double *ll, double *mm, double *sI, int N, int Nh, hpoint *hull){

 int ci,cj,ck;
 double *p, // params m x 1
     *x; // observed data n x 1, the image pixel fluxes
 double *xdummy, *xsub; //extra arrays
 int m,n;
 double *b; /* affine combination */

 double opts[CLM_OPTS_SZ], info[CLM_INFO_SZ];

 double penalty; /* penalty for solutions with components out of pixel range */

 fit_double_point_data lmdata;

 opts[0]=CLM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-15;
  opts[4]=-CLM_DIFF_DELTA; // relevant only if the Jacobian is approximated using finite differences; specifies forward differencing
  //opts[4]=-LM_DIFF_DELTA; // specifies central differencing to approximate Jacobian; more accurate but more expensive to compute!
  /* I: opts[0-4] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
   * scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
   * the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
   * If \delta<0, the Jacobian is approximated with central differences which are more accurate
   * (but slower!) compared to the forward differences employed by default. 
  */

 //maxiter=maxiter/max_em_iter;
 m=3*N; /* Nx2 positions, Nx1 flux format l0,m0,sI0, l1,m1,sI1, etc*/
 n=npix; /* no of pixels */

 if ((p=(double*)calloc((size_t)(m),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((x=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((xsub=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((xdummy=(double*)calloc((size_t)(n),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }
 if ((b=(double*)calloc((size_t)(N),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
 }

 for (ci=0; ci<npix; ci++) {
       x[ci]=parr[ci].sI;
 }

 lmdata.parr=parr;
 lmdata.bmaj=bmaj;
 lmdata.bmin=bmin;
 lmdata.bpa=bpa;
 double sbpa,cbpa;
 sincos(bpa,&sbpa,&cbpa);
 double sbpa_bmaj=sbpa/bmaj;
 double sbpa_bmin=sbpa/bmin;
 double cbpa_bmaj=cbpa/bmaj;
 double cbpa_bmin=cbpa/bmin;
 lmdata.a3=sin(2.0*bpa)*(1.0/(bmaj*bmaj)-1.0/(bmin*bmin));
 //lmdata.a4=-(sin(bpa)/bmaj)*(sin(bpa)/bmaj)-(cos(bpa)/bmin)*(cos(bpa)/bmin);
 //lmdata.a5=-(cos(bpa)/bmaj)*(cos(bpa)/bmaj)-(sin(bpa)/bmin)*(sin(bpa)/bmin);
 lmdata.a4=-(sbpa_bmaj)*(sbpa_bmaj)-(cbpa_bmin)*(cbpa_bmin);
 lmdata.a5=-(cbpa_bmaj)*(cbpa_bmaj)-(sbpa_bmin)*(sbpa_bmin);

 /* initial values are at peak pixel locations, 
    when each component is subtracted one by one */
 memcpy(xdummy,x,(size_t)(n)*sizeof(double));
 for (ci=0; ci<N; ci++) {
   /* find peak value and set position (start from 1) */  
   cj=my_idamax(n,xdummy,1);
   cj--;
   p[ci*3]=parr[cj].l;
   p[ci*3+1]=parr[cj].m;
   p[ci*3+2]=parr[cj].sI;
   /* subtract this component from data */
   mylm_fit_single(&p[3*ci], xsub, 3, n, (void*)&lmdata);
   /* xdummy=xdummy-b*xsub */
   my_daxpy(n, xsub, -1.0, xdummy);
 }

 for (ci=0; ci<max_em_iter; ci++) {
   for (cj=0; cj<N; cj++) {
    /* calculate contribution from hidden data, subtract from x */
    memcpy(xdummy,x,(size_t)(n)*sizeof(double));
    for (ck=0; ck<N; ck++) {
     if (ck!=cj) {
       mylm_fit_single(&p[3*ck], xsub, 3, n, (void*)&lmdata);
       /* xdummy=xdummy-xsub */
       my_daxpy(n, xsub, -1.0, xdummy);
     }
    }
    //dlevmar_dif(mylm_fit_single, &p[3*cj], xdummy, 3, n, maxiter, opts, info, NULL, NULL, (void*)&lmdata);  // no Jacobian
    //clevmar_der_single_nocuda(mylm_fit_single, NULL, &p[3*cj], xdummy, 3, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
    // weighted least squares estimation
    dweighted_ls_fit(&p[3*cj], xdummy, 3, n, maxiter, (void*)&lmdata);
   } 
 }


 /* final global fit */
 //dlevmar_dif(mylm_fit_N, p, x, m, n, maxiter, opts, info, NULL, NULL, (void*)&lmdata);  // no Jacobian
 clevmar_der_single_nocuda(mylm_fit_N, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
#ifdef DEBUG
  print_levmar_info(info[0],info[1],(int)info[5], (int)info[6], (int)info[7], (int)info[8], (int)info[9]);
  printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
#endif
   /* O: information regarding the minimization. Set to NULL if don't care
   * info[0]= ||e||_2 at initial p.
   * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
   * info[5]= # iterations,
   * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
   * 2 - stopped by small Dp
   * 3 - stopped by itmax
   * 4 - singular matrix. Restart from current p with increased mu 
   * 5 - no further error reduction is possible. Restart with increased mu
   * 6 - stopped by small ||e||_2
   * 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
   * info[7]= # function evaluations
   * info[8]= # Jacobian evaluations
   * info[9]= # linear systems solved, i.e. # attempts for reducing error
   */

 /* check for solutions such that l_min <= ll <= l_max and m_min <= mm <= m_max */
 penalty=0.0;
 for (ci=0; ci<N; ci++) {
   ll[ci]=p[3*ci];
   mm[ci]=p[3*ci+1];
   sI[ci]=p[3*ci+2];
   /* if not inside hull, add penalty */
   if (!inside_hull(Nh, hull, ll[ci],mm[ci])) {
    penalty+=INFINITY_L;
   }
 }

 free(p);
 free(x);
 free(b);
 free(xsub);
 free(xdummy);

 /* AIC, 3*N parms */
 return 2*3*N+npix*log(info[1])*2.0+penalty;
}
