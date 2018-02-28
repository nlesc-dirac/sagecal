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

//#define DEBUG



/** keep interface almost the same as in levmar **/
int
clevmar_der_single_nocuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10], 
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  int nu=2,nu2;
  double p_L2, Dp_L2=DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0, pDp_eL2, init_p_eL2;
  double tmp,mu=0.0;
  double tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  double *hxd,*hxm=0;
  double *ed;
  double *jac;

  double *jacTjacd,*jacTjacd0;

  double *pnew,*Dpd,*bd;
  double *aones;
  double *jacTed;

  /* used in QR solver */
  double *WORK;
  int lwork=0;
  double w[1];

  int status;

  /* used in SVD solver */
  double *Ud;
  double *VTd;
  double *Sd;

  /* for Jacobian evaluation */
  int jac_given;
  double delta,tempp,ddiff;
  if (!jacf) {
   jac_given=0;
   /* need more memory for jacobian calculation */
   if ((hxm=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
   }
   delta=CLM_DIFF_DELTA;
  } else {
   jac_given=1;
  }

  int solve_axb=linsolv;

  /* setup default settings */
  if(opts){
    tau=opts[0];
    eps1=opts[1];
    eps2=opts[2];
    eps2_sq=opts[2]*opts[2];
    eps3=opts[3];
  } else {
    tau=CLM_INIT_MU;
    eps1=CLM_STOP_THRESH;
    eps2=CLM_STOP_THRESH;
    eps2_sq=CLM_STOP_THRESH*CLM_STOP_THRESH;
    eps3=CLM_STOP_THRESH;
  }

  if ((hxd=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((ed=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jac=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd0=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTed=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Dpd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((bd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((pnew=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((aones=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }

  WORK=Ud=Sd=VTd=0;
  me_data_t *dt=(me_data_t*)adata;
  setweights(M,aones,1.0,dt->Nt);
  /* memory allocation: different solvers */
  if (solve_axb==0) {

  } else if (solve_axb==1) {
    /* workspace query */
    status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  } else {
    if ((Ud=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((VTd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((Sd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    
    status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  }


  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  (*func)(p, hxd, M, N, adata);


  /* e=x */
  my_dcopy(N, x, 1, ed, 1);
  /* e=x-hx */
  my_daxpy(N, hxd, -1.0, ed);

  /* norm ||e|| */
  p_eL2=my_dnrm2(N, ed);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finite(p_eL2)) stop=7;


  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("iter=%d err=%lf\n",k,p_eL2);
#endif
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }

    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
    */
    if (jac_given) {
     (*jacf)(p, jac, M, N, adata);
    } else {
      /* estimate jacobian using central differences */
       for (ci=0;ci<M; ci++) {
        /* Jacobian in row major order, so jac[ci],jac[ci+M],jac[ci+2*M]...N values are modified */
        /* modify ci-th parameter */
        tempp=p[ci];
        ddiff=fabs(p[ci]*(1e-4));
        if (ddiff<delta) {
          ddiff=delta;
        }
        p[ci]+=ddiff;
        /* f(p+delta) */
        (*func)(p, hxd, M, N, adata);
        p[ci]=tempp-ddiff;
        /* f(p-delta) */
        (*func)(p, hxm, M, N, adata);
        p[ci]=tempp;
        ddiff=0.5/ddiff;
        /* hxd=hxd-hxm */
        my_daxpy(N, hxm, -1.0, hxd);
        /* hxd=hxd/delta */
        my_dscal(N, ddiff, hxd);

        my_dcopy(N, hxd, 1, &jac[ci], M);
       }
    }


     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     my_dgemm('N','T',M,M,N,1.0,jac,M,jac,M,0.0,jacTjacd,M);
     
     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     //cbstatus=cublasDcopy(cbhandle, M*M, jacTjacd, 1, jacTjacd0, 1);
     my_dcopy(M*M,jacTjacd,1,jacTjacd0,1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     my_dgemv('N',M,N,1.0,jac,M,ed,1,0.0,jacTed,1);


     /* Compute ||J^T e||_inf and ||p||^2 */
     /* find infinity norm of J^T e, 1 based indexing*/
     //cbstatus=cublasIdamax(cbhandle, M, jacTed, 1, &ci);
     ci=my_idamax(M,jacTed,1);
     //err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
     memcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double));
     /* L2 norm of current parameter values */
     /* norm ||Dp|| */
     //cbstatus=cublasDnrm2(cbhandle, M, pd, 1, &p_L2);
     p_L2=my_dnrm2(M,p);
     p_L2=p_L2*p_L2;
     if(jacTe_inf<0.0) {jacTe_inf=-jacTe_inf;}
#ifdef DEBUG
     printf("Inf norm=%lf\n",jacTe_inf);
#endif
     
    /* check for convergence */
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0; /* no increment for p in this case */
      stop=1;
      break;
    }

    /* compute initial (k=0) damping factor */
    if (k==0) {
      /* find max diagonal element (stride is M+1) */
      /* should be MAX not MAX(ABS) */
      //cbstatus=cublasIdamax(cbhandle, M, jacTjacd, M+1, &ci); /* 1 based index */
      ci=my_idamax(M,jacTjacd,M+1);
      ci=(ci-1)*(M+1); /* right value of the diagonal */

      //err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
      memcpy(&tmp,&(jacTjacd[ci]),sizeof(double));
      mu=tau*tmp;
    }

    
    /* determine increment using adaptive damping */
    while(1){
      /* augment normal equations */
      /* increment A => A+ mu*I, increment diagonal entries */
      /* copy jacTjacd<=jacTjacd0 */
      //cbstatus=cublasDcopy(cbhandle, M*M, jacTjacd0, 1, jacTjacd, 1);
      memcpy(jacTjacd,jacTjacd0,M*M*sizeof(double));
      //cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
      my_daxpys(M,aones,1,mu,jacTjacd,M+1);

#ifdef DEBUG
      printf("mu=%lf\n",mu);
#endif
/*************************************************************************/
      /* solve augmented equations A x = b */
      /* A==jacTjacd, b==Dpd, after solving, x==Dpd */
      /* b=jacTed : intially right hand side, at exit the solution */
      if (solve_axb==0) {
        /* Cholesky solver **********************/
        /* lower triangle of Ad is destroyed */
        status=my_dpotrf('U',M,jacTjacd,M);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         //cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
         memcpy(Dpd,jacTed,M*sizeof(double));
         status=my_dpotrs('U',M,1,jacTjacd,M,Dpd,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix info=%d\n",status);
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        /* copy Dpd<=jacTed */
        memcpy(Dpd,jacTed,M*sizeof(double));
        status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,WORK,lwork);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }

      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
        status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,WORK,lwork);
        /* copy Dpd<=jacTed */
        //memcpy(Dpd,jacTed,M*sizeof(double));
        memcpy(bd,jacTed,M*sizeof(double));
        /* b<=U^T * b */
        my_dgemv('T',M,M,1.0,Ud,M,bd,1,0.0,Dpd,1);
        /* robust correction */
        /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        for (ci=0; ci<M; ci++) {
         if (Sd[ci]>eps1) {
          Dpd[ci]=Dpd[ci]/Sd[ci];
         } else {
          Dpd[ci]=0.0;
         }
        }

        /* b<=VT^T * b */
        memcpy(bd,Dpd,M*sizeof(double));
        my_dgemv('T',M,M,1.0,VTd,M,bd,1,0.0,Dpd,1);

        issolved=1;
      }
/*************************************************************************/

      /* compute p's new estimate and ||Dp||^2 */
      if (issolved) {
          /* compute p's new estimate and ||Dp||^2 */
          /* pnew=p+Dp */
          /* pnew=p */
        //  cbstatus=cublasDcopy(cbhandle, M, pd, 1, pnewd, 1);
          memcpy(pnew,p,M*sizeof(double));
          /* pnew=pnew+Dp */
          //cbstatus=cublasDaxpy(cbhandle, M, &alpha, Dpd, 1, pnewd, 1);
          my_daxpy(M,Dpd,1.0,pnew);

          /* norm ||Dp|| */
          //cbstatus=cublasDnrm2(cbhandle, M, Dpd, 1, &Dp_L2);
          Dp_L2=my_dnrm2(M,Dpd);
          Dp_L2=Dp_L2*Dp_L2;

#ifdef DEBUG
printf("norm ||dp|| =%lf, norm ||p||=%lf\n",Dp_L2,p_L2);
#endif
          if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
           stop=2;
           break;
          }

         if(Dp_L2>=(p_L2+eps2)/(CLM_EPSILON*CLM_EPSILON)){ /* almost singular */
          stop=4;
          break;
         }

        /* new function value */
        (*func)(pnew, hxd, M, N, adata); /* evaluate function at p + Dp */

        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        /* copy to device */
        /* hxd<=hx */
        //err=cudaMemcpy(hxd, hx, N*sizeof(double), cudaMemcpyHostToDevice);

        /* e=x */
        //cbstatus=cublasDcopy(cbhandle, N, xd, 1, ed, 1);
        memcpy(ed,x,N*sizeof(double));
        /* e=x-hx */
        //cbstatus=cublasDaxpy(cbhandle, N, &alpha, hxd, 1, ed, 1);
        my_daxpy(N,hxd,-1.0,ed);
        /* note: e is updated */

        /* norm ||e|| */
        //cbstatus=cublasDnrm2(cbhandle, N, ed, 1, &pDp_eL2);
        pDp_eL2=my_dnrm2(N,ed);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finite(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
                                  */
          stop=7;
          break;
        }

        /* dL=Dp'*(mu*Dp+jacTe) */
        /* bd=jacTe+mu*Dp */
        //cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, bd, 1);
        memcpy(bd,jacTed,M*sizeof(double));
        //cbstatus=cublasDaxpy(cbhandle, M, &mu, Dpd, 1, bd, 1);
        my_daxpy(M,Dpd,mu,bd);
        //cbstatus=cublasDdot(cbhandle, M, Dpd, 1, bd, 1, &dL);
        dL=my_ddot(M,Dpd,bd);

        dF=p_eL2-pDp_eL2;

#ifdef DEBUG
        printf("dF=%lf, dL=%lf\n",dF,dL);
#endif
        if(dL>0.0 && dF>0.0){ /* reduction in error, increment is accepted */
          tmp=(2.0*dF/dL-1.0);
          tmp=1.0-tmp*tmp*tmp;
          mu=mu*((tmp>=CLM_ONE_THIRD)? tmp : CLM_ONE_THIRD);
          nu=2;

          /* update p's estimate */
          //cbstatus=cublasDcopy(cbhandle, M, pnewd, 1, pd, 1);
          memcpy(p,pnew,M*sizeof(double));

          /* update ||e||_2 */
          p_eL2=pDp_eL2;
          break;
        }

      }
      /* if this point is reached, either the linear system could not be solved or
       * the error did not reduce; in any case, the increment must be rejected
       */

      mu*=(double)nu;
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). */
        stop=5;
        break;
      }

      nu=nu2;

    } /* inner loop */

  }
  /**** end iteration loop ***********/


  if(k>=itmax) stop=3;


  free(jac);
  free(jacTjacd);
  free(jacTjacd0);
  free(jacTed);
  free(Dpd);
  free(bd);
  free(hxd);
  if (!jac_given) { free(hxm); }
  free(ed);
  free(aones);
  free(pnew);
 
  if (solve_axb==0) {
  } else if (solve_axb==1) {
   free(WORK);
  } else {
   free(Ud);
   free(VTd);
   free(Sd);
   free(WORK);
  }
#ifdef DEBUG
  printf("stop=%d\n",stop);
#endif
  if(info){
    info[0]=init_p_eL2;
    info[1]=p_eL2;
    info[2]=jacTe_inf;
    info[3]=Dp_L2;
    info[4]=mu;
    info[5]=(double)k;
    info[6]=(double)stop;
    info[7]=(double)0;
    info[8]=(double)0;
    info[9]=(double)0;
  }
  return 0;
}


int
mlm_der_single(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[6],   /* I: minim. options [\mu, \m, \p0, \p1, \p2, \delta].
                        delta: 1 or 2
                       */
  double info[10], 
                      /* O: information regarding the minimization. Set to NULL if don't care
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{
  /* NOTE F()=func()-data */
  double *Fxk, *Fyk;
  double *Jk, *JkTJk, *JkTJk0, *JkTe, *JkTe0;
  double *dk,*yk,*dhatk,*sk;
  double *Jkdk;

  double lambda;
  double *aones;
  double mu,m,p0,p1,p2; int delta;
  double Fxknrm,Fyknrm,Fykdhatknrm,Fxksknrm,FJkdknrm;
  int niter=0;
  int p_update=1;
  double Fxknrm2,Fxksknrm2;

  double Ak,Pk,rk;

  int ci;

  /* used in QR solver */
  double *WORK=0,*TAU=0,*R=0;
  /* used in SVD solver */
  double *Ud=0;
  double *VTd=0;
  double *Sd=0;

  int lwork=0;
  double w[1];
  int status,issolved;
  int solve_axb=linsolv;



  if (opts) {
    mu=opts[0];
    m=opts[1];
    p0=opts[2];
    p1=opts[3];
    p2=opts[4];
    delta=(int)opts[5];  
  } else {
    mu=1e-5;
    m=1e-3;
    p0=0.0001;
    p1=0.25;
    p2=0.75;
    delta=1;  /* 1 or 2 */
  }

  double epsilon=CLM_EPSILON;

  if ((aones=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
//  for (ci=0;ci<M; ci++) {
//   aones[ci]=1.0;
//  }
  me_data_t *dt=(me_data_t*)adata;
  setweights(M,aones,1.0,dt->Nt);

  if ((dk=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((sk=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((dhatk=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((yk=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Jkdk=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Fxk=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Fyk=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Jk=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((JkTJk=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((JkTJk0=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((JkTe=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((JkTe0=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  /* memory allocation: different solvers */
  if (solve_axb==1) {
    /* QR solver ********************************/
    /* workspace query */
    status=my_dgeqrf(M,M,JkTJk,M,TAU,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
        printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
        exit(1);
      }
    }
    if ((R=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((TAU=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
  } else if (solve_axb==2) {
    if ((Ud=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((VTd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((Sd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }

    status=my_dgesvd('A','A',M,M,JkTJk,M,Sd,Ud,M,VTd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  }

  /* F(x_k) = func()-data */
  /* func() */
  (*func)(p, Fxk, M, N, adata);
  /* func() - data */
  my_daxpy(N, x, -1.0, Fxk);
  /* find ||Fxk|| */
  Fxknrm=my_dnrm2(N,Fxk);

  double init_Fxknrm=Fxknrm;

  while (niter<itmax) {
     if (delta>1) {
      lambda=mu*Fxknrm*Fxknrm;
     } else {
      lambda=mu*Fxknrm;
     }
     Fxknrm2=Fxknrm*Fxknrm;

     if ( p_update==1 ) {
      /* J_k */
      (*jacf)(p, Jk, M, N, adata);
      /* Compute J_k^T J_k and -J_k^T F(x_k) */
      my_dgemm('N','T',M,M,N,1.0,Jk,M,Jk,M,0.0,JkTJk0,M);
      my_dgemv('N',M,N,-1.0,Jk,M,Fxk,1,0.0,JkTe0,1);
     }
     /* if || J_k^T F(x_k) || < epsilon, stop */
     Fyknrm=my_dnrm2(M,JkTe0);
     if (Fyknrm<epsilon) { 
#ifdef DEBUG
      printf("stopping 1 at iter %d\n",niter);
#endif
      break; 
     }
     memcpy(JkTe,JkTe0,M*sizeof(double));
     memcpy(JkTJk,JkTJk0,M*M*sizeof(double));
     /* add lambdaxI to J^T J */
     my_daxpys(M,aones,1,lambda,JkTJk,M+1);
  
/********************************************************************/
     if (solve_axb==0) {
       /* Cholesky solver **********************/
       status=my_dpotrf('U',M,JkTJk,M);
       if (!status) {
         issolved=1;
       } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
       }
       if (issolved) {
         /* copy dk<=JkTe */
         memcpy(dk,JkTe,M*sizeof(double));
         status=my_dpotrs('U',M,1,JkTJk,M,dk,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix info=%d\n",status);
#endif
         }
        }
     } else if (solve_axb==1) {
       /* QR solver ********************************/
       /* QR factorization: JkTJk and TAU now have that */
       status=my_dgeqrf(M,M,JkTJk,M,TAU,w,lwork);
       /* copy JkTJk as R (only upper triangle is used) */
       memcpy(R,JkTJk,M*M*sizeof(double));
       /* form Q in JkTJk */
       my_dorgqr(M,M,M,JkTJk,M,TAU,WORK,lwork);
       /* dk <= Q^T jacTed */
       my_dgemv('T',M,M,1.0,JkTJk,M,JkTe,1,0.0,dk,1);
       /* solve R x = b */
       status=my_dtrtrs('U','N','N',M,1,R,M,dk,M);
       if (!status) {
        issolved=1;
       } else {
        issolved=0;
#ifdef DEBUG
        printf("Singular matrix info=%d\n",status);
#endif
       }
     } else if (solve_axb==2) { 
       /* SVD solver *********************************/
       /* U S VT = A */
       status=my_dgesvd('A','A',M,M,JkTJk,M,Sd,Ud,M,VTd,M,WORK,lwork);
       /* dk <= U^T jacTed */
       my_dgemv('T',M,M,1.0,Ud,M,JkTe,1,0.0,dk,1);
       /* robust correction */
       /* divide by singular values  dk[]/Sd[]  for Sd[]> epsilon */
       for (ci=0; ci<M; ci++) {
         if (Sd[ci]>epsilon) {
          dk[ci]=dk[ci]/Sd[ci];
         } else {
          dk[ci]=0.0;
         }
       }

       /* dk <= VT^T dk */
       memcpy(yk,dk,M*sizeof(double));
       my_dgemv('T',M,M,1.0,VTd,M,yk,1,0.0,dk,1);

     }
/********************************************************************/

     /* y_k<= x_k+ d_k */
     my_dcopy(M,p,1,yk,1);
     my_daxpy(M,dk,1.0,yk);

     /* compute F(y_k) */
     /* func() */
     (*func)(yk, Fyk, M, N, adata);
     /* func() - data */
     my_daxpy(N, x, -1.0, Fyk);

     /* Compute -J_k^T F(y_k) */
     my_dgemv('N',M,N,-1.0,Jk,M,Fyk,1,0.0,JkTe,1);
  
/********************************************************************/
     if (solve_axb==0) {
       /* Cholesky solver **********************/
       /* copy dk<=JkTe */
         memcpy(dhatk,JkTe,M*sizeof(double));
         status=my_dpotrs('U',M,1,JkTJk,M,dhatk,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix info=%d\n",status);
#endif
         }
     } else if (solve_axb==1) {
       /* QR solver ********************************/
       /* dhatk <= Q^T jacTed */
       my_dgemv('T',M,M,1.0,JkTJk,M,JkTe,1,0.0,dhatk,1);
       /* solve R x = b */
       status=my_dtrtrs('U','N','N',M,1,R,M,dhatk,M);
       if (!status) {
         issolved=1;
       } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
       }
     } else  if (solve_axb==2) {
       /* SVD solver *********************************/
       /* dhatk <= U^T jacTed */
       my_dgemv('T',M,M,1.0,Ud,M,JkTe,1,0.0,dhatk,1);
       /* robust correction */
       /* divide by singular values  dk[]/Sd[]  for Sd[]> epsilon */
       for (ci=0; ci<M; ci++) {
         if (Sd[ci]>epsilon) {
          dhatk[ci]=dhatk[ci]/Sd[ci];
         } else {
          dhatk[ci]=0.0;
         }
       }
       /* dk <= VT^T dk */
       memcpy(yk,dhatk,M*sizeof(double));
       my_dgemv('T',M,M,1.0,VTd,M,yk,1,0.0,dhatk,1);
     }
/********************************************************************/



  /* s_k<= d_k+ dhat_k */
  my_dcopy(M,dk,1,sk,1);
  my_daxpy(M,dhatk,1.0,sk);

  /* find norms */
  /* || F(y_k) || */
  Fyknrm=my_dnrm2(N,Fyk);
  Fyknrm=Fyknrm*Fyknrm;

  /* || F(y_k) + J_k dhat_k || */
  my_dgemv('T',M,N,1.0,Jk,M,dhatk,1,0.0,Jkdk,1);
  /* Fyk <= Fyk+ J_k dhat_k */
  my_daxpy(N,Jkdk,1.0,Fyk);
  Fykdhatknrm=my_dnrm2(N,Fyk);
  Fykdhatknrm=Fykdhatknrm*Fykdhatknrm;

  /* ||F(x_k+d_k+dhat_k)|| == ||F(x_k+s_k)|| */
  /* y_k<= x_k+ s_k */
  my_dcopy(M,p,1,yk,1);
  my_daxpy(M,sk,1.0,yk);
  (*func)(yk, Fyk, M, N, adata);
  /* func() - data */
  my_daxpy(N, x, -1.0, Fyk);
  Fxksknrm=my_dnrm2(N,Fyk);
  Fxksknrm2=Fxksknrm*Fxksknrm;

  /* || Fxk + J_k d_k || */
  /* J d_k : since J is row major, transpose */
  my_dgemv('T',M,N,1.0,Jk,M,dk,1,0.0,Jkdk,1);
  /* Fxk <= Fxk+ J_k d_k or, J_k d_k <= Fxk+ J_k d_k */
  my_daxpy(N,Fxk,1.0,Jkdk);
  FJkdknrm=my_dnrm2(N,Jkdk);
  FJkdknrm=FJkdknrm*FJkdknrm;

  /* find ratio */
  Ak=Fxknrm2-Fxksknrm2;
  Pk=Fxknrm2-FJkdknrm+Fyknrm-Fykdhatknrm;
  /* if Pk<epsilon or rk<epsilon, also stop */
  if (fabs(Pk)<epsilon) {
#ifdef DEBUG
   printf("stopping 2 at iter %d\n",niter);
#endif
   break; 
  }
  rk=Ak/Pk;


  //printf("Ak=%lf Pk=%lf rk=%lf mu=%lf ||Fxk||=%lf\n",Ak,Pk,rk,mu,Fxknrm);
  if (rk>=p0) {
    p_update=1;
    /* update p<= p+sk */
    my_daxpy(M,sk,1.0,p);
    /* also update auxiliary info */
    /* Fxk <= Fyk */
    my_dcopy(N,Fyk,1,Fxk,1);
    Fxknrm=Fxksknrm;
    /* new Jk needed */
  } else { /* else no p update */
    p_update=0;
    /* use previous Jk, Fxk, JkTJk, JkTe */
  }
  if (rk<p1) {
   mu=4.0*mu; 
  } else if (rk<p2) {
   /* no update */
  } else {
   if (m>0.25*mu) {
    mu=m;
   } else {
    mu=0.25*mu;
   }
  }
   niter++;
  }

  free(aones);
  if (solve_axb==1) {
   free(WORK);
   free(TAU);
   free(R);
  } else if (solve_axb==2) {
    free(WORK);
    free(Ud);
    free(VTd);
    free(Sd);
  }
  free(Jkdk);
  free(dk);
  free(dhatk);
  free(sk);
  free(yk);
  free(Fxk);
  free(Fyk);
  free(Jk);
  free(JkTJk0);
  free(JkTJk);
  free(JkTe);
  free(JkTe0);

  if(info){
    info[0]=init_Fxknrm;
    info[1]=Fxknrm;
  }
  return 0;
}


/** keep interface almost the same as in levmar **/
/* OS accel */
int
oslevmar_der_single_nocuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian \part x / \part p */
  double *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
  double *x,         /* I: measurement vector. NULL implies a zero vector */
  int M,              /* I: parameter vector dimension (i.e. #unknowns) */
  int N,              /* I: measurement vector dimension */
  int itmax,          /* I: maximum number of iterations */
  double opts[4],   /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  double info[10], 
                      /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */

  int linsolv, /* 0 Cholesky, 1 QR, 2 SVD */
  int randomize, /* if >0 randomize */
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{

  /* general note: all device variables end with a 'd' */
  int stop=0;
  int nu=2,nu2;
  double p_L2, Dp_L2=DBL_MAX, dF, dL, p_eL2, jacTe_inf=0.0, pDp_eL2, init_p_eL2;
  double tmp,mu=0.0;
  double tau, eps1, eps2, eps2_sq, eps3;
  int k,ci,issolved;

  double *hxd;
  double *ed;
  double *jac;

  double *jacTjacd,*jacTjacd0;

  double *pnew,*Dpd,*bd;
  double *aones;
  double *jacTed;

  /* used in QR solver */
  double *WORK;
  int lwork=0;
  double w[1];

  int status;

  /* used in SVD solver */
  double *Ud;
  double *VTd;
  double *Sd;

  int solve_axb=linsolv;

  /* setup default settings */
  if(opts){
    tau=opts[0];
    eps1=opts[1];
    eps2=opts[2];
    eps2_sq=opts[2]*opts[2];
    eps3=opts[3];
  } else {
    tau=CLM_INIT_MU;
    eps1=CLM_STOP_THRESH;
    eps2=CLM_STOP_THRESH;
    eps2_sq=CLM_STOP_THRESH*CLM_STOP_THRESH;
    eps3=CLM_STOP_THRESH;
  }

  if ((hxd=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((ed=(double*)calloc((size_t)N,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jac=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTjacd0=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((jacTed=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((Dpd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((bd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((pnew=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((aones=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }

  WORK=Ud=Sd=VTd=0;
//  for (ci=0;ci<M; ci++) {
//   aones[ci]=1.0;
//  }
  me_data_t *dt=(me_data_t*)adata;
  setweights(M,aones,1.0,dt->Nt);

  /* memory allocation: different solvers */
  if (solve_axb==0) {

  } else if (solve_axb==1) {
    /* workspace query */
    status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  } else {
    if ((Ud=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((VTd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    if ((Sd=(double*)calloc((size_t)M,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
    }
    
    status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
     }
    }
  }


  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  (*func)(p, hxd, M, N, adata);


  /* e=x */
  my_dcopy(N, x, 1, ed, 1);
  /* e=x-hx */
  my_daxpy(N, hxd, -1.0, ed);

  /* norm ||e|| */
  p_eL2=my_dnrm2(N, ed);
  /* square */
  p_eL2=p_eL2*p_eL2;

  init_p_eL2=p_eL2;
  if(!finite(p_eL2)) stop=7;

  /* setup OS subsets and stating offsets */
  /* ME data for Jacobian calculation (need a new one) */
  me_data_t lmdata;
  me_data_t *lmdata0=(me_data_t*)adata;
  lmdata.clus=lmdata0->clus;
  lmdata.u=lmdata.v=lmdata.w=0;  /* not needed */
  lmdata.Nbase=lmdata0->Nbase;
  lmdata.tilesz=lmdata0->tilesz;
  lmdata.N=lmdata0->N;
  lmdata.carr=lmdata0->carr;
  lmdata.M=lmdata0->M;
  lmdata.Mt=lmdata0->Mt;
  lmdata.freq0=lmdata0->freq0;
  lmdata.Nt=lmdata0->Nt;
  lmdata.barr=lmdata0->barr;
  lmdata.coh=lmdata0->coh;
  lmdata.tileoff=lmdata0->tileoff;
  /* we work with lmdata0->tilesz tiles, and offset from 0 is lmdata0->tileoff,
     so, OS needs to divide this many tiles with the right offset per subset */
  /* barr and coh offsets will be calculated internally */
  /* ed : N, cohd : Nbase*8, bbd : Nbase*2 full size */
  /* if ntiles<Nsubsets, make Nsubsets=ntiles */
  int Nsubsets=10;
  if (lmdata0->tilesz<Nsubsets) { Nsubsets=lmdata0->tilesz; }
  /* FIXME: is 0.1 of subsets enough ? */
  int max_os_iter=(int)ceil(0.1*(double)Nsubsets);
  int Npersubset=(N+Nsubsets-1)/Nsubsets;
  int Ntpersubset=(lmdata0->tilesz+Nsubsets-1)/Nsubsets;
  int *Nos,*edI,*subI=0,*tileI,*tileoff;
  if ((Nos=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((edI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((tileI=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  if ((tileoff=(int*)calloc((size_t)Nsubsets,sizeof(int)))==0) {
#ifndef USE_MIC
      printf("%s: %d: no free memory\n",__FILE__,__LINE__);
#endif
      exit(1);
  }
  int l,ositer;;
  k=l=0;
  for (ci=0; ci<Nsubsets; ci++) {
    edI[ci]=k;
    tileoff[ci]=lmdata0->tileoff+l;
    if (l+Ntpersubset<lmdata0->tilesz) {
      Nos[ci]=Npersubset;
      tileI[ci]=Ntpersubset;
    } else {
      Nos[ci]=N-k;
      tileI[ci]=lmdata0->tilesz-l;
    }
    k=k+Npersubset;
    l=l+Ntpersubset;
  }

#ifdef DEBUG
  for (ci=0; ci<Nsubsets; ci++) {
   printf("ci=%d, Nos=%d, edI=%d, tilesz=%d, tileoff=%d\n",ci,Nos[ci],edI[ci],tileI[ci],tileoff[ci]);
  }
#endif


  /**** iteration loop ***********/
  for(k=0; k<itmax && !stop; ++k){
#ifdef DEBUG
    printf("iter=%d err=%lf\n",k,p_eL2);
#endif
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }
    if (randomize) {
     /* random permutation of subsets */
     subI=random_permutation(Nsubsets,0,0);
    }
/**************** OS loop ***************************/
    for (ositer=0; ositer<max_os_iter; ositer++) {
     /* select subset to compute Jacobian */
     if (randomize) {
      l=subI[ositer];
     } else {
      /* a deterministic value in 0...Nsubsets-1 */
      l=(k+ositer)%Nsubsets;
     }
     /* NOTE: no. of subsets >= no. of OS iterations, so select
        a random set of subsets */
     /* N, Nbase changes with subset, cohd,bbd,ed gets offsets */
     /* ed : N, cohd : Nbase*8, bbd : Nbase*2 full size */
    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
    */
     /* note: adata has to advance */
     lmdata.tileoff=tileoff[l];
     lmdata.tilesz=tileI[l];
     (*jacf)(p, jac, M, Nos[l], (void*)&lmdata);
     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     my_dgemm('N','T',M,M,Nos[l],1.0,jac,M,jac,M,0.0,jacTjacd,M);
     
     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     my_dcopy(M*M,jacTjacd,1,jacTjacd0,1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     my_dgemv('N',M,Nos[l],1.0,jac,M,&ed[edI[l]],1,0.0,jacTed,1);


     /* Compute ||J^T e||_inf and ||p||^2 */
     /* find infinity norm of J^T e, 1 based indexing*/
     ci=my_idamax(M,jacTed,1);
     memcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double));
     /* L2 norm of current parameter values */
     /* norm ||Dp|| */
     p_L2=my_dnrm2(M,p);
     p_L2=p_L2*p_L2;
     if(jacTe_inf<0.0) {jacTe_inf=-jacTe_inf;}
#ifdef DEBUG
     printf("Inf norm=%lf\n",jacTe_inf);
#endif
     
    /* check for convergence */
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0; /* no increment for p in this case */
      stop=1;
      break;
    }

    /* compute initial (k=0) damping factor */
    if (k==0) {
      /* find max diagonal element (stride is M+1) */
      /* should be MAX not MAX(ABS) */
      ci=my_idamax(M,jacTjacd,M+1);
      ci=(ci-1)*(M+1); /* right value of the diagonal */

      memcpy(&tmp,&(jacTjacd[ci]),sizeof(double));
      mu=tau*tmp;
    }

    
    /* determine increment using adaptive damping */
    while(1){
      /* augment normal equations */
      /* increment A => A+ mu*I, increment diagonal entries */
      /* copy jacTjacd<=jacTjacd0 */
      memcpy(jacTjacd,jacTjacd0,M*M*sizeof(double));
      my_daxpys(M,aones,1,mu,jacTjacd,M+1);

#ifdef DEBUG
      printf("mu=%lf\n",mu);
#endif
/*************************************************************************/
      /* solve augmented equations A x = b */
      /* A==jacTjacd, b==Dpd, after solving, x==Dpd */
      /* b=jacTed : intially right hand side, at exit the solution */
      if (solve_axb==0) {
        /* Cholesky solver **********************/
        /* lower triangle of Ad is destroyed */
        status=my_dpotrf('U',M,jacTjacd,M);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         memcpy(Dpd,jacTed,M*sizeof(double));
         status=my_dpotrs('U',M,1,jacTjacd,M,Dpd,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           printf("Singular matrix info=%d\n",status);
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        /* copy Dpd<=jacTed */
        memcpy(Dpd,jacTed,M*sizeof(double));
        status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,WORK,lwork);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         printf("Singular matrix info=%d\n",status);
#endif
        }

      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
        status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,WORK,lwork);
        /* copy Dpd<=jacTed */
        memcpy(bd,jacTed,M*sizeof(double));
        /* b<=U^T * b */
        my_dgemv('T',M,M,1.0,Ud,M,bd,1,0.0,Dpd,1);
        /* robust correction */
        /* divide by singular values  Dpd[]/Sd[]  for Sd[]> eps1 */
        for (ci=0; ci<M; ci++) {
         if (Sd[ci]>eps1) {
          Dpd[ci]=Dpd[ci]/Sd[ci];
         } else {
          Dpd[ci]=0.0;
         }
        }

        /* b<=VT^T * b */
        memcpy(bd,Dpd,M*sizeof(double));
        my_dgemv('T',M,M,1.0,VTd,M,bd,1,0.0,Dpd,1);

        issolved=1;
      }
/*************************************************************************/

      /* compute p's new estimate and ||Dp||^2 */
      if (issolved) {
          /* compute p's new estimate and ||Dp||^2 */
          /* pnew=p+Dp */
          /* pnew=p */
          memcpy(pnew,p,M*sizeof(double));
          /* pnew=pnew+Dp */
          my_daxpy(M,Dpd,1.0,pnew);

          /* norm ||Dp|| */
          Dp_L2=my_dnrm2(M,Dpd);
          Dp_L2=Dp_L2*Dp_L2;

#ifdef DEBUG
printf("norm ||dp|| =%lf, norm ||p||=%lf\n",Dp_L2,p_L2);
#endif
          if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
           stop=2;
           break;
          }

         if(Dp_L2>=(p_L2+eps2)/(CLM_EPSILON*CLM_EPSILON)){ /* almost singular */
          stop=4;
          break;
         }

        /* new function value */
        (*func)(pnew, hxd, M, N, adata); /* evaluate function at p + Dp */

        /* compute ||e(pDp)||_2 */
        /* ### hx=x-hx, pDp_eL2=||hx|| */
        memcpy(ed,x,N*sizeof(double));
        /* e=x-hx */
        my_daxpy(N,hxd,-1.0,ed);
        /* note: e is updated */

        /* norm ||e|| */
        pDp_eL2=my_dnrm2(N,ed);
        pDp_eL2=pDp_eL2*pDp_eL2;


        if(!finite(pDp_eL2)){ /* sum of squares is not finite, most probably due to a user error.
                                  */
          stop=7;
          break;
        }

        /* dL=Dp'*(mu*Dp+jacTe) */
        /* bd=jacTe+mu*Dp */
        memcpy(bd,jacTed,M*sizeof(double));
        my_daxpy(M,Dpd,mu,bd);
        dL=my_ddot(M,Dpd,bd);

        dF=p_eL2-pDp_eL2;

#ifdef DEBUG
        printf("dF=%lf, dL=%lf\n",dF,dL);
#endif
        if(dL>0.0 && dF>0.0){ /* reduction in error, increment is accepted */
          tmp=(2.0*dF/dL-1.0);
          tmp=1.0-tmp*tmp*tmp;
          mu=mu*((tmp>=CLM_ONE_THIRD)? tmp : CLM_ONE_THIRD);
          nu=2;

          /* update p's estimate */
          memcpy(p,pnew,M*sizeof(double));

          /* update ||e||_2 */
          p_eL2=pDp_eL2;
          break;
        }

      }
      /* if this point is reached, either the linear system could not be solved or
       * the error did not reduce; in any case, the increment must be rejected
       */

      mu*=(double)nu;
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). */
        stop=5;
        break;
      }

      nu=nu2;

    } /* inner loop */

  }
  if (randomize) {
   free(subI);
  }
/**************** end OS loop ***************************/

  }
  /**** end iteration loop ***********/
  free(Nos);
  free(edI);
  free(tileI);
  free(tileoff);

  if(k>=itmax) stop=3;


  free(jac);
  free(jacTjacd);
  free(jacTjacd0);
  free(jacTed);
  free(Dpd);
  free(bd);
  free(hxd);
  free(ed);
  free(aones);
  free(pnew);
 
  if (solve_axb==0) {
  } else if (solve_axb==1) {
   free(WORK);
  } else {
   free(Ud);
   free(VTd);
   free(Sd);
   free(WORK);
  }
#ifdef DEBUG
  printf("stop=%d\n",stop);
#endif
  if(info){
    info[0]=init_p_eL2;
    info[1]=p_eL2;
    info[2]=jacTe_inf;
    info[3]=Dp_L2;
    info[4]=mu;
    info[5]=(double)k;
    info[6]=(double)stop;
    info[7]=(double)0;
    info[8]=(double)0;
    info[9]=(double)0;
  }
  return 0;
}
