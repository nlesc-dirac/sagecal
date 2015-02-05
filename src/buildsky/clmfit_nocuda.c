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


#include "buildsky.h"

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
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
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
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((ed=(double*)calloc((size_t)N,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jac=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jacTjacd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jacTjacd0=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jacTed=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((Dpd=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((bd=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((pnew=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((aones=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }

  WORK=Ud=Sd=VTd=0;
  for (ci=0;ci<M; ci++) {
   aones[ci]=1.0;
  }
  /* memory allocation: different solvers */
  if (solve_axb==0) {

  } else if (solve_axb==1) {
    /* workspace query */
    status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }
  } else {
    if ((Ud=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    if ((VTd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    if ((Sd=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    
    status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
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
     //status=culaDeviceDgemm('N','T',M,M,N,1.0,jacd,M,jacd,M,0.0,jacTjacd,M);
     my_dgemm('N','T',M,M,N,1.0,jac,M,jac,M,0.0,jacTjacd,M);
     
     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     //cbstatus=cublasDcopy(cbhandle, M*M, jacTjacd, 1, jacTjacd0, 1);
     my_dcopy(M*M,jacTjacd,1,jacTjacd0,1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     //status=culaDeviceDgemv('N',M,N,1.0,jacd,M,ed,1,0.0,jacTed,1);
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
        //status=culaDeviceDpotrf('U',M,jacTjacd,M);
        status=my_dpotrf('U',M,jacTjacd,M);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         //cbstatus=cublasDcopy(cbhandle, M, jacTed, 1, Dpd, 1);
         memcpy(Dpd,jacTed,M*sizeof(double));
         //status=culaDeviceDpotrs('U',M,1,jacTjacd,M,Dpd,M);
         status=my_dpotrs('U',M,1,jacTjacd,M,Dpd,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
         }
        }
      } else if (solve_axb==1) {
        /* QR solver ********************************/
        //status=culaDeviceDgeqrf(M,M,jacTjacd,M,taud);
        /* copy Dpd<=jacTed */
        memcpy(Dpd,jacTed,M*sizeof(double));
        status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,WORK,lwork);
        if (!status) {
         issolved=1;
        } else {
         issolved=0;
#ifdef DEBUG
         fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
        }

      } else {
        /* SVD solver *********************************/
        /* U S VT = A */
       // status=culaDeviceDgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M);
        status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,WORK,lwork);
        /* copy Dpd<=jacTed */
        //memcpy(Dpd,jacTed,M*sizeof(double));
        memcpy(bd,jacTed,M*sizeof(double));
        /* b<=U^T * b */
        //status=culaDeviceDgemv('T',M,M,1.0,Ud,M,Dpd,1,0.0,Dpd,1);
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
        //status=culaDeviceDgemv('T',M,M,1.0,VTd,M,Dpd,1,0.0,Dpd,1);
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

typedef struct thread_data_vec_{ 
  int starti,endi;
  double *ed;
  double *wtd;
} thread_data_vec_t;

static void *
odot_threadfn(void *data) {
 thread_data_vec_t *t=(thread_data_vec_t*)data;
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   t->ed[ci]*=t->wtd[ci];
 }
 return NULL;
}


/* Hadamard product */
/* ed <= ed*wtd , size Nx1
  Nt threads */
static int
my_odot(double *ed,double *wtd,int N,int Nt) {
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_vec_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(N+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_vec_t*)malloc((size_t)Nt*sizeof(thread_data_vec_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

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
    threaddata[nth].wtd=wtd;
    pthread_create(&th_array[nth],&attr,odot_threadfn,(void*)(&threaddata[nth]));
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

 return 0;

}

/* robust LM */
int
clevmar_der_single_nocuda0(
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
  double *ed,*wtd;
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
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
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
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((ed=(double*)calloc((size_t)N,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jac=(double*)calloc((size_t)N*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jacTjacd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jacTjacd0=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((jacTed=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((Dpd=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((bd=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((pnew=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((aones=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((wtd=(double*)calloc((size_t)N,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  WORK=Ud=Sd=VTd=0;
  for (ci=0;ci<M; ci++) {
   aones[ci]=1.0;
  }
  /*W set initial weights to 1 */
  for (ci=0;ci<N; ci++) {
   wtd[ci]=1.0;
  }
  /* memory allocation: different solvers */
  if (solve_axb==0) {

  } else if (solve_axb==1) {
    /* workspace query */
    status=my_dgels('N',M,M,1,jacTjacd,M,Dpd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }
  } else {
    if ((Ud=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    if ((VTd=(double*)calloc((size_t)M*M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    if ((Sd=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    
    status=my_dgesvd('A','A',M,M,jacTjacd,M,Sd,Ud,M,VTd,M,w,-1);
    if (!status) {
      lwork=(int)w[0];
      if ((WORK=(double*)calloc((size_t)lwork,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }
  }

  int nw,wt_itmax=3;
  double wt_sum,robust_nu=3.0;
  int Nt=4; /* threads */

  /* EM iteration loop */
  /************************************************************/
  for (nw=0; nw<wt_itmax; nw++) {
  /* ### compute e=x - f(p) and its L2 norm */
  /* ### e=x-hx, p_eL2=||e|| */
  (*func)(p, hxd, M, N, adata);

  /* e=x */
  my_dcopy(N, x, 1, ed, 1);
  /* e=x-hx */
  my_daxpy(N, hxd, -1.0, ed);

  /*W e<= wt\odot e */
  my_odot(ed,wtd,N,Nt);

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

     /*W J<= wt\odot J, each row mult by wt[] */
     /* jac[0..M-1] <- wtd[0],  jac[M...2M-1] <- wtd[1] ... */
     for (ci=0; ci<N; ci++) {
      my_dscal(M, wtd[ci], &jac[ci*M]);
     }

     /* Compute J^T J and J^T e */
     /* Cache efficient computation of J^T J based on blocking
     */
     /* since J is in ROW major order, assume it is transposed,
       so actually calculate A=J*J^T, where J is size MxN */
     my_dgemm('N','T',M,M,N,1.0,jac,M,jac,M,0.0,jacTjacd,M);
     
     /* create backup */
     /* copy jacTjacd0<=jacTjacd */
     my_dcopy(M*M,jacTjacd,1,jacTjacd0,1);
     /* J^T e */
     /* calculate b=J^T*e (actually compute b=J*e, where J in row major (size MxN) */
     my_dgemv('N',M,N,1.0,jac,M,ed,1,0.0,jacTed,1);


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
         fprintf(stderr,"Singular matrix info=%d\n",status);
#endif
        }
        if (issolved) {
         /* copy Dpd<=jacTed */
         memcpy(Dpd,jacTed,M*sizeof(double));
         status=my_dpotrs('U',M,1,jacTjacd,M,Dpd,M);
         if (status) {
           issolved=0;
#ifdef DEBUG
           fprintf(stderr,"Singular matrix info=%d\n",status);
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
         fprintf(stderr,"Singular matrix info=%d\n",status);
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
        /* copy to device */
        /* hxd<=hx */

        /* e=x */
        memcpy(ed,x,N*sizeof(double));
        /* e=x-hx */
        my_daxpy(N,hxd,-1.0,ed);
        /* note: e is updated */

        /*W e<= wt\odot e */
        my_odot(ed,wtd,N,Nt);

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
  /**** end iteration loop ***********/


  if(k>=itmax) stop=3;

  /*W if not at first or last iteration, recalculate error */
  if (nw>0 && nw<wt_itmax-1) {
    (*func)(p, hxd, M, N, adata);
    /* e=x */
    my_dcopy(N, x, 1, ed, 1);
    /* e=x-hx */
    my_daxpy(N, hxd, -1.0, ed);
  }

  /*W if not at the last iteration, update weights */
  if (nw<wt_itmax-1) {
   for (ci=0; ci<N; ci++) {
    wtd[ci]=sqrt((robust_nu+1.0)/(robust_nu+ed[ci]*ed[ci]));
   }
   wt_sum=my_dasum(N,wtd);
   wt_sum=(double)N/wt_sum;
   my_dscal(N,wt_sum,wtd);
   stop=0; /* restart LM */
  }

  } /* end EM iteration loop */


  free(jac);
  free(jacTjacd);
  free(jacTjacd0);
  free(jacTed);
  free(Dpd);
  free(bd);
  free(hxd);
  if (!jac_given) { free(hxm); }
  free(ed);
  free(wtd);
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
