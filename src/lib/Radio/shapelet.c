/*
 *
 Copyright (C) 2023 Sarod Yatawatta <sarod@users.sf.net>  
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

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Dirac.h"
#include "Dirac_radio.h"

/* evaluate Hermite polynomial value using recursion
 */
static double 
H_e(double x, int n) {
	if(n==0) return 1.0;
	if(n==1) return 2*x;
	return 2*x*H_e(x,n-1)-2*(n-1)*H_e(x,n-2);
}

/******************** shapelet stuff **********************/
/** calculate the UV mode vectors (scalar version, only 1 point)
 * in: u,v: arrays of the grid points in UV domain
 *      beta: scale factor
 *      n0: number of modes in each dimension
 * out:
 *      Av: array of mode vectors size 1 times n0.n0, in column major order
 *      cplx: array of integers, size n0*n0, if 1 this mode is imaginary, else real
 *
 */
static int
calculate_uv_mode_vectors_scalar(double u, double v, double beta, int n0, double **Av, int **cplx) {

	int xci,zci,Ntot;

	double **shpvl, *fact;
	int n1,n2,start;
	double xval;
	int signval;

  Ntot=2; /* u,v seperately */
	/* set up factorial array */
  if ((fact=(double*)calloc((size_t)(n0),sizeof(double)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
  fact[0]=1.0;
	for (xci=1; xci<(n0); xci++) {
		fact[xci]=(xci)*fact[xci-1];
	}

	/* setup array to store calculated shapelet value */
	/* need max storage Ntot x n0 */
  if ((shpvl=(double**)calloc((size_t)(Ntot),sizeof(double*)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
	for (xci=0; xci<Ntot; xci++) {
   if ((shpvl[xci]=(double*)calloc((size_t)(n0),sizeof(double)))==0) {
	   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	   exit(1);
	 }
	}


	/* start filling in the array from the positive values */
	zci=0;
  xval=u*beta;
  double expval=exp(-0.5*xval*xval);
	for (xci=0; xci<n0; xci++) {
		shpvl[zci][xci]=H_e(xval,xci)*expval/sqrt((double)(2<<xci)*fact[xci]);
	}
	zci=1;
  xval=v*beta;
  expval=exp(-0.5*xval*xval);
	for (xci=0; xci<n0; xci++) {
		shpvl[zci][xci]=H_e(xval,xci)*expval/sqrt((double)(2<<xci)*fact[xci]);
	}


	/* now calculate the mode vectors */
	/* each vector is 1 x 1 length and there are n0*n0 of them */
  if ((*Av=(double*)calloc((size_t)((n0)*(n0)),sizeof(double)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
  if ((*cplx=(int*)calloc((size_t)((n0)*(n0)),sizeof(int)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}

  for (n2=0; n2<(n0); n2++) {
	 for (n1=0; n1<(n0); n1++) {
	  (*cplx)[n2*n0+n1]=((n1+n2)%2==0?0:1) /* even (real) or odd (imaginary)*/;
		/* sign */
		if ((*cplx)[n2*n0+n1]==0) {
			signval=(((n1+n2)/2)%2==0?1:-1);
		} else {
			signval=(((n1+n2-1)/2)%2==0?1:-1);
		}

    /* fill in 1*1*(zci) to 1*1*(zci+1)-1 */
		start=(n2*(n0)+n1);
		if (signval==-1) {
        (*Av)[start]=-shpvl[0][n1]*shpvl[1][n2];
		} else {
        (*Av)[start]=shpvl[0][n1]*shpvl[1][n2];
		}
	 }
	}

	free(fact);
	for (xci=0; xci<Ntot; xci++) {
	 free(shpvl[xci]);
	}
	free(shpvl);

  return 0;
}


/* Fourier space contribution with a scalar shapelet model */
complex double
shapelet_contrib(void*dd, double u, double v, double w) {
  exinfo_shapelet *dp=(exinfo_shapelet*)dd;
  int *cplx;
  double *Av;
  int ci,M;
  double a,b,cosph,sinph,ut,vt,up,vp;

  double realsum,imagsum;

  /* first the rotation due to projection */
 // up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
 // vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);
  if (dp->use_projection) {
   up=-u*(dp->cxi)+v*(dp->cphi)*(dp->sxi)-w*(dp->sphi)*(dp->sxi);
   vp=-u*(dp->sxi)-v*(dp->cphi)*(dp->cxi)+w*(dp->sphi)*(dp->cxi);
  } else {
   up=u;
   vp=v;
  }

  /* linear transformations, if any */
//  a=1.0;
//  b=dp->eY/dp->eX;
  a=1.0/dp->eX;
  b=1.0/dp->eY;
  //cosph=cos(dp->eP);
  //sinph=sin(dp->eP);
  sincos(dp->eP,&sinph,&cosph);
  ut=a*(cosph*up-sinph*vp);
  vt=b*(sinph*up+cosph*vp);
  /* note: we decompose f(-l,m) so the Fourier transform is F(-u,v)
   so negate the u grid */
  calculate_uv_mode_vectors_scalar(-ut, vt, dp->beta, dp->n0, &Av, &cplx);
  realsum=imagsum=0.0;
  M=(dp->n0)*(dp->n0);
  for (ci=0; ci<M; ci++) {
    if (cplx[ci]) {
     imagsum+=dp->modes[ci]*Av[ci];
    } else {
     realsum+=dp->modes[ci]*Av[ci];
    }
  }

  free(Av);
  free(cplx);
  //return 2.0*M_PI*(realsum+_Complex_I*imagsum);
  return 2.0*M_PI*(realsum+_Complex_I*imagsum)*a*b;
}

/* Fourier space contribution with a vector (4 correlations) shapelet model */
/* modes: 4*n0*n0 models (2x2)x modes
 * n0: model order (n0^2 basis)
 * beta: model scale
 * u,v,w: baseline coords 
 * out:
 * coh: 4x1 correlations output
 */
int
shapelet_contrib_vector(complex double *modes, int n0, double beta, double u, double v, double w, complex double *coh) {
  int *cplx;
  double *Av;
  int ci,M;

  /* note: we decompose f(-l,m) so the Fourier transform is F(-u,v)
   so negate the u grid */
  calculate_uv_mode_vectors_scalar(-u, v, beta, n0, &Av, &cplx);
  M=n0*n0;
  coh[0]=coh[1]=coh[2]=coh[3]=0.0;
  for (ci=0; ci<M; ci++) {
    /* modes[4*ci],modes[4*ci+1],modes[4*ci+2],modes[4*ci+3] map to
     * XX,XY,YX,YY */
    complex double coeff;
    if (cplx[ci]) {
     coeff=0.0+_Complex_I*Av[ci];
    } else {
     coeff=Av[ci]+_Complex_I*0.0;
    }
    coh[0]+=modes[4*ci]*coeff;
    coh[1]+=modes[4*ci+1]*coeff;
    coh[2]+=modes[4*ci+2]*coeff;
    coh[3]+=modes[4*ci+3]*coeff;
  }

  coh[0] *=2.0*M_PI;
  coh[1] *=2.0*M_PI;
  coh[2] *=2.0*M_PI;
  coh[3] *=2.0*M_PI;

  free(Av);
  free(cplx);
  return 0;
}



/* struct for sorting coordinates */
typedef struct coordval_{
 double val;
 int idx;
} coordval_t;

/* comparison */
static int
compare_coordinates(const void *a, const void *b) {
 const coordval_t *da=(const coordval_t *)a;
 const coordval_t *db=(const coordval_t *)b;

 return(da->val>=db->val?1:-1);
}

//#define DEBUG

int
shapelet_modes(int n0,double beta, double *x, double *y, int N, complex double *output) {
/* calculate mode vectors for each (x,y) point given by the arrays x, y
 * of equal length.
 *
 * in: x,y: arrays of the grid points
 *      N: number of grid points
 *      beta: scale factor
 *      n0: number of modes in each dimension
 * out:        
 *      Av: array of mode vectors size N times n0.n0, in column major order
 *
 */

	double *grid;
	int *xindex,*yindex;
	int xci,yci,zci,Ntot;
	int *neg_grid;

	double **shpvl, *fact;
	int n1,n2;

  /* for sorting */
  coordval_t *cx_val,*cy_val; 

  /* image size: N pixels
   */
	/* allocate memory to store grid points: max unique points are 2N */
  if ((grid=(double*)calloc((size_t)(2*N),sizeof(double)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}

  if ((xindex=(int*)calloc((size_t)(N),sizeof(int)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
  if ((yindex=(int*)calloc((size_t)(N),sizeof(int)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}

  /* sort coordinates */
  if ((cx_val=(coordval_t*)calloc((size_t)(N),sizeof(coordval_t)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
  if ((cy_val=(coordval_t*)calloc((size_t)(N),sizeof(coordval_t)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}

  for (xci=0; xci<N; xci++) {
   cx_val[xci].idx=xci;
   cx_val[xci].val=x[xci];
   cy_val[xci].idx=xci;
   cy_val[xci].val=y[xci];
  }

#ifdef DEBUG
  printf("Before sort id x, y\n");
  for (xci=0; xci<N; xci++) {
   printf("%d %lf %lf\n",xci,cx_val[xci].val,cy_val[xci].val);
  }
#endif
  qsort(cx_val,N,sizeof(coordval_t),compare_coordinates);
  qsort(cy_val,N,sizeof(coordval_t),compare_coordinates);
#ifdef DEBUG
  printf("After sort id x, y\n");
  for (xci=0; xci<N; xci++) {
   printf("%d %lf %lf\n",xci,cx_val[xci].val,cy_val[xci].val);
  }
#endif

	/* merge axes coordinates */
	xci=yci=zci=0;
	while(xci<N && yci<N ) {
	 if (cx_val[xci].val==cy_val[yci].val){
		/* common index */
		grid[zci]=cx_val[xci].val;
    xindex[cx_val[xci].idx]=zci;
    yindex[cy_val[yci].idx]=zci;
	  zci++;
	  xci++;	 
	  yci++;	 
	 } else if (cx_val[xci].val<cy_val[yci].val){
		 grid[zci]=cx_val[xci].val;
     xindex[cx_val[xci].idx]=zci;
	   zci++;
	   xci++;	 
	 } else {
		 grid[zci]=cy_val[yci].val;
     yindex[cy_val[yci].idx]=zci;
	   zci++;
	   yci++;	 
	 }	 
	}
	/* copy the tail */
	if(xci<N && yci==N) {
		/* tail from x */
		while(xci<N) {
		 grid[zci]=cx_val[xci].val;
     xindex[cx_val[xci].idx]=zci;
	   zci++;
	   xci++;	 
		}
	} else if (xci==N && yci<N) {
		/* tail from y */
		while(yci<N) {
		 grid[zci]=cy_val[yci].val;
     yindex[cy_val[yci].idx]=zci;
	   zci++;
	   yci++;	 
		}
	}
	Ntot=zci;

	if (Ntot<2) {
		fprintf(stderr,"Error: Need at least 2 grid points\n");
		exit(1);
	}
#ifdef DEBUG
	printf("Input coord points\n");
	for (xci=0; xci<N; xci++) {
	 printf("[%d]=%lf %lf ",xci,x[xci],y[xci]);
	}
	printf("Grid\n");
	for (xci=0; xci<Ntot; xci++) {
	 printf("[%d]=%lf ",xci,grid[xci]);
	}
	printf("Index x,y\n");
	for (xci=0; xci<N; xci++) {
	 printf("[%d]=%d %d ",xci,xindex[xci],yindex[xci]);
	}
	printf("\n");
#endif
	/* wrap up negative values to positive ones if possible */
  if ((neg_grid=(int*)calloc((size_t)(Ntot),sizeof(int)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
	for (zci=0; zci<Ntot; zci++) {
			neg_grid[zci]=-1;
	}
	zci=Ntot-1;
	xci=0;
	/* find positive values to all negative values if possible */
	while(xci<Ntot && grid[xci]<0) {
	 /* try to find a positive value for this is possible */
	 while(zci>=0 && grid[zci]>0 && -grid[xci]<grid[zci]) {
				zci--;
	 }
	 /* now we might have found a correct value */
	 if (zci>=0 && grid[zci]>0 && -grid[xci]==grid[zci]) {
			neg_grid[xci]=zci;
	 }
	 xci++;
	}

#ifdef DEBUG
	printf("Neg grid\n");
	for (xci=0; xci<Ntot; xci++) {
	 printf("[%d]=%d ",xci,neg_grid[xci]);
	}
	printf("\n");
#endif


	/* set up factorial array */
  if ((fact=(double*)calloc((size_t)(n0),sizeof(double)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
  fact[0]=1;
	for (xci=1; xci<(n0); xci++) {
		fact[xci]=(xci)*fact[xci-1];
	}

#ifdef DEBUG
	printf("Factorials\n");
	for (xci=0; xci<(n0); xci++) {
	 printf("[%d]=%lf ",xci,fact[xci]);
	}
	printf("\n");
#endif

	/* setup array to store calculated shapelet value */
	/* need max storage Ntot x n0 */
  if ((shpvl=(double**)calloc((size_t)(Ntot),sizeof(double*)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
	for (xci=0; xci<Ntot; xci++) {
   if ((shpvl[xci]=(double*)calloc((size_t)(n0),sizeof(double)))==0) {
	   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	   exit(1);
	 }
	}

	/* start filling in the array from the positive values */
	for (zci=Ntot-1; zci>=0; zci--) {
    /* check to see if there are any positive values */
     if (neg_grid[zci] !=-1) {
			/* copy in the values from positive one, with appropriate sign change */
	     for (xci=0; xci<(n0); xci++) {
				 shpvl[zci][xci]=(xci%2==1?-shpvl[neg_grid[zci]][xci]:shpvl[neg_grid[zci]][xci]);
			 }
		 } else {
	     for (xci=0; xci<(n0); xci++) {
				/*take into account the scaling
				*/
				 double xvalt=grid[zci]/(beta);
				 shpvl[zci][xci]=H_e(xvalt,xci)*exp(-0.5*xvalt*xvalt)/sqrt((2<<xci)*fact[xci]);
		   }
		 }
	}


#ifdef DEBUG
  printf("x, shapelet val\n");
	for (zci=0; zci<Ntot; zci++) {
		printf("%lf= ",grid[zci]);
	  for (xci=0; xci<(n0); xci++) {
		  printf("%lf, ",shpvl[zci][xci]);
		}
		printf("\n");
	}
#endif

	/* now calculate the mode vectors */
	/* each vector is N length and there are n0*n0 of them */
	for (xci=0; xci<N; xci++) {
	for (n2=0; n2<(n0); n2++) {
	 for (n1=0; n1<(n0); n1++) {
      // per each point: nmodes values, idx: iterate over model for given point
      double prod=shpvl[xindex[xci]][n1]*shpvl[yindex[xci]][n2];
      output[n0*n0*xci+n2*n0+n1]=prod+_Complex_I*0.0;//real-valued basis;
		}
	 }
	}

#ifdef DEBUG
	printf("%%Matrix dimension=%d by %d\n",N,(n0)*(n0));
	for (xci=0; xci<N; xci++) {
  printf("%d ",xci);
	for (n1=0; n1<(n0); n1++) {
	 for (n2=0; n2<(n0); n2++) {
    printf("%lf ",creal(output[n0*n0*xci+n2*n0+n1]));
	 }
	 }
	 printf("\n");
	}
#endif
	free(grid);
	free(xindex);
	free(yindex);
  free(cx_val);
  free(cy_val);
	free(neg_grid);
	free(fact);
	for (xci=0; xci<Ntot; xci++) {
	 free(shpvl[xci]);
	}
	free(shpvl);
  return 0;
}


/* calculate recurrance relation
 * equations
% if l+m+n odd, = H(l,m,n)= 0
% H(0,0,0)=1, circular symmetry
% H(l+1,m,n) = 2 l (a^2-1) H(l-1,m,n) + 2m a b H(l,m-1,n) + 2n a c H(l,m,n-1)
% H(l,m+1,n) = 2 m (b^2-1) H(l,m-1,n) + 2n b c H(l,m,n-1) + 2l b a H(l-1,m,n)
% H(l,m,n+1) = 2 n (c^2-1) H(l,m,n-1) + 2l c a H(l-1,m,n) + 2m c b H(l,m-1,n)
l: 0..L-1, m: 0..M-1, n: 0..N-1

L: LxMxN storage

Note for large values of L,M,N the values of H will be very large

*/
static int
L_mat(int L, int M, int N, double a, double b, double c, double *H) {

  int *flag;
  /* Note: initially all flags are set to invalid: or to zero */
  if ((flag=(int*)calloc((size_t)(L*M*N),sizeof(int)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
  /* as the tensor is updated, corresponding flags are also set to valid */
  for (int l=0; l<L; l++) {
    for (int m=0; m<M; m++) {
      for (int n=0; n<N; n++) {
        /* H(0,0,0)=1 */
        if (!l && !m && !n) {
          H[l*M*N+m*N+n]=1.0;
          flag[l*M*N+m*N+n]=1;
        }
        /* l+m+n odd, H(l,m,n)= 0 */
        if ((l+m+n)%2 !=0) {
          H[l*M*N+m*N+n]=0.0;
          flag[l*M*N+m*N+n]=1;
        }

        if ((n+1<=N-1) && ((l+m+n+1)%2 ==0)) {
          /* valid lhs l,m,n+1 */
          /* H(l,m,n+1) = 2 n (c^2-1) H(l,m,n-1) + 2l c a H(l-1,m,n) + 2m c b H(l,m-1,n)
           * find valid rhs */
          double rhs=0.0;
          if ((n-1>=0) && flag[l*M*N+m*N+n-1]) {
            rhs += ((double)2*n)*(c*c-1.0)*H[l*M*N+m*N+n-1];
          }
          if ((l-1>=0) && flag[(l-1)*M*N+m*N+n]) {
            rhs += ((double)2*l)*(c*a)*H[(l-1)*M*N+m*N+n];
          }
          if ((m-1>=0) && flag[l*M*N+(m-1)*N+n]) {
            rhs += ((double)2*m)*(c*b)*H[l*M*N+(m-1)*N+n];
          }
          if (rhs!=0.0) {
            H[l*M*N+m*N+n+1]=rhs;
            flag[l*M*N+m*N+n+1]=1;
          }
        }

        if ((m+1<=M-1) && ((l+m+1+n)%2 ==0)) {
          /* valid lhs l,m+1,n */
          /* H(l,m+1,n) = 2 m (b^2-1) H(l,m-1,n) + 2n b c H(l,m,n-1) + 2l b a H(l-1,m,n)
           * find valid rhs */
          double rhs=0.0;
          if ((m-1>=0) && flag[l*M*N+(m-1)*N+n]) {
            rhs += ((double)2*m)*(b*b-1.0)*H[l*M*N+(m-1)*N+n];
          }
          if ((n-1>=0) && flag[l*M*N+m*N+n-1]) {
            rhs += ((double)2*n)*(b*c)*H[l*M*N+m*N+n-1];
          }
          if ((l-1>=0) && flag[(l-1)*M*N+m*N+n]) {
            rhs += ((double)2*l)*(b*a)*H[(l-1)*M*N+m*N+n];
          }
          if (rhs!=0.0) {
            H[l*M*N+(m+1)*N+n]=rhs;
            flag[l*M*N+(m+1)*N+n]=1;
          }
        }

        if ((l+1<=L-1) && ((l+1+m+n)%2 ==0)) {
          /* valid lhs l+1,m,n */
          /* H(l+1,m,n) = 2 l (a^2-1) H(l-1,m,n) + 2m a b H(l,m-1,n) + 2n a c H(l,m,n-1)
           * find valid rhs */
          double rhs=0.0;
          if ((l-1>=0) && flag[(l-1)*M*N+m*N+n]) {
            rhs += ((double)2*l)*(a*a-1.0)*H[(l-1)*M*N+m*N+n];
          }
          if ((m-1>=0) && flag[l*M*N+(m-1)*N+n]) {
            rhs += ((double)2*m)*(a*b)*H[l*M*N+(m-1)*N+n];
          }
          if ((n-1>=0) && flag[l*M*N+m*N+n-1]) {
            rhs += ((double)2*n)*(a*c)*H[l*M*N+m*N+n-1];
          }
          if (rhs!=0.0) {
            H[(l+1)*M*N+m*N+n]=rhs;
            flag[(l+1)*M*N+m*N+n]=1;
          }
        }

      }
    }
  }

  free(flag);

  return 0;
}


/* pre-calculate tensor B used in shapelet multiplication (1D)
 * h = f x g
 * where f, g, and h are given as (real space) 1D shapelet decompositions
 * h: L modes, alpha scale
 * f: M modes, beta scale
 * g: N modes, gamma scale
 * output :
 * B: LxMxN tensor C(l,m,n) : out
 * The same B can be used in 2D decompositions by using the (kronecker) product
 */
int
shapelet_product_tensor(int L, int M, int N, double alpha, double beta, double gamma,
    double *B) {

  double *H;
  if ((H=(double*)calloc((size_t)(L*M*N),sizeof(double)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}

  double nu=1.0/sqrt(1.0/(alpha*alpha)+1.0/(beta*beta)+1.0/(gamma*gamma));
  L_mat(L, M, N, sqrt(2.0)*nu/alpha, sqrt(2.0)*nu/beta, sqrt(2.0)*nu/gamma, H);

  /* setup factorial array : max value of L,M or N*/
  double *fact;
  int n0=MAX(L,MAX(M,N));
  if ((fact=(double*)calloc((size_t)(n0),sizeof(double)))==0) {
	  fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
	  exit(1);
	}
  fact[0]=1.0;
	for (int ci=1; ci<(n0); ci++) {
		fact[ci]=(ci)*fact[ci-1];
	}

  /* B_lmn(a,b,c) = nu/sqrt(-2^{l+m+n} sqrt(pi) l! m! n! a b c) L_lmn(sqrt(2)nu/a, sqrt(2)nu/b, sqrt(2)nu/c)
  */
  for (int l=0; l<L; l++) {
    for (int m=0; m<M; m++) {
      for (int n=0; n<N; n++) {
        /* only even l+m+n are non zero, so -2^{l+m+n}=2^{l+m+n} */
        if ((l+m+n)%2 ==0) {
         /* note that we use column major order in B, row m and col n = m+nM, NOT mN+n */
         B[l*M*N+m+n*M]=nu/sqrt((double)(1<<(l+m+n))*sqrt(M_PI)* fact[l]*fact[m]*fact[n]*alpha*beta*gamma) * H[l*M*N+m*N+n];
        } else {
         B[l*M*N+m+n*M]=0.0;
        }
      }
    }
  }

  /* Normalize B to account for the very large values expected in H */
  double Bnorm=my_dnrm2(L*M*N,B);
  my_dscal(L*M*N,1.0/Bnorm,B);

  free(H);
  free(fact);

  return 0;
}


/* find C=kron(A,B)
 * A: MxN input
 * B: MxN input
 * Note: A and B can be the same
 * C: M^2 x N^2 output
 * All matrices in column major order, (row i, col j) = i + j*M
 */
static int
kronecker_prod(int M, int N, double *A, double *B, double *C) {
  /* storage for a_ij B */
  double *aB;
  if ((aB=(double*)calloc((size_t)(M*N),sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      /* find a_ij B */
      my_dcopy(M*N,B,1,aB,1);
      my_dscal(M*N,A[i+j*M],aB);

      /* starting location of C to copy a_ij B */
      /* i rows of M blocks = iM, j columns of N blocks, each of size M^2 = jN*M^2 */
      int st=i*M+j*N*(M*M);
      /* start copying N columns of a_ij B to C */
      for (int col=0; col<N; col++) {
        my_dcopy(M,&aB[col*M],1,&C[st+col*M*M],1);
      }
    }
  }

  free(aB);
  return 0;
}

/* find in terms of shapelet decompositions (2D)
 * h = f x g
 * where f, g, and h are given as (real space) shapelet decompositions
 * h: L^2 modes, alpha scale
 * f: M^2 modes, beta scale
 * g: N^2 modes, gamma scale
 * input : f, g
 * output : h
 * h: LxL modes : out
 * f: MxM modes : in 
 * g: NxN modes : in
 * C: LxMxN tensor C(l,m,n) : in (pre-calculated)
 */
int
shapelet_product(int L, int M, int N, double alpha, double beta, double gamma,
    double *h, double *f, double *g, double *C) {

  for (int m=0; m<M*M; m++) {
    printf("f %d %lf\n",m,f[m]);
  }
  for (int n=0; n<N*N; n++) {
    printf("g %d %lf\n",n,g[n]);
  }
  for (int l=0; l<L; l++) {
    printf("%d =\n",l);
    for (int m=0; m<M; m++) {
      for (int n=0; n<N; n++) {
         /* column major order */
         printf("%lf ",C[l*M*N+m+n*M]);
      }
      printf("\n");
    }
  }

  /* find f x g^T : M^2 x N^2 matrix, stored as M^2*N^2 vector */
  double *fg;
  if ((fg=(double*)calloc((size_t)(M*M*N*N),sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (int ci=0; ci<N*N; ci++) {
    my_dcopy(M*M,f,1,&fg[ci*M*M],1);
    my_dscal(M*M,g[ci],&fg[ci*M*M]);
  }

  printf("fg=\n");
  for (int ci=0; ci<M*M; ci++) {
    for (int cj=0; cj<N*N; cj++) {
      printf("%lf ",fg[ci+cj*M*M]);
    }
    printf("\n");
  }

  /* storage to find kronecker product */
  double *Cl;
  if ((Cl=(double*)calloc((size_t)(M*M*N*N),sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (int l1=0; l1<L; l1++) {
    double *Cl1=&C[l1*M*N]; //C[l1,:,:] MxN matrix
    for (int l2=0; l2<L; l2++) {
      double *Cl2=&C[l2*M*N]; //C[l2,:,:] MxN matrix
      /* find kronecker product Cl=kron(Cl2,Cl1) */
      kronecker_prod(M,N,Cl2,Cl1,Cl);
      /* Hadamard prod Cl . fg and sum */
      double sum=0.0;
#pragma GCC ivdep
      for (int ci=0; ci<M*M*N*N; ci++) {
        sum+=Cl[ci]*fg[ci];
      }
      printf("h(%d,%d) %lf\n",l1,l2,sum);
      h[l1+l2*L]=sum;
    }
  }

  for (int l=0; l<L*L; l++) {
    printf("h %d %lf\n",l,h[l]);
  }

  free(fg);
  free(Cl);
  return 0;
}

/* Jones matrix multiplication
   C=A*B
*/
static void
amb(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=(a[0]*b[0]+a[1]*b[2]);
 c[1]=(a[0]*b[1]+a[1]*b[3]);
 c[2]=(a[2]*b[0]+a[3]*b[2]);
 c[3]=(a[2]*b[1]+a[3]*b[3]);
}

/* Jones matrix multiplication
   C=A*B^H
*/
static void
ambt(complex double * __restrict a, complex double * __restrict b, complex double * __restrict c) {
 c[0]=a[0]*conj(b[0])+a[1]*conj(b[1]);
 c[1]=a[0]*conj(b[2])+a[1]*conj(b[3]);
 c[2]=a[2]*conj(b[0])+a[3]*conj(b[1]);
 c[3]=a[2]*conj(b[2])+a[3]*conj(b[3]);
}


/* find in terms of shapelet decompositions (2D)
 * h = f x g
 * where f, g, and h are given as shapelet decompositions, each item in h,f,g are Jones matrices of 2x2 size
 * h: L^2 x 2x2 modes, alpha scale
 * f: M^2 x 2x2 modes, beta scale
 * g: N^2 x 2x2 modes, gamma scale
 * hermitian: if 1, find f x g^H (Jones matrix Hermitian)
 * input : f, g
 * output : h
 * h: LxL modes : out
 * f: MxM modes : in 
 * g: NxN modes : in
 * C: LxMxN tensor C(l,m,n) : in (pre-calculated)
 */
int
shapelet_product_jones(int L, int M, int N, double alpha, double beta, double gamma,
    complex double *h, complex double *f, complex double *g, double *C, int hermitian) {

  //printf("Input h %dx%d %lf f %dx%d %lf g %dx%d %lf (x 4)\n",L,L,alpha,M,M,beta,N,N,gamma);
#ifdef DEBUG
  for (int m=0; m<M*M; m++) {
    printf("f %d %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf)\n",m,creal(f[4*m]),cimag(f[4*m]),creal(f[4*m+1]),cimag(f[4*m+1]),creal(f[4*m+2]),cimag(f[4*m+2]),creal(f[4*m+3]),cimag(f[4*m+3]));
  }
  for (int m=0; m<N*N; m++) {
    printf("g %d %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf)\n",m,creal(g[4*m]),cimag(g[4*m]),creal(g[4*m+1]),cimag(g[4*m+1]),creal(g[4*m+2]),cimag(g[4*m+2]),creal(g[4*m+3]),cimag(g[4*m+3]));
  }
  for (int l=0; l<L; l++) {
    printf("%d =\n",l);
    for (int m=0; m<M; m++) {
      for (int n=0; n<N; n++) {
         /* column major order */
         printf("%lf ",C[l*M*N+m+n*M]);
      }
      printf("\n");
    }
  }
#endif

  /* cannot find f x g^T : M^2 x N^2 matrix, stored as M^2*N^2 vector
   * because each item in f,g are 2x2 matrices,
   * fg: rows 4*M^2 cols N^2 */
  complex double *fg;
  if ((fg=(complex double*)calloc((size_t)(M*M*N*N*4),sizeof(complex double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (int m=0; m<M*M; m++) {
    for (int n=0; n<N*N; n++) {
      /* select J1=f[4*m,..4*m+3] and J2=g[4*n..4*n+3],
       * find J=J1 x J2^H, or J=J1 x J2, store at fg[4*M^2*n+4*m] */
      if (hermitian) {
       ambt(&f[4*m],&g[4*n],&fg[4*M*M*n+4*m]);
      } else {
       amb(&f[4*m],&g[4*n],&fg[4*M*M*n+4*m]);
      }
    }
  }

#ifdef DEBUG
  for (int m=0; m<M*M; m++) {
    for (int n=0; n<N*N; n++) {
      printf("fg %d %d %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf)\n",creal(fg[4*M*M*n+4*m]),cimag(fg[4*M*M*n+4*m]),creal(fg[4*M*M*n+4*m+1]),cimag(fg[4*M*M*n+4*m+1]),creal(fg[4*M*M*n+4*m+2]),cimag(fg[4*M*M*n+4*m+2]),creal(fg[4*M*M*n+4*m+3]),cimag(fg[4*M*M*n+4*m+3]));
    }
  }
#endif
 
  /* storage to find kronecker product */
  double *Cl;
  if ((Cl=(double*)calloc((size_t)(M*M*N*N),sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (int l1=0; l1<L; l1++) {
    double *Cl1=&C[l1*M*N]; //C[l1,:,:] MxN matrix
    for (int l2=0; l2<L; l2++) {
      double *Cl2=&C[l2*M*N]; //C[l2,:,:] MxN matrix
      /* find kronecker product Cl=kron(Cl2,Cl1) */
      kronecker_prod(M,N,Cl2,Cl1,Cl);
      /* Hadamard prod Cl . (f x g)  and sum, fxg=J_f * J_g^H, 2x2 product */
      complex double sum[4]={0.0,0.0,0.0,0.0};
#pragma GCC ivdep
      for (int ci=0; ci<M*M*N*N; ci++) {
        sum[0]+=Cl[ci]*fg[4*ci];
        sum[1]+=Cl[ci]*fg[4*ci+1];
        sum[2]+=Cl[ci]*fg[4*ci+2];
        sum[3]+=Cl[ci]*fg[4*ci+3];
      }
//      printf("h(%d,%d) %lf+j*(%lf)\n",l1,l2,creal(sum[0]),cimag(sum[0]));
//      printf("h(%d,%d) %lf+j*(%lf)\n",l1,l2,creal(sum[1]),cimag(sum[1]));
//      printf("h(%d,%d) %lf+j*(%lf)\n",l1,l2,creal(sum[2]),cimag(sum[2]));
//      printf("h(%d,%d) %lf+j*(%lf)\n",l1,l2,creal(sum[3]),cimag(sum[3]));
      h[(l1+l2*L)*4]=sum[0];
      h[(l1+l2*L)*4+1]=sum[1];
      h[(l1+l2*L)*4+2]=sum[2];
      h[(l1+l2*L)*4+3]=sum[3];
    }
  }

#ifdef DEBUG
  for (int m=0; m<L*L; m++) {
    printf("h %d %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf) %lf+j*(%lf)\n",m,creal(h[4*m]),cimag(h[4*m]),creal(h[4*m+1]),cimag(h[4*m+1]),creal(h[4*m+2]),cimag(h[4*m+2]),creal(h[4*m+3]),cimag(h[4*m+3]));
  }
#endif

  //printf("Norms Cl:%e f:%e g:%e h:%e\n",my_dnrm2(M*M*N*N,Cl),my_dnrm2(4*M*M,(double*)f),my_dnrm2(4*N*N,(double*)g),my_dnrm2(4*L*L,(double*)h));
  free(fg);
  free(Cl);
  return 0;
}


/* 
 * Zspat: spatial model storage 4*N*Npoly*G, shape 2*N*Npoly x 2 : Note 2 cols!
 * B: consensus polynomials Npoly*Nfreq
 * Npoly: consensus poly (in freq) terms
 * N: stations
 * n0: spatial terms (G = n0 x n0)
 * axes_M: image size axes_M x axes_M
 * freq: which freq to plot ? 0...Nfreq-1
 * plot_type: what to plot, 0: ||J||^2, ....
 * basis: spatial basis type: SP_SHAPELET, SP_SHARMONIC
 * beta: for shapelet basis, scale factor
 * filename: output filename
 *
 */
int
plot_spatial_model(complex double *Zspat, double *B, int Npoly, int N, int n0, int Nfreq, int axes_M, int freq, int plot_type, int spatialreg_basis, double beta, const char *filename) {

   int G=n0*n0;
   double *pn_ll=0,*pn_mm=0;
   double *pn_phi=0,*pn_theta=0;
   int pn_grid_M=axes_M*axes_M;
   complex double *pn_phivec=0; /* basis vector */
   complex double *pn_Phi=0; /* basis matrix */
   complex double *pn_Zbar=0; /* constraint for each pixel, 2*Npoly*N x 2 x pn_grid_M */
   double *J=0; /* storage for B_f Z, for selected freq, all pixels, (complex) 2Nx2xpn_grid_M */


   /* plotting : initialize */
   if ((pn_ll=(double*)calloc((size_t)axes_M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
   if ((pn_mm=(double*)calloc((size_t)axes_M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
   if ((pn_phivec=(complex double*)calloc((size_t)pn_grid_M*G,sizeof(complex double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
   /* depending on the basis, determine plotting domain,
    * shapelet: [-beta*n0/2,beta*n0/2], spherical harmonic: [-0.9,0.9] */
   double plot_limit=(spatialreg_basis==SP_SHAPELET?0.5*beta*(double)n0:0.9);
   for (int ci=0; ci<axes_M; ci++) {
       pn_ll[ci]=(plot_limit-(-plot_limit))*((double)ci+0.5)/(double)(axes_M)-plot_limit;
       pn_mm[ci]=pn_ll[ci];
   }
   if ((pn_theta=(double*)calloc((size_t)pn_grid_M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
   if ((pn_phi=(double*)calloc((size_t)pn_grid_M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }

   for (int ci=0; ci<axes_M; ci++) {
       for (int cj=0; cj<axes_M; cj++) {
          if (spatialreg_basis==SP_SHAPELET) {
           /* diffuse sky shapelet model is in (-l,m) so negate */
           pn_theta[ci*axes_M+cj]=-pn_ll[ci];
           pn_phi[ci*axes_M+cj]=pn_mm[cj];
          } else {
           /* map (l,m) to r [0,pi/2] and theta[0,2*pi] */
           double rr=sqrt(pn_ll[ci]*pn_ll[ci]+pn_mm[cj]*pn_mm[cj])*M_PI_2;
           double tt=atan2(pn_mm[cj],pn_ll[ci]);
           pn_theta[ci*axes_M+cj]=rr;
           pn_phi[ci*axes_M+cj]=tt;
          }
       }
   }

   if (spatialreg_basis==SP_SHAPELET) {
       shapelet_modes(n0,beta,pn_theta,pn_phi,pn_grid_M,pn_phivec);
   } else {
       sharmonic_modes(n0,pn_theta,pn_phi,pn_grid_M,pn_phivec);
   }
   /* Phi = I \kron phi_vec : (2Gx2) x pixels */
   if ((pn_Phi=(complex double*)calloc((size_t)pn_grid_M*2*G*2,sizeof(complex double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
   for (int ci=0; ci<pn_grid_M; ci++) {
       memcpy(&pn_Phi[ci*2*G*2],&pn_phivec[ci*G],G*sizeof(complex double));
       memcpy(&pn_Phi[ci*2*G*2+3*G],&pn_phivec[ci*G],G*sizeof(complex double));
   }

   if ((pn_Zbar=(complex double*)calloc((size_t)N*4*Npoly*pn_grid_M,sizeof(complex double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
   }
   if ((J=(double*)calloc((size_t)N*8*pn_grid_M,sizeof(double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
   }


   /* Z_k = Zspat Phi_k : (2*2*N x Npoly) x pixels, same row ordering as Z */
   /* Z_k : 2*Npoly*N x 2 */
   for(int cm=0; cm<pn_grid_M; cm++) {
     my_zgemm('N','N',2*Npoly*N,2,2*G,1.0+_Complex_I*0.0,Zspat,2*Npoly*N,&pn_Phi[cm*2*G*2],2*G,0.0+_Complex_I*0.0,&pn_Zbar[cm*2*Npoly*N*2],2*Npoly*N);
   }

   /* evaluate B_f Z_k, k all pixels : (2Nx2) x pixels */
   for (int p=0; p<pn_grid_M; p++) {
     memset(&J[8*N*p],0,sizeof(double)*(size_t)N*8);
     for (int ci=0; ci<Npoly; ci++) {
       /* 2 columns separately */
       my_daxpy(4*N, (double*)&pn_Zbar[p*4*N*Npoly+ci*2*N], B[freq*Npoly+ci], &J[8*N*p]);
       my_daxpy(4*N, (double*)&pn_Zbar[p*4*N*Npoly+ci*2*N+2*N*Npoly], B[freq*Npoly+ci], &J[8*N*p+4*N]);
     }
   }

   /* re-arrange pixel values into pn_grid_M blocks, N times for stations */
   double *pixval=0;
   if ((pixval=(double*)calloc((size_t)pn_grid_M*N,sizeof(double)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
   }

//#define DEBUG
#ifdef DEBUG
       FILE *dfp;
       if ((dfp=fopen("debug.m","w+"))==0) {
        fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
        exit(1);
       }
       fprintf(dfp,"N=%d;\nM=%d;\n",N,axes_M);
       fprintf(dfp,"%% each station will have M*M*5 values, offset 8*N\n");
       fprintf(dfp,"%% for example J1_11=Jvec(1:4*N:end); J1_11=reshape(J1_11,M,M);\n");
       fprintf(dfp,"%% and J2_11=Jvec(1*4+1:4*N:end);\n");
       fprintf(dfp,"Jvec=[\n");
       for (int ci=0; ci<pn_grid_M*4*N; ci++) {
         fprintf(dfp,"%lf+j*(%lf)\n",J[2*ci],J[2*ci+1]);
       }
       fprintf(dfp,"];\n");
       //fprintf(dfp,"Pix=[\n");
       //for (int ci=0; ci<pn_grid_M*N; ci++) {
       //  fprintf(dfp,"%lf\n",pixval[ci]);
       //}
       //fprintf(dfp,"];\n");

       /* Jvec above is calculated as 1) Z_k = Z Phi_k and 2) J = B_f Z_k 
        * this can also be calculated as 1) Z_f = B_f Z and 2) J = Z_f Phi_k
        * do the second form and compare 
        */
       complex double *Z_f=0; /* 2N x 2G */
       double *J_f=0; /* 2N x 2 x pixels */
       if ((Z_f=(complex double*)calloc((size_t)N*4*G,sizeof(complex double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
       }
       if ((J_f=(double*)calloc((size_t)N*8*pn_grid_M,sizeof(double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
       }
       /* Z_f = B_f Z : 2N x 2G */
       for (int col=0; col<2*G; col++) {
         for (int np=0; np<Npoly; np++) {
           my_daxpy(4*N, (double*)&Zspat[col*2*N*Npoly+np*2*N], B[freq*Npoly+np], (double*)&Z_f[col*2*N]);
         }
       }

       /* J = Z_f Phi_k for all pixels */
       for(int cm=0; cm<pn_grid_M; cm++) {
         my_zgemm('N','N',2*N,2,2*G,1.0+_Complex_I*0.0,Z_f,2*N,&pn_Phi[cm*2*G*2],2*G,0.0+_Complex_I*0.0,(complex double *)&J_f[cm*2*N*2*2],2*N);
       }

       fprintf(dfp,"Jvec1=[\n");
       for (int ci=0; ci<pn_grid_M*4*N; ci++) {
         fprintf(dfp,"%lf+j*(%lf)\n",J_f[2*ci],J_f[2*ci+1]);
       }
       fprintf(dfp,"];\n");

       fclose(dfp);
#endif /* DEBUG */

   double *pn_J=J; // or J_f
   for (int cm=0; cm<N; cm++) {
     if (plot_type==0) { /* ||J||^2 */
#pragma GCC ivdep
         for (int ci=0; ci<pn_grid_M; ci++) {
          pixval[ci+cm*pn_grid_M]=pn_J[ci*8*N+4*cm]*pn_J[ci*8*N+4*cm] // real J11
              +pn_J[ci*8*N+4*cm+1]*pn_J[ci*8*N+4*cm+1] // imag J11
              +pn_J[ci*8*N+4*cm+2]*pn_J[ci*8*N+4*cm+2] // real J21
              +pn_J[ci*8*N+4*cm+3]*pn_J[ci*8*N+4*cm+3] // imag J21
              +pn_J[ci*8*N+4*cm+4*N+0]*pn_J[ci*8*N+4*cm+4*N+0] // real J12
              +pn_J[ci*8*N+4*cm+4*N+1]*pn_J[ci*8*N+4*cm+4*N+1] // imag J12
              +pn_J[ci*8*N+4*cm+4*N+2]*pn_J[ci*8*N+4*cm+4*N+2] // real J22
              +pn_J[ci*8*N+4*cm+4*N+3]*pn_J[ci*8*N+4*cm+4*N+3]; // imag J22

         }
     } else if (plot_type==1) { /* angle(J11) */
#pragma GCC ivdep
         for (int ci=0; ci<pn_grid_M; ci++) {
           pixval[ci+cm*pn_grid_M]=atan2(pn_J[4*cm+ci*8*N+1],pn_J[4*cm+ci*8*N]);
         }
     } else if (plot_type==2) { /* angle(J22) */
#pragma GCC ivdep
         for (int ci=0; ci<pn_grid_M; ci++) {
           pixval[ci+cm*pn_grid_M]=atan2(pn_J[4*cm+ci*8*N+4*N+3],pn_J[4*cm+ci*8*N+4*N+2]);
         }
     }
   }
   convert_tensor_to_image(pixval, filename, N, axes_M, 1); // last argument ==1 : normalize

   free(pixval);
   free(pn_ll);
   free(pn_mm);
   free(pn_theta);
   free(pn_phi);
   free(pn_phivec);
   free(pn_Phi);
   free(pn_Zbar);
   free(J);
#ifdef DEBUG
    free(Z_f);
    free(J_f);
#endif


   return 0;
}
