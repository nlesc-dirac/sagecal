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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Dirac_radio.h"

/* evaluate Hermite polynomial value using recursion
 */
static double 
H_e(double x, int n) {
	if(n==0) return 1.0;
	if(n==1) return 2*x;
	return 2*x*H_e(x,n-1)-2*(n-1)*H_e(x,n-2);
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
      output[n0*n0*xci+n2*n0+n1]=prod+_Complex_I*prod;
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
