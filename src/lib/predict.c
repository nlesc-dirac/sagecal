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


#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "sagecal.h"

/******************** shapalet stuff **********************/
/* evaluate Hermite polynomial value using recursion
 */
static double
H_e(double x, int n) {
  if(n==0) return 1.0;
  if(n==1) return 2*x;
  return 2*x*H_e(x,n-1)-2*(n-1)*H_e(x,n-2);
}

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
		fact[xci]=(xci+1)*fact[xci-1];
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
	for (xci=0; xci<n0; xci++) {
		xval=u*beta;
		shpvl[zci][xci]=H_e(xval,xci)*exp(-0.5*xval*xval)/sqrt((double)(2<<xci)*fact[xci]);
	}
	zci=1;
	for (xci=0; xci<n0; xci++) {
		xval=v*beta;
		shpvl[zci][xci]=H_e(xval,xci)*exp(-0.5*xval*xval)/sqrt((double)(2<<xci)*fact[xci]);
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



/************* extended source contributions ************/

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


complex double
gaussian_contrib(void*dd, double u, double v, double w) {
  exinfo_gaussian *dp=(exinfo_gaussian*)dd;
  double up,vp,a,b,ut,vt,cosph,sinph;

  /* first the rotation due to projection */
  if (dp->use_projection) {
   up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
   vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);
  } else {
   up=u;
   vp=v;
  }

  /* linear transformations, if any */
  a=dp->eX;
  b=dp->eY;
  //cosph=cos(dp->eP);
  //sinph=sin(dp->eP);
  sincos(dp->eP,&sinph,&cosph);
  ut=a*(cosph*up-sinph*vp);
  vt=b*(sinph*up+cosph*vp);

  /* Fourier TF is normalized with integrated flux,
    so to get the peak value right, scale the flux */
  //return 0.5*exp(-(ut*ut+vt*vt))/sqrt(2.0*a*b);
  return 0.5*M_PI*exp(-(ut*ut+vt*vt));
}

complex double
ring_contrib(void*dd, double u, double v, double w) {
  exinfo_ring *dp=(exinfo_ring*)dd;
  double up,vp,a,b;

  /* first the rotation due to projection */
  up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
  vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);

  a=dp->eX; /* diameter */ 
  b=sqrt(up*up+vp*vp)*a*2.0*M_PI; 

  return j0(b);
}

complex double
disk_contrib(void*dd, double u, double v, double w) {
  exinfo_disk *dp=(exinfo_disk*)dd;
  double up,vp,a,b;

  /* first the rotation due to projection */
  up=u*(dp->cxi)-v*(dp->cphi)*(dp->sxi)+w*(dp->sphi)*(dp->sxi);
  vp=u*(dp->sxi)+v*(dp->cphi)*(dp->cxi)-w*(dp->sphi)*(dp->cxi);

  a=dp->eX; /* diameter */ 
  b=sqrt(up*up+vp*vp)*a*2.0*M_PI; 

  return j1(b);
}

/* worker thread function for prediction */
static void *
predict_threadfn(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 int ci,cm,cn;
 double phterm,sinph,cosph;
 complex double prodterm;
 complex double C[4];
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   memset(&(t->x[8*ci]),0,sizeof(double)*8);
   /* if this baseline is flagged, we do not compute */
   if (!t->barr[ci+t->boff].flag) {
    for (cm=0; cm<(t->M); cm++) { /* clusters */

     memset(C,0,sizeof(complex double)*4);
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* note u=u/c, v=v/c, w=w/c here */
       /* phterm is 2pi(u/c l +v/c m +w/c n) */
       phterm=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[cn]+t->v[ci]*t->carr[cm].mm[cn]+t->w[ci]*t->carr[cm].nn[cn]);
       //sinph=sin(phterm*t->freq0);
       //cosph=cos(phterm*t->freq0);
       sincos(phterm*t->freq0,&sinph,&cosph);
       /* calculate coherency, NOT scaled by 1/2 */
       prodterm=t->carr[cm].sI[cn]*(cosph+_Complex_I*sinph);
       /* freq smearing : extra term delta * sinc(delta/2 * phterm) */
       phterm*=t->fdelta*0.5;
       if (phterm!=0.0) {
         sinph=sin(phterm)/phterm;
         prodterm*=sinph;
       }
       /* full formula
             exp(j*2*pi*k)*sinc(2*pi*k*delta/2), k=(ul+vm+wn)
        */
       
       /* check if source type is not a point source for additional 
          calculations */
       if (t->carr[cm].stype[cn]==STYPE_POINT) {
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_SHAPELET) {
        prodterm*=shapelet_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_GAUSSIAN) {
        prodterm*=gaussian_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_DISK) {
        prodterm*=disk_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_RING) {
        prodterm*=ring_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       }
     }
     /* add to baseline visibilities */
     t->x[8*ci]+=creal(C[0]);
     t->x[8*ci+1]+=cimag(C[0]);
     t->x[8*ci+6]+=creal(C[3]);
     t->x[8*ci+7]+=cimag(C[3]);
    }
   }
 }

 return NULL;
}

int
predict_visibilities(double *u, double *v, double *w, double *x, int N,   
   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, int Nt) {
  /* u,v,w : size Nbase*tilesz x 1  x: size Nbase*8*tilesz x 1 */
  /* barr: size Nbase*tilesz x 1 carr: size Mx1 */

  int nth,nth1,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  int Nbase1=Nbase*tilesz;

  /* calculate min baselines a thread can handle */
  //Nthb0=ceil((double)Nbase1/(double)Nt);
  Nthb0=(Nbase1+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase1; nth++) {
    /* this thread will handle baselines [ci:min(Nbase1-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase1) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase1-ci;
    }
    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]); 
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].x=&(x[8*ci]);
    threaddata[nth].freq0=freq0;
    threaddata[nth].fdelta=fdelta;
#ifdef DEBUG
    printf("thread %d writing to data from %d baselines %d\n",nth,8*ci,Nthb);
#endif
    
    
    pthread_create(&th_array[nth],&attr,predict_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for precalculation*/
static void *
precal_threadfn(void *data) {
 thread_data_base_t *t=(thread_data_base_t*)data;
 
 /* memory ordering: x[0:4M-1] baseline 0
                     x[4M:2*4M-1] baseline 2 ... */
 int ci,cm,cn;
 int M=(t->M);
 double phterm,sinph,cosph,uvdist;
 complex double prodterm;
 complex double C[4];
 for (ci=0; ci<t->Nb; ci++) {
   /* iterate over the sky model and calculate contribution */
   /* for this x[8*ci:8*(ci+1)-1] */
   memset(&(t->coh[4*M*ci]),0,sizeof(complex double)*4*M);
   /* even if this baseline is flagged, we do compute */
    for (cm=0; cm<M; cm++) { /* clusters */
     memset(C,0,sizeof(complex double)*4);
     for (cn=0; cn<t->carr[cm].N; cn++) {
       /* note u=u/c, v=v/c, w=w/c here */
       /* phterm is 2pi(u/c l +v/c m +w/c n) */
       phterm=2.0*M_PI*(t->u[ci]*t->carr[cm].ll[cn]+t->v[ci]*t->carr[cm].mm[cn]+t->w[ci]*t->carr[cm].nn[cn]);
       //sinph=sin(phterm*t->freq0);
       //cosph=cos(phterm*t->freq0);
       sincos(phterm*t->freq0,&sinph,&cosph);

       /* caltulate coherency, NOT scaled by 1/2 */
       prodterm=t->carr[cm].sI[cn]*(cosph+_Complex_I*sinph);
       /* freq smearing : extra term delta * sinc(delta/2 * phterm) */
       phterm*=t->fdelta*0.5;
       if (phterm!=0.0) {
         sinph=sin(phterm)/phterm;
         prodterm*=sinph;
       }
       /* full formula
             exp(j*2*pi*k)*sinc(2*pi*k*delta/2), k=(ul+vm+wn)
        */
       

       /* check if source type is not a point source for additional 
          calculations */
       if (t->carr[cm].stype[cn]==STYPE_POINT) {
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_SHAPELET) {
        prodterm*=shapelet_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_GAUSSIAN) {
        prodterm*=gaussian_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_DISK) {
        prodterm*=disk_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       } else if (t->carr[cm].stype[cn]==STYPE_RING) {
        prodterm*=ring_contrib(t->carr[cm].ex[cn],t->u[ci]*t->freq0,t->v[ci]*t->freq0,t->w[ci]*t->freq0);
        C[0]+=prodterm;
        C[3]+=prodterm;
       }
     }
     /* add to baseline visibilities */
     t->coh[4*M*ci+4*cm]=C[0];
     t->coh[4*M*ci+4*cm+1]=C[1];
     t->coh[4*M*ci+4*cm+2]=C[2];
     t->coh[4*M*ci+4*cm+3]=C[3];
    }
    if (!t->barr[ci+t->boff].flag) {
    /* change the flag to 2 if baseline length is < uvmin */
    uvdist=sqrt(t->u[ci]*t->u[ci]+t->v[ci]*t->v[ci])*t->freq0;
    if (uvdist<t->uvmin) {
      t->barr[ci+t->boff].flag=2;
    }
   }
 }

 return NULL;
}


int
precalculate_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double uvmin, int Nt) {
  /* u,v,w : size Nbasex 1  x: size Nbase*4*M x 1 */
  /* barr: size Nbasex 1 carr: size Mx1 */
  /* ordering of x: [0,4M-1] coherencies for baseline 0
                    [4M,2*4M-1] coherencies for baseline 1 ... */

  int nth,nth1,ci;

  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_base_t *threaddata;

  /* calculate min baselines a thread can handle */
  //Nthb0=ceil((double)Nbase/(double)Nt);
  Nthb0=(Nbase+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_base_t*)malloc((size_t)Nt*sizeof(thread_data_base_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }

    threaddata[nth].boff=ci;
    threaddata[nth].Nb=Nthb;
    threaddata[nth].barr=barr;
    threaddata[nth].u=&(u[ci]); 
    threaddata[nth].v=&(v[ci]);
    threaddata[nth].w=&(w[ci]);
    threaddata[nth].carr=carr;
    threaddata[nth].M=M;
    threaddata[nth].uvmin=uvmin;
    threaddata[nth].coh=&(x[4*M*ci]);
    threaddata[nth].freq0=freq0;
    threaddata[nth].fdelta=fdelta;
    //printf("thread %d precalculating to data from %d baselines %d\n",nth,4*M*ci,Nthb);
    pthread_create(&th_array[nth],&attr,precal_threadfn,(void*)(&threaddata[nth]));
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




/* worker thread function for rearranging coherencies*/
static void *
rearrange_threadfn(void *data) {
 thread_data_coharr_t *t=(thread_data_coharr_t*)data;
 
 /* NOTE: t->ddcoh must have all zeros intially (using calloc) */
 int ci,cj;
 double *realcoh=(double*)t->coh;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
   if (!t->barr[ci].flag) {
     t->ddbase[2*ci]=(char)t->barr[ci].sta1;
     t->ddbase[2*ci+1]=(char)t->barr[ci].sta2;
     /* loop over directions and copy coherencies */
     for (cj=0; cj<t->M; cj++) {
       memcpy(&(t->ddcoh[cj*(t->Nbase)*8+8*ci]),&realcoh[8*cj+8*(t->M)*ci],8*sizeof(double));
     }
   } else {
     t->ddbase[2*ci]=t->ddbase[2*ci+1]=-1;
   }
 }

 return NULL;
}

/* rearranges coherencies for GPU use later */
/* barr: 2*Nbase x 1
   coh: M*Nbase*4 x 1 complex
   ddcoh: M*Nbase*8 x 1
   ddbase: 2*Nbase x 1
*/
int
rearrange_coherencies(int Nbase, baseline_t *barr, complex double *coh, double *ddcoh, char *ddbase, int M, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_coharr_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_coharr_t*)malloc((size_t)Nt*sizeof(thread_data_coharr_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].M=M;
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].barr=barr;
    threaddata[nth].coh=coh;
    threaddata[nth].ddcoh=ddcoh;
    threaddata[nth].ddbase=ddbase;
    pthread_create(&th_array[nth],&attr,rearrange_threadfn,(void*)(&threaddata[nth]));
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

/* worker thread function for rearranging baselines*/
static void *
rearrange_base_threadfn(void *data) {
 thread_data_coharr_t *t=(thread_data_coharr_t*)data;
 
 int ci;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
   if (!t->barr[ci].flag) {
     t->ddbase[2*ci]=(char)t->barr[ci].sta1;
     t->ddbase[2*ci+1]=(char)t->barr[ci].sta2;
   } else {
     t->ddbase[2*ci]=t->ddbase[2*ci+1]=-1;
   }
 }

 return NULL;
}

/* rearranges baselines for GPU use later */
/* barr: 2*Nbase x 1
   ddbase: 2*Nbase x 1
*/
int
rearrange_baselines(int Nbase, baseline_t *barr, char *ddbase, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_coharr_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_coharr_t*)malloc((size_t)Nt*sizeof(thread_data_coharr_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].barr=barr;
    threaddata[nth].ddbase=ddbase;
    pthread_create(&th_array[nth],&attr,rearrange_base_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for preflag data */
static void *
preflag_threadfn(void *data) {
 thread_data_preflag_t *t=(thread_data_preflag_t*)data;
 
 int ci;
 for (ci=t->startbase; ci<=t->endbase; ci++) {
   if (t->flag[ci]>0.0) { /* flagged data */
     t->barr[ci].flag=1;
     /* set data points to 0 */
     t->x[8*ci]=0.0;
     t->x[8*ci+1]=0.0;
     t->x[8*ci+2]=0.0;
     t->x[8*ci+3]=0.0;
     t->x[8*ci+4]=0.0;
     t->x[8*ci+5]=0.0;
     t->x[8*ci+6]=0.0;
     t->x[8*ci+7]=0.0;
   } else {
     t->barr[ci].flag=0;
   }
 }

 return NULL;
}

/* update baseline flags, also make data zero if flagged
  this is needed for solving (calculate error) ignore flagged data */
/* Nbase: total actual data points = Nbasextilesz
   flag: flag array Nbasex1
   barr: baseline array Nbasex1
   x: data Nbase*8 x 1 ( 8 value per baseline ) 
   Nt: no of threads 
*/
int
preset_flags_and_data(int Nbase, double *flag, baseline_t *barr, double *x, int Nt){
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_preflag_t *threaddata;

  /* calculate min baselines a thread can handle */
  Nthb0=(Nbase+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_preflag_t*)malloc((size_t)Nt*sizeof(thread_data_preflag_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating baselines per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<Nbase; nth++) {
    /* this thread will handle baselines [ci:min(Nbase-1,ci+Nthb0-1)] */
    /* determine actual no. of baselines */
    if (ci+Nthb0<Nbase) {
     Nthb=Nthb0;
    } else {
     Nthb=Nbase-ci;
    }
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].startbase=ci;
    threaddata[nth].endbase=ci+Nthb-1;
    threaddata[nth].flag=flag;
    threaddata[nth].barr=barr;
    threaddata[nth].x=x;
    pthread_create(&th_array[nth],&attr,preflag_threadfn,(void*)(&threaddata[nth]));
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


/* worker thread function for type conversion */
static void *
double_to_float_threadfn(void *data) {
 thread_data_typeconv_t *t=(thread_data_typeconv_t*)data;
 
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   t->farr[ci]=(float)t->darr[ci];
 }
 return NULL;
}
static void *
float_to_double_threadfn(void *data) {
 thread_data_typeconv_t *t=(thread_data_typeconv_t*)data;
 
 int ci;
 for (ci=t->starti; ci<=t->endi; ci++) {
   t->darr[ci]=(double)t->farr[ci];
 }
 return NULL;
}

/* convert types */
/* both arrays size nx1 
*/
int
double_to_float(float *farr, double *darr,int n, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_typeconv_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(n+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_typeconv_t*)malloc((size_t)Nt*sizeof(thread_data_typeconv_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<n; nth++) {
    if (ci+Nthb0<n) {
     Nthb=Nthb0;
    } else {
     Nthb=n-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].farr=farr;
    threaddata[nth].darr=darr;
    pthread_create(&th_array[nth],&attr,double_to_float_threadfn,(void*)(&threaddata[nth]));
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

/* convert types */
/* both arrays size nx1 
*/
int
float_to_double(double *darr, float *farr,int n, int Nt) {

  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_typeconv_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(n+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_typeconv_t*)malloc((size_t)Nt*sizeof(thread_data_typeconv_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<n; nth++) {
    if (ci+Nthb0<n) {
     Nthb=Nthb0;
    } else {
     Nthb=n-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].farr=farr;
    threaddata[nth].darr=darr;
    pthread_create(&th_array[nth],&attr,float_to_double_threadfn,(void*)(&threaddata[nth]));
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


static void *
baselinegen_threadfn(void *data) {
 thread_data_baselinegen_t *t=(thread_data_baselinegen_t*)data;
 
 int ci,cj,sta1,sta2;
 for (ci=t->starti; ci<=t->endi; ci++) {
   sta1=0; sta2=sta1+1;
   for (cj=0; cj<t->Nbase; cj++) {
    t->barr[ci*t->Nbase+cj].sta1=sta1;
    t->barr[ci*t->Nbase+cj].sta2=sta2;
    if(sta2<(t->N-1)) {
     sta2=sta2+1;
    } else {
      if (sta1<(t->N-2)) {
      sta1=sta1+1;
      sta2=sta1+1;
      } else {
       sta1=0;
       sta2=sta1+1;
      }
    }
   }
 }
 return NULL;
}

/* generte baselines -> sta1,sta2 pairs for later use */
/* barr: Nbasextilesz
   N : stations
   Nt : threads 
*/
int
generate_baselines(int Nbase, int tilesz, int N, baseline_t *barr,int Nt) {
  int nth,nth1,ci;
  int Nthb0,Nthb;
  pthread_attr_t attr;
  pthread_t *th_array;
  thread_data_baselinegen_t *threaddata;

  /* calculate min values a thread can handle */
  Nthb0=(tilesz+Nt-1)/Nt;

  /* setup threads */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

  if ((th_array=(pthread_t*)malloc((size_t)Nt*sizeof(pthread_t)))==0) {
   fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
   exit(1);
  }
  if ((threaddata=(thread_data_baselinegen_t*)malloc((size_t)Nt*sizeof(thread_data_baselinegen_t)))==0) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  /* iterate over threads, allocating indices per thread */
  ci=0;
  for (nth=0;  nth<Nt && ci<tilesz; nth++) {
    if (ci+Nthb0<tilesz) {
     Nthb=Nthb0;
    } else {
     Nthb=tilesz-ci;
    }
    threaddata[nth].starti=ci;
    threaddata[nth].endi=ci+Nthb-1;
    threaddata[nth].Nbase=Nbase;
    threaddata[nth].N=N;
    threaddata[nth].barr=barr;
    pthread_create(&th_array[nth],&attr,baselinegen_threadfn,(void*)(&threaddata[nth]));
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



