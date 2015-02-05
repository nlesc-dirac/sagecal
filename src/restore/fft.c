/* FST - a Fast Shapelet Transformer
 *
   Copyright (C) 2006-2011 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <complex.h>
#include <fftw3.h>

#include "shapelet.h"

/* y = a.x + y */
void
daxpy(int N, double *x, double a, double *y) {
    int i=1; /* strides */
    extern void daxpy_(int *N, double *alpha, double *x, int *incx, double *y, int *incy);
    daxpy_(&N,&a,x,&i,y,&i);
}


/* evaluate 2D fft using fftw
   data: column major data, fortran array
   Nx, Ny: data dimension Nx x Ny (rows, columns)
   Nud, Nvd: FFT dimension Nud x Nvd  Nud> Nx and Nvd>Ny (proper zero padding)
   freal, fimag: copied values, size Nu x Nv (smaller than Nud x Nvd), C arrays
   In a nutshell:
   Nx x Ny image -> zero pad to get Nud x Nvd image -> FFT to get Nud x Nvd values 
   -> extract inner Nu x Nv value -> copy back to output (columns,rows)
   colmajor: if 1, return the result in column major order, else row major order
 */
int
eval_fft(double *data, int Nx, int Ny, double *freal, double *fimag, int Nud, int Nvd, int Nu, int Nv, int colmajor) {
    int ci,cj,kk;
    int Ncx, Ncy, cu,cv;
    int Ncu, Ncv;
    double rescale; 
    int nthreads=2;
    /* fftw_complex is complex double */
    fftw_complex *in, *out;
    fftw_plan p;
    /* thread fftw */
    fftw_init_threads();
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nud*Nvd);
    if (!in) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    /* initialize the data to zero */
    for (kk=0; kk<Nud*Nvd; kk++) {
      in[kk]=0.0+_Complex_I*0.0;
    }

    /* copy data in row major order, with proper FFT shift */
    /*
       .................      ................
       |               |      |              |
       |  A        B   | to   |   D      C   |
       |               |      |              |
       |  C        D   |      |   B      A   |
       .................      ................

    */
    /* the mapping works for a single dimension:
     Image length Nx: (count from 1), Nc=Nx/2
     Nc                     Nx-1       0                  Nc-1
     |-----------------------|---------|-------------------|
     0                      Nx-Nc     Nud-Nc               Nud-1
     U axis, length Nud: (count from 1)
    */
    Ncx=Nx/2;
    Ncy=Ny/2;

    for (kk=0; kk<Nx*Ny; kk++) {
      /* row */
      ci=kk%Nx; /* map to u */
      if (ci>=0 && ci<=Ncx-1) {
       cu=Nud-Ncx+ci;
      } else if (ci>=Ncx && ci<=Nx-1) {
       cu=ci-Ncx;
      } else {
       cu=-1;
      }
   
      /* column */
      cj=kk/Nx; /* map to v */
      if (cj>=0 && cj<=Ncy-1) {
       cv=Nvd-Ncy+cj;
      } else if (cj>=Ncy && cj<=Ny-1) {
       cv=cj-Ncy;
      } else {
       cv=-1;
      }
      if (cu>=0 && cv>=0)  {
        in[cv*Nud+cu]=data[kk]+_Complex_I*0.0;
        //in[cu*Nvd+cv]=data[kk]+_Complex_I*0.0;
      }
    } 
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nud*Nvd);
    if (!out) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }

    /* thread fftw */
    fftw_plan_with_nthreads(nthreads);
    /* we have row major order, so flip v,u here  */
    p=fftw_plan_dft_2d(Nvd, Nud, in, out,
                               FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */
    
    fftw_destroy_plan(p);


    /* copy back, flip quadrants, cut off tails */
    /* the reverse mapping is identical to the forward mapping, except Nx,Ny are replaced with Nu,Nv */
    /* the mapping works for a single dimension:
     New U axis FFT length Nu: (count from 1), Nc=Nu/2
     Nc                     Nu-1       0                  Nc-1
     |-----------------------|---------|-------------------|
     0                      Nu-Nc     Nud-Nc               Nud-1
     old U axis, length Nud: (count from 1)
    */

    /* determine the amout to cutoff from Nud to get Nu */
    Ncu=Nu/2;
    Ncv=Nv/2;

    printf("FFT UV=[%d,%d], UdVd=[%d,%d] Im=[%d,%d]\n",Nu,Nv,Nud,Nvd,Nx,Ny);
    /* calculate normalizing constant */
    rescale=0.5/sqrt((double)Nud*Nvd);
    for (kk=0; kk<Nud*Nvd; kk++) {
     /* column */
     //cv=kk%Nvd;
      cv=kk/Nud;
      if (cv>=0 && cv<Nv-Ncv) {
       cj=cv+Ncv;
      } else if (cv>=Nvd-Ncv && cv<=Nvd-1) {
       cj=cv-Nvd+Ncv;
      } else {
       cj=-1;
      }

     /* row */
     //cu=kk/Nvd;
     cu=kk%Nud;
      if (cu>=0 && cu<Nu-Ncu) {
       ci=cu+Ncu;
      } else if (cu>=Nud-Ncu && cu<=Nud-1) {
       ci=cu-Nud+Ncu;
      } else {
       ci=-1;
      }

      
      if (ci>=0 && cj>=0)  {
      if (colmajor) {
        freal[cj*Nu+ci]=rescale*creal(out[kk]);
        fimag[cj*Nu+ci]=rescale*cimag(out[kk]); 
       } else {
         freal[ci*Nv+cj]=rescale*creal(out[kk]);
       fimag[ci*Nv+cj]=rescale*cimag(out[kk]); 
       }
      }
    }
   
  
    fftw_free(in); fftw_free(out);
    /* thread fftw */
    fftw_cleanup_threads();

    return 0;
}

/** convolves the mode vectors with the given PSF  (using the FFT)
 *      x,y: arrays of the grid points, sorted!
 *      M: number of modes
 *      Av: array of mode vectors size Nx.Ny times n0.n0, in column major order
 *      n0: number of modes in each dimension
 *      bmaj (radians),bmin (radians),bpa (degrees): Gaussian PSF parameters 
 */
int
convolve_with_psf(double *x, int Nx, double *y, int Ny,
                  double *Av, int n0, double bmaj, double bmin, double bpa) {

    int xci,yci,kk;
    double sc;
    /* fftw_complex is complex double */
    fftw_complex *in, *psfout,*out, *deltaout;
    fftw_plan p,q;
    double lr,mr;
    double *xr,*yr;
    /* take care of asymmetric grid points, make them symmetric */
    if ((xr=(double*)calloc((size_t)(Nx),sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    if ((yr=(double*)calloc((size_t)(Ny),sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }   
    /* middle point of grid */
    kk=Nx/2;
    for (xci=0; xci<Nx; xci++) {
      xr[xci]=x[xci]-x[kk];
    }
    kk=Ny/2;
    for (yci=0; yci<Ny; yci++) {
      yr[yci]=y[yci]-y[kk];
    }

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx*Ny);
    if (!in) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    /* initialize the data to fit the gaussian , with linear transform
      exp(-(lr^2/bmaj^2+mr^2/bmin^2))
    */
    for (yci=0; yci<Ny; yci++) {
     for (xci=0; xci<Nx; xci++) {
      lr=(-xr[xci]*sin(bpa)+yr[yci]*cos(bpa))/bmaj;
      mr=(-xr[xci]*cos(bpa)-yr[yci]*sin(bpa))/bmin;
      in[Nx*yci+xci]=exp(-(lr*lr+mr*mr))+_Complex_I*0.0;
     }
    }
    free(xr);
    free(yr);

    psfout = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    if (!psfout) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }

    /* we have column major order, so flip y,x here  */
    p=fftw_plan_dft_2d(Ny, Nx, in, psfout,
                               FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */
    fftw_destroy_plan(p);

    /* fixing the unknown delay of the PSF w.r.t the image */
    /* calculate the phase gradient of the FFT of a single pixel */
    for (kk=0; kk<Nx*Ny; kk++) {
      in[kk]=0.0+_Complex_I*0.0;
    }
    /* peak of the PSF */ /* FIXME: make sure mpy*Nx+mpx-1 is within array */
    in[(Ny/2)*Nx+(Nx/2)]=1.0+_Complex_I*0.0;
    deltaout= (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    if (!deltaout) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }


    p=fftw_plan_dft_2d(Ny, Nx, in, deltaout,
                               FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p); 
    fftw_destroy_plan(p);

    /* normalize psf max value to 1 */
    sc=0.0;
    for (yci=0; yci<Nx*Ny; yci++) {
      psfout[yci] /=deltaout[yci];
      if (sc<cabs(psfout[yci]))  {
         sc=cabs(psfout[yci]);
      }
    }
    fftw_free(deltaout);
    for (yci=0; yci<Nx*Ny; yci++) {
       psfout[yci]/=sc;
    }

    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    if (!out) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }


    p=fftw_plan_dft_2d(Ny, Nx, in, out,
                               FFTW_FORWARD, FFTW_ESTIMATE);
    q=fftw_plan_dft_2d(Ny, Nx, out, in,
                               FFTW_BACKWARD, FFTW_ESTIMATE);

    /* scale */
    sc=1.0/(double)(Nx*Ny);
    /* for the mode vectors */   
    for (kk=0; kk<n0*n0; kk++) {
#pragma omp parallel for
     for (yci=0; yci<Nx*Ny; yci++) {
       in[yci]=Av[Nx*Ny*kk+yci]+_Complex_I*0.0;
     }
     fftw_execute(p);
     /* multiply with the psf */
#pragma omp parallel for
     for (yci=0; yci<Nx*Ny; yci++) {
       out[yci]*=(psfout[yci]);
     }
     /* take ifft */
     fftw_execute(q);
     /* copy back the result, do proper scale */
#pragma omp parallel for
     for (yci=0; yci<Nx*Ny; yci++) {
       Av[Nx*Ny*kk+yci]=creal(in[yci])*sc;
     }
    }
    fftw_destroy_plan(p);
    fftw_destroy_plan(q);
    fftw_free(in); fftw_free(psfout);
    fftw_free(out);

 return 0;
}

/** convolves the mode vectors with the given PSF fits array (using the FFT)
 *      x,y: arrays of the grid points, sorted!
 *      M: number of modes
 *      Av: array of mode vectors size Nx.Ny times n0.n0, in column major order
 *      n0: number of modes in each dimension
 *      PSF parameters 
 *      psf: array in column major order, size Npx x Npy
 *      constraint Npx<= Nx and Npy <= Ny
 */
int
convolve_with_psf_fits(double *x, int Nx, double *y, int Ny,
                  double *Av, int n0, double *psf, int Npx, int Npy) {

    int xci,yci,kk;
    int Ntx, Nty;
    double sc;
    /* find the peak of psf */
    double psfmax;
    int mpx, mpy;

    /* fftw_complex is complex double */
    fftw_complex *in, *psfout,*out, *deltaout;
    fftw_plan p,q;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx*Ny);
    if (!in) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }


    for (kk=0; kk<Nx*Ny; kk++) {
      in[kk]=0.0+_Complex_I*0.0;
    }
    /*  Fill the Nx x Ny array with the psf Npx x Npy, the missing area 
        is set to zero
    */
    /* first: make sure the pixel sizes are in order */
    if (Npx<=Nx) {
      Ntx=Npx;
    } else {
      Ntx=Nx;
    }
    if (Npy<=Ny) {
      Nty=Npy;
    } else {
      Nty=Ny;
    }

    psfmax=-1e10; /* FIXME: put a negative large value */
    mpx=Ntx/2;
    mpy=Nty/2;
    for (yci=0; yci<Nty; yci++) {
     for (xci=0; xci<Ntx; xci++) {
      if (psfmax< psf[Npx*yci+xci]) {
        psfmax=psf[Npx*yci+xci];
        mpx=xci;
        mpy=yci;
      }
      in[Nx*yci+xci]=psf[Npx*yci+xci]+_Complex_I*0.0;
     }
    }
    psfout = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    if (!psfout) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }



    /* we have column major order, so flip y,x here  */
    p=fftw_plan_dft_2d(Ny, Nx, in, psfout,
                               FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */

    fftw_destroy_plan(p);
    /* fixing the unknown delay of the PSF w.r.t the image */
    /* calculate the phase gradient of the FFT of a single pixel */
    for (kk=0; kk<Nx*Ny; kk++) {
      in[kk]=0.0+_Complex_I*0.0;
    }
    /* peak of the PSF */ /* FIXME: make sure mpy*Nx+mpx-1 is within array */
    /* NOTE: the -1 below is due to original grid being 1 off in l */
    in[mpy*Nx+mpx-1]=1.0+_Complex_I*0.0;
    deltaout= (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    if (!deltaout) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }


    p=fftw_plan_dft_2d(Ny, Nx, in, deltaout,
                               FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p); 
 
    fftw_destroy_plan(p);
    /* normalize psf max value to 1 */
    sc=0.0;
    for (yci=0; yci<Nx*Ny; yci++) {
      psfout[yci] /=deltaout[yci];
      if (sc<cabs(psfout[yci]))  {
         sc=cabs(psfout[yci]);
      }
    }
    fftw_free(deltaout);
    printf("psfmax=%lf scale=%lf\n",psfmax,sc);
    for (yci=0; yci<Nx*Ny; yci++) {
       psfout[yci]/=sc*psfmax; /* FIXME: check this is the right scale */
       //psfout[yci]/=psfmax;
    }

    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nx*Ny);
    if (!out) { 
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }


    p=fftw_plan_dft_2d(Ny, Nx, in, out,
                               FFTW_FORWARD, FFTW_ESTIMATE);
    q=fftw_plan_dft_2d(Ny, Nx, out, in,
                               FFTW_BACKWARD, FFTW_ESTIMATE);

    /* scale */
    sc=1.0/(double)(Nx*Ny);
    /* for the mode vectors */   
    for (kk=0; kk<n0*n0; kk++) {
     for (yci=0; yci<Nx*Ny; yci++) {
       in[yci]=Av[Nx*Ny*kk+yci]+_Complex_I*0.0;
     }
     fftw_execute(p);
     /* multiply with the psf */
     for (yci=0; yci<Nx*Ny; yci++) {
       //out[yci]*=cabs(psfout[yci]);
       out[yci]*=(psfout[yci]);
     }
     /* take ifft */
     fftw_execute(q);
     /* copy back the result, do proper scale */
     for (yci=0; yci<Nx*Ny; yci++) {
       Av[Nx*Ny*kk+yci]=creal(in[yci])*sc;
     }
    }
    fftw_destroy_plan(p);
    fftw_destroy_plan(q);
    fftw_free(in); fftw_free(psfout);
    fftw_free(out);

 return 0;
}
