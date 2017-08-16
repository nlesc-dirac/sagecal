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

#ifndef BUILDSKY_H
#define BUILDSKY_H
#define _GNU_SOURCE /* for sincos() */
#include <glib.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>

#include <fitsio.h>
#include <wcs.h>
#include <prj.h>
#include <wcshdr.h>
#include <wcsfix.h>

#ifndef INFINITY_L
#define INFINITY_L 1e6
#endif
#ifndef TOL_L
#define TOL_L 1e-6
#endif
#ifndef MIN
#define MIN(x,y) \
  ((x)<=(y)? (x): (y))
#endif

#ifndef MAX
#define MAX(x,y) \
  ((x)>=(y)? (x): (y))
#endif

/********* constants - from levmar ******************/
#define CLM_INIT_MU       1E-03
#define CLM_STOP_THRESH   1E-17
#define CLM_DIFF_DELTA    1E-06
#define CLM_EPSILON       1E-12
#define CLM_ONE_THIRD     0.3333333334 /* 1.0/3.0 */
#define CLM_OPTS_SZ       5 /* max(4, 5) */
#define CLM_INFO_SZ       10


//#define DEBUG
typedef struct nlimits_ {
    int naxis;
    long int *d;
    long int *lpix;
    long int *hpix;
    double tol;
    int datatype;
} nlims; /* struct to store array dimensions */

typedef struct __io_buff__ {
    fitsfile *fptr;
    nlims arr_dims;
    struct wcsprm *wcs;
} io_buff;


/* data structure for a pixel */
typedef struct hash_pix_ {
  int x,y; /* pixel coords */
  double l,m; /* l,m coords (rad) */
  double ra,dec; /* ra,dec (rad) */ 
  double sI; /* flux */
} hpixel;

/* data structure for a source */
typedef struct ext_src_ {
  double l,m; /* lm of centroid of source */
  double sI; /* flux of source */
} extsrc;


/* data structure for stack: for convex hull */
typedef struct stacknode_
{
  struct stacknode_ *next;  /* next node */
  void *rec;      /* data */
} stack_node;
/* stack for constructing hull */
typedef struct stack_
{
  stack_node *head;
  int count; /* no of elements */
} stack_type;
/* points for hull computation */
typedef struct hpoint_
{
  float x; /* actually l */
  float y; /* actually m */
} hpoint;



/* data structure for a priority queue node for clustering */
typedef struct pq_src_ {
  double rd; /* radius of this cluster */
  extsrc *src; /* pointer to source */
} pqsrc;

/* data structure for a source to cluster the full sky */
typedef struct cl_src_ {
 char *name; /* name printed in the sky model */
 double l,m; /* coords for clustering */
 double sI; /* flux for sorting cluster list */
 double ra,dec; /* coords for annotations, in degrees (not rad) */
} clsrc;

/* struct in hash is GList */
typedef struct pixellist_{
 GList *pix; /* list of pixels type hpixel */
 GList *slist; /* list of extracted sources type extsrc */
 double stI; /* total flux */
 int Nh; /* no of points in the hull */
 hpoint *hull; /* array of (l,m) points in the hull, size Nh+1, first and last point are the same */
} pixellist;

/* data structure for FITS iterator */
typedef struct fiter_struct_ {
  GHashTable *pixtable;
  io_buff *fbuff;
  GList *ignlist; /* list of island numbers to ignore */
} fits_iter_struct;

/* data struct for book keeping various fitting results (we use an array of these) */
typedef struct fit_result_{
 double aic; /* AIC: minimum one is chosen */
 int nsrc; /* no of sources */
 double *ll, *mm, *sI; /* arrays of size nsrcx1 */
} fit_result;

/* data struct for levmar */
typedef struct fit_double_point_data_{
  hpixel *parr;
  double bmaj, bmin, bpa; 
  /* extra parameters, only used in weighted least squares */
  double a3,a4,a5;
} fit_double_point_data;




/**** multiple FITS files ****/
/* data structure for a pixel, with freq dependence */
typedef struct hash_pixf_ {
  int x,y; /* pixel coords */
  double l,m; /* l,m coords (rad) */
  double ra,dec; /* ra,dec (rad) */ 
  int Nf; /* no of freq components */
  double *sI; /* flux Nf x 1 array */
} hpixelf;

/* data structure for a source, with freq dependence */
typedef struct ext_srcf_ {
  double l,m; /* lm of centroid of source */
  double f0; /* reference freq */
  double sI; /* flux of source, at freq f0 */
  double sP,sP1,sP2; /* spectral index sI=exp(log(I_0)+{sP}*log(f/f0)+sP1*log(f/f0)^2+sP2*log(f/f0)^3 */
} extsrcf;

/* data structure for a priority queue node for clustering */
typedef struct pq_srcf_ {
  double rd; /* radius of this cluster */
  extsrcf *src; /* pointer to source */
} pqsrcf;


/* struct in hash is GList, with freq dependence */
typedef struct pixellistf_{
 GList *pix; /* list of pixels type hpixelf */
 GList *slist; /* list of extracted sources type extsrcf */
 double stI; /* total flux */
 int Nh; /* no of points in the hull */
 hpoint *hull; /* array of (l,m) points in the hull, size Nh+1, first and last point are the same */
} pixellistf;

/* data struct for book keeping various fitting results (we use an array of these) */
typedef struct fit_resultf_{
 double aic; /* AIC: minimum one is chosen */
 int nsrc; /* no of sources */
 double f0; /* reference freq */
 double *ll, *mm, *sI, *sP; /* arrays of size nsrcx1, coords, flux, spectral index */
} fit_resultf;

/* data struct for levmar: multiple FITS files */
typedef struct fit_double_point_dataf_{
  hpixelf *parr;
  double *bmaj, *bmin, *bpa, *freqs;  /* arrays size Nfx1 */
  int Nf;
  double *sP; /* not used always, sometimes point to spectral index when its not a parameter */
  double *ll; /* not used always, sometimes point to position */
  double *mm; /* not used always, sometimes point to position */
  double ref_freq,ref_flux; /* reference values for spectral index */
} fit_double_point_dataf;





/******************************** buildsky.c *******************************/
/*** rad to RA and Dec ***/
extern void
rad_to_ra(double ra,int *ra_h,int *ra_m,double *ra_s);

extern void
rad_to_dec(double dec, int *dec_d, int *dec_m, double *dec_s);


/* FITS reading
  imagfile: image file
  maskfile: mask of the image, created by Duchamp 
  pixtable: hash table of blobs in the mask
  bmaj (rad),bmin (rad) ,pa (rad): PSF parameters from the FITS file
  minpix: footprint of psf in pixels
  ignlist: list of islands to ignore (integer list)
*/
extern int 
read_fits_file(const char *imgfile, const char *maskfile, GHashTable **pixtable, double *bmaj, double *bmin, double *pa, int beam_given, double *minpix, GList *ignlist);



/*  calculates the centroids, etc 
  pixtable: hash table of blobs
  bmaj,bmin,bpa: PSF
  minpix: PSF foot print
  threshold: guard pixels will have this amout of the smallest flux
  maxiter: max no of LM iterations
  maxemiter: max no of EM iterations
  use_em: if not zero, will use EM instead of LM
  maxfits: limit max attempted fits to this value, if >0
*/
extern int 
process_pixels(GHashTable *pixtable, double bmaj, double bmin, double bpa, double minpix, double threshold, int maxiter, int maxemiter, int use_em, int maxfits);

/* add guard pixels to the image pixels 
   pixlist: list of pixels
   threshold: guard pixels will have a flux of (threshold)x (minimum flux of pixels)
   parr: final array of pixels, plus guard pixels
   n: no of total pixels in array */
extern int
add_guard_pixels(GList *pixlist, double threshold, hpixel **parr, int *n);



/* try to remove detections of slidelobes etc
   actually not remove anything, but suggest probable sidelobes
   from the pixel list */
/* pixtable: hash table of blobs 
   wcutoff: threshold to detect sidelobes 
*/
extern int
filter_pixels(GHashTable *pixtable, double wcutoff);


/* write output file
   imgfile: FITS image
   pixtable: hash table of source models
   minpix: psf footprint
   bmaj,bmin,bpa: psf
   outformat: 0 BBS 1 LSM
   clusterratio: components closer than clusteratio*(bmaj+bmin)/2 will be merged
   nclusters: no of max clusters to cluster the sky
   unistr: unique string for source names
   scaleflux: if 1, scale model flux to match total flux of island
*/
extern int
write_world_coords(const char *imgfile, GHashTable *pixtable, double minpix, double bmaj, double bmin, double bpa, int outformat, double clusterratio, int nclusters,const char *unistr, int scaleflux);

/******************************** buildmultisky.c ****************************/
/* FITS reading, multiple files
  fitsdir: directory of image files
  maskfile: mask of the image, created by Duchamp 
  pixtable: hash table of blobs in the mask
  Nf: no. of FITS files (frequency components)
  freqs: freq array  Nfx1
  bmaj (rad),bmin (rad) ,pa (rad): PSF parameters from the FITS files: Nfx1 arrays
  minpix: footprint of MEAN psf in pixels
  ignlist: list of islands to ignore (integer list)
  donegative: if >0, fit -ve pixels instead of +ve pixels
*/
extern int 
read_fits_file_f(const char *fitsdir, const char *maskfile, GHashTable **pixtable, int *Nf, double **freqs, double **bmaj, double **bmin, double **pa, int beam_given, double *minpix, GList *ignlist, int donegative);


/*  calculates the centroids, etc 
  pixtable: hash table of blobs
  Nf: no. of FITS files (frequency components)
  freqs: freq array  Nfx1
  bmaj (rad),bmin (rad) ,pa (rad): PSF parameters from the FITS files: Nfx1 arrays
  ref_freq: spectral indices will be calculated at this freq (mean freq)
  minpix: PSF foot print
  threshold: guard pixels will have this amount of the smallest flux
  maxiter: max no of LM iterations
  maxemiter: max no of EM iterations
  use_em: if not zero, will use EM instead of LM
  maxfits: limit max attempted fits to this value, if >0
*/
extern int 
process_pixels_f(GHashTable *pixtable, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double *ref_freq, double minpix, double threshold, int maxiter, int maxemiter, int use_em, int maxfits);


/* add guard pixels to the image pixels 
   pixlist: list of pixels
   Nf: no. of FITS files (frequency components)
   threshold: guard pixels will have a flux of (threshold)x (minimum flux of pixels)
   parr: final array of pixels, plus guard pixels
   n: no of total pixels in array */
extern int
add_guard_pixels_f(GList *pixlist, int Nf, double threshold, hpixelf **parr, int *n);


/* try to remove detections of slidelobes etc
   actually not remove anything, but suggest probable sidelobes
   from the pixel list */
/* pixtable: hash table of blobs 
   wcutoff: threshold to detect sidelobes 
*/
extern int
filter_pixels_f(GHashTable *pixtable, double wcutoff);


/* write output file
   imgfile: FITS image: mask file
   pixtable: hash table of source models
   minpix: psf footprint
   Nf: no. of FITS files (frequency components)
   freqs: freq array  Nfx1 
   bmaj (rad),bmin (rad) ,pa (rad): PSF parameters from the FITS files: Nfx1 arrays
   ref_freq: spectral indices will be calculated at this freq (mean freq)
   outformat: 0 BBS 1 LSM
   clusterratio: components closer than clusteratio*(bmaj+bmin)/2 will be merged
   nclusters: no of max clusters to cluster the sky
   unistr: unique string for source names
   donegative: if >0, print -ve flux instead of +ve flux
   scaleflux: if 1, scale model flux to match total flux of island
*/
extern int
write_world_coords_f(const char *imgfile, GHashTable *pixtable, double minpix, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, int outformat, double clusterratio, int nclusters, const char *unistr,int donegative, int scaleflux);


/******************************** fitpixels.c *******************************/
/* fit (just find centroid and peak flux) for a single component
  parr: array of pixel information (npix x 1)
  npix: array size
  bmaj,bmin,bpa: psf (all radians)
  maxiter: max LM iterations
  ll,mm: centroid positions
  sI: peak flux
*/
extern double 
fit_single_point0(hpixel *parr, int npix, double bmaj, double bmin, double bpa,int maxiter, double *ll, double *mm, double *sI);


/* fit for a single component
  parr: array of pixel information (npix x 1)
  npix: array size
  bmaj,bmin,bpa: psf (all radians)
  maxiter: max LM iterations
  ll,mm: centroid positions
  sI: peak flux
*/
extern double 
fit_single_point(hpixel *parr, int npix, double bmaj, double bmin, double bpa,int maxiter, double *ll, double *mm, double *sI);


/* fit for N components
  parr: array of pixel information (npix x 1)
  npix: array size
  bmaj,bmin,bpa: psf (all radians)
  maxiter: max LM iterations
  ll,mm: centroid positions Nx1
  sI: peak flux Nx1

 N>1 */
extern double
fit_N_point(hpixel *parr, int npix, double bmaj, double bmin, double bpa, int maxiter, double *ll, double *mm, double *sI, int N);

/* fit for N components, using EM
  parr: array of pixel information (npix x 1)
  npix: array size
  bmaj,bmin,bpa: psf (all radians)
  maxiter: max LM iterations
  maxemiter: max no of EM iterations
  ll,mm: centroid positions Nx1
  sI: peak flux Nx1

 N>1 
 hull: Nh x 1 array of hpoints
*/
extern double
fit_N_point_em(hpixel *parr, int npix, double bmaj, double bmin, double bpa, int maxiter, int maxemiter, double *ll, double *mm, double *sI, int N, int Nh, hpoint *hull);




/* print info for levmar return values */
extern void
print_levmar_info(double e_0, double e_final, int itermax, int info, int fnum, int jnum, int lnum);


/****************************** myblas.c ****************************/
/* BLAS wrappers */
/* blas dcopy */
/* y = x */
/* read x values spaced by Nx (so x size> N*Nx) */
/* write to y values spaced by Ny  (so y size > N*Ny) */
extern void
my_dcopy(int N, double *x, int Nx, double *y, int Ny);

/* blas scale */
/* x = a. x */
extern void
my_dscal(int N, double a, double *x);

/* x^T*y */
extern double
my_ddot(int N, double *x, double *y);

/* ||x||_2 */
extern double
my_dnrm2(int N, double *x);

/* sum||x||_1 */
extern double
my_dasum(int N, double *x);

/* BLAS y = a.x + y */
extern void
my_daxpy(int N, double *x, double a, double *y);

/* BLAS y = a.x + y */
extern void
my_daxpys(int N, double *x, int incx, double a, double *y, int incy);

/* max |x|  index (start from 1...)*/
extern int
my_idamax(int N, double *x, int incx);

/* BLAS DGEMM C = alpha*op(A)*op(B)+ beta*C */
extern void
my_dgemm(char transa, char transb, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

/* BLAS DGEMV  y = alpha*op(A)*x+ beta*y : op 'T' or 'N' */
extern void
my_dgemv(char trans, int M, int N, double alpha, double *A, int lda, double *x, int incx,  double beta, double *y, int incy);

/* following routines used in LAPACK solvers */
/* cholesky factorization: real symmetric */
extern int
my_dpotrf(char uplo, int N, double *A, int lda);

/* solve Ax=b using cholesky factorization */
extern int
my_dpotrs(char uplo, int N, int nrhs, double *A, int lda, double *b, int ldb);

/* solve Ax=b using QR factorization */
extern int
my_dgels(char TRANS, int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *WORK, int LWORK);

/* A=U S VT, so V needs NOT to be transposed */
extern int
my_dgesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S,
   double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK);
/*********************************** fitmultipixels.c ************************/
/* fit (just find centroid and peak flux) for a single component
  parr: array of pixel information (npix x 1)
  npix: array size
  Nf: no. of FITS files (frequency components)
  freqs: freq array  Nfx1
  bmaj (rad),bmin (rad) ,pa (rad): PSF parameters from the FITS files: Nfx1 arrays
  maxiter: max LM iterations
  ll,mm: centroid positions
  sI: peak flux
  sP: spectral index
*/
extern double 
fit_single_point0_f(hpixelf *parr, int npix, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa,int maxiter, double *ll, double *mm, double *sI, double *sP);


/* fit for a single component
  parr: array of pixel information (npix x 1)
  npix: array size
  Nf: no. of FITS files (frequency components)
  freqs: freq array  Nfx1
  bmaj (rad),bmin (rad) ,pa (rad): PSF parameters from the FITS files: Nfx1 arrays
  ref_freq: reference freq
  maxiter: max LM iterations
  ll,mm: centroid positions
  sI: peak flux
  sP: spectral index
*/
extern double 
fit_single_point_f(hpixelf *parr, int npix, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, int maxiter, double *ll, double *mm, double *sI, double *sP);


/* fit for N components, using EM
  parr: array of pixel information (npix x 1)
  npix: array size
  Nf: no. of FITS files (frequency components)
  freqs: freq array  Nfx1
  bmaj (rad),bmin (rad) ,pa (rad): PSF parameters from the FITS files: Nfx1 arrays
  ref_freq: reference freq
  maxiter: max LM iterations
  maxemiter: max no of EM iterations
  ll,mm: centroid positions Nx1
  sI: peak flux Nx1
  sP: spectral index 3Nx1 (order 3)

  hull: Nh x 1 array of hpoints
 N>1 */
extern double
fit_N_point_em_f(hpixelf *parr, int npix, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, int maxiter, int maxemiter, double *ll, double *mm, double *sI, double *sP, int N, int Nh, hpoint *hull);


/*********************************** scluster.c *******************************/

/* cluster sources closer than radius r (in lm coords) (rad)
   inlist : input source list
   pixlist : list of pixels
   bmaj,bmin,bpa: PSF
   outlist : output source list (initially empty)
*/
extern int
cluster_sources(double r, GList *inlist, GList *pixlist, double bmaj, double bmin, double bpa, GList **outlist);

/* cluster sources closer than radius r (in lm coords) (rad)
   multi freq version
   inlist : input source list
   pixlist : list of pixels
   freqs: Nfx1 array of frequencies 
   bmaj,bmin,bpa: PSF Nfx1 arrays
   ref_freq: reference freq
   outlist : output source list (initially empty)
*/
extern int
cluster_sources_f(double r, GList *inlist, GList *pixlist, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, GList **outlist);



/*********************************** scluster.c *******************************/

/* cluster whole sky using C-clustering lib
*/
extern int
cluster_sky(const char *imgfile, GList *skylist, int ncluster);


/****************************** clmfit_nocuda.c ****************************/
/* LM with LAPACK */
/** keep interface almost the same as in levmar **/
extern int
clevmar_der_single_nocuda(
  void (*func)(double *p, double *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
  void (*jacf)(double *p, double *j, int m, int n, void *adata),  /* function to evaluate the Jacobian : NULL if not given */
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
  void *adata); /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                * Set to NULL if not needed
                */

/****************************** hull.c ****************************/
/* function to build a convex hull of a given set of pixels */
/* pixset: set of pixels to find the boundary
   pixset->pix: list of pixels
   pixset->hull: array of hull (l,m) pixels
   pixset->Nh: no. of boundary (hull) points
*/
extern int
construct_boundary(pixellist *pixset);
/* function to build a convex hull of a given set of pixels */
/* pixset: set of pixels to find the boundary
   pixset->pix: list of pixels
   pixset->hull: array of hull (l,m) pixels
   pixset->Nh: no. of boundary (hull) points
*/
extern int
construct_boundary_f(pixellistf *pixset);
/* check if point (x,y) is inside hull 
   if true, return 1, else 0 */
/* hull : Nhx1 array of points */
extern int 
inside_hull(int Nh, hpoint *hull, double x, double y);
#endif /* BUILDSKY_H */
