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

#ifndef DIRAC_RADIO_H
#define DIRAC_RADIO_H

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#endif /* HAVE_CUDA */

#ifdef __cplusplus
        extern "C" {
#endif

#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>

#include <pthread.h>

/* for gcc 4.8 and above */
#ifndef complex
#define complex _Complex
#endif
 

#ifdef HAVE_CUDA
#include <Dirac_GPUtune.h>

/* define float constants if not defined */
#ifndef M_PIf
#define M_PIf (float)M_PI
#endif

#ifndef M_PI_2f
#define M_PI_2f (float)M_PI_2
#endif

#ifndef M_PI_4f
#define M_PI_4f (float)M_PI_4
#endif
#endif /* HAVE_CUDA */

/* max source name length, increase it if names get longer */
#define MAX_SNAME 2048

/* source types */
#define STYPE_POINT 0
#define STYPE_GAUSSIAN 1
#define STYPE_DISK 2
#define STYPE_RING 3
#define STYPE_SHAPELET 4

/* simulation options */
#define SIMUL_ONLY 1 /* only predict model */
#define SIMUL_ADD 2 /* add to input */
#define SIMUL_SUB 3 /* subtract from input */

#include <Dirac_common.h>

/****************************** readsky.c ****************************/
/* struct for a cluster GList item */
struct clust_t;

typedef struct clust_n_{
 char *name; /* source name (string)*/
} clust_n;

/* read sky/cluster files, 
   carr:  return array size Mx1 of clusters
   M : no of clusters
   freq0: obs frequency Hz
   ra0,dec0 : ra,dec of phase center (radians)
   format: 0: LSM, 1: LSM with 3 order spec index
   each element has source infor for that cluster */
extern int
read_sky_cluster(const char *skymodel, const char *clusterfile, clus_source_t **carr, int *M, double freq0, double ra0, double dec0,int format);

/* read solution file, only a set of solutions and load to p
  sfp: solution file pointer
  p: solutions vector Mt x 1
  carr: for getting correct offset in p
  N : stations 
  M : clusters
*/
extern int
read_solutions(FILE *sfp,double *p,clus_source_t *carr,int N,int M);

/* set ignlist[ci]=1 if 
  cluster id 'cid' is mentioned in ignfile and carr[ci].id==cid
*/
extern int
update_ignorelist(const char *ignfile, int *ignlist, int M, clus_source_t *carr);

/* read ADMM regularization factor per cluster from text file:
 *
 *
 * format without spatial regularization:
#cluster_id  hybrid_parameter admm_rho
 ...
 ...
 (M values)
 admm_rho : can be 0 to ignore consensus, just normal calibration
# end file

 format with spatial regularization:
#cluster_id  hybrid_parameter admm_rho spatial_alpha
 ...
 ...
 (M values)
# end file

 and store it in array arho : size Mtx1, taking into account the hybrid parameter
 also in array arhoslave : size Mx1, without taking hybrid params into account

 if spatialreg>0, also read spatial regularization factors (read M and store Mt values)
 alpha: Mtx1 spatial regularization values
*/

extern int
read_arho_fromfile(const char *admm_rho_file,int Mt,double *arho, int M, double *arhoslave, int spatialreg, double *alpha);

/****************************** predict.c ****************************/
/************* extended source contributions ************/
extern complex double
gaussian_contrib(void*dd, double u, double v, double w);

extern complex double
ring_contrib(void*dd, double u, double v, double w);

extern complex double
disk_contrib(void*dd, double u, double v, double w);


/* time smearing TMS eq. 6.80 for EW-array formula 
  note u,v,w: meter/c so multiply by freq. to get wavelength 
  ll,mm: source
  dec0: phase center declination
  tdelta: integration time */
extern double
time_smear(double ll,double mm,double dec0,double tdelta,double u,double v,double w,double freq0);

/* predict visibilities
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: data to write size Nbase*8*tileze x 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  N: no of stations
  Nbase: no of baselines
  tilesz: tile size
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  Nt: no of threads
*/
extern int
predict_visibilities(double *u, double *v, double *w, double *x, int N,
   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, int Nt);

/* precalculate cluster coherencies
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: coherencies size Nbase*4*Mx 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  N: no of stations
  Nbase: no of baselines (including more than one tile)
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  uvmin: baseline length sqrt(u^2+v^2) below which not to include in solution
  uvmax: baseline length higher than this not included in solution
  Nt: no of threads

  NOTE: prediction is done for all baselines, even flagged ones
  and flags are set to 2 for baselines lower than uvcut
*/
extern int
precalculate_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int Nt);

/* multi-freq version of precalculate_coherencies() */
/*
  x: coherencies size Nbase*tilesz*4*M*Nchan x 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
   same order repeated per each channel
  freqs: Nchanx1 array of frequencies
  Nbase: is actually Nbase*tilesz of original problem
*/
extern int
precalculate_coherencies_multifreq(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double *freqs, int Nchan, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int Nt);


/****************************** diffuse_predict.c ****************************/
/* have_cuda: if 1, use GPU version, else only CPU version */
extern int
recalculate_diffuse_coherencies(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int diffuse_cluster, int sh_n0, double sh_beta, complex double *Z, int Nt, int use_cuda);
/****************************** transforms.c ****************************/
#ifndef ASEC2RAD
#define ASEC2RAD 4.848136811095359935899141e-6
#endif

/* 
 convert xyz ITRF 2000 coords (m) to
 long,lat, (rad) height (m)
 References:
*/
extern int
xyz2llh(double *x, double *y, double *z, double *longitude, double *latitude, double *height, int N);

/* convert ra,dec to az,el
   ra,dec: radians
   longitude,latitude: rad,rad 
   time_jd: JD days

   az,el: output  rad,rad

References: Darin C. Koblick MATLAB code, based on
  % Fundamentals of Astrodynamics and Applications 
 % D. Vallado, Second Edition
 % Example 3-5. Finding Local Siderial Time (pg. 192) 
 % Algorithm 28: AzElToRaDec (pg. 259)
*/
extern int
radec2azel(double ra, double dec, double longitude, double latitude, double time_jd, double *az, double *el);

/* convert time to Greenwitch Mean Sideral Angle (deg)
   time_jd : JD days
   thetaGMST : GMST angle (deg)
*/
extern int
jd2gmst(double time_jd, double *thetaGMST);


/* convert ra,dec to az,el
   ra,dec: radians
   longitude,latitude: rad,rad 
   thetaGMST : GMST angle (deg)

   az,el: output  rad,rad

*/
extern int
radec2azel_gmst(double ra, double dec, double longitude, double latitude, double thetaGMST, double *az, double *el);

/* given the epoch jd_tdb2, 
 calculate rotation matrix params needed to precess from J2000 
   NOVAS (Naval Observatory Vector Astronomy Software)
   PURPOSE:
      Precesses equatorial rectangular coordinates from one epoch to
      another.  One of the two epochs must be J2000.0.  The coordinates
      are referred to the mean dynamical equator and equinox of the two
      respective epochs.

   REFERENCES:
      Explanatory Supplement To The Astronomical Almanac, pp. 103-104.
      Capitaine, N. et al. (2003), Astronomy And Astrophysics 412,
         pp. 567-586.
      Hilton, J. L. et al. (2006), IAU WG report, Celest. Mech., 94,
         pp. 351-367.

*/
extern int
get_precession_params(double jd_tdb2, double Tr[9]);
/* precess  ra0,dec0 at J2000
   to ra,dec at epoch given by transform Tr
 using NOVAS library */
extern int
precession(double ra0, double dec0, double Tr[9], double *ra, double *dec);

/****************************** stationbeam.c ****************************/
/* 
  ra,dec: source direction (rad)
  int bf_type:  beamformer type STAT_NONE, STAT_SINGLE or STAT_TILE
  b_ra0,b_dec0: tile bem center (rad), only used in STAT_TILE mode
  ra0,dec0: beam center (rad)
  f: frequency (Hz)
  f0: beam forming frequency (Hz)
  
  longitude,latitude : Nx1 array of station positions (rad,rad)
  time_jd: JD (day) time
  Nelem : Nx1 array, no. of elements used in each station
  x,y,z: Nx1 pointer arrays to station positions, each station has Nelem[]x1 arrays

  beamgain: Nx1 array of station beam gain along the source direction

wideband: if 0, use f0 as beamformer freq, elese, use f as beamformer freq (for wideband data), also the element coeffients are calculated for each f, not only for f0
*/ 
extern int
arraybeam(double ra, double dec, int bf_type, double b_ra0, double b_dec0, double ra0, double dec0, double f, double f0, int N, double *longitude, double *latitude, double time_jd, int *Nelem, double **x, double **y, double **z, double *beamgain, int wideband);


/*
  ecoeff: elementcoeff struct of element beam coefficients
  elementgain: 8Nx1 array of element beam EJones along the source direction

findex: in wideband mode, the index of f needed to calculate the offset of element coefficients
  */
extern int
array_element_beam(double ra, double dec, int bf_type, double b_ra0, double b_dec0, double ra0, double dec0, double f, double f0, int N, double *longitude, double *latitude, double time_jd, int *Nelem, double **x, double **y, double **z, elementcoeff *ecoeff, double *beamgain, double *elementgain, int wideband, int findex);

extern int
element_beam(double ra, double dec, double f, double f0, int N, double *longitude, double *latitude, double time_jd, elementcoeff *ecoeff, double *elementgain, int wideband, int findex);
/****************************** elementbeam.c ************************************/
typedef struct elementval_{
  complex double phi, theta; /* tuple for element beam Ejones */
} elementval;

/* get beam type LBA/HBA and frequency
   return beam pattern coeff vectors for theta/phi patterns
element_type: ELEM_HBA, ELEM_LBA, ELEM_ALO, ... */
extern int
set_elementcoeffs(int element_type,  double frequency, elementcoeff *ecoeff);

/* get beam type LBA/HBA and for each frequency in frequencies array
   return beam pattern coeff vectors for theta/phi patterns
   frequencies: in Hz, Nf x 1 array
*/
extern int
set_elementcoeffs_wb(int element_type,  double *frequencies, int Nf,  elementcoeff *ecoeff);

/* free storage */
extern int
free_elementcoeffs(elementcoeff ecoeff);

/* calculate elementbeam values for given r,theta coordinates */
extern elementval
eval_elementcoeffs(double r, double theta, elementcoeff *ecoeff);

/* with wideband model, use findex to offset coefficients to match the freq */
extern elementval
eval_elementcoeffs_wb(double r, double theta, elementcoeff *ecoeff, int findex);

/* spherical harmonic basis functions
 * n0: max modes, starts from 1,2,...
 l=0,1,2,....,n0-1 : total n0
 m=(0),(-1,0,1),(-2,-1,0,1,2),....(-l,-l+1,...,l-1,l) : total 2*l+1
 total no of modes=(n0)^2
 * th,ph: array of theta,phi values, both of size Nt (not a uniform grid)
 * range th: 0..pi/2, ph: 0..2*pi
 * output: n0^2 (per each mode) x Nt vector
 */
extern int
sharmonic_modes(int n0,double *th, double *ph, int Nt, complex double *output);


/****************************** cspice_utils.c ************************************/
#ifdef HAVE_CSPICE
extern void
cspice_load_kernels(void);

extern int
cspice_xyz_to_latlon(double x,double y, double z,double *lon, double *lat, double *alt);

/* calculate element beam, same as element_beam(..) but transforms are in lunar frame
 * also a mutex is needed as CSPICE is not thread safe
 */
extern int
cspice_element_beam_lunar(double ra, double dec, double f, double f0, int N, double *longitude, double *latitude, double time_jd, elementcoeff *ecoeff, double *elementgain, int wideband, int findex, pthread_mutex_t *mutex);
#endif /* HAVE_CSPICE */

/****************************** shapelet.c ****************************/
extern complex double
shapelet_contrib(void*dd, double u, double v, double w);
extern int
shapelet_contrib_vector(complex double *modes, int n0, double beta, double u, double v, double w, complex double *coh);

/* shapelet basis (rectangular, real valued),
 * n0: total modes=n0^2
 * beta: scale factor
 * x,y : grid points, total N
 * output: n0^2 (per mode) x N, real/imag same value
 */
extern int
shapelet_modes(int n0,double beta, double *x, double *y, int N, complex double *output);

extern int
shapelet_product_tensor(int L, int M, int N, double alpha, double beta, double gamma,
    double *B);

extern int
shapelet_product(int L, int M, int N, double alpha, double beta, double gamma,
    double *h, double *f, double *g, double *C);

extern int
shapelet_product_jones(int L, int M, int N, double alpha, double beta, double gamma,
    complex double *h, complex double *f, complex double *g, double *C, int hermitian);

extern int
plot_spatial_model(complex double *Zspat, double *B, int Npoly, int N, int G, int Nfreq, int axes_M, int freq, int plot_type, int basis, double beta, const char *filename);
/****************************** predict_withbeam.c ****************************/
/* precalculate cluster coherencies
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  x: coherencies size Nbase*4*Mx 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  N: no of stations
  Nbase: total no of baselines (including more than one tile or timeslot)
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  uvmin: baseline length sqrt(u^2+v^2) below which not to include in solution
  uvmax: baseline length higher than this not included in solution

  Station beam specific parameters:
  bf_type: STAT_NONE, STAT_SINGLE, or STAT_TILE (type of beamformer)
  b_ra0, b_dec0: (if HBA), tile beam pointing  rad,rad
  ph_ra0,ph_dec0: full beam pointing rad,rad
  ph_freq0: beam reference freq
  longitude,latitude: Nx1 arrays (rad,rad) station locations
  time_utc: JD (day) : tilesz x 1 
  tilesz: how many tiles: == unique time_utc
  Nelem: Nx1 array, size of stations (elements)
  xx,yy,zz: Nx1 arrays of station element locations arrays xx[],yy[],zz[]

  ecoeff: struct storing information used to calculate element beam pattern
  doBeam: flag to determine if full (element+array), array only, or element only beam is calculated
  Nt: no of threads

  NOTE: prediction is done for all baselines, even flagged ones
  and flags are set to 2 for baselines lower than uvcut
*/

extern int
precalculate_coherencies_withbeam(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tileze, int *Nelem, double **xx, double **yy, double **zz,
 elementcoeff *ecoeff, int doBeam, int Nt);

/* multi-freq version of precalculate_coherencies_withbeam */
extern int
precalculate_coherencies_multifreq_withbeam(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double *freqs, int Nchan, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tileze, int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int doBeam, int Nt);


extern int
predict_visibilities_multifreq_withbeam(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int doBeam, int Nt, int add_to_data);

/* predict with beam and solutions */
extern int
predict_visibilities_multifreq_withsol_withbeam(double *u,double *v,double *w,double *p,double *x,int *ignorelist,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz,
  elementcoeff *ecoeff, int doBeam,
  int Nt,int add_to_data, int ccid, double rho,int phase_only);

extern int
calculate_residuals_multifreq_withbeam(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int doBeam, int Nt, int ccid, double rho, int phase_only);


/* change epoch of soure ra,dec from J2000 to JAPP */
/* also the beam pointing ra_beam,dec_beam */
extern int
precess_source_locations_deprecated(double jd_tdb, clus_source_t *carr, int M, double *ra_beam, double *dec_beam, int Nt);

/****************************** predict_withbeam_cuda.c ****************************/
#ifdef HAVE_CUDA
/* copy Nx1 double array x to device as float
   first allocate device memory (need to be freed later) */
extern void
dtofcopy(int N, float **x_d, double *x);

/* if dobeam==0, beam calculation is off
   else, flag to determine if full (element+array), array only, or element only beam is calculated
 */
extern int
precalculate_coherencies_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tileze, int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int dobeam, int Nt);

extern int
predict_visibilities_multifreq_withbeam_gpu(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int dobeam, int Nt, int add_to_data);

extern int
calculate_residuals_multifreq_withbeam_gpu(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int dobeam, int Nt, int ccid, double rho, int phase_only);

extern int
predict_visibilities_withsol_withbeam_gpu(double *u,double *v,double *w,double *p,double *x, int *ignorelist, int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int dobeam, int Nt, int add_to_data, int ccid, double rho, int phase_only);

/* note fdelta is bandwidth per channel here, not the full bandwidth */
extern int
precalculate_coherencies_multifreq_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double *freqs,int Nchan, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, 
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc, int tileze, int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int dobeam, int Nt);


#endif /*!HAVE_CUDA */

/****************************** predict_model.cu ****************************/
#ifdef HAVE_CUDA

extern void
cudakernel_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, int *Nelem, float **xx, float **yy, float **zz, float *ra, float *dec, float ph_ra0, float  ph_dec0, float ph_freq0, float *beam, int wideband);

extern void
cudakernel_tile_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, int *Nelem, float **xx, float **yy, float **zz, float *ra, float *dec, float b_ra0, float b_dec0, float ph_ra0, float ph_dec0, float ph_freq0, float *beam, int wideband);

extern void
cudakernel_element_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
 double *time_utc, float *ra, float *dec, int Nmodes, int M, float beta, float *pattern_phi, float *pattern_theta, float *pattern_preamble, float *elementbeam, int wideband);

extern void
cudakernel_coherencies(int B, int N, int T, int K, int F, double *u, double *v, double *w,baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
  unsigned char *stype, double *sI0, double *sQ0, double *sU0, double *sV0, double *f0, double *spec_idx, double *spec_idx1, double *spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh, int dobeam);

extern void
cudakernel_residuals(int B, int N, int T, int K, int F, double *u, double *v, double *w, double *p, int nchunk, baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
  unsigned char *stype, double *sI0, double *sQ0, double *sU0, double *sV0, double *f0, double *spec_idx, double *spec_idx1, double *spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *coh,int dobeam);

void
cudakernel_coherencies_and_residuals(int B, int N, int T, int K, int F, double *u, double *v, double *w, double *p, int nchunk, baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
  unsigned char *stype, double *sI0, double *sQ0, double *sU0, double *sV0, double *f0, double *spec_idx, double *spec_idx1, double *spec_idx2, int **exs, double deltaf, double deltat, double dec0, double *mod, double *coh, int dobeam);

/* B: total baselines
   N: stations
   Nb: baselines worked by this kernel 
   boff: baseline offset 0..B-1
   F: frequencies
   nchunk: how many solutions (hybrid mode)
   x: residual to be corrected: size Nb*8*F
   p: solutions (inverted) size N*8*nchunk
   barr: baseline array to get station indices: size Nb
*/
extern void
cudakernel_correct_residuals(int B, int N, int Nb, int boff, int F, int nchunk, double *x, double *p, baseline_t *barr);

extern void
cudakernel_convert_time(int T, double *time_utc);

extern void
cudakernel_calculate_shapelet_coherencies(float u, float v, float *modes, float *fact, int n0, float beta, double *coh);
#endif /* !HAVE_CUDA */


/****************************** mdl.c ****************************/
/*
  change polynomial order from Kstart to Kfinish
  evaluate Z for each poly order, then find MDL
   N: stations
   M: clusters
   F: frequencies
   J: weightxrhoxJ solutions (note: not true J, but J scaled by each slaves' rho), 8NMxF blocks
   rho: regularization, no weighting applied, Mx1 
   freqs: frequencies, Fx1
   freq0: reference freq
   weight: weight for each freq, based on flagged data, Fx1
   polytype: type of polynomial
  Kstart, Kfinish: range of order of polynomials to calculate the MDL
   Nt: no. of threads
*/
extern int
minimum_description_length(int N, int M, int F, double *J, double *rho, double *freqs, double freq0, double *weight, int polytype, int Kstart, int Kfinish, int Nt);
/****************************** residual.c ****************************/
/* residual calculation, with/without linear interpolation */
/* 
  u,v,w: u,v,w coordinates (wavelengths) size Nbase*tilesz x 1 
  u,v,w are ordered with baselines, timeslots
  p0,p: parameter arrays 8*N*M x1 double values (re,img) for each station/direction
  p0: old value, p new one, interpolate between the two
  x: data to write size Nbase*8*tilesz x 1
   ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots
  input: x is actual data, output: x is the residual
  N: no of stations
  Nbase: no of baselines
  tilesz: tile size
  barr: baseline to station map, size Nbase*tilesz x 1
  carr: sky model/cluster info size Mx1 of clusters
  coh: coherencies size Nbase*tilesz*4*M x 1
  M: no of clusters
  freq0: frequency
  fdelta: bandwidth for freq smearing
  tdelta: integration time for time smearing
  dec0: declination for time smearing
  Nt: no. of threads
  ccid: which cluster to use as correction
  rho: MMSE robust parameter J+rho I inverted

  phase_only: if >0, and if there is any correction done, use only phase of diagonal elements for correction 
*/
extern int
calculate_residuals(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double freq0,double fdelta,double tdelta,double dec0, int Nt, int ccid, double rho);

/* 
  residuals for multiple channels
  data to write size Nbase*8*tilesz*Nchan x 1
  ordered by XX(re,im),XY(re,im),YX(re,im), YY(re,im), baseline, timeslots, channels
  input: x is actual data, output: x is the residual
  freqs: Nchanx1 of frequency values
  fdelta: total bandwidth, so divide by Nchan to get each channel bandwith
  tdelta: integration time for time smearing
  dec0: declination for time smearing
*/
extern int
calculate_residuals_multifreq(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0, int Nt, int ccid, double rho, int phase_only);

/* 
  calculate visibilities for multiple channels, no solutions are used
  note: output column x is set to 0 if add_to_data ==0, else model is added/subtracted (==1 or 2) to data
*/
extern int
predict_visibilities_multifreq(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,int Nt,int add_to_data);


/* predict with solutions in p , ignore clusters flagged in ignorelist (Mx1) array
 also correct final data with solutions for cluster ccid, if valid
*/
extern int
predict_visibilities_multifreq_withsol(double *u,double *v,double *w,double *p,double *x,int *ignorelist,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta,double dec0,int Nt,int add_to_data, int ccid, double rho,int phase_only);

/****************************** diagnostics.c ****************************/
/* Instead of calculating the residuals, calculate influence function
 * and replace the residuals with the influence function,
 * the input arguments are similar to calculate_residuals_multifreq_withbeam_gpu()
 * or calculate_residuals_multifreq_withbeam()
 */
extern int
calculate_diagnostics_gpu(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
 int bf_type, double b_ra0, double b_dec0, double ph_ra0, double ph_dec0, double ph_freq0, double *longitude, double *latitude, double *time_utc,int *Nelem, double **xx, double **yy, double **zz, elementcoeff *ecoeff, int dobeam, int Nt);

/****************************** influence_function.cu ****************************/
#ifdef HAVE_CUDA
/* B: total baselines (baselines x tile size)
   N: stations
   T: tile size (not really needed)
   F: frequencies
   barr: baseline to station map, size B x 1
   p: parameter arrays 8*N*M x1 double values (re,img) for each station/direction
   nchunk: how many solutions (hybrid mode)
   coh: coherencies Bx4 complex
   res: residual Bx4 complex
   hess: hessian (output) 4N x 4N complex, or 4N*4N*2 complex float
*/
extern void
cudakernel_hessian(int B, int N, int T, int F, baseline_t *barr, double *p, int nchunk, float *coh, float *res, float *hess);


#endif /* HAVE_CUDA */

#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* DIRAC_RADIO_H */
