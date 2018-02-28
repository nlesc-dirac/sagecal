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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>

#include <fitsio.h>
#include <wcs.h>
#include <prj.h>
#include <wcshdr.h>
#include <wcsfix.h>

#include <shapelet.h>
#include "restore.h"

#include <time.h>

//#define DEBUG

/* function to zero the fits file */
int 
zero_image(long totalrows, long offset, long firstrow, long nrows,
   int ncols, iteratorCol *cols, void *user_struct) {

		int ii;

		static float *charp;

    if (firstrow == 1)
    {
       if (ncols != 1)
           return(-1);  /* number of columns incorrect */
       /* assign the input pointers to the appropriate arrays and null ptrs*/
       charp= (float *)  fits_iter_get_array(&cols[0]);
		}

    /*  NOTE: 1st element of array is the null pixel value!  */
    /*  Loop from 1 to nrows, not 0 to nrows - 1.  */
    for (ii = 1; ii <= nrows; ii++) {
			    charp[ii]=0.0; 
    }

		return 0;
}
 



#ifndef TOL
#define TOL 5e-6
#endif
/* calculate l,m using SIN projection, and evaluate Gaussian */
/* ll,mm: pixel l,m (rad)
   ss: source info
   bmaj,bmin,pa : PSF
   freq0: evaluate at this freq
*/
static double
calculate_contribution1(struct wcsprm *wcs, double ll, double mm, sinfo *ss, double bmaj, double bmin, double pa, double freq0) {
  double l,m,lr,mr;
  double stokesI;

  double a,b,A,B,alpha,theta,num,den,X,Y;
  
  int ncoord,*statc,status;
  double *pixelra,*pixeldec,*imgphi,*imgtheta,*imgl,*imgm;
  /* find source l,m coords */
  ncoord=1;
  if ((pixelra=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((pixeldec=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((imgphi=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((imgtheta=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((imgl=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((imgm=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }

  if ((statc=(int*)calloc((size_t)ncoord,sizeof(int)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  pixelra[0]=(double)(ss->ra)*180.0/M_PI;
  pixeldec[0]=(double)(ss->dec)*180.0/M_PI;
  if ((status = cels2x(&wcs->cel, 1,1,1,1, pixelra, pixeldec, imgphi, imgtheta, imgl, imgm, statc))) {
    fprintf(stderr,"%s: %d: wcsp2s ERROR %2d\n",__FILE__,__LINE__,status);
    /* Handle Invalid pixel coordinates. */
    if (status == 6) status = 0;
  }

  l=-(ll-imgl[0]*M_PI/180.0);
  m=mm-imgm[0]*M_PI/180.0;

  free(pixelra);
  free(pixeldec);
  free(imgphi);
  free(imgtheta);
  free(imgl);
  free(imgm);
  free(statc);

  /* rotate l,m by angle pa to get new values */
  double spa,cpa;
  sincos(pa,&spa,&cpa);
  //lr=-l*sin(pa)+m*cos(pa);
  //mr=-l*cos(pa)-m*sin(pa); 
  lr=-l*spa+m*cpa;
  mr=-l*cpa-m*spa; 
  double fratio,fratio1,fratio2;
  if (ss->spec_idx!=0.0) {
   //stokesI=ss->sI*pow(freq0/ss->f0,ss->spec_idx);
   fratio=log(freq0/ss->f0);
   fratio1=fratio*fratio;
   fratio2=fratio1*fratio;
   /* catch -ve and 0 values for sI */
   if (ss->sI==0.0) {
    stokesI=0.0;
   } else if (ss->sI>0.0) {
    stokesI=exp(log(ss->sI)+ss->spec_idx*fratio+ss->spec_idx1*fratio1+ss->spec_idx2*fratio2);
   } else {
    stokesI=-exp(log(-ss->sI)+ss->spec_idx*fratio+ss->spec_idx1*fratio1+ss->spec_idx2*fratio2);
   }
  } else {
   stokesI=ss->sI;
  }

  if (ss->type==STYPE_POINT) {
   /* point source, just convolue with the beam */
   /* gaussian exp(-(lr^2/bmaj^2+mr^2/bmin^2)) */
   l=lr/bmaj;
   m=mr/bmin;
   return stokesI*exp(-(l*l+m*m));
  } else { /* we have an extended source */
   if (ss->type==STYPE_DISK) {
     /* disk source */
     if (sqrt(lr*lr+mr*mr)<=ss->eX) {
       return stokesI;
      } else {
      /* smooth with the beam */
       l=(sqrt(lr*lr+mr*mr)-ss->eX)/bmaj;
       return stokesI*exp(-(l*l));
      }
    } else if (ss->type==STYPE_RING) {
     /* ring source */
      /* smooth with the beam */
      l=(sqrt(lr*lr+mr*mr)-ss->eX)/bmaj;
      return stokesI*exp(-(l*l));
   } else if (ss->type==STYPE_GAUSSIAN) {
    /* Gaussian source */
    // dont add the beam return ss->sI*exp(-(lr*lr/(bmaj*bmaj+ss->eX*ss->eX)+mr*mr/(bmin*bmin+ss->eY*ss->eY)));
   //num=1/2*Y^2*a^2+1/2*B^2*Y^2-1/2*X^2*a^2*cos(2*alpha)+1/2*A^2*Y^2+1/2*b^2*X^2+1/2*b^2*Y^2+1/2*B^2*X^2+1/2*A^2*X^2+1/2*X^2*a^2-X*Y*a^2*sin(2*alpha)+Y*B^2*X*sin(2*theta)-A^2*Y*X*sin(2*theta)+b^2*X*Y*sin(2*alpha)+1/2*b^2*X^2*cos(2*alpha)+1/2*Y^2*a^2*cos(2*alpha)-1/2*b^2*Y^2*cos(2*alpha)+1/2*B^2*X^2*cos(2*theta)-1/2*B^2*Y^2*cos(2*theta)-1/2*A^2*X^2*cos(2*theta)+1/2*A^2*Y^2*cos(2*theta)
   //den=1/2*b^2*B^2+1/2*a^2*B^2+1/2*b^2*A^2+1/2*a^2*A^2+A^2*B^2+a^2*b^2+1/2*b^2*A^2*cos(2*alpha-2*theta)-1/2*b^2*B^2*cos(2*alpha-2*theta)+1/2*a^2*B^2*cos(2*alpha-2*theta)-1/2*a^2*A^2*cos(2*alpha-2*theta)
  // scale=pi*a*b*A*B
  // e^{-num/den}*scale 
    alpha=ss->eP;
    theta=pa;
    A=bmaj; B=bmin;
    a=ss->eX; b=ss->eY; 
    X=l; Y=m;
    num=0.5*Y*Y*a*a+0.5*B*B*Y*Y-0.5*X*X*a*a*cos(2.0*alpha)+0.5*A*A*Y*Y+0.5*b*b*X*X+0.5*b*b*Y*Y+0.5*B*B*X*X+0.5*A*A*X*X+0.5*X*X*a*a-X*Y*a*a*sin(2.0*alpha)+Y*B*B*X*sin(2.0*theta)-A*A*Y*X*sin(2.0*theta)+b*b*X*Y*sin(2.0*alpha)+0.5*b*b*X*X*cos(2.0*alpha)+0.5*Y*Y*a*a*cos(2.0*alpha)-0.5*b*b*Y*Y*cos(2.0*alpha)+0.5*B*B*X*X*cos(2.0*theta)-0.5*B*B*Y*Y*cos(2.0*theta)-0.5*A*A*X*X*cos(2.0*theta)+0.5*A*A*Y*Y*cos(2.0*theta);
    den=0.5*b*b*B*B+0.5*a*a*B*B+0.5*b*b*A*A+0.5*a*a*A*A+A*A*B*B+a*a*b*b+0.5*b*b*A*A*cos(2.0*alpha-2.0*theta)-0.5*b*b*B*B*cos(2.0*alpha-2.0*theta)+0.5*a*a*B*B*cos(2.0*alpha-2.0*theta)-0.5*a*a*A*A*cos(2.0*alpha-2.0*theta);
    /* preseve peak flux, not total, so scale is omitted */
    return stokesI*exp(-num/den);//*M_PI*a*b*A*B/sqrt(den);
    /* do right transformation with PA */
//    l=-lr*cos(ss->eP)-mr*sin(ss->eP);
//    m=+lr*sin(ss->eP)-mr*cos(ss->eP); 
//    return stokesI*exp(-(l*l/(bmaj*bmaj+ss->eX*ss->eX)+m*m/(bmin*bmin+ss->eY*ss->eY)));
   }
  }
  return 0;
}

/* filename: file name
 */
int 
read_fits_file_restore(const char *filename, glist *slist, double bmaj,double bmin, double pa, int add_to_pixel, int beam_given, int format) {
    io_buff fbuff;
    int status;
		int naxis;
		int bitpix;
    long int new_naxis[4];

		int jj,kk,ll;
		int datatype=0;
		long int totalpix;
		double bscale,bzero;
		long int increment[4]={1,1,1,1};
		int null_flag=0;
    sinfo *ss;
    exinfo_shapelet *exs=0;
    double fits_bmaj,fits_bmin,fits_bpa;

    iteratorCol cols[3];  /* structure used by the iterator function */
    int n_cols=1;
    long rows_per_loop=0, offset=0;

		/* stuctures from WCSLIB */
		char *header;
		int ncard,nreject,nwcs;
		//extern const char *wcshdr_errmsg[];
		int ncoord;
		double *pixelc, *imgc, *worldc, *phic, *thetac, *myarr;
		int *statc;

		int stat[NWCSFIX];

    double freq0=1e6; /* reference freq */
    double tempI;
    char ctypeS[16]; /* for handling velocity */

    double pixel_extent,beam_ext;
    double minpix=400.0;

    int fullstokes=1; /* full IQUV image */
		
    /* for shapelet modes */
    double *x,*y,*Av,*z;
    double l0,m0,x0,y0; /* shapelet center l,m and pixels */
    double nn,phi,xi; /* for rotation */
    
    int Nx,Ny,n0_var,M,ii;

    l0=m0=x0=y0=0.0;

    status = 0; 
#ifdef DEBUG
    printf("File =%s\n",filename);
#endif
    fits_open_file(&fbuff.fptr, filename, READWRITE, &status); /* open file */

/* WCSLIB et al. */
		/* read FITS header */
		if ((status = fits_hdr2str(fbuff.fptr, 1, NULL, 0, &header, &ncard, &status))) {
		 fits_report_error(stderr, status);
		 return 1;
		}

    if (!beam_given) {
    /* try to read psf params from file */
    fits_read_key(fbuff.fptr,TDOUBLE,"BMAJ",&fits_bmaj,0,&status);
    /* recover error from missing key */
    if (status) { 
     status=0; 
     fits_bmaj=fits_bmin=-1; /* no key present */ 
    } else {
     fits_read_key(fbuff.fptr,TDOUBLE,"BMIN",&fits_bmin,0,&status);
     if (status) { 
      status=0; 
      fits_bmaj=fits_bmin=-1; /* no key present */ 
     } else {
      fits_read_key(fbuff.fptr,TDOUBLE,"BPA",&fits_bpa,0,&status);
      if (status) { 
       status=0;
       fits_bmaj=fits_bmin=-1; /* no key present */ 
      } else { /* convert to radians */
       fits_bpa=(fits_bpa)/180.0*M_PI; 
       printf("beam= (%lf,%lf,%lf)\n",fits_bmaj,fits_bmin,fits_bpa);
       bmaj=fits_bmaj/360.0*M_PI;
       bmin=fits_bmin/360.0*M_PI;
       pa=fits_bpa;

       /* also read pixel deltas */
       fits_bmaj=fits_bmin=1.0;
       fits_read_key(fbuff.fptr,TDOUBLE,"CDELT1",&fits_bmaj,0,&status);
       if (status) {
        status=0;
       }
       fits_read_key(fbuff.fptr,TDOUBLE,"CDELT2",&fits_bmin,0,&status);
       if (status) {
        status=0;
       }
       fits_bmaj*=M_PI/180.0;
       fits_bmin*=M_PI/180.0;
       /* calculate foot print of psf (pi*a*b) no of pixels*/
       minpix=M_PI*(bmaj)*(bmin)/(fits_bmaj*fits_bmin);
       if (minpix<0) { minpix=-minpix; }
       printf("foot print %lf pix\n",minpix);
     }
    }
   }
   } else { /* beam given, use a default value for footprint */
     minpix=400.0;
   }
   beam_ext=(sqrt(minpix)*4.0);
#ifdef DEBUG
   printf("PSF extent =%lf\n",beam_ext);
#endif

/* try to Parse the primary header of the FITS file. */
    if ((status = wcspih(header, ncard, WCSHDR_all, 2, &nreject, &nwcs, &fbuff.wcs))) {
	      fprintf(stderr, "wcspih ERROR %d, ignoring WCS\n", status);
		}

		/* Fix non-standard WCS keyvalues. */
		if ((status = wcsfix(7, 0, fbuff.wcs, stat))) {
		  printf("wcsfix ERROR, status returns: (");
			  for (ii = 0; ii < NWCSFIX; ii++) {
					printf(ii ? ", %d" : "%d", stat[ii]);
				}
				printf(")\n\n");
	  }

		if ((status = wcsset(fbuff.wcs))) {
		  fprintf(stderr, "wcsset ERROR %d:\n", status);
		  return 1;
		}

#ifdef DEBUG
	  /* Print the struct. */
	  if ((status = wcsprt(fbuff.wcs))) return status;
#endif

		/* turn off scaling so that we copy the pixel values */
		bscale=1.0; bzero=0.0;
    fits_set_bscale(fbuff.fptr,  bscale, bzero, &status);


		fits_get_img_dim(fbuff.fptr, &naxis, &status);
#ifdef DEBUG
		printf("Axis=%d\n",naxis);
#endif
    /* fix zero length axes */
		if (naxis<4) naxis=4;

		if ((fbuff.arr_dims.d=(long int*)calloc((size_t)naxis,sizeof(long int)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			return 1;
		}
		if ((fbuff.arr_dims.lpix=(long int*)calloc((size_t)naxis,sizeof(long int)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			return 1;
		}
		if ((fbuff.arr_dims.hpix=(long int*)calloc((size_t)naxis,sizeof(long int)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			return 1;
		}
		/* get axis dimensions */
		fits_get_img_size(fbuff.fptr, naxis, fbuff.arr_dims.d, &status);
		fbuff.arr_dims.naxis=naxis;
		/* get data type */
		fits_get_img_type(fbuff.fptr, &bitpix, &status);
		if(bitpix==BYTE_IMG) {
#ifdef DEBUG
			printf("Type Bytes\n");
#endif
			datatype=TBYTE;
		}else if(bitpix==SHORT_IMG) {
#ifdef DEBUG
			printf("Type Short Int\n");
#endif
			datatype=TSHORT;
		}else if(bitpix==LONG_IMG) {
#ifdef DEBUG
			printf("Type Long Int\n");
#endif
			datatype=TLONG;
		}else if(bitpix==FLOAT_IMG) {
#ifdef DEBUG
			printf("Type Float\n");
#endif
			datatype=TFLOAT;
		}else if(bitpix==DOUBLE_IMG) {
#ifdef DEBUG
			printf("Type Double\n");
#endif
			datatype=TDOUBLE;
		}
		fbuff.arr_dims.datatype=datatype;

    /* if not adding to image, zero all pixels */
    if (!add_to_pixel) {
    /* define input column structure members for the iterator function */
     fits_iter_set_file(&cols[0], fbuff.fptr);
     fits_iter_set_iotype(&cols[0], InputOutputCol);
     fits_iter_set_datatype(&cols[0], TFLOAT);

     fits_iterate_data(n_cols, cols, offset, rows_per_loop,
                      zero_image, (void*)0, &status);
    }

    /* use WCS to get freq coordinate */
    /* handle velocity values */
    memset((void*)ctypeS,0,16);
    strcpy(ctypeS, "FREQ-???");
    kk = -1;
    if ((status = wcssptr(fbuff.wcs, &kk, ctypeS))) {
        fprintf(stderr,"wcssptr ERROR %d\n", status);
        status=0; 
    }

    /* iterate list I,Q,U,V */
    glist_set_iter_forward(slist);
    ss=(sinfo*)glist_iter_forward(slist);
    while (ss) {
      /* if ss is a shapelet, parse the file here 
         and get modes, beta */

   		ncoord=1; 
  	  if ((pixelc=(double*)calloc((size_t)ncoord*4,sizeof(double)))==0) {
		  	fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
		  	exit(1);
		  }
  	  if ((imgc=(double*)calloc((size_t)ncoord*4,sizeof(double)))==0) {
		  	fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
		  	exit(1);
		  }
  	  if ((worldc=(double*)calloc((size_t)ncoord*4,sizeof(double)))==0) {
		  	fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
		  	exit(1);
		  }
		  if ((phic=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
		  	fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
		  	exit(1);
		  }
		  if ((thetac=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
		  	fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
		  	exit(1);
		  }
		  if ((statc=(int*)calloc((size_t)ncoord,sizeof(int)))==0) {
		  	fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
		  	exit(1);
		  }
     /* if format==FORMAT_LM, convert l,m of source to ra,dec */
     if (format==FORMAT_LM) {
       double imgl,imgm,ra_c,dec_c;
       if ((status = celx2s(&(fbuff.wcs->cel), 1,1,1,1, &ss->ra, &ss->dec, &imgl, &imgm, &ra_c, &dec_c, statc))) {
          fprintf(stderr,"celx2s ERROR %2d\n", status);
          /* Handle Invalid pixel coordinates. */
          if (status == 5) status = 0;
       }
       ss->ra=ra_c*M_PI/180.0;
       ss->dec=dec_c*M_PI/180.0;
     }

      /* first find pixel location for this source */
      worldc[0]=ss->ra/M_PI*180.0;
      worldc[1]=ss->dec/M_PI*180.0;
      worldc[2]=(double)1.0;
      worldc[3]=(double)1.0;

      /* check WCS has same no of axes */
      if (fbuff.wcs->naxis!=naxis) {
        fprintf(stderr,"WCS no of axes %d using %d\n",fbuff.wcs->naxis,naxis);
      }
      if ((status = wcss2p(fbuff.wcs, ncoord, naxis,   worldc, phic, thetac, imgc, pixelc, statc))) {
       fprintf(stderr,"wcss2p ERROR %2d\n", status);
       /* Handle Invalid pixel coordinates. */
       if (status == 8) status = 0;
      }
#ifdef DEBUG
      printf("RADEC (%lf,%lf) Pixel cen (%lf,%lf)\n",ss->ra,ss->dec,pixelc[0],pixelc[1]);
     printf("world %lf %lf\n",worldc[0],worldc[1]);
     printf("pixel %lf %lf\n",pixelc[0],pixelc[1]);
     printf("image %lf %lf\n",imgc[0],imgc[1]);
#endif


      /* calculate extent of this source (in pixels) */
      if (ss->type==STYPE_POINT) {
        pixel_extent=beam_ext;
      } else if (ss->type==STYPE_DISK||ss->type==STYPE_RING) {
        pixel_extent=beam_ext+sqrt(ss->eX*ss->eY/fabs(fits_bmaj*fits_bmin));
      } else if (ss->type==STYPE_GAUSSIAN) {
        pixel_extent=beam_ext+2.0*sqrt(ss->eX*ss->eY/fabs(fits_bmaj*fits_bmin));
      } else if (ss->type==STYPE_SHAPELET) {
        /* get shapelet struct */
        exs=(exinfo_shapelet*)ss->exdata;
        /* shapelet extent: no of modes sqrt */
        pixel_extent=beam_ext+sqrt((double)exs->n0)*(exs->beta*exs->beta)/fabs(fits_bmaj*fits_bmin);
        /* also check if too many pixels are used, truncate it to a safe level */
        if (pixel_extent>1000.0) {
         fprintf(stderr,"Warning: truncating shapelet modes from %lf pixels to 1000\n",pixel_extent);
         pixel_extent=1000.0;
        }
      } else {
        pixel_extent=beam_ext;
      }
      fbuff.arr_dims.lpix[0]=(long int)(pixelc[0]-pixel_extent);
      fbuff.arr_dims.hpix[0]=(long int)(pixelc[0]+pixel_extent);
      fbuff.arr_dims.lpix[1]=(long int)(pixelc[1]-pixel_extent);
      fbuff.arr_dims.hpix[1]=(long int)(pixelc[1]+pixel_extent);
      if (ss->type==STYPE_POINT) {
       /* calculate ra,dec for the nearest pixel for this source */
       pixelc[0]=round(pixelc[0]);
       pixelc[1]=round(pixelc[1]);
       pixelc[2]=1.0;
       pixelc[3]=1.0;
  		 if ((status = wcsp2s(fbuff.wcs, ncoord, naxis, pixelc, imgc, phic, thetac,
			 worldc, statc))) {
		 	 fprintf(stderr,"%s: %d: wcsp2s ERROR %2d\n",__FILE__,__LINE__,status);
			 /* Handle Invalid pixel coordinates. */
			 if (status == 8) status = 0;
  	   }

       /* find freq coordinate (will be done again later) */
       freq0=worldc[2];
       if (freq0<=2.0) freq0=worldc[3];


       /* scale up peak flux such that nearest pixel center 
         has the peak value, not the centroid given by ra,dec */
       tempI=calculate_contribution1(fbuff.wcs,imgc[0]*M_PI/180.0, imgc[1]*M_PI/180.0, ss, bmaj,bmin,pa, freq0);
       double fratio,fratio1,fratio2;
       if (ss->spec_idx!=0.0) {
        fratio=log(freq0/ss->f0);
        fratio1=fratio*fratio;
        fratio2=fratio1*fratio;
        /* catch -ve and 0 values for sI */
        if (ss->sI==0.0) {
         ss->Iscale=0.0;
        } else if (ss->sI>0.0) {
         ss->Iscale=exp(log(ss->sI)+ss->spec_idx*fratio+ss->spec_idx1*fratio1+ss->spec_idx2*fratio2)/tempI;
        } else {
         ss->Iscale=-exp(log(-ss->sI)+ss->spec_idx*fratio+ss->spec_idx1*fratio1+ss->spec_idx2*fratio2)/tempI;
        }
       } else {
        if (tempI!=0.0) {
         ss->Iscale=fabs(ss->sI/tempI);
        } else {
         ss->Iscale=0.0;
        }
       }
      } else {
       ss->Iscale=1.0;
       x0=pixelc[0];
       y0=pixelc[1];
       l0=imgc[0]*M_PI/180.0; 
       m0=imgc[1]*M_PI/180.0;
      }

#ifdef DEBUG
     printf("temp=%lf (%lf %lf %lf %lf %lf)\n",tempI,ss->sI,ss->spec_idx,ss->ra,ss->dec,ss->Iscale);
     printf("Original pixel range [%ld,%ld] to [%ld,%ld]\n",fbuff.arr_dims.lpix[0],fbuff.arr_dims.lpix[1],fbuff.arr_dims.hpix[0],fbuff.arr_dims.hpix[1]);
#endif
      free(pixelc);
      free(imgc);
      free(worldc);
      free(phic);
      free(thetac);
      free(statc);


     fbuff.arr_dims.lpix[2]=fbuff.arr_dims.lpix[3]=1;
     /* sanity check for pixel boundaries */
     if (fbuff.arr_dims.lpix[0]<1) fbuff.arr_dims.lpix[0]=1;
     if (fbuff.arr_dims.lpix[0]>fbuff.arr_dims.d[0]) fbuff.arr_dims.lpix[0]=fbuff.arr_dims.d[0];
     if (fbuff.arr_dims.hpix[0]<1) fbuff.arr_dims.hpix[0]=1;
     if (fbuff.arr_dims.hpix[0]>fbuff.arr_dims.d[0]) fbuff.arr_dims.hpix[0]=fbuff.arr_dims.d[0];
     if (fbuff.arr_dims.lpix[1]<1) fbuff.arr_dims.lpix[1]=1;
     if (fbuff.arr_dims.lpix[1]>fbuff.arr_dims.d[1]) fbuff.arr_dims.lpix[1]=fbuff.arr_dims.d[1];
     if (fbuff.arr_dims.hpix[1]<1) fbuff.arr_dims.hpix[1]=1;
     if (fbuff.arr_dims.hpix[1]>fbuff.arr_dims.d[1]) fbuff.arr_dims.hpix[1]=fbuff.arr_dims.d[1];

#ifdef DEBUG
     printf("Pixel range [%ld,%ld] to [%ld,%ld]\n",fbuff.arr_dims.lpix[0],fbuff.arr_dims.lpix[1],fbuff.arr_dims.hpix[0],fbuff.arr_dims.hpix[1]);
#endif
     if (fbuff.arr_dims.d[2]>0) {
      fbuff.arr_dims.hpix[2]=fbuff.arr_dims.d[2]; /* freq axes */
     } else {
      fbuff.arr_dims.hpix[2]=1; /* freq axes */
     }

     if (fbuff.arr_dims.d[3]>0) {
      fbuff.arr_dims.hpix[3]=fbuff.arr_dims.d[3];
     } else {
      fbuff.arr_dims.hpix[3]=1;
     }

	  /******* create new array **********/	
		new_naxis[0]=fbuff.arr_dims.hpix[0]-fbuff.arr_dims.lpix[0]+1;
		new_naxis[1]=fbuff.arr_dims.hpix[1]-fbuff.arr_dims.lpix[1]+1;
		new_naxis[2]=fbuff.arr_dims.hpix[2]-fbuff.arr_dims.lpix[2]+1;
		new_naxis[3]=fbuff.arr_dims.hpix[3]-fbuff.arr_dims.lpix[3]+1;
    if (new_naxis[3]!=4) {
     fullstokes=0; /* only work with I */
    } 
		/* calculate total number of pixels */
    totalpix=((fbuff.arr_dims.hpix[0]-fbuff.arr_dims.lpix[0]+1)
     *(fbuff.arr_dims.hpix[1]-fbuff.arr_dims.lpix[1]+1)
     *(fbuff.arr_dims.hpix[2]-fbuff.arr_dims.lpix[2]+1)
     *(fbuff.arr_dims.hpix[3]-fbuff.arr_dims.lpix[3]+1));

#ifdef DEBUG
		printf("selecting %ld pixels\n",totalpix);
#endif
		
		if ((myarr=(double*)calloc((size_t)totalpix,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}

		/* allocate memory for pixel/world coordinate arrays */
		ncoord=new_naxis[0]*new_naxis[1]*1*1; /* consider only one plane fron freq, and stokes axes because RA,Dec will not change */
  	if ((pixelc=(double*)calloc((size_t)ncoord*4,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
  	if ((imgc=(double*)calloc((size_t)ncoord*4,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
  	if ((worldc=(double*)calloc((size_t)ncoord*4,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
		if ((phic=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
		if ((thetac=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
		if ((statc=(int*)calloc((size_t)ncoord,sizeof(int)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}

		/* fill up the pixel coordinate array */
    kk=0;
    for (ii=fbuff.arr_dims.lpix[0];ii<=fbuff.arr_dims.hpix[0];ii++)
     for (jj=fbuff.arr_dims.lpix[1];jj<=fbuff.arr_dims.hpix[1];jj++) {
						 pixelc[kk+0]=(double)ii;
						 pixelc[kk+1]=(double)jj;
						 pixelc[kk+2]=(double)1.0;
						 pixelc[kk+3]=(double)1.0;
						 kk+=4;
		 }
#ifdef DEBUG
		/* now kk has passed the last pixel */
		printf("total %d, created %d\n",ncoord,kk);
#endif
    if (fbuff.wcs->naxis!=naxis) {
        fprintf(stderr,"WCS no of axes %d using %d\n",fbuff.wcs->naxis,naxis);
    }

		if ((status = wcsp2s(fbuff.wcs, ncoord, naxis, pixelc, imgc, phic, thetac,
			 worldc, statc))) {
			 fprintf(stderr,"%s: %d: wcsp2s ERROR %2d\n",__FILE__,__LINE__,status);
			 /* Handle Invalid pixel coordinates. */
			 if (status == 8) status = 0;
	  }

    /* find freq coordinate */
    freq0=worldc[2];
    if (freq0<=2.0) freq0=worldc[3];

#ifdef DEBUG
      printf("freq=%lf\n",freq0);
#endif


    fits_read_subset(fbuff.fptr, TDOUBLE, fbuff.arr_dims.lpix, fbuff.arr_dims.hpix, increment,
                   0, myarr, &null_flag, &status);


    /* calculate contribution for shapelets in one call for whole myarr */
    if (ss->type!=STYPE_SHAPELET) {
    ll=kk=0;
    for (ii=1;ii<=new_naxis[0];ii++)
     for (jj=1;jj<=new_naxis[1];jj++) {
       if ( !statc[kk/4] ) {
        /* for pixel (i,j,k,l) in column major it should be 
          (l-1)*Ax3*Ax2*Ax1+(k-1)*Ax2*Ax1+(j-1)*Ax1+i-1 */
           ll=(jj-1)*new_naxis[0]+ii-1;
           //tempI=calculate_contribution(worldc[kk]*M_PI/180.0, worldc[kk+1]*M_PI/180.0, ss, bmaj,bmin,pa, freq0);
           tempI=calculate_contribution1(fbuff.wcs,imgc[kk]*M_PI/180.0, imgc[kk+1]*M_PI/180.0, ss, bmaj,bmin,pa, freq0);
           tempI *=ss->Iscale;
           if (add_to_pixel<0) { 
             tempI=-tempI;
           }
           myarr[ll]+=tempI;
           /* Q, U, V */
           if (fullstokes) {
            myarr[ll+new_naxis[0]*new_naxis[1]]+=ss->sQ*tempI/ss->sI;
            myarr[ll+2*new_naxis[0]*new_naxis[1]]+=ss->sU*tempI/ss->sI;
            myarr[ll+3*new_naxis[0]*new_naxis[1]]+=ss->sV*tempI/ss->sI; 
           }
       } else {
         myarr[ll]=0;
         if (fullstokes) {
          myarr[ll+new_naxis[0]*new_naxis[1]]=0;
          myarr[ll+2*new_naxis[0]*new_naxis[1]]=0;
          myarr[ll+3*new_naxis[0]*new_naxis[1]]=0;
         }
       }
			 kk+=4;
     }
     } else { /* shapelet source */
/*******************************************/
  /* calculate right rotation */
  nn=sqrt(1.0-l0*l0-m0*m0);
#ifdef DEBUG
 printf("(l0,m0,n0)=%lf,%lf,%lf [x,y]=%lf,%lf\n",l0,m0,nn,x0,y0);
#endif

  /* calculate projection from [0,0,1] -> [l,m,n] */
  /* the whole story is:
        [0,0,1]->[l,m,n] with
         l=sin(phi)sin(xi), m=-sin(phi)cos(xi), n=cos(phi) so
         phi=acos(n), xi=atan2(-l,m) and then map
         [u,v,w] ->[ut,vt,wt] with
         |cos(xi)    -cos(phi)sin(xi)     sin(phi)sin(xi)|
         |sin(xi)     cos(phi)cos(xi)     -sin(phi)cos(xi)|
         |0           sin(phi)             cos(phi)       |
  */
  phi=acos(nn);
  xi=atan2(-l0,m0);
  /* transform a=sin(xi), b=cos(xi), theta=phi */
  n0_var=exs->n0;
  /* reset wcs to right pixel center */
  l0=fbuff.wcs->crpix[0];
  m0=fbuff.wcs->crpix[1];
  //fbuff.wcs->crpix[0]=fbuff.arr_dims.d[0]-x0+1;
  fbuff.wcs->crpix[0]=fbuff.arr_dims.hpix[0]-(x0-fbuff.arr_dims.lpix[0]);
  fbuff.wcs->crpix[1]=y0;
  /* rerun wcs with shifted center */
  if ((status = wcsp2s(fbuff.wcs, ncoord, naxis, pixelc, imgc, phic, thetac,
			 worldc, statc))) {
			 fprintf(stderr,"%s: %d: wcsp2s ERROR %2d\n",__FILE__,__LINE__,status);
			 /* Handle Invalid pixel coordinates. */
			 if (status == 8) status = 0;
	}
  /* reset wcs to right pixel center */
  fbuff.wcs->crpix[0]=l0;
  fbuff.wcs->crpix[1]=m0;

  M=exs->n0*exs->n0-1; /* always less than */
  if ((x=(double*)calloc((size_t)new_naxis[0],sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if ((y=(double*)calloc((size_t)new_naxis[1],sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  Nx=new_naxis[0];
  Ny=new_naxis[1];
  for (ii=0;ii<new_naxis[0];ii++) {
     /* reversed axis */
     x[new_naxis[0]-ii-1]=(imgc[ii*4*new_naxis[1]])*M_PI/180.0;
  }
  /* different strides */
  for (ii=0;ii<new_naxis[1];ii++) {
     y[ii]=(imgc[ii*4+1])*M_PI/180.0;
  }
#ifdef DEBUG
  printf("LT phi=%lf xi=%lf\n",phi,xi);
#endif
  //if (nn<0.998 || exs->linear_tf) {
  if (nn<0.999 || (exs->eX!=1.0 && exs->eY!=1.0)) {
   printf("using TF %d %dx%d\n",exs->linear_tf,Nx,Ny);
   calculate_mode_vectors_tf(x, Nx, y, Ny, exs->eX,exs->eY*cos(phi),M_PI-xi,&n0_var,&exs->beta, &Av);
  } else {
   printf("nn=%lf not using TF %dx%d\n",nn,Nx,Ny);
   calculate_mode_vectors(x, Nx, y, Ny, &M, &exs->beta, &Av, &n0_var);
  }
#ifdef DEBUG
  printf("returned modes=%d M=%d\n",n0_var,M);
#endif

  convolve_with_psf(x,Nx,y,Ny,Av,n0_var,bmaj,bmin,pa);

  if ((z=(double*)calloc((size_t)(Nx*(Ny)),sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  for(ii=0;ii<M; ii++) {
    /* y=y+A(:,i)*a[i] */
    daxpy(Nx*(Ny), &(Av[ii*(Nx)*(Ny)]), exs->modes[ii], z);
  }
  if (add_to_pixel<0) { 
   daxpy(Nx*(Ny), z, -ss->sI, myarr);
   if (fullstokes) {
      daxpy(Nx*(Ny), z, -ss->sQ/ss->sI, &myarr[new_naxis[0]*new_naxis[1]]);
      daxpy(Nx*(Ny), z, -ss->sU/ss->sI, &myarr[2*new_naxis[0]*new_naxis[1]]);
      daxpy(Nx*(Ny), z, -ss->sV/ss->sI, &myarr[3*new_naxis[0]*new_naxis[1]]);
   }
  } else {
   daxpy(Nx*(Ny), z, ss->sI, myarr);
   if (fullstokes) {
      daxpy(Nx*(Ny), z, ss->sQ/ss->sI, &myarr[new_naxis[0]*new_naxis[1]]);
      daxpy(Nx*(Ny), z, ss->sU/ss->sI, &myarr[2*new_naxis[0]*new_naxis[1]]);
      daxpy(Nx*(Ny), z, ss->sV/ss->sI, &myarr[3*new_naxis[0]*new_naxis[1]]);
   }
  }
  free(x);
  free(y);
  free(z);
  free(Av);
/*******************************************/
     }
     fits_write_subset(fbuff.fptr, TDOUBLE, fbuff.arr_dims.lpix, fbuff.arr_dims.hpix, myarr, &status);

      free(pixelc);
      free(imgc);
      free(worldc);
      free(phic);
      free(thetac);
      free(statc);
      free(myarr);



    ss=(sinfo*)glist_iter_forward(slist);
    }

    fits_close_file(fbuff.fptr, &status);      /* all done */

    if (status) 
        fits_report_error(stderr, status);  /* print out error messages */

		free(header);
		
		free(fbuff.arr_dims.d);
		free(fbuff.arr_dims.lpix);
		free(fbuff.arr_dims.hpix);
		wcsfree(fbuff.wcs);
    free(fbuff.wcs);

    return(status);
}

void
print_help(void) {
   fprintf(stderr,"Usage:\n");
   fprintf(stderr,"-f infile.fits -i lsm.txt\n");
   fprintf(stderr,"-a : add the value, instead of replacing the value at each pixel\n");
   fprintf(stderr,"-s : subtract the value, instead of replacing the value at each pixel\n");
   fprintf(stderr,"-o : %d: BBS %d: LSM format %d: LSM (3 order spec.idx): default 0\n",FORMAT_BBS,FORMAT_LSM,FORMAT_LSM_SP);
   fprintf(stderr,"-c : filename: cluster file name\n");
   fprintf(stderr,"-l : filename: solutions file name (new format!)\n");
   fprintf(stderr,"-g : filename: station numbers (0,1,..) whose solutions to ignore\n");
   fprintf(stderr,"Note: application of solutions only works for unpolarized sky model\n\n");
  
   fprintf(stderr,"PSF options\n-m : major axis width (arcsec): default 200\n");
   fprintf(stderr,"-n : minor axis width (arcsec): default 200\n");
   fprintf(stderr,"-p : position angle, measured from positive y axis, clockwise (deg): default 0\n");
   fprintf(stderr,"If NO PSF is given, it will be read from the FITS file\n");
   fprintf(stderr,"some extended sources  are also supported\n");
   fprintf(stderr,"Gaussain name G*, Disk name D*, Ring name R*, Shapelet name S*, Default Points\n");
   fprintf(stderr,"Report bugs to Sarod Yatawatta <sarod@users.sf.net>\n");
}

void
print_copyright(void) {
  printf("Restore 0.0.10 (C) 2011-2015 Sarod Yatawatta\n");
}

/* for getopt() */
extern char *optarg;
extern int optind, opterr, optopt;


int main(int argc, char **argv) {
 int c;
 char *ffile, *slistname, *clusterfile, *solfile, *ignfile;
 glist slist;
 ffile=slistname=clusterfile=solfile=ignfile=0;
 double pa=0; /* psf position angle rad */
 double bmaj=0.001; /* psf major axis rad */
 double bmin=0.001; /* psf minor axis rad */

 int outformat=0;
 int add_to_pixel=0;
 int subtract_from_pixel=0;
 int beam_given=0;

 /* for parsing integers */
 int base=10;
 char *endptr;
 print_copyright();
 if (argc==1) {
  print_help();
  return 1;
 }
 while ((c=getopt(argc,argv,"f:i:m:n:p:o:c:l:g:as"))!=-1) {
   switch(c) {
    case 'f':
     if (optarg) {
      ffile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( ffile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(ffile,(char*)optarg);
     }
   break;
   case 'i':
     if (optarg) {
      slistname=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( slistname== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(slistname,(char*)optarg);
     }
   break;
   case 'c':
     if (optarg) {
      clusterfile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( clusterfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(clusterfile,(char*)optarg);
     }
   break;
   case 'l':
     if (optarg) {
      solfile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( solfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(solfile,(char*)optarg);
     }
   break;
   case 'g':
     if (optarg) {
      ignfile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( ignfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(ignfile,(char*)optarg);
     }
   break;
   case 'm':
     if (optarg) {
       bmaj=strtod(optarg,0);
       /* convert arcsec to radians , divide by 2 to get radius*/
       if(!bmaj) { bmaj=0.001; }
       else {
        bmaj=(bmaj/3600.0)/360.0*M_PI; 
       }
       beam_given=1;
     }
   break;
   case 'n':
     if (optarg) {
       bmin=strtod(optarg,0);
       /* convert arcsec to radians , divide by 2 to get radius*/
       if(!bmin) { bmin=0.001; }
       else {
        bmin=(bmin/3600.0)/360.0*M_PI; 
       }
       beam_given=1;
     }
   break;
   case 'p':
     if (optarg) {
       pa=strtod(optarg,0);
       if(!pa) { pa=0.01; }
       else {
       /* convert deg to rad */
        pa=(pa)/180.0*M_PI; 
       }
       beam_given=1;
     }
   break;
   case 'o':
     if (optarg) {
       outformat=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) outformat=0;
     }
    break;
   break;

   case 'a':
     add_to_pixel=1;
   break;
   case 's':
     subtract_from_pixel=1;
   break;
   default:
    print_help();
    break;
  }
 } 

 if (add_to_pixel && subtract_from_pixel) {
  print_help();
  exit(0);
 }
 if (subtract_from_pixel) {
   add_to_pixel=-1;
 }
 
 if (ffile && slistname) {
  if (solfile && clusterfile) {
   read_sky_model_withgain(slistname,&slist,outformat,clusterfile,solfile,ignfile);
  } else {
   read_sky_model(slistname,&slist,outformat);
  }
  printf("read in %d sources\n",slist.count);
  /* read fits file, rewrite pixels */
  read_fits_file_restore(ffile,&slist, bmaj,bmin,pa, add_to_pixel, beam_given,outformat);
  free(ffile);
  free(slistname);
  free(solfile);
  free(clusterfile);
  glist_delete(&slist);
  free(ignfile);
 } else {
  if (ffile) free(ffile);
  if (slistname) free(slistname);
  if (solfile) free(solfile);
  if (clusterfile) free(clusterfile);
  if (ignfile) free(ignfile);
 }
 return 0;
}
