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
#include <dirent.h>

/*** rad to RA and Dec ***/
void
rad_to_ra(double ra,int *ra_h,int *ra_m,double *ra_s) {
 /* convert negative ra to positive by adding 2pi */
 if (ra<0.0) { ra=ra+2.0*M_PI; }
 double w=ra*12.0/M_PI;
 *ra_h=(int)((int)w%24);
 *ra_m=(int)((w-*ra_h)*60.0);
 *ra_s=(w-*ra_h-*ra_m/60.0)*3600.0;
 /* for negative ra,dec, only keep ra_h or dec_d negative */
 if(*ra_m<0) {
  *ra_m=-(*ra_m);
 }
 if(*ra_s<0) {
  *ra_s=-(*ra_s);
 }
}
void
rad_to_dec(double dec, int *dec_d, int *dec_m, double *dec_s) {
  double w=dec*180.0/M_PI;
  *dec_d=(int)((int)w%180);
  *dec_m=(int)((w-*dec_d)*60.0);
  *dec_s=(w-*dec_d-*dec_m/60.0)*3600.0;
   /* for negative ra,dec, only keep ra_h or dec_d negative */
  if (*dec_m<0) {
    *dec_m=-(*dec_m);
  }
  if(*dec_s<0) {
    *dec_s=-(*dec_s);
  }
}

/* key destroy function */
static void  
destroy_key(gpointer data) {
 free((uint32_t*)data);
}
/* value destroy function */
static void  
destroy_value(gpointer data) {
 pixellist *pixl=(pixellist*)data;
 GList *li; 
 for(li=pixl->pix; li!=NULL; li=g_list_next(li)) {
        hpixel *ii= li->data;
        g_free(ii);
 }
 g_list_free(pixl->pix);
 /* free also the source list */
 for(li=pixl->slist; li!=NULL; li=g_list_next(li)) {
        extsrc *ii= li->data;
        g_free(ii);
 }
 g_list_free(pixl->slist);
 free(pixl->hull);
 free(pixl);
}


/* comparison function to find islands to ignore */
static int
compare_two_keys(const void *a, const void *b) {
 uint32_t *akey,*bkey;
 akey=(uint32_t *)a;
 bkey=(uint32_t *)b; 
 if (*akey==*bkey) return 0;
 return 1;
}

/* FITS iterator function */
int
fillup_pixel_hashtable(long totalrows, long offset, long firstrow, long nrows,
   int ncols, iteratorCol *cols, void *user_struct) {

   int ii;
   static long int pt,d1,d2,d3,d4;
   static double *dptr;
   int tmpval;
   GList *ignorethis;

   uint32_t *key;
   hpixel *idx;
   pixellist *x,*a;

   fits_iter_struct *fiter=(fits_iter_struct*)user_struct;
   nlims arr_dims=fiter->fbuff->arr_dims;

   if (firstrow == 1) {
       if (ncols != 1)
           return(-1); 
    /* do any initialization here */
   }

   dptr= (double *)  fits_iter_get_array(&cols[0]);

   for (ii = 1; ii <= nrows; ii++) {
           //printf("arr =%f\n",counts[ii]);
           //counts[ii] = 1.;
           tmpval=(int)dptr[ii];
           if (tmpval> 0) {
             /* calculate 4D coords */
             pt=firstrow+ii-1;
             //printf("coord point=%ld ",pt);
             d4=pt/(arr_dims.d[0]*arr_dims.d[1]*arr_dims.d[2]);
             pt-=(arr_dims.d[0]*arr_dims.d[1]*arr_dims.d[2])*d4;
             d3=pt/(arr_dims.d[0]*arr_dims.d[1]);
             pt-=(arr_dims.d[0]*arr_dims.d[1])*d3;
             d2=pt/(arr_dims.d[0]);
             pt-=(arr_dims.d[0])*d2;
             d1=pt;
             if (d1>0 && d2+1>0){ /* make sure x,y coords are valid */
#ifdef DEBUG
             printf("arr=%d coords =(%ld,%ld,%ld,%ld)\n",tmpval,d1,d2+1,d3,d4);
#endif
             /* find current limit */

          /* insert this coord to the hash table */
         /* insert this pixel to hash table */
         if ((key = (uint32_t *)malloc(sizeof(uint32_t)))==0) {
           fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
          return 1;
         }
         *key = (uint32_t)tmpval;
   //      printf("finding %d in ignore? %d \n",*key,g_list_length(fiter->ignlist));
         /* lookup if this value is in ignore list */
         ignorethis=g_list_find_custom(fiter->ignlist,key,compare_two_keys);
         if (ignorethis==NULL) { /* not found */

           if ((idx= (hpixel *)malloc(sizeof(hpixel)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            return 1;
           }
           idx->x=d1;
           idx->y=d2+1;
           idx->l=0.0;
           idx->m=0.0;
           idx->ra=0.0;
           idx->dec=0.0;
 

           idx->sI=0.0;

           x=(pixellist*)g_hash_table_lookup(fiter->pixtable,key);
           if (!x) { /* new key */
             if((a  = (pixellist *)malloc(sizeof(pixellist)))==0) {
               fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
               return 1;
             }
             a->pix=NULL;
             a->slist=NULL; /* also initialize source list to null, (not being used now) */
             a->hull=NULL; a->Nh=0;
             a->pix=g_list_prepend(a->pix,idx);
             g_hash_table_insert(fiter->pixtable,(gpointer)key,(gpointer)a);
#ifdef DEBUG
             printf("New key: %d -->  %u\n",*key,g_list_length(a->pix));
#endif
           } else { /* key already found */
             x->pix=g_list_prepend(x->pix,idx);
#ifdef DEBUG
             printf("Old key: %d -->  %u\n",*key,g_list_length(x->pix));
#endif
             free(key);
           }
         } else {
           free(key);
         }


           }
          }
    }

  return 0;
}

int 
read_fits_file(const char *imgfile, const char *maskfile, GHashTable **pixtable, double *bmaj, double *bmin, double *pa, int beam_given, double *minpix, GList *ignlist){
 // hash table, list params
 GHashTableIter iter;
 GList *li; 
 pixellist *val;
 uint32_t *key_;
 hpixel *ppix;


 //FITS stuff
 io_buff fbuff;
 int status;
 int naxis;
 int bitpix;
 long int new_naxis[4];


		int ii,jj,kk,ll;
		int datatype=0;
		long int totalpix;
		double bscale,bzero;
		long int increment[4]={1,1,1,1};
		int null_flag=0;
    double fits_bmaj,fits_bmin,fits_bpa;

		/* stuctures from WCSLIB */
		char *header;
		int ncard,nreject,nwcs;
		int ncoord;
		double *pixelc, *imgc, *worldc, *phic, *thetac, *myarr;
		int *statc;

		int stat[NWCSFIX];

    iteratorCol cols[3];  /* structure used by the iterator function */
    int n_cols;
    long rows_per_loop, offset;
    fits_iter_struct fiter;




 status=0;
 fits_open_file(&fbuff.fptr, maskfile, READONLY, &status);


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
       *bmaj=fits_bmaj/360.0*M_PI;
       *bmin=fits_bmin/360.0*M_PI;
       *pa=fits_bpa;
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
       *minpix=M_PI*(*bmaj)*(*bmin)/(fits_bmaj*fits_bmin);
       if (*minpix<0) { *minpix=-*minpix; }
#ifdef DEBUG
       printf("foot print %lf pix\n",*minpix);
#endif

     }
    }
   }
   } else {
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
       *minpix=M_PI*(*bmaj)*(*bmin)/(fits_bmaj*fits_bmin);
       if (*minpix<0) { *minpix=-*minpix; }
#ifdef DEBUG
       printf("foot print %lf pix\n",*minpix);
#endif
   }

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

		/* turn off scaling so that we copy the pixel values */
		bscale=1.0; bzero=0.0;
    fits_set_bscale(fbuff.fptr,  bscale, bzero, &status);


		fits_get_img_dim(fbuff.fptr, &naxis, &status);
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
#ifdef DEBUG
    printf("Image size %ld, %ld, %ld, %ld\n",fbuff.arr_dims.d[0],fbuff.arr_dims.d[1],fbuff.arr_dims.d[2],fbuff.arr_dims.d[3]);
#endif
    /* if dimensions for axes 2,3 are zero, make them 1 */
    if (!fbuff.arr_dims.d[2]) {
       fbuff.arr_dims.d[2]=1;
    }
    if (!fbuff.arr_dims.d[3]) {
       fbuff.arr_dims.d[3]=1;
    }
		fbuff.arr_dims.naxis=naxis;
		/* get data type */
		fits_get_img_type(fbuff.fptr, &bitpix, &status);
		if(bitpix==BYTE_IMG) {
			datatype=TBYTE;
		}else if(bitpix==SHORT_IMG) {
			datatype=TSHORT;
		}else if(bitpix==LONG_IMG) {
			datatype=TLONG;
		}else if(bitpix==FLOAT_IMG) {
			datatype=TFLOAT;
		}else if(bitpix==DOUBLE_IMG) {
			datatype=TDOUBLE;
		}


		fbuff.arr_dims.datatype=datatype;

	 /* only work with stokes I when reading, else read full cube */
     fbuff.arr_dims.lpix[0]=fbuff.arr_dims.lpix[1]=fbuff.arr_dims.lpix[2]=fbuff.arr_dims.lpix[3]=1;
     fbuff.arr_dims.hpix[0]=fbuff.arr_dims.d[0];
     fbuff.arr_dims.hpix[1]=fbuff.arr_dims.d[1];
     fbuff.arr_dims.hpix[2]=1; /* freq axes */
     fbuff.arr_dims.hpix[3]=1;
	  /******* create new array **********/	
		new_naxis[0]=fbuff.arr_dims.hpix[0]-fbuff.arr_dims.lpix[0]+1;
		new_naxis[1]=fbuff.arr_dims.hpix[1]-fbuff.arr_dims.lpix[1]+1;
		new_naxis[2]=fbuff.arr_dims.hpix[2]-fbuff.arr_dims.lpix[2]+1;
		new_naxis[3]=fbuff.arr_dims.hpix[3]-fbuff.arr_dims.lpix[3]+1;
		/* calculate total number of pixels */
    totalpix=((fbuff.arr_dims.hpix[0]-fbuff.arr_dims.lpix[0]+1)
     *(fbuff.arr_dims.hpix[1]-fbuff.arr_dims.lpix[1]+1)
     *(fbuff.arr_dims.hpix[2]-fbuff.arr_dims.lpix[2]+1)
     *(fbuff.arr_dims.hpix[3]-fbuff.arr_dims.lpix[3]+1));

		if ((myarr=(double*)calloc((size_t)totalpix,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			return 1;
		}

    /* define input column structure members for the iterator function */
    fits_iter_set_file(&cols[0], fbuff.fptr);
    fits_iter_set_iotype(&cols[0], InputCol);
    fits_iter_set_datatype(&cols[0], TDOUBLE);
    rows_per_loop = 0;  /* use default optimum number of rows */
    offset = 0;         /* process all the rows */
    n_cols=1;


    /* initialize fits iterator data */
    *pixtable=g_hash_table_new_full(g_int_hash, g_int_equal,destroy_key,destroy_value);
    fiter.pixtable=*pixtable;
    fiter.fbuff=&fbuff;
    fiter.ignlist=ignlist;
    fits_iterate_data(n_cols, cols, offset, rows_per_loop,
                      fillup_pixel_hashtable, (void*)&fiter, &status);


  g_hash_table_iter_init (&iter, *pixtable);
  while (g_hash_table_iter_next (&iter, (gpointer) &key_, (gpointer) &val))
  {

		/* ******************BEGIN create grid for the cells using WCS */
    ncoord=g_list_length(val->pix);
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

    kk=0;
    for(li=val->pix; li!=NULL; li=g_list_next(li)) {
        ppix= li->data;
        ii=ppix->x;
        jj=ppix->y;
				pixelc[kk+0]=(double)ii;
				pixelc[kk+1]=(double)jj;
				pixelc[kk+2]=(double)1.0;
				pixelc[kk+3]=(double)1.0;
				kk+=4;
   }
   /* make sure WCS uses 4 axes as well */
   if (fbuff.wcs->naxis != naxis) {
    fprintf(stderr,"WCS axes =%d but should be %d\n",fbuff.wcs->naxis,naxis);
   }
	 if ((status = wcsp2s(fbuff.wcs, ncoord, naxis, pixelc, imgc, phic, thetac,
			 worldc, statc))) {
			 fprintf(stderr,"wcsp2s ERROR %2d\n", status);
			 /* Handle Invalid pixel coordinates. */
			 if (status == 8) status = 0;
	 }

    kk=0;
    for(li=val->pix; li!=NULL; li=g_list_next(li)) {
        ppix= li->data;
        ppix->l=imgc[kk]*M_PI/180.0;
        ppix->m=imgc[kk+1]*M_PI/180.0;
        ppix->ra=worldc[kk]*M_PI/180.0;
        ppix->dec=worldc[kk+1]*M_PI/180.0;
				kk+=4;
   }
    
		free(pixelc);
		free(imgc);
		free(worldc);
		free(phic);
		free(thetac);
		free(statc);
   }

    fits_close_file(fbuff.fptr, &status);      /* all done */

    if (status) 
        fits_report_error(stderr, status);  /* print out error messages */

		free(header);
		
 fits_open_file(&fbuff.fptr, imgfile, READONLY, &status);
 /*FIXME: could only read the needed pixels later, instead of full image */
 fits_read_subset(fbuff.fptr, TDOUBLE, fbuff.arr_dims.lpix, fbuff.arr_dims.hpix, increment,
    0, myarr, &null_flag, &status);
 /*************************************/
 /* NOTE: pixels here are +1 from the pixels in kvis */
  g_hash_table_iter_init (&iter, *pixtable);
  while (g_hash_table_iter_next (&iter, (gpointer) &key_, (gpointer) &val))
  {
#ifdef DEBUG
     printf("key %u ---> %u: ",(uint32_t)*key_,g_list_length(val->pix));
#endif
     /* calculate total flux */
     val->stI=0;
     for(li=val->pix; li!=NULL; li=g_list_next(li)) {
        ppix= li->data;
        ii=ppix->x;
        jj=ppix->y;
        ll=(jj-1)*new_naxis[0]+ii-1;
        ppix->sI=myarr[ll];
        val->stI+=myarr[ll];
#ifdef DEBUG
        printf("[pixel (%d,%d), lm (%lf,%lf), radec (%lf,%lf), sI %lf]",ppix->x,ppix->y,ppix->l,ppix->m,ppix->ra,ppix->dec,ppix->sI);
#endif
     }
#ifdef DEBUG
     printf("Total flux=%lf\n",val->stI);
     printf("\n");
#endif
      
  }
 /*************************************/
 

 fits_close_file(fbuff.fptr, &status);

 free(myarr);
 free(fbuff.arr_dims.d);
 free(fbuff.arr_dims.lpix);
 free(fbuff.arr_dims.hpix);

 wcsfree(fbuff.wcs);
 free(fbuff.wcs);


 if (status) {
    fits_report_error(stderr, status);
    return 1;
 }

 
  return 0;
}



int 
write_world_coords(const char *imgfile, GHashTable *pixtable, double minpix, double bmaj, double bmin, double bpa, int outformat, double clusterratio, int nclusters,const char *unistr, int scaleflux){
 GHashTableIter iter;
 GList *li,*pli; 
 pixellist *val;
 uint32_t *key_;
 hpixel *ppix;

 //FITS stuff
 io_buff fbuff;
 int status,ii;
  /* stuctures from WCSLIB */
  char *header;
	int ncard,nreject,nwcs;
	int ncoord;
	double *pixell, *pixelm, *imgl, *imgm, *ra_c, *dec_c;
	int *statc;
	int stat[NWCSFIX];

 char *textfile;
 char *regionfile;
 FILE *outf,*regf;
 int count;

 extsrc *srcx;
 double ra_s,dec_s;
 int ra_h,ra_m,dec_d,dec_m;

 double fluxscale,total_model_flux,peak_abs;


 GList *cluslist;
 
 /* list to cluster the full sky */
 GList *skylist=NULL;
 clsrc *csrc;

 status=0;
 fits_open_file(&fbuff.fptr, imgfile, READONLY, &status);

 /* text file name: imgfile+sky.txt */
 textfile=(char*)calloc((size_t)strlen(imgfile)+strlen(".sky.txt")+1,sizeof(char));
 if ( textfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
 }
 strcpy(textfile,imgfile);
 ii=strlen(imgfile);
 strcpy(&textfile[ii],".sky.txt\0");
#ifdef DEBUG
 printf("Text =%s\n",textfile);
#endif

 /* DS9 region file name: imgfile+sky.txt */
 regionfile=(char*)calloc((size_t)strlen(imgfile)+strlen(".ds9.reg")+1,sizeof(char));
 if ( textfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
 }
 strcpy(regionfile,imgfile);
 ii=strlen(imgfile);
 strcpy(&regionfile[ii],".ds9.reg\0");
#ifdef DEBUG
 printf("Region =%s\n",regionfile);
#endif


/* WCSLIB et al. */
		/* read FITS header */
		if ((status = fits_hdr2str(fbuff.fptr, 1, NULL, 0, &header, &ncard, &status))) {
		 fits_report_error(stderr, status);
		 exit(1);
		}

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
		  //return 1; just a warning is enough here
		}

    ncoord=1;
  	if ((pixell=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
  	if ((pixelm=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
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
  	if ((ra_c=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
  	if ((dec_c=(double*)calloc((size_t)ncoord,sizeof(double)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}
  	if ((statc=(int*)calloc((size_t)ncoord,sizeof(int)))==0) {
			fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
			exit(1);
		}


    outf=fopen(textfile,"w+");
    if(!outf) {
    	fprintf(stderr,"%s: %d: unable to open file\n",__FILE__,__LINE__);
			exit(1);
    }
    regf=fopen(regionfile,"w+");
    if(!regf) {
    	fprintf(stderr,"%s: %d: unable to open file\n",__FILE__,__LINE__);
			exit(1);
    }
    if (!outformat) {
     fprintf(outf,"# (Name, Type, Ra, Dec, I, Q, U, V, ReferenceFrequency='60e6',  SpectralIndex='[0.0,0.0,0.0]', MajorAxis, MinorAxis, Orientation) = format\n");
     fprintf(outf,"# The above line defines the field order and is required.\n");
    } else if (outformat==1) {
     fprintf(outf,"## this is an LSM text (hms/dms) file\n");
     fprintf(outf,"##  fields are (where h:m:s is RA, d:m:s is Dec):\n");
     fprintf(outf,"##  name h m s d m s I Q U V spectral_index0 spectral_index1 spectral_index2 RM extent_X(rad) extent_Y(rad) pos_angle(rad) freq0\n");
    } else {
    }
   
    /* DS9 region format */
    fprintf(regf,"# Region file format: DS9 version 4.1\n");
    fprintf(regf,"global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n");
    g_hash_table_iter_init (&iter, pixtable);
    while (g_hash_table_iter_next (&iter, (gpointer) &key_, (gpointer) &val))
    {
#ifdef DEBUG
     printf("key %u ---> %u: ",(uint32_t)*key_,g_list_length(val->slist));
#endif
     fluxscale=1.0;
     total_model_flux=0.0;
     if (g_list_length(val->slist) >1) {
      /* iterate over list once, to find total model flux */
      total_model_flux=0.0;
      for(li=val->slist; li!=NULL; li=g_list_next(li)) {
        srcx= li->data;
        total_model_flux+=srcx->sI;
      }
      //printf("BEAM %lf,%lf\n",bmaj,bmin);
      total_model_flux*=1.1*minpix; /* FIXME: no scale change */
      if (scaleflux) {
       fluxscale=val->stI/(total_model_flux);
      }
     } 
     printf("(%d) Total flux/beam=%lf model=%lf scale=%lf\n",*key_,val->stI/minpix,total_model_flux/minpix,fluxscale);
     /* try to cluster */
     cluslist=NULL;
     if (clusterratio>0.0) {
      /* for domain calculation val->pix, bmaj, bmin, bpa */
      cluster_sources(0.5*(bmaj+bmin)*clusterratio,val->slist,val->pix,bmaj,bmin,bpa,&cluslist);
     }
     count=1;
     if (g_list_length(cluslist)>0) {
      printf("Choosing clustered version ");
      li=cluslist;
      /* if cluster only have 1 source, normalize by peak flux, not the sum */
      if (g_list_length(cluslist)==1 && scaleflux) {
        srcx=li->data;
        /* peak absolute flux */
        peak_abs=-1e6;
        for(pli=val->pix; pli!=NULL; pli=g_list_next(pli)) {
          ppix=pli->data;
          if (peak_abs<fabs(ppix->sI)) {
           peak_abs=fabs(ppix->sI);
          }
        }
        fluxscale=peak_abs/fabs(srcx->sI);
        printf("scale=%lf\n",fluxscale);
      } else {
        printf("\n");
      }
     } else {
      li=val->slist;
     }
     while(li!=NULL) {
        srcx= li->data;
        /**************************************************/
        /* allocate memory for sky cluster source */
        if((csrc= (clsrc*)malloc(sizeof(clsrc)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        /* *key_=one or more digits, count=one or more digits, '\0' end */
        if((csrc->name=(char*)calloc(strlen("PC")+strlen(unistr)+(*key_/10)+1+(count/10)+1+1,sizeof(char)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        sprintf(csrc->name,"P%s%dC%d",unistr,*key_,count);
        csrc->l=srcx->l;
        csrc->m=srcx->m;
        csrc->sI=srcx->sI;
        /**************************************************/

			  pixell[0]=(double)srcx->l*180.0/M_PI;
				pixelm[0]=(double)srcx->m*180.0/M_PI;
		    if ((status = celx2s(&fbuff.wcs->cel, 1,1,1,1, pixell, pixelm, imgl, imgm, ra_c,dec_c, statc))) {
			    fprintf(stderr,"wcsp2s ERROR %2d\n", status);
			    /* Handle Invalid pixel coordinates. */
			    if (status == 6) status = 0;
	      }
#ifdef DEBUG
        printf("Proj (%lf,%lf) to (%lf,%lf) (%lf,%lf)\n",pixell[0],pixelm[0],imgl[0],imgm[0],ra_c[0],dec_c[0]);
#endif
        rad_to_ra(ra_c[0]*M_PI/180.0,&ra_h,&ra_m,&ra_s);
        rad_to_dec(dec_c[0]*M_PI/180.0,&dec_d,&dec_m,&dec_s);

        csrc->ra=ra_c[0];
        csrc->dec=dec_c[0];
        /* insert this sky node to sky cluster list */
        skylist=g_list_prepend(skylist,csrc);

#ifdef DEBUG
       printf("sI=%lf centroid (%d:%d:%4.3lf, %d:%d:%4.3lf)\n",srcx->sI,ra_h,ra_m,ra_s,dec_d,dec_m,dec_s);
#endif
       fprintf(regf,"fk5;ellipse(%lf,%lf,%lf,%lf,%lf) # text={P%s%dC%d}\n",ra_c[0],dec_c[0],bmaj*180.0/M_PI,bmin*180.0/M_PI,bpa*180/M_PI+90.0,unistr,*key_,count);
       if (!outformat) { /* BBS */
         fprintf(outf,"P%s%dC%d, POINT, %d:%d:%4.2lf, +%d.%d.%4.2lf, %lf, 0.0, 0.0, 0.0\n",unistr,*key_,count++,ra_h,ra_m,ra_s,dec_d,dec_m,dec_s,srcx->sI*fluxscale);
       } else if (outformat==1) { /* LSM */
         fprintf(outf,"P%s%dC%d %d %d %4.3lf %d %d %4.3lf %lf 0 0 0 0 0 0 0 0 0 0 1000000.0\n",unistr,*key_,count++,ra_h,ra_m,ra_s,dec_d,dec_m,dec_s,srcx->sI*fluxscale);
      } else { /* position in (l,m)  (rad) instead of ra,dec */
         fprintf(outf,"P%s%dC%d %9.8lf %9.8lf %lf\n",unistr,*key_,count++,srcx->l,srcx->m,srcx->sI*fluxscale);
      }
      li=g_list_next(li);
     }
     /* free cluster list */
     if (g_list_length(cluslist)>0) {
       for(li=cluslist; li!=NULL; li=g_list_next(li)) {
        srcx=li->data;
        g_free(srcx);
      }
     }
     g_list_free(cluslist);

     /* also write the pixel x,y list as a box in the region file */
/*     xmin=ymin=1e6;
     xmax=ymax=-1e6;
     for(li=val->pix; li!=NULL; li=g_list_next(li)) {
       ppix= li->data;
       if (ppix->x>xmax) xmax=ppix->x;
       if (ppix->x<xmin) xmin=ppix->x;
       if (ppix->y>ymax) ymax=ppix->y;
       if (ppix->y<ymin) ymin=ppix->y;
     }
     fprintf(regf,"physical;box %lf %lf %lf %f 0",(double)(xmax+xmin)*0.5, (double)(ymax+ymin)*0.5, (double)(xmax-xmin), (double)(ymax-ymin));
*/
      fprintf(regf,"fk5;polygon(");
      for (ii=0; ii<val->Nh-2; ii++) { 
			 pixell[0]=val->hull[ii].x*180.0/M_PI;
		   pixelm[0]=val->hull[ii].y*180.0/M_PI;
		   if ((status = celx2s(&fbuff.wcs->cel, 1,1,1,1, pixell, pixelm, imgl, imgm, ra_c,dec_c, statc))) {
			    fprintf(stderr,"wcsp2s ERROR %2d\n", status);
			    /* Handle Invalid pixel coordinates. */
			    if (status == 6) status = 0;
	     }
       fprintf(regf,"%lf,%lf,",ra_c[0],dec_c[0]);
     }
     /* last point */
		 pixell[0]=val->hull[ii].x*180.0/M_PI;
		 pixelm[0]=val->hull[ii].y*180.0/M_PI;
		 if ((status = celx2s(&fbuff.wcs->cel, 1,1,1,1, pixell, pixelm, imgl, imgm, ra_c,dec_c, statc))) {
			    fprintf(stderr,"wcsp2s ERROR %2d\n", status);
			    /* Handle Invalid pixel coordinates. */
			    if (status == 6) status = 0;
	   }
     fprintf(regf,"%lf,%lf",ra_c[0],dec_c[0]);

     fprintf(regf,") # color=red text={%u}\n",*key_);
#ifdef DEBUG
     printf("\n");
#endif
    }

    /* cluster sky */
    cluster_sky(imgfile,skylist,nclusters);
    /* free sky cluster list */
    if (g_list_length(skylist)>0) {
       for(li=skylist; li!=NULL; li=g_list_next(li)) {
        csrc=li->data;
        free(csrc->name);
        g_free(csrc);
      }
    }
    g_list_free(skylist);

    fclose(outf);
    fclose(regf);
    fits_close_file(fbuff.fptr, &status);      /* all done */

    if (status) 
        fits_report_error(stderr, status);  /* print out error messages */

		free(header);
		
    free(textfile);
    free(regionfile);
		free(pixell);
		free(pixelm);
		free(imgl);
		free(imgm);
    free(ra_c);
    free(dec_c);
		free(statc);
    wcsfree(fbuff.wcs);
    free(fbuff.wcs);


   if (status) {
    fits_report_error(stderr, status);
    return 1;
   }
   return 0;
}

/* hash table functions */
typedef struct xyhash_ {
 unsigned int x,y;
} xyhash;

static guint
pixel_hash(gconstpointer key) {

 guint k,l;
 xyhash *e=(xyhash*)key;

 k=(int)e->x;
 l=(int)e->y;

 k+=~(k<<15);
 l^=(k>>10);
 k+=(l<<3);
 l^=(k>>6);
 k+=~(l<<11);
 l^=(k>>16);
 k+=l;

 return(k);
}

static gboolean 
pixel_equal(gconstpointer a, gconstpointer b) {
 xyhash *p1=(xyhash*)a;
 xyhash *p2=(xyhash*)b;

 return ((p1->x==p2->x) && (p1->y==p2->y));
}

typedef struct xhash_ {
 int xy;
 double lm;
} xhash;

/* key destroy function */
static void  
destroy_xykey(gpointer data) {
 free((xyhash*)data);
}

static void  
destroy_xyvalue(gpointer data) {
 free((int*)data);
}

static void  
destroy_xkey(gpointer data) {
 free((int*)data);
}

static void  
destroy_xvalue(gpointer data) {
 free((double*)data);
}


/* comparison function for sorting pixels */
static int
pix_compare(const void *a, const void *b) {
 uint32_t *akey,*bkey;
 akey=(uint32_t *)a;
 bkey=(uint32_t *)b;
 if (*akey<*bkey) return -1;
 if (*akey==*bkey) return 0;
 /* a>b */
 return 1;
}


int
add_guard_pixels(GList *pixlist, double threshold, hpixel **parr, int *n) {
 /* we use 3 hash tables: pixels coords are (x,y)
    key: (x,y), data: none
    key: x    data: l
    key: y    data: m
    total pixels will be >= (x pixels) x (y pixels)
*/

  GHashTable *xyreg; /* track x,y */
  GHashTable *xtol, *ytom; /* x to l, y to m (Note: same x (or y) will have different l (or m) values, we take the first one) */


 GList *pixval;
 hpixel *ppix;
 xyhash *xykey;
 uint32_t *xkey,*ykey; 
 double *xp,*xv,*lp,*lv,*mp,*mv;

 uint32_t *xarr,*yarr;
 int ci,cj,ck;
 double deltav;
 int origx, origy;

 double minflux;

  xyreg=g_hash_table_new_full(pixel_hash, pixel_equal,destroy_xykey,destroy_xyvalue);
  xtol=g_hash_table_new_full(g_int_hash, g_int_equal,destroy_xkey,destroy_xvalue);
  ytom=g_hash_table_new_full(g_int_hash, g_int_equal,destroy_xkey,destroy_xvalue);


#ifdef DEBUG
   printf("Orig No of pixels %u\n",g_list_length(pixlist));
#endif
   for(pixval=pixlist; pixval!=NULL; pixval=g_list_next(pixval)) {
        ppix= pixval->data;
#ifdef DEBUG
        printf("[pixel (%d,%d), lm (%lf,%lf), radec (%lf,%lf), sI %lf]",ppix->x,ppix->y,ppix->l,ppix->m,ppix->ra,ppix->dec,ppix->sI);
#endif
    
        /* insert this pixel to hash table */
        if ((xykey = (xyhash*)malloc(sizeof(xyhash)))==0) {
          fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
          exit(1);
        }
        if ((xkey = (uint32_t*)malloc(sizeof(int)))==0) {
          fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
          exit(1);
        }
        if ((ykey = (uint32_t*)malloc(sizeof(int)))==0) {
          fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
          exit(1);
        }



        xykey->x=(uint32_t)ppix->x;
        xykey->y=(uint32_t)ppix->y;
        xp=(double*)g_hash_table_lookup(xyreg,xykey);
        if (!xp) { /* new key */
           if ((xv= (double*)malloc(sizeof(double)))==0) {
             fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
             return 1;
           }
           *xv=1.0;
           g_hash_table_insert(xyreg,(gpointer)xykey,(gpointer)xv);
        } else { /* key already found */
#ifdef DEBUG
           printf("Old key (%d, %d)\n", ppix->x, ppix->y);
#endif
           free(xykey);
        }
        /* insert x,y values in their hash tables too */
        *xkey=ppix->x;
        *ykey=ppix->y;
        lp=(double*)g_hash_table_lookup(xtol,xkey);
        mp=(double*)g_hash_table_lookup(ytom,ykey);
        if (!lp) { /* new key */
           if ((lv= (double*)malloc(sizeof(double)))==0) {
             fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
             return 1;
           }
           *lv=ppix->l;
           g_hash_table_insert(xtol,(gpointer)xkey,(gpointer)lv);
        } else {
           free(xkey);
        }

        if (!mp) { /* new key */
           if ((mv= (double*)malloc(sizeof(double)))==0) {
             fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
             return 1;
           }
           *mv=ppix->m;
           g_hash_table_insert(ytom,(gpointer)ykey,(gpointer)mv);
        } else {
           free(ykey);
        }

   }
  origx=g_hash_table_size(xtol);
  origy=g_hash_table_size(ytom);
#ifdef DEBUG
  printf("XY size %d X size %d Y size %d\n",g_hash_table_size(xyreg),origx,origy);
#endif
 pixval=g_hash_table_get_keys(xtol);
 /* SORT */
  if ((xarr= (uint32_t*)malloc(sizeof(uint32_t)*(origx+2)))==0) {
          fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
          return 1;
  }
  GList *pixval0=pixval;
  ci=1;
#ifdef DEBUG
  printf("Before x:");
#endif
  for (;pixval!=NULL;pixval=g_list_next(pixval)) {
    xkey=(uint32_t*)pixval->data;
    xarr[ci++]=*xkey;
#ifdef DEBUG
   printf("%u ",*xkey);
#endif
  }
  g_list_free(pixval0);
#ifdef DEBUG
  printf("\n");
#endif
  qsort(&xarr[1],origx,sizeof(uint32_t),pix_compare);
  /* add first and last pixel */
  xarr[0]=xarr[1]-1;
  xarr[origx+1]=xarr[origx]+1;
  lp=(double*)g_hash_table_lookup(xtol,&xarr[1]);
  mp=(double*)g_hash_table_lookup(xtol,&xarr[origx]);
  deltav=((*mp-*lp)/(double)origx);
  if ((lv= (double*)malloc(sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *lv=*lp-deltav;
  if ((xkey = (uint32_t*)malloc(sizeof(int)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *xkey=xarr[0];
  g_hash_table_insert(xtol,(gpointer)xkey,(gpointer)lv);
  if ((lv= (double*)malloc(sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *lv=*mp+deltav;
  if ((xkey = (uint32_t*)malloc(sizeof(int)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *xkey=xarr[origx+1];
  g_hash_table_insert(xtol,(gpointer)xkey,(gpointer)lv);



#ifdef DEBUG
  for(ci=0; ci<g_hash_table_size(xtol); ci++) {
   printf("%u ",xarr[ci]);
  }
  printf("\n");
#endif

  pixval=g_hash_table_get_keys(ytom);
  /* SORT */
  if ((yarr= (uint32_t*)malloc(sizeof(uint32_t)*(origy+2)))==0) {
          fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
          exit(1);
  }
  pixval0=pixval;
  ci=1;
#ifdef DEBUG
  printf("Before y:");
#endif
  for (;pixval!=NULL;pixval=g_list_next(pixval)) {
    xkey=(uint32_t*)pixval->data;
    yarr[ci++]=*xkey;
#ifdef DEBUG
   printf("%u ",*xkey);
#endif
  }
  g_list_free(pixval0);
#ifdef DEBUG
  printf("\n");
#endif
  qsort(&yarr[1],origy,sizeof(uint32_t),pix_compare);
  /* add first and last pixel */
  yarr[0]=yarr[1]-1;
  yarr[origy+1]=yarr[origy]+1;
  lp=(double*)g_hash_table_lookup(ytom,&yarr[1]);
  mp=(double*)g_hash_table_lookup(ytom,&yarr[origy]);
  deltav=((*mp-*lp)/(double)origy);
  if ((lv= (double*)malloc(sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *lv=*lp-deltav;
  if ((xkey = (uint32_t*)malloc(sizeof(int)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *xkey=yarr[0];
  g_hash_table_insert(ytom,(gpointer)xkey,(gpointer)lv);
  if ((lv= (double*)malloc(sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *lv=*mp+deltav;
  if ((xkey = (uint32_t*)malloc(sizeof(int)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  *xkey=yarr[origy+1];
  g_hash_table_insert(ytom,(gpointer)xkey,(gpointer)lv);


#ifdef DEBUG
  for(ci=0; ci<g_hash_table_size(ytom); ci++) {
   printf("%u ",yarr[ci]);
  }
  printf("\n");
#endif

  /* now, write back to the array */
  *n=g_hash_table_size(xtol)*g_hash_table_size(ytom);
  if ((*parr= (hpixel *)malloc(sizeof(hpixel)*(*n)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
  }
  minflux=1e6;
  ck=0;
  /* first the original pixels, without any change */
  for(pixval=pixlist; pixval!=NULL; pixval=g_list_next(pixval)) {
        ppix=pixval->data;
#ifdef DEBUG
        printf("[pixel (%d,%d), lm (%lf,%lf), radec (%lf,%lf), sI %lf]",ppix->x,ppix->y,ppix->l,ppix->m,ppix->ra,ppix->dec,ppix->sI);
#endif
        (*parr)[ck].x=ppix->x;
        (*parr)[ck].y=ppix->y;
        (*parr)[ck].l=ppix->l;
        (*parr)[ck].m=ppix->m;
        (*parr)[ck].ra=ppix->ra;
        (*parr)[ck].dec=ppix->dec;
        (*parr)[ck].sI=ppix->sI;
        if (minflux>ppix->sI){ 
         minflux=ppix->sI;
        }
        ck++;
    
  }
#ifdef DEBUG
  printf("\n");
#endif
  /* now the guard pixels */
  if ((xykey = (xyhash*)malloc(sizeof(xyhash)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((xkey = (uint32_t*)malloc(sizeof(int)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  if ((ykey = (uint32_t*)malloc(sizeof(int)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }

  minflux*=threshold;
  for(ci=0; ci<g_hash_table_size(xtol); ci++) 
    for(cj=0; cj<g_hash_table_size(ytom); cj++) {
        xykey->x=xarr[ci];
        xykey->y=yarr[cj];
        xp=(double*)g_hash_table_lookup(xyreg,xykey);
        if (!xp) { /* this key not in the hash table, therfore a guard pixel */
          *xkey=xykey->x;
          *ykey=xykey->y;
          lp=(double*)g_hash_table_lookup(xtol,xkey);
          mp=(double*)g_hash_table_lookup(ytom,ykey);
          if (lp && mp) {
            (*parr)[ck].x=xykey->x;
            (*parr)[ck].y=xykey->y;
            (*parr)[ck].l=*lp;
            (*parr)[ck].m=*mp;
            (*parr)[ck].ra=-1;
            (*parr)[ck].dec=-1;
            (*parr)[ck].sI=minflux;
            ck++;
          }
        }
  }
  
#ifdef DEBUG
  printf("Max %d pixels, filled %d pixels\n",*n,ck);
  for(ci=0; ci<*n; ci++)  {
    printf("[pixel (%d,%d), lm (%lf,%lf), radec (%lf,%lf), sI %lf]",(*parr)[ci].x,(*parr)[ci].y,(*parr)[ci].l,(*parr)[ci].m,(*parr)[ci].ra,(*parr)[ci].dec, (*parr)[ci].sI);
  }
#endif
  free(xykey);
  free(xkey);
  free(ykey);

  free(xarr);
  free(yarr);
  g_hash_table_destroy(xyreg);
  g_hash_table_destroy(xtol);
  g_hash_table_destroy(ytom);
  return 0;
}



int 
process_pixels(GHashTable *pixtable, double bmaj, double bmin, double bpa, double minpix, double threshold, int maxiter, int maxemiter, int use_em,int maxfits){
 GHashTableIter iter;
 pixellist *val;
 uint32_t *key_;
 hpixel *parr;
 int ntrue;
 int ci,cs;

 double aic;

 extsrc *srcx;

 fit_result *res; /* array for bookkeeping all fit results (reallocated for each blob in hash table)*/
 int nfits; /* no of different fits */
 double low_aic;
 int chosen;

 /*************************************/
 parr=0;
 /* NOTE: pixels here are +1 from the pixels in kvis */
  g_hash_table_iter_init (&iter, pixtable);
  while (g_hash_table_iter_next (&iter, (gpointer) &key_, (gpointer) &val))
  {
     printf("Island %u (%u pixels)\n",(uint32_t)*key_,g_list_length(val->pix));
     /* find max. no of possible fits (degrees of freedom) */
     /* also catch if beam < 3 pixels case */
     if (minpix>3) {
      nfits=(int)((double)g_list_length(val->pix)/minpix)+1;
     } else {
      nfits=(int)((double)g_list_length(val->pix)/3.0)+1;
     }
     /* override possible fits if maxfits>0 */
     if (maxfits && nfits>maxfits) {
      nfits=maxfits;
     }

     printf("Possible fits %d\n",nfits);
     /* determine convex hull for the pixels, in (l,m) coordinates */
     /* val->hull will be created */
     construct_boundary(val);
     add_guard_pixels(val->pix, threshold, &parr, &ntrue);

     if ((res= (fit_result*)malloc(sizeof(fit_result)*nfits))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       return 1;
     }

     /* evaluate all possible fits */
     for (cs=0; cs<nfits; cs++) {
       if (cs==0) {
       /************** strategy 1: single point *************************/
       res[cs].nsrc=cs+1;
       if ((res[cs].ll= (double*)malloc(sizeof(double)*res[cs].nsrc))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         return 1;
       }
       if ((res[cs].mm= (double*)malloc(sizeof(double)*res[cs].nsrc))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         return 1;
       }
       if ((res[cs].sI= (double*)malloc(sizeof(double)*res[cs].nsrc))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         return 1;
       }
       aic=fit_single_point0(parr,ntrue,bmaj,bmin,bpa,maxiter,res[cs].ll,res[cs].mm,res[cs].sI);
       //aic=fit_single_point(parr,ntrue,bmaj,bmin,bpa,maxiter,res[cs].ll,res[cs].mm,res[cs].sI);
       printf("AIC=%lg\n",aic);
       res[cs].aic=aic;
       } else {
       /************** strategy 2: multiple points ***********************/
       res[cs].nsrc=cs+1;
       if ((res[cs].ll= (double*)malloc(sizeof(double)*res[cs].nsrc))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        return 1;
       }
       if ((res[cs].mm= (double*)malloc(sizeof(double)*res[cs].nsrc))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        return 1;
       }
       if ((res[cs].sI= (double*)malloc(sizeof(double)*res[cs].nsrc))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        return 1;
       }
       aic=fit_N_point_em(parr,ntrue,bmaj,bmin,bpa,maxiter,maxemiter,res[cs].ll,res[cs].mm,res[cs].sI,res[cs].nsrc, val->Nh, val->hull);
       res[cs].aic=aic;
       printf("AIC=%lg\n",aic);
       }
     }

     /* now, go through the AIC values to find the lowest one */ 
     low_aic=INFINITY_L;
     chosen=0;
     for (ci=0; ci<nfits; ci++) {
      if (res[ci].aic<low_aic) {
       low_aic=res[ci].aic;
       chosen=ci;
      } 
     }
     /* now we have the lowest AIC, include only these sources */
     for (ci=0; ci<res[chosen].nsrc; ci++) {
       if ((srcx= (extsrc*)malloc(sizeof(extsrc)))==0) {
          fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
          return 1;
       }
       srcx->l=res[chosen].ll[ci];
       srcx->m=res[chosen].mm[ci];
       srcx->sI=res[chosen].sI[ci];
       val->slist=g_list_prepend(val->slist,srcx);
     }

     free(parr);
     for (ci=0; ci<nfits; ci++) {
       free(res[ci].ll);
       free(res[ci].mm);
       free(res[ci].sI);
     }
     free(res);
  }
 /*************************************/
 
 return 0;
}



/* lapack driver for symmetric eigenvalue problem */
/* A: size nxn matrix, only lower triangle (column major) is given
   n: matrix dimension
   ev: eigenvalues nx1
   work: size lwork x 1
   lwork: work length
*/
#ifdef EXTRA
static int
my_dsyev(double *A,int n, double *ev, double *work, int lwork) {
 extern void dsyev_(char *JOBZ,char* UPLO,int *N,double *A,int *LDA,double *W,double *WORK,int *LWORK,int *INFO);
 int info;
 char jobz='N'; /* only eigenvalues */
 char uplo='L'; /* lower triangle */
 dsyev_(&jobz,&uplo,&n,A,&n,ev,work,&lwork,&info);

 return info;
}
#endif /* EXTRA */




int
filter_pixels(GHashTable *pixtable, double wcutoff) {
  GHashTableIter iter;
  pixellist *val;
  uint32_t *key_;

  GList *pixval;
  hpixel *ppix;


  /* storage for LAPACK */
  double *A,*W;
  int N=2;
  double T,D,xmean,ymean;
  double *xpix,*ypix,*sI;
  int Npix,ci;
  double sItot,sIpeak,sImean;

  /* A: size 2x2 */
  if ((A=(double*)malloc(sizeof(double)*(size_t)N*N))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* W: size 2x1 */
  if ((W=(double*)malloc(sizeof(double)*(size_t)N))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  printf("################## probable ignore list (%lf) ##############\n",wcutoff);
  /* NOTE: pixels here are +1 from the pixels in kvis */
  g_hash_table_iter_init (&iter, pixtable);
  while (g_hash_table_iter_next (&iter, (gpointer) &key_, (gpointer) &val))
  {
     //printf("Island %u (%u pixels)\n",(uint32_t)*key_,g_list_length(val->pix));
     /* find the covariance of pixels: pixel (px,py)
      so C=|px^T.px  px^T.py|
           |py^T.px  py^T.py|
      lower triangle of A=[px^T.px py^T.px py^T.py]
     */
      A[0]=A[1]=A[2]=A[3]=0.0;   
      Npix=g_list_length(val->pix);
      if ((xpix=(double*)malloc(sizeof(double)*(size_t)Npix))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      if ((ypix=(double*)malloc(sizeof(double)*(size_t)Npix))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      if ((sI=(double*)malloc(sizeof(double)*(size_t)Npix))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      ci=0;
      xmean=ymean=0.0;
      for(pixval=val->pix; pixval!=NULL; pixval=g_list_next(pixval)) {
        ppix=pixval->data;
        xpix[ci]=ppix->l;
        ypix[ci]=ppix->m;
        sI[ci]=ppix->sI;
        xmean+=xpix[ci];
        ymean+=ypix[ci];
        ci++;
      }
      xmean/=(double)Npix;
      ymean/=(double)Npix;
      for (ci=0; ci<Npix;ci++) {
       xpix[ci]-=xmean;
       ypix[ci]-=ymean;
       A[0]+=xpix[ci]*xpix[ci];
       A[1]+=xpix[ci]*ypix[ci];
       A[2]+=ypix[ci]*ypix[ci];
      }
/*      printf("A=[%lf %lf;\n",A[0],A[1]);
      printf("   %lf %lf];\n",A[1],A[2]); */
      /* trace */
      T=A[0]+A[2];
      /* determinant */
      D=A[0]*A[2]-A[1]*A[1];
      W[0]=T*0.5+sqrt(T*T*0.25-D);
      W[1]=T*0.5-sqrt(T*T*0.25-D);
      //printf("%lf\n",W[0]/W[1]);

      /* second criterion, total flux, peak flux and mean flux */
      sItot=sIpeak=0.0;
      for (ci=0; ci<Npix;ci++) {
        sItot+=sI[ci];
        if (sIpeak<fabs(sI[ci])) {
          sIpeak=fabs(sI[ci]);
        }
      }
      sImean=sItot/(double)Npix;
     // printf("%lf %lf %lf\n",sItot,sImean,sIpeak);
     //printf("%lf\n",W[0]/(W[1]*sIpeak*sImean));

      if (W[0]/(W[1]*sIpeak*sImean) > wcutoff) {
        printf("%d\n",(uint32_t)*key_);
      }
      
      free(xpix);
      free(ypix);
      free(sI);

  } 
  free(A);
  free(W);

  printf("################## end ignore list                  ##############\n");

 return 0;
}
