/*
 *
 Copyright (C) 2015 Sarod Yatawatta <sarod@users.sf.net>  
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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include "Dirac.h"

// This takes the float value 'val', converts it to red, green & blue values, then 
// sets those values into the image memory buffer location pointed to by 'ptr'
// ptr: 3 char array
inline void 
setRGB(unsigned char *ptr, float val) {
  int v = (int)(val * 767);
  if (v < 0) v = 0;
  if (v > 767) v = 767;
  int offset = v % 256;

  if (v<256) {
    ptr[0] = 0; ptr[1] = 0; ptr[2] = offset;
  }
  else if (v<512) {
    ptr[0] = 0; ptr[1] = offset; ptr[2] = 255-offset;
  }
  else {
    ptr[0] = offset; ptr[1] = 255-offset; ptr[2] = 0;
  }
}


/* write image to filename, size widthxheight pixels
  use buffer as data*/
static int 
write_ppm_image(const char* filename, int width, int height, float *buffer) {
  FILE *fp;

  int oversample=1;

  // Open file for writing (binary mode)
  if ((fp = fopen(filename, "wb")) == NULL)  {
    fprintf(stderr, "%s: %d: Could not open file %s for writing\n",__FILE__,__LINE__,filename);
    exit(1);
  }
  fprintf(fp, "P6\n%i %i 255\n", oversample*width, oversample*height);

  // Write image data
  int x, y;
  unsigned char rgb[3];
  for (y=0; y<height; y++) {
#pragma GCC ivdep
    for (x=0; x<width; x++) {
      setRGB(rgb, buffer[y*width + x]);
      fputc(rgb[0],fp);
      fputc(rgb[1],fp);
      fputc(rgb[2],fp);
    }
  }

  if (fp != NULL) fclose(fp);
  return 0;
}


/* W : tensor each plane (MxM),  N planes
   write as N squares of size  MxM onto an image (PPM file) 
   normalize: if 1, scale each panel to fit to colour range, else if 0, normalize the full panel to have max of 1 */
int
convert_tensor_to_image(double *W, const char *filename, int N, int M, int normalize) {

 /* determine size of float buffer needed for image */
 int panel_m=(int)ceil(sqrt((double)N));
 int panel_n=(N+panel_m-1)/panel_m;
 int patch_size=M*M;
 int ci,cj,ck,col;
 
 /* so we have panels of panel_m x panel_n, of size MxM */
 int P=MAX(panel_m,panel_n);

 /* image size (P.M) x (P.M) pixels */
 float *B=0;
 if((B=(float*)calloc((size_t)P*P*M*M,sizeof(float)))==0) {
   fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
   exit(1);
 }

 /* if normalize==1, scale each column of W to fit [0,1] */
 double W_max_diff=0.0;
 double *W_min,*W_max;
 if((W_min=(double*)calloc((size_t)N,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
   exit(1);
 }
 if((W_max=(double*)calloc((size_t)N,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
   exit(1);
 }
 for (col=0; col<N; col++) {
   W_min[col]=1e9;
   W_max[col]=-1e9;
#pragma GCC ivdep
   for(ci=0; ci<patch_size; ci++) {
     if (W_min[col]>W[col*patch_size+ci]) { W_min[col]=W[col*patch_size+ci];}
     if (W_max[col]<W[col*patch_size+ci]) { W_max[col]=W[col*patch_size+ci];}
   }
   if (W_max_diff<(W_max[col]-W_min[col])) {
     W_max_diff=W_max[col]-W_min[col];
   }
 }

 /* cut off very small values (prevent plotting noise), say less than 0.1 of the peak column */
 for (col=0; col<N; col++) {
   if ((W_max_diff*0.1 > (W_max[col]-W_min[col]))
       && (W_max[col]-W_min[col] < 1.0) ) {
     W_max[col]=1.0;
     W_min[col]=0.0;
   }
 }

 /* copy W to B, also while scaling values to [0,1] */
 for (col=0; col<N; col++) {
   double scalefactor=(normalize?1.0/(W_max[col]-W_min[col]):1.0/W_max_diff);
   /* map each column of MxM values in W to panel (x,y) */
   int x=col%P; // in 0...panel_m-1
   int y=col/P; // in 0...panel_n-1
   /* row and column offsets for this panel */
   int col_off=(x*M)*M*panel_n; /* full columns to the left, each panel has M columns,  each column size M*panel_n */
   int row_off=y*M; /* rows above, each of thikness 1 */
   /* ranges to write
    * col_off+row_off+[0:M-1]
    * col_off+row_off+1*panel_n*M+[0:M-1]
    * col_off+row_off+2*panel_n*M+[0:M-1]
    * .. until all M columns are written */
   ck=0;
   for (ci=0; ci<M; ci++) {
#pragma GCC ivdep
     for (cj=0; cj<M; cj++) {
       B[col_off+row_off+ci*panel_n*M+cj]=(float)(W[col*patch_size+ck]-W_min[col])*scalefactor;
       ck++;
     }
   }
 }

 write_ppm_image(filename, P*M, P*M, B);

 free(B);
 free(W_min);
 free(W_max);
 return 0;
}
