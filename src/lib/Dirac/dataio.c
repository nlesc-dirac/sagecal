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


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "Dirac.h"

int 
open_data_stream(int file, double **d, int *count, int *N, double *freq0, double *ra0, double *dec0) {
  struct stat statbuf;

  int ig;
  

 /* find the file size */
 if (fstat (file,&statbuf) < 0) {
   fprintf(stderr,"%s: %d: no file open\n",__FILE__,__LINE__);
   exit(1);
 }

 //printf("file size (bytes) %d\n",(int)statbuf.st_size);
 /* total double values is size/8 */
 *count=statbuf.st_size/8;
 //printf("total double values %d\n",*count);
  
  /* map the file to memory */
  *d= (double*)mmap(NULL,  statbuf.st_size, PROT_READ|PROT_WRITE, MAP_SHARED, file, 0);
  if ( !d) {
     fprintf(stderr,"%s: %d: no file open\n",__FILE__,__LINE__);
		 exit(1);
  }

  /* remove header from data */
  *N=(int)(*d)[0];
  *freq0=(*d)[1];
  *ra0=(*d)[2];
  *dec0=(*d)[3];
  /* read ignored stations and discard them */
  ig=(int)(*d)[4]; 
  /* make correct value for N */
  *N=*N-ig;
 
  printf("Ignoring %d stations\n",ig);
  /* increment to data */
  *d=&((*d)[5+ig]); 
  
  return(0);
}


int
close_data_stream(double *d, int count) {
  
  /* sync to disk */
  msync(d, (size_t)count*sizeof(double), MS_SYNC );
  munmap((void*)d, (size_t)count*sizeof(double));
  return 0;
}

