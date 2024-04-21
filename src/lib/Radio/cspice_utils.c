/*
 *
 Copyright (C) 2024 Sarod Yatawatta <sarod@users.sf.net>
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


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <SpiceUsr.h>
#include "Dirac_radio.h"

void
cspice_load_kernels(void) {
  /* get env CSPICE_KERNEL_PATH */
  char* cspice_path = getenv("CSPICE_KERNEL_PATH");
  if (cspice_path) {
    const char *kname="/pck00011.tpc\0";
    char *fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    kname="/naif0012.tls\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    kname="/moon_de440_220930.tf\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    kname="/moon_pa_de440_200625.bpc\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);

    /* following kernel only needed for ITRF93 frame */
    /*
    kname="/earth_000101_240713_240419.bpc\0";
    fullname=(char*)calloc((size_t)strlen((char*)cspice_path)+1+strlen((char*)kname),sizeof(char));
    if (fullname == 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
    }
    strcpy(fullname,cspice_path);
    strcpy((char*)&(fullname[strlen(cspice_path)]),kname);
    printf("loading %s\n",fullname);
    furnsh_c(fullname);
    free(fullname);
    */

  } else {
    fprintf(stderr,"CSPICE kernel path 'CSPICE_KERNEL_PATH' is not found in environment variables\n");
    fprintf(stderr,"Download the kernels pck00011.tpc, naif0012.tls,\n moon_de440_220930.tf, moon_pa_de440_200625.bpc\n");
    fprintf(stderr,"And rerun the program after setting the directory where these kernels stored as CSPICE_KERNEL_PATH\n");
    exit(1);
  }
}
