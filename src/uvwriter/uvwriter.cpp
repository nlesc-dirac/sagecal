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
#include <iostream>
#include <casacore/ms/MeasurementSets/MSIter.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableVector.h>
#include <casacore/tables/Tables/TableRecord.h>
#include <casacore/tables/Tables/TableColumn.h>
#include <casacore/tables/Tables/TableIter.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Cube.h>
#include <fstream>
#include <math.h>
#include <complex>
#include <SpiceUsr.h>
#include "Dirac_radio.h"

//#define DEBUG

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
  } else {
    fprintf(stderr,"CSPICE kernel path 'CSPICE_KERNEL_PATH' is not found in environment variables\n");
    fprintf(stderr,"Download the kernels pck00011.tpc, naif0012.tls,\n moon_de440_220930.tf, moon_pa_de440_200625.bpc\n");
    fprintf(stderr,"And rerun the program after setting the directory where these kernels stored as CSPICE_KERNEL_PATH\n");
    exit(1);
  }
}

void
print_help(void) {
   fprintf(stderr,"Calculate the UVW coordinates for the given MS based on lunar frame.\n");
   fprintf(stderr,"Usage:\n");
   fprintf(stderr,"uvwriter -d MS\n");
   fprintf(stderr,"-d : input MS (TIME and ANTENNA positions will be used to calculate the UVW coordinates)\n");
}


/* for getopt() */
extern char *optarg;
extern int optind, opterr, optopt;

using namespace casacore;

int 
main(int argc, char **argv) {

  int c;
  char *inms=0;
  while ((c=getopt(argc,argv,"d:h"))!=-1) {
    switch(c) {
    case 'd':
      if (optarg) {
        inms=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
        if (inms== 0 ) {
         fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
         exit(1);
        }
        strcpy(inms,(char*)optarg);
      }
      break;
    default:
      print_help();
      break;
    }
  }
  if (!inms) {
    print_help();
    exit(0);
  }

  cspice_load_kernels();

  MeasurementSet ms(inms,Table::Update);
  MSAntenna _ant = ms.antenna();
  size_t N=_ant.nrow();
  ArrayColumn<double> position(_ant,MSAntenna::columnName(MSAntennaEnums::POSITION));
  double *xyz=new double[3*N];
  for (size_t ci=0; ci<N; ci++) {
    double *_pos=position(ci).data();
    //printf("Ant %ld %lf %lf %lf\n",ci,_pos[0],_pos[1],_pos[2]);
    xyz[3*ci]=_pos[0];
    xyz[3*ci+1]=_pos[1];
    xyz[3*ci+2]=_pos[2];
  }

  MSField _field = Table(ms.field());
  ArrayColumn<double> ref_dir(_field, MSField::columnName(MSFieldEnums::PHASE_DIR));
  Array<double> dir = ref_dir(0);
  double *ph = dir.data();
  double ra0=ph[0];
  double dec0=ph[1];

  printf("Antennas %ld phase center %lf,%lf (rad)\n",N,ra0,dec0);

  Block<int> sort(1);
  sort[0]=MS::TIME;
  MSIter mi(ms,sort,100.0);

  mi.origin();
  while(mi.more()) {
    /**************************************************************/
    Block<String> iv1(3);
    iv1[0] = "TIME";
    iv1[1] = "ANTENNA1";
    iv1[2] = "ANTENNA2";
    Table t=mi.table().sort(iv1,Sort::Ascending);
    size_t n_row=t.nrow();

    ROScalarColumn<int> a1(t,"ANTENNA1"), a2(t,"ANTENNA2");
    ArrayColumn<double> uvwCol(t,"UVW");
    ROScalarColumn<double> tut(t,"TIME");

    double old_t0=0.0;

    /* unit vector pointing to phase center */
    double e[3]={0.0,0.0,0.0};
    for (size_t row=0; row<n_row; row++) {
      size_t i=a1(row);
      size_t j=a2(row);
      Array<double> uvw=uvwCol(row);
      double *uvwp=uvw.data();
//      printf("utc %lf ant %ld %ld uvw %lf %lf %lf\n",tut(row),i,j,uvwp[0],uvwp[1],uvwp[2]);
      double t0=tut(row);
      if (old_t0 != t0) {
        /* rotate direction vector to this time */
        /* convert t0 (s) to JD (days) */
        double t0_d=t0/86400.0+2400000.5;
        /* ephemeris time from t0 (s) */
        double ep_t0=unitim_c(t0_d,"JED","ET");
        //printf("UTC %le (s) ET %le (s)\n",t0,ep_t0);
        
        /* rectangular coords of phace center */
        SpiceDouble srcrect[3],mtrans[3][3],v2000[3];
        /* ra,dec to rectangular */
        radrec_c(1.0,ra0*rpd_c(),dec0*rpd_c(),v2000);
        /* precess ep_t0 on lunar frame ME: mean Earth/polar axis, PA: principle axis */
        pxform_c("J2000","MOON_PA",ep_t0,mtrans);//MOON_PA,MOON_ME,IAU_EARTH,ITRF93
        /* rotate v2000 onto lunar frame */
        mxv_c(mtrans,v2000,srcrect);

        //printf("source J2000 %lf,%lf,%lf MOON %lf,%lf,%lf\n",v2000[0],v2000[1],v2000[2],
        // srcrect[0],srcrect[1],srcrect[2]);

        /* unit vector pointing to phase center */
        double smag=sqrt(srcrect[0]*srcrect[0]+srcrect[1]*srcrect[1]+srcrect[2]*srcrect[2]);
        e[0]=srcrect[0]/smag;
        e[1]=srcrect[1]/smag;
        e[2]=srcrect[2]/smag;

        old_t0=t0;
      }

      /* find vector stat1-stat2 */
      if (i!=j) {
       double v[3];
       v[0]=xyz[3*i]-xyz[3*j];
       v[1]=xyz[3*i+1]-xyz[3*j+1];
       v[2]=xyz[3*i+2]-xyz[3*j+2];
      
       /* project v onto plane normal to unit vector e */
       /* dot product v.e */
       double v_e=v[0]*e[0]+v[1]*e[1]+v[2]*e[2];
       /* projected v = v - (v . e) e */
       v[0] -= v_e*e[0];
       v[1] -= v_e*e[1];
       v[2] -= v_e*e[2];

       uvwp[0]=v[0];
       uvwp[1]=v[1];
       uvwp[2]=v[2];
       uvwCol.put(row,uvw);
      }
    }
    /**************************************************************/
    mi++;
  }

  delete [] xyz;
  if (inms) free(inms);
  return 0;
}
