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
#include <time.h> /* for timing */

#include "sagecal.h"

void
print_help(void) {
   fprintf(stderr,"Usage:\n");
   fprintf(stderr,"sagecal -d datafile -s sky.txt -c cluster.txt\n");
   fprintf(stderr,"Additional options:\n");
   fprintf(stderr,"-n no. of threads\n");
   fprintf(stderr,"-t tile size\n");
   fprintf(stderr,"-p solution file\n");
   fprintf(stderr,"-e max EM iterations: default 3\n");
   fprintf(stderr,"-g max iterations (within single EM): default 2\n");
   fprintf(stderr,"-l max LBFGS iterations: default 10\n");
   fprintf(stderr,"-m LBFGS memory size: default 7\n");
   fprintf(stderr,"-x exclude baselines length (lambda) lower than this in calibration: default 0\n");
   fprintf(stderr,"-w 1 or 0 : write ouput: default 1\n");
   fprintf(stderr,"-k cluster_id : correct residuals with solution of this cluster : default -99999\n");
   fprintf(stderr,"-o robust rho, robust matrix inversion during correction: default 1e-9\n");
   fprintf(stderr,"-f bandwidth MHz, for freq. smearing: default 0\n");
   fprintf(stderr,"\n\nAdvanced options:\n");
   fprintf(stderr,"-i 1 or 0 : interpolate or not : default 0 (disabled)\n");
   fprintf(stderr,"-j 0,1,2... 0 : OSaccel, 1 no OSaccel, 2: RLM, default 0\n");
   fprintf(stderr,"-b no. of BLAS threads: default 1\n");
   fprintf(stderr,"-u tiles debug mode: stop after this many solutions : default 0\n");
   fprintf(stderr,"-y GPU threads per block: default 128\n");
   fprintf(stderr,"-z reset current solution if error goes up: default 0\n");
   fprintf(stderr,"-a simulation mode (do not solve)\n");
   fprintf(stderr,"-q 0 (Chol),1 (QR),2 (SVD) : default 1\n");
   fprintf(stderr,"-r reset initial values if residual increases\n by this amount between iterations: default 5\n");
   fprintf(stderr,"Report bugs to <sarod@users.sf.net>\n");
}

void
print_copyright(void) {
  printf("SAGEcal HD 0.1.03 %s%s (C) 2011-2013 Sarod Yatawatta\nOSaccel by Sanaz Kazemi\n",CLMV,CLBFGSV);
}

/* for getopt() */
extern char *optarg;
extern int optind, opterr, optopt;


/********** reading and writing routines *************************/
static int
write_output_data(int xcount, double *x, int xstep, double *data, int offset, int dstep) {

   /* copy back data from x */
   /* d[4:dstep:(Nbase-1)*dstep+4] <<== x[0:Nbase-1] : real XX */
   my_dcopy(xcount, &x[0], xstep, &data[offset+4], dstep);
   /* d[5:dstep:(Nbase-1)*11+5] <<== x[1:Nbase-1+1] imag XX */
   my_dcopy(xcount, &x[1], xstep, &data[offset+5], dstep);
   /* d[6:dstep:(Nbase-1)*11+6] <<== x[2:Nbase-1+2] real XY */
   my_dcopy(xcount, &x[2], xstep, &data[offset+6], dstep);
   /* d[7:dstep:(Nbase-1)*11+7] <<== x[3:Nbase-1+3] imag XY */
   my_dcopy(xcount, &x[3], xstep, &data[offset+7], dstep);
   /* d[8:dstep:(Nbase-1)*11+8] <<== x[4:Nbase-1+4] real YX */
   my_dcopy(xcount, &x[4], xstep, &data[offset+8], dstep);
   /* d[9:dstep:(Nbase-1)*11+9] <<== x[5:Nbase-1+5] imag YX */
   my_dcopy(xcount, &x[5], xstep, &data[offset+9], dstep);
   /* d[10:dstep:(Nbase-1)*11+10] <<== x[6:Nbase-1+6] real YY */
   my_dcopy(xcount, &x[6], xstep, &data[offset+10], dstep);
   /* d[11:dstep:(Nbase-1)*11+11] <<== x[7:Nbase-1+7] imag YY */
   my_dcopy(xcount, &x[7], xstep, &data[offset+11], dstep);

 return 0;
} 

static int
read_input_aux_data(int count, double *data, int offset, int dstep,
   double *uu, double *vv, double *ww, double *flag, int incx) {
   /* d[0:dstep:(Nbase-1)*dstep] ==> u[0:Nbase-1] */
   my_dcopy(count, &data[offset], dstep, uu, incx);
   /* d[1:dstep:(Nbase-1)*dstep+1] ==> v[0:Nbase-1] */
   my_dcopy(count, &data[offset+1], dstep, vv, incx);
   /* d[2:dstep:(Nbase-1)*dstep+2] ==> w[0:Nbase-1] */
   my_dcopy(count, &data[offset+2], dstep, ww, incx);
   /* d[3:dstep:(Nbase-1)*dstep+3] flag  */
   my_dcopy(count, &data[offset+3], dstep, flag, incx);

   return 0;
}

static int
read_input_data(int count, double *data,int offset,int dstep,double *x, int xstep) {

   /* d[4:dstep:(Nbase-1)*dstep+4] ==> x[0:Nbase-1] : real XX */
   my_dcopy(count, &data[offset+4], dstep, &x[0], xstep);
   /* d[5:dstep:(Nbase-1)*dstep+5] ==> x[1:Nbase-1+1] imag XX */
   my_dcopy(count, &data[offset+5], dstep, &x[1], xstep);
   /* d[6:dstep:(Nbase-1)*dstep+6] ==> x[2:Nbase-1+2] real XY */
   my_dcopy(count, &data[offset+6], dstep, &x[2], xstep);
   /* d[7:dstep:(Nbase-1)*dstep+7] ==> x[3:Nbase-1+3] imag XY */
   my_dcopy(count, &data[offset+7], dstep, &x[3], xstep);
   /* d[8:dstep:(Nbase-1)*dstep+8] ==> x[4:Nbase-1+4] real YX */
   my_dcopy(count, &data[offset+8], dstep, &x[4], xstep);
   /* d[9:dstep:(Nbase-1)*dstep+9] ==> x[5:Nbase-1+5] imag YX */
   my_dcopy(count, &data[offset+9], dstep, &x[5], xstep);
   /* d[10:dstep:(Nbase-1)*dstep+10] ==> x[6:Nbase-1+6] real YY */
   my_dcopy(count, &data[offset+10], dstep, &x[6], xstep);
   /* d[11:dstep:(Nbase-1)*dstep+11] ==> x[7:Nbase-1+7] imag YY */
   my_dcopy(count, &data[offset+11], dstep, &x[7], xstep);

   return 0;
}


int main (int argc, char **argv) {
  double *data;
  int file;
  int count,c;
  int tcount;

  clus_source_t *carr;
  baseline_t *barr; 
  int M,Mt,ci,cj,ck;

  int N,Nbase,Nt,Nbt;
  double freq0,ra0,dec0;
  int max_lbfgs,lbfgs_m;
  int limit_timeslots=0;
  int write_output=1;
  int gpu_threads=128;

  /* for parsing integers */
  int base=10;
  char *endptr;

  FILE *sfp=0;


  complex double *coh;

  int doff,idxs,tilesz;

  double *uu,*vv,*ww,*x,*flag;
  /* u,v,w: arrays of Nbase*tileszx1 */
  /* x: array of Nbase*8*tilesz x 1: eash baseline has 8 value re,im XX,XY,YX,YY */
  int dstep=12; /* how many double values per baseline */

  /* parameters */
  double *p;
  double **pm;

  /* copy of old values for interpolation */
  baseline_t *barr0;
  double *uu0,*vv0,*ww0,*x0;
  double *p0, *pinit; /* parameter arrays */
  int Ntilebase,Ntilebase2;
  int do_interpolate=0;
  int tilesz2,Ntilebase1;
  int firsttile=1;
  int lasttile=0;

  char *skyfile=0,*clusterfile=0,*datafile=0,*solfile=0;
  char *nbt=0;

  int predict_vis=0;
  int retval;

  /* short baselines */
  double uvmin=0.0;
  /* bandwidth for freq smearing */
  double fdelta=0.0;

  /* initial and final residuals */
  double res_0,res_1;
  /* previous residual */
  double res_prev=CLM_DBL_MAX;
  double res_ratio=5; /* how much can the residual increase before resetting solutions */
  res_0=res_1=0.0;

  int solver_mode=0;  /* 0: with OS, 1: No OS, 2: Robust LM */
  int max_iter,max_emiter;
  int reset_sol=0;
  max_iter=2;
  max_emiter=3;
  Nbt=1;
  max_lbfgs=10;
  lbfgs_m=7;

  Nt=2; 
  /* tile size: read more than one timeslot */
  tilesz=4;
  /* linear solver to use 0:Chol, 1: QR, 2: SVD*/
  int linsolv=1; /* QR solver good enough */

  time_t start_time, end_time;
  double elapsed_time;

  int ccid=-99999; /* cluster id for correction of residual (default -99999) */
  double rho=1e-9; /* MMSE robust parameter */

  int format=1;
  print_copyright();
  /* free j,v */
  while ((c=getopt(argc,argv,"d:s:c:t:n:p:e:f:g:b:i:j:l:m:u:x:w:y:z:q:r:k:o:ha"))!=-1) {
   switch(c) {
    case 'd':
     if (optarg) {
      datafile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( datafile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(datafile,(char*)optarg);
     }
    break;
    case 's':
     if (optarg) {
      skyfile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( skyfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(skyfile,(char*)optarg);
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
    case 'p':
     if (optarg) {
      solfile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( solfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(solfile,(char*)optarg);
     }
    break;
   case 'n':
     if (optarg) {
       Nt=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) Nt=1;
     }
    break;
   case 't':
     if (optarg) {
       tilesz=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) tilesz=1;
     }
    break;
   case 'e':
     if (optarg) {
       max_emiter=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) max_emiter=4;
     }
    break;
   case 'g':
     if (optarg) {
       max_iter=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) max_iter=3;
     }
    break;
   case 'f':
     if (optarg) {
       fdelta=strtod(optarg,0);
       if (fdelta<0) fdelta=0.0;
       /* input is MHz, convert to Hz */
       fdelta*=1e6;
     }
    break;
   case 'b':
     if (optarg) {
       Nbt=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) Nbt=1;
     }
    break;
   case 'l':
     if (optarg) {
       max_lbfgs=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) max_lbfgs=0;
     }
    break;
   case 'm':
     if (optarg) {
       lbfgs_m=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) lbfgs_m=6;
     }
    break;
   case 'u':
     if (optarg) {
       limit_timeslots=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) limit_timeslots=0;
     }
    break;
   case 'x':
     if (optarg) {
       uvmin=strtod(optarg,0);
       if (uvmin<0) uvmin=0.0;
     }
    break;
   case 'w':
     if (optarg) {
       write_output=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) write_output=1;
     }
    break;
   case 'y':
     if (optarg) {
       gpu_threads=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) gpu_threads=128;
       if (gpu_threads!=32 || gpu_threads!=64 || gpu_threads!=128 || gpu_threads!=256 || gpu_threads!=512) {
         fprintf(stderr,"Use a power of two (32,64,..512) for GPU threads. Instead of %d using 128\n",gpu_threads);
         gpu_threads=128;
       }
     }
    break;
   case 'z':
     if (optarg) {
       reset_sol=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) reset_sol=0;
     }
    break;
   case 'q':
     if (optarg) {
       linsolv=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) linsolv=0;
     }
    break;
   case 'r':
     if (optarg) {
       res_ratio=strtod(optarg,0);
       if (res_ratio<0.0) res_ratio=5.0;
     }
    break;
   case 'i':
     if (optarg) {
       do_interpolate=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) do_interpolate=1; 
     }
    break;
   case 'j':
     if (optarg) {
       solver_mode=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) solver_mode=0; 
       if (solver_mode<0|| solver_mode>2) solver_mode=0;
     }
    break;
   case 'k':
     if (optarg) {
       ccid=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) ccid=-99999;
     }
    break;
   case 'o':
     if (optarg) {
       rho=strtod(optarg,0);
       if (rho<0) rho=1e-9;
     }
    break;
   case 'a':
    predict_vis=1;
    break;
   case 'h':
    print_help();
    exit(0);
    break;
   default:
    fprintf(stderr,"%s: %d: invalid arguments",__FILE__,__LINE__);
    print_help();
    exit(0);
    break;
   }
  }
  

  if (!skyfile || !clusterfile || !datafile) {
    fprintf(stderr,"%s: %d: no input files specified\n",__FILE__,__LINE__);
    print_help();
    exit(0);
  }
  if (solfile && !predict_vis) {
    if ((sfp=fopen(solfile,"w+"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      return 1;
    }
  }

  nbt=(char*)calloc((size_t)4,sizeof(char));
  if ( nbt== 0 ) {
    fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
    exit(1);
  }
  sprintf(nbt,"%d",Nbt);
  /* update env variable for GOTO blas */
  setenv("GOTO_NUM_THREADS",nbt,1);
//void goto_set_num_threads(int num_threads);
//void openblas_set_num_threads(int num_threads);
  free(nbt);
  /* open the file for reading and  writing*/
  file = open(datafile, O_RDWR);
  if ( file <= 0 ) {
     fprintf(stderr,"%s: %d: no file open\n",__FILE__,__LINE__);
     exit(1);
  }
 
  open_data_stream(file, &data, &count, &N, &freq0, &ra0, &dec0);

  /* pass freq: for spectral index, ra,dec of phase center to calculate l,m */
  read_sky_cluster(skyfile,clusterfile,&carr,&M,freq0,ra0,dec0,format);
  printf("got %d clusters\n",M);

  /* note: actual no of data values=count-stations-freq-ra-dec-ignored_stations =count-5 */
  /* note: per baseline, we have 12 values u,v,w,flag,XX(re,im),XY(re,im),YX(re,im),YY(re,im) */
  tcount=count-5; /* never exceed this value when accessing data */
  Nbase=N*(N-1)/2;
  printf("No. stations=%d, baselines=%d, timeslots=%d, freq=%lf ph=(%lf,%lf)\n",N,Nbase,tcount/(Nbase*dstep),freq0,ra0,dec0);
  printf("using %d threads, tile size=%d\n",Nt,tilesz);
  /* override tilesz if it is too large  */
  if (tilesz*Nbase*dstep>tcount) {
   tilesz=tcount/(Nbase*dstep);
   printf("limiting tilesize to %d\n",tilesz);
  }

  /* allocate memory */
  if ((uu=(double*)calloc((size_t)Nbase*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((vv=(double*)calloc((size_t)Nbase*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((ww=(double*)calloc((size_t)Nbase*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((flag=(double*)calloc((size_t)Nbase*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((x=(double*)calloc((size_t)Nbase*8*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* array to store baseline->sta1,sta2 map */
  if ((barr=(baseline_t*)calloc((size_t)Nbase*tilesz,sizeof(baseline_t)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  generate_baselines(Nbase,tilesz,N,barr,Nt);

  /* calculate actual no of parameters needed,
   this could be > M */
  Mt=0;
  for (ci=0; ci<M; ci++) {
    //printf("cluster %d has %d time chunks\n",carr[ci].id,carr[ci].nchunk);
    Mt+=carr[ci].nchunk;
  }
  printf("total effective clusters=%d\n",Mt);

  /* parameters 8*N*M ==> 8*N*Mt */
  if ((p=(double*)calloc((size_t)N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* update cluster array with correct pointers to parameters */
  cj=0;
  for (ci=0; ci<M; ci++) {
    if ((carr[ci].p=(int*)calloc((size_t)carr[ci].nchunk,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    for (ck=0; ck<carr[ci].nchunk; ck++) {
      carr[ci].p[ck]=cj*8*N;
      cj++;
    }
  }

  /* pointers to parameters */
  if ((pm=(double**)calloc((size_t)Mt,sizeof(double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* setup the pointers */
  for (ci=0; ci<Mt; ci++) {
   pm[ci]=&(p[ci*8*N]);
  }
  /* initilize parameters to [1,0,0,0,0,0,1,0] */
  for (ci=0; ci<Mt; ci++) {
    for (cj=0; cj<N; cj++) {
      pm[ci][8*cj]=1.0;
      pm[ci][8*cj+6]=1.0;
    }
  }
  /* backup of default initial values */
  if ((pinit=(double*)calloc((size_t)N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  memcpy(pinit,p,(size_t)N*8*Mt*sizeof(double));

  /* coherencies */
  if ((coh=(complex double*)calloc((size_t)(M*Nbase*tilesz*4),sizeof(complex double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  if (write_output && do_interpolate) {
  /* backup memory for interpolation */
  if ((uu0=(double*)calloc((size_t)Nbase*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((vv0=(double*)calloc((size_t)Nbase*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((ww0=(double*)calloc((size_t)Nbase*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((barr0=(baseline_t*)calloc((size_t)Nbase*tilesz,sizeof(baseline_t)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((p0=(double*)calloc((size_t)N*8*M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((x0=(double*)calloc((size_t)Nbase*8*tilesz,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  } else {
   uu0=vv0=ww0=p0=x0=0;
   barr0=0;
  }

#ifdef DEBUG
   printf("********************************\n");
   for (cj=0; cj<N*8; cj++) {
     printf("%d/%d:",cj/8,cj);
     for (ci=0; ci<M; ci++) {
      printf(" %lf",pm[ci][cj]);
    }
    printf("\n");
   }
   printf("********************************\n");
#endif

  int randomize=1;

  /* having tilesz>1 means Nbase becomes Nbase*tilesz */
  doff=0;  
  /* override tcount if limit_timeslots>0 */
  if ((limit_timeslots>0) && (doff+tilesz)*Nbase*dstep*limit_timeslots<tcount ) {
   tcount=(doff+tilesz)*Nbase*dstep*limit_timeslots;
   printf("limiting timeslots to %d\n",tcount/(Nbase*dstep));
  }
  retval=0;

  int oldtilesz=tilesz;
  int doff0=doff;

  tilesz2=tilesz/2;
  Ntilebase=Nbase*tilesz;
  Ntilebase2=Nbase*tilesz2; /* second half */
  Ntilebase1=Ntilebase-Ntilebase2; /* first half */

  while(tilesz>0 && (doff+tilesz)*Nbase*dstep<=tcount) {
   start_time = time(0);
   idxs=doff*Nbase*dstep;
   if (firsttile||lasttile) {
     Ntilebase=Nbase*tilesz;
   }

   if (write_output && do_interpolate) {
     /* backup old values */
     memcpy(p0,p,sizeof(double)*N*8*M);
     /* interpolation: split array into
      | 0.....(tilesz-1)/2 | (tilesz-1)/2+1....(tilesz-1) |
      | 0..... tilesz2-1   | tilesz2  ......    tilesz-1  |
     */
     if (firsttile) {
       tilesz2=tilesz/2;
       Ntilebase2=Nbase*tilesz2; /* second half */
       Ntilebase1=Ntilebase-Ntilebase2; /* first half */
     } else if (lasttile) {
       tilesz2=oldtilesz/2;
       Ntilebase1=Nbase*(oldtilesz-tilesz2); /* first half */
       Ntilebase2=Nbase*(tilesz/2); /* second half */
       /* only copy the last half of old u,v,w,barr */
       memcpy(uu0,&uu[Nbase*(oldtilesz/2)],Ntilebase1*sizeof(double));
       memcpy(vv0,&vv[Nbase*(oldtilesz/2)],Ntilebase1*sizeof(double));
       memcpy(ww0,&ww[Nbase*(oldtilesz/2)],Ntilebase1*sizeof(double));
       memcpy(barr0,&barr[Nbase*(oldtilesz/2)],(size_t)Ntilebase1*sizeof(baseline_t));
     } else { 
       /* only copy the last half of old u,v,w,barr */
       memcpy(uu0,&uu[Ntilebase2],Ntilebase1*sizeof(double));
       memcpy(vv0,&vv[Ntilebase2],Ntilebase1*sizeof(double));
       memcpy(ww0,&ww[Ntilebase2],Ntilebase1*sizeof(double));
       memcpy(barr0,&barr[Ntilebase2],(size_t)Ntilebase1*sizeof(baseline_t));
     }
   }

   read_input_aux_data(Ntilebase,data,idxs,dstep,uu,vv,ww,flag,1);
   read_input_data(Ntilebase,data,idxs,dstep,x,8);

   printf("timeslot %d\n",doff);
   /* rescale u,v,w by 1/c NOT to wavelengths, that is done later */
   my_dscal(Ntilebase,1.0/CONST_C,uu);
   my_dscal(Ntilebase,1.0/CONST_C,vv);
   my_dscal(Ntilebase,1.0/CONST_C,ww);
   /* update baseline flags */
   /* and set x[]=0 for flagged values */
   preset_flags_and_data(Ntilebase,flag,barr,x,Nt);

   if (predict_vis) {
    predict_visibilities(uu,vv,ww,x,N,Nbase,tilesz,barr,carr,M,freq0,fdelta,Nt);
   } else {
    precalculate_coherencies(uu,vv,ww,coh,N,Ntilebase,barr,carr,M,freq0,fdelta,uvmin,Nt); 
#ifndef HAVE_CUDA
    retval=sagefit_visibilities(uu,vv,ww,x,N,Nbase,tilesz,barr,carr,coh,M,Mt,freq0,fdelta,p,uvmin,Nt,max_emiter,max_iter,max_lbfgs,lbfgs_m,gpu_threads,linsolv,solver_mode,2.0,30.0,randomize,&res_0,&res_1);
#endif
#ifdef HAVE_CUDA
    // in COMA one_gpu
    //retval=sagefit_visibilities_dual_pt_one_gpu(uu,vv,ww,x,N,Nbase,tilesz,barr,carr,coh,M,Mt,freq0,fdelta,p,uvmin,Nt,max_emiter,max_iter,max_lbfgs,lbfgs_m,gpu_threads,linsolv,solver_mode,&res_0,&res_1);
    retval=sagefit_visibilities_dual_pt_flt(uu,vv,ww,x,N,Nbase,tilesz,barr,carr,coh,M,Mt,freq0,fdelta,p,uvmin,Nt,max_emiter,max_iter,max_lbfgs,lbfgs_m,gpu_threads,linsolv,solver_mode,2.0,30.0,randomize,&res_0,&res_1);
#endif

   /* if residual has increased, better restart from nominal values for
     solutions */
   if (retval==-1 && reset_sol) {
    /* initilize parameters to [1,0,0,0,0,0,1,0] */
    memcpy(p,pinit,sizeof(double)*N*8*M);

   /* copy original data from file */
   read_input_data(Ntilebase,data,idxs,dstep,x,8);

   preset_flags_and_data(Ntilebase,flag,barr,x,Nt);
   /* re-run minimization */
#ifndef HAVE_CUDA
   retval=sagefit_visibilities(uu,vv,ww,x,N,Nbase,tilesz,barr,carr,coh,M,Mt,freq0,fdelta,p,uvmin,Nt,max_emiter,max_iter,max_lbfgs,lbfgs_m,gpu_threads,linsolv,solver_mode,2.0,30.0,randomize,&res_0,&res_1);
#endif
#ifdef HAVE_CUDA
   // in COMA one_gpu
   //retval=sagefit_visibilities_dual_pt_one_gpu(uu,vv,ww,x,N,Nbase,tilesz,barr,carr,coh,M,Mt,freq0,fdelta,p,uvmin,Nt,max_emiter,max_iter,max_lbfgs,lbfgs_m,gpu_threads,linsolv,solver_mode,&res_0,&res_1);
   retval=sagefit_visibilities_dual_pt_flt(uu,vv,ww,x,N,Nbase,tilesz,barr,carr,coh,M,Mt,freq0,fdelta,p,uvmin,Nt,max_emiter,max_iter,max_lbfgs,lbfgs_m,gpu_threads,linsolv,solver_mode,2.0,30.0,randomize,&res_0,&res_1);
#endif
    }
   }

   if (write_output) {
   /* interpolate visibilities between p0  and p parameters,
      so initial solution, no data is written */
     if (do_interpolate) {
       if (firsttile) {
         /* copy only first half */
         write_output_data(Nbase*(tilesz/2),x,8,data,doff0*Nbase*dstep,dstep);
         doff0=doff0+tilesz/2;
       } else if (lasttile) {
         /* only copy the first half of current u,v,w,barr to last half */
         memcpy(&uu0[Ntilebase1],uu,Ntilebase2*sizeof(double));
         memcpy(&vv0[Ntilebase1],vv,Ntilebase2*sizeof(double));
         memcpy(&ww0[Ntilebase1],ww,Ntilebase2*sizeof(double));
         memcpy(&barr0[Ntilebase1],barr,(size_t)Ntilebase2*sizeof(baseline_t));

         read_input_data(Ntilebase1+Ntilebase2,data,doff0*Nbase*dstep,dstep,x0,8);

         precalculate_coherencies(uu0,vv0,ww0,coh,N,Nbase*(oldtilesz-oldtilesz/2+tilesz/2),barr0,carr,M,freq0,fdelta,0,Nt); 
         calculate_residuals(uu0,vv0,ww0,p0,p,x0,N,Nbase,oldtilesz-oldtilesz/2+tilesz/2,barr0,carr,coh,M,freq0,fdelta,Nt,ccid,rho);
         write_output_data(Ntilebase1+Ntilebase2,x0,8,data,doff0*Nbase*dstep,dstep);
         doff0=doff0+oldtilesz-oldtilesz/2+tilesz/2;
         /* now write the tail end */
         write_output_data(Nbase*(tilesz-tilesz/2),&x[Nbase*8*(tilesz/2)],8,data,doff0*Nbase*dstep,dstep);
       } else { 
         /* only copy the first half of current u,v,w,barr to last half */
         memcpy(&uu0[Ntilebase1],uu,Ntilebase2*sizeof(double));
         memcpy(&vv0[Ntilebase1],vv,Ntilebase2*sizeof(double));
         memcpy(&ww0[Ntilebase1],ww,Ntilebase2*sizeof(double));
         memcpy(&barr0[Ntilebase1],barr,(size_t)Ntilebase2*sizeof(baseline_t));

         read_input_data(Ntilebase,data,doff0*Nbase*dstep,dstep,x,8);
         precalculate_coherencies(uu0,vv0,ww0,coh,N,Nbase*tilesz,barr0,carr,M,freq0,fdelta,0,Nt); 
         calculate_residuals(uu0,vv0,ww0,p0,p,x,N,Nbase,tilesz,barr0,carr,coh,M,freq0,fdelta,Nt,ccid,rho);
         write_output_data(Ntilebase,x,8,data,doff0*Nbase*dstep,dstep);
         doff0=doff0+tilesz;
       }
     } else {
       if (!predict_vis) {
         /* to calculate residual of flagged data points, need 
         a fresh copy of the data */
         read_input_data(Ntilebase,data,idxs,dstep,x,8);
         /* now calculate residual for all data points */
         calculate_residuals(uu,vv,ww,p,p,x,N,Nbase,tilesz,barr,carr,coh,M,freq0,fdelta,Nt,ccid,rho);
       } 
       write_output_data(Ntilebase,x,8,data,idxs,dstep);
     }
   }
   //printf("range %d:%d (%d)\n",idxs,idxs+Nbase*tilesz*dstep,doff);

   doff=doff+tilesz;

   /* print solutions to file */
   if (solfile && !predict_vis) {
    /* solutions are reverse ordered by cluster */
   /* for (cj=0; cj<N*8; cj++) {
     fprintf(sfp,"%d ",cj);
     for (ci=M-1; ci>=0; ci--) {
      fprintf(sfp," %lf",pm[ci][cj]);
     }
     fprintf(sfp,"\n");
    } */

    for (cj=0; cj<N*8; cj++) {
     fprintf(sfp,"%d ",cj);
     for (ci=M-1; ci>=0; ci--) {
       for (ck=0; ck<carr[ci].nchunk; ck++) {
        fprintf(sfp," %lf",p[carr[ci].p[ck]+cj]);
       }
     }
     fprintf(sfp,"\n");
    }
   }

   /* if residual has increased too much, reset solutions to original
      initial values */
   if (res_1>res_ratio*res_prev) {
     /* reset solutions so next iteration has default initial values */
     memcpy(p,pinit,sizeof(double)*N*8*M);
   } else if (res_1<res_prev) { /* only store the min value */
    res_prev=res_1;
   }

   end_time = time(0);
   elapsed_time = ((double) (end_time-start_time)) / 60.0;
   printf("Residual: initial=%lf, final=%lf, Time spent=%lf minutes\n",res_0,res_1,elapsed_time);
   fflush(stdout);

   /* handle tail end of data here */
   if ((doff+tilesz)*Nbase*dstep>tcount) {
    oldtilesz=tilesz;
    /* since here (doff+tilesz)*Nbase*dstep>tcount, 
    reduce the tilesz until (doff+tilesz)*Nbase*dstep>tcount */
    do {
     tilesz--;
    } while ((doff+tilesz)*Nbase*dstep>tcount);
    lasttile=1;
   }
   /* reset first tile flag */
   if (firsttile) { firsttile=0; }
  }

  printf("Done.\n");
  close_data_stream(data,count);
  close(file);
  exinfo_gaussian *exg;
  exinfo_disk *exd;
  exinfo_ring *exr;
  exinfo_shapelet *exs;

  for (ci=0; ci<M; ci++) {
    free(carr[ci].ll);
    free(carr[ci].mm);
    free(carr[ci].nn);
    free(carr[ci].sI);
    free(carr[ci].p);
    for (cj=0; cj<carr[ci].N; cj++) {
     /* do a proper typecast before freeing */
     switch (carr[ci].stype[cj]) {
      case STYPE_GAUSSIAN:
        exg=(exinfo_gaussian*)carr[ci].ex[cj];
        if (exg) free(exg);
        break;
      case STYPE_DISK:
        exd=(exinfo_disk*)carr[ci].ex[cj];
        if (exd) free(exd);
        break;
      case STYPE_RING:
        exr=(exinfo_ring*)carr[ci].ex[cj];
        if (exr) free(exr);
        break;
      case STYPE_SHAPELET:
        exs=(exinfo_shapelet*)carr[ci].ex[cj];
        if (exs)  {
          if (exs->modes) {
            free(exs->modes);
          }
          free(exs);
        }
        break;
      default:
        break;
     }
    }
    free(carr[ci].ex);
    free(carr[ci].stype);
    free(carr[ci].sI0);
    free(carr[ci].f0);
    free(carr[ci].spec_idx);
  }
  free(carr);
  free(skyfile);
  free(clusterfile);
  free(datafile);
  if (solfile) {
   if (!predict_vis) {
    fclose(sfp);
   }
   free(solfile);
  }
  free(barr);

  free(uu);
  free(vv);
  free(ww);
  free(flag);
  free(x);
  free(p);
  free(pinit);
  free(pm);
  free(coh);

  if (write_output && do_interpolate) {
   free(uu0);
   free(vv0);
   free(ww0);
   free(barr0);
   free(p0);
   free(x0);
  }
   
  return(0);

}
