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

void
print_help(void) {
   fprintf(stderr,"Usage:\n");
   fprintf(stderr,"buildsky -f image.fits -m mask.fits\n");
   fprintf(stderr,"or:\n");
   fprintf(stderr,"buildsky -d fits_directory -m mask.fits\n\n");
   fprintf(stderr,"Additional options:\n");
   fprintf(stderr,"-t threshold: guard pixels will have this amount of flux as the lowest flux value (defalut 0)\n");
   fprintf(stderr,"-i maxiter: max no of LM iterations\n");
   fprintf(stderr,"-e maxemiter: max no of EM iterations\n");
   fprintf(stderr,"-n : Use normal LM instead of EM algorithm\n");
   fprintf(stderr,"PSF options\n-a : major axis width (arcsec): default 10\n");
   fprintf(stderr,"-b : minor axis width (arcsec): default 10\n");
   fprintf(stderr,"-p : position angle, measured from positive y axis, counter clockwise (deg): default 0\n");
   fprintf(stderr,"If NO PSF is given, it will be read from the FITS file\n\n");
   fprintf(stderr,"-o format: output format (0: BBS, 1: LSM (with 3 order spec.idx) 2: (l,m) positions) default BBS\n");
   fprintf(stderr,"-g ignorelist: text file of island numbers to ignore: default none\n");
   fprintf(stderr,"-w cutoff: cutoff value to detect possible sidelobes: defalut 0\n");
   fprintf(stderr,"-c rd: merge components closer than rd*(bmaj+bmin)/2: default 1.0\n");
   fprintf(stderr,"-l maxfits: limit the number of attempted fits to this value: default 10\n");
   fprintf(stderr,"-k clusters: max no of clusters to cluster the sky\n(-ve clusters for hierarchical, 0 for no clustering)\n");
   fprintf(stderr,"-s string: additional unique string to use with source names: default nothing\n");
   fprintf(stderr,"-N : if used, fit negative flux instead of positive\n");
   fprintf(stderr,"-q : if 1, scale model fluxes to match the total flux of detected island: default 0\n");
   fprintf(stderr,"Report bugs to <sarod@users.sf.net>\n");
}


void
print_copyright(void) {
  printf("Buildsky 0.1.0 (C) 2011-2016 Sarod Yatawatta\n");
}


/* for getopt() */
extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char **argv) {
  GHashTable *pixtable;
  int c;
  char *ffile,*mfile,*ignfile,*unistr;

  GList *ignlist=NULL; /* list of islands to ignore */
  GList *ignli; 
  FILE *cfp;
  int ignc, *ignp;

  double threshold=0.0;
  int use_em=1;
  int donegative=0;
  int maxiter=1000;
  int maxemiter=4;
  int outformat=0; /* 0: BBS, 1: LSM */
  /* for parsing integers */
  int base=10;
  char *endptr;
  int beam_given=0;
  double bmaj=0.001;
  double bmin=0.001;
  double bpa=0.0;
  double minpix;
  double clusterratio=1.0;
  int maxfits=10;
  int multifits=0;
  int nclusters=0;
  double wcutoff=0.0;

  int scaleflux=0; /* if 1, scale flux to match the total flux of island */

  /* for multiple FITS files */
  int Nf=0;
  double *freqs,*bmajs,*bmins,*bpas;
  double ref_freq;


  print_copyright();
  ffile=mfile=ignfile=unistr=0;
  if (argc<2) {
    print_help();
    return 1;
  }
  while ((c=getopt(argc,argv,"a:b:c:d:e:f:g:i:k:l:m:o:p:q:s:t:w:Nnh"))!=-1) {
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
    case 'd':
     if (optarg) {
      ffile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( ffile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(ffile,(char*)optarg);
      multifits=1;
     }
   break;
   case 'm':
     if (optarg) {
      mfile=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( mfile== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(mfile,(char*)optarg);
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
   case 's':
     if (optarg) {
      unistr=(char*)calloc((size_t)strlen((char*)optarg)+1,sizeof(char));
      if ( unistr== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
      }
      strcpy(unistr,(char*)optarg);
     }
    break;
   case 't':
     if (optarg) {
       threshold=strtod(optarg,0);
     }
    break;
   case 'c':
     if (optarg) {
       clusterratio=strtod(optarg,0);
     }
    break;
   case 'i':
     if (optarg) {
       maxiter=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) maxiter=1000;
     }
    break;
   case 'k':
     if (optarg) {
       nclusters=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) nclusters=0;
     }
    break;
   case 'l':
     if (optarg) {
       maxfits=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) maxfits=10;
     }
    break;
   case 'o':
     if (optarg) {
       outformat=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) outformat=0;
     }
    break;
   case 'e':
     if (optarg) {
       maxemiter=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) maxemiter=1000;
     }
    break;
   case 'q':
     if (optarg) {
       scaleflux=(int)strtol(optarg,&endptr,base);
       if (endptr==optarg) scaleflux=0;
     }
    break;
   case 'a':
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
   case 'b':
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
       bpa=strtod(optarg,0);
       if(!bpa) { bpa=0.01; }
       else {
       /* convert deg to rad */
        bpa=(bpa)/180.0*M_PI;
       }
       beam_given=1;
     }
   break;
   case 'n':
    use_em=0;
    break;
   case 'N':
    donegative=1;
    break;
   case 'w':
     if (optarg) {
       wcutoff=strtod(optarg,0);
     }
     if (wcutoff<0.0) { wcutoff=0.0; }
    break;
   case 'h':
    print_help();
    return 1;
   default:
    print_help();
    break;
  }
 }


  if (ffile && mfile) {
   if (!unistr) {
     unistr=(char*)calloc((size_t)1,sizeof(char));
     if ( unistr== 0 ) {
       fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
       exit(1);
     }
     unistr[0]='\0';
   }

   /* use a new random seed */
   srand(time(0));
   /* build ignore list, if given */
   if (ignfile) {
     if ((cfp=fopen(ignfile,"r"))==0) {
      fprintf(stderr,"%s: %d: no file %s\n",__FILE__,__LINE__,ignfile);
      exit(1);
     }
     do {
      c=fscanf(cfp,"%d",&ignc);
      if (c>0) {
        if ((ignp= (int*)malloc(sizeof(int)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
        }
        *ignp=ignc;
        ignlist=g_list_prepend(ignlist,ignp);
      }
     } while (c>= 0);

    printf("Ignore %d islands\n",g_list_length(ignlist));
    fclose(cfp);
   }
   if (!multifits) {
    read_fits_file(ffile,mfile, &pixtable,&bmaj,&bmin,&bpa, beam_given, &minpix, ignlist);
   } else {
    freqs=bmajs=bmins=bpas=0;
    read_fits_file_f(ffile,mfile, &pixtable,&Nf,&freqs,&bmajs,&bmins,&bpas, beam_given, &minpix, ignlist,donegative);
   }
   /* dont need ignore list anymore */
   for(ignli=ignlist; ignli!=NULL; ignli=g_list_next(ignli)) {
        ignp=ignli->data;
        g_free(ignp);
   }
   g_list_free(ignlist);
   /* filter pixel blobs, remove dodgy ones  */
   if (wcutoff>0.0) {
    if (!multifits ) {
     filter_pixels(pixtable,wcutoff);
    } else {
     filter_pixels_f(pixtable,wcutoff);
    }
    /* if filter is on, quit now */
    printf("quitting. re-run without filter\n");
    free(unistr);
    return 0;
   }
   if (!multifits) {
    process_pixels(pixtable,bmaj,bmin,bpa,minpix,threshold,maxiter,maxemiter,use_em,maxfits);
    write_world_coords(ffile,pixtable,minpix,bmaj,bmin,bpa,outformat,clusterratio,nclusters,unistr,scaleflux);
   } else {
     process_pixels_f(pixtable,Nf,freqs,bmajs,bmins,bpas,&ref_freq,minpix,threshold,maxiter,maxemiter,use_em,maxfits);
     write_world_coords_f(mfile,pixtable,minpix,Nf,freqs,bmajs,bmins,bpas,ref_freq,outformat,clusterratio,nclusters,unistr,donegative, scaleflux);
     free(freqs);
     free(bmajs);
     free(bmins);
     free(bpas);
   }
   g_hash_table_destroy(pixtable);
   free(ffile);
   free(mfile);
   free(ignfile);
   free(unistr);
  } else {
   print_help();
  }
  return 0;
}
