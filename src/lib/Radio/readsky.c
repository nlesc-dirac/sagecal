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


#include "Radio.h"
#include <ctype.h>

//#define DEBUG

/* key destroy function */
static void
destroy_hash_key(gpointer data) {
 free((char*)data);
}
/* value destroy function */
static void
destroy_hash_value(gpointer data) {
 sinfo_t *ss=(sinfo_t*)data;
 free(ss);
}

/* skips comment lines */
static int
skip_lines(FILE *fin)
{

  int c;
  do {
  if ( ( c = getc(fin) ) == EOF )
    return(-1);
  /* handle empty lines */
  if ( c == '\n' )
  continue; /* next line */
  if ( (c != '#') ) {
  ungetc(c,fin);
  return(0);
  } else { /* skip this line */
  do {
  if ( ( c = getc(fin) ) == EOF )
  return(-1);
  } while (  c != '\n') ;
  }
  } while( 1 );
}

/* skips rest of line */
static int
skip_restof_line(FILE *fin)
{
  int c;
  do {
  if ( ( c = getc(fin) ) == EOF )
  return(-1);
  } while (  c != '\n') ;
  return(1);
}


/* reads the next string (isalphanumeric() contiguous set of characters)
  separated by spaces, tabs or a newline. If the last character read is newline
  1 is returned, else 0 returned. */
/* buffer is automatically adjusted is length is not enough */
static int
read_next_string(char **buff, int *buff_len, FILE *infd) {
   int k,c,flag;
   k = 0;
   /* intialize buffer */
   (*buff)[0]='\0';
   /* skip leading white space */
   do {
   c=fgetc(infd);
   /* also handle DOS end of line \r\n */
   if(c=='\n' || c=='\r' || c==EOF) return 1;
   } while(c != EOF && isblank(c));
   if(c=='\n' || c=='\r' || c==EOF) return 1;
   /* now we have read a non whitespace character */
   (*buff)[k++]=c;
  if (k==*buff_len) {
    /* now we have run out of buffer */
    *buff_len += 30;
    if ((*buff = (char*)realloc((void*)(*buff),sizeof(char)*(size_t)(*buff_len)))==NULL) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
    }
  }
   flag=0;
   while ( ((c = fgetc(infd)) != EOF ) && k < *buff_len) {
      if ( c == '\n' || c=='\r' ) {  flag=1; break; }
      if ( isblank(c) ) {  break; }/* not end of line */
      (*buff)[k++] = c;
      if (k==*buff_len) {
       /* now we have run out of buffer */
       *buff_len += 30;
       if((*buff = (char*)realloc((void*)(*buff),sizeof(char)*(size_t)(*buff_len)))==NULL) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
       }
      }
  }
  /* now c == blank , \n or EOF */
  if (k==*buff_len-1) {
    /* now we have run out of buffer */
    *buff_len += 2;
    if((*buff = (char*)realloc((void*)(*buff),sizeof(char)*(size_t)(*buff_len)))==NULL) {
    fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
    exit(1);
   }
  }

  /* add '\0' to end */
  (*buff)[k++]='\0';
  return flag;
}




/* read shapalet mode file and build model */
/* buff: source name, mode file will be buff.fits.modes
   n0: model order, total modes will be n0*n0
   beta: scale
   modes: n0*n0 array of model parameters, memory will be allocated
*/ 
static int
read_shapelet_modes(char *buff,int *n0,double *beta,double **modes) {
  char *input_modes;
  int c,M,ci;
  double ra_s,dec_s;
  int ra_h,ra_m,dec_d,dec_m;

  FILE *cfp;
  if((input_modes= (char*)malloc(sizeof(char)*(size_t)(strlen(buff)+strlen(".fits.modes")+1)))==NULL) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
  }
  strcpy(input_modes,buff);
  strcpy((char*)&(input_modes[strlen(buff)]),".fits.modes\0");
  if ((cfp=fopen(input_modes,"r"))==0) {
      fprintf(stderr,"%s: %d: no file %s\n",__FILE__,__LINE__,input_modes);
      exit(1);
  }

  /* read RA, Dec: ignored */
  c=fscanf(cfp,"%d %d %lf %d %d %lf",&ra_h,&ra_m,&ra_s,&dec_d,&dec_m,&dec_s);

  /* read modes, beta */
  c=fscanf(cfp,"%d %lf",n0,beta);

  /* there are n0*n0 values for modes */
  M=(*n0)*(*n0);
  if ((*modes=(double*)calloc((size_t)M,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  int retval;
  for (ci=0; ci<M; ci++) {
   retval=fscanf(cfp,"%d %lf",&c,&(*modes)[ci]);
   if (retval==EOF) {
      fprintf(stderr,"%s: %d: read error %s\n",__FILE__,__LINE__,input_modes);
      exit(1);
   }
  }

  fclose(cfp);
  free(input_modes);
  return 0;
} 


int
read_sky_cluster(const char *skymodel, const char *clusterfile, clus_source_t **carr, int *M, double freq0, double ra0, double dec0, int format) {

  FILE *cfp;
  int c,buff_len,ci,cj;
  char *buff;

  GList *clusters;
  clust_t *clus;
  clust_n *sclus;

  GList *li,*ln;

  double sI,sQ,sU,sV,rahr,ramin,rasec,decd,decmin,decsec,spec_idx,spec_idx1,spec_idx2,dummy_RM,eX,eY,eP,f0;
  double fratio,fratio1,fratio2;
  double myra,mydec;

  exinfo_gaussian *exg;
  exinfo_disk *exd;
  exinfo_ring *exr;
  exinfo_shapelet *exs;

  double nn,xi,phi;

  GHashTable *stable;
  char *hkey;
  sinfo_t *source;

  /* first read the cluster file, construct a list of clusters*/
  /* each element of list is a list of source names */
  if ((cfp=fopen(clusterfile,"r"))==0) {
      fprintf(stderr,"%s: %d: no file %s\n",__FILE__,__LINE__,clusterfile);
      exit(1);
  }
  /* allocate memory for buffer */
  buff_len = MAX_SNAME; /* handle long names */
  if((buff = (char*)malloc(sizeof(char)*(size_t)(buff_len)))==NULL) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
  }


  clusters=NULL;
  c=skip_lines(cfp);
  while(c>=0) {
    /* we have a new line */
    memset(buff,0,buff_len);
    /* first read cluster number */
    c=read_next_string(&buff,&buff_len,cfp);
    clus=NULL;
    if (c!=1) {
     /* new cluster found */
     if ((clus= (clust_t*)malloc(sizeof(clust_t)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
     }
     sscanf(buff,"%d",&clus->id);
     clus->slist=NULL;
    }
    /* next read no of chunks */
    memset(buff,0,buff_len);
    c=read_next_string(&buff,&buff_len,cfp);
    sscanf(buff,"%d",&clus->nchunk);

    while (c!=1) { 
     memset(buff,0,buff_len);
     c=read_next_string(&buff,&buff_len,cfp);
     if (strlen(buff)>0) {
      /* source found for this cluster */
      if ((sclus= (clust_n*)malloc(sizeof(clust_n)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
      }     
      if ((sclus->name=(char*)malloc((size_t)(strlen(buff)+1)*sizeof(char)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
      }
      strcpy(sclus->name,buff);
      clus->slist=g_list_prepend(clus->slist,sclus);
     }
    }

    /* add this cluster */
    clusters=g_list_prepend(clusters,clus);
    c=skip_lines(cfp);
  }
  fclose(cfp);


  /* now read the sky model */
  /* format: LSM format */
  /* ### Name  | RA (hr,min,sec) | DEC (deg,min,sec) | I | Q | U | V | SI | RM | eX (rad) | eY (rad) | eP (rad) | ref_freq */
  /* NAME first letter : G/g Gaussian
      D/d : disk
      R/r : ring
      S/s : shapelet
      else: point
  */
  if ((cfp=fopen(skymodel,"r"))==0) {
      fprintf(stderr,"%s: %d: no file %s\n",__FILE__,__LINE__,skymodel);
      exit(1);
  }

  if ((buff = (char*)realloc((void*)(buff),sizeof(char)*(size_t)(MAX_SNAME)))==NULL) {
     fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  stable=g_hash_table_new_full(g_str_hash,g_str_equal,destroy_hash_key,destroy_hash_value);
  c=skip_lines(cfp);
  while(c>=0) {
    if (format==0) {
    c=fscanf(cfp,"%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",buff,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&sI,&sQ,&sU,&sV,&spec_idx,&dummy_RM,&eX,&eY,&eP, &f0);
    spec_idx1=spec_idx2=0.0;
    } else { /* 3 order spectral idx */
    c=fscanf(cfp,"%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",buff,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&sI,&sQ,&sU,&sV,&spec_idx,&spec_idx1,&spec_idx2,&dummy_RM,&eX,&eY,&eP, &f0);
    }

    /* add this to hash table */
    if (c!=EOF && c>0) {
      if ((hkey=(char*)malloc((size_t)(strlen(buff)+1)*sizeof(char)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
      }    
      strcpy(hkey,buff);
      if ((source=(sinfo_t*)malloc(sizeof(sinfo_t)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
      }      
      /* calculate l,m */
      /* Rad=(hr+min/60+sec/60*60)*pi/12 */
      if (rahr<0.0) {
        myra=-(-rahr+ramin/60.0+rasec/3600.0)*M_PI/12.0;
      } else {
        myra=(rahr+ramin/60.0+rasec/3600.0)*M_PI/12.0;
      }
      /* Rad=(hr+min/60+sec/60*60)*pi/180 */
      if (decd<0.0) {
        mydec=-(-decd+decmin/60.0+decsec/3600.0)*M_PI/180.0;
      } else {
        mydec=(decd+decmin/60.0+decsec/3600.0)*M_PI/180.0;
      }
      /* convert to l,m: NOTE we use -l here */
      source->ll=cos(mydec)*sin(myra-ra0);
      source->mm=sin(mydec)*cos(dec0)-cos(mydec)*sin(dec0)*cos(myra-ra0);
      source->ra=myra;
      source->dec=mydec;
      
      /* use spetral index, if != 0, to update sI to match data freq */
      if (spec_idx!=0.0) {
       fratio=log(freq0/f0);
       fratio1=fratio*fratio;
       fratio2=fratio1*fratio;
       /* catch -ve and 0 sI */
       if (sI>0.0) {
        source->sI[0]=exp(log(sI)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2);
       } else {
        source->sI[0]=(sI==0.0?0.0:-exp(log(-sI)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2));
       }
       if (sQ>0.0) {
        source->sI[1]=exp(log(sQ)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2);
       } else {
        source->sI[1]=(sQ==0.0?0.0:-exp(log(-sQ)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2));
       }
       if (sU>0.0) {
        source->sI[2]=exp(log(sU)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2);
       } else {
        source->sI[2]=(sU==0.0?0.0:-exp(log(-sU)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2));
       }
       if (sV>0.0) {
        source->sI[3]=exp(log(sV)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2);
       } else {
        source->sI[3]=(sV==0.0?0.0:-exp(log(-sV)+spec_idx*fratio+spec_idx1*fratio1+spec_idx2*fratio2));
       }

      } else {
       source->sI[0]=sI;
       source->sI[1]=sQ;
       source->sI[2]=sU;
       source->sI[3]=sV;
      }
      source->sI0[0]=sI; /* original sI */
      source->sI0[1]=sQ; /* original sQ */
      source->sI0[2]=sU; /* original sU */
      source->sI0[3]=sV; /* original sV */
      source->f0=f0;
      source->spec_idx=spec_idx;
      source->spec_idx1=spec_idx1;
      source->spec_idx2=spec_idx2;
      
      /* correction for projection, only for extended sources */
      /* calculate n */
      nn=sqrt(1.0-source->ll*source->ll-source->mm*source->mm);
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
       //printf("nn=%lf\n",nn);
       phi=acos(nn);
       xi=atan2(-source->ll,source->mm);

      /* determine source type */
      if (buff[0]=='G' || buff[0]=='g') {
       source->stype=STYPE_GAUSSIAN;
       if((exg=(exinfo_gaussian *)malloc(sizeof(exinfo_gaussian)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
       } 
       exg->eX=2.0*eX; /* scale by 2 */
       exg->eY=2.0*eY;
       exg->eP=eP;
       /* negate angles */
       exg->cxi=cos(xi);
       exg->sxi=sin(-xi);
       exg->cphi=cos(phi);
       exg->sphi=sin(-phi);
       if (nn<PROJ_CUT) {
         /* only then consider projection */
         exg->use_projection=1;
       } else {
         exg->use_projection=0;
       }
       source->exdata=(void*)exg;

      } else if (buff[0]=='D' || buff[0]=='d') {
       source->stype=STYPE_DISK;
       if((exd=(exinfo_disk*)malloc(sizeof(exinfo_disk)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
       } 
       exd->eX=eX;
       /* negate angles */
       exd->cxi=cos(xi);
       exd->sxi=sin(-xi);
       exd->cphi=cos(phi);
       exd->sphi=sin(-phi);
       if (nn<PROJ_CUT) {
         /* only then consider projection */
         exd->use_projection=1;
       } else {
         exd->use_projection=0;
       }
       source->exdata=(void*)exd;

      } else if (buff[0]=='R' || buff[0]=='r') {
       source->stype=STYPE_RING;
       if((exr=(exinfo_ring*)malloc(sizeof(exinfo_ring)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
       } 
       exr->eX=eX;
       /* negate angles */
       exr->cxi=cos(xi);
       exr->sxi=sin(-xi);
       exr->cphi=cos(phi);
       exr->sphi=sin(-phi);
       if (nn<PROJ_CUT) {
         /* only then consider projection */
         exr->use_projection=1;
       } else {
         exr->use_projection=0;
       }
       source->exdata=(void*)exr;

      } else if (buff[0]=='S' || buff[0]=='s') {
       source->stype=STYPE_SHAPELET;
       if((exs=(exinfo_shapelet*)malloc(sizeof(exinfo_shapelet)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
       } 
       exs->eX=eX;
       exs->eY=eY;
       /* sanity check if eX !=0 and eY !=0 */
       if (!exs->eX) {
         exs->eX=1.0;
         fprintf(stderr,"Warning: shapelet %s eX is zero. resetting to 1\n",buff);
       }
       if (!exs->eY) {
         exs->eY=1.0;
         fprintf(stderr,"Warning: shapelet %s eY is zero. resetting to 1\n",buff);
       }
       exs->eP=eP;
       /* open mode file and build up info */
       read_shapelet_modes(buff,&exs->n0,&exs->beta,&exs->modes);
       
       /* negate angles */
       exs->cxi=cos(xi);
       exs->sxi=sin(-xi);
       exs->cphi=cos(phi);
       exs->sphi=sin(-phi);
       if (nn<PROJ_CUT) {
         /* only then consider projection */
         exs->use_projection=1;
       } else {
         exs->use_projection=0;
       }
       source->exdata=(void*)exs;
 
      } else {
       source->stype=STYPE_POINT;
       source->exdata=NULL;
      }

      g_hash_table_insert(stable,(gpointer)hkey,(gpointer)source);
    }
    c=skip_restof_line(cfp);
    c=skip_lines(cfp);
  }
  fclose(cfp);
  free(buff);

  *M=g_list_length(clusters);
  /* setup the array of cluster/source information */
  if ((*carr=(clus_source_t*)malloc((size_t)(g_list_length(clusters))*sizeof(clus_source_t)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  } 
  
  ci=0;
  for(li=clusters; li!=NULL; li=g_list_next(li)) {
    clus=li->data;
#ifdef DEBUG
    printf("cluster %d has %d elements\n",clus->id,g_list_length(clus->slist));
#endif

    /* remember id, because -ve ids are not subtracted */
    (*carr)[ci].id=clus->id;
    (*carr)[ci].nchunk=clus->nchunk;
    (*carr)[ci].N=g_list_length(clus->slist);

    if (((*carr)[ci].ll=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].mm=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].nn=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].sI=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].sQ=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].sU=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].sV=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].stype=(unsigned char*)malloc((size_t)((*carr)[ci].N)*sizeof(unsigned char)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].ex=(void**)malloc((size_t)((*carr)[ci].N)*sizeof(void*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* for handling multi channel data */
    if (((*carr)[ci].sI0=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].sQ0=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].sU0=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].sV0=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].f0=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].spec_idx=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].spec_idx1=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].spec_idx2=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].ra=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if (((*carr)[ci].dec=(double*)malloc((size_t)((*carr)[ci].N)*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }


    cj=0;
    for(ln=clus->slist; ln!=NULL; ln=g_list_next(ln)) {
     sclus=ln->data;
#ifdef DEBUG
     printf(" %s",sclus->name);
#endif
     /* lookup hash table */
     source=NULL;
     source=(sinfo_t*)g_hash_table_lookup(stable,sclus->name);
     if (source) {
      (*carr)[ci].ll[cj]=source->ll;
      (*carr)[ci].mm[cj]=source->mm;
      (*carr)[ci].nn[cj]=sqrt(1.0-source->ll*source->ll-source->mm*source->mm)-1.0;
      (*carr)[ci].sI[cj]=source->sI[0];
      (*carr)[ci].sQ[cj]=source->sI[1];
      (*carr)[ci].sU[cj]=source->sI[2];
      (*carr)[ci].sV[cj]=source->sI[3];
      (*carr)[ci].stype[cj]=source->stype;
#ifdef DEBUG
      printf(" (%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf)",source->ll,source->mm,source->ra,source->dec,(*carr)[ci].nn[cj],source->sI[0],source->sI[1],source->sI[2],source->sI[3]);
#endif
      (*carr)[ci].ex[cj]=source->exdata; /* FIXME: duplicate sources could create double free error */
      (*carr)[ci].ra[cj]=source->ra;
      (*carr)[ci].dec[cj]=source->dec;
      
      /* for multi channel data */
      (*carr)[ci].sI0[cj]=source->sI0[0];
      (*carr)[ci].sQ0[cj]=source->sI0[1];
      (*carr)[ci].sU0[cj]=source->sI0[2];
      (*carr)[ci].sV0[cj]=source->sI0[3];
      (*carr)[ci].f0[cj]=source->f0;
      (*carr)[ci].spec_idx[cj]=source->spec_idx;
      (*carr)[ci].spec_idx1[cj]=source->spec_idx1;
      (*carr)[ci].spec_idx2[cj]=source->spec_idx2;
      cj++;
     } else {
      fprintf(stderr,"Error: source %s not found\n",sclus->name);
     }
    }
    /* sanity check */
    if (cj!=(*carr)[ci].N) {
      fprintf(stderr,"Error: Expected %d no of sources for cluster %d but found %d, check your sky model!\nError: Continuing anyway but will get wrong results.\n",(*carr)[ci].N,*M-ci,cj);
    }
//    printf("\n");
    ci++;
  }


  /* FIXME: free source info for sources that were not part of the clusters */
  /* free cluster data */
  for(li=clusters; li!=NULL; li=g_list_next(li)) {
    clus=li->data;
    for(ln=clus->slist; ln!=NULL; ln=g_list_next(ln)) {
     sclus=ln->data;
     free(sclus->name);
     free(sclus);
    }
    g_list_free(clus->slist);
    free(clus);
  }
  g_list_free(clusters);
  g_hash_table_destroy(stable);
 return 0;
}



int
read_solutions(FILE *cfp,double *p,clus_source_t *carr,int N,int M) {
 /* read 8N valid rows and Mt columns */
 int Nc=8*N-1;
 int c,buff_len,ci,ck,cn;
 double jtmp;
 char *buf;
 /* allocate memory for buffer */
 buff_len = 128;
 if((buf = (char*)malloc(sizeof(char)*(size_t)(buff_len)))==NULL) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
 }
#ifdef DEBUG
printf("Nc=%d\n",Nc);
#endif
 c=skip_lines(cfp);
 while(Nc>=0 && c>=0) {
    /* we have a new line */
    memset(buf,0,buff_len);
    c=read_next_string(&buf,&buff_len,cfp);
    if (c!=1) {
     /* first column is solution number (int) 0..8N-1 */
     sscanf(buf,"%d",&cn);
     /* also do a sanity check */
     if ( cn<0 || cn>8*N-1 ) cn=0;
    }
#ifdef DEBUG
    printf("%d ",cn);
#endif
    /* read the rest of the line */
    for (ci=M-1; ci>=0; ci--) {
        for (ck=0; ck<carr[ci].nchunk; ck++) {
        if (strlen(buf)>0)  {
         memset(buf,0,buff_len);
         c=read_next_string(&buf,&buff_len,cfp);
          sscanf(buf,"%lf",&jtmp);
          p[carr[ci].p[ck]+cn]=jtmp;
#ifdef DEBUG
          printf("%e ",jtmp);
#endif
        }
       }
    }
#ifdef DEBUG
    printf("\n"); 
#endif
    c=skip_lines(cfp);
    Nc--;
 }
 /* if Nc>=0 and we have reached the EOF, something wrong with solution file
   so display warning */
 if (Nc>=0) {
  fprintf(stderr,"Warning: solution file EOF reached, check your solution file\n");
  exit(1);
 } 

 free(buf);
 return 0;
}


int
update_ignorelist(const char *ignfile, int *ignlist, int M, clus_source_t *carr) {
    FILE *cfp;
    int ci,c,ignc,cn;
    if ((cfp=fopen(ignfile,"r"))==0) {
      fprintf(stderr,"%s: %d: no file %s\n",__FILE__,__LINE__,ignfile);
      exit(1);
    }
    cn=0;
    do {
      c=fscanf(cfp,"%d",&ignc);
      if (c>0) {
#ifdef DEBUG
      printf("searching for %d\n",ignc);
#endif
        /* search for this id in carr */
        for (ci=0; ci<M; ci++) {
         if (carr[ci].id==ignc) {
            ignlist[ci]=1;
            cn++;
#ifdef DEBUG
            printf("cid=%d found at %d\n",ignc,ci);
#endif
            break;
         }
        }
      }
    } while (c>= 0);

    fclose(cfp);
    printf("Total %d clusters ignored in simulation.\n",cn);
    return 0;
}




int
read_arho_fromfile(const char *admm_rho_file,int Mt,double *arho, int M, double *arhoslave) {

  FILE *cfp;
  int c,ci,cj,cluster_id,hybrid,hb;
  double admm_rho;
  if ((cfp=fopen(admm_rho_file,"r"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      exit(1);
  }

  c=skip_lines(cfp);
  ci=0; /* store it in reverse order */
  cj=0;
//printf("Mt=%d M=%d\n",Mt,M);
  while(c>=0 && cj<M) {
    c=fscanf(cfp,"%d %d %lf",&cluster_id,&hybrid,&admm_rho);
//printf("c=%d ci=%d cj=%d\n",c,ci,cj);
    /* add this value to arho array */
    if (c!=EOF && c>0) {
      /* found a valid line */
      arhoslave[M-1-cj]=admm_rho; /* reverse order */
      /* FIXME: for hybrid>1, since data is divided by that amount, 
         the actual value of rho should be also reduced ??? */
      for (hb=0; hb<hybrid; hb++) {
        if (hb==0) {
         arhoslave[M-1-cj]=admm_rho; /* reverse order */
         //printf("clus=%d arr=%d rhoslave=%lf\n",cluster_id,M-1-cj,admm_rho);
         cj++;
        }
        arho[Mt-1-ci]=admm_rho; /* reverse order */
        //printf("clus=%d arr=%d rho=%lf\n",cluster_id,Mt-1-ci,admm_rho);
        if (ci<Mt-1) {
         ci++;
        } else {
         /* array size does not match one given by text file */
         break;
        }
      }
    }
    c=skip_restof_line(cfp);
    c=skip_lines(cfp);
  }
//printf("c=%d ci=%d cj=%d\n",c,ci,cj);
  /* report any errors */
  if (!(c==EOF && ci==Mt-1 && cj==M)) {
    fprintf(stderr,"%s: %d: Error: cluster numbers in cluster file and regularization file do not match up.\n",__FILE__,__LINE__);
    exit(1);
  }
  fclose(cfp);


 return 0;
}
