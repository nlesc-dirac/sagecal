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


#include <ctype.h>
#include <complex.h>
#include "restore.h"
#include <glib.h>


static void
matprod_JHJ(double aa[8],double a0,double a1,double a2,double a3,double a4,double a5,double a6,double a7) {
 complex double a,b,c,d,e00,e01,e10,e11;
 a=a0+_Complex_I*a1;
 b=a2+_Complex_I*a3;
 c=a4+_Complex_I*a5;
 d=a6+_Complex_I*a7;
 /* [a b; c d]^H [a b; c d] */
 e00=conj(a)*a+conj(c)*c;
 e01=conj(a)*b+conj(c)*d;
 e10=conj(b)*a+conj(d)*c;
 e11=conj(b)*b+conj(d)*d;
 aa[0]=creal(e00);
 aa[1]=cimag(e00);
 aa[2]=creal(e01);
 aa[3]=cimag(e01);
 aa[4]=creal(e10);
 aa[5]=cimag(e10);
 aa[6]=creal(e11);
 aa[7]=cimag(e11);
}

/* compare sources, on their location */
static int
listcomp(const void *s1, const void *s2) {
  sinfo *e1=(sinfo *)s1;
  sinfo *e2=(sinfo *)s2;
  if ((e1->ra==e2->ra) && (e1->dec==e2->dec)) return 0;
  return 1;
}
static void
listfree(void * d)
{
 sinfo *e=(sinfo*)d;
 exinfo_shapelet *ex;
 /* if type is shapelet, free file name */
 if (e->type==STYPE_SHAPELET) { 
  ex=(exinfo_shapelet*)e->exdata;
  free(ex->modes); 
  free(ex);
 }
 free(e);
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

/* read shapalet mode file and build model */
/* buff: source name, mode file will be buff.fits.modes
   n0: model order, total modes will be n0*n0
   beta: scale
   modes: n0*n0 array of model parameters, memory will be allocated
   eX,eY,eP : linear transform parameters, if any
   ltf: linear transform used ?
*/
static int
read_shapelet_modes_img(char *buff,int *n0,double *beta,double **modes, double *eX, double *eY, double *eP, int *ltf) {
  char *input_modes;
  int c,M,ci;
  double ra_s,dec_s;
  int ra_h,ra_m,dec_d,dec_m;
  double l_a,l_b,l_theta;
  int linear_tf=0;

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
  /* downscale beta */
  *beta /=2.0*M_PI;

  /* there are n0*n0 values for modes */
  M=(*n0)*(*n0);
  if ((*modes=(double*)calloc((size_t)M,sizeof(double)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  for (ci=0; ci<M; ci++) {
   c=fscanf(cfp,"%d %lf",&c,&(*modes)[ci]);
  }

  /* need to find a line starting with 'L' */
  c=skip_lines(cfp);
  while(c>=0) {
    if ((c=(getc(cfp))=='L')) {
     /* found line starting with L, parse this line */
     c=fscanf(cfp,"%lf %lf %lf",&l_a,&l_b,&l_theta);
     if (c!=EOF && l_a>0.0 && l_b>0.0) {
       linear_tf=1;
       l_theta-=M_PI/2;
       printf("Linear Transform a=%lf, b=%lf, theta=%lf\n",l_a,l_b,l_theta);
     }
    }
    c=skip_lines(cfp);
  }

  /* double check linear TF, ignore default values */
  if (linear_tf && l_a==1.0 && l_b==1.0 && l_theta==0.0) {
    linear_tf=0;
  } 
  *eX=l_a; *eY=l_b; *eP=l_theta; *ltf=linear_tf;

  fclose(cfp);
  free(input_modes);
  return 0;
}


/* f0: ref. freq, 
   t0,t1: spectral idx parameters (normally t1=0) within a [ ]
*/
static int
read_bbs_skyline(FILE *cfp, char *buf, double *rahr, double *ramin, double *rasec, double *decd, double *decmin, double *decsec, double *sI, double *sQ, double *sU, double *sV, int *stype, double *eX, double *eY, double *eP, double *f0, double *t0) {
      /* BBS format, comma delimiter */
      /*  (Name, Type, Ra, Dec, I, Q, U, V, .... */
      /*  Name, Type, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis,
Orientation, ReferenceFrequency, SpectralIndex= with '[]' */
      /* NOTE: no default values taken, for point sources
        major,minor,orientation has to be all zero  */
      /* P165C1, POINT, 7:58:15.92, +51.19.43.35, 0.532894, 0.0, 0.0, 0.0 */
/* Example
  # bmaj,bmin, Gaussian radius in degrees, bpa also in degrees
  Gtest1, GAUSSIAN, 18:59:16.309, -22.46.26.616, 100, 100, 100, 100, 0.222, 0.111, 100, 150e6, [-1.0]
  Ptest2, POINT, 18:59:20.309, -22.53.16.616, 100, 100, 100, 100, 0, 0, 0, 140e6, [-2.100] */

    int c;
    char type[128];
    char rastr[128];
    char decstr[128];
    char sIstr[128];
    char sQstr[128];
    char sUstr[128];
    char sVstr[128];
    char Majorstr[128];
    char Minorstr[128];
    char Orientstr[128];
    char f0str[128];
    char Sidxstr[128];
    char *typep;
    c=fscanf(cfp,"%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%127[^,],%s",buf,type,rastr,decstr,sIstr,sQstr,sUstr,sVstr,Majorstr,Minorstr,Orientstr,f0str,Sidxstr);
   printf("%s %s %s %s %s %s %s %s %s %s %s %s\n",type,rastr,decstr,sIstr,sQstr,sUstr,sVstr,Majorstr,Minorstr,Orientstr,f0str,Sidxstr);
   typep=&type[0];
   while(isspace(*typep)) typep++;
   if (typep[0]=='P' || typep[0]=='p') {
     *stype=STYPE_POINT;
   } else if  (typep[0]=='G' || typep[0]=='g') {
     *stype=STYPE_GAUSSIAN;
   } 
   sscanf(rastr,"%lf:%lf:%lf",rahr,ramin,rasec);
   /* split dec to smaller parts */
   char decdstr[16];
   char decminstr[16];
   char decsecstr[16];
   sscanf(decstr,"%15[^.].%15[^.].%s",decdstr,decminstr,decsecstr);
   sscanf(decdstr,"%lf",decd);
   sscanf(decminstr,"%lf",decmin);
   sscanf(decsecstr,"%lf",decsec);

   sscanf(sIstr,"%lf",sI);
   sscanf(sQstr,"%lf",sQ);
   sscanf(sUstr,"%lf",sU);
   sscanf(sVstr,"%lf",sV);
   sscanf(Majorstr,"%lf",eX);
   sscanf(Minorstr,"%lf",eY);
   /* bmaj,bmin are in degrees, convert to radians and radii */
   *eX *= (*eX)/180.0*M_PI;
   *eY *= (*eY)/180.0*M_PI;
   sscanf(Orientstr,"%lf",eP);
   sscanf(f0str,"%lf",f0);
   sscanf(Sidxstr,"[%lf]",t0);
return c;
}


int
read_sky_model(const char *slistname, glist *slist, int format) {

  FILE *cfp;
  int c;
  char buf[128];
  sinfo ss;
  exinfo_shapelet *exs;
  double rahr, ramin, rasec, decd, decmin, decsec;
  double dummy_RM;


   /* read, parse source list */
  glist_sorted_init(slist,sizeof(sinfo),&listfree,&listcomp);

  /* read in the source list, parse each source info */
  if ((cfp=fopen(slistname,"r"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      return 1;
  }

  c=skip_lines(cfp);
  while(c>=0) {
    if (format==FORMAT_BBS) {
      c=read_bbs_skyline(cfp,buf,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&ss.sI,&ss.sQ,&ss.sU,&ss.sV, &ss.type, &ss.eX, &ss.eY, &ss.eP, &ss.f0, &ss.spec_idx);
      printf("BBS: (%lf,%lf,%lf) (%lf,%lf,%lf) [%lf,%lf,%lf,%lf] (%lf,%lf,%lf) [%lf,%lf]\n",rahr,ramin,rasec,decd,decmin,decsec,ss.sI,ss.sQ,ss.sU,ss.sV,ss.eX,ss.eY,ss.eP,ss.f0,ss.spec_idx);
      ss.exdata=NULL;
      ss.spec_idx1=ss.spec_idx2=0.0;
    } else if (format==FORMAT_LM) {
     /* l,m format */
     /* NAME l m sI */
     c=fscanf(cfp,"%s %lf %lf %lf",buf,&ss.ra,&ss.dec,&ss.sI);
     ss.sQ=ss.sU=ss.sV=0.0;
     ss.eX=ss.eY=ss.eP=0.0;
     ss.spec_idx=ss.spec_idx1=ss.spec_idx2=0.0;
     ss.f0=10000.0;
     ss.type=STYPE_POINT;
     /* convert l,m radians to degrees */
     ss.ra *=(180.0/M_PI);
     ss.dec *=(180.0/M_PI);
     ss.exdata=NULL;
#ifdef DEBUG
     printf("l,m=%lf,%lf\n",ss.ra,ss.dec);
#endif
    } else if (format==FORMAT_LSM_SP) { /* new format */
      ss.exdata=NULL;
     /* LSM format */
    /* ### Name  | RA (hr,min,sec) | DEC (deg,min,sec) | I | Q | U | V | SI | RM | eX (rad) | eY (rad) | eP (rad) | ref_freq */
      c=fscanf(cfp,"%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",buf,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&ss.sI,&ss.sQ,&ss.sU,&ss.sV,&ss.spec_idx,&ss.spec_idx1,&ss.spec_idx2,&dummy_RM,&ss.eX,&ss.eY,&ss.eP, &ss.f0);
    /* use first letter of name to determine which type of source */
    if (buf[0]=='G' || buf[0]=='g') {
      ss.type=STYPE_GAUSSIAN;
    } else if (buf[0]=='D' || buf[0]=='d') {
      ss.type=STYPE_DISK;
    } else if (buf[0]=='R' || buf[0]=='r') {
      ss.type=STYPE_RING;
    } else if (buf[0]=='S' || buf[0]=='s') {
      ss.type=STYPE_SHAPELET;
      /* need mode file called "buf.fits.modes" */
      if ((exs=(exinfo_shapelet*)calloc(1,sizeof(exinfo_shapelet)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      /* open mode file and build up info */
      read_shapelet_modes_img(buf,&exs->n0,&exs->beta,&exs->modes,&exs->eX,&exs->eY,&exs->eP,&exs->linear_tf);
      ss.exdata=(void*)exs;
    } else {
      ss.type=STYPE_POINT;
    }
    } else { /* format==FORMAT_LSM */
      ss.exdata=NULL;
     /* LSM format */
    /* ### Name  | RA (hr,min,sec) | DEC (deg,min,sec) | I | Q | U | V | SI | RM | eX (rad) | eY (rad) | eP (rad) | ref_freq */
      c=fscanf(cfp,"%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",buf,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&ss.sI,&ss.sQ,&ss.sU,&ss.sV,&ss.spec_idx,&dummy_RM,&ss.eX,&ss.eY,&ss.eP, &ss.f0);
    /* use first letter of name to determine which type of source */
    if (buf[0]=='G' || buf[0]=='g') {
      ss.type=STYPE_GAUSSIAN;
    } else if (buf[0]=='D' || buf[0]=='d') {
      ss.type=STYPE_DISK;
    } else if (buf[0]=='R' || buf[0]=='r') {
      ss.type=STYPE_RING;
    } else if (buf[0]=='S' || buf[0]=='s') {
      ss.type=STYPE_SHAPELET;
      /* need mode file called "buf.fits.modes" */
      if ((exs=(exinfo_shapelet*)calloc(1,sizeof(exinfo_shapelet)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      /* open mode file and build up info */
      read_shapelet_modes_img(buf,&exs->n0,&exs->beta,&exs->modes,&exs->eX,&exs->eY,&exs->eP,&exs->linear_tf);
      ss.exdata=(void*)exs;
    } else {
      ss.type=STYPE_POINT;
    }
     ss.spec_idx1=ss.spec_idx2=0.0;
    }

#ifdef DEBUG
    printf("read %d items:  %s %lf %lf %lf %lf %lf %lf [%lf %lf %lf %lf] (%lf,%lf,%lf,%lf) [%lf,%lf,%lf]\n",c, buf,rahr,ramin,rasec,decd,decmin,decsec,ss.sI,ss.sQ,ss.sU,ss.sV,ss.spec_idx, ss.spec_idx1,ss.spec_idx2,ss.f0, ss.eX,ss.eY,ss.eP);
#endif

    if (format!=FORMAT_LM) {
    /* Rad=(hr+min/60+sec/60*60)*pi/12 */
    if (rahr<0) {
     ss.ra=-(-rahr+ramin/60.0+rasec/3600.0)*M_PI/12.0;
    } else {
     ss.ra=(rahr+ramin/60.0+rasec/3600.0)*M_PI/12.0;
    }
    /* Rad=(hr+min/60+sec/60*60)*pi/180 */
    if (decd<0) {
     ss.dec=-(-decd+decmin/60.0+decsec/3600.0)*M_PI/180.0;
    } else {
     ss.dec=(decd+decmin/60.0+decsec/3600.0)*M_PI/180.0;
    }
    }
    glist_sorted_insert(slist,(void*)&ss);
    c=skip_restof_line(cfp);
    c=skip_lines(cfp);
  }
  fclose(cfp);


 return 0;
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
   if(c=='\n' || c==EOF) return 1;
   } while(c != EOF && isblank(c));
   if(c=='\n' || c==EOF) return 1;
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
      if ( c == '\n') {  flag=1; break; }
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



/* struct for a cluster GList item */
typedef struct clust_t_{
 int id; /* cluster id */
 int nchunk; /* no. of time chunks (hybrid solution) */
 int ex,ey; /* valid soultions (columns in solution file for this cluster)
             0.....[ex...ey] ..... */
 GList *slist; /* list of sources in this cluster (string)*/
} clust_t;

typedef struct clust_n_{
 char *name; /* source name (string)*/
} clust_n;


/* hash table */
/* name (*char) ->id (int) */
/* key destroy function */
static void
destroy_hash_key(gpointer data) {
 free((char*)data);
}
/* value (pointer to clust_t) destroy function */
static void
destroy_hash_value(gpointer data) {
 free((clust_t**)data);
}


int
read_sky_model_withgain(const char *slistname, glist *slist, int format,const char *clusterfile, const char *solfile, const char *ignfile) {

  FILE *cfp;
  int c,buff_len;
  char *buf;
  sinfo ss;
  exinfo_shapelet *exs;
  double rahr, ramin, rasec, decd, decmin, decsec;
  double dummy_RM;


  GList *clusters;
  clust_t *clus;
  clust_t **cptr;
  clust_n *sclus;
  GList *li,*ln;

  GList *ignlist=NULL;
  GList *ignli;
  int ignc, *ignp;
  int igncnt,*ignidx;
  int ck,sta,use_this_line;


  GHashTable *stable;
  char *hkey;
  int cid;
  int ci,Nc,Ns,nt,Nt;
  
  complex double **JHJ,**JsHJ; /* arrays for storing solutions per cluster */
  double **Jf1,**Jf2; /* arrays for storing solutions per cluster */
  double aa[8];
  double jtmp;

  double sIxc,sQxc,sUxc,sVxc;


  /************** read in cluster file *************************************/
  /* first read the cluster file, construct a list of clusters*/
  /* each element of list is a list of source names */
  if ((cfp=fopen(clusterfile,"r"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      return 1;
  }
  /* allocate memory for buffer */
  buff_len = MAX_SNAME;
  if((buf = (char*)malloc(sizeof(char)*(size_t)(buff_len)))==NULL) {
        fprintf(stderr,"%s: %d: No free memory\n",__FILE__,__LINE__);
        exit(1);
  }

  clusters=NULL;
  c=skip_lines(cfp);
  ci=0; /* use this as index for colum numbers */
  while(c>=0) {
    /* we have a new line */
    memset(buf,0,buff_len);
    /* first read cluster number */
    c=read_next_string(&buf,&buff_len,cfp);
    clus=NULL;
    if (c!=1) {
     /* new cluster found */
     if ((clus=(clust_t*)malloc(sizeof(clust_t)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
     }
     sscanf(buf,"%d",&clus->id);
     clus->slist=NULL;
    }
#ifdef DEBUG
    printf("Cluster %s\n",buf);
#endif
    /* next item is hybrid chunk size */
    if (c!=1) { 
     memset(buf,0,buff_len);
     c=read_next_string(&buf,&buff_len,cfp);
     sscanf(buf,"%d",&clus->nchunk);
#ifdef DEBUG
     printf("Hybrid %s ",buf);
#endif
     /* store the valid colum ranges */
     clus->ex=ci;
     clus->ey=ci+clus->nchunk-1;
     /* increment column index */
     ci+=clus->nchunk;
    }
    while (c!=1) {
     memset(buf,0,buff_len);
     c=read_next_string(&buf,&buff_len,cfp);
     if (strlen(buf)>0) {
      /* source found for this cluster */
      if ((sclus= (clust_n*)malloc(sizeof(clust_n)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
      }
      if ((sclus->name=(char*)malloc((size_t)(strlen(buf)+1)*sizeof(char)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
      }
      strcpy(sclus->name,buf);
#ifdef DEBUG
      printf("%s ",buf);
#endif
      clus->slist=g_list_prepend(clus->slist,sclus);
     }
    }
#ifdef DEBUG
    printf("\n");
#endif
    /* add this cluster */
    clusters=g_list_prepend(clusters,clus);
    c=skip_lines(cfp);
  }
  fclose(cfp);
  printf("Total columns needed in solutions file=%d\n",ci);
  Nc=ci; /* this will be the effective no of clusters */

  /* build ignore list, if given */
  igncnt=0; ignidx=0;
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

    printf("Ignore %d stations\n",g_list_length(ignlist));
    fclose(cfp);
  } 
  igncnt=g_list_length(ignlist);
  if (igncnt>0) {
    if ((ignidx= (int*)malloc(sizeof(int)*(size_t)igncnt))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }
    ci=0;
    /* dont need ignore list anymore */
    for(ignli=ignlist; ignli!=NULL; ignli=g_list_next(ignli)) {
        ignp=ignli->data;
        ignidx[ci]=*ignp;
        ci++;
        g_free(ignp);
    }
    g_list_free(ignlist);
  }


  /* build up hash table to map source name -> cluster id */
  stable=g_hash_table_new_full(g_str_hash,g_str_equal,destroy_hash_key,destroy_hash_value);

  /* build up hash table */
  ci=0;
  for(li=clusters; li!=NULL; li=g_list_next(li)) {
    clus=li->data;
    for(ln=clus->slist; ln!=NULL; ln=g_list_next(ln)) {
     sclus=ln->data;
     if ((hkey=(char*)malloc((size_t)(strlen(sclus->name)+1)*sizeof(char)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
     }
     strcpy(hkey,sclus->name);
     if ((cptr=(clust_t**)malloc((size_t)(1)*sizeof(clust_t*)))==0) {
            fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
            exit(1);
     }
     *cptr=clus;
     g_hash_table_insert(stable,(gpointer)hkey,(gpointer)cptr);
    }
    ci++;
  }

  printf("Found %d (true) %d (effective) clusters\n",ci,Nc);

 /* first determine no of stations by counting 0....8*Ns-1 (first col)*/
  if ((cfp=fopen(solfile,"r"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      return 1;
  }
  /* for new versions of sagecal, need to skip first 3 lines */
  for (ci=0; ci<3; ci++) {
   do {
    c = fgetc(cfp);
   } while (c != '\n');
  }

  c=skip_lines(cfp);
  Ns=0;
  while(c>=0) {
    /* we have a new line */
    memset(buf,0,buff_len);
    c=read_next_string(&buf,&buff_len,cfp);
    if (c!=1) {
     /* first column is solution number (int) */
     sscanf(buf,"%d",&ci);
    }
    if (!Ns && !ci) {
     Ns++; /* first zero in ci */
    } else if (ci>0) {
     Ns++;
    } else {
     break;
    } 
    c=skip_restof_line(cfp);
    c=skip_lines(cfp);
  }
  Ns=Ns/8;
  printf("No of stations=%d->%d\n",Ns,Ns-igncnt);
  Ns=Ns-igncnt;
 
  fclose(cfp);
  /* memory to store solutions */
  if ((JHJ=(complex double**)calloc((size_t)(4),sizeof(complex double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((JsHJ=(complex double**)calloc((size_t)(4),sizeof(complex double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  for (ci=0; ci<4; ci++) {
   if ((JHJ[ci]=(complex double*)calloc((size_t)(Nc),sizeof(complex double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   if ((JsHJ[ci]=(complex double*)calloc((size_t)(Nc),sizeof(complex double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
  }
  if ((Jf1=(double**)calloc((size_t)(8),sizeof(double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((Jf2=(double**)calloc((size_t)(8),sizeof(double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  for (ci=0; ci<8; ci++) {
   if ((Jf1[ci]=(double*)calloc((size_t)(Nc),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   if ((Jf2[ci]=(double*)calloc((size_t)(Nc),sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
  }

 /* re-read the solution file */
 /* calculate sum(J_i)^H sum(J_i)-sum(J_i^H J_i)
   = J_1^H J_2+J_1^H J_3....+J_1 J_2^H+J_1 J_3^H....
 */
  if ((cfp=fopen(solfile,"r"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      exit(1);
  }
  /* for new versions of sagecal, need to skip first 3 lines */
  for (ci=0; ci<3; ci++) {
   do {
    c = fgetc(cfp);
   } while (c != '\n');
  }

  Nt=0;
  c=skip_lines(cfp);
  while(c>=0) {
    /* we have a new line */
    memset(buf,0,buff_len);
    c=read_next_string(&buf,&buff_len,cfp);
    if (c!=1) {
     /* first column is solution number (int) */
     sscanf(buf,"%d",&nt);
    }
    /* determine if this row belongs to a flagged station */
    use_this_line=1;
    if (igncnt>0) {
     sta=nt/8;
     for (ck=0; ck<igncnt; ck++) {
      if (ignidx[ck]==sta){
       use_this_line=0;
#ifdef DEBUG
       printf("station %d row %d ignored\n",sta,nt);
#endif
      }
     } 
    }
    /* if nt>0 and nt%8==0, new station has passed, so form J^H J for previous
       station : stored in Jf2 now */
    if (!(nt%8)) {
     for (ci=0; ci<Nc; ci++){
       matprod_JHJ(aa,Jf2[0][ci],Jf2[1][ci],Jf2[2][ci],Jf2[3][ci],Jf2[4][ci],Jf2[5][ci],Jf2[6][ci],Jf2[7][ci]);
       JHJ[0][ci]+=aa[0]+_Complex_I*aa[1];
       JHJ[1][ci]+=aa[2]+_Complex_I*aa[3];
       JHJ[2][ci]+=aa[4]+_Complex_I*aa[5];
       JHJ[3][ci]+=aa[6]+_Complex_I*aa[7];
     }
    } 
    /* if nt==0, new timeslot, so reset everything */
    if (!nt) {
      //printf("reset %d\n",nt);
     /* form sum(J_i)^H * J_i : stored in Jf1 */
     for (ci=0; ci<Nc; ci++){
       matprod_JHJ(aa,Jf1[0][ci],Jf1[1][ci],Jf1[2][ci],Jf1[3][ci],Jf1[4][ci],Jf1[5][ci],Jf1[6][ci],Jf1[7][ci]);
       JsHJ[0][ci]+=aa[0]+_Complex_I*aa[1];
       JsHJ[1][ci]+=aa[2]+_Complex_I*aa[3];
       JsHJ[2][ci]+=aa[4]+_Complex_I*aa[5];
       JsHJ[3][ci]+=aa[6]+_Complex_I*aa[7];
       
       //printf("J=[%lf+j*%lf, %lf+j*%lf; %lf+j*%lf, %lf+j*%lf] JJ=[%lf+j*%lf, %lf+j*%lf; %lf+j*%lf, %lf+j*%lf]\n",Jf1[0][ci],Jf1[1][ci],Jf1[2][ci],Jf1[3][ci],Jf1[4][ci],Jf1[5][ci],Jf1[6][ci],Jf1[7][ci],aa[0],aa[1],aa[2],aa[3],aa[4],aa[5],aa[6],aa[7]);
     }

     for (ci=0; ci<8; ci++) {
      memset(Jf1[ci],0,(size_t)Nc*sizeof(double));
     }
     Nt++; /* total no of time slots */
    }
    ci=0; /* cluster number */
    while (c!=1) {
     memset(buf,0,buff_len);
     c=read_next_string(&buf,&buff_len,cfp);
     if (strlen(buf)>0 && ci<Nc)  {
       sscanf(buf,"%lf",&jtmp);
       /* only accumulate if this station is not ignored */
       if (use_this_line) {
        Jf1[nt%8][ci]+=jtmp; /* sum(J_i) */
        Jf2[nt%8][ci]=jtmp; /* J_i */
       }
     }
     ci++;
    }
    c=skip_lines(cfp);
  }
  fclose(cfp);
  free(ignidx);

  /* final averaging */
    if (nt) {
      //printf("reset %d\n",nt);
     /* form sum(J_i)^H * J_i : stored in Jf1 */
     for (ci=0; ci<Nc; ci++){
       matprod_JHJ(aa,Jf1[0][ci],Jf1[1][ci],Jf1[2][ci],Jf1[3][ci],Jf1[4][ci],Jf1[5][ci],Jf1[6][ci],Jf1[7][ci]);
       JsHJ[0][ci]+=aa[0]+_Complex_I*aa[1];
       JsHJ[1][ci]+=aa[2]+_Complex_I*aa[3];
       JsHJ[2][ci]+=aa[4]+_Complex_I*aa[5];
       JsHJ[3][ci]+=aa[6]+_Complex_I*aa[7];
       
       //printf("J=[%lf+j*%lf, %lf+j*%lf; %lf+j*%lf, %lf+j*%lf] JJ=[%lf+j*%lf, %lf+j*%lf; %lf+j*%lf, %lf+j*%lf]\n",Jf1[0][ci],Jf1[1][ci],Jf1[2][ci],Jf1[3][ci],Jf1[4][ci],Jf1[5][ci],Jf1[6][ci],Jf1[7][ci],aa[0],aa[1],aa[2],aa[3],aa[4],aa[5],aa[6],aa[7]);
       matprod_JHJ(aa,Jf2[0][ci],Jf2[1][ci],Jf2[2][ci],Jf2[3][ci],Jf2[4][ci],Jf2[5][ci],Jf2[6][ci],Jf2[7][ci]);
       JHJ[0][ci]+=aa[0]+_Complex_I*aa[1];
       JHJ[1][ci]+=aa[2]+_Complex_I*aa[3];
       JHJ[2][ci]+=aa[4]+_Complex_I*aa[5];
       JHJ[3][ci]+=aa[6]+_Complex_I*aa[7];

     }

    }

  printf("Total timeslots=%d\n",Nt);
  /* form sum(J_i)^J sum(J_i) - J_i^H J_i */
  for (ci=0; ci<Nc; ci++) {
   JsHJ[0][ci]=(0.5/(double)(Nt*Ns*(Ns-1)))*(JsHJ[0][ci]-JHJ[0][ci]);
   JsHJ[1][ci]=(0.5/(double)(Nt*Ns*(Ns-1)))*(JsHJ[1][ci]-JHJ[1][ci]);
   JsHJ[2][ci]=(0.5/(double)(Nt*Ns*(Ns-1)))*(JsHJ[2][ci]-JHJ[2][ci]);
   JsHJ[3][ci]=(0.5/(double)(Nt*Ns*(Ns-1)))*(JsHJ[3][ci]-JHJ[3][ci]);
#ifdef DEBUG
   printf("%d : [%lf+j*%lf, %lf+j*%lf; %lf+j*%lf, %lf+j*%lf]\n",ci,creal(JsHJ[0][ci]),cimag(JsHJ[0][ci]),creal(JsHJ[1][ci]),cimag(JsHJ[1][ci]),creal(JsHJ[2][ci]),cimag(JsHJ[2][ci]),creal(JsHJ[3][ci]),cimag(JsHJ[3][ci]));
#endif
  }

  /***************done reading cluster file*********************************/

   /* read, parse source list */
  glist_sorted_init(slist,sizeof(sinfo),&listfree,&listcomp);

  /* read in the source list, parse each source info */
  if ((cfp=fopen(slistname,"r"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      exit(1);
  }

  c=skip_lines(cfp);
  while(c>=0) {
    memset(buf,0,buff_len);
    if (format==FORMAT_BBS) {
      c=read_bbs_skyline(cfp,buf,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&ss.sI,&ss.sQ,&ss.sU,&ss.sV, &ss.type, &ss.eX, &ss.eY, &ss.eP, &ss.f0, &ss.spec_idx);
      ss.exdata=NULL;
      ss.spec_idx1=ss.spec_idx2=0.0;
    } else if (format==FORMAT_LM) {
     /* l,m format */
     /* NAME l m sI */
     c=fscanf(cfp,"%s %lf %lf %lf",buf,&ss.ra,&ss.dec,&ss.sI);
     ss.sQ=ss.sU=ss.sV=0.0;
     ss.eX=ss.eY=ss.eP=0.0;
     ss.spec_idx=ss.spec_idx1=ss.spec_idx2=0.0;
     ss.f0=10000.0;
     ss.type=STYPE_POINT;
     /* convert l,m radians to degrees */
     ss.ra *=(180.0/M_PI);
     ss.dec *=(180.0/M_PI);
#ifdef DEBUG
     printf("l,m=%lf,%lf\n",ss.ra,ss.dec);
#endif
    }  else if (format==FORMAT_LSM_SP) { /* new format */
     /* LSM format */
    /* ### Name  | RA (hr,min,sec) | DEC (deg,min,sec) | I | Q | U | V | SI0 |SI1 | SI2 | RM | eX (rad) | eY (rad) | eP (rad) | ref_freq */
      c=fscanf(cfp,"%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",buf,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&ss.sI,&ss.sQ,&ss.sU,&ss.sV,&ss.spec_idx,&ss.spec_idx1,&ss.spec_idx2,&dummy_RM,&ss.eX,&ss.eY,&ss.eP, &ss.f0);
    /* use first letter of name to determine which type of source */
    if (buf[0]=='G' || buf[0]=='g') {
      ss.type=STYPE_GAUSSIAN;
    } else if (buf[0]=='D' || buf[0]=='d') {
      ss.type=STYPE_DISK;
    } else if (buf[0]=='R' || buf[0]=='r') {
      ss.type=STYPE_RING;
    } else if (buf[0]=='S' || buf[0]=='s') {
      ss.type=STYPE_SHAPELET;
      /* need mode file called "buf.fits.modes" */
      if ((exs=(exinfo_shapelet*)calloc(1,sizeof(exinfo_shapelet)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      /* open mode file and build up info */
      read_shapelet_modes_img(buf,&exs->n0,&exs->beta,&exs->modes,&exs->eX,&exs->eY,&exs->eP,&exs->linear_tf);
      ss.exdata=(void*)exs;
    } else {
      ss.type=STYPE_POINT;
    }
    }  else { /* format==FORMAT_LSM */
     /* LSM format */
    /* ### Name  | RA (hr,min,sec) | DEC (deg,min,sec) | I | Q | U | V | SI | RM | eX (rad) | eY (rad) | eP (rad) | ref_freq */
      c=fscanf(cfp,"%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",buf,&rahr,&ramin,&rasec,&decd,&decmin,&decsec,&ss.sI,&ss.sQ,&ss.sU,&ss.sV,&ss.spec_idx,&dummy_RM,&ss.eX,&ss.eY,&ss.eP, &ss.f0);
    /* use first letter of name to determine which type of source */
    if (buf[0]=='G' || buf[0]=='g') {
      ss.type=STYPE_GAUSSIAN;
    } else if (buf[0]=='D' || buf[0]=='d') {
      ss.type=STYPE_DISK;
    } else if (buf[0]=='R' || buf[0]=='r') {
      ss.type=STYPE_RING;
    } else if (buf[0]=='S' || buf[0]=='s') {
      ss.type=STYPE_SHAPELET;
      /* need mode file called "buf.fits.modes" */
      if ((exs=(exinfo_shapelet*)calloc(1,sizeof(exinfo_shapelet)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      /* open mode file and build up info */
      read_shapelet_modes_img(buf,&exs->n0,&exs->beta,&exs->modes,&exs->eX,&exs->eY,&exs->eP,&exs->linear_tf);
      ss.exdata=(void*)exs;
    } else {
      ss.type=STYPE_POINT;
    }
     ss.spec_idx1=ss.spec_idx2=0.0;
    }


    /* lookup hash table */
    //cid=(int*)g_hash_table_lookup(stable,buf); 
    cptr=(clust_t**)g_hash_table_lookup(stable,buf); 
#ifdef DEBUG
    printf("looking for %s\n",buf);
    printf("found cluster %d cols %d,%d\n",(*cptr)->id,(*cptr)->ex,(*cptr)->ey);
#endif
    if (cptr && *cptr) {
    cid=(*cptr)->id;
    /* actual cluster id in solution is Nc-(cid+1) */
    if ((*cptr)->id>=0) {
      sIxc=sQxc=sUxc=sVxc=0.0;
      /* columns are (*cptr)->ex...(*cptr)->ey */
      for (cid=(*cptr)->ex; cid<=(*cptr)->ey; cid++) {
       /* correct flux, for _unpolarized_ sky FIXME  */
       /* also take abs() for I gain (can be -ve for noise limit)*/
       sIxc+=fabs(creal(JsHJ[0][cid]+JsHJ[3][cid]));
       sQxc+=creal(JsHJ[0][cid]-JsHJ[3][cid]);
       sUxc+=creal(JsHJ[1][cid]+JsHJ[2][cid]);
       sUxc+=creal(JsHJ[1][cid]-JsHJ[2][cid]);
      }
      sIxc/=(double)(*cptr)->nchunk;
      sQxc/=(double)(*cptr)->nchunk;
      sUxc/=(double)(*cptr)->nchunk;
      sVxc/=(double)(*cptr)->nchunk;
      ss.sI*=sIxc;
      ss.sQ*=sQxc;
      ss.sU*=sUxc;
      ss.sV*=sVxc;
#ifdef DEBUG
      printf("source %s : clu %d\n",buf,(*cptr)->id);
#endif
    }
    } else {
      fprintf(stderr,"Error: %s: %d: cluster for source %s not found.\n",__FILE__,__LINE__,buf);
    }
#ifdef DEBUG
    printf("read %d items:  %s %lf %lf %lf %lf %lf %lf [%lf %lf %lf %lf] (%lf,%lf,%lf,%lf) [%lf,%lf,%lf]\n",c, buf,rahr,ramin,rasec,decd,decmin,decsec,ss.sI,ss.sQ,ss.sU,ss.sV,ss.spec_idx,ss.spec_idx1,ss.spec_idx2,ss.f0, ss.eX,ss.eY,ss.eP);
#endif

    if (format!=FORMAT_LM) {
    /* Rad=(hr+min/60+sec/60*60)*pi/12 */
    if (rahr<0) {
     ss.ra=-(-rahr+ramin/60.0+rasec/3600.0)*M_PI/12.0;
    } else {
     ss.ra=(rahr+ramin/60.0+rasec/3600.0)*M_PI/12.0;
    }
    /* Rad=(hr+min/60+sec/60*60)*pi/180 */
    if (decd<0) {
     ss.dec=-(-decd+decmin/60.0+decsec/3600.0)*M_PI/180.0;
    } else {
     ss.dec=(decd+decmin/60.0+decsec/3600.0)*M_PI/180.0;
    }
    }
    glist_sorted_insert(slist,(void*)&ss);
    c=skip_restof_line(cfp);
    c=skip_lines(cfp);
  }
  fclose(cfp);

  free(buf);
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

  for (ci=0; ci<4; ci++) {
   free(JHJ[ci]);
   free(JsHJ[ci]);
  }
  free(JHJ);
  free(JsHJ);
  for (ci=0; ci<8; ci++) {
   free(Jf1[ci]);
   free(Jf2[ci]);
  }
  free(Jf1);
  free(Jf2);
 return 0;
}
