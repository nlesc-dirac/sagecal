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
#include "cluster.h" /* C-clustering lib */

//#define DEBUG

/* comparison of radii for sorting */
static int
compare_radii(const void *a, const void *b, void *dat) {
   pqsrc *ai, *bi;
   ai=(pqsrc *)a; 
   bi=(pqsrc *)b; 
   if (ai->rd<bi->rd) return -1;
   if (ai->rd>bi->rd) return 1;
   return 0;
}


static int
compare_radii_f(const void *a, const void *b, void *dat) {
   pqsrcf *ai, *bi;
   ai=(pqsrcf *)a; 
   bi=(pqsrcf *)b; 
   if (ai->rd<bi->rd) return -1;
   if (ai->rd>bi->rd) return 1;
   return 0;
}


/***************** fitting functions **************************/

/* data struct for levmar */
typedef struct merge_two_to_one_data_ {
  int Ngrid;
  double *lgrid,*mgrid; /* size Ngridx1 */
  double bmaj; 
  double bmin;
  double bpa; 
  /* all above for the grid and PSF */

  double l0,m0; /* position of fitted source */
} merge_two_to_one_data;


/* fitting function  - single point  (ref flux, spectral index)
 p: mx1 format sI0,sP0
 x: nx1 pixel fluxes, from model, pixels ordered with freq
*/
static void
mylm_fit_single_pfmult(double *p, double *x, int m, int n, void *data) {
 merge_two_to_one_data *dp=(merge_two_to_one_data*)data;
 int ci,ck;
 double ll,mm;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 ci=0;
 /* iterate over grid */
 for (ck=0; ck<dp->Ngrid; ck++) {
   /* grid iteration */
    ll=(-(dp->lgrid[ck]-dp->l0)*sin(dp->bpa)+(dp->mgrid[ck]-dp->m0)*cos(dp->bpa))/dp->bmaj;
    mm=(-(dp->lgrid[ck]-dp->l0)*cos(dp->bpa)-(dp->mgrid[ck]-dp->m0)*sin(dp->bpa))/dp->bmin;
    x[ci]=p[0]*exp(-(ll*ll+mm*mm));
    ci++;
 }
}



/* merge two sources to one */
/* pixlist : list of pixels
   bmaj,bmin,bpa: PSF
   lval,mval,sIval: 2x1 arrays of two sources parameters
   ll,mm,sI : output value for merged source
*/

static double
fit_two_to_one(GList *pixlist, double bmaj, double bmin, double bpa, double *lval, double *mval, double *sIval, double *ll, double *mm, double *sI) {

 int m,n,npix;
 hpixel *ppix;
 double *pixell,*pixelm,*p,*x,*x1;
 GList *pli;
 int ci;
 /* setup levmar */
 double opts[CLM_OPTS_SZ], info[CLM_INFO_SZ];
 opts[0]=CLM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-15;
 opts[4]=-CLM_DIFF_DELTA;

 int maxiter=100;
 merge_two_to_one_data lmdata;
 
 m=1; /* sI*/
 npix=g_list_length(pixlist);
 n=npix;  /* no of pixels x freqs */

 if ((pixell=(double*)calloc((size_t)npix,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((pixelm=(double*)calloc((size_t)npix,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }

 if ((p=(double*)calloc((size_t)m,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((x1=(double*)calloc((size_t)n,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 ci=0;
 for(pli=pixlist; pli!=NULL; pli=g_list_next(pli)) {
   ppix=pli->data;
   pixell[ci]=ppix->l;
   pixelm[ci]=ppix->m;
   ci++;
 }

 /* ll,mm is the centroid */
 *ll=lval[0]*fabs(sIval[0])+lval[1]*fabs(sIval[1]);
 *mm=mval[0]*fabs(sIval[0])+mval[1]*fabs(sIval[1]);
 *ll/=(fabs(sIval[0])+fabs(sIval[1]));
 *mm/=(fabs(sIval[0])+fabs(sIval[1]));

 lmdata.Ngrid=npix;
 lmdata.lgrid=pixell;
 lmdata.mgrid=pixelm;
 lmdata.bmaj=bmaj;
 lmdata.bmin=bmin;
 lmdata.bpa=bpa;

 /* first source contrib */
 p[0]=sIval[0];
 lmdata.l0=lval[0];
 lmdata.m0=mval[0];
 mylm_fit_single_pfmult(p, x, m, n, (void*)&lmdata);
 /* second source contrib */
 p[0]=sIval[1];
 lmdata.l0=lval[1];
 lmdata.m0=mval[1];
 mylm_fit_single_pfmult(p, x1, m, n, (void*)&lmdata);
 /* x=x+x1 */
 my_daxpy(n, x1, 1.0, x);
 free(x1);

 lmdata.l0=*ll;
 lmdata.m0=*mm;
 /* initial values */
 p[0]=0.5*(sIval[0]+sIval[1]);

#ifdef DEBUG
 printf("Initial merge val (%lf)\n",p[0]);
#endif

 clevmar_der_single_nocuda(mylm_fit_single_pfmult, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);

 /* output */
 *sI=p[0];
 
#ifdef DEBUG
 printf("Final merge val (%lf)\n",p[0]);
#endif
 free(pixell);
 free(pixelm);
 free(p);
 free(x);
 return 0;
}


/* data struct for levmar */
typedef struct merge_two_to_one_f_data_ {
  int Ngrid;
  double *lgrid,*mgrid; /* size Ngridx1 */
  int Nf;
  double *freqs; /* size Nfx1 */
  double *bmaj; /* size Nfx1 */
  double *bmin; /* size Nfx1 */
  double *bpa; /* size Nfx1 */
  /* all above for the grid and PSF */

  double ref_freq;
  double l0,m0; /* position of fitted source */
} merge_two_to_one_f_data;


/* fitting function  - single point  (ref flux, spectral index)
 p: mx1 format sI0,sP0
 x: nx1 pixel fluxes, from model, pixels ordered with freq
*/
static void
mylm_fit_single_sipfmult(double *p, double *x, int m, int n, void *data) {
 merge_two_to_one_f_data *dp=(merge_two_to_one_f_data*)data;
 int ci,cj,ck;
 double ll,mm;
 /* first reset x to all zeros */
 memset(x,0,sizeof(double)*n);
 ci=0;
 /* iterate over grid, per freq */
 for (ck=0; ck<dp->Ngrid; ck++) {
   for (cj=0; cj<dp->Nf; ++cj) { /* freq iteration */
   /* grid iteration */

    ll=(-(dp->lgrid[ck]-dp->l0)*sin(dp->bpa[cj])+(dp->mgrid[ck]-dp->m0)*cos(dp->bpa[cj]))/dp->bmaj[cj];
    mm=(-(dp->lgrid[ck]-dp->l0)*cos(dp->bpa[cj])-(dp->mgrid[ck]-dp->m0)*sin(dp->bpa[cj]))/dp->bmin[cj];
    x[ci]=p[0]*pow(dp->freqs[cj]/dp->ref_freq,p[1])*exp(-(ll*ll+mm*mm));
    ci++;
   }
 }
}


/* merge two sources to one */
/* pixlist : list of pixels
   freqs: Nfx1 array of frequencies 
   bmaj,bmin,bpa: PSF Nfx1 arrays 
   ref_freq: reference freq
   All above needed to evaluate the grid
   lval,mval,sIval,sPval : 2x1 arrays of two sources parameters
   ll,mm,sI,sP : output value for merged source
*/

static double
fit_two_to_one_f(GList *pixlist, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, double *lval, double *mval, double *sIval, double *sPval, double *ll, double *mm, double *sI, double *sP) {

 int m,n,npix;
 hpixelf *ppix;
 double *pixell,*pixelm,*p,*x,*x1;
 GList *pli;
 int ci;
 /* setup levmar */
 double opts[CLM_OPTS_SZ], info[CLM_INFO_SZ];
 opts[0]=CLM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-15;
 opts[4]=-CLM_DIFF_DELTA;

 int maxiter=100;
 merge_two_to_one_f_data lmdata;
 
 m=2; /* sI, sP */
 npix=g_list_length(pixlist);
 n=npix*Nf;  /* no of pixels x freqs */

 if ((pixell=(double*)calloc((size_t)npix,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((pixelm=(double*)calloc((size_t)npix,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }

 if ((p=(double*)calloc((size_t)m,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((x=(double*)calloc((size_t)n,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 if ((x1=(double*)calloc((size_t)n,sizeof(double)))==0) {
   fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
   exit(1);
 }
 ci=0;
 for(pli=pixlist; pli!=NULL; pli=g_list_next(pli)) {
   ppix=pli->data;
   pixell[ci]=ppix->l;
   pixelm[ci]=ppix->m;
   ci++;
 }

 /* ll,mm is the centroid */
 *ll=lval[0]*fabs(sIval[0])+lval[1]*fabs(sIval[1]);
 *mm=mval[0]*fabs(sIval[0])+mval[1]*fabs(sIval[1]);
 *ll/=(fabs(sIval[0])+fabs(sIval[1]));
 *mm/=(fabs(sIval[0])+fabs(sIval[1]));

 lmdata.Ngrid=npix;
 lmdata.lgrid=pixell;
 lmdata.mgrid=pixelm;
 lmdata.Nf=Nf;
 lmdata.freqs=freqs;
 lmdata.bmaj=bmaj;
 lmdata.bmin=bmin;
 lmdata.bpa=bpa;
 lmdata.l0=*ll;
 lmdata.m0=*mm;
 lmdata.ref_freq=ref_freq;

 /* first source contrib */
 p[0]=sIval[0];
 p[1]=sPval[0];
 lmdata.l0=lval[0];
 lmdata.m0=mval[0];
 mylm_fit_single_sipfmult(p, x, m, n, (void*)&lmdata);
 /* second source contrib */
 p[0]=sIval[1];
 p[1]=sPval[1];
 lmdata.l0=lval[1];
 lmdata.m0=mval[1];
 mylm_fit_single_sipfmult(p, x1, m, n, (void*)&lmdata);
 /* x=x+x1 */
 my_daxpy(n, x1, 1.0, x);
 free(x1);

 lmdata.l0=*ll;
 lmdata.m0=*mm;
 /* initial values */
 p[0]=0.5*(sIval[0]+sIval[1]);
 p[1]=0.5*(sPval[0]+sPval[1]);

#ifdef DEBUG
 printf("Initial merge val (%lf,%lf)\n",p[0],p[1]);
#endif

 clevmar_der_single_nocuda(mylm_fit_single_sipfmult, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);

 /* output */
 *sI=p[0];
 *sP=p[1];
 
#ifdef DEBUG
 printf("Final merge val (%lf,%lf)\n",p[0],p[1]);
#endif
 free(pixell);
 free(pixelm);
 free(p);
 free(x);
 return 0;
}

/***************** end fitting functions **************************/

int 
cluster_sources(double r, GList *inlist, GList *pixlist, double bmaj, double bmin, double bpa, GList **outlist) {
 /* priority queue */
 GQueue *pq;

 /* list iterators */
 GList *li;

 pqsrc *qnode, *headnode;

 int nct,ci,myclose,we_cluster_this;
 double mydist,tmp,ll,mm,sI;

 /* arrays for fitting */
 double lval[2],mval[2],sIval[2];


 /* no clustering if only one source */
 if (g_list_length(inlist)<=1) {
  return -1;
 }

/* FIXME: what if only few nodes are close and others are far ? */
#ifdef DEBUG
 printf("min radius =%lf\n",r);
#endif
 /* put all sources in inlist to a priority queue, with radius 0 */
 /* queue is sourced by radius, smallest first */
 pq=g_queue_new();
 ci=0;
 for (li=inlist; li!=NULL; li=g_list_next(li)) {
   if((qnode= (pqsrc*)malloc(sizeof(pqsrc)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
   qnode->rd=0.0;
   if((qnode->src= (extsrc*)malloc(sizeof(extsrc)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
  
   memcpy((void*)qnode->src,(void*)li->data,sizeof(extsrc)); /* copy source data */
 
   g_queue_insert_sorted(pq,(gpointer)qnode,compare_radii,(gpointer)0);

 }

#ifdef DEBUG
    printf("total nodes to cluster %d\n",g_queue_get_length(pq));
#endif

 /************** now the clustering *****************************/
 we_cluster_this=1;
 while (we_cluster_this && (g_queue_get_length(pq)>1)) {
   /* first test if we need to merge */
   headnode=g_queue_peek_head(pq);
   if ((headnode!=NULL) && (headnode->rd<r) ) {
    /* we can cluster */
    headnode=g_queue_pop_head(pq);
    /* find the closest source to this one */
    nct=g_queue_get_length(pq);
#ifdef DEBUG
    printf("remaining nodes %d\n",nct);
#endif
    mydist=1e6;
    myclose=-1; /* no source found yet */
    for (ci=0; ci<nct; ci++) {
      qnode=g_queue_peek_nth(pq,ci);
      if (qnode!=NULL) {
        /* calculate distance */
        tmp=sqrt((qnode->src->l-headnode->src->l)*(qnode->src->l-headnode->src->l)+ (qnode->src->m-headnode->src->m)*(qnode->src->m-headnode->src->m));
        if (tmp<mydist) {
         mydist=tmp;
         myclose=ci;
        }
      }
    }
 
#ifdef DEBUG
    printf("closest is %d with distance %lf\n",myclose,mydist);
#endif
    if (myclose>=0) {
     if (mydist<r) {
      /* get this node and merge */
      qnode=g_queue_pop_nth(pq,myclose);
      /* distance */
      qnode->rd=mydist;
      ll=qnode->src->l*qnode->src->sI+headnode->src->l*headnode->src->sI;
      mm=qnode->src->m*qnode->src->sI+headnode->src->m*headnode->src->sI;
      ll/=(qnode->src->sI+headnode->src->sI);
      mm/=(qnode->src->sI+headnode->src->sI);
      qnode->src->l=ll;
      qnode->src->m=mm;
      /* FIXME: new flux : preserve total or peak? */
      sI=(qnode->src->sI+headnode->src->sI)*0.5; /* preserve peak */
      //qnode->src->sI+=headnode->src->sI; /* preserve total*/
      /* minimize to find p[0]=sI for merged source */
      /* minimization domain [l,m] : determined by pixlist */
      lval[0]=qnode->src->l;
      lval[1]=headnode->src->l;
      mval[0]=qnode->src->m;
      mval[1]=headnode->src->m;
      sIval[0]=qnode->src->sI;
      sIval[1]=headnode->src->sI;

      /* do non linear fit for sI and sP over domain [l,m] pixels and freq range */
      fit_two_to_one(pixlist, bmaj, bmin, bpa, lval, mval, sIval, &ll, &mm, &sI);
      /* update new source */
      qnode->src->l=ll;
      qnode->src->m=mm;
      qnode->src->sI=sI;


      g_queue_insert_sorted(pq,(gpointer)qnode,compare_radii,(gpointer)0);
      /* free the headnode */
      free(headnode->src);
      free(headnode);
     } else {
      /* put back the headnode, with 'updated' radius*/
      headnode->rd=mydist;
      g_queue_insert_sorted(pq,(gpointer)headnode,compare_radii,(gpointer)0);
      //we_cluster_this=0;
     }
    } else {
     /* no source found , put back head node and stop clustering */
      /* put back the headnode and stop clustering */
      g_queue_insert_sorted(pq,(gpointer)headnode,compare_radii,(gpointer)0);
      we_cluster_this=0;
    }
   } else { 
    we_cluster_this=0;
   }
 } 

 /* create a new list with clustered sources */
 /* also free the queue */
 /* print whats in */
 qnode=g_queue_pop_head(pq);
 *outlist=NULL;
 while(qnode!=NULL) {
#ifdef DEBUG
  printf("PQ: radius=%lf lm=(%lf,%lf) flux=%lf\n",qnode->rd,qnode->src->l,qnode->src->m,qnode->src->sI);
#endif
  *outlist=g_list_prepend(*outlist,qnode->src);
  free(qnode);
  qnode=g_queue_pop_head(pq);
 }

 g_queue_free(pq);
 return 0;
}


int 
cluster_sources_f(double r, GList *inlist, GList *pixlist, int Nf, double *freqs, double *bmaj, double *bmin, double *bpa, double ref_freq, GList **outlist) {
 /* priority queue */
 GQueue *pq;

 /* list iterators */
 GList *li;

 pqsrcf *qnode, *headnode;

 int nct,ci,myclose,we_cluster_this;
 double mydist,tmp,ll,mm,sI,sP;

 /* arrays for fitting */
 double lval[2],mval[2],sIval[2],sPval[2];

 /* no clustering if only one source */
 if (g_list_length(inlist)<=1) {
  return -1;
 }

/* FIXME: what if only few nodes are close and others are far ? */
#ifdef DEBUG
 printf("min radius =%lf\n",r);
#endif
 /* put all sources in inlist to a priority queue, with radius 0 */
 /* queue is sourced by radius, smallest first */
 pq=g_queue_new();
 ci=0;
 for (li=inlist; li!=NULL; li=g_list_next(li)) {
   if((qnode= (pqsrcf*)malloc(sizeof(pqsrcf)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
   qnode->rd=0.0;
   if((qnode->src=(extsrcf*)malloc(sizeof(extsrcf)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
   }
  
   memcpy((void*)qnode->src,(void*)li->data,sizeof(extsrcf)); /* copy source data */
 
   g_queue_insert_sorted(pq,(gpointer)qnode,compare_radii_f,(gpointer)0);

 }

#ifdef DEBUG
    printf("total nodes to cluster %d\n",g_queue_get_length(pq));
#endif

 /************** now the clustering *****************************/
 we_cluster_this=1;
 while (we_cluster_this && (g_queue_get_length(pq)>1)) {
   /* first test if we need to merge */
   headnode=g_queue_peek_head(pq);
   if ((headnode!=NULL) && (headnode->rd<r) ) {
    /* we can cluster */
    headnode=g_queue_pop_head(pq);
    /* find the closest source to this one */
    nct=g_queue_get_length(pq);
#ifdef DEBUG
    printf("remaining nodes %d\n",nct);
#endif
    mydist=INFINITY_L;
    myclose=-1; /* no source found yet */
    for (ci=0; ci<nct; ci++) {
      qnode=g_queue_peek_nth(pq,ci);
      if (qnode!=NULL) {
        /* calculate distance */
        tmp=sqrt((qnode->src->l-headnode->src->l)*(qnode->src->l-headnode->src->l)+ (qnode->src->m-headnode->src->m)*(qnode->src->m-headnode->src->m));
        if (tmp<mydist) {
         mydist=tmp;
         myclose=ci;
        }
      }
    }
 
#ifdef DEBUG
    printf("closest is %d with distance %lf\n",myclose,mydist);
#endif
    if (myclose>=0) {
     if (mydist<r) {
      /* get this node and merge */
      qnode=g_queue_pop_nth(pq,myclose);
      /* distance */
      qnode->rd=mydist;
      ll=qnode->src->l*qnode->src->sI+headnode->src->l*headnode->src->sI;
      mm=qnode->src->m*qnode->src->sI+headnode->src->m*headnode->src->sI;
      ll/=(qnode->src->sI+headnode->src->sI);
      mm/=(qnode->src->sI+headnode->src->sI);
      qnode->src->l=ll;
      qnode->src->m=mm;
      /* FIXME: new flux : preserve total or peak? */
      sI=(qnode->src->sI+headnode->src->sI)*0.5; /* preserve peak */
      sP=(qnode->src->sP+headnode->src->sP)*0.5; /* spectral idx*/
      //qnode->src->sI+=headnode->src->sI; /* preserve total*/
      lval[0]=qnode->src->l;
      lval[1]=headnode->src->l;
      mval[0]=qnode->src->m;
      mval[1]=headnode->src->m;
      sIval[0]=qnode->src->sI;
      sIval[1]=headnode->src->sI;
      sPval[0]=qnode->src->sP;
      sPval[1]=headnode->src->sP;

      /* do non linear fit for sI and sP over domain [l,m] pixels and freq range */
      fit_two_to_one_f(pixlist, Nf, freqs, bmaj, bmin, bpa, ref_freq, lval, mval, sIval, sPval, &ll, &mm, &sI, &sP);
      /* update new source */
      qnode->src->l=ll;
      qnode->src->m=mm;
      qnode->src->sI=sI;
      qnode->src->sP=sP;

      g_queue_insert_sorted(pq,(gpointer)qnode,compare_radii_f,(gpointer)0);
      /* free the headnode */
      free(headnode->src);
      free(headnode);
     } else {
      /* put back the headnode, with 'updated' radius*/
      headnode->rd=mydist;
      g_queue_insert_sorted(pq,(gpointer)headnode,compare_radii_f,(gpointer)0);
     }
    } else {
     /* no source found , put back head node and stop clustering */
      /* put back the headnode and stop clustering */
      g_queue_insert_sorted(pq,(gpointer)headnode,compare_radii_f,(gpointer)0);
      we_cluster_this=0;
    }
   } else { 
    we_cluster_this=0;
   }
 } 

 /* create a new list with clustered sources */
 /* also free the queue */
 /* print whats in */
 qnode=g_queue_pop_head(pq);
 *outlist=NULL;
 while(qnode!=NULL) {
#ifdef DEBUG
  printf("PQ: radius=%lf lm=(%lf,%lf) flux=%lf\n",qnode->rd,qnode->src->l,qnode->src->m,qnode->src->sI);
#endif
  *outlist=g_list_prepend(*outlist,qnode->src);
  free(qnode);
  qnode=g_queue_pop_head(pq);
 }

 g_queue_free(pq);
 return 0;
}


/*********************** clustering sky (multiple sources) using C-cluster lib ******/
/* Perform k-means clustering */
/* nrows: no of entries (sources)
   ncols: no of parameters for each source (l,m)
   data: nrows x ncols matrix
   mask: nrows x ncols matrix
   weight: ncols x 1 array
   clusterid: nrows x 1 array of cluser ids (output)
   nclusters: max no of clusters
*/
static void 
kmeans_clustering(int nrows, int ncols, double** data, int** mask, double *weight, int *clusterid, int nclusters) { 
  const int transpose = 0;
  const char dist = 'e';
  const char method = 'a';
  int npass = 1;
  int ifound = 0;
  double error;

  printf("clustering %d passes of the EM algorithm\n",npass);
  npass = 1000;
  kcluster(nclusters,nrows,ncols,data,mask,weight,transpose,npass,method,dist,
    clusterid, &error, &ifound);
  printf ("Solution found %d times; ", ifound);
  printf ("within-cluster sum of distances is %f\n", error);
#ifdef DEBUG
  printf ("Cluster assignments:\n");
  for (i = 0; i < nrows; i++)
    printf ("Source %d: cluster %d\n", i, clusterid[i]);
#endif

}


/* Perform hierarchical clustering */
/* nrows: no of entries (sources)
   ncols: no of parameters for each source (l,m)
   data: nrows x ncols matrix
   mask: nrows x ncols matrix
   weight: ncols x 1 array
   clusterid: nrows x 1 array of cluser ids (output)
   ncut: no of clusters for the tree to be cut (actual clusters might be less)
   nclusters: no of clusters produced
*/
static void 
hierarchical_clustering(int nrows, int ncols, double** data, int** mask, double *weight, int *clusterid, int ncut, int *nclusters) { 
 
  double **distmatrix;
  int ci;
  Node *tree;
  /* calculate Euclidean distance */
  distmatrix=distancematrix(nrows, ncols, data, mask, weight, 'e', 0);
  if (!distmatrix) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  /* pairwise centroid linkage clustering */
  tree = treecluster(nrows, ncols, data, mask, weight, 0, 'e', 'c', 0);
  if (!tree) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  } 
  cuttree (nrows, tree, ncut, clusterid);
  free(tree);
  /* find max cluster id to get no of clusters */
  *nclusters=0;
  for(ci=0; ci<nrows; ci++) {
   if (clusterid[ci]>(*nclusters)) {
     (*nclusters)=clusterid[ci];
   }
  }
  /* adjust for 0 indexing */
  (*nclusters)++;
  printf("clusters=%d\n",(*nclusters));
  /* distmatrix is only triangular */
  for (ci=1; ci<nrows; ci++) free(distmatrix[ci]);
  free(distmatrix);
}

/* struct for sorting fluxes */
typedef struct flux_id_{
 double sI;
 int idx;
} flux_id; 
/* comparison function for sorting fluxes */
static int
flux_compare(const void *a, const void *b) {
  flux_id *aa,*bb;
  aa=(flux_id*)a;
  bb=(flux_id*)b;
  if (aa->sI>bb->sI) return -1;
  if (aa->sI==bb->sI) return 0;
  return 1;
}

int
cluster_sky(const char *imgfile, GList *skylist, int ncluster) {
  
  clsrc *csrc;
  GList *li;
  int nrows=g_list_length(skylist);
  int ncols=2; /* l,m */
  double** data;
  int** mask;
  double* weight;
  int* clusterid;

  int ci,cj;

  int nhcluster;

  char *clusterfile,*clusterann;
  FILE *cfilep,*annfp;

  GList **csarray; /* array of size ncluster x 1 of lists, for sources in each cluster */
  flux_id *cflux; /* array for sorting clusters according to their fluxes */

  if (ncluster==0) { return -1; }

  if ((data=(double**)malloc(nrows*sizeof(double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((mask=(int**)malloc(nrows*sizeof(int*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  for (ci=0; ci<nrows; ci++) {
    if ((data[ci]=(double*)malloc(ncols*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if ((mask[ci]=(int*)malloc(ncols*sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
  }

  if ((weight=(double*)malloc(ncols*sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if ((clusterid=(int*)malloc(nrows*sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  for (ci=0; ci<ncols; ci++) weight[ci] = 1.0;

  ci=0;
  for (li=skylist; li!=NULL; li=g_list_next(li)) {
    csrc=li->data;
    data[ci][0]=csrc->l;
    data[ci][1]=csrc->m;
    mask[ci][0]=1;
    mask[ci][1]=1;
    ci++;
  }

  if (ncluster>0) {
   kmeans_clustering(nrows, ncols, data, mask, weight, clusterid, ncluster);
  } else {
   /* hierarchical clustering */
   hierarchical_clustering(nrows,ncols,data,mask,weight,clusterid,abs(ncluster),&nhcluster);
   ncluster=nhcluster;
  }

  for (ci=0; ci<nrows; ci++) {
    free(data[ci]);
    free(mask[ci]);
  }
  free(data);
  free(mask);
  free(weight);


  /* now write output of clustering to text file */
  if ((csarray=(GList**)malloc(ncluster*sizeof(GList*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  for (ci=0; ci<ncluster; ci++) {
   csarray[ci]=NULL;
  }
  if ((cflux=(flux_id*)calloc((size_t)ncluster,sizeof(flux_id)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }

  /* go through all sources, insert its info in the correct list */
  ci=0;
  for (li=skylist; li!=NULL; li=g_list_next(li)) {
    csrc=li->data;
    csarray[clusterid[ci]]=g_list_prepend(csarray[clusterid[ci]],csrc);
    cflux[clusterid[ci]].sI+=csrc->sI;
    cflux[clusterid[ci]].idx=clusterid[ci];
    ci++;
  }
  free(clusterid);
  
  /* sort clusters according to their fluxes */
  qsort(cflux,ncluster,sizeof(flux_id),flux_compare);

  /* create files to write output */
  clusterfile=(char*)calloc((size_t)strlen(imgfile)+strlen(".sky.txt.cluster")+1,sizeof(char));
  if ( clusterfile== 0 ) {
     fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
     exit(1);
  }
  strcpy(clusterfile,imgfile);
  ci=strlen(imgfile);
  strcpy(&clusterfile[ci],".sky.txt.cluster\0");

  clusterann=(char*)calloc((size_t)strlen(imgfile)+strlen(".sky.txt.cluster.reg")+1,sizeof(char));
  if ( clusterann== 0 ) {
     fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);
     exit(1);
  }
  strcpy(clusterann,imgfile);
  ci=strlen(imgfile);
  strcpy(&clusterann[ci],".sky.txt.cluster.reg\0");

  cfilep=fopen(clusterfile,"w+");
  if(!cfilep) {
      fprintf(stderr,"%s: %d: unable to open file\n",__FILE__,__LINE__);
      exit(1);
  }
  annfp=fopen(clusterann,"w+");
  if(!annfp) {
      fprintf(stderr,"%s: %d: unable to open file\n",__FILE__,__LINE__);
      exit(1);
  }

  /* print header info in the files */
  /* DS9 region format */
  fprintf(annfp,"# Region file format: DS9 version 4.1\n");
  fprintf(annfp,"global color=blue dashlist=8 3 width=1 font=\"helvetica 10 normal\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n");



  fprintf(cfilep,"# cluster file format (sorted in descending order)\n# id hybrid_slots source1 source2 ....\n");
  for (cj=0; cj<ncluster; cj++) {
   ci=cflux[cj].idx;
#ifdef DEBUG
   printf("Cluster %d has %d sources, flux=%lf\n",ci,g_list_length(csarray[ci]),cflux[cj].sI);
#endif
   if (g_list_length(csarray[ci])>0) {
    fprintf(cfilep,"%d 1 ",ci);
    for (li=csarray[ci]; li!=NULL; li=g_list_next(li)) {
      csrc=li->data;
      fprintf(cfilep,"%s ",csrc->name);
#ifdef DEBUG
      printf("%s ",csrc->name);
#endif
      fprintf(annfp,"fk5;point(%lf,%lf) # point=x color=yellow text={%d}\n",csrc->ra,csrc->dec,ci);
    }
#ifdef DEBUG
    printf("\n");
#endif
    fprintf(cfilep,"\n");
   }
  }

  /* since data in the lists were not alloc'd we dont free them */
  for (ci=0; ci<ncluster; ci++) {
   g_list_free(csarray[ci]);
  }
  free(csarray);
  free(cflux);
  free(clusterfile);
  free(clusterann);
  fclose(cfilep);
  fclose(annfp);

  return 0;
}
/*********************** end clustering sky using C-cluster lib ******/
