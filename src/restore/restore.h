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


#ifndef RESTORE_H
#define RESTORE_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>

#include <fitsio.h>
#include <wcs.h>
#include <prj.h>
#include <wcshdr.h>
#include <wcsfix.h>

/* max source name length, increase it if names get longer */
#define MAX_SNAME 2048

/***************** glist.c  *************************/
/* generic list structure */
typedef struct glist_node_ {
  struct glist_node_ *next;
  struct glist_node_ *prev;
  void *data;
} glist_node;

typedef struct glist_ {
  glist_node *head;
  glist_node *tail;
  glist_node *iter; /* used to iterate the list */
  /* destructor function for each list data */
  void (*list_data_free)(void *);
  /* function to compare two data records */
  int (*list_data_comp)(const void *d1, const void *d2);
  /*  data size */
  size_t datasize;
  /* number of elements */
  int count;
} glist;

/* intialize a generic doubly linked list */
/* datasize: size of list data 
 *  * list_data_free: function to free list data 
 *   */
extern void
glist_init(glist *L,size_t datasize, void (*list_data_free)(void *));
/* initialize a generic sorted doubly linked list */
/* datasize: size of list data
 *  * list_data_free: function to free list data
 *   * list_data_comp: compare two data items: return 0 if match, negative if d1>d2 
 *    */
extern void
glist_sorted_init(glist *L,size_t datasize, void (*list_data_free)(void *), int (*list_data_comp)(const void *d1, const void *d2));

/* insert data to list */
extern void *
glist_insert(glist *L, const void *data );
/* insert data to a sorted list */
extern void *
glist_sorted_insert(glist *L, const void *data);
/* delete a complete list */
extern void
glist_delete(glist *L);
/* initialize list traversalt from head to tail */
extern void
glist_set_iter_forward(glist *L);
/* traverse from head to tail */
extern void *
glist_iter_forward(glist *L);
/* initialize list traversal from tail to head */
extern void
glist_set_iter_backward(glist *L);
/* traverse list from tail to head */
extern void *
glist_iter_backward(glist *L);
/* check if the list is empty */
/* return 1 if empty, 0 if not */
extern int
glist_empty(glist *L);
/* remove first element from the list (at head)*/
/* and return its address */
extern void  *
glist_remove(glist *L);
/* remove element from the list if it exists */
/* returns 0 if not deleted (not exist), else 1*/
extern int
glist_sorted_remove(glist *L, const void *data);
/* look for element from the list if it exists */
/* returns 0 if not found (not exist), else the address*/
extern void *
glist_sorted_lookup(glist *L, const void *data);


/***************** restore.c  *************************/
/* soure types */
#define STYPE_POINT 0
#define STYPE_GAUSSIAN 1
#define STYPE_DISK 2
#define STYPE_RING 3
#define STYPE_SHAPELET 4

/*typedef struct nlimits_ {
    int naxis;
    long int *d;
    long int *lpix;
    long int *hpix;
    double tol;
    int datatype;
} nlims; 

typedef struct __io_buff__ {
    fitsfile *fptr;
    nlims arr_dims;
    struct wcsprm *wcs;
} io_buff; */

/* structure fore source info */
typedef struct sourceinfo_{
 double sI,sQ,sU,sV,ra,dec;
 /* for disk (and maybe Gaussian) source */
 double eX,eY,eP;
 /* for spectral index */
 double spec_idx,spec_idx1,spec_idx2, f0;
 /* source type */
 int type;
 /* scale up flux by this value
   to compensate rounding  ra,dec of centroid to the nearest pixel */
 double Iscale;

 /* for shapelets, file name of mode file */
 void *exdata;
} sinfo;

typedef struct exinfo_shapelet_ {
  int n0; /* model order, no of modes=n0*n0 */
  double beta; /* scale*/
  double *modes; /* array of n0*n0 x 1 values */
  double eX,eY,eP; /* linear transform parameters */
  int linear_tf; /* 0: no linear tf, 1: linear tf */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
} exinfo_shapelet;

/* filename: file name
   format: 2 (l,m) special case, override ra,dec
 */
extern int
read_fits_file_restore(const char *filename, glist *slist, double bmaj,double bmin, double pa, int add_to_pixel, int beam_given, int format);

/***************** readsky.c  *************************/
/* format 1: LSM,
   format 0: BBS 
   format 2: (l,m) format */
#define FORMAT_BBS 0
#define FORMAT_LSM 1
#define FORMAT_LSM_SP 2 /* 3 order spectral index */
#define FORMAT_LM 9 /* l,m format (not used) */
extern int
read_sky_model(const char *filename, glist *slist, int format);


/* 
  filaname: sky model,
  format 0,1,2
  slist: source list (output)
  clusterfile: cluster file name
  solfile: solution file name
  ignfile: stations numbers whose solutions to ignore
*/  
extern int
read_sky_model_withgain(const char *slistname, glist *slist, int format,const char *clusterfile, const char *solfile, const char *ignfile);
#endif /* RESTORE_H */
