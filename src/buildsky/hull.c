
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
#include <stdio.h>
#include <stdlib.h>

static void
init_stack (stack_type * s)
{
  s->head = 0;
  s->count=0;
}

static int
push (void *rec, stack_type * s)
{
  stack_node *n;
  if ((n = (stack_node *) malloc (sizeof (stack_node))) == 0)
    {
      fprintf (stderr, "%s: %d: no free memory", __FILE__, __LINE__);
      exit (-1);
    }
  n->rec = rec;
  n->next = s->head;
  s->head = n;
  s->count++;

  return (0);
}

static void *
pop (stack_type * s)
{
  void *temprec;
  stack_node *temp;

  if (s->head == 0)
    return (0);
  else
    {
      temp = s->head;
      temprec = (void *) temp->rec;
      s->head = temp->next;
      free (temp);
      s->count--;
      return (temprec);
    }
}

static int
count(stack_type *s) {
 return s->count;
}

/* comparison function for sorting the array of points */
static int
compare_points (const void *a, const void *b)
{
  hpoint *p1, *p2;

  p1 = (hpoint *) a;
  p2 = (hpoint *) b;

  if ((p1->x < p2->x) || ((p1->x == p2->x) && (p1->y < p2->y)))
    return (-1);
  else
    return (1);

  /* won't reach here */
  return (0);
}

/* determinant of three points */
static double 
determ (hpoint p0, hpoint p1, hpoint p2)
{
  return (p0.x * (p1.y - p2.y) + p1.x * (p2.y - p0.y) + p2.x * (p0.y - p1.y));
}

#define MYPUSH(j,p,s)    if ((j=(int *)malloc(sizeof(j)))==0) {\
          fprintf(stderr,"%s: %d: no free memory",__FILE__,__LINE__);\
          exit(-1);\
  }\
  *j=p;\
  push((void *)j,&s);


/* function to build a convex hull of a given set of pixels */
/* pixset: set of pixels to find the boundary
   pixset->pix: list of pixels
   pixset->hull: array of hull (l,m) pixels
   pixset->Nh: no. of boundary (hull) points
*/
int
construct_boundary(pixellist *pixset)
{
  stack_type sl,su; /* two stacks for upper and lowr hulls */
  hpoint *parray;
  int i, *j, n, p0, p1, p2;
  GList *pixval;
  hpixel *ppix;

  n = g_list_length(pixset->pix);

  /* generate random set of points */
  if ((parray = (hpoint *) malloc ((size_t) n * sizeof (hpoint))) == 0) {
      fprintf (stderr, "%s: %d: no free memory", __FILE__, __LINE__);
      exit (-1);
  }
  pixval=pixset->pix;
  for (i = 0; i < n; i++) {
      ppix=pixval->data;
      parray[i].x = ppix->l;
      parray[i].y = ppix->m;
      //printf("insert %f %f\n",parray[i].x,parray[i].y);
      pixval=g_list_next(pixval);
  }

  qsort ((void *) parray, (size_t) n, sizeof (hpoint), &compare_points);

  init_stack (&su);
  /* construct upper hull */
  p0 = 0;
  p1 = 1;
  p2 = 2;
  MYPUSH (j, p0, su);
  MYPUSH (j, p1, su);
  MYPUSH (j, p2, su);

  for (i = 3; i < n; i++) {
      if (determ (parray[p0], parray[p1], parray[p2]) < 0.0) {
     /* clockwise */
	  /* we can insert p2 to the hull */
	  p0 = p1;
	  p1 = p2;
	  p2 = i;
	  MYPUSH (j, p2, su);
    } else {
		/* counterclockwise or colinear */
	  p1 = p2;
	  j=(int*)pop (&su);		/* remove p2 */
    free(j);
	  j=(int*)pop (&su);		/* remove p1 */
    free(j);
	  MYPUSH (j, p1, su);
	  p2 = i;
	  MYPUSH (j, p2, su);
	  }
  }


  /* construct lower hull */
  init_stack (&sl);
  p0 = n - 1;
  p1 = n - 2;
  p2 = n - 3;
  MYPUSH (j, p0, sl);
  MYPUSH (j, p1, sl);
  MYPUSH (j, p2, sl);

  for (i = n - 4; i >= 0; i--) {
      if (determ (parray[p0], parray[p1], parray[p2]) < 0.0) {	
     /*counterclockwise */
	  /* we can insert p2 to the hull */
	  p0 = p1;
	  p1 = p2;
	  p2 = i;
	  MYPUSH (j, p2, sl);
    } else {
		/* counterclockwise or colinear */
	  p1 = p2;
	  j=(int*)pop (&sl);		/* remove p2 */
    free(j);
	  j=(int*)pop (&sl);		/* remove p1 */
    free(j);
	  MYPUSH (j, p1, sl);
	  p2 = i;
	  MYPUSH (j, p2, sl);
	  }
  }


#ifdef DEBUG
  printf("Has %d hull pixels\n",count(&sl)+count(&su)-2);
#endif
  pixset->Nh=count(&sl)+count(&su)-1;
  if ((pixset->hull= (hpoint*)malloc((size_t)pixset->Nh*sizeof(hpoint)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  
  i=0;
  j = (int *) pop (&su);
  while (j != 0) {
#ifdef DEBUG
      printf ("%d\n", *j);
#endif
      pixset->hull[i].x=parray[*j].x;
      pixset->hull[i].y=parray[*j].y;
      i++;
      free (j);
      j = (int *) pop (&su);
  }
  j = (int *) pop (&sl); /* throw away first element */
#ifdef DEBUG
  printf ("x %d\n", *j);
#endif
  free(j);
  j = (int *) pop (&sl); 
  while (j != 0) {
#ifdef DEBUG
      printf ("%d\n", *j);
#endif
      pixset->hull[i].x=parray[*j].x;
      pixset->hull[i].y=parray[*j].y;
      i++;
      free (j);
      j = (int *) pop (&sl);
  }
  /* last item is first item */

  free(parray);
  return (0);
}


/* function to build a convex hull of a given set of pixels */
/* pixset: set of pixels to find the boundary
   pixset->pix: list of pixels
   pixset->hull: array of hull (l,m) pixels
   pixset->Nh: no. of boundary (hull) points
*/
int
construct_boundary_f(pixellistf *pixset)
{
  stack_type sl,su; /* two stacks for upper and lowr hulls */
  hpoint *parray;
  int i, *j, n, p0, p1, p2;
  GList *pixval;
  hpixel *ppix;

  n = g_list_length(pixset->pix);

  /* generate random set of points */
  if ((parray = (hpoint *) malloc ((size_t) n * sizeof (hpoint))) == 0) {
      fprintf (stderr, "%s: %d: no free memory", __FILE__, __LINE__);
      exit (-1);
  }
  pixval=pixset->pix;
  for (i = 0; i < n; i++) {
      ppix=pixval->data;
      parray[i].x = ppix->l;
      parray[i].y = ppix->m;
      //printf("insert %f %f\n",parray[i].x,parray[i].y);
      pixval=g_list_next(pixval);
  }

  qsort ((void *) parray, (size_t) n, sizeof (hpoint), &compare_points);

  init_stack (&su);
  /* construct upper hull */
  p0 = 0;
  p1 = 1;
  p2 = 2;
  MYPUSH (j, p0, su);
  MYPUSH (j, p1, su);
  MYPUSH (j, p2, su);

  for (i = 3; i < n; i++) {
      if (determ (parray[p0], parray[p1], parray[p2]) < 0.0) {
     /* clockwise */
	  /* we can insert p2 to the hull */
	  p0 = p1;
	  p1 = p2;
	  p2 = i;
	  MYPUSH (j, p2, su);
    } else {
		/* counterclockwise or colinear */
	  p1 = p2;
	  j=(int*)pop (&su);		/* remove p2 */
    free(j);
	  j=(int*)pop (&su);		/* remove p1 */
    free(j);
	  MYPUSH (j, p1, su);
	  p2 = i;
	  MYPUSH (j, p2, su);
	  }
  }


  /* construct lower hull */
  init_stack (&sl);
  p0 = n - 1;
  p1 = n - 2;
  p2 = n - 3;
  MYPUSH (j, p0, sl);
  MYPUSH (j, p1, sl);
  MYPUSH (j, p2, sl);

  for (i = n - 4; i >= 0; i--) {
      if (determ (parray[p0], parray[p1], parray[p2]) < 0.0) {	
     /*counterclockwise */
	  /* we can insert p2 to the hull */
	  p0 = p1;
	  p1 = p2;
	  p2 = i;
	  MYPUSH (j, p2, sl);
    } else {
		/* counterclockwise or colinear */
	  p1 = p2;
	  j=(int*)pop (&sl);		/* remove p2 */
    free(j);
	  j=(int*)pop (&sl);		/* remove p1 */
    free(j);
	  MYPUSH (j, p1, sl);
	  p2 = i;
	  MYPUSH (j, p2, sl);
	  }
  }


#ifdef DEBUG
  printf("Has %d hull pixels\n",count(&sl)+count(&su)-2);
#endif
  pixset->Nh=count(&sl)+count(&su)-1;
  if ((pixset->hull= (hpoint*)malloc((size_t)pixset->Nh*sizeof(hpoint)))==0) {
    fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
    exit(1);
  }
  
  i=0;
  j = (int *) pop (&su);
  while (j != 0) {
#ifdef DEBUG
      printf ("%d\n", *j);
#endif
      pixset->hull[i].x=parray[*j].x;
      pixset->hull[i].y=parray[*j].y;
      i++;
      free (j);
      j = (int *) pop (&su);
  }
  j = (int *) pop (&sl); /* throw away first element */
#ifdef DEBUG
  printf ("x %d\n", *j);
#endif
  free(j);
  j = (int *) pop (&sl); 
  while (j != 0) {
#ifdef DEBUG
      printf ("%d\n", *j);
#endif
      pixset->hull[i].x=parray[*j].x;
      pixset->hull[i].y=parray[*j].y;
      i++;
      free (j);
      j = (int *) pop (&sl);
  }
  /* last item is first item */

  free(parray);
  return (0);
}





typedef struct line_
{
  double p1x, p2x, p1y, p2y;
} line;
/* functions from sedgewick and dirk stuker */
static int
ccw (double p0x, double p0y, double p1x, double p1y, double p2x, double p2y)
{
  double dx1, dx2, dy1, dy2;
  dx1 = p1x - p0x;
  dy1 = p1y - p0y;
  dx2 = p2x - p0x;
  dy2 = p2y - p0y;
#ifdef DEBUG
  printf ("ccw dx1=%lf, dy1=%lf, dx2=%lf dy2=%lf\n", dx1, dy1, dx2, dy2);
#endif
  if (dx1 * dy2 > dy1 * dx2)
    return (1);
  if (dx1 * dy2 < dy1 * dx2)
    return (-1);
  if (dx1 * dy2 == dy1 * dx2)
    {
      if ((dx1 * dx2 < 0) || (dy1 * dy2 < 0))
  {
    return (-1);
  }
      else if (dx1 * dx1 + dy1 * dy1 >= dx2 * dx2 + dy2 * dy2)
  {
    return (0);
  }
      /* else */
      return (1);
    }
/* will not reach here */
  return (0);
}


/* general intersection, point overlap taken as intersection */
static int
intersect (line e1, line e2)
{
  int ccw11, ccw12, ccw21, ccw22;
  ccw11 = ccw (e1.p1x, e1.p1y, e1.p2x, e1.p2y, e2.p1x, e2.p1y);
  ccw12 = ccw (e1.p1x, e1.p1y, e1.p2x, e1.p2y, e2.p2x, e2.p2y);
  ccw21 = ccw (e2.p1x, e2.p1y, e2.p2x, e2.p2y, e1.p1x, e1.p1y);
  ccw22 = ccw (e2.p1x, e2.p1y, e2.p2x, e2.p2y, e1.p2x, e1.p2y);

#ifdef DEBUG
  printf ("intersect %d %d %d %d\n", ccw11, ccw12, ccw21, ccw22);
#endif
  return (((ccw11 * ccw12 < 0) && (ccw21 * ccw22 < 0))
    || (!ccw11 || !ccw12 || !ccw21 || !ccw22));
}

/* check point is inside the polygon */
/* note the points, and the hull edges has to be in ccw order */
/* check if point (x,y) is inside hull 
   if true, return 1, else 0 */
/* hull : Nhx1 array of points */
int
inside_hull(int Nh, hpoint *hull, double x, double y) {
  line lt, lv, lp;
  int i, j, count;

#ifdef DEBUG
  printf ("inside_poly: %d \n", parray[0]);
#endif
  /* run the vanilla algorithm */
  count = 0;
  j = 0;
  /* test line */
  lt.p1x = x;
  lt.p1y = y;
  srand(time(0)); /* FIXME! */
  lt.p2y = 10.0 + rand();
  lt.p2x = 10.0 + rand();	/* use a random number */

#ifdef DEBUG
  printf ("detect point (%lf,%lf)\n", lt.p1x, lt.p1y);
#endif
  lv.p1x = x;
  lv.p1y = y;
  lv.p2x = x;
  lv.p2y = y;

  for (i=0; i<Nh-1; i++) {
      lp.p1x = hull[i+1].x;
      lp.p1y = hull[i+1].y;
      lp.p2x = hull[i].x;
      lp.p2y = hull[i].y;
      if (intersect (lv, lp)) {
	      return (1);
	    }
      lp.p2x = hull[i+1].x;
      lp.p2y = hull[i+1].y;
      if (!intersect (lp, lt)) {
	      lp.p2x = hull[j].x;
	      lp.p2y = hull[j].y;
	      if (intersect (lp, lt)) {
	        count += 1;
#ifdef DEBUG
	        printf ("case 1\n");
	        printf("intersection (%lf,%lf)-(%lf,%lf) with (%lf,%lf)-(%lf,%lf)\n",
		         lp.p1x, lp.p1y, lp.p2x, lp.p2y, lt.p1x, lt.p1y, lt.p2x,
		         lt.p2y);
	        printf ("intersect with line (%d,%d)\n", i, j);
#endif
	    } else {
	      if ((i != j + 1) 
          && (ccw(lt.p1x, lt.p1y, lt.p2x, lt.p2y, hull[j].x, hull[j].y) 
          * ccw (lt.p1x, lt.p1y, lt.p2x, lt.p2y, hull[i].x, hull[i].y) < 1)) {
		     count += 1;
#ifdef DEBUG
		     printf ("case 2\n");
#endif
		    }
	    }
	    j = i;
	   }

  }

  if ((j != Nh-1)
      && (ccw (lt.p1x, lt.p1y, lt.p2x, lt.p2y,  hull[j].x, hull[j].y)
	  * ccw (lt.p1x, lt.p1y, lt.p2x, lt.p2y, hull[0].x, hull[0].y) == 1)) {
      count -= 1;
#ifdef DEBUG
      printf ("case 3\n");
      printf ("count=%d\n", count);
#endif
  }

  return ((count % 2) == 1);
}
