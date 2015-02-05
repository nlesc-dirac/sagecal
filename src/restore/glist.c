/*  pdnmesh - a 2D finite element solver
    Copyright (C) 2001-2005 Sarod Yatawatta <sarod@users.sf.net>  
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
  $Id: glist.c,v 1.2 2005/04/18 08:30:00 sarod Exp $
*/

/* generic list */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "restore.h"

#define DEBUG
/* insert elements to the list, returns new location of data in list */
void *
glist_insert(glist *L, const void *data ) {
  glist_node *temp=NULL;
  if (!L) {exit(1);}
  if((temp=(glist_node*)malloc(sizeof(glist_node)))==NULL) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if((temp->data=(void*)malloc(L->datasize))==NULL) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  /* copy data */
  memcpy(temp->data,data,L->datasize);
  if ( L->head== 0 ) { /* empty */ 
    temp->next=temp->prev=0;
    L->tail=L->head=temp;
  }else {
   temp->next=L->head;
   temp->prev=0;
   L->head->prev=temp;
   L->head=temp;
  }

  L->count++;
  /* return address of new data location */
  return (void*)temp->data;
}

/* insert elements to the list, returns new location of data in list */
void *
glist_sorted_insert(glist *L, const void *data) {
  glist_node *temp=NULL;
  glist_node *idx;
  if (!L) {exit(1);}
  if((temp=(glist_node*)malloc(sizeof(glist_node)))==NULL) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  if((temp->data=(void*)malloc(L->datasize))==NULL) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
  }
  /* copy data */
  memcpy(temp->data,data,L->datasize);
  if ( L->head== 0 ) { /* empty */ 
    temp->next=temp->prev=0;
    L->tail=L->head=temp;
    L->count++;
  }else {
   /* find correct location for data */
   idx=L->head;
   while(idx && (L->list_data_comp(idx->data,temp->data) <0))
     idx=idx->next;
   /* now either idx==0 or correct location found */
   if (!idx) { /* at tail */
    if(L->list_data_comp(L->tail->data,temp->data) != 0) {
    L->tail->next=temp;
    temp->prev=L->tail;
    temp->next=0;
    L->tail=temp;
    L->count++;
    } else {
   /* not inserted, free memory */
   L->list_data_free(temp->data);
   free(temp);
   return(0);
    }
   } else if(L->list_data_comp(idx->data,temp->data) !=0) {
   /* if equal, do not insert */
    if (!idx->prev) { /* insert at head */
     temp->prev=idx->prev;
     temp->next=idx;
     idx->prev=temp;
     L->head=temp;
    } else { /* insert at middle */
     idx->prev->next=temp;
     temp->prev=idx->prev;
     temp->next=idx;
     idx->prev=temp;
    }
    L->count++;
   } else {
   /* not inserted, free memory */
   L->list_data_free(temp->data);
   free(temp);
   return(0);
   }
  }
  /* return address of new data location */
  return (void*)temp->data;
}

/* delete the list */
void  
glist_delete(glist *L) {
  glist_node *temp;
  if (L) {
   while(L->head) {
     temp=L->head;
     L->head=temp->next;
     if ( L->head ) {
	 L->head->prev=NULL;
     } 
    L->list_data_free((void *)temp->data);
    free(temp);
   }
 }
  L->head=L->tail=L->iter=NULL;
  L->datasize=0;
  L->count=0;
}

/* initialize the list */
void 
glist_init(glist *L,size_t datasize, void (*list_data_free)(void *)) {
   if (L) {
    L->head=L->tail=0;
    L->datasize=datasize;
    L->list_data_free=list_data_free;
    L->iter=L->head;
    L->count=0;
   }
}

/* initialize the list */
void 
glist_sorted_init(glist *L,size_t datasize, void (*list_data_free)(void *), int (*list_data_comp)(const void *, const void*)) {
   if (L) {
    L->head=L->tail=0;
    L->datasize=datasize;
    L->list_data_free=list_data_free;
    L->list_data_comp=list_data_comp;
    L->iter=L->head;
    L->count=0;
   }
}
/* initialize before traversing the list */
void 
glist_set_iter_forward(glist *L) {
   if (L) {
    L->iter=L->head;
   }
}

/* traverse the list if NULL end has reached */
void *
glist_iter_forward(glist *L) {
   void *p;
   if (!L) return 0;
   if (!L->iter) {L->iter=L->head; return(0);}
   p=L->iter->data;
   L->iter=L->iter->next;
   return(p);
}

/* initialize before traversing the list */
void 
glist_set_iter_backward(glist *L) {
   if (L) {
    L->iter=L->tail;
   }
}

/* traverse the list if NULL end has reached */
void *
glist_iter_backward(glist *L) {
   void *p;
   if (!L) return 0;
   if (!L->iter) {L->iter=L->tail; return(0);}
   p=L->iter->data;
   L->iter=L->iter->prev;
   return(p);
}

/* check if the list is empty */
/* return 1 if empty, 0 if not */
int
glist_empty(glist *L) {
  return( L->head== 0 );
}


/* remove element from the list */
void  *
glist_remove(glist *L) {
  glist_node *temp;
  void *data;
  if (L && L->head) {
     temp=L->head;
     L->head=temp->next;
     if ( L->head ) {
	 L->head->prev=NULL;
     } else {
	 L->tail=NULL;
     }
     data=(void *)temp->data;
     free(temp);

     L->count--;
     return(data);
  }
  return(0);
}

/* remove element from the list if it exists */
/* returns 0 if not deleted (not exist), 1 if deleted */
int
glist_sorted_remove(glist *L, const void *data) {
  glist_node *idx;
  if ( !L || (L->head== 0) ) { /* empty */ 
    return(0);
  }else{
   /* find correct location for data */
   idx=L->head;
   while(idx && (L->list_data_comp(idx->data,data) <0))
     idx=idx->next;
   /* now either idx==0 or correct location found */
   if(idx && L->list_data_comp(idx->data,data) ==0) {
   /* element exists */
#ifdef DEBUG
	   printf("glist_sorted_remove: element found\n");
#endif
    if (!idx->prev) { 
     if(!idx->next) {/* delete at head, tail - one element*/
      L->head=L->tail=NULL;
#ifdef DEBUG
	   printf("glist_sorted_remove: delete at head+tail\n");
#endif
     } else {/* delete at head */
      L->head=idx->next;
      idx->next->prev=NULL;
#ifdef DEBUG
	   printf("glist_sorted_remove: delete at head\n");
#endif
     }
    } else { 
     if(!idx->next) {/* delete at tail */
      L->tail=idx->prev;
      idx->prev->next=NULL;
#ifdef DEBUG
	   printf("glist_sorted_remove: delete at tail\n");
#endif
     } else {/* delete at middle */
      idx->next->prev=idx->prev;
      idx->prev->next=idx->next;
#ifdef DEBUG
	   printf("glist_sorted_remove: delete at middle\n");
#endif
     }
    }
      L->list_data_free((void *)idx->data);
      free(idx);
      return(1);
   } else { /* element does not exist */
     return(0);
   }
    L->count--;
  }
  return(0);
}

/* look for element from the list if it exists */
/* returns 0 if not found (not exist), else the address*/
void *
glist_sorted_lookup(glist *L, const void *data) {
  glist_node *idx;
  if ( !L || (L->head== 0) ) { /* empty */ 
    return(0);
  }else{
   /* find correct location for data */
   idx=L->head;
   while(idx && (L->list_data_comp(idx->data,data) <0))
     idx=idx->next;
   /* now either idx==0 or correct location found */
   if(idx && L->list_data_comp(idx->data,data) ==0) {
   /* element exists */
#ifdef DEBUG
	   printf("glist_sorted_lookup: element found\n");
#endif
     return(idx->data);
   } else { /* element does not exist */
     return(0);
   }
  }
  return(0);
}
