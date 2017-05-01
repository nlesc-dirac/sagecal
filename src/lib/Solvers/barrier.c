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

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Dirac.h"
/* implementation of a barrier to sync threads.
  The barrier has two doors (enter and exit). Only one door 
  can be open at a time. Initially the enter door is open.
  All threads that enter the barrier are sleeping (wait).
  The last thread to enter the barrier will 
   1)close the enter door
   2)wakeup all sleeping threads.
   3)open the exit door.
  So the woken up threads will leave the barrier one by 
  one, as they are awoken. The last thread to leave the barrier
  will
   1)open the enter door 
   2)close the exit door,
  So finally the barrier reaches its initial state
*/

/* initialize barrier */
/* N - no. of accomodated threads */
void
init_th_barrier(th_barrier *barrier, int N)
{
 barrier->tcount=0; /* initially empty */
 barrier->nthreads=N;
 pthread_mutex_init(&barrier->enter_mutex,NULL);
 pthread_mutex_init(&barrier->exit_mutex,NULL);
 pthread_cond_init(&barrier->lastthread_cond,NULL);
 pthread_cond_init(&barrier->exit_cond,NULL);
}
/* destroy barrier */
void
destroy_th_barrier(th_barrier *barrier)
{
 pthread_mutex_destroy(&barrier->enter_mutex);
 pthread_mutex_destroy(&barrier->exit_mutex);
 pthread_cond_destroy(&barrier->lastthread_cond);
 pthread_cond_destroy(&barrier->exit_cond);
 barrier->tcount=barrier->nthreads=0;
}
/* the main operation of the barrier */
void
sync_barrier(th_barrier *barrier)
{
 /* trivial case */
 if(barrier->nthreads <2) return;
 /* else */
 /* new threads enters the barrier. Now close the entry door
  so that other threads cannot enter the barrier until we are done */
 pthread_mutex_lock(&barrier->enter_mutex);
 /* next lock the exit mutex - no threads can leave either */
 pthread_mutex_lock(&barrier->exit_mutex);
 /* now check to see if this is the last expected thread */
 if( ++(barrier->tcount) < barrier->nthreads) {
  /* no. this is not the last thread. so open the entry door */
  pthread_mutex_unlock(&barrier->enter_mutex);
 /* go to sleep */
  pthread_cond_wait(&barrier->exit_cond,&barrier->exit_mutex);
 } else {
 /* this is the last thread */
 /* wakeup sleeping threads */
 pthread_cond_broadcast(&barrier->exit_cond);
 /* go to sleep until all threads are woken up
   and leave the barrier */
 pthread_cond_wait(&barrier->lastthread_cond,&barrier->exit_mutex);
/* now all threads have left the barrier. so open the entry door again */
 pthread_mutex_unlock(&barrier->enter_mutex);
 } 
 /* next to the last thread leaving the barrier */
 if(--(barrier->tcount)==1) {
  /* wakeup the last sleeping thread */
  pthread_cond_broadcast(&barrier->lastthread_cond);
 }
 pthread_mutex_unlock(&barrier->exit_mutex);
} 
 


/* master and two slaves */
//int
//main(int argc, char *argv[]) {
// th_pipeline p;
// 
// gbdata g;
//
// init_pipeline(&p,&g);
//sync_barrier(&(p.gate1)); /* stop at gate 1 */
//   g.status=0; /* master work */
//sync_barrier(&(p.gate2)); /* stop at gate 2 */
// //exec_pipeline(&p);
//sync_barrier(&(p.gate1)); /* stop at gate 1 */
// g.status=10; /* master work */
//sync_barrier(&(p.gate2)); /* stop at gate 2 */
// //exec_pipeline(&p);
// destroy_pipeline(&p);
// /* still need to free slave_data structs, from data */
// return 0;
//}
