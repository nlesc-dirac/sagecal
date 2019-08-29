/*
 *
 Copyright (C) 2006-2015 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "Dirac.h"


#include <nvml.h>

//#define MPI_BUILD
#ifdef MPI_BUILD
#include <mpi.h>
#endif

//#define DEBUG

/* return random value in 0,1,..,maxval */
#ifndef MPI_BUILD
static int
random_pick(int maxval, taskhist *th) {
  double rat=(double)random()/(double)RAND_MAX;
  double y=rat*(double)(maxval+1);
  int x=(int)floor(y); 
  return x;
}
#endif

void
init_task_hist(taskhist *th) {
 th->prev=-1;
 th->rseed=0;
 pthread_mutex_init(&th->prev_mutex,NULL);
}

void
destroy_task_hist(taskhist *th) {
 th->prev=-1;
 th->rseed=0;
 pthread_mutex_destroy(&th->prev_mutex);
}

/* select a GPU from 0,1..,max_gpu
   in such a way to allow load balancing */
int
select_work_gpu(int max_gpu, taskhist *th) {
  if (!max_gpu) return 0; /* no need to spend time if only one GPU is available */
#ifdef MPI_BUILD
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /* check if max_gpu > no. of actual devices */
  int actual_devcount;
  cudaGetDeviceCount(&actual_devcount);
  if (max_gpu+1>actual_devcount) {
   return rank%(actual_devcount);
  }
  return rank%(max_gpu+1); /* modulo value */
#endif

#ifndef MPI_BUILD
  /* sequentially query the devices to find 
     one with the min load/memory usage */
  nvmlReturn_t result;
  result = nvmlInit();
  int retval;
  int minid=-1;
  int maxid=-1;


  if (result!=NVML_SUCCESS) {
    fprintf(stderr,"%s: %d: cannot access NVML\n",__FILE__,__LINE__);
    /* return random pick */
    retval=random_pick(max_gpu, th);
    /* if this matches the previous value, select again */
    pthread_mutex_lock(&th->prev_mutex);
    while (retval==th->prev) {
     retval=random_pick(max_gpu, th);
    }

    th->prev=retval;
    pthread_mutex_unlock(&th->prev_mutex);
    return retval;
  } else {
    /* iterate */
    nvmlDevice_t device;
    nvmlUtilization_t nvmlUtilization;
    nvmlMemory_t nvmlMemory; 
    unsigned int min_util=101; /* GPU utilization */
    unsigned int max_util=0; /* GPU utilization */
    unsigned long long int max_free=0; /* max free memory */
    unsigned long long int min_free=ULLONG_MAX; /* max free memory */
    int ci;
    for (ci=0; ci<=max_gpu; ci++) {
      result=nvmlDeviceGetHandleByIndex(ci, &device);
      result=nvmlDeviceGetUtilizationRates(device, &nvmlUtilization);
      result=nvmlDeviceGetMemoryInfo(device, &nvmlMemory); 
      if (min_util>nvmlUtilization.gpu) {
          min_util=nvmlUtilization.gpu;
          minid=ci;
      }
      if (max_util<nvmlUtilization.gpu) {
          max_util=nvmlUtilization.gpu;
      }
      if (max_free<nvmlMemory.free) {
         max_free=nvmlMemory.free;
         maxid=ci;
      }
      if (min_free>nvmlMemory.free) {
         min_free=nvmlMemory.free;
      }
    }
    result = nvmlShutdown();
    /* give priority for selection a GPU with max free memory,
       if there is a tie, use min utilization as second criterion */ 
    /* if all have 0 usage, again use random */
    if (max_free==min_free && max_util==min_util) {
     retval=random_pick(max_gpu,th);
     /* if this value matches previous one, select again */
     pthread_mutex_lock(&th->prev_mutex);
     while(retval==th->prev) {
      retval=random_pick(max_gpu,th);
     }
     th->prev=retval;
     pthread_mutex_unlock(&th->prev_mutex);
     return retval;
    } else {
     if (max_free==min_free) { /* all cards have equal free mem */
       retval=(int)minid;
     } else {
       retval=(int)maxid;
     }
    }
  }

  /* update last pick */
  pthread_mutex_lock(&th->prev_mutex);
  th->prev=retval;
  pthread_mutex_unlock(&th->prev_mutex);
  
  return retval;
#endif
}
