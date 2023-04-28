/*
 *
 Copyright (C) 2018 Sarod Yatawatta <sarod@users.sf.net>  
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

#ifndef DIRAC_GPUTUNE_H
#define DIRAC_GPUTUNE_H
#ifdef __cplusplus
        extern "C" {
#endif

/********************************************/
#ifdef HAVE_CUDA

/* include tunable parameters of GPU version here */
/* such as number of blocks, threads per block etc */
#ifndef MAX_GPU_ID
#define MAX_GPU_ID 3 /* use 0 (1 GPU), 1 (2 GPUs), ... */
#endif
/* default value for threads per block */
#ifndef DEFAULT_TH_PER_BK 
#define DEFAULT_TH_PER_BK 64
#endif
#ifndef DEFAULT_TH_PER_BK_2
#define DEFAULT_TH_PER_BK_2 32
#endif

#ifndef ARRAY_USE_SHMEM /* use shared memory for calculation station beam */
#define ARRAY_USE_SHMEM 1
#endif
#ifndef ARRAY_MAX_ELEM /* if using shared memory, max possible elements for a station */
/* this is increased from 512 (=16x32) to 16x96 */
#define ARRAY_MAX_ELEM 1536
#endif
/* default GPU heap size (in MB) needed to calculate some shapelet models,
    if model has n0>20 or so, try increasing this and recompiling
   the default GPU values is ~ 8MB */
#ifndef GPU_HEAP_SIZE
#define GPU_HEAP_SIZE 32
#endif
/* shared memory size for element beam coefficients */
#ifndef ELEMENT_MAX_SIZE
#define ELEMENT_MAX_SIZE 64 // should be > (BEAM_ELEM_MODES*(BEAM_ELEM_MODES+1)/2)
#endif

//Max no. of frequencies for a single kernel to work (predict_model.cu) on
//Make this large ~64 for handling data with many channels
#ifndef MODEL_MAX_F
#define MODEL_MAX_F 16
#endif

#endif
/********************************************/

#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* DIRAC_GPUTUNE_H */
