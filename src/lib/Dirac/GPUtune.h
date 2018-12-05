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

#ifndef GPUTUNE_H
#define GPUTUNE_H
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

#endif
/********************************************/

#ifdef __cplusplus
     } /* extern "C" */
#endif
#endif /* GPUTUNE_H */
