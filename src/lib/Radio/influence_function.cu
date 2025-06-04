/*
 *
 Copyright (C) 2025 Sarod Yatawatta <sarod@users.sf.net>  
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

#include "cuda.h"
#include <cuComplex.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "Dirac_radio.h"

/* enable this for checking for kernel failure */
#define CUDA_DBG

/* matrix multiplications */
/* C=A*B */
__device__ void
amb(const cuFloatComplex*__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],b[0]),cuCmulf(a[1],b[2]));
 c[1]=cuCaddf(cuCmulf(a[0],b[1]),cuCmulf(a[1],b[3]));
 c[2]=cuCaddf(cuCmulf(a[2],b[0]),cuCmulf(a[3],b[2]));
 c[3]=cuCaddf(cuCmulf(a[2],b[1]),cuCmulf(a[3],b[3]));
}
/* C=A*B^H */
__device__ void
ambt(const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b, cuFloatComplex *__restrict__ c) {
 c[0]=cuCaddf(cuCmulf(a[0],cuConjf(b[0])),cuCmulf(a[1],cuConjf(b[1])));
 c[1]=cuCaddf(cuCmulf(a[0],cuConjf(b[2])),cuCmulf(a[1],cuConjf(b[3])));
 c[2]=cuCaddf(cuCmulf(a[2],cuConjf(b[0])),cuCmulf(a[3],cuConjf(b[1])));
 c[3]=cuCaddf(cuCmulf(a[2],cuConjf(b[2])),cuCmulf(a[3],cuConjf(b[3])));
}


/* only use extern if calling code is C */
extern "C"
{

#ifdef CUDA_DBG
static void
checkCudaError(cudaError_t err, const char *file, int line)
{
    if(!err)
        return;
    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
}
#endif


} /* extern "C" */
