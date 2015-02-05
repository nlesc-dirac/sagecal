/* FST - a Fast Shapelet Transformer
 *
   Copyright (C) 2006 Sarod Yatawatta <sarod@users.sf.net>  
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
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "shapelet.h"

/* evaluate Hermite polynomial value using recursion
 */
double 
H_e(double x, int n) {
	if(n==0) return 1.0;
	if(n==1) return 2*x;
	return 2*x*H_e(x,n-1)-2*(n-1)*H_e(x,n-2);
}
