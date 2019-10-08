/*
 *
 Copyright (C) 2019 Sarod Yatawatta <sarod@users.sf.net>  
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

#ifndef __SAGECALMAIN_H__
#define __SAGECALMAIN_H__

#ifndef LMCUT
#define LMCUT 40
#endif


/********* main.cpp ***************************************************/
extern void
print_copyright(void);
extern void
print_help(void);
extern void
ParseCmdLine(int ac, char **av);


/********* fullbatch_mode.cpp *****************************************/
extern int
run_fullbatch_calibration(void);

/********* minibatch_mode.cpp *****************************************/
extern int
run_minibatch_calibration(void);

/********* minibatch_consensus_mode.cpp *****************************************/
extern int
run_minibatch_consensus_calibration(void);
#endif /* __SAGECALMAIN_H__ */

