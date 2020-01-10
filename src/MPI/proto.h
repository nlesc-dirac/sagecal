/*
 *
 Copyright (C) 2014 Sarod Yatawatta <sarod@users.sf.net>  
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

#ifndef __PROTO_H__
#define __PROTO_H__

/*** MPI message tags *****/
#ifndef TAG_MSNAME
#define TAG_MSNAME 100
#endif
#ifndef TAG_MSAUX
#define TAG_MSAUX 101
#endif
#ifndef TAG_CTRL 
#define TAG_CTRL 102
#endif
#ifndef TAG_YDATA
#define TAG_YDATA 103
#endif
#ifndef TAG_CONSENSUS
#define TAG_CONSENSUS 104
#endif
#ifndef TAG_RHO
#define TAG_RHO 105
#endif
#ifndef TAG_FRATIO
#define TAG_FRATIO 106
#endif
#ifndef TAG_RHO_UPDATE
#define TAG_RHO_UPDATE 107
#endif
#ifndef TAG_CONSENSUS_OLD
#define TAG_CONSENSUS_OLD 108
#endif
#ifndef TAG_CHUNK
#define TAG_CHUNK 109
#endif





/* control flags */
#ifndef CTRL_START
#define CTRL_START 90
#endif
#ifndef CTRL_END
#define CTRL_END 91
#endif
#ifndef CTRL_DONE
#define CTRL_DONE 92
#endif
#ifndef CTRL_RESET
#define CTRL_RESET 93
#endif
#ifndef CTRL_SKIP
#define CTRL_SKIP 94
#endif



/*** structure to store info in the master, similar to IOData */
typedef struct MPIData_ {
       int N; /* no of stations */
       int M; /* effective clusters */
       int tilesz;
       int Nms; /* no. of MS (not necessarily equivalent to slaves) */
       int totalt; /* total no of time slots */
       double *freqs; /* channel freqs, size Nmsx 1 */
       double freq0; /* reference freq (use average) */

} MPIData;

/********* main.cpp ***************************************************/
extern void
print_copyright(void);
extern void
print_help(void);
extern void
ParseCmdLine(int ac, char **av);


/********* sagecal_master.cpp ******************************************/
extern int 
sagecal_master(int argc, char **argv);


/********* sagecal_slave.cpp ******************************************/
extern int 
sagecal_slave(int argc, char **argv);


/********* sagecal_stochastic_master.cpp ******************************************/
extern int 
sagecal_stochastic_master(int argc, char **argv);

/********* sagecal_stochastic_slave.cpp ******************************************/
extern int 
sagecal_stochastic_slave(int argc, char **argv);


#endif /* __PROTO_H__ */
