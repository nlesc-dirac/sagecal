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

#ifndef __DATA_H__
#define __DATA_H__
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <ms/MeasurementSets/MSIter.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/TableVector.h>
#include <tables/Tables/TableRecord.h>
#include <tables/Tables/TableColumn.h>
#include <tables/Tables/TableIter.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/ArrayColumn.h>
#include <casa/Arrays/Array.h>
#include <casa/Arrays/Cube.h>
#include <fstream>
#include <math.h>
#include <complex>


using namespace casa;

namespace Data
{

    struct IOData {
       int N; /* no of stations */
       int Nbase; /* baselines, exclude autocorrelations */
       int tilesz;
       int Nchan; /* total no of channels */
       int Nms; /* no. of MS */
       int *NchanMS; /* total channels per MS : Nms x 1 vector */
       double deltat; /* integration time (s) */
       int totalt; /* total no of time slots */
       double ra0; /* phase center */
       double dec0; /* phase center */
       double *u; /* uvw coords, size Nbase*tilesz x 1 */
       double *v;
       double *w;
       double *x; /* averaged data, size Nbase*8*tilez x 1
       ordered by XX(re,im),XY(re,im),YX(re,im),YY(re,im), baseline, timeslots */
       double *xo; /* unaveraged data, size Nbase*8*tilesz*Nchan x 1
       ordered by  XX(re,im),XY(re,im),YX(re,im),YY(re,im), baseline, timeslots, channel */
       double *freqs; /* channel freqs, size Nchan x 1 */
       double freq0; /* averaged freq */
       double *flag; /* double for conforming with old routines size Nbase*tilesz x 1 */
       double deltaf; /* total bandwidth for freq. smearing */
    };

    /* read Auxilliary info and setup memory */
    void readAuxData(const char *fname, IOData *data);
    void readAuxDataList(vector<string> msnames, IOData *data);

    void readMSlist(char *fname, vector<string> *msnames);
    /* load data using MS Iterator */
    void loadData(Table t, IOData iodata);
    void loadDataList(vector<MSIter*> msitr, Data::IOData iodata);
    /* write back data using MS Iterator */
    void writeData(Table t, IOData iodata);
    void writeDataList(vector<MSIter*> msitr, IOData iodata);
    void freeData(IOData data);

    extern int numChannels; 
    extern unsigned long int numRows;

   struct float2 {
     float x,y;
   };


    extern char *TableName; /* MS name */
    extern char *MSlist; /* text file with MS names */
    extern float min_uvcut;
    extern float max_uvcut;
    extern float max_uvtaper;
    extern casa::String DataField; /* input column DATA/CORRECTED_DATA */
    extern casa::String OutField; /* output column DATA/CORRECTED_DATA */
    extern int TileSize; //Tile size
    extern int Nt; /* no of worker threads */
    extern char *SkyModel; /* sky model file */
    extern char *Clusters; /* cluster file */
    extern int format; /* sky model format 0: LSM, 1: LSM with 3 order spec idx*/
 
    /* sagecal paramters */
    extern int max_emiter;
    extern int max_iter;
    extern int max_lbfgs;
    extern int lbfgs_m;
    extern int gpu_threads;
    extern int linsolv;
    extern int solver_mode;
    extern int ccid;
    extern double rho;
    extern char *solfile;
    extern char *ignorefile;
    extern double nulow,nuhigh;
    extern int randomize;
    extern int whiten;
    extern int DoSim; /* if 1, simulation mode */
    extern int doChan; /* if 1, solve for each channel in multi channel data */
    extern int DoDiag; /* if >0, enables diagnostics (Leverage) 1: write leverage as output (no residual), 2: only calculate fractions of leverage/noise */

    /* distributed sagecal parameters */
    extern int Nadmm; /* ADMM iterations >=1 */
    extern int Npoly; /* polynomial order >=1 */
    extern double admm_rho; /* regularization */
    extern char *admm_rho_file; /* text file for regularization of each cluster */
    /* for debugging, upper limit on time slots */
    extern int Nmaxtime;
}
#endif //__DATA_H__
