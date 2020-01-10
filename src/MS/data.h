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
#include <casacore/ms/MeasurementSets/MSIter.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableVector.h>
#include <casacore/tables/Tables/TableRecord.h>
#include <casacore/tables/Tables/TableColumn.h>
#include <casacore/tables/Tables/TableIter.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Cube.h>
#include <fstream>
#include <math.h>
#include <complex>


using namespace casacore;

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

       double fratio; /* flagged data ratio = flagged/total, not counting data excluded from uv cut */
                      /* if 1, all usable data are flagged */
    };

    /* Station beam info */
    struct LBeam {
      double *time_utc; /* time coord UTC (s), size tileszx1, 
                         convert from MJD (s) to JD (days) */
      int *Nelem; /* no of elements in each station, size Nx1 */
      /* position (ITRF) of stations (m)
       later changed to logitude,latitude,height (rad,rad,m) */
      double *sx; /* x: size Nx1 */
      double *sy; /* y: ... */
      double *sz; /* z: ... */
      /* x,y,z coords of elements, projected, converted to ITRF (m) */
      double **xx; /* x coord pointer, size Nx1, each *x: x coord of station, size Nelem[]x1 */
      double **yy; /* y ... */
      double **zz; /* z ... */
      /* pointing center of beams (only one) (could be different from phase center) */
      double p_ra0;
      double p_dec0;
    };


    /* read Auxilliary info and setup memory */
    void readAuxData(const char *fname, IOData *data);
    void readAuxData(const char *fname, IOData *data, LBeam *binfo);

    void readAuxDataList(vector<string> msnames, IOData *data);

    void readMSlist(char *fname, vector<string> *msnames);
    /* load data using MS Iterator */
    void loadData(Table t, IOData iodata, double *fratio);
    void loadData(Table t, IOData iodata, LBeam binfo, double *fratio);
    void loadDataMinibatch(Table ti, Data::IOData iodata, int minibatch, double *fratio);
    void loadDataMinibatch(Table ti, Data::IOData iodata, LBeam binfo, int minibatch, double *fratio);

    void loadDataList(vector<MSIter*> msitr, Data::IOData iodata, double *fratio);
    /* write back data using MS Iterator */
    void writeData(Table t, IOData iodata);
    void writeDataMinibatch(Table ti, Data::IOData iodata, int minibatch);
    void writeDataList(vector<MSIter*> msitr, IOData iodata);
    void freeData(IOData data);
    void freeData(IOData data, LBeam binfo);

    extern int numChannels; 
    extern unsigned long int numRows;

    struct float2 {
      float x,y;
    };


    extern char *TableName; /* MS name */
    extern char *MSlist; /* text file with MS names */
    extern char *MSpattern; /* pattern to match all MS names used in calibration */
    extern float min_uvcut;
    extern float max_uvcut;
    extern float max_uvtaper;
    extern casacore::String DataField; /* input column DATA/CORRECTED_DATA */
    extern casacore::String OutField; /* output column DATA/CORRECTED_DATA */
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
    extern char *initsolfile;
    extern char *ignorefile;
    extern double nulow,nuhigh;
    extern int randomize;
    extern int whiten;
    extern int DoSim; /* if 1, simulation mode */
    extern int doChan; /* if 1, solve for each channel in multi channel data */
    extern int doBeam; /* if 1, predict (LOFAR) beam array factor */
    extern int DoDiag; /* if >0, enables diagnostics (Leverage) 1: write leverage as output (no residual), 2: only calculate fractions of leverage/noise */
    extern int phaseOnly; /* if >0, and if any correction is done, extract phase and do phase only correction */

    /* stochastic calibration parameters */
    extern int stochastic_calib_epochs; /* if > 1, stochastic calibration */
    extern int stochastic_calib_minibatches; /* number of minibatches the data is split */
    extern int stochastic_calib_bands; /* number of bands to split channels for solutions over full band */
    /* federated averaging, global - local constraint regularization factor */
    extern double federated_reg_alpha;

    /* distributed sagecal parameters */
    extern int Nadmm; /* ADMM iterations >=1 */
    extern int Npoly; /* polynomial order >=1 */
    extern int PolyType; /* what kind on polynomials to use 0,1,2,3 */
    extern double admm_rho; /* regularization */
    extern char *admm_rho_file; /* text file for regularization of each cluster */
    extern int aadmm; /* if >0, enable adaptive update of rho */
    /* for debugging, upper limit on time slots */
    extern int Nmaxtime;
    /* skipping initial timeslots */
    extern int Nskip;
    extern int verbose; /* if >0, enable verbose output */
    extern int mdl; /* if given, calculate AIC/MDL for different poly configs and find minimum */
    extern int GPUpredict; /* if given, use GPU for model calculation */
    extern int heapsize; /* heap size in GPU (MB), for using malloc() */
    /* for client server mode */
    extern int servermode; /* 0: client, 1: server, else default operation */
    extern char *servername; /* server host name or ip address */
    extern char *portnumber; /* which port number to use for communication */
}
#endif //__DATA_H__
