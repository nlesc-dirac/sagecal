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

#include "data.h"
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <casacore/casa/Quanta/Quantum.h>

#include "Radio.h"
#include "Dirac.h"
#include "sagecalmain.h"


using namespace std;
using namespace Data;

void
print_copyright(void) {
  cout<<"SAGECal 0.7.1 (C) 2011-2020 Sarod Yatawatta"<<endl;
}


void
print_help(void) {
   cout << "Usage:" << endl;
   cout<<"sagecal -d MS -s sky.txt -c cluster.txt"<<endl;
   cout<<"or"<<endl;
   cout<<"sagecal -f MSlist -s sky.txt -c cluster.txt"<<endl;
   cout<<endl<<"Stochastic calibration:"<<endl;
   cout<<"sagecal -d MS -s sky.txt -c cluster.txt -N epochs -M minibatches"<<endl;
   cout<<endl;
   cout << "-d MS name" << endl;
   cout << "-f MSlist: text file with MS names" << endl;
   cout << "-s sky.txt: sky model file"<< endl;
   cout << "-c cluster.txt: cluster file"<< endl;
   cout << "-p solutions.txt: if given, save solution in this file, or read the solutions if doing simulations"<< endl;
   cout << "-F sky model format: 0: LSM, 1: LSM with 3 order spectra : default "<< Data::format<<endl;
   cout << "-I input column (DATA/CORRECTED_DATA/...) : default " <<Data::DataField<< endl;
   cout << "-O ouput column (DATA/CORRECTED_DATA/...) : default " <<Data::OutField<< endl;
   cout << "-e max EM iterations : default " <<Data::max_emiter<< endl;
   cout << "-g max iterations  (within single EM) : default " <<Data::max_iter<< endl;
   cout << "-l max LBFGS iterations : default " <<Data::max_lbfgs<< endl;
   cout << "-m LBFGS memory size : default " <<Data::lbfgs_m<< endl;
   cout << "-n no of worker threads : default "<<Data::Nt << endl;
   cout << "-t tile size : default " <<Data::TileSize<< endl;
   cout << "-a 0,1,2,3 : if "<<SIMUL_ONLY<<", only simulate, if "<<SIMUL_ADD<<", simulate and add to input, if "<<SIMUL_SUB<<", simulate and subtract from input (For a>0, multiplied by solutions if solutions file is also given): default " <<Data::DoSim<< endl;
   cout << "-z ignore_clusters: if only doing a simulation, ignore the cluster ids listed in this file" << endl;
   cout << "-b 0,1 : if 1, solve for each channel: default " <<Data::doChan<< endl;
   cout << "-B 0,1 : if 1, predict array beam: default " <<Data::doBeam<< endl;
#ifdef HAVE_CUDA
   cout << "-E 0,1 : if 1, use GPU for model computing: default " <<Data::GPUpredict<< endl;
#endif
   cout << "-x exclude baselines length (lambda) lower than this in calibration : default "<<Data::min_uvcut << endl;
   cout << "-y exclude baselines length (lambda) higher than this in calibration : default "<<Data::max_uvcut << endl;
   cout <<endl<<"Advanced options:"<<endl;
   cout << "-k cluster_id : correct residuals with solution of this cluster : default "<<Data::ccid<< endl;
   cout << "-o robust rho, robust matrix inversion during correction: default "<<Data::rho<< endl;
   cout << "-J 0,1 : if >0, use phase only correction: default "<<Data::phaseOnly<< endl;
   cout << "-j 0,1,2... 0 : OSaccel, 1 no OSaccel, 2: OSRLM, 3: RLM, 4: RTR, 5: RRTR, 6:NSD : default "<<Data::solver_mode<< endl;
   cout << "-L robust nu, lower bound: default "<<Data::nulow<< endl;
   cout << "-H robust nu, upper bound: default "<<Data::nuhigh<< endl;
   cout << "-W pre-whiten data: default "<<Data::whiten<< endl;
   cout << "-R randomize iterations: default "<<Data::randomize<< endl;
#ifdef HAVE_CUDA
   cout << "-S GPU heap size (MB): default "<<Data::heapsize<< endl;
#endif
   cout << "-D 0,1,2 : if >0, enable diagnostics (Jacobian Leverage) 1 replace Jacobian Leverage as output, 2 only fractional noise/leverage is printed: default " <<Data::DoDiag<< endl;
   cout << "-q solutions.txt: if given, initialize solutions by reading this file (need to have the same format as a solution file, only solutions for 1 timeslot needed)"<< endl;

   cout<<endl<<"Stochastic mode:"<<endl;
   cout << "-N epochs, if >0, use stochastic calibration: default "<<Data::stochastic_calib_epochs<< endl;
   cout << "-M minibatches, must be >0, split data to this many minibatches: default "<<Data::stochastic_calib_minibatches<< endl;
   cout << "-w mini-bands, must be >0, split channels to this many mini-bands for bandpass calibration: default "<<Data::stochastic_calib_bands<< endl;

   cout<<endl<<"Stochastic mode with consensus:"<<endl;
   cout << "-A ADMM iterations: default " <<Data::Nadmm<< endl;
   cout << "-P consensus polynomial terms: default " <<Data::Npoly<< endl;
   cout << "-Q consensus polynomial type (0,1,2,3): default " <<Data::PolyType<< endl;
   cout << "-r regularization factor: default " <<Data::admm_rho<< endl;
   cout << "Note: In stochastic mode, no hybrid solutions are allowed."<<endl<<"All clusters should have 1 in the second column of cluster file."<<endl;

   cout <<"Report bugs to <sarod@users.sf.net>"<<endl;

}

void
ParseCmdLine(int ac, char **av) {
    print_copyright();
    int c;
    if(ac < 2)
    {
        print_help();
        exit(0);
    }
    while((c=getopt(ac, av, ":a:b:c:d:e:f:g:j:k:l:m:n:o:p:q:r:s:t:w:x:y:z:A:B:D:E:F:H:I:J:L:M:N:O:P:Q:R:S:W:E:h"))!= -1)
    {
        switch(c)
        {
            case 'd':
                TableName = optarg;
                break;
            case 'f':
                MSlist=optarg;
                break;
            case 's':
                SkyModel= optarg;
                break;
            case 'c':
                Clusters= optarg;
                break;
            case 'p':
                solfile= optarg;
                break;
            case 'q':
                initsolfile= optarg;
                break;
            case 'g':
                max_iter= atoi(optarg);
                break;
            case 'a':
                DoSim= atoi(optarg);
                if (DoSim<0) { DoSim=1; }
                break;
            case 'b':
                doChan= atoi(optarg);
                if (doChan>1) { doChan=1; }
                break;
            case 'B':
                doBeam= atoi(optarg);
                if (doBeam>1) { doBeam=1; }
                break;
            case 'E':
                GPUpredict=atoi(optarg);
                break;
            case 'F':
                format= atoi(optarg);
                if (format>1) { format=1; }
                break;
            case 'e':
                max_emiter= atoi(optarg);
                break;
            case 'l':
                max_lbfgs= atoi(optarg);
                break;
            case 'm':
                lbfgs_m= atoi(optarg);
                break;
            case 'j':
                solver_mode= atoi(optarg);
                break;
            case 't':
                TileSize = atoi(optarg);
                break;
            case 'I':
                DataField = optarg;
                break;
            case 'O':
                OutField = optarg;
                break;
            case 'n':
                Nt= atoi(optarg);
                break;
            case 'k':
                ccid= atoi(optarg);
                break;
            case 'o':
                rho= atof(optarg);
                break;
            case 'L':
                nulow= atof(optarg);
                break;
            case 'H':
                nuhigh= atof(optarg);
                break;
            case 'R':
                randomize= atoi(optarg);
                break;
            case 'W':
                whiten= atoi(optarg);
                break;
            case 'J':
                phaseOnly= atoi(optarg);
                break;
#ifdef HAVE_CUDA
            case 'S':
                heapsize= atoi(optarg);
                break;
#endif
            case 'x':
                Data::min_uvcut= atof(optarg);
                break;
            case 'y':
                Data::max_uvcut= atof(optarg);
                break;
            case 'z':
                ignorefile= optarg;
                break;
            case 'N':
                Data::stochastic_calib_epochs= atoi(optarg);
                break;
            case 'M':
                Data::stochastic_calib_minibatches= atoi(optarg);
                break;
            case 'w':
                Data::stochastic_calib_bands= atoi(optarg);
                break;
            case 'A':
                Nadmm= atoi(optarg);
                break;
            case 'P':
                Npoly= atoi(optarg);
                break;
            case 'Q':
                PolyType= atoi(optarg);
                break;
            case 'r':
                admm_rho= atof(optarg);
                break;
            case 'D':
                DoDiag= atoi(optarg);
                if (DoDiag<0) { DoDiag=0; }
                break;
            case 'h':
                print_help();
                exit(1);
            case ':':
                cout<<"Error: A value is missing for one of the options"<<endl;
                print_help();
                exit(1);
            default:
                print_help();
                exit(1);
        }
    }

    if (TableName) {
     cout<<" MS: "<<TableName<<endl;
    } else if (MSlist && !stochastic_calib_epochs) {
     cout<<" MS list: "<<MSlist<<endl;
    } else {
     print_help();
     exit(1);
    }
    cout<<"Selecting baselines > "<<min_uvcut<<" and < "<<max_uvcut<<" wavelengths."<<endl;
    if (!DoSim) {
    cout<<"Using ";
    if (solver_mode==SM_LM_LBFGS || solver_mode==SM_OSLM_LBFGS || solver_mode==SM_RTR_OSLM_LBFGS ||  solver_mode==SM_NSD_RLBFGS) {
     cout<<"Gaussian noise model for solver."<<endl;
    } else {
     cout<<"Robust noise model for solver with degrees of freedom ["<<nulow<<","<<nuhigh<<"]."<<endl;
    }
    } else {
     cout<<"Only doing simulation (with possible correction for cluster id "<<ccid<<")."<<endl;
    }
}

int
main(int argc, char **argv) {
    ParseCmdLine(argc, argv);
    
    if (!Data::SkyModel || !Data::Clusters || !(Data::TableName || Data::MSlist)) {
      print_help();
      exit(1);
    }
    if (Data::stochastic_calib_epochs>0) {
      if (Data::Nadmm>1 && Data::stochastic_calib_bands>1) {
       /* stochastic calibration with consensus */ 
       run_minibatch_consensus_calibration();
      } else {
       /* stochastic calibration */
       run_minibatch_calibration();
      }
    } else {
      /* normal calibration */
      run_fullbatch_calibration();
    }
   return 0;
}
