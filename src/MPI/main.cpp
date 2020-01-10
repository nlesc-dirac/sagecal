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

#include "data.h"
#include "proto.h"
#include <fstream>
#include <vector> 
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <Dirac.h>
#include <Radio.h>
#include <mpi.h>

using namespace std;
using namespace Data;

void
print_copyright(void) {
  cout<<"SAGECal-MPI 0.7.1 (C) 2011-2020 Sarod Yatawatta"<<endl;
}


void
print_help(void) {
   cout << "Usage:" << endl;
   cout<<"mpirun (MPI options) sagecal-mpi -f '*.MS' -s sky.txt -c cluster.txt (other options)"<<endl;
   cout<<endl;
   cout<<"Mandatory options:"<<endl;
   cout << "-f '*.MS': text pattern to search MS names" << endl;
   cout << "-s sky.txt: sky model file"<< endl;
   cout << "-c cluster.txt: cluster file"<< endl;
   cout<<endl;
   cout<<"Other options:"<<endl;
   cout << "-p solutions.txt: if given, save (global) solutions in this file, but slaves will always write to 'XXX.MS.solutions'"<< endl;
   cout << "-F sky model format: 0: LSM, 1: LSM with 3 order spectra : default "<< Data::format<<endl;
   cout << "-I input column (DATA/CORRECTED_DATA/...) : default " <<Data::DataField<< endl;
   cout << "-O ouput column (DATA/CORRECTED_DATA/...) : default " <<Data::OutField<< endl;
   cout << "-e max EM iterations : default " <<Data::max_emiter<< endl;
   cout << "-g max iterations  (within single EM) : default " <<Data::max_iter<< endl;
   cout << "-l max LBFGS iterations : default " <<Data::max_lbfgs<< endl;
   cout << "-m LBFGS memory size : default " <<Data::lbfgs_m<< endl;
   cout << "-n no of worker threads : default "<<Data::Nt << endl;
   cout << "-t tile size : default " <<Data::TileSize<< endl;
   cout << "-B 0,1 : if 1, predict array beam: default " <<Data::doBeam<< endl;
#ifdef HAVE_CUDA
   cout << "-E 0,1 : if >0, use GPU for model computing: default " <<Data::GPUpredict<< endl;
#endif
   cout << "-A ADMM iterations: default " <<Data::Nadmm<< endl;
   cout << "-P consensus polynomial terms: default " <<Data::Npoly<< endl;
   cout << "-Q consensus polynomial type (0,1,2,3): default " <<Data::PolyType<< endl;
   cout << "-r regularization factor: default " <<Data::admm_rho<< endl;
   cout << "-G regularization factor of each cluster (text file instead of -r, has to match _exactly_ the cluster file's first 2 columns): default : None" << endl;
   cout << "-C if >0, adaptive update of regularization factor: default "<<Data::aadmm<< endl;
   cout << "-x exclude baselines length (lambda) lower than this in calibration : default "<<Data::min_uvcut << endl;
   cout << "-y exclude baselines length (lambda) higher than this in calibration : default "<<Data::max_uvcut << endl;
   cout <<endl<<"Advanced options:"<<endl;
   cout << "-k cluster_id : correct residuals with solution of this cluster : default "<<Data::ccid<< endl;
   cout << "-o robust rho, robust matrix inversion during correction: default "<<Data::rho<< endl;
   cout << "-J 0,1 : if >0, use phase only correction: default "<<Data::phaseOnly<< endl;
   cout << "-j 0,1,2... 0 : OSaccel, 1 no OSaccel, 2: OSRLM, 3: RLM, 4: RTR, 5: RRTR: 6: NSD, default "<<Data::solver_mode<< endl;
   cout << "-L robust nu, lower bound: default "<<Data::nulow<< endl;
   cout << "-H robust nu, upper bound: default "<<Data::nuhigh<< endl;
   cout << "-W pre-whiten data: default "<<Data::whiten<< endl;
   cout << "-R randomize iterations: default "<<Data::randomize<< endl;
#ifdef HAVE_CUDA
   cout << "-S GPU heap size (MB): default "<<Data::heapsize<< endl;
#endif
   cout << "-T stop after this number of solutions (0 means no limit): default "<<Data::Nmaxtime<< endl;
   cout << "-K skip this number of solutions before starting calibration: default "<<Data::Nskip<< endl;
   cout << "Note: if -K a -T b, then calibration will start at 'a' and end at 'b', so b > a always."<<endl;
   cout << "Note: a,b are measured in number of solutions (tiles), so amount of data calibrated depends on -t parameter."<<endl;
   cout << "-V if given, enable verbose output: default "<<Data::verbose<<endl;
   //cout << "-M if given, evaluate AIC/MDL criteria for polynomials starting from 1 term to the one given by -P and suggest the best polynomial terms to use based on the minimum AIC/MDL: default "<<Data::mdl<<endl;
   cout << "-q solutions.txt: if given, initialize solutions by reading this file (need to have the same format as a solution file, only solutions for 1 timeslot needed)"<< endl;
   cout<<endl<<"Stochastic mode:"<<endl;
   cout << "-N epochs, if >0, use stochastic calibration: default "<<Data::stochastic_calib_epochs<< endl;
   cout << "-M minibatches, must be >0, split data to this many minibatches: default "<<Data::stochastic_calib_minibatches<< endl;
   cout << "-w mini-bands, must be >0, split channels to this many mini-bands for bandpass calibration: default "<<Data::stochastic_calib_bands<< endl;
   cout << "-u alpha, must be >0, alpha is the regularization factor used in passing global Z to local value: default "<<Data::federated_reg_alpha<< endl;
   cout <<"Report bugs to <sarod@users.sf.net>"<<endl;
}

/* command line parsing for both master/slaves */
void 
ParseCmdLine(int ac, char **av) {
    int c;
    while((c=getopt(ac, av, ":c:e:f:g:j:k:l:m:n:o:p:q:r:s:t:u:w:x:y:A:B:C:E:F:G:H:I:J:K:L:M:N:O:P:Q:R:S:T:W:E:MVh"))!= -1)
    {
        switch(c)
        {
            case 'f':
                MSpattern=optarg;
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
            case 'T':
                Nmaxtime= atoi(optarg);
                if (Nmaxtime<0) { Nmaxtime=0; }
                break;
            case 'K':
                Nskip= atoi(optarg);
                if (Nskip<0) { Nskip=0; }
                break;
            case 'F':
                format= atoi(optarg);
                if (format>1) { format=1; }
                break;
            case 'B':
                doBeam= atoi(optarg);
                if (doBeam>1) { doBeam=1; }
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
            case 'P': 
                Npoly= atoi(optarg);
                break;
            case 'Q': 
                PolyType= atoi(optarg);
                break;
            case 'J': 
                phaseOnly= atoi(optarg);
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
            case 'A': 
                Nadmm= atoi(optarg);
                break;
/*            case 'M': 
                Data::mdl=1;
                break;  */
            case 'E': 
                GPUpredict=atoi(optarg);
                break; 
            case 'H': 
                nuhigh= atof(optarg);
                break;
            case 'r': 
                admm_rho= atof(optarg);
                break;
            case 'G': 
                admm_rho_file= optarg;
                break;
            case 'R': 
                randomize= atoi(optarg);
                break;
            case 'C': 
                aadmm= atoi(optarg);
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
            case 'V': 
                Data::verbose=1;
                break; 
            case 'W':
                whiten= atoi(optarg);
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
            case 'u':
                Data::federated_reg_alpha= atof(optarg);
                break;
            case 'h': 
                print_help();
                MPI_Finalize();
                exit(1);
            case ':':
                cout<<"Error: A value is missing for one of the options"<<endl;
                print_help();
                exit(1);
            default:
                print_help();
                MPI_Finalize();
                exit(1);
        }
    }

    if (!MSpattern) {
     cout<<"Error: MS pattern is mandatory."<<endl;
     print_help();
     MPI_Finalize();
     exit(1);
    }
    cout<<"Selecting baselines > "<<min_uvcut<<" and < "<<max_uvcut<<" wavelengths."<<endl;
    cout<<"Using ";
    if (solver_mode==SM_LM_LBFGS || solver_mode==SM_OSLM_LBFGS || solver_mode==SM_RTR_OSLM_LBFGS) {
     cout<<"Gaussian noise model for solver."<<endl;
    } else {
     cout<<"Robust noise model for solver with degrees of freedom ["<<nulow<<","<<nuhigh<<"]."<<endl;
    }
    if (Nskip && Nmaxtime && Nskip>=Nmaxtime) {
     cout<<"Error: Start time of calibration "<< Nskip <<" is later than end time "<<Nmaxtime<<"."<<endl;
     print_help();
     MPI_Finalize();
     exit(1);
    }
}

/* real main program */
int
main(int argc, char **argv) {
 print_copyright();
 int myrank;
 /* init MPI */
 MPI_Init(&argc,&argv);
 if(argc<2) {
   print_help();
   MPI_Finalize();
   exit(0);
 }

 /* find out my identity and default communicator */
 MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
 /* both slave and master will parse command line again */
 /* full batch mode */

 ParseCmdLine(argc, argv); /* need to parse input here as well */
 if (Data::stochastic_calib_epochs==0) {
  if (myrank==0) {
   sagecal_master(argc,argv);
  } else {
   sagecal_slave(argc,argv);
  }
 } else {
  /* stochastic calibration */
  if (myrank==0) {
   sagecal_stochastic_master(argc,argv);
  } else {
   sagecal_stochastic_slave(argc,argv);
  }
 }
 /* shutdown MPI */
 MPI_Finalize();
 return 0;
}
