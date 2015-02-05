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

#include<sagecal.h>
#include <mpi.h>

using namespace std;
using namespace Data;

void
print_copyright(void) {
  cout<<"SAGECal-MPI 0.3.5 (C) 2011-2015 Sarod Yatawatta"<<endl;
}


void
print_help(void) {
   cout << "Usage:" << endl;
   cout<<"mpirun (MPI options) sagecal-mpi -f MSlist -s sky.txt -c cluster.txt"<<endl;
   cout << "-f MSlist: text file with MS names" << endl;
   cout << "-s sky.txt: sky model file"<< endl;
   cout << "-c cluster.txt: cluster file"<< endl;
   cout << "-p solutions.txt: if given, save solution in this file, if not given 'XXX.MS.solutions' will be used"<< endl;
   cout << "-F sky model format: 0: LSM, 1: LSM with 3 order spectra : default "<< Data::format<<endl;
   cout << "-I input column (DATA/CORRECTED_DATA) : default " <<Data::DataField<< endl;
   cout << "-O ouput column (DATA/CORRECTED_DATA) : default " <<Data::OutField<< endl;
   cout << "-e max EM iterations : default " <<Data::max_emiter<< endl;
   cout << "-g max iterations  (within single EM) : default " <<Data::max_iter<< endl;
   cout << "-l max LBFGS iterations : default " <<Data::max_lbfgs<< endl;
   cout << "-m LBFGS memory size : default " <<Data::lbfgs_m<< endl;
   cout << "-n no of worker threads : default "<<Data::Nt << endl;
   cout << "-t tile size : default " <<Data::TileSize<< endl;
   cout << "-A ADMM iterations: default " <<Data::Nadmm<< endl;
   cout << "-P consensus polynomial order: default " <<Data::Npoly<< endl;
   cout << "-r regularization factor: default " <<Data::admm_rho<< endl;
   cout << "-x exclude baselines length (lambda) lower than this in calibration : default "<<Data::min_uvcut << endl;
   cout << "-y exclude baselines length (lambda) higher than this in calibration : default "<<Data::max_uvcut << endl;
   cout <<endl<<"Advanced options:"<<endl;
   cout << "-k cluster_id : correct residuals with solution of this cluster : default "<<Data::ccid<< endl;
   cout << "-o robust rho, robust matrix inversion during correction: default "<<Data::rho<< endl;
   cout << "-j 0,1,2... 0 : OSaccel, 1 no OSaccel, 2: OSRLM, 3: RLM, 4: RTR, 5: RRTR: default "<<Data::solver_mode<< endl;
   cout << "-L robust nu, lower bound: default "<<Data::nulow<< endl;
   cout << "-H robust nu, upper bound: default "<<Data::nuhigh<< endl;
   cout << "-R randomize iterations: default "<<Data::randomize<< endl;
   cout << "-D 0,1,2 : if >0, enable diagnostics (Jacobian Leverage) 1 replace Jacobian Leverage as output, 2 only fractional noise/leverage is printed: default " <<Data::DoDiag<< endl;
   cout << "-T stop after this number of solutions (0 means no limit): default "<<Data::Nmaxtime<< endl;
   cout <<"Report bugs to <sarod@users.sf.net>"<<endl;
}

/* command line parsing for both master/slaves */
void 
ParseCmdLine(int ac, char **av) {
    char c;
    while((c=getopt(ac, av, "c:e:f:g:j:k:l:m:n:o:p:r:s:t:x:y:A:D:F:I:L:O:P:H:R:T:h"))!= -1)
    {
        switch(c)
        {
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
            case 'g':
                max_iter= atoi(optarg);
                break;
            case 'T':
                Nmaxtime= atoi(optarg);
                if (Nmaxtime<0) { Nmaxtime=0; }
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
            case 'P': 
                Npoly= atoi(optarg);
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
            case 'H': 
                nuhigh= atof(optarg);
                break;
            case 'r': 
                admm_rho= atof(optarg);
                break;
            case 'R': 
                randomize= atoi(optarg);
                break;
            case 'x': 
                Data::min_uvcut= atof(optarg);
                break;
            case 'y': 
                Data::max_uvcut= atof(optarg);
                break;
            case 'h': 
                print_help();
                MPI_Finalize();
                exit(1);
            default:
                print_help();
                MPI_Finalize();
                exit(1);
        }
    }

    if (!MSlist) {
     cout<<"MS list is mandatory."<<endl;
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
 if (myrank==0) {
  sagecal_master(argc,argv);
 } else {
  sagecal_slave(argc,argv);
 }
 /* shutdown MPI */
 MPI_Finalize();
 return 0;
}
