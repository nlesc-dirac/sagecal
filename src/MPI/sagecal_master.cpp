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

//#define DEBUG
int 
sagecal_master(int argc, char **argv) {
    ParseCmdLine(argc, argv);
    if (!Data::SkyModel || !Data::Clusters || !Data::MSlist) {
      print_help();
      MPI_Finalize();
      exit(1);
    }
    MPIData iodata;
    MPI_Status status;
    iodata.tilesz=Data::TileSize;
    vector<string> msnames;
    if (Data::MSlist) {
     Data::readMSlist(Data::MSlist,&msnames);
     cout<<"Total MS "<<msnames.size()<<endl;
    }
    fflush(stdout);
    /**** send MS names to slaves ***************************************/
    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    cout<<"Total slaves "<<ntasks-1<<endl;
    if ((int)msnames.size()+1!=ntasks) {
     cout<<"Number of slaves != Number of MS"<<endl;
     MPI_Finalize();
     exit(1);
    }
    for(unsigned int cm=0; cm<msnames.size(); cm++) {
       /* length+1 to include trailing '\0' */
       MPI_Send(const_cast<char*>(msnames[cm].c_str()), msnames[cm].length()+1, MPI_CHAR, cm+1,TAG_MSNAME, MPI_COMM_WORLD);
    }

    /**********************************************************/

   iodata.Nms=ntasks-1;
   /**** get info from slaves ***************************************/
   int *bufint=new int[4];
   double *bufdouble=new double[1];
   iodata.freqs=new double[iodata.Nms];
   iodata.freq0=0.0;
   iodata.N=iodata.M=iodata.totalt=0;
   /* use iodata to store the results, also check for consistency of results */
   for (int cm=0; cm<iodata.Nms; cm++) {
     MPI_Recv(bufint, 4, /* N,M,tilesz,totalt */
       MPI_INT, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);
cout<<"Slave "<<cm+1<<" N="<<bufint[0]<<" M="<<bufint[1]<<" tilesz="<<bufint[2]<<" totaltime="<<bufint[3]<<endl;
     if (cm==0) { /* update data */
      iodata.N=bufint[0];
      iodata.M=bufint[1];
      iodata.tilesz=bufint[2];
      iodata.totalt=bufint[3];
     } else { /* compare against others */
       if ((iodata.N != bufint[0]) || (iodata.M != bufint[1]) || (iodata.tilesz != bufint[2])) {
        cout<<"Slave "<<cm+1<<" parameters do not match  N="<<bufint[0]<<" M="<<bufint[1]<<" tilesz="<<bufint[2]<<endl;
       }
       if (iodata.totalt<bufint[3]) {
        /* use max value as total time */
        iodata.totalt=bufint[3];
       }
     }
     MPI_Recv(bufdouble, 1, /* freq */
       MPI_DOUBLE, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);
     iodata.freqs[cm]=bufdouble[0];
     iodata.freq0 +=bufdouble[0];
     cout<<"Slave "<<cm+1<<" freq="<<bufdouble[0]<<endl;
   }
    iodata.freq0/=(double)iodata.Nms;
    delete [] bufint;
    delete [] bufdouble;
cout<<"Reference freq="<<iodata.freq0<<endl;
    /* ADMM memory */
    double *Z,*Y,*z;
    /* Z: 2Nx2 x Npoly x M */
    /* keep ordered by M (one direction together) */
    if ((Z=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* z : 2Nx2 x M x Npoly vector, so each block is 8NM */
    if ((z=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* copy of Y+rho J, M times, for each slave */
    /* keep ordered by M (one direction together) */
    if ((Y=(double*)calloc((size_t)iodata.N*8*iodata.M*iodata.Nms,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* need a copy to calculate dual residual */
    double *Zold;
    if ((Zold=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* interpolation polynomial */
    double *B,*Bi;
    /* Npoly terms, for each frequency, so Npoly x Nms */
    if ((B=(double*)calloc((size_t)Npoly*iodata.Nms,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* pseudoinverse */
    if ((Bi=(double*)calloc((size_t)Npoly*Npoly,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* regularization factor array */
    double *arho;
    if ((arho=(double*)calloc((size_t)Nadmm,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

#ifdef DEBUG
    FILE *dfp;
    if ((dfp=fopen("debug.m","w+"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      return 1;
    }
#endif

    /* each Npoly blocks is bases evaluated at one freq */
    setup_polynomials(B, Npoly, iodata.Nms, iodata.freqs, iodata.freq0, 1);
    /* find sum B(:,i)B(:,i)^T, and its pseudoinverse */
    find_prod_inverse(B,Bi,Npoly,iodata.Nms);

    /* setup regularization factor array */
    for (int p=0; p<Nadmm; p++) {
     arho[p]=admm_rho; 
    }

#ifdef DEBUG
    fprintf(dfp,"B=[\n");
    for (int p=0; p<Npoly; p++) {
     for (int q=0; q<iodata.Nms; q++) {
       fprintf(dfp,"%lf ",B[p*Npoly+q]);
     } 
     fprintf(dfp,";\n");
    }
    fprintf(dfp,"];\n");
    fprintf(dfp,"rho=%lf;\narho=[",admm_rho);
    for (int p=0; p<Nadmm; p++) {
      fprintf(dfp,"%lf ",arho[p]);
    }
    fprintf(dfp,"];\n");
    fprintf(dfp,"Bi=[\n");
    for (int p=0; p<Npoly; p++) {
     for (int q=0; q<Npoly; q++) {
       fprintf(dfp,"%lf ",Bi[p*Npoly+q]);
     } 
     fprintf(dfp,";\n");
    }
    fprintf(dfp,"];\n");
#endif

    /* determine how many iterations are needed */
    int Ntime=(iodata.totalt+iodata.tilesz-1)/iodata.tilesz;
    /* override if input limit is given */
    if (Nmaxtime>0 && Ntime>Nmaxtime) {
      Ntime=Nmaxtime;
    } 
#ifdef DEBUG
    Ntime=1;
#endif
cout<<"Master total timeslots="<<Ntime<<endl;
cout<<"ADMM iterations="<<Nadmm<<" polynomial order="<<Npoly<<" regularization="<<admm_rho<<endl;
    int msgcode;
    for (int ct=0; ct<Ntime; ct++)  {
      /* send start processing signal to slaves */
      msgcode=CTRL_START;
      for(int cm=0; cm<iodata.Nms; cm++) {
        MPI_Send(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD);
      }

      for (int admm=0; admm<Nadmm; admm++) {
         /* update rho on each slave */
         for(int cm=0; cm<iodata.Nms; cm++) {
          MPI_Send(&arho[admm], 1, MPI_DOUBLE, cm+1,TAG_RHO, MPI_COMM_WORLD);
         }

         /* get Y_i+rho J_i from each slave */
         /* note: for first iteration, reorder values as
            2Nx2 complex  matrix blocks, M times from each  slave 
            then project values to the mean, and reorder again
            and pass back to the slave */
         for(int cm=0; cm<iodata.Nms; cm++) {
          MPI_Recv(&Y[cm*iodata.N*8*iodata.M], iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_YDATA, MPI_COMM_WORLD, &status);
         }
#ifdef DEBUG
         fprintf(dfp,"%%%%%%%%%%%%%% time=%d admm=%d\n",ct,admm);
         for(int cm=0; cm<iodata.Nms; cm++) {
          for(int m=0; m<iodata.M; m++) {
           fprintf(dfp,"%%%%%%%%%%%% (Y+rho J) slave=%d dir=%d\n",cm,m);
           fprintf(dfp,"Y_%d_%d=[\n",cm,m);
           for (int p=0; p<iodata.N; p++) {
            int off=cm*iodata.N*8*iodata.M+m*8*iodata.N+p*8;
            fprintf(dfp,"%e+j*(%e), %e+j*(%e);\n%e+j*(%e), %e+j*(%e);\n",Y[off],Y[off+1],Y[off+2],Y[off+3],Y[off+4],Y[off+5],Y[off+6],Y[off+7]);
           }
           fprintf(dfp,"];\n");
          }
         }
#endif
         if (admm==0) {
           calculate_manifold_average(iodata.N,iodata.M,iodata.Nms,Y,20,Data::Nt);
#ifdef DEBUG
         fprintf(dfp,"%%%%%%%%%%%%%% time=%d admm=%d\n",ct,admm);
         for(int cm=0; cm<iodata.Nms; cm++) {
          for(int m=0; m<iodata.M; m++) {
           fprintf(dfp,"%%%%%%%%%%%% Averaged (Y+rho J) slave=%d dir=%d\n",cm,m);
           fprintf(dfp,"Yav_%d_%d=[\n",cm,m);
           for (int p=0; p<iodata.N; p++) {
            int off=cm*iodata.N*8*iodata.M+m*8*iodata.N+p*8;
            fprintf(dfp,"%e+j*(%e), %e+j*(%e);\n%e+j*(%e), %e+j*(%e);\n",Y[off],Y[off+1],Y[off+2],Y[off+3],Y[off+4],Y[off+5],Y[off+6],Y[off+7]);
           }
           fprintf(dfp,"];\n");
          }
         }
#endif
          /* send updated Y back to each slave */
          for(int cm=0; cm<iodata.Nms; cm++) {
           MPI_Send(&Y[cm*iodata.N*8*iodata.M], iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_YDATA, MPI_COMM_WORLD);
          }

         }

        
         /* update Z */
         /* add to 8NM vector, multiplied by Npoly different scalars, Nms times */
         for (int ci=0; ci<Npoly; ci++) {
           my_dcopy(8*iodata.N*iodata.M,Y,1,&z[ci*8*iodata.N*iodata.M],1);
           my_dscal(8*iodata.N*iodata.M,B[ci],&z[ci*8*iodata.N*iodata.M]);
         }
         for (int cm=1; cm<iodata.Nms; cm++) {
           for (int ci=0; ci<Npoly; ci++) {
            my_daxpy(8*iodata.N*iodata.M, &Y[cm*8*iodata.N*iodata.M], B[cm*Npoly+ci], &z[ci*8*iodata.N*iodata.M]);
           }
         }
         /* also scale by 1/rho */
         my_dscal(8*iodata.N*iodata.M*Npoly,1.0/arho[admm],z);

#ifdef DEBUG
         fprintf(dfp,"%%%%%%%%%%%%%% time=%d admm=%d\n",ct,admm);
          for(int m=0; m<iodata.M; m++) {
           fprintf(dfp,"%%%%%%%%%%%% sum (Y+rho J) dir=%d\n",m);
           fprintf(dfp,"Ysum_%d=[\n",m);
           for (int p=0; p<iodata.N; p++) {
            int off=m*8*iodata.N+p*8;
            fprintf(dfp,"%lf+j*(%lf), %lf+j*(%lf);\n%lf+j*(%lf), %lf+j*(%lf);\n",z[off],z[off+1],z[off+2],z[off+3],z[off+4],z[off+5],z[off+6],z[off+7]);
           }
           fprintf(dfp,"];\n");
          }
#endif

         /* find product z_tilde x Bi^T, z_tilde with proper reshaping */
         my_dcopy(iodata.N*8*Npoly*iodata.M,Z,1,Zold,1);
         update_global_z(Z,iodata.N,iodata.M,Npoly,z,Bi);
         /* find dual error ||Zold-Znew|| */
         my_daxpy(iodata.N*8*Npoly*iodata.M,Z,-1.0,Zold);
         /* dual residual per one real parameter */
         cout<<"ADMM : "<<admm<<" dual residual="<<my_dnrm2(iodata.N*8*Npoly*iodata.M,Zold)/sqrt((double)8*iodata.N*Npoly*iodata.M)<<endl;

 #ifdef DEBUG
         fprintf(dfp,"%%%%%%%%%%%%%% time=%d admm=%d\n",ct,admm);
          for(int m=0; m<iodata.M; m++) {
           for (int ci=0;ci<Npoly; ci++) {
            fprintf(dfp,"%%%%%%%%%%%% Z dir=%d poly=%d\n",m,ci);
            fprintf(dfp,"Z_%d_%d=[\n",m,ci);
            for (int p=0; p<iodata.N; p++) {
             int off=m*8*iodata.N*Npoly+ci*8*iodata.N+p*8;
             fprintf(dfp,"%lf+j*(%lf), %lf+j*(%lf);\n%lf+j*(%lf), %lf+j*(%lf);\n",Z[off],Z[off+1],Z[off+2],Z[off+3],Z[off+4],Z[off+5],Z[off+6],Z[off+7]);
            }
            fprintf(dfp,"];\n");
           }
          }
#endif


         /* send B_i Z to each slave */
         for (int cm=0; cm<iodata.Nms; cm++) {
           for (int p=0; p<iodata.M; p++) {
            memset(&z[8*iodata.N*p],0,sizeof(double)*(size_t)iodata.N*8);
            for (int ci=0; ci<Npoly; ci++) {
             my_daxpy(8*iodata.N, &Z[p*8*iodata.N*Npoly+ci*8*iodata.N], B[cm*Npoly+ci], &z[8*iodata.N*p]);
            }
           }
#ifdef DEBUG
         fprintf(dfp,"%%%%%%%%%%%%%% time=%d admm=%d\n",ct,admm);
          for(int m=0; m<iodata.M; m++) {
           fprintf(dfp,"%%%%%%%%%%%% consensus for slave=%d dir=%d\n",cm,m);
           fprintf(dfp,"BZ_%d_%d=[\n",cm,m);
           for (int p=0; p<iodata.N; p++) {
            int off=m*8*iodata.N+p*8;
            fprintf(dfp,"%lf+j*(%lf), %lf+j*(%lf);\n%lf+j*(%lf), %lf+j*(%lf);\n",z[off],z[off+1],z[off+2],z[off+3],z[off+4],z[off+5],z[off+6],z[off+7]);
           }
           fprintf(dfp,"];\n");
          }
#endif
           MPI_Send(z, iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_CONSENSUS, MPI_COMM_WORLD);
         }
      }

      /* wait till all slaves are done writing data */
      int resetcount=0;
      for(int cm=0; cm<iodata.Nms; cm++) {
        MPI_Recv(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD,&status);
        if (msgcode==CTRL_RESET) {
          resetcount++;
        }
      }
    }

    /* send end signal to each slave */
    msgcode=CTRL_END;
    for(int cm=0; cm<iodata.Nms; cm++) {
        MPI_Send(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD);
    }


#ifdef DEBUG
    fclose(dfp);
#endif

    /**********************************************************/

   delete [] iodata.freqs;
   free(Z);
   free(Zold);
   free(z);
   free(Y);
   free(B);
   free(Bi);
   free(arho);
  /**********************************************************/

   cout<<"Done."<<endl;    
   return 0;
}
