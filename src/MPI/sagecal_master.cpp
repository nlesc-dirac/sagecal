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

#include <map>
#include <string>
#include <cstring>
#include <iostream>

#include <Dirac.h>
#include <Radio.h>
#include <mpi.h>

using namespace std;
using namespace Data;

//#define DEBUG
int 
sagecal_master(int argc, char **argv) {
    ParseCmdLine(argc, argv);
    if (!Data::SkyModel || !Data::Clusters || !Data::MSpattern) {
      print_help();
      MPI_Finalize();
      exit(1);
    }

    openblas_set_num_threads(1);//Always use 1 thread for openblas to avoid conflicts;

    MPIData iodata;
    MPI_Status status;
    iodata.tilesz=Data::TileSize;
    vector<string> msnames;
    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    int nslaves=ntasks-1;
    /**** setup MS names ***************************************/
    multimap<string, int> nodenames;

    const char *filebase=Data::MSpattern;

    if (Data::randomize) {
     srand(time(0)); /* use different seed */
    } else {
     srand(0);
    }

    // master gets host name of all slaves and enters them to map
    for (int cm=0; cm<nslaves; cm++) {
     int count;
     MPI_Probe(cm+1,TAG_MSNAME,MPI_COMM_WORLD,&status);
     MPI_Get_count(&status,MPI_CHAR,&count);
     // allocate memory
     char *buf=new char[count];
     MPI_Recv(buf,count,MPI_CHAR,cm+1,TAG_MSNAME,MPI_COMM_WORLD,&status);
     string buf1(buf,count);
     //Insert key, value as pairs
     nodenames.insert(pair<string, int>(buf1,cm+1));
     delete [] buf;
    }

    int nhosts=0;
    //also create array to store unique node names
    vector<string> nnames;
    for( multimap<string,int>::iterator it1 = nodenames.begin(), end = nodenames.end(); it1 != end; it1 = nodenames.upper_bound(it1->first)) {
      nnames.push_back(string(it1->first));
      nhosts++;
    }
    cout<<"Found "<<nhosts<<" unique hosts"<<endl;
    multimap<string, int>::iterator it; //Iterator for map
    vector<string>::iterator ss;

    vector<vector<int> > PP(nhosts, vector<int>() ); //one row contains all slaves of one node
    int cmi=0;
    for (ss=nnames.begin(); ss<nnames.end(); ss++) {
     pair<multimap<string,int>::iterator, multimap<string,int>::iterator> ii;
     ii = nodenames.equal_range(*ss); //We get the first and last entry in ii;

     //now, master sends one slave per node request, the remaining nodes null 
     //also keep track which slaves
     bool sentone=0;
     for(it = ii.first; it != ii.second; ++it) {
      if (!sentone) {
        MPI_Send(filebase,strlen(filebase),MPI_CHAR,it->second,TAG_MSNAME,MPI_COMM_WORLD);
        PP[cmi].push_back(it->second);
        sentone=1;
      } else {
        //send null
        MPI_Send("",0,MPI_CHAR,it->second,TAG_MSNAME,MPI_COMM_WORLD);
        PP[cmi].push_back(it->second);
      }
     }

     cmi++;
    }

    //get back the results, only from slaves we sent a proper request
    vector<vector<string> > MP(nhosts, vector<string>() ); //one row contains all MS of one node
    int totalfiles=0;
    for (int cm=0; cm<nhosts; cm++) {
     int nfiles;
     MPI_Recv(&nfiles,1,MPI_INT,PP[cm][0],TAG_MSNAME,MPI_COMM_WORLD,&status);
     totalfiles +=nfiles;
     //get all MS names from this host
     int cj;
     for (cj=0; cj<nfiles; cj++) {
      int count;
      MPI_Probe(PP[cm][0],TAG_MSNAME,MPI_COMM_WORLD,&status);
      MPI_Get_count(&status,MPI_CHAR,&count);
      char *buf=new char[count];
      MPI_Recv(buf,count,MPI_CHAR,PP[cm][0],TAG_MSNAME,MPI_COMM_WORLD,&status);
      string buf1(buf,count);
      //Insert key, value as pairs
      MP[cm].push_back(buf1);
      delete [] buf;
     }
    }


cout<<"Master received all "<<totalfiles<<" files"<<endl;
   // check if we have more slaves than there are files, print a warning and exit
   if (totalfiles < nslaves) {
    cout<<"Error: The total number of datasets "<<totalfiles<<" is lower than the slaves used ("<<nslaves<<")."<<endl;
    cout<<"Error: Reduce the number of slaves to a value lower than "<<totalfiles<<" and rerun."<<endl;
    cout<<"Error: Value for -np should be something less than or equal to "<<totalfiles+1<<"."<<endl;
    exit(1);
   }

   vector<string>::iterator mp;
   vector<vector<string> >::iterator mrow;
   vector<int> hostms(totalfiles); /* i-th row stores the slave rank for i-th MS */
   // keeping track on which MS each slave will work on
   // begin: current : end
   vector<int> Sbegin(nslaves);
   vector<int> Send(nslaves);
   vector<int> Scurrent(nslaves); /* relative offset from Sbegin: 0,1,... */

   cmi=0;
   int chost=0;
   for (mrow=MP.begin(); mrow<MP.end(); mrow++) {
      int Nms=mrow->size();
      int Nslaves=PP[cmi].size();
      int NperH=(Nms+Nslaves-1)/Nslaves;
      int ck=0;
      mp=mrow->begin();
      int nperH;
      int nslave;
      for (nslave=0; nslave<Nslaves && ck<Nms; nslave++) {
        if (ck+NperH<Nms) { 
          nperH=NperH;
        } else {
          nperH=Nms-ck;
        }
        MPI_Send(&nperH,1,MPI_INT,PP[cmi][nslave],TAG_MSNAME,MPI_COMM_WORLD);
        //keep track of MS id ranges sent to each slave
        Sbegin[PP[cmi][nslave]-1]=chost;
        if (Data::randomize) {
         Scurrent[PP[cmi][nslave]-1]=random_int(nperH-1); /* relative offset, randomly initialized in 0,1..,nperH-1 */
        } else {
         Scurrent[PP[cmi][nslave]-1]=0; /* relative offset, always 0 */
        }
        Send[PP[cmi][nslave]-1]=chost+nperH-1;
        for (int ch=0; ch<nperH; ch++) {
          MPI_Send((*mp).c_str(),(*mp).length(),MPI_CHAR,PP[cmi][nslave],TAG_MSNAME,MPI_COMM_WORLD);
          hostms[chost]=PP[cmi][nslave];
          /* also send chost to slave */
          MPI_Send(&chost,1,MPI_INT,PP[cmi][nslave],TAG_MSNAME,MPI_COMM_WORLD);
          chost++;
          mp++;
        }
        ck+=nperH;
      }
      while(nslave<Nslaves) {
        // remaining slaves (-1 means no working MS for this slave)
        Sbegin[PP[cmi][nslave]-1]=-1;
        Scurrent[PP[cmi][nslave]-1]=-1;
        Send[PP[cmi][nslave]-1]=-1;
        int zero=0;
        MPI_Send(&zero,1,MPI_INT,PP[cmi][nslave],TAG_MSNAME,MPI_COMM_WORLD);
        nslave++;
      }
      cmi++;
   }


   //print report of allocation
   int mintotal=0;
   for (int cm=0; cm<nslaves; cm++) {
    int thistotal=Send[cm]-Sbegin[cm]+1;
    cout<<"Slave "<<cm+1<<" MS range "<<Sbegin[cm]<<":"<<Send[cm]<<" total "<<thistotal<<endl;
    if (mintotal<thistotal) { mintotal=thistotal; }
   }
   //print a warning if no of ADMM iterations is too low
   if (mintotal>Nadmm) {
    cout<<"Warning: current ADMM iterations "<<Nadmm<<" lower than max no. of MS some slaves need to calibrate ("<<mintotal<<")."<<endl;
    cout<<"Warning: increase ADMM iterations to at least "<<mintotal<<"."<<endl;
   }


    fflush(stdout);
    /**** end setup MS names ***************************************/

   iodata.Nms=totalfiles;
   /**** get info from slaves ***************************************/
   int *bufint=new int[6];
   double *bufdouble=new double[1];
   iodata.freqs=new double[iodata.Nms];
   iodata.freq0=0.0;
   iodata.N=iodata.M=iodata.totalt=0;
   int Mo=0;

   /* use iodata to store the results, also check for consistency of results */
   for (int cm=0; cm<nslaves; cm++) {
     /* for each slave, iterate over different MS */
     if (Sbegin[cm]>=0) {
        int scount=Send[cm]-Sbegin[cm]+1;
        for (int ct=0; ct<scount; ct++) {
         MPI_Recv(bufint, 6, /* MS-id, N,Mo(actual clusters),M(with hybrid),tilesz,totalt */
           MPI_INT, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);
         int thismsid=bufint[0];
cout<<"Slave "<<cm+1<<" MS="<<thismsid<<" N="<<bufint[1]<<" M="<<bufint[2]<<"/"<<bufint[3]<<" tilesz="<<bufint[4]<<" totaltime="<<bufint[5]<<endl;
         if (cm==0 && ct==0) { /* update metadata */
          iodata.N=bufint[1];
          Mo=bufint[2];
          iodata.M=bufint[3];
          iodata.tilesz=bufint[4];
          iodata.totalt=bufint[5];

         } else { /* check metadata for problem consistency */
           if ((iodata.N != bufint[1]) || (iodata.M != bufint[3]) || (iodata.tilesz != bufint[4])) {
            cout<<"Slave "<<cm+1<<" parameters do not match  N="<<bufint[1]<<" M="<<bufint[3]<<" tilesz="<<bufint[4]<<endl;
            exit(1);
           }
           if (iodata.totalt<bufint[5]) {
            /* use max value as total time */
            iodata.totalt=bufint[5];
           }
         }
         MPI_Recv(bufdouble, 1, /* freq */
           MPI_DOUBLE, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);
         iodata.freqs[thismsid]=bufdouble[0];
         iodata.freq0 +=bufdouble[0];
         cout<<"Slave "<<cm+1<<" MS="<<thismsid<<" frequency (MHz)="<<bufdouble[0]*1e-6<<endl;
        }
     }
   }
   iodata.freq0/=(double)iodata.Nms;
   delete [] bufint;
   delete [] bufdouble;
cout<<"Reference frequency (MHz)="<<iodata.freq0*1.0e-6<<endl;

    /* ADMM memory : allocated together for all MS */
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
    /* need a copy to update rho */
    double *Zold,*zold;
    if ((Zold=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* zold : 2Nx2 x M x Npoly vector, so each block is 8NM, storage for calculating Bf Z_old */
    if ((zold=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* storage to calculate dual residual */
    double *Zerr;
    if ((Zerr=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    
    /* file for saving solutions */
    FILE *sfp=0;
    if (solfile) {
     if ((sfp=fopen(solfile,"w+"))==0) {
       fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
       exit(1);
     }
    }

    /* write additional info to solution file */
    if (solfile) {
      fprintf(sfp,"# solution file (Z) created by SAGECal\n");
      fprintf(sfp,"# reference_freq(MHz) polynomial_order stations clusters effective_clusters\n");
      fprintf(sfp,"%lf %d %d %d %d\n",iodata.freq0*1e-6,Npoly,iodata.N,Mo,iodata.M);
    }




    /* interpolation polynomial */
    double *B;
    /* Npoly terms, for each frequency, so Npoly x Nms */
    if ((B=(double*)calloc((size_t)Npoly*iodata.Nms,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* regularization factor array, size Mx1
       one per each hybrid cluster */
    double *arho,*arhoslave;
    if ((arho=(double*)calloc((size_t)iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    if ((arhoslave=(double*)calloc((size_t)Mo,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* array to store flag ratio per each slave */
    double *fratio;
    if ((fratio=(double*)calloc((size_t)iodata.Nms,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    /* if text file is given, read it and update rho array */
    if (Data::admm_rho_file) {
     read_arho_fromfile(Data::admm_rho_file,iodata.M,arho,Mo,arhoslave);
    } else {
     /* copy common value */
     /* setup regularization factor array */
     for (int p=0; p<iodata.M; p++) {
      arho[p]=admm_rho; 
     }
     for (int p=0; p<Mo; p++) {
      arhoslave[p]=admm_rho; 
     }
    }

    /* send array to slaves */
    /* update rho on each slave */
    for(int cm=0; cm<nslaves; cm++) {
      MPI_Send(arhoslave, Mo, MPI_DOUBLE, cm+1,TAG_RHO, MPI_COMM_WORLD);
    }

    /*BB: get from 1st slave info about chunk sizes */
    int *chunkvec;
    if ((chunkvec=(int*)calloc((size_t)Mo,sizeof(int)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }   
    MPI_Recv(chunkvec,Mo,MPI_INT,1,TAG_CHUNK,MPI_COMM_WORLD,&status);    

    /* rho for each freq can be different (per each direction),
       so need to store each value, each column is Mx1 rho of one freq */
    double *rhok,*Bii;
    if ((rhok=(double*)calloc((size_t)iodata.Nms*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* initilized with default values */
    for(int cm=0; cm<iodata.Nms; cm++) {
       my_dcopy(iodata.M,arho,1,&rhok[cm*iodata.M],1);
    }
    /* pseudoinverse  M values of NpolyxNpoly matrices */
    if ((Bii=(double*)calloc((size_t)iodata.M*Npoly*Npoly,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }


#ifdef DEBUG
    FILE *dfp;
    if ((dfp=fopen("debug.m","w+"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      exit(1);
    }
    fprintf(dfp,"%% timeslot admmIter dir1 dir2 (reversed) ...; (each row=freq)\n");  
    fprintf(dfp,"rhoK=[\n");
#endif

    /* each Npoly blocks is bases evaluated at one freq, catch if Npoly=1 */
    setup_polynomials(B, Npoly, iodata.Nms, iodata.freqs, iodata.freq0,(Npoly==1?1:PolyType));


    /* determine how many iterations are needed */
    int Ntime=(iodata.totalt+iodata.tilesz-1)/iodata.tilesz;
    /* override if input limit is given */
    if (Nmaxtime>0 && Ntime>Nmaxtime) {
      Ntime=Nmaxtime;
    } 
#ifdef DEBUG
    //Ntime=2;
#endif
    cout<<"Master total timeslots="<<Ntime<<endl;

    if (!Data::admm_rho_file) {
     cout<<"ADMM iterations="<<Nadmm<<" polynomial order="<<Npoly<<" regularization="<<admm_rho<<endl;
    } else {
     cout<<"ADMM iterations="<<Nadmm<<" polynomial order="<<Npoly<<" regularization given by text file "<<Data::admm_rho_file<<endl;
    }
    int msgcode;
    /* important thing to remember: no. of MS != no. of slaves */
    for (int ct=0; ct<Ntime; ct++)  {
      /* send start processing signal to slaves */
      if (Nskip>0 && ct<Nskip) {
       msgcode=CTRL_SKIP;
      } else {
       msgcode=CTRL_START;
      }
      for(int cm=0; cm<nslaves; cm++) {
        MPI_Send(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD);
      }
      if (Nskip>0 && ct<Nskip) {
        cout<<"Skipping timeslot "<<ct<<endl;
        continue;
      }

      /* receive flag ratio (0 means all flagged) from each slave  for all MS */
      for(int cm=0; cm<nslaves; cm++) {
        if (Sbegin[cm]>=0) {
         int scount=Send[cm]-Sbegin[cm]+1;
         for (int ct1=0; ct1<scount; ct1++) {
          int mmid=Sbegin[cm]+ct1;
          MPI_Recv(&fratio[mmid], 1, MPI_DOUBLE, cm+1,TAG_FRATIO, MPI_COMM_WORLD, &status);
         }
        }
      }
      /* initialize rho with default values, also scale each column of rhok based on fratio */
      for(int cm=0; cm<iodata.Nms; cm++) {
       my_dcopy(iodata.M,arho,1,&rhok[cm*iodata.M],1);
       my_dscal(iodata.M,fratio[cm],&rhok[cm*iodata.M]);
      }

      /* Note(x): sum rho*B(:,i)*B(:,i)^T and its inverse changes with iteration, because rho might be updated */
      /* find sum fratio[i] * B(:,i)B(:,i)^T, and its pseudoinverse */
      /* find sum (Nms values) rho[] B(:,i) B(:,i)^T per cluster (M values) */

      for (int admm=0; admm<Nadmm; admm++) {
#ifdef DEBUG
       /* at each iteration, save rhok array
          each row correspond to one frequency */
       for (int cm=0; cm<iodata.Nms; cm++) {
          fprintf(dfp,"%d %d ",ct,admm);
          for(int ct1=0; ct1<iodata.M; ct1++) {
           fprintf(dfp,"%lf ",rhok[cm*iodata.M+ct1]);
          }
          fprintf(dfp,";\n");
       }
#endif
         /* BB : update Bi since rho is possibly updated, see Note(x) above */
         find_prod_inverse_full(B,Bii,Npoly,iodata.Nms,iodata.M,rhok,Data::Nt);

         /* send which MS to work on */
         for(int cm=0; cm<nslaves; cm++) {
          if (Sbegin[cm]>=0) {
           MPI_Send(&Scurrent[cm], 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD);
          }
         }
         /* for later iterations, need to update B_i Z for each slave MS and send it to them
          if the current MS is different from previous one */
         if (admm==Nadmm-1) { /* last ADMM, update all */
           for(int cm=0; cm<nslaves; cm++) {
            if (Sbegin[cm]>=0) {
             int scount=Send[cm]-Sbegin[cm]+1;

              for (int ct1=0; ct1<scount; ct1++) {
               int mmid=Sbegin[cm]+ct1;
               for (int p=0; p<iodata.M; p++) {
                memset(&z[8*iodata.N*p],0,sizeof(double)*(size_t)iodata.N*8);
                for (int ci=0; ci<Npoly; ci++) {
                 my_daxpy(8*iodata.N, &Z[p*8*iodata.N*Npoly+ci*8*iodata.N], B[mmid*Npoly+ci], &z[8*iodata.N*p]);
                }
               }
               MPI_Send(z, iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_CONSENSUS, MPI_COMM_WORLD);
              }
             }
           }
         } else if (admm>0) {
          for (int cm=0; cm<nslaves; cm++) {
           if (Sbegin[cm]>=0 && (Sbegin[cm]<Send[cm])) {
           int mmid=Sbegin[cm]+Scurrent[cm];
           for (int p=0; p<iodata.M; p++) {
            memset(&z[8*iodata.N*p],0,sizeof(double)*(size_t)iodata.N*8);
            for (int ci=0; ci<Npoly; ci++) {
             my_daxpy(8*iodata.N, &Z[p*8*iodata.N*Npoly+ci*8*iodata.N], B[mmid*Npoly+ci], &z[8*iodata.N*p]);
            }
           }
           MPI_Send(z, iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_CONSENSUS, MPI_COMM_WORLD);
           }
          }
         }

         /* get Y_i+rho J_i from each slave */
         /* note: for first iteration, reorder values as
            2Nx2 complex  matrix blocks, M times from each  slave 
            then project values to the mean, and reorder again
            and pass back to the slave */
         if (admm==0) {
          for(int cm=0; cm<nslaves; cm++) {
            if (Sbegin[cm]>=0) {
             int scount=Send[cm]-Sbegin[cm]+1;
             for (int ct1=0; ct1<scount; ct1++) {
              int mmid=Sbegin[cm]+ct1;
              MPI_Recv(&Y[mmid*iodata.N*8*iodata.M], iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_YDATA, MPI_COMM_WORLD, &status);
             }
            }
          }
         } else {
         for(int cm=0; cm<nslaves; cm++) {
           if (Sbegin[cm]>=0) {
            int mmid=Sbegin[cm]+Scurrent[cm];
            MPI_Recv(&Y[mmid*iodata.N*8*iodata.M], iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_YDATA, MPI_COMM_WORLD, &status);
           }
          }
         }

         if (admm==0) {
           calculate_manifold_average(iodata.N,iodata.M,iodata.Nms,Y,20,Data::randomize,Data::Nt);
          /* send updated Y back to each slave (for all MS) */
          for(int cm=0; cm<nslaves; cm++) {
           if (Sbegin[cm]>=0) {
            int scount=Send[cm]-Sbegin[cm]+1;
            for (int ct1=0; ct1<scount; ct1++) {
              int mmid=Sbegin[cm]+ct1;
              MPI_Send(&Y[mmid*iodata.N*8*iodata.M], iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_YDATA, MPI_COMM_WORLD);
            }
           }
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
            /* Note: no weighting of Y is needed, because slave has already weighted their rho (we have rho J here) */
            my_daxpy(8*iodata.N*iodata.M, &Y[cm*8*iodata.N*iodata.M], B[cm*Npoly+ci], &z[ci*8*iodata.N*iodata.M]);
           }
         }
         /* no need to scale by 1/rho here, because Bii is already divided by 1/rho */

         /* find product z_tilde x Bi^T, z_tilde with proper reshaping */
         my_dcopy(iodata.N*8*Npoly*iodata.M,Z,1,Zold,1);
         my_dcopy(iodata.N*8*Npoly*iodata.M,Z,1,Zerr,1);
         update_global_z_multi(Z,iodata.N,iodata.M,Npoly,z,Bii,Data::Nt);
         /* find dual error ||Zold-Znew|| (note Zerr is destroyed here)  */
         my_daxpy(iodata.N*8*Npoly*iodata.M,Z,-1.0,Zerr);
         /* dual residual per one real parameter */
         if (Data::verbose) {
          cout<<"ADMM : "<<admm<<" dual residual="<<my_dnrm2(iodata.N*8*Npoly*iodata.M,Zerr)/sqrt((double)8*iodata.N*Npoly*iodata.M)<<endl;
         } else {
          cout<<"Timeslot:"<<ct<<" ADMM:"<<admm<<endl;
         }

         /* find the MDL if admm==0 */
         /* At admm=0, Y = 0 + rho J, possibly projected to a common unitary ambiguity
            so estimate Z for different polynomial degrees, find ||rho J - rho B Z||^2 error
            sum up over Nms
          */
         if (Data::mdl && admm==0) {
          minimum_description_length(iodata.N,iodata.M,iodata.Nms,Y,arho,iodata.freqs,iodata.freq0,fratio,(Npoly==1?1:PolyType),1,Npoly,Data::Nt);
         }

         /* BB : send B_i Z_old to each slave */
         /* send B_i Z to each slave, 
            NOTE that this should be calculated for the current MS as well as the
            next MS each slave will work on, so increment current MS 'after' finding this
            and find it again (at the beginning of the loop) (if not changed) */
         for (int cm=0; cm<nslaves; cm++) {
           if (Sbegin[cm]>=0) {
           int mmid=Sbegin[cm]+Scurrent[cm];
           for (int p=0; p<iodata.M; p++) {
            memset(&z[8*iodata.N*p],0,sizeof(double)*(size_t)iodata.N*8);
            memset(&zold[8*iodata.N*p],0,sizeof(double)*(size_t)iodata.N*8);
            for (int ci=0; ci<Npoly; ci++) {
             my_daxpy(8*iodata.N, &Z[p*8*iodata.N*Npoly+ci*8*iodata.N], B[mmid*Npoly+ci], &z[8*iodata.N*p]);
             my_daxpy(8*iodata.N, &Zold[p*8*iodata.N*Npoly+ci*8*iodata.N], B[mmid*Npoly+ci], &zold[8*iodata.N*p]);
            }
           }

           /* old */
           MPI_Send(zold, iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_CONSENSUS_OLD, MPI_COMM_WORLD);
           /* new */
	         if (admm>0) {
	           MPI_Send(z, iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_CONSENSUS, MPI_COMM_WORLD);
	         } else if (admm==0){//update and send B_i Z for all ms
	           int scount=Send[cm]-Sbegin[cm]+1;
       
	           for (int ct=0; ct<scount; ct++) {
	            mmid = Sbegin[cm]+ct;
	            for (int p=0; p<iodata.M; p++) {
		            memset(&z[8*iodata.N*p],0,sizeof(double)*(size_t)iodata.N*8);
		            for (int ci=0; ci<Npoly; ci++) {
		              my_daxpy(8*iodata.N, &Z[p*8*iodata.N*Npoly+ci*8*iodata.N], B[mmid*Npoly+ci], &z[8*iodata.N*p]);
		            }
	            }
	            MPI_Send(z, iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_CONSENSUS, MPI_COMM_WORLD);   
	          }
	     
	        }
	       }
	      }

        /* BB : get updated rho from slaves */
        for (int cm=0; cm<nslaves; cm++) {
           if (Sbegin[cm]>=0) {
            int mmid=Sbegin[cm]+Scurrent[cm];
            MPI_Recv(arhoslave,Mo,MPI_DOUBLE,cm+1,TAG_RHO_UPDATE,MPI_COMM_WORLD,&status);
            /* now copy this to proper locations, using chunkvec as guide */
            //MPI_Recv(&rhok[mmid*iodata.M],iodata.M,MPI_DOUBLE,cm+1,TAG_RHO_UPDATE,MPI_COMM_WORLD,&status);
            int chk1=0;
            double *rhoptr=&rhok[mmid*iodata.M];
            for (int chk2=0; chk2<Mo; chk2++) {
              for (int chk3=0; chk3<chunkvec[chk2]; chk3++) {
                rhoptr[chk1]=arhoslave[chk2];
                chk1++;
              }
            }
           }
        }
  
        /* go to the next MS in the next ADMM iteration */
        for (int cm=0; cm<nslaves; cm++) {
          if (Sbegin[cm]>=0 && (Sbegin[cm]!=Send[cm])) { /* skip cases with only one MS */
           /* increment current MS for each slave */
           Scurrent[cm]=(Sbegin[cm]+Scurrent[cm]>=Send[cm]?0:Scurrent[cm]+1);
          }
        }
      }

      /* wait till all slaves are done writing data */
      int resetcount=0;
      for(int cm=0; cm<nslaves; cm++) {
        MPI_Recv(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD,&status);
        if (msgcode==CTRL_RESET) {
          resetcount++;
        }
      }

    /* write Z to solution file, same format as J, but we have Npoly times more
       values per timeslot per column */
     if (solfile) {
      for (int p=0; p<iodata.N*8*Npoly; p++) {
       fprintf(sfp,"%d ",p);
       //for (int ppi=0; ppi<iodata.M; ppi++) {
       for (int ppi=iodata.M-1; ppi>=0; ppi--) { /* reverse ordering */
        fprintf(sfp," %e",Z[ppi*iodata.N*8*Npoly+p]);
       }
       fprintf(sfp,"\n");
      }
     }
     if (resetcount>nslaves/2) {
       /* if most slaves have reset, print a warning only */
       //memset(Z,0,sizeof(double)*(size_t)iodata.N*8*Npoly*iodata.M);
       cout<<"Warning: Most slaves did not converge."<<endl;
     }


    }

    /* send end signal to each slave */
    msgcode=CTRL_END;
    for(int cm=0; cm<nslaves; cm++) {
        MPI_Send(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD);
    }

    if (solfile) {
      fclose(sfp);
    }


#ifdef DEBUG
    fprintf(dfp,"];\n");
    fclose(dfp);
#endif

    /**********************************************************/

   delete [] iodata.freqs;
   free(Z);
   free(Zold);
   free(Zerr);
   free(z);
   free(zold);
   free(Y);
   free(B);
   free(arho);
   free(arhoslave);
   free(fratio);

   free(rhok);
   free(Bii);
   free(chunkvec);
  /**********************************************************/

   cout<<"Done."<<endl;    
   return 0;
}
