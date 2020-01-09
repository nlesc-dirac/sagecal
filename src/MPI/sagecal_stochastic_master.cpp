/*
 *
 Copyright (C) 2020 Sarod Yatawatta <sarod@users.sf.net>  
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
sagecal_stochastic_master(int argc, char **argv) {
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
          iodata.M=bufint[3]; /* Note M here is Mt in slave */
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

    /* get min,max freq from each slave to find full freq. range */
    bufdouble = new double[2];
    double min_f,max_f;
    min_f=1e15; max_f=-1e15;
    for (int cm=0; cm<nslaves; cm++) {
       MPI_Recv(bufdouble, 2, /* min, max freq */
           MPI_DOUBLE, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);

      min_f=(min_f>bufdouble[0]?bufdouble[0]:min_f);
      max_f=(max_f<bufdouble[1]?bufdouble[1]:max_f);
    }
cout<<"Freq range (MHz) ["<<min_f*1e-6<<","<<max_f*1e-6<<"]"<<endl;
    delete [] bufdouble;
    bufdouble = new double[3];
    /* send min,max and reference freq to each slave to setup polynomials */
    bufdouble[0]=min_f;
    bufdouble[1]=max_f;
    bufdouble[2]=iodata.freq0;
    for (int cm=0; cm<nslaves; cm++) {
     MPI_Send(bufdouble, 3, MPI_DOUBLE, cm+1,TAG_MSAUX, MPI_COMM_WORLD);
    }
    delete [] bufdouble;
    /* ADMM global memory : allocated together for all slaves */
    double *Z;
    /* Z: 2Nx2 x Npoly x M  x nslaves */
    /* keep ordered by M (one direction together) , per each slave */
    if ((Z=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M*nslaves,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }
    /* need a copy */
    double *Zavg;
    if ((Zavg=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

   /**********************************************************/
    /* determine how many iterations are needed */
    int Ntime=(iodata.totalt+iodata.tilesz-1)/iodata.tilesz;
    /* override if input limit is given */
    if (Nmaxtime>0 && Ntime>Nmaxtime) {
      Ntime=Nmaxtime;
    }

    cout<<"Master total batches="<<Ntime<<endl;

    int msgcode;

   /**********************************************************/
   /* iterate over all data */
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


    for (int nadmm=0; nadmm<Nadmm; nadmm++) {

    /* receive Z from each slave */
    for (int cm=0; cm<nslaves; cm++) {
       MPI_Recv(&Z[cm*iodata.N*8*Npoly*iodata.M], iodata.N*8*Npoly*iodata.M,
           MPI_DOUBLE, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);
    }
    /* find the average over quotient manifold, say Zav 
      and project it back to Z for each freq, Z : 2N.Npoly x 2 blocks, 
     total nslaves, per each M direction : Z input and output */
    calculate_manifold_average_projectback(iodata.N*Npoly,iodata.M,nslaves,Z,20,Data::randomize,Data::Nt);
    /* send  back to all slaves */
    for (int cm=0; cm<nslaves; cm++) {
     MPI_Send(&Z[cm*iodata.N*8*Npoly*iodata.M], iodata.N*8*Npoly*iodata.M, MPI_DOUBLE, cm+1,TAG_MSAUX, MPI_COMM_WORLD);
    }

    cout<<"Master admm "<<nadmm<<endl;
    }

    /* wait till all slaves are done writing data */
    int resetcount=0;
    for(int cm=0; cm<nslaves; cm++) {
        MPI_Recv(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD,&status);
        if (msgcode==CTRL_RESET) {
          resetcount++;
        }
    }

    cout<<"Master batch "<<ct<<" done with "<<resetcount<<endl;

   }
   /**********************************************************/


    /* send end signal to each slave */
    msgcode=CTRL_END;
    for(int cm=0; cm<nslaves; cm++) {
        MPI_Send(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD);
    }



   delete [] iodata.freqs;
   free(Z);
   free(Zavg);
  /**********************************************************/

   cout<<"Masted Done."<<endl;    
   return 0;
}
