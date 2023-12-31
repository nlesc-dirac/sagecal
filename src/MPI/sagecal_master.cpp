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
#include <Dirac_radio.h>
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

#ifdef HAVE_OPENBLAS
    openblas_set_num_threads(1);//Always use 1 thread for openblas to avoid conflicts;
#endif

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
    cout<<"Worker "<<cm+1<<" MS range "<<Sbegin[cm]<<":"<<Send[cm]<<" total "<<thistotal<<endl;
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
   double *bufdouble=new double[1+(Data::spatialreg?2:0)];
   iodata.freqs=new double[iodata.Nms];
   iodata.freq0=0.0;
   iodata.N=iodata.M=iodata.totalt=0;
   int Mo=0;
   /* extra parameters for spatial regularization*/
   double ra0=0.0,dec0=0.0;

   /* use iodata to store the results, also check for consistency of results */
   for (int cm=0; cm<nslaves; cm++) {
     /* for each slave, iterate over different MS */
     if (Sbegin[cm]>=0) {
        int scount=Send[cm]-Sbegin[cm]+1;
        for (int ct=0; ct<scount; ct++) {
         MPI_Recv(bufint, 6, /* MS-id, N,Mo(actual clusters),M(with hybrid),tilesz,totalt */
           MPI_INT, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);
         int thismsid=bufint[0];
    cout<<"Worker "<<cm+1<<" MS="<<thismsid<<" N="<<bufint[1]<<" M="<<bufint[2]<<"/"<<bufint[3]<<" tilesz="<<bufint[4]<<" totaltime="<<bufint[5]<<endl;
         if (cm==0 && ct==0) { /* update metadata */
          iodata.N=bufint[1];
          Mo=bufint[2];
          iodata.M=bufint[3];
          iodata.tilesz=bufint[4];
          iodata.totalt=bufint[5];

         } else { /* check metadata for problem consistency */
           if ((iodata.N != bufint[1]) || (iodata.M != bufint[3]) || (iodata.tilesz != bufint[4])) {
            cout<<"Worker "<<cm+1<<" parameters do not match  N="<<bufint[1]<<" M="<<bufint[3]<<" tilesz="<<bufint[4]<<endl;
            exit(1);
           }
           if (iodata.totalt<bufint[5]) {
            /* use max value as total time */
            iodata.totalt=bufint[5];
           }
         }
         /* freq, if spatialreg>0, ra0, dec0 */
         MPI_Recv(bufdouble, 1+(Data::spatialreg?2:0), /* freq, ra0, dec0 */
           MPI_DOUBLE, cm+1, TAG_MSAUX, MPI_COMM_WORLD, &status);
         iodata.freqs[thismsid]=bufdouble[0];
         iodata.freq0 +=bufdouble[0];
         if (Data::spatialreg) {
           if (cm==0 && ct==0) {
            ra0=bufdouble[1];
            dec0=bufdouble[2];
           } else {
             /* check ra0,dec0 for sanity */
             if (ra0!=bufdouble[1] || dec0!=bufdouble[2]) {
              cout<<"Warning: worker "<<cm+1<<" parameters do not match  ra0="<<bufdouble[1]<<" dec0="<<bufdouble[2]<<endl;
             }
           }
         }
         cout<<"Worker "<<cm+1<<" MS="<<thismsid<<" frequency (MHz)="<<bufdouble[0]*1e-6<<endl;
        }
     }
   }
   iodata.freq0/=(double)iodata.Nms;
   delete [] bufint;
   delete [] bufdouble;
   cout<<"Reference frequency (MHz)="<<iodata.freq0*1.0e-6<<endl;

//#define DEBUG1
#ifdef DEBUG1
    FILE *dfp;
#endif
   clus_source_t *carr;
   double *ll=0,*mm=0; /* Mx1 centroid coords (polar) */
   int spatialreg_basis=SP_SHAPELET; /* SP_SHAPELET: shapelet (l,m) basis, SP_SHARMONIC: spherical harmonic (phi,theta) basis */
   /* input parameter int sh_n0 shapelet or spherical harmonic model order */
   int G=sh_n0*sh_n0; /* total modes */
   double sh_beta=1.0; /* scale factor for shapelet basis - will be reset later */
   /* elastic net regularization parameters sh_lambda (L2), sh_mu (L1) */
   complex double *phivec=0; /* vector to store spherical harmonic modes n0^2 per each  polar coordinate */
   complex double *Phi=0; /* basis matrices 2Gx2, M times */
   complex double *Phikk=0; /* sum of Phi_k x Phi_k^H : 2Gx2G */
   int sp_diffuse_id=-1; /* if -D 'id' gives a matching cluster id, set this to matching ordinal number in 0,1,... */
   if (Data::spatialreg) {
#ifdef DEBUG1
    if ((dfp=fopen("debug.m","w+"))==0) {
      fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
      exit(1);
    }
    fprintf(dfp,"G=%d;\nK=%d;\n",G,iodata.M);
#endif

     int M1;
     read_sky_cluster(Data::SkyModel,Data::Clusters,&carr,&M1,iodata.freq0,ra0,dec0,Data::format);
     /* Note: we use hybrid cluster size as M, as we have this many solutions */
     if ((ll=(double*)calloc((size_t)iodata.M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if ((mm=(double*)calloc((size_t)iodata.M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     double *P;
     double lmean,mmean;
     double l_max=0; /* max +ve or -ve value, to determine scale factor for shapelet basis */
     /* find centroid of each cluster, weighted by P=|sI|+|sQ|+|sU|+|sV|
      * for example, llmean=dot(P,ll)/N, N: sources */
     int idx=0;
     /* Note: cluster odering is in reverse */
     for (int ci=0; ci<M1; ci++) {
       /* check if a cluster id exists matching Data::ddid */
       if (carr[ci].id==Data::ddid) {
         sp_diffuse_id=ci;
         printf("Cluster id %d (ordinal %d) is being used as foreground (diffuse) model\n",Data::ddid,sp_diffuse_id);
       }
       if (carr[ci].N>1) {
        if ((P=(double*)calloc((size_t)carr[ci].N,sizeof(double)))==0) {
         fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
         exit(1);
        }
        for (int cj=0; cj<carr[ci].N; cj++) {
          P[cj]=fabs(carr[ci].sI[cj])+fabs(carr[ci].sQ[cj])+fabs(carr[ci].sU[cj])
             +fabs(carr[ci].sV[cj]);
        }
        double sumP=my_dasum(carr[ci].N,P);
        lmean=my_ddot(carr[ci].N,P,carr[ci].ll)/sumP;
        mmean=my_ddot(carr[ci].N,P,carr[ci].mm)/sumP;
        free(P);
       } else {
         /* just one source, so copy values */
        lmean=carr[ci].ll[0];
        mmean=carr[ci].mm[0];
       }
       if (l_max<MAX(fabs(lmean),fabs(mmean))) {
         l_max=MAX(fabs(lmean),fabs(mmean));
       }
       double rr,tt;
       if (spatialreg_basis==SP_SHAPELET) {
         /* diffuse sky shapelet model is in (-l,m) so negate */
         rr=-lmean;
         tt=mmean;
       } else {
         /* transform l,m in [-1,1] to r in [0,pi/2],theta [0,2*pi] */
         rr=sqrt(lmean*lmean+mmean*mmean)*M_PI_2;
         tt=atan2(mmean,lmean);
       }
       /* copy coordinates to array, considering hybrid clustering */
       for (int cj=0; cj<carr[ci].nchunk; cj++) {
         ll[idx]=rr;
         mm[idx]=tt;
         idx++;
       }
     }
     if ((phivec=(complex double*)calloc((size_t)iodata.M*G,sizeof(complex double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if (spatialreg_basis==SP_SHAPELET) {
      sh_beta=4.0*sqrt(l_max*l_max/(double)iodata.M); /* scale ~ 2 x sqrt(range(l)*delta(l)) or m */
      printf("Using shaplet spatial basis with scale %lf\n",sh_beta);
      /* shapelet basis: real basis, complex part is zero */
      shapelet_modes(sh_n0,sh_beta,ll,mm,iodata.M,phivec);
     } else {
      printf("Using spherical harmonic spatial basis\n");
      sharmonic_modes(sh_n0,ll,mm,iodata.M,phivec);
     }
#ifdef DEBUG1
     fprintf(dfp,"phi=[\n");
     for (int ci=0; ci<iodata.M*G; ci++) {
       fprintf(dfp,"%lf+j*(%lf)\n",creal(phivec[ci]),cimag(phivec[ci]));
     }
     fprintf(dfp,"];\n");
#endif
     if ((Phi=(complex double*)calloc((size_t)iodata.M*2*G*2,sizeof(complex double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     for (int ci=0; ci<iodata.M; ci++) {
       memcpy(&Phi[ci*2*G*2],&phivec[ci*G],G*sizeof(complex double));
       memcpy(&Phi[ci*2*G*2+3*G],&phivec[ci*G],G*sizeof(complex double));
     }
     if ((Phikk=(complex double*)calloc((size_t)2*G*2*G,sizeof(complex double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     /* find product Phik*Phik^H (2Gx2G) and add up */
     for (int ci=0; ci<iodata.M; ci++) {
       /* C= alpha * op(A)*op(B)+beta * C */
       my_zgemm('N','C',2*G,2*G,2,1.0+_Complex_I*0.0,&Phi[ci*2*G*2],2*G,&Phi[ci*2*G*2],2*G,1.0+_Complex_I*0.0,Phikk,2*G);
     }
     /* add \lambda I to Phikk */
     for (int ci=0; ci<2*G; ci++) {
       Phikk[ci*2*G+ci]+=sh_lambda;
     }
#ifdef DEBUG1
     fprintf(dfp,"Phikk=[\n");
     for (int ci=0; ci<2*G*2*G; ci++) {
       fprintf(dfp,"%lf+j*(%lf)\n",creal(Phikk[ci]),cimag(Phikk[ci]));
     }
     fprintf(dfp,"];\n");
#endif
   }
#ifdef DEBUG1
   if (Data::spatialreg) {
    fclose(dfp);
   }
#endif

   /**************** for plotting of spatial model *****************/
   int pn_axes_M=30; /* l,m axis size M */
   int pn_nfreq=0;//iodata.Nms-1; /* freq to plot */
   /**************** end plotting of spatial model *****************/

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
    

    complex double *Zbar=0; /* constraint for each direction, 2Nx2 x Npoly x M */
    complex double *Zspat=0; /* spatial constraint matrix, 2*Npoly*N x 2G */
    complex double *Zspat_diff=0; /* spatial constrait matrix used by the diffuse model if any, 2*Npoly*N x 2G similar to  Zspat */
    complex double *Zspat_diff0=0; /* initial value for spatial constrait matrix used by the diffuse model if any, 2*Npoly*N x 2G similar to  Zspat */
    complex double *Psi_diff=0; /* Lagrange multiplier for constraint Zspat=Zspat_diff used by the diffuse model if any, 2*Npoly*N x 2G similar to  Zspat */
    double *X=0; /* Lagrange multiplier for spatial reg Z=Zbar, 2*2*Npoly*N x 2 x M (double) */
    /*SP: spatial update */
    if (Data::spatialreg) {
      if ((Zbar=(complex double*)calloc((size_t)iodata.N*4*Npoly*iodata.M,sizeof(complex double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      if ((Zspat=(complex double*)calloc((size_t)iodata.N*4*Npoly*G,sizeof(complex double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      if ((X=(double*)calloc((size_t)iodata.N*8*Npoly*iodata.M,sizeof(double)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
    }

    /* file for saving solutions */
    FILE *sfp=0;
    FILE *sp_sfp=0;
    if (solfile) {
     if ((sfp=fopen(solfile,"w+"))==0) {
       fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
       exit(1);
     }
     if (Data::spatialreg) {
      string filebuff=std::string("spatial_")+std::string(solfile);
      if ((sp_sfp=fopen(filebuff.c_str(),"w+"))==0) {
       fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
       exit(1);
      }
     }
    }

    /* write additional info to solution file */
    if (solfile) {
      fprintf(sfp,"# solution file (Z) created by SAGECal\n");
      fprintf(sfp,"# reference_freq(MHz) polynomial_order stations clusters effective_clusters\n");
      fprintf(sfp,"%lf %d %d %d %d\n",iodata.freq0*1e-6,Npoly,iodata.N,Mo,iodata.M);
      if (Data::spatialreg) {
        fprintf(sp_sfp,"# spatial regularization solution file (Zspat) created by SAGECal\n");
        fprintf(sp_sfp,"# Top two rows are the polar coordinates of the centroids (rad)\n");
        fprintf(sp_sfp,"# reference_freq(MHz) polynomial_order(freq) polynomial_order(spatial) stations clusters effective_clusters\n");
        fprintf(sp_sfp,"%lf %d %d %d %d %d\n",iodata.freq0*1e-6,Npoly,G,iodata.N,Mo,iodata.M);
        /* write spatial centroids to solution file */
        for (int ci=0; ci<iodata.M; ci++) {
          fprintf(sp_sfp," %lf",ll[ci]);
        }
        fprintf(sp_sfp,"\n");
        for (int ci=0; ci<iodata.M; ci++) {
          fprintf(sp_sfp," %lf",mm[ci]);
        }
        fprintf(sp_sfp,"\n");
      }
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

    double *alphak=0; /* alpha array Mx1, for spatial regularization */
    if (Data::spatialreg) {
     if ((alphak=(double*)calloc((size_t)iodata.M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }

    /* if text file is given, read it and update rho array 
     * also if spatialreg is on, text file should have those values */
    if (Data::admm_rho_file) {
     read_arho_fromfile(Data::admm_rho_file,iodata.M,arho,Mo,arhoslave,Data::spatialreg,alphak);
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
    if (Data::spatialreg && (!Data::admm_rho_file)) {
     /* if spatial regularization is not given in text file,
      * scale up/down each alpha based on initial rho value,
      * for cluster with max rho, scale is 1 */
     double maxrho=arho[my_idamax(iodata.M,arho,1)-1];
     for (int cm=0; cm<iodata.M; cm++) {
       alphak[cm]=Data::federated_reg_alpha*arho[cm]/maxrho;
     }
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

    if (Data::spatialreg && sp_diffuse_id>=0) {
       /* Initialize spatial model used in diffuse sky to a nominal value,
        * for example to match the nomianal beam model or initialize to all zero */
       if ((Zspat_diff=(complex double*)calloc((size_t)iodata.N*4*Npoly*G,sizeof(complex double)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
       }
       if ((Zspat_diff0=(complex double*)calloc((size_t)iodata.N*4*Npoly*G,sizeof(complex double)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
       }

       /* B_f Zspat_diff_0 Phi_k = J_0, where J_0 = 1_N \kron I_2 */
       /* B_f : 2N x 2Npoly N, Phi_k : 2G x 2, J_0 : 2N x 2, Z_spat_diff: 2Npoly N x 2G */
       /* so Zspat_diff_0 = (\sum_f B_f^T B_f)^{-1} (\sum_f B_f^T) J_0 (\sum_k Phi_k^H) (\sum_k Phi_k Phi_k^H)^{-1} */
       find_initial_spatial(B,phivec,Npoly,iodata.N,iodata.Nms,iodata.M,G,Zspat_diff0);
       if ((Psi_diff=(complex double*)calloc((size_t)iodata.N*4*Npoly*G,sizeof(complex double)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
       }
       /* before admm iteration, send all necessary aux info */
       for (int cm=0; cm<nslaves; cm++) {
         MPI_Send(&sh_n0, 1, MPI_INT, cm+1,TAG_SPATIAL, MPI_COMM_WORLD);
         MPI_Send(&iodata.Nms, 1, MPI_INT, cm+1,TAG_SPATIAL, MPI_COMM_WORLD);
         MPI_Send(&sh_beta, 1, MPI_DOUBLE, cm+1, TAG_SPATIAL, MPI_COMM_WORLD);
         MPI_Send(B, Npoly*iodata.Nms, MPI_DOUBLE, cm+1,TAG_SPATIAL, MPI_COMM_WORLD);
       }
    }


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

        /* spatial regularization with a valid diffuse model: send each worker updated spatial model for its next MS */
        if (Data::spatialreg && sp_diffuse_id>=0 && !(admm%Data::admm_cadence)) {
          /* at start of each ADMM iteration, set to initial value */
          if (!admm) {
            memcpy(Zspat_diff,Zspat_diff0,sizeof(complex double)*(size_t)iodata.N*4*Npoly*G);
          }
          for (int cm=0; cm<nslaves; cm++) {
             MPI_Send(Zspat_diff, iodata.N*8*Npoly*G, MPI_DOUBLE, cm+1,TAG_SPATIAL, MPI_COMM_WORLD);
          }
        }

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
         if (!Data::spatialreg) {
          find_prod_inverse_full(B,Bii,Npoly,iodata.Nms,iodata.M,rhok,Data::Nt);
         } else {
          find_prod_inverse_full_fed(B,Bii,Npoly,iodata.Nms,iodata.M,rhok,alphak,Data::Nt);
         }

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
         /* no need to scale by 1/rho above, because Bii is already divided by 1/rho */
         /* add (alpha Zbar - X) if spatial regularization is enabled */
         /* Zbar-2col, X,z-1col */
         if (Data::spatialreg && admm>0) {
           double *Zbar_r=(double*)Zbar;
           for (int cm=0; cm<iodata.M; cm++) {
             for (int np=0; np<Npoly; np++) {
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+0],4,alphak[cm],&z[cm*iodata.N*8*Npoly+0],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+1],4,alphak[cm],&z[cm*iodata.N*8*Npoly+1],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+2],4,alphak[cm],&z[cm*iodata.N*8*Npoly+4],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+3],4,alphak[cm],&z[cm*iodata.N*8*Npoly+5],8);

              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+0],4,alphak[cm],&z[cm*iodata.N*8*Npoly+2],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+1],4,alphak[cm],&z[cm*iodata.N*8*Npoly+3],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+2],4,alphak[cm],&z[cm*iodata.N*8*Npoly+6],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+3],4,alphak[cm],&z[cm*iodata.N*8*Npoly+7],8);
             }
           }
           my_daxpy(iodata.N*8*Npoly*iodata.M,X,-1.0,z);
         }

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

         if (Data::spatialreg && !(admm%Data::admm_cadence)) {
           /*SP: spatial update */
           /* 1. update Zbar  from global sol Z (copy) */
           /* Note the row ordering Z: 2*2*N x Npoly (for each M)
            * Zbar: 2*N*Npoly x 2 ( note 2 columns ) for each M */
           /* Z-1col Zbar-2col */
           /* Split each 4N of Z into two 2N values, copy to first half and second half of Zbar */
           double *Zbar_r=(double*)Zbar;
           for (int cm=0; cm<iodata.M; cm++) {
            for (int np=0; np<Npoly; np++) {
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+0],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+0],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+1],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+1],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+4],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+2],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+5],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+3],4);

             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+2],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+0],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+3],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+1],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+6],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+2],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+7],8,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+3],4);
            }
           }
           /* 2. update Zspat taking proximal step (FISTA) */
           if (sp_diffuse_id>=0) {
            if (!admm) {
             memset(Psi_diff,0,sizeof(complex double)*(size_t)iodata.N*4*Npoly*G);
            }
            /* find Z, min \sum_k (Z_k -Z Phi_k) + \lambda ||Z||^2 + \mu ||Z||_1 + \Psi^H(Z-Z_diff) + \gamma/2 ||Z-Z_diff||^2, note Phikk already has \lambda I added  */
            update_spatialreg_fista_with_diffconstraint(Zspat,Zbar,Phikk,Phi,Zspat_diff,Psi_diff,iodata.N,iodata.M,Npoly,G,sh_mu, sp_gamma, fista_maxiter);
            /* find Z_diff, min ||Z_diff - Z_diff0||^2 + \Psi^H(Z-Z_diff) + \gamma/2||Z-Z_diff||^2 + \lambda ||Z_diff||^2 */
            /* grad = (Z_diff-Z_diff0) -1/2 Psi - gamma/2 (Z-Z_diff) + \lambda Z_diff */
            /* Z_diff <= (Z_diff0 + 1/2 Psi + gamma/2 Z) / ( 1  + gamma/2 + \lambda)  */

            memcpy(Zspat_diff,Zspat_diff0,sizeof(complex double)*(size_t)iodata.N*4*Npoly*G);
            my_caxpy(2*Npoly*iodata.N*2*G, Psi_diff, 0.5, Zspat_diff);
            my_caxpy(2*Npoly*iodata.N*2*G, Zspat, 0.5*sp_gamma, Zspat_diff);
            my_cscal(2*Npoly*iodata.N*2*G, 1.0/(1+0.5*sp_gamma+sh_lambda), Zspat_diff);

            /* update Lagrange multiplier \Psi = \Psi + \gamma (Z-Zdiff) */
            my_caxpy(2*Npoly*iodata.N*2*G, Zspat, sp_gamma, Psi_diff);
            my_caxpy(2*Npoly*iodata.N*2*G, Zspat_diff, -sp_gamma, Psi_diff);
           } else {
            /* min \sum_k (Z_k -Z Phi_k) + \lambda ||Z||^2 + \mu ||Z||_1, note Phikk already has \lambda I added  */
            update_spatialreg_fista(Zspat,Zbar,Phikk,Phi,iodata.N,iodata.M,Npoly,G,sh_mu, fista_maxiter);
           }
           /* 3. update Zbar from Zspat, Z_k = Z Phi_k */
           /* Note the row ordering is 2*N*Npoly x 2 x M (each M has 2 cols) */
           for (int cm=0; cm<iodata.M; cm++) {
             my_zgemm('N','N',2*Npoly*iodata.N,2,2*G,1.0+_Complex_I*0.0,Zspat,2*Npoly*iodata.N,&Phi[cm*2*G*2],2*G,0.0+_Complex_I*0.0,&Zbar[cm*2*Npoly*iodata.N*2],2*Npoly*iodata.N);
           }
           /* 4. update X comparing Zbar, Z */
           /* Zerr=Z-Zbar */
           /* Note the row ordering Z: 2N*2 x Npoly (for each M) and
            * Zbar: 2*N*Npoly x 2 (for each M) */
           /* Z-1col Zerr-2col */
           for (int cm=0; cm<iodata.M; cm++) {
            for (int np=0; np<Npoly; np++) {
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+0],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+0],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+1],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+1],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+4],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+2],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+5],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+3],4);

             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+2],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+0],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+3],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+1],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+6],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+2],4);
             my_dcopy(iodata.N,&Z[cm*iodata.N*8*Npoly+np*iodata.N*8+7],8,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+3],4);
            }
           }
           my_daxpy(iodata.N*8*Npoly*iodata.M,(double*)Zbar,-1.0,Zerr);
           /* X = X + alpha (Z - Zbar) */
           if (!admm) {
            memset(X,0,sizeof(double)*(size_t)iodata.N*8*Npoly*iodata.M);
           }
           /* Zerr-2col X-1col */
           for (int cm=0; cm<iodata.M; cm++) {
             //my_daxpy(iodata.N*8*Npoly,&Zerr[cm*iodata.N*8*Npoly],alphak[cm],&X[cm*iodata.N*8*Npoly]);
             for (int np=0; np<Npoly; np++) {
              //my_daxpy(iodata.N*4,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4],alphak[cm],&X[cm*iodata.N*8*Npoly]);
              //my_daxpy(iodata.N*4,&Zerr[cm*iodata.N*8*Npoly+Npoly*iodata.N*4+np*iodata.N*4],alphak[cm],&X[cm*iodata.N*8*Npoly+iodata.N*4]);

              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+0],4,alphak[cm],&X[cm*iodata.N*8*Npoly+0],8);
              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+1],4,alphak[cm],&X[cm*iodata.N*8*Npoly+1],8);
              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+2],4,alphak[cm],&X[cm*iodata.N*8*Npoly+4],8);
              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+3],4,alphak[cm],&X[cm*iodata.N*8*Npoly+5],8);

              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+0],4,alphak[cm],&X[cm*iodata.N*8*Npoly+2],8);
              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+1],4,alphak[cm],&X[cm*iodata.N*8*Npoly+3],8);
              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+2],4,alphak[cm],&X[cm*iodata.N*8*Npoly+6],8);
              my_daxpys(iodata.N,&Zerr[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+3],4,alphak[cm],&X[cm*iodata.N*8*Npoly+7],8);
             }
           }
           printf("SP: ADMM %d: ||Z-Zbar||=%lf ||Z||=%lf ||X||=%lf",admm,my_dnrm2(iodata.N*8*Npoly*iodata.M,Zerr)/((double)iodata.N*8*Npoly*iodata.M),my_dnrm2(iodata.N*8*Npoly*iodata.M,Z)/((double)iodata.N*8*Npoly*iodata.M),my_dnrm2(iodata.N*8*Npoly*iodata.M,X)/((double)iodata.N*8*Npoly*iodata.M));
           if (sp_diffuse_id>=0) {
            printf(" ||Z_diff||=%lf ||Psi||=%lf\n", my_dnrm2(iodata.N*8*Npoly*G,(double*)Zspat_diff)/((double)iodata.N*8*Npoly*G), my_dnrm2(iodata.N*8*Npoly*G,(double*)Psi_diff)/((double)iodata.N*8*Npoly*G));
           } else {
            printf("\n");
           }
           /* 5. feed Zbar and X to next update of Z
             already done above*/
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
       
	           for (int ct1=0; ct1<scount; ct1++) {
	            mmid = Sbegin[cm]+ct1;
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

        if (Data::aadmm) {
        /* BB : get updated rho from slaves */
        for (int cm=0; cm<nslaves; cm++) {
           if (Sbegin[cm]>=0) {
            int mmid=Sbegin[cm]+Scurrent[cm];
            MPI_Recv(arhoslave,Mo,MPI_DOUBLE,cm+1,TAG_RHO_UPDATE,MPI_COMM_WORLD,&status);
            /* now copy this to proper locations, using chunkvec as guide */
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
        }
  
        /* go to the next MS in the next ADMM iteration */
        for (int cm=0; cm<nslaves; cm++) {
          if (Sbegin[cm]>=0 && (Sbegin[cm]!=Send[cm])) { /* skip cases with only one MS */
           /* increment current MS for each slave */
           Scurrent[cm]=(Sbegin[cm]+Scurrent[cm]>=Send[cm]?0:Scurrent[cm]+1);
          }
        }

      }
      
      if (Data::use_global_solution) {
      cout<<"Using global solution for residual calculation"<<endl;

      /* get back final solutions from each worker */
      /* get Y_i+rho J_i */
      for(int cm=0; cm<nslaves; cm++) {
            if (Sbegin[cm]>=0) {
             int scount=Send[cm]-Sbegin[cm]+1;
             for (int ct1=0; ct1<scount; ct1++) {
              int mmid=Sbegin[cm]+ct1;
              MPI_Recv(&Y[mmid*iodata.N*8*iodata.M], iodata.N*8*iodata.M, MPI_DOUBLE, cm+1,TAG_YDATA, MPI_COMM_WORLD, &status);
             }
            }
      }


      /* recalculate Z based on final solutions */
  
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
         /* add (alpha Zbar - X) if spatial regularization is enabled */
         /* Zbar-2col, X,z-1col */
         if (Data::spatialreg) {
           double *Zbar_r=(double*)Zbar;
           for (int cm=0; cm<iodata.M; cm++) {
             for (int np=0; np<Npoly; np++) {
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+0],4,alphak[cm],&z[cm*iodata.N*8*Npoly+0],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+1],4,alphak[cm],&z[cm*iodata.N*8*Npoly+1],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+2],4,alphak[cm],&z[cm*iodata.N*8*Npoly+4],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+3],4,alphak[cm],&z[cm*iodata.N*8*Npoly+5],8);

              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+0],4,alphak[cm],&z[cm*iodata.N*8*Npoly+2],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+1],4,alphak[cm],&z[cm*iodata.N*8*Npoly+3],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+2],4,alphak[cm],&z[cm*iodata.N*8*Npoly+6],8);
              my_daxpys(iodata.N,&Zbar_r[cm*iodata.N*8*Npoly+np*iodata.N*4+Npoly*iodata.N*4+3],4,alphak[cm],&z[cm*iodata.N*8*Npoly+7],8);
             }
           }
           my_daxpy(iodata.N*8*Npoly*iodata.M,X,-1.0,z);
         }

         /* find product z_tilde x Bi^T, z_tilde with proper reshaping */
         my_dcopy(iodata.N*8*Npoly*iodata.M,Z,1,Zerr,1);
         update_global_z_multi(Z,iodata.N,iodata.M,Npoly,z,Bii,Data::Nt);
         /* find dual error ||Zold-Znew|| (note Zerr is destroyed here)  */
         my_daxpy(iodata.N*8*Npoly*iodata.M,Z,-1.0,Zerr);
         /* dual residual per one real parameter */
         if (Data::verbose) {
          cout<<"ADMM : "<<Nadmm<<" dual residual="<<my_dnrm2(iodata.N*8*Npoly*iodata.M,Zerr)/sqrt((double)8*iodata.N*Npoly*iodata.M)<<endl;
         } else {
          cout<<"Timeslot:"<<ct<<" ADMM:"<<Nadmm<<endl;
         }

      /* calculate global solution for each MS and send them to workers */

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
      /* 2Nx2 x Npoly x M */
      for (int p=0; p<iodata.N*8*Npoly; p++) {
       fprintf(sfp,"%d ",p);
       for (int ppi=iodata.M-1; ppi>=0; ppi--) { /* reverse ordering */
        fprintf(sfp," %e",Z[ppi*iodata.N*8*Npoly+p]);
       }
       fprintf(sfp,"\n");
      }
      if (Data::spatialreg) {
       /* 2N*Npoly x 2G */
       double *Zspre=(double*)Zspat;
       for (int p=0; p<iodata.N*8*Npoly; p++) {
        fprintf(sp_sfp,"%d ",p);
        for (int ppi=0; ppi<G; ppi++) {
         fprintf(sp_sfp," %e",Zspre[ppi*iodata.N*8*Npoly+p]);
        }
        fprintf(sp_sfp,"\n");
       }
      }
     }
     if (resetcount>nslaves/2) {
       /* if most slaves have reset, print a warning only */
       cout<<"Warning: Most slaves did not converge."<<endl;
     }


     if (Data::spatialreg) { 
       /* plotting : create plot */
       std::string imagename=std::string("J_")+std::to_string(ct)+std::string(".ppm");
       /* this is slow, because the basis vectors need to be re-calculated, but this keeps the code cleaner */
       plot_spatial_model(Zspat,B,Npoly,iodata.N,sh_n0,iodata.Nms,pn_axes_M,pn_nfreq,0,spatialreg_basis,sh_beta,imagename.c_str());
     }

    } /* time */

    /* send end signal to each slave */
    msgcode=CTRL_END;
    for(int cm=0; cm<nslaves; cm++) {
        MPI_Send(&msgcode, 1, MPI_INT, cm+1,TAG_CTRL, MPI_COMM_WORLD);
    }

    if (solfile) {
      fclose(sfp);
      if (Data::spatialreg) {
        fclose(sp_sfp);
      }
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

   if (Data::spatialreg) {
      exinfo_gaussian *exg;
      exinfo_disk *exd;
      exinfo_ring *exr;
      exinfo_shapelet *exs;

      for (int ci=0; ci<Mo; ci++) {
       free(carr[ci].ll);
       free(carr[ci].mm);
       free(carr[ci].nn);
       free(carr[ci].sI);
       free(carr[ci].sQ);
       free(carr[ci].sU);
       free(carr[ci].sV);
       //Do not free(carr[ci].p) as it is not allocated;
       free(carr[ci].ra);
       free(carr[ci].dec);
       for (int cj=0; cj<carr[ci].N; cj++) {
        /* do a proper typecast before freeing */
        switch (carr[ci].stype[cj]) {
          case STYPE_GAUSSIAN:
           exg=(exinfo_gaussian*)carr[ci].ex[cj];
           if (exg) { free(exg); carr[ci].ex[cj]=0; }
           break;
          case STYPE_DISK:
           exd=(exinfo_disk*)carr[ci].ex[cj];
           if (exd) { free(exd); carr[ci].ex[cj]=0; }
           break;
          case STYPE_RING:
           exr=(exinfo_ring*)carr[ci].ex[cj];
           if (exr) { free(exr); carr[ci].ex[cj]=0; }
           break;
          case STYPE_SHAPELET:
           exs=(exinfo_shapelet*)carr[ci].ex[cj];
           if (exs)  {
            if (exs->modes) {
            free(exs->modes);
           }
           free(exs);
           carr[ci].ex[cj]=0;
           }
           break;
          default:
           break;
        }
      }
      free(carr[ci].ex);
      free(carr[ci].stype);
      free(carr[ci].sI0);
      free(carr[ci].sQ0);
      free(carr[ci].sU0);
      free(carr[ci].sV0);
      free(carr[ci].f0);
      free(carr[ci].spec_idx);
      free(carr[ci].spec_idx1);
      free(carr[ci].spec_idx2);
     }
     free(carr);
     free(ll);
     free(mm);
     free(phivec);
     free(Phi);
     free(Phikk);
     free(Zbar);
     free(Zspat);
     if (sp_diffuse_id>=0) {
      free(Zspat_diff);
      free(Zspat_diff0);
      free(Psi_diff);
     }
     free(X);
     free(alphak);
   }
  /**********************************************************/

   cout<<"Done."<<endl;    
   return 0;
}
