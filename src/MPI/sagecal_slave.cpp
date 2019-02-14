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
#include <time.h>

#include <map>
#include <string>
#include <cstring>
#include <glob.h>


#include <Dirac.h>
#include <Radio.h>
#include <mpi.h>

#ifndef LMCUT
#define LMCUT 40
#endif

using namespace std;
using namespace Data;
//#define DEBUG

int 
sagecal_slave(int argc, char **argv) {
    ParseCmdLine(argc, argv);
    if (!Data::SkyModel || !Data::Clusters || !Data::MSpattern) {
      print_help();
      MPI_Finalize();
      exit(1);
    }

    openblas_set_num_threads(1);//Data::Nt;

    /* determine my MPI rank */
    int myrank;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    /**** setup MS names ***************************************/
    // slave sends host name to master
    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);
    MPI_Send(name,strlen(name),MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD);

    //slaves wait for request from master, or idle signal
    int count;
    MPI_Probe(0,TAG_MSNAME,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_CHAR,&count);
    // allocate memory
    char *buf=new char[count];
    MPI_Recv(buf,count,MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
    string buf1(buf,count);
    delete [] buf;
    //if we have received a non null string
    if (count > 0) {
      // search for matching file names
      glob_t glob_result;
      int globret=glob(buf1.c_str(),GLOB_TILDE,NULL,&glob_result);
      vector<string> ret;
      for(unsigned int i=0;i<glob_result.gl_pathc;++i){
          ret.push_back(string(glob_result.gl_pathv[i]));
      }
      globfree(&glob_result);
      // check for errors and exist
      if (globret>0) {
cout<<"Error in checking files matching pattern "<<buf1<<". Exiting."<<endl;
       exit(1);
      } else {
       //slave finds the file list and send back to master
       int nfiles=ret.size();
        MPI_Send(&nfiles,1,MPI_INT,0,TAG_MSNAME,MPI_COMM_WORLD);
        //also send the file names
        int cj;
        for (cj=0; cj<nfiles; cj++) {
          MPI_Send(ret[cj].c_str(),ret[cj].length(),MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD);
        }
      }
   }

     //get back the MS names for this slave
     vector<string> myms;
     vector<int> myids;
     int mymscount,mymsid;
     MPI_Recv(&mymscount,1,MPI_INT,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
     if (mymscount>0) {
      for (int ch=0; ch<mymscount; ch++) {
        MPI_Probe(0,TAG_MSNAME,MPI_COMM_WORLD,&status);
        MPI_Get_count(&status,MPI_CHAR,&count);
        buf=new char[count];
        MPI_Recv(buf,count,MPI_CHAR,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
        string buf2(buf,count);
        myms.push_back(buf2);
        // also get the id of this MS
        MPI_Recv(&mymsid,1,MPI_INT,0,TAG_MSNAME,MPI_COMM_WORLD,&status);
        myids.push_back(mymsid);

        delete [] buf;
      }
     } else {
cout<<"Slave "<<myrank<<" has nothing to do"<<endl;
       return 0;
     }


    /**** end setup MS names ***************************************/
    //create vectors to store data, beam, sky info etc for each MS
    vector<Data::IOData> iodata_vec(mymscount);
    vector<Data::LBeam> beam_vec(mymscount);

    if (Data::randomize) {
     srand(time(0)); /* use different seed */
    } else {
     srand(0);
    }
    /**********************************************************/
     int M,Mt,ci,cj,ck;  
    /* parameters */
    double *pinit;

    for(int cm=0; cm<mymscount; cm++) {
     iodata_vec[cm].tilesz=Data::TileSize;
     iodata_vec[cm].deltat=1.0;
     if (!doBeam) {
      Data::readAuxData(myms[cm].c_str(),&iodata_vec[cm]);
     } else {
      Data::readAuxData(myms[cm].c_str(),&iodata_vec[cm],&beam_vec[cm]);
     }
    }
    fflush(stdout);

    vector<FILE *> sfp_vec(mymscount);
    for(int cm=0; cm<mymscount; cm++) {
     /* always create default solution file name MS+'.solutions' */
     string filebuff=std::string(myms[cm])+std::string(".solutions\0");
     if ((sfp_vec[cm]=fopen(filebuff.c_str(),"w+"))==0) {
       fprintf(stderr,"%s: %d: no file\n",__FILE__,__LINE__);
       exit(1);
     }
    }

     double mean_nu;
     vector<clus_source_t *> carr_vec(mymscount);
     vector<baseline_t *> barr_vec(mymscount);

     for(int cm=0; cm<mymscount; cm++) {
      read_sky_cluster(Data::SkyModel,Data::Clusters,&carr_vec[cm],&M,iodata_vec[cm].freq0,iodata_vec[cm].ra0,iodata_vec[cm].dec0,Data::format);
     }

     /* exit if there are 0 clusters (incorrect sky model/ cluster file)*/
     if (M<=0) {
      fprintf(stderr,"%s: %d: no clusters to solve\n",__FILE__,__LINE__);
      exit(1);
     } else {
      printf("Got %d clusters\n",M);
     }

     /* array to store baseline->sta1,sta2 map */
     for(int cm=0; cm<mymscount; cm++) {
      if ((barr_vec[cm]=(baseline_t*)calloc((size_t)iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,sizeof(baseline_t)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      generate_baselines(iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,iodata_vec[cm].N,barr_vec[cm],Data::Nt);
     }

     /* calculate actual no of parameters needed,
      this could be > M */
     Mt=0;
     for (ci=0; ci<M; ci++) {
       //printf("cluster %d has %d time chunks\n",carr_vec[0][ci].id,carr_vec[0][ci].nchunk);
       Mt+=carr_vec[0][ci].nchunk;
     }
     printf("Total effective clusters: %d\n",Mt);
     /* create an array with chunk sizes for each cluster */
     int *chunkvec=0;
     if (myrank==1) {
      if ((chunkvec=(int *)calloc((size_t)M,sizeof(int)))==0) {
       fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
       exit(1);
      }
      for (ci=0; ci<M; ci++) {
       chunkvec[ci]=carr_vec[0][ci].nchunk;
      }
     }

    vector<double *> p_vec(mymscount);
    vector<double **> pm_vec(mymscount);

    for(int cm=0; cm<mymscount; cm++) {
    /* parameters 8*N*M ==> 8*N*Mt */
     if ((p_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].N*8*Mt,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }
    for(int cm=0; cm<mymscount; cm++) {
     /* update cluster array with correct pointers to parameters */
     cj=0;
     for (ci=0; ci<M; ci++) {
       if ((carr_vec[cm][ci].p=(int*)calloc((size_t)carr_vec[cm][ci].nchunk,sizeof(int)))==0) {
        fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
        exit(1);
       }
       for (ck=0; ck<carr_vec[cm][ci].nchunk; ck++) {
         carr_vec[cm][ci].p[ck]=cj*8*iodata_vec[cm].N;
         cj++;
       }
     }
   }

  for(int cm=0; cm<mymscount; cm++) {
   /* pointers to parameters */
   if ((pm_vec[cm]=(double**)calloc((size_t)Mt,sizeof(double*)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
   /* setup the pointers */
   for (ci=0; ci<Mt; ci++) {
    pm_vec[cm][ci]=&(p_vec[cm][ci*8*iodata_vec[cm].N]);
   }
   /* initilize parameters to [1,0,0,0,0,0,1,0] */
   if (!initsolfile) {
    for (ci=0; ci<Mt; ci++) {
     for (cj=0; cj<iodata_vec[cm].N; cj++) {
       pm_vec[cm][ci][8*cj]=1.0;
       pm_vec[cm][ci][8*cj+6]=1.0;
     }
    }
   } 
  }
  /* initialize solutions by reading from a file */
  if ((pinit=(double*)calloc((size_t)iodata_vec[0].N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
  }
  if (initsolfile) {
      FILE *sfq;
      if ((sfq=fopen(initsolfile,"r"))==0) {
       fprintf(stderr,"%s: %d: no solution file present\n",__FILE__,__LINE__);
       exit(1);
      }
      /* remember to skip first 3 lines from solution file */
      char chr;
      for (ci=0; ci<3; ci++) {
       do {
        chr = fgetc(sfq);
       } while (chr != '\n');
      }
     printf("Initializing solutions from %s\n",initsolfile);
     read_solutions(sfq,pinit,carr_vec[0],iodata_vec[0].N,M);
     fclose(sfq);

   /* backup of default initial values */
   for(int cm=0; cm<mymscount; cm++) {
    memcpy(p_vec[cm],pinit,(size_t)iodata_vec[0].N*8*Mt*sizeof(double));
   }
  } else {
    /* backup of default initial values */
    memcpy(pinit,p_vec[0],(size_t)iodata_vec[0].N*8*Mt*sizeof(double));
  }

  vector<complex double *> coh_vec(mymscount);
  /* coherencies */
  for(int cm=0; cm<mymscount; cm++) {
   if ((coh_vec[cm]=(complex double*)calloc((size_t)(M*iodata_vec[cm].Nbase*iodata_vec[cm].tilesz*4),sizeof(complex double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
   }
  }


   double res_0,res_1;
   vector<double> res_00(mymscount),res_01(mymscount),res_0vec(mymscount),res_1vec(mymscount),res_prev(mymscount);   
   double res_ratio=15.0; /* how much can the residual increase before resetting solutions, set higher than stand alone mode */
    res_0=res_1=0.0;
    for(int cm=0; cm<mymscount; cm++) {
     res_00[cm]=res_01[cm]=0.0;
     /* previous residual */
     res_prev[cm]=CLM_DBL_MAX;
    }

    /**********************************************************/
    Block<int> sort(1);
    sort[0] = MS::TIME; /* note: only sort over TIME for ms iterator to work */
    /* timeinterval in seconds */

    for(int cm=0; cm<mymscount; cm++) {
     cout<<"For "<<iodata_vec[cm].tilesz<<" samples, solution time interval (s): "<<iodata_vec[cm].deltat*(double)iodata_vec[cm].tilesz<<endl;
     cout<<"Freq: "<<iodata_vec[cm].freq0/1e6<<" MHz, Chan: "<<iodata_vec[cm].Nchan<<" Bandwidth: "<<iodata_vec[cm].deltaf/1e6<<" MHz"<<endl;
    }
    vector<MSIter*> msitr;
    vector<MeasurementSet*> msvector;
    for(int cm=0; cm<mymscount; cm++) {
      MeasurementSet *ms=new MeasurementSet(myms[cm],Table::Update); 
      MSIter *mi=new MSIter(*ms,sort,iodata_vec[cm].deltat*(double)iodata_vec[cm].tilesz);
      msitr.push_back(mi);
      msvector.push_back(ms);
    }
   
    time_t start_time, end_time;
    double elapsed_time;

    int tilex=0;
    for(int cm=0; cm<mymscount; cm++) {
      msitr[cm]->origin();
    }

    /* write additional info to solution file */
    for(int cm=0; cm<mymscount; cm++) {
     fprintf(sfp_vec[cm],"# solution file created by SAGECal\n");
     fprintf(sfp_vec[cm],"# freq(MHz) bandwidth(MHz) time_interval(min) stations clusters effective_clusters\n");
     fprintf(sfp_vec[cm],"%lf %lf %lf %d %d %d\n",iodata_vec[cm].freq0*1e-6,iodata_vec[cm].deltaf*1e-6,(double)iodata_vec[cm].tilesz*iodata_vec[cm].deltat/60.0,iodata_vec[cm].N,M,Mt);
    }



    /**** send info to master ***************************************/
    /* send msid, freq (freq0), no. stations (N), total timeslots (totalt), no. of clusters (M), true no. of clusters with hybrid (Mt), integration time (deltat), bandwidth (deltaf) */
    int *bufint=new int[6];
    double *bufdouble=new double[1];
    for(int cm=0; cm<mymscount; cm++) {
     bufint[0]=myids[cm];
     bufint[1]=iodata_vec[cm].N;
     bufint[2]=M;
     bufint[3]=Mt;
     bufint[4]=iodata_vec[cm].tilesz;
     bufint[5]=iodata_vec[cm].totalt;
     bufdouble[0]=iodata_vec[cm].freq0;
     MPI_Send(bufint, 6, MPI_INT, 0,TAG_MSAUX, MPI_COMM_WORLD);
     MPI_Send(bufdouble, 1, MPI_DOUBLE, 0,TAG_MSAUX, MPI_COMM_WORLD);
    }

    delete [] bufint;
    delete [] bufdouble;


    /* ADMM memory : seperate for each MS */
    vector<double *> Z_vec(mymscount);
    vector<double *> Y_vec(mymscount);

    /* BB */
    vector<double *> Yhat0_vec(mymscount);
    vector<double *> J0_vec(mymscount);
    double *Yhat;
    for(int cm=0; cm<mymscount; cm++) {
     /* Z: (store B_f Z) 2Nx2 x M */
     if ((Z_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].N*8*Mt,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     /* Zold: (store B_f Z_old) 2Nx2 x M */
     if ((Yhat0_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].N*8*Mt,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     /* to store old solution */
     if ((J0_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].N*8*Mt,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }


     /* Y, 2Nx2 , M times */
     if ((Y_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].N*8*Mt,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }
    /* All freqs has N equal is assumed here */
    if ((Yhat=(double*)calloc((size_t)iodata_vec[0].N*8*Mt,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
    }

    /* primal residual J-BZ : use only 1 for all MS */
    double *pres;
    if ((pres=(double*)calloc((size_t)iodata_vec[0].N*8*Mt,sizeof(double)))==0) {
     fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
     exit(1);
    }

    vector<double *> arho_vec(mymscount);
    vector<double *> arho0_vec(mymscount);
    vector<double *> arhoupper_vec(mymscount); /* upper limit of rho */
    
    for(int cm=0; cm<mymscount; cm++) {
     if ((arho_vec[cm]=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if ((arho0_vec[cm]=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
     if ((arhoupper_vec[cm]=(double*)calloc((size_t)M,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
     }
    }

    /* get regularization factor array */
    MPI_Recv(arho0_vec[0],M,MPI_DOUBLE,0,TAG_RHO,MPI_COMM_WORLD,&status);
   
    /* send chunk size info to master */
    if (myrank==1) {
      MPI_Send(chunkvec,M,MPI_INT,0,TAG_CHUNK,MPI_COMM_WORLD);
    }

    /* keep backup of regularization factor, per frequency */
    for(int cm=1; cm<mymscount; cm++) {
      memcpy(arho0_vec[cm],arho0_vec[0],(size_t)M*sizeof(double));
    }

    /* if we have more than 1 channel, or if we whiten data, need to backup raw data */
    vector<double *> xbackup_vec(mymscount);
    for(int cm=0; cm<mymscount; cm++) {
     if (iodata_vec[cm].Nchan>1 || Data::whiten) {
      if ((xbackup_vec[cm]=(double*)calloc((size_t)iodata_vec[cm].Nbase*8*iodata_vec[cm].tilesz,sizeof(double)))==0) {
      fprintf(stderr,"%s: %d: no free memory\n",__FILE__,__LINE__);
      exit(1);
      }
     } else {
      xbackup_vec[cm]=0;
     }
    }


    int msgcode=0;
    /* starting iteration, inner iterations doubled */
    int start_iter=1;
    int sources_precessed=0;

    double inv_c=1.0/CONST_C;

#ifdef HAVE_CUDA
   /* setup Heap of GPU,  only need to be done once, before any kernel is launched  */
    if (GPUpredict>0) {
     for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
        cudaSetDevice(gpuid);
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
     }
    }
#endif


    while(1) {
     start_time = time(0);
     /* get start/end signal from master */
     MPI_Recv(&msgcode,1,MPI_INT,0,TAG_CTRL,MPI_COMM_WORLD,&status);
     /* assume all MS are the same size */  
     if (msgcode==CTRL_END || !msitr[0]->more()) {
cout<<"Slave "<<myrank<<" quitting"<<endl;
      break;
     } else if (msgcode==CTRL_SKIP) {
      /* skip to next timeslot */
      for(int cm=0; cm<mymscount; cm++) {
       (*msitr[cm])++;
      }
      tilex+=iodata_vec[0].tilesz;
      continue;
     }

     for(int cm=0; cm<mymscount; cm++) {
      /* else, load data, do the necessary preprocessing */
      if (!doBeam) {
       Data::loadData(msitr[cm]->table(),iodata_vec[cm],&iodata_vec[cm].fratio);
      } else {
       Data::loadData(msitr[cm]->table(),iodata_vec[cm],beam_vec[cm],&iodata_vec[cm].fratio);
      }
     }
     /* downweight factor for regularization, depending on amount of data flagged, 
        0.0 means all data are flagged */
     for(int cm=0; cm<mymscount; cm++) {
      iodata_vec[cm].fratio=1.0-iodata_vec[cm].fratio;
      if (Data::verbose) {
cout<<myrank<<" : "<<cm<<": downweight ratio ("<<iodata_vec[cm].fratio<<") based on flags."<<endl;
      }
     }
     for(int cm=0; cm<mymscount; cm++) {
      /* send flag ratio (0 means all flagged) to master */
      MPI_Send(&iodata_vec[cm].fratio, 1, MPI_DOUBLE, 0,TAG_FRATIO, MPI_COMM_WORLD);
     }

     for(int cm=0; cm<mymscount; cm++) {
      /* reweight regularization factors with weight based on flags */
      memcpy(arho_vec[cm],arho0_vec[cm],(size_t)M*sizeof(double));
      my_dscal(M,iodata_vec[cm].fratio,arho_vec[cm]);
      /* use upper limit x10 this value */
      memcpy(arhoupper_vec[cm],arho_vec[cm],(size_t)M*sizeof(double));
      my_dscal(M,10.0,arhoupper_vec[cm]);
     }

     for(int cm=0; cm<mymscount; cm++) {
      /* rescale u,v,w by 1/c NOT to wavelengths, that is done later in prediction */
      my_dscal(iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,inv_c,iodata_vec[cm].u);
      my_dscal(iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,inv_c,iodata_vec[cm].v);
      my_dscal(iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,inv_c,iodata_vec[cm].w);
     }


     /**********************************************************/

     for(int cm=0; cm<mymscount; cm++) {
      /* update baseline flags */
      /* and set x[]=0 for flagged values */
      preset_flags_and_data(iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,iodata_vec[cm].flag,barr_vec[cm],iodata_vec[cm].x,Data::Nt);
      /* if data is being whitened, whiten x here before copying */
      if (Data::whiten) {
        whiten_data(iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,iodata_vec[cm].x,iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].freq0,Data::Nt);
      }
      if (iodata_vec[cm].Nchan>1 || Data::whiten) { /* keep fresh copy of raw data */
        my_dcopy(iodata_vec[cm].Nbase*8*iodata_vec[cm].tilesz, iodata_vec[cm].x, 1, xbackup_vec[cm], 1);
      }
     }


     for(int cm=0; cm<mymscount; cm++) {
     /* precess source locations (also beam pointing) from J2000 to JAPP if we do any beam predictions,
      using first time slot as epoch */
      if (doBeam && !sources_precessed) {
       precess_source_locations(beam_vec[cm].time_utc[iodata_vec[cm].tilesz/2],carr_vec[cm],M,&beam_vec[cm].p_ra0,&beam_vec[cm].p_dec0,Data::Nt);
       sources_precessed=1;
      }
     }

     for(int cm=0; cm<mymscount; cm++) {
#ifndef HAVE_CUDA
      if (!doBeam) {
       precalculate_coherencies(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,coh_vec[cm],iodata_vec[cm].N,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::min_uvcut,Data::max_uvcut,Data::Nt);
      } else {
       precalculate_coherencies_withbeam(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,coh_vec[cm],iodata_vec[cm].N,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::min_uvcut,Data::max_uvcut,
        beam_vec[cm].p_ra0,beam_vec[cm].p_dec0,iodata_vec[cm].freq0,beam_vec[cm].sx,beam_vec[cm].sy,beam_vec[cm].time_utc,iodata_vec[cm].tilesz,beam_vec[cm].Nelem,beam_vec[cm].xx,beam_vec[cm].yy,beam_vec[cm].zz,Data::Nt);
      }
#endif
#ifdef HAVE_CUDA
     if (GPUpredict) {
       precalculate_coherencies_withbeam_gpu(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,coh_vec[cm],iodata_vec[cm].N,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::min_uvcut,Data::max_uvcut,
  beam_vec[cm].p_ra0,beam_vec[cm].p_dec0,iodata_vec[cm].freq0,beam_vec[cm].sx,beam_vec[cm].sy,beam_vec[cm].time_utc,iodata_vec[cm].tilesz,beam_vec[cm].Nelem,beam_vec[cm].xx,beam_vec[cm].yy,beam_vec[cm].zz,doBeam,Data::Nt);
     } else {
      if (!doBeam) {
       precalculate_coherencies(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,coh_vec[cm],iodata_vec[cm].N,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::min_uvcut,Data::max_uvcut,Data::Nt);
      } else {
       precalculate_coherencies_withbeam(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,coh_vec[cm],iodata_vec[cm].N,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::min_uvcut,Data::max_uvcut,
        beam_vec[cm].p_ra0,beam_vec[cm].p_dec0,iodata_vec[cm].freq0,beam_vec[cm].sx,beam_vec[cm].sy,beam_vec[cm].time_utc,iodata_vec[cm].tilesz,beam_vec[cm].Nelem,beam_vec[cm].xx,beam_vec[cm].yy,beam_vec[cm].zz,Data::Nt);
      }
     }
#endif
     }
 
     /******************** ADMM  *******************************/
     int mmid_prev=-1;
     for (int admm=0; admm<Nadmm; admm++) {
       /* receive which MS to work on in this ADMM iteration */
      int mmid;
      MPI_Recv(&mmid,1,MPI_INT,0,TAG_CTRL,MPI_COMM_WORLD,&status);
      /* for later iterations, if working MS has changed,  the B_i Z value needs to be updated
         so get it from the master */
      if (admm==Nadmm-1) { /* for last ADMM, update for all MS */
       for(int cm=0; cm<mymscount; cm++) {
         MPI_Recv(Z_vec[cm], iodata_vec[cm].N*8*Mt, MPI_DOUBLE, 0,TAG_CONSENSUS, MPI_COMM_WORLD, &status);
       }
      } else if (admm>0 && (mmid_prev!=mmid)) {
       MPI_Recv(Z_vec[mmid], iodata_vec[mmid].N*8*Mt, MPI_DOUBLE, 0,TAG_CONSENSUS, MPI_COMM_WORLD, &status);
      }
      mmid_prev=mmid;

      /* ADMM 1: minimize cost function, for all MS */
      if (admm==0) { 
       for(int cm=0; cm<mymscount; cm++) {
#ifndef HAVE_CUDA
      if (start_iter) {
       sagefit_visibilities(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,(iodata_vec[cm].N<=LMCUT?4*Data::max_emiter:6*Data::max_emiter),Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata_vec[cm].N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata_vec[cm].N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1); 
      } else {
       sagefit_visibilities(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
      }
#endif /* !HAVE_CUDA */
#ifdef HAVE_CUDA
      if (start_iter) {
       sagefit_visibilities_dual_pt_flt(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,(iodata_vec[cm].N<=LMCUT?4*Data::max_emiter:6*Data::max_emiter),Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata_vec[cm].N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata_vec[cm].N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
      } else {
       sagefit_visibilities_dual_pt_flt(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
      }
#endif /* HAVE_CUDA */
       /* remember initial residual (taken equal over all MS) */
        res_0vec[cm]=res_00[cm]=res_0;
        res_1vec[cm]=res_01[cm]=res_1;
#ifdef DEBUG
        cout<<myrank<<": MS :"<<cm<<" residual "<<res_0<<" -> "<<res_1<<endl;
#endif
       }
       if (start_iter) {
        start_iter=0;
       }
      } else if (admm==Nadmm-1) { /* minimize augmented Lagrangian for all MS */ 
       for(int cm=0; cm<mymscount; cm++) {
       /* since original data is now residual, get a fresh copy of data */
       if (iodata_vec[cm].Nchan>1 || Data::whiten) {
        my_dcopy(iodata_vec[cm].Nbase*8*iodata_vec[cm].tilesz, xbackup_vec[cm], 1, iodata_vec[cm].x, 1);
       } else {
        /* only 1 channel is assumed */
        my_dcopy(iodata_vec[cm].Nbase*8*iodata_vec[cm].tilesz, iodata_vec[cm].xo, 1, iodata_vec[cm].x, 1);
       }
 
#ifndef HAVE_CUDA
       sagefit_visibilities_admm(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Y_vec[cm],Z_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[cm],&mean_nu,&res_0,&res_1);
#endif /* !HAVE_CUDA */
#ifdef HAVE_CUDA
       sagefit_visibilities_admm_dual_pt_flt(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Y_vec[cm],Z_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[cm],&mean_nu,&res_0,&res_1);
#endif /* HAVE_CUDA */
       res_0vec[cm]=res_0;
       res_1vec[cm]=res_1;

       }
      } else { /* minimize augmented Lagrangian */
       /* since original data is now residual, get a fresh copy of data */
       if (iodata_vec[mmid].Nchan>1 || Data::whiten) {
        my_dcopy(iodata_vec[mmid].Nbase*8*iodata_vec[mmid].tilesz, xbackup_vec[mmid], 1, iodata_vec[mmid].x, 1);
       } else {
        /* only 1 channel is assumed */
        my_dcopy(iodata_vec[mmid].Nbase*8*iodata_vec[mmid].tilesz, iodata_vec[mmid].xo, 1, iodata_vec[mmid].x, 1);
       }
 
#ifndef HAVE_CUDA
       sagefit_visibilities_admm(iodata_vec[mmid].u,iodata_vec[mmid].v,iodata_vec[mmid].w,iodata_vec[mmid].x,iodata_vec[mmid].N,iodata_vec[mmid].Nbase,iodata_vec[mmid].tilesz,barr_vec[mmid],carr_vec[mmid],coh_vec[mmid],M,Mt,iodata_vec[mmid].freq0,iodata_vec[mmid].deltaf,p_vec[mmid],Y_vec[mmid],Z_vec[mmid],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[mmid],&mean_nu,&res_0,&res_1);
#endif /* !HAVE_CUDA */
#ifdef HAVE_CUDA
       //sagefit_visibilities_admm(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr_vec[mmid],carr_vec[mmid],coh_vec[mmid],M,Mt,iodata.freq0,iodata.deltaf,p,Y_vec[mmid],Z_vec[mmid],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[mmid],&mean_nu,&res_0,&res_1);
       sagefit_visibilities_admm_dual_pt_flt(iodata_vec[mmid].u,iodata_vec[mmid].v,iodata_vec[mmid].w,iodata_vec[mmid].x,iodata_vec[mmid].N,iodata_vec[mmid].Nbase,iodata_vec[mmid].tilesz,barr_vec[mmid],carr_vec[mmid],coh_vec[mmid],M,Mt,iodata_vec[mmid].freq0,iodata_vec[mmid].deltaf,p_vec[mmid],Y_vec[mmid],Z_vec[mmid],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[mmid],&mean_nu,&res_0,&res_1);
#endif /* HAVE_CUDA */
       res_0vec[mmid]=res_0;
       res_1vec[mmid]=res_1;

      }

      /* ADMM 2: send Y_i+rho J_i to master */
      /* calculate Y <= Y + rho J */
      if (admm==0) {
       for(int cm=0; cm<mymscount; cm++) {
        /* If initial solution has diverged, reset solutions before sending 
           anything to master */
        if (res_01[cm] > res_ratio*res_00[cm]) {
         cout<<myrank<<": MS :"<<cm<<" initial residual "<<res_00[cm]<<" increased to "<<res_01[cm]<<". Resetting!"<<endl;
         memcpy(p_vec[cm],pinit,(size_t)iodata_vec[cm].N*8*Mt*sizeof(double));
        }

        /* Y is set to 0 : so original is just rho * J*/
        my_dcopy(iodata_vec[cm].N*8*Mt, p_vec[cm], 1, Y_vec[cm], 1);
        /* scale by individual rho for each cluster */
        /* if rho<=0, do nothing */
        ck=0;
        for (ci=0; ci<M; ci++) {
         /* Y will be set to 0 if rho<=0 */
         my_dscal(iodata_vec[cm].N*8*carr_vec[cm][ci].nchunk, arho_vec[cm][ci], &Y_vec[cm][ck]);
         ck+=iodata_vec[cm].N*8*carr_vec[cm][ci].nchunk;
        }
       }
      } else {
       ck=0;
       for (ci=0; ci<M; ci++) {
        if (arho_vec[mmid][ci]>0.0) {
         my_daxpy(iodata_vec[mmid].N*8*carr_vec[mmid][ci].nchunk, &p_vec[mmid][ck], arho_vec[mmid][ci], &Y_vec[mmid][ck]);
        }
        ck+=iodata_vec[mmid].N*8*carr_vec[mmid][ci].nchunk;
#ifdef DEBUG
      cout<<myrank<<": MS="<<mmid<<" Clus="<<ci<<" Chunk="<<carr_vec[mmid][ci].nchunk<<" Rho="<<arho_vec[mmid][ci]<<endl;
#endif
       }
      }
     
      if (admm==0) {
       for(int cm=0; cm<mymscount; cm++) {
         MPI_Send(Y_vec[cm], iodata_vec[cm].N*8*Mt, MPI_DOUBLE, 0,TAG_YDATA, MPI_COMM_WORLD);
       }
      } else {
       /* if most data are flagged, only send the original Y we got at the beginning */
       MPI_Send(Y_vec[mmid], iodata_vec[mmid].N*8*Mt, MPI_DOUBLE, 0,TAG_YDATA, MPI_COMM_WORLD);
      }
      /* for initial ADMM iteration, get back Y with common unitary ambiguity (for all MS) */
      if (admm==0) {
       for(int cm=0; cm<mymscount; cm++) {
        //MPI_Recv(Y_vec[mmid], iodata_vec[mmid].N*8*Mt, MPI_DOUBLE, 0,TAG_YDATA, MPI_COMM_WORLD, &status);
        MPI_Recv(Y_vec[cm], iodata_vec[cm].N*8*Mt, MPI_DOUBLE, 0,TAG_YDATA, MPI_COMM_WORLD, &status);
       }
      }

      MPI_Recv(Yhat, iodata_vec[mmid].N*8*Mt, MPI_DOUBLE, 0,TAG_CONSENSUS_OLD, MPI_COMM_WORLD, &status);
      ck=0;
      for (ci=0; ci<M; ci++) {
	if (arho_vec[mmid][ci]>0.0) {
         /* first update Yhat, because it needs Y_i , Yhat now = (B_i Z_old) */
         /* scale by -rho */
          my_dscal(iodata_vec[mmid].N*8*carr_vec[mmid][ci].nchunk,-arho_vec[mmid][ci],&Yhat[ck]);
         /* add Y_i + rho J */
         my_daxpy(iodata_vec[mmid].N*8*carr_vec[mmid][ci].nchunk, &Y_vec[mmid][ck], 1.0, &Yhat[ck]);
	}
	ck+=iodata_vec[mmid].N*8*carr_vec[mmid][ci].nchunk;
      }
      /* now update Y_i with B_i Z*/
      if(admm==0){
	/*MM: the first iteration you have to do this for all MS!*/
        for(int cm=0; cm<mymscount; cm++) {
	  /* ADMM 3: get B_i Z from master */
	  MPI_Recv(Z_vec[cm], iodata_vec[cm].N*8*Mt, MPI_DOUBLE, 0,TAG_CONSENSUS, MPI_COMM_WORLD, &status);
	  /* BB : also need Yhat_i <= Y_i + rho (J_i - B_i Z_old), 
	     node we already have Y_i + rho J_i */
	  /* update Y_i <= Y_i + rho (J_i-B_i Z)
	     since we already have Y_i + rho J_i, only need -rho (B_i Z) */
	  ck=0;
	  for (ci=0; ci<M; ci++) {
	    if (arho_vec[cm][ci]>0.0) {
	      /* now update Y_i to new value */
	      my_daxpy(iodata_vec[cm].N*8*carr_vec[cm][ci].nchunk, &Z_vec[cm][ck], -arho_vec[cm][ci], &Y_vec[cm][ck]);
	    }
	    ck+=iodata_vec[cm].N*8*carr_vec[cm][ci].nchunk;
	  }
	}

      }
      else {

      /* ADMM 3: get B_i Z from master */
      MPI_Recv(Z_vec[mmid], iodata_vec[mmid].N*8*Mt, MPI_DOUBLE, 0,TAG_CONSENSUS, MPI_COMM_WORLD, &status);
     
      /* BB : also need Yhat_i <= Y_i + rho (J_i - B_i Z_old), 
              node we already have Y_i + rho J_i */
      /* update Y_i <= Y_i + rho (J_i-B_i Z)
          since we already have Y_i + rho J_i, only need -rho (B_i Z) */
      ck=0;
      for (ci=0; ci<M; ci++) {
        if (arho_vec[mmid][ci]>0.0) {
         /* now update Y_i to new value */
         my_daxpy(iodata_vec[mmid].N*8*carr_vec[mmid][ci].nchunk, &Z_vec[mmid][ck], -arho_vec[mmid][ci], &Y_vec[mmid][ck]);
        }
        ck+=iodata_vec[mmid].N*8*carr_vec[mmid][ci].nchunk;
      }
       
      }
       


      /* BB : update rho, only after each MS is given 1 ADMM iteration
       and also if mymscount == 1, only update skipping every iteration */
      if (Data::aadmm && ((mymscount>1 && admm>=mymscount)|| (mymscount==1 && admm>1 && admm%2==0))) {
       update_rho_bb(arho_vec[mmid],arhoupper_vec[mmid],iodata_vec[mmid].N,M,Mt,carr_vec[mmid],Yhat,Yhat0_vec[mmid],p_vec[mmid],J0_vec[mmid],Data::Nt);
      }
      /* BB : send updated rho to master */
      MPI_Send(arho_vec[mmid],M,MPI_DOUBLE,0,TAG_RHO_UPDATE,MPI_COMM_WORLD);

      /* BB : store current Yhat and J as reference (k0) values */
      my_dcopy(iodata_vec[mmid].N*8*Mt, Yhat, 1, Yhat0_vec[mmid], 1);
      my_dcopy(iodata_vec[mmid].N*8*Mt, p_vec[mmid], 1, J0_vec[mmid], 1);

      /* calculate primal residual J-BZ */
      my_dcopy(iodata_vec[mmid].N*8*Mt, p_vec[mmid], 1, pres, 1);
      my_daxpy(iodata_vec[mmid].N*8*Mt, Z_vec[mmid], -1.0, pres);
      
      /* primal residual : per one real parameter */ 
      /* to remove a load of network traffic and screen output, disable this info */
      if (Data::verbose) {
       cout<<myrank<< ": ADMM : "<<admm<<" : "<<mmid<<" residual: primal="<<my_dnrm2(iodata_vec[mmid].N*8*Mt,pres)/sqrt((double)8*iodata_vec[mmid].N*Mt)<<", initial="<<res_0vec[mmid]<<", final="<<res_1vec[mmid]<<endl;
      }
     }
     /******************** END ADMM *******************************/

     /* write residuals to output */
     for(int cm=0; cm<mymscount; cm++) {
#ifndef HAVE_CUDA
      if (!doBeam) {
       calculate_residuals_multifreq(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,p_vec[cm],iodata_vec[cm].xo,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freqs,iodata_vec[cm].Nchan,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
      } else {
       calculate_residuals_multifreq_withbeam(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,p_vec[cm],iodata_vec[cm].xo,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freqs,iodata_vec[cm].Nchan,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,
       beam_vec[cm].p_ra0,beam_vec[cm].p_dec0,iodata_vec[cm].freq0,beam_vec[cm].sx,beam_vec[cm].sy,beam_vec[cm].time_utc,beam_vec[cm].Nelem,beam_vec[cm].xx,beam_vec[cm].yy,beam_vec[cm].zz,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
      }
#endif
#ifdef HAVE_CUDA
     if (GPUpredict) {
       calculate_residuals_multifreq_withbeam_gpu(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,p_vec[cm],iodata_vec[cm].xo,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freqs,iodata_vec[cm].Nchan,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,
          beam_vec[cm].p_ra0,beam_vec[cm].p_dec0,iodata_vec[cm].freq0,beam_vec[cm].sx,beam_vec[cm].sy,beam_vec[cm].time_utc,beam_vec[cm].Nelem,beam_vec[cm].xx,beam_vec[cm].yy,beam_vec[cm].zz,doBeam,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
     } else {
      if (!doBeam) {
       calculate_residuals_multifreq(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,p_vec[cm],iodata_vec[cm].xo,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freqs,iodata_vec[cm].Nchan,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
      } else {
       calculate_residuals_multifreq_withbeam(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,p_vec[cm],iodata_vec[cm].xo,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freqs,iodata_vec[cm].Nchan,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,
       beam_vec[cm].p_ra0,beam_vec[cm].p_dec0,iodata_vec[cm].freq0,beam_vec[cm].sx,beam_vec[cm].sy,beam_vec[cm].time_utc,beam_vec[cm].Nelem,beam_vec[cm].xx,beam_vec[cm].yy,beam_vec[cm].zz,Data::Nt,Data::ccid,Data::rho,Data::phaseOnly);
      }
     }
#endif
     }
     tilex+=iodata_vec[0].tilesz;

     for(int cm=0; cm<mymscount; cm++) {
     /* print solutions to file */
      for (cj=0; cj<iodata_vec[cm].N*8; cj++) {
       fprintf(sfp_vec[cm],"%d ",cj);
       for (ci=M-1; ci>=0; ci--) {
         for (ck=0; ck<carr_vec[cm][ci].nchunk; ck++) {
          /* print solution */
          fprintf(sfp_vec[cm]," %e",p_vec[cm][carr_vec[cm][ci].p[ck]+cj]);
         }
       }
       fprintf(sfp_vec[cm],"\n");
      }
     }

     for(int cm=0; cm<mymscount; cm++) {
      Data::writeData(msitr[cm]->table(),iodata_vec[cm]);
     }

     /* advance all MS to next data chunk */    
     for(int cm=0; cm<mymscount; cm++) {
       (*msitr[cm])++;
     }
     /* do some quality control */
    /* if residual has increased too much, or all are flagged (0 residual)
      or NaN
      reset solutions to original
      initial values : use residual at 1st ADMM */
    /* do not reset if initial residual is 0, because by def final one will be higher */
     for(int cm=0; cm<mymscount; cm++) {
      if (res_00[cm]!=0.0 && (res_01[cm]==0.0 || !isfinite(res_01[cm]) || res_01[cm]>res_ratio*res_prev[cm])) {
        cout<<"Resetting Solution "<<cm<<endl;
        /* reset solutions so next iteration has default initial values */
        memcpy(p_vec[cm],pinit,(size_t)iodata_vec[cm].N*8*Mt*sizeof(double));
        /* also assume iterations have restarted from scratch */
        start_iter=1;
        /* also forget min residual (otherwise will try to reset it always) */
        if (res_01[cm]!=0.0 && isfinite(res_01[cm])) {
         res_prev[cm]=res_01[cm];
        }
      } else if (res_01[cm]<res_prev[cm]) { /* only store the min value */
       res_prev[cm]=res_01[cm];
      }
    }
    end_time = time(0);
    elapsed_time = ((double) (end_time-start_time)) / 60.0;
    if (solver_mode==SM_OSLM_OSRLM_RLBFGS||solver_mode==SM_RLM_RLBFGS||solver_mode==SM_RTR_OSRLM_RLBFGS || solver_mode==SM_NSD_RLBFGS) { 
     if (Data::verbose) {
      cout<<"nu="<<mean_nu<<endl;
     }
    }
    if (Data::verbose) {
     for(int cm=0; cm<mymscount; cm++) {
      cout<<myrank<< ": Timeslot: "<<tilex<<" residual MS "<<cm<<": initial="<<res_00[cm]<<"/"<<res_0vec[cm]<<",final="<<res_01[cm]<<"/"<<res_1vec[cm]<<", Time spent="<<elapsed_time<<" minutes"<<endl;
     }
    }

#ifdef HAVE_CUDA
   /* if -E uses a large value ~say 100, at each multiple of this, clear GPU memory */
   if (GPUpredict>1 && tilex>0 && !(tilex%GPUpredict)) {
    for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
       cudaSetDevice(gpuid);
       cudaDeviceReset();
       cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
    }
   }
#endif

     /* now send to master signal that we are ready for next data chunk */
     if (start_iter==1) {
      msgcode=CTRL_RESET;
     } else {
      msgcode=CTRL_DONE;
     }

     MPI_Send(&msgcode, 1, MPI_INT, 0,TAG_CTRL, MPI_COMM_WORLD);
    }

    for(int cm=0; cm<mymscount; cm++) {
     delete msitr[cm];
     delete msvector[cm];
    }

    for(int cm=0; cm<mymscount; cm++) {
     /* free data memory */
     if (!doBeam) {
      Data::freeData(iodata_vec[cm]);
     } else {
      Data::freeData(iodata_vec[cm],beam_vec[cm]);
     }
    }
 

    /**********************************************************/

  exinfo_gaussian *exg;
  exinfo_disk *exd;
  exinfo_ring *exr;
  exinfo_shapelet *exs;

  for(int cm=0; cm<mymscount; cm++) {
  for (ci=0; ci<M; ci++) {
    free(carr_vec[cm][ci].ll);
    free(carr_vec[cm][ci].mm);
    free(carr_vec[cm][ci].nn);
    free(carr_vec[cm][ci].sI);
    free(carr_vec[cm][ci].sQ);
    free(carr_vec[cm][ci].sU);
    free(carr_vec[cm][ci].sV);
    free(carr_vec[cm][ci].p);
    free(carr_vec[cm][ci].ra);
    free(carr_vec[cm][ci].dec);
    for (cj=0; cj<carr_vec[cm][ci].N; cj++) {
     /* do a proper typecast before freeing */
     switch (carr_vec[cm][ci].stype[cj]) {
      case STYPE_GAUSSIAN:
        exg=(exinfo_gaussian*)carr_vec[cm][ci].ex[cj];
        if (exg) free(exg);
        break;
      case STYPE_DISK:
        exd=(exinfo_disk*)carr_vec[cm][ci].ex[cj];
        if (exd) free(exd);
        break;
      case STYPE_RING:
        exr=(exinfo_ring*)carr_vec[cm][ci].ex[cj];
        if (exr) free(exr);
        break;
      case STYPE_SHAPELET:
        exs=(exinfo_shapelet*)carr_vec[cm][ci].ex[cj];
        if (exs)  {
          if (exs->modes) {
            free(exs->modes);
          }
          free(exs);
        }
        break;
      default:
        break;
     }
    }
    free(carr_vec[cm][ci].ex);
    free(carr_vec[cm][ci].stype);
    free(carr_vec[cm][ci].sI0);
    free(carr_vec[cm][ci].sQ0);
    free(carr_vec[cm][ci].sU0);
    free(carr_vec[cm][ci].sV0);
    free(carr_vec[cm][ci].f0);
    free(carr_vec[cm][ci].spec_idx);
    free(carr_vec[cm][ci].spec_idx1);
    free(carr_vec[cm][ci].spec_idx2);
  }
  free(carr_vec[cm]);
  free(barr_vec[cm]);
  free(p_vec[cm]);
  free(pm_vec[cm]);
  free(coh_vec[cm]);
  fclose(sfp_vec[cm]);
  if (iodata_vec[cm].Nchan>1 || Data::whiten) {
    free(xbackup_vec[cm]);
  }
  free(Z_vec[cm]);
  free(Yhat0_vec[cm]);
  free(J0_vec[cm]);
  free(Y_vec[cm]);
  free(arho_vec[cm]);
  free(arho0_vec[cm]);
  free(arhoupper_vec[cm]);
  }
  free(Yhat);
  free(pinit);
  free(pres);
  if (myrank==1) {
    free(chunkvec);
  }

  /**********************************************************/

   cout<<"Done."<<endl;    
   return 0;
}
